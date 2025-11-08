"""
Debugger Agent
Handles pipeline errors and implements retry logic
"""
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState
from config import get_llm

def debugger_agent(state: AgentState) -> AgentState:
    """
    LLM-based debugger agent:
    Analyzes the error_message and decides next step dynamically.
    """
    print("\n\n" + "=" * 80)
    print("\n🤖 Starting debugging ...")

    # Use .get() with default values for safety
    error_message = state.get("error_message", "Unknown")
    current_agent = state.get("current_agent", "Unknown")
    retry_count = state.get("retry_count", 0)

    print(f"error: {error_message} \nfrom: {current_agent}")

    # System prompt instructing the LLM
    system_prompt = SystemMessage(content=f"""
You are a debugger agent in a multi-agent workflow system.
Analyze the error message and decide:
1. If the workflow should retry the current agent
2. If it should replan via orchestrator
3. If it should stop completely
 - when same issue happened more than twice
 - when it's an external error, which cannot handled by llm

Return JSON ONLY:
{{
  "debug_decision": "retry/replan/stop",
  "can_retry": true/false,
  "retry_count": int,
  "target_agent": "agent_name",
  "error_message": "concise reason for decision and specific fix suggestions"
}}
""")

    human_prompt = HumanMessage(content=f"""
Error message: {error_message}
Current agent: {current_agent}
Retry count: {retry_count}""")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    llm = get_llm()
    llm_chain = prompt | llm

    response = llm_chain.invoke({})

    if response.content.startswith("```"):
        response_content = response.content.strip("`").split("\n", 1)[-1]
    else:
        response_content = response.content

    result = json.loads(response_content)

    state["debug_decision"] = result["debug_decision"]
    state["can_retry"] = result["can_retry"]
    state["retry_count"] = result["retry_count"] + 1
    state["target_agent"] = result["target_agent"]
    state["error_message"] = result["error_message"]

    error_feedback = result["error_message"]

    # Reset has_error
    state["has_error"] = False

    print(f"error_feedback: {error_feedback}")

    print("\n✅ Debugging completed.")

    return state
