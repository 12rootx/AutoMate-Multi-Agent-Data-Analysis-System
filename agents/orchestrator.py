"""
Orchestrator Agent
Plans the optimal workflow based on user query
"""
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState
from config import get_llm, AGENT_DESCRIPTIONS, FUNCTIONAL_NODES

def print_workflow_plan(result: dict):
    """Print workflow plan in a user-friendly format"""
    print("🔧 WORKFLOW PLAN")
    print("=" * 50)

    # Task status
    status = "✅ SUPPORTED" if result["supported_task"] else "❌ NOT SUPPORTED"
    print(f"Task Status: {status}")
    print()

    # Deliverables
    print("📋 EXPECTED DELIVERABLES:")
    for i, deliverable in enumerate(result["deliverables"], 1):
        print(f"  {i}. {deliverable}")
    print()

    # Workflow steps
    print("🔄 WORKFLOW STEPS:")
    for i, step in enumerate(result["workflow"], 1):
        print(f"  Step {i}: {step['agent']}")
        print(f"    Reason: {step['reason']}")
        print()

def agent_orchestrator(state: AgentState) -> AgentState:
    """LLM decides the optimal workflow"""
    print("\n\n" + "=" * 80)
    print("\n🤖 Starting agent orchestration...")

    user_query = state["user_prompt"]

    # System message with orchestration instructions
    system_message = SystemMessage(content=f"""Based on the query, create an optimal workflow using these agents: {AGENT_DESCRIPTIONS}

# CRITICAL: Respond ONLY one VALID JSON — ALWAYS no markdown, comments and explanations:
{{
  "supported_task": true | false,
  "deliverables": ["list of expected business-ready outputs, concise"],
  "workflow": [{{"agent": "name", "reason": "why"}}]
}}

- If it's a simple data lookup without analysis needs, create a minimal workflow.
- If the user query involves complex tasks, such as modeling (clustering), nlp, recommendations, you should include optimization_agent and business_insight_agent in your workflow.
- Simple data query will not need optimization_agent.
- ALWAYS to HAVE VISUALIZATION AGENT AFTER business_insight_agent.
- Only call eda_agent when user specify or imply to, such as overview, eda.
- Only call 1 functional agent {FUNCTIONAL_NODES}.
- Example workflow (no redundant agents, minimal path to achieve user query):
  1. data_acquisition_agent -> data_query_agent: non-analysis based data query
  2. data_acquisition_agent -> data_query_agent -> business insight: analysis required data query
  3. data_acquisition_agent -> recommendation_agent -> optimization_agent -> business_insight_agent: non data query (e.g., modeling, recommendations) analysis.
  4. data_acquisition_agent -> eda_agent -> business_insight_agent.
""")

    human_message = HumanMessage(content=f"User Query: {user_query}")

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = get_llm()
    llm_chain = prompt | llm

    response = llm_chain.invoke({})

    # Parse response
    if response.content.startswith("```"):
        response_content = response.content.strip("`").split("\n", 1)[-1]
    else:
        response_content = response.content

    result = json.loads(response_content)

    # Update state
    state["workflow_plan"] = result.get("workflow", [])
    state["current_agent"] = state["workflow_plan"][0]["agent"] if state["workflow_plan"] else "END"
    state["supported_task"] = result.get("supported_task", False)
    state["deliverables"] = result.get("deliverables", "")

    # Print plan
    print_workflow_plan(result)

    print("\n✅ Agent orchestration completed.")

    return state
