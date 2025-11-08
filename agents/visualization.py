"""
Visualization Agent
Provides visualization support for business insights
"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm
from graph_utils import get_next_agent

def visualization_agent(state: AgentState) -> AgentState:
    """Provide visualization support for business insight translations"""

    print("\n\n" + "=" * 80)
    print("\n📊 Starting visualization generation...")

    user_query = state["user_prompt"]
    business_insight = state.get("business_insight", "")
    query_result = state.get("query_result", "none")
    task_result = state.get("task_result", "none")

    system_message = SystemMessage(content=f"""You are a Visualization generator for a multi-agent system.
Your job is to enhance business insight by generating supporting visualizations 
by returning a Python code block that
retain business insight but in user-frendly way, and integrete with visualizations.

INPUTS YOU HAVE:
- Business insights already generated
- Specific visualization recommendations
- User's original question
- Available data results

GUIDELINES:
1. PRIORITIZE readability, sense-making, clarity and business relevance over technical complexity
2. CHOOSE chart types that match the business insight, such as:
   - Comparisons → bar charts, grouped bars
   - Trends over time → line charts, area charts  
   - Distributions → histograms, box plots
   - Relationships → scatter plots, heatmaps
   - Proportions → pie charts, donut charts (only for few categories)
3. ENSURE each chart has:
   - Clear business-focused title
   - Readable labels and legends
   - Brief caption explaining the business takeaway
4. Visualization muse be user friendly and be straight to the point
""".strip())

    human_message = HumanMessage(content=f"""
USER QUERY: {user_query}

BUSINESS INSIGHTS: {business_insight}

DATA AVAILABLE:
Query Results: {query_result}
Task Results: {task_result}

Generate visualizations to support these insights.
""")

    try:
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        llm = get_llm()
        llm_chain = prompt | llm

        response = llm_chain.invoke({})
        full_output = response.content

        # Extract and run the code block (if any)
        code_match = re.search(r"```python(.*?)```", full_output, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            print("\n🧪 Executing Visualization Code...\n")
            try:
                exec_globals = {}
                exec(code, exec_globals)
            except Exception as e:
                print("❌ Error executing chart code:", e)
        else:
            print("⚠️ No Python code block found in the output.")

        state["current_agent"] = get_next_agent(state, "visualization_agent")
        print("\n✅ Visualization generation completed.")
        return state

    except Exception as e:
        # Visualization is optional, don't fail the whole pipeline
        print(f"⚠️ Visualization generation failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}")
        state["current_agent"] = get_next_agent(state, "visualization_agent")
        return state
