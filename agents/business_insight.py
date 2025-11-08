"""
Business Insight Agent
Translates analytical findings into actionable business strategies
"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm
from graph_utils import get_next_agent

def business_insight_agent(state: AgentState) -> AgentState:
    """Generate comprehensive insights from analyses"""
    print("\n\n" + "=" * 80)
    print("\n💼 Starting business insight translation...")
    
    deliverables = state.get("deliverables")
    user_query = state["user_prompt"]

    system_message = SystemMessage(content="""
You are the Business Insight Translator explaining data findings to non-technical business stakeholders.

STRICT REQUIREMENTS:
**Business Insight**
1. AUDIENCE: Non-technical executives. Use everyday business language - NO technical jargon
2. STRUCTURE - name each section in user-friendly way, include them based on user query and available data:
   2.1. EXECUTIVE SUMMARY (2-3 sentences)
   - Key findings at a glance
   - Main takeaway for decision makers

   2.2. KEY INSIGHTS (3-5 bullet points)
   - What the data tells us
   - Patterns and trends discovered
   - Surprising or important findings

   2.3. BUSINESS IMPLICATIONS
   - What this means for the business
   - Opportunities identified
   - Risks or challenges highlighted

   2.4. ACTIONABLE RECOMMENDATIONS (3-5 specific actions)
   - Concrete steps to take
   - Prioritized by impact
   - Include quick wins and long-term strategies

   2.5. NEXT STEPS
   - Follow-up analyses needed
   - Data to collect
   - Metrics to monitor
3. CONTENT: Every key point must trace back to data evidence (metrics, segments, patterns)
4. TONE: Clear, concise, business-focused with specific actionable insights

**Visualization Needs**
    - If needed, add "VISUALIZATION_RECOMMENDATION:" at the end with specific chart types.
    - Only for visualization providing essential support, otherwise, skip it.
    - If it provide little added information, and can be messy shown in graph, skip it.

REQUIREMENTS:
- Audience: non-technical. Use everyday words. NO technical jargons.
- Use plain language equivalents (e.g., "more likely to respond" instead of "segment lift")
- Traceability: Statements must map to evidence when applicable (figures, features, metrics, segments).
- Focus more on business outcomes, less on methodology
- Keep recommendations practical and executable
- If no relevant data provided, state what's missing

OUTPUT: Provide a comprehensive but concise business response that answers the original question. 
                                   """)

    human_message = HumanMessage(content=f"""
USER QUERY: {user_query}

DELIVERABLES: {deliverables}

ANALYSIS RESULTS:
Data query: {state.get("query_result", 'no results yet')}
Task analysis: {state.get("task_result", 'no results yet')}

Generate comprehensive business insights and recommendations.
""")

    try:
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        llm = get_llm()
        llm_chain = prompt | llm

        response = llm_chain.invoke({})
        business_insight = response.content

        # Process visualization recommendation
        if "VISUALIZATION_RECOMMENDATION:" in business_insight:
            parts = business_insight.split("VISUALIZATION_RECOMMENDATION:")
            insight = parts[0].strip()
            viz_recommendation = parts[1].strip() if len(parts) > 1 else ""

            state["business_insight"] = insight
            state["viz_recommendation"] = viz_recommendation
            state["needs_visualization"] = bool(viz_recommendation)
        else:
            state["needs_visualization"] = False
            state["business_insight"] = business_insight
            state["viz_recommendation"] = ""

        # Print results
        print("💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print(state["business_insight"])
        if state["needs_visualization"]:
            print(f"\n📊 Visualization Recommendation: {state['viz_recommendation']}")
        print("=" * 80)

        state["current_agent"] = get_next_agent(state, "business_insight_agent")
        print("\n✅ Business insight generation completed.")
        return state

    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"Business insight generation failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}"
        state["can_retry"] = state.get("retry_count", 0) < 3
        return state