"""
Graph Construction Module
Builds the dynamic LangGraph workflow based on planned agents
"""
from typing import List, Dict
from langgraph.graph import StateGraph, END

from state import AgentState
from agents import (
    agent_orchestrator,
    data_acquisition_agent,
    data_query_agent,
    eda_agent,
    recommendation_agent,
    clustering_agent,
    nlp_agent,
    optimization_agent,
    business_insight_agent,
    visualization_agent,
    debugger_agent
)
from graph_utils import get_next_agent

# Define agent mappings
AGENT_MAPPINGS = {
    'agent_orchestrator': agent_orchestrator,
    'data_acquisition': data_acquisition_agent,
    'data_query_agent': data_query_agent,
    'eda_agent': eda_agent,
    'nlp_agent': nlp_agent,
    'clustering_agent': clustering_agent,
    'recommendation_agent': recommendation_agent,
    'optimization_agent': optimization_agent,
    'business_insight_agent': business_insight_agent,
    'visualization_agent': visualization_agent,
    'debugger_agent': debugger_agent
}

def route_after_debugger(state: AgentState) -> str:
    """
    Decide where to go after debugger.
    LLM has populated 'debug_decision' and 'target_agent'.
    """
    decision = state.get("debug_decision", "stop")

    if decision == "retry":
        return state.get("target_agent", "data_acquisition")

    elif decision == "replan":
        return state.get("target_agent", "agent_orchestrator")
    else:
        print("⚠️ Debug decision unresolved, stop.")
        return END

def build_dynamic_graph(workflow_plan: List = None) -> StateGraph:
    """
    Dynamically build LangGraph based on planner's workflow

    Args:
        workflow_plan: List of workflow steps from orchestrator

    Returns:
        Compiled StateGraph
    """

    builder = StateGraph(AgentState)

    # Add all agents as nodes
    for agent_name, agent_func in AGENT_MAPPINGS.items():
        builder.add_node(agent_name, agent_func)

    # ALWAYS set an entry point
    if workflow_plan:
        builder.set_entry_point(workflow_plan[0]["agent"])
    else:
        builder.set_entry_point("agent_orchestrator")

    # Build linear workflow if plan exists
    if workflow_plan:
        for i, step in enumerate(workflow_plan):
            current_agent = step["agent"]
            if i < len(workflow_plan) - 1:
                next_agent = workflow_plan[i + 1]["agent"]
                builder.add_edge(current_agent, next_agent)
            else:
                builder.add_edge(current_agent, END)
    else:
        # Default flow when no plan exists yet
        builder.add_edge("agent_orchestrator", "data_acquisition")

    def create_error_handler(agent_name: str):
        """Create error handler for specific agent"""
        def error_handler(state: AgentState) -> str:
            if state.get("has_error"):
                return "debugger_agent"
            else:
                return get_next_agent(state, agent_name)
        return error_handler

    # Add conditional edges for debugger
    all_destinations = list(AGENT_MAPPINGS.keys()) + [END]
    builder.add_conditional_edges(
        "debugger_agent",
        route_after_debugger,
        all_destinations
    )

    # Add conditional edges for error handling to all agents
    for agent_name in AGENT_MAPPINGS.keys():
        if agent_name != "debugger_agent":
            builder.add_conditional_edges(
                agent_name,
                create_error_handler(agent_name),
                ["debugger_agent"] + all_destinations
            )

    return builder.compile()

def create_workflow_graph(initial_state: Dict = None) -> StateGraph:
    """
    Convenience function to create a workflow graph

    Args:
        initial_state: Initial state dictionary with user_prompt and dataset_path

    Returns:
        Compiled StateGraph ready for execution
    """
    return build_dynamic_graph([])
