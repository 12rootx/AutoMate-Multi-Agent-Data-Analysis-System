"""
Graph utility functions
Helper functions for graph construction and routing
"""
from typing import Dict, List
from langgraph.graph import END

def get_next_agent(state: dict, current_agent: str) -> str:
    """Get the next agent, with optimization loop handling"""
    workflow = state.get("workflow_plan", [])
    agent_names = [step["agent"] for step in workflow]

    try:
        current_index = agent_names.index(current_agent)

        # If optimization_agent failed approval, go back to previous agent
        if current_agent == "optimization_agent" and not state.get("opt_approval", True):
            return agent_names[current_index - 1] if current_index > 0 else agent_names[0]

        # Normal forward flow
        if current_index < len(agent_names) - 1:
            next_agent = agent_names[current_index + 1]

            # Skip optimization_agent if max count reached
            if next_agent == "optimization_agent" and state.get("opt_cnt", 0) >= 3:
                return agent_names[current_index + 2] if current_index + 2 < len(agent_names) else END

            return next_agent
        else:
            return END
    except ValueError:
        # If current agent is orchestrator, start workflow
        if current_agent == "agent_orchestrator" and workflow:
            return workflow[0]["agent"]
        return END

def print_workflow_plan(result: Dict) -> None:
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
