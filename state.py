"""
State definition for Multi-Agent System
Defines the AgentState TypedDict used across all agents
"""
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import pandas as pd
from langgraph.graph import add_messages

class AgentState(TypedDict, total=False):
    """
    Global state shared across all agents in the workflow

    Data Pipeline Fields:
        dataset_path: Path to the dataset directory
        datasets: Dictionary of loaded DataFrames {name: DataFrame}
        query_result: Result from data query operations
        task_code: Generated code for tasks
        task_result: Results from task execution
        opt_approval: Whether optimization suggestions were approved
        opt_suggestions: Suggestions from optimization agent
        opt_cnt: Number of optimization cycles performed
        business_insight: Business insights and recommendations
        needs_visualization: Whether visualization is needed
        viz_recommendation: Visualization recommendations

    Control Flow Fields:
        user_prompt: Original user query/request
        messages: Message history for LLM interactions
        workflow_plan: Planned workflow steps
        current_agent: Currently executing agent
        supported_task: Whether the task is supported
        deliverables: Expected deliverables from the workflow

    Error Handling Fields:
        has_error: Whether an error occurred
        error_message: Error message details
        can_retry: Whether retry is possible
        retry_count: Number of retries attempted
        debug_decision: Decision from debugger agent
        target_agent: Target agent for retry/replan
    """

    # Data pipeline
    dataset_path: str
    datasets: Dict[str, pd.DataFrame]
    query_result: Any  # Can be DataFrame or dict
    task_code: Any
    task_result: Any
    opt_approval: bool
    opt_suggestions: Any
    opt_cnt: int
    business_insight: str
    needs_visualization: bool
    viz_recommendation: str

    # Control flow
    user_prompt: str
    messages: Annotated[List[Dict], add_messages]
    workflow_plan: List[Dict[str, Any]]
    current_agent: str
    supported_task: bool
    deliverables: List[str]

    # Error handling
    has_error: bool
    error_message: Optional[str]
    can_retry: bool
    retry_count: int
    debug_decision: str
    target_agent: str
