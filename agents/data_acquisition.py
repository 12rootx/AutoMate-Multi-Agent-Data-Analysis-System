"""
Data Acquisition Agent
Loads and validates datasets from multiple file formats
"""
from state import AgentState
from utils import discover_files, load_multiple_files
from graph_utils import get_next_agent
import traceback

def data_acquisition_agent(state: AgentState) -> AgentState:
    """Ingest data into a dict with dataset_name and pandas dataframe."""

    print("\n\n" + "=" * 80)
    print("\n🤖 Starting data acquisition...")

    try:
        path = state.get("dataset_path", "./dataset")  # Default path

        files = discover_files(path)
        datasets = load_multiple_files(files)

        state["datasets"] = datasets
        state["current_agent"] = get_next_agent(state, "data_acquisition")

        print("\n✅ Data acquisition completed.")
        return state

    except Exception as e:
        return {
            "has_error": True,
            "error_message": str(e),
            "ERROR TRACEBACK": {traceback.format_exc()},
            "can_retry": state.get("retry_count", 0) < 3
        }
