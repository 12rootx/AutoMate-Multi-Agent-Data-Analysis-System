"""
Main entry point for Multi-Agent System
Provides both CLI and programmatic interfaces
"""
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.markdown import Markdown

from graph import create_workflow_graph
from config import DATA_PATH, OPENAI_API_KEY

console = Console()

def run_analysis(user_prompt: str, dataset_path: str = None, api_key: str = None) -> dict:
    """
    Run multi-agent analysis

    Args:
        user_prompt: User's business question or analysis request
        dataset_path: Path to dataset directory (default: config.DATA_PATH)
        api_key: OpenAI API key (default: config.OPENAI_API_KEY)

    Returns:
        Final state dictionary with results
    """

    # Use defaults from config if not provided
    dataset_path = dataset_path or DATA_PATH
    api_key = api_key or OPENAI_API_KEY

    # Validate inputs
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    if not dataset_path:
        raise ValueError("Dataset path is required. Set DATA_PATH environment variable or pass dataset_path parameter.")

    # Update config with provided API key
    import config
    config.OPENAI_API_KEY = api_key

    # Initialize state
    initial_state = {
        "user_prompt": user_prompt,
        "dataset_path": dataset_path,
        "messages": [],
        "workflow_plan": [],
        "opt_cnt": 0,
        "retry_count": 0,
        "has_error": False
    }

    # Build and execute graph
    print("\n" + "=" * 80)
    print("🚀 MULTI-AGENT SYSTEM STARTING")
    print("=" * 80)
    print(f"\n📝 User Query: {user_prompt}")
    print(f"📁 Dataset Path: {dataset_path}")
    print()

    app = create_workflow_graph()

    # Execute workflow
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 80)
    print("✅ MULTI-AGENT SYSTEM COMPLETED")
    print("=" * 80)

    return final_state

def quick_analysis(user_prompt: str, dataset_path: str = None) -> dict:
    """
    Quick analysis with minimal configuration
    Uses environment variables for API key

    Args:
        user_prompt: User's question
        dataset_path: Path to data (optional)

    Returns:
        Final state with results
    """
    return run_analysis(user_prompt, dataset_path)

def print_results(state: dict):
    """
    Pretty print analysis results

    Args:
        state: Final state from analysis
    """
    console.print("\n[bold cyan]📊 Analysis Results[/bold cyan]\n")

    # Query results
    if state.get("query_result"):
        console.print("[bold]Query Results:[/bold]")
        query_result = state["query_result"]
        if isinstance(query_result, list) and len(query_result) > 0:
            console.print(f"  - {len(query_result)} records returned")
        console.print()

    # Task results
    if state.get("task_result"):
        console.print("[bold]Task Results:[/bold]")
        task_result = state["task_result"]
        if isinstance(task_result, dict):
            for key, value in task_result.items():
                console.print(f"  - {key}: {value}")
        console.print()

    # Business insights
    if state.get("business_insight"):
        console.print("\n[bold cyan]💼 Business Insights[/bold cyan]\n")
        md = Markdown(state["business_insight"])
        console.print(md)
        console.print()

    # Workflow info
    if state.get("workflow_plan"):
        console.print("[bold]Workflow Executed:[/bold]")
        for i, step in enumerate(state["workflow_plan"], 1):
            console.print(f"  {i}. {step.get('agent', 'Unknown')}")
        console.print()

def main():
    """CLI entry point"""
    import sys

    if len(sys.argv) < 2:
        console.print("[bold red]Error:[/bold red] Please provide a query")
        console.print("\nUsage:")
        console.print("  python main.py \"Your business question here\"")
        console.print("\nExample:")
        console.print('  python main.py "What are the top 5 product categories by revenue?"')
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:])

    try:
        # Run analysis
        final_state = run_analysis(user_prompt)

        # Print results
        print_results(final_state)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
