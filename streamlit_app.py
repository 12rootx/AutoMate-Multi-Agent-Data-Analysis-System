"""
Streamlit UI for Multi-Agent System
Clean 3-tab structure: Business Insights & Data | Progress Log | Technical Details
"""
import streamlit as st
import pandas as pd
import os
import sys
import io
import matplotlib.pyplot as plt
import time
import traceback
from PIL import Image

# Import from modular structure
from main import run_analysis
from config import DATA_PATH
from utils import discover_files


icon = Image.open("/Users/rootx/Downloads/icon.png")

st.set_page_config(
    page_title="Multi-Agent Data Analysis System",
    page_icon=icon,
    #page_icon="🤖",
    layout="wide"
)


# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'process_logs' not in st.session_state:
    st.session_state.process_logs = []
if 'figures' not in st.session_state:
    st.session_state.figures = []
if 'figures_by_agent' not in st.session_state:
    st.session_state.figures_by_agent = {}

# Main title
st.title("🤖 AutoMate Multi-Agent System")
st.markdown("**Intelligent data analysis with LLM-powered agents**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key"
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Data path input
data_path = st.sidebar.text_input(
    "Data Path",
    value=DATA_PATH,
    help="Path to your dataset directory"
)

if data_path:
    os.environ["DATA_PATH"] = data_path

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Query Input")

    # Query input
    user_query = st.text_area(
        "Enter your business question:",
        height=100,
        placeholder="Example: What are the top 5 product categories by revenue?"
    )

    # Example queries
    st.subheader("💡 Example Queries")
    example_queries = [
        "How many customers are there by state?",
        "show me Top 20 popular product categories with their orders, average price and corresponding revenues in latest 3 months, and provide your thoughts",
        "Give me an overview about product categroy in state SP, for the lastest 3 months",
        "What product categories are frequently bought together with sports and leisure?",
        "Recommend 3–6 category groups suitable for targeted marketing campaigns, using the latest 3 months data?",
        "compare review language across top 5 origin countries. what should we suggest to customers when describing those coffee bean origins? what are the distincts between them"
    ]

    selected_example = st.selectbox("Or choose an example:", [""] + example_queries)
    if selected_example:
        user_query = selected_example

with col2:
    st.header("🎯 System Status")

    # Check if API key is provided
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.error("❌ API Key required")

    # Check if data path exists
    if os.path.exists(data_path):
        st.success("✅ Data path found")

        # Show available files
        try:
            files = discover_files(data_path)
            if files:
                st.info(f"📁 Found {len(files)} data files")
                with st.expander("View data files"):
                    for file in files[:10]:
                        st.text(os.path.basename(file))
            else:
                st.warning("⚠️ No data files found")
        except:
            st.warning("⚠️ Cannot access data files")
    else:
        st.error("❌ Data path not found")

# Analysis section
st.header("🔍 Analysis Results")

# Custom output capture class with agent tracking
class StreamlitLogger:
    """Capture stdout and track which agent is currently executing"""
    def __init__(self, status_container):
        self.status_container = status_container
        self.logs = []
        self.current_agent = None

    def write(self, text):
        if text.strip():
            self.logs.append(text)

            # Detect agent changes for tracking
            if "Starting" in text and "agent" in text.lower():
                if "orchestrat" in text.lower():
                    self.current_agent = "orchestrator"
                elif "data acquisition" in text.lower():
                    self.current_agent = "data_acquisition"
                elif "data query" in text.lower():
                    self.current_agent = "data_query"
                elif "eda" in text.lower():
                    self.current_agent = "eda"
                elif "clustering" in text.lower():
                    self.current_agent = "clustering"
                elif "recommendation" in text.lower():
                    self.current_agent = "recommendation"
                elif "nlp" in text.lower():
                    self.current_agent = "nlp"
                elif "optimization" in text.lower():
                    self.current_agent = "optimization"
                elif "business insight" in text.lower():
                    self.current_agent = "business_insight"
                elif "visualization" in text.lower():
                    self.current_agent = "visualization"
                elif "debug" in text.lower():
                    self.current_agent = "debugger"

            # Update status container
            with self.status_container:
                st.text(text)

    def flush(self):
        pass

    def get_current_agent(self):
        return self.current_agent

# Enhanced matplotlib figure capture with agent tracking
class FigureCapture:
    """Capture matplotlib figures and track which agent created them"""
    def __init__(self, logger):
        self.figures = []
        self.figures_by_agent = {}
        self.logger = logger
        self._original_show = plt.show

    def __enter__(self):
        def custom_show():
            fig = plt.gcf()
            if fig.get_axes():
                self.figures.append(fig)

                # Track which agent created this figure
                current_agent = self.logger.get_current_agent()
                if current_agent:
                    if current_agent not in self.figures_by_agent:
                        self.figures_by_agent[current_agent] = []
                    self.figures_by_agent[current_agent].append(fig)

        plt.show = custom_show
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self._original_show

if st.button("🚀 Run Analysis", type="primary", disabled=not (api_key and user_query)):
    if not api_key:
        st.error("Please provide your OpenAI API key")
    elif not user_query:
        st.error("Please enter a query")
    else:
        # Clear previous results
        st.session_state.analysis_results = None
        st.session_state.process_logs = []
        st.session_state.figures = []
        st.session_state.figures_by_agent = {}

        # Create containers for real-time updates
        status_container = st.container()
        progress_bar = st.progress(0)

        with status_container:
            st.subheader("📊 Analysis Progress")
            process_output = st.empty()

        try:
            # Create logger first
            logger = StreamlitLogger(process_output)

            # Capture stdout and figures
            with FigureCapture(logger) as fig_capture:
                old_stdout = sys.stdout
                sys.stdout = logger

                try:
                    progress_bar.progress(10)
                    result = run_analysis(user_query, data_path, api_key)
                    progress_bar.progress(90)

                    # Store results
                    st.session_state.analysis_results = result
                    st.session_state.process_logs = logger.logs
                    st.session_state.figures = fig_capture.figures
                    st.session_state.figures_by_agent = fig_capture.figures_by_agent

                    progress_bar.progress(100)
                finally:
                    sys.stdout = old_stdout

            st.success("✅ Analysis completed successfully!")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}")
            st.exception(e)
            progress_bar.empty()

# Display results if available
if st.session_state.analysis_results is not None:
    result = st.session_state.analysis_results

    # CLEAN 3-TAB STRUCTURE
    tab1, tab2, tab3 = st.tabs([
        "💼 Business Insights & Data Visualizations",
        "🔧 Progress Log",
        "⚙️ Technical Details"
    ])

    # ============================================================================
    # TAB 1: BUSINESS INSIGHTS & DATA VISUALIZATIONS
    # ============================================================================
    with tab1:
        st.subheader("💼 Business Insights & Recommendations")

        insights = result.get("business_insight")

            # Clean up the text before displaying
        if insights:
            # Fix common formatting issues
            cleaned_insights = insights.replace('”', '"').replace('“', '"')
            cleaned_insights = cleaned_insights.replace('’', "'").replace('‘', "'")
        
            st.markdown(cleaned_insights)
        else:
            st.info("No business insights generated")

        # Get figures from business_insight and visualization agents
        business_figures = st.session_state.figures_by_agent.get("business_insight", [])
        viz_agent_figures = st.session_state.figures_by_agent.get("visualization", [])
        relevant_figures = business_figures + viz_agent_figures

        # Display supporting visualizations inline
        if relevant_figures:
            st.markdown("---")
            st.markdown("### 📊 Supporting Visualizations")

            # Smart layout based on number of figures
            if len(relevant_figures) == 1:
                st.pyplot(relevant_figures[0])
            elif len(relevant_figures) == 2:
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(relevant_figures[0])
                with col2:
                    st.pyplot(relevant_figures[1])
            else:
                # Grid layout for 3+ figures
                for i in range(0, len(relevant_figures), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(relevant_figures):
                            with col:
                                st.pyplot(relevant_figures[i + j])

            # Download buttons for visualizations
            st.markdown("**Download visualizations:**")
            cols = st.columns(min(3, len(relevant_figures)))
            for i, fig in enumerate(relevant_figures):
                with cols[i % 3]:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        f"📥 Chart {i+1}",
                        buf,
                        f"chart_{i+1}.png",
                        "image/png",
                        key=f'download-viz-{i}'
                    )

        # Display data results (query results and task results)
        st.markdown("---")
        st.markdown("### 📊 Data Results")

        # Query Results
        query_result = result.get("query_result")
        if query_result:
            try:
                if isinstance(query_result, list) and len(query_result) > 0:
                    st.markdown("**Query Results:**")
                    df = pd.DataFrame(query_result)
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Query Results (CSV)",
                        csv,
                        "query_results.csv",
                        "text/csv",
                        key='download-query'
                    )
                elif isinstance(query_result, dict):
                    st.markdown("**Query Results:**")
                    st.json(query_result)
                else:
                    st.write(query_result)
            except Exception as e:
                st.error(f"Error displaying query results: {e}\nERROR TRACEBACK:{traceback.format_exc()}")
                st.json(query_result)

        # Task Results
        task_result = result.get("task_result")
        if task_result:
            st.markdown("---")
            try:
                if isinstance(task_result, list) and len(task_result) > 0:
                    st.markdown("**Analysis Results:**")
                    df_task = pd.DataFrame(task_result)
                    st.dataframe(df_task, use_container_width=True)

                    csv_task = df_task.to_csv(index=False)
                    st.download_button(
                        "📥 Download Analysis Results (CSV)",
                        csv_task,
                        "analysis_results.csv",
                        "text/csv",
                        key='download-task'
                    )
                elif isinstance(task_result, dict):
                    st.markdown("**Analysis Results:**")
                    for key, value in task_result.items():
                        with st.expander(f"📌 {key}"):
                            if isinstance(value, (dict, list)):
                                st.json(value)
                            else:
                                st.write(value)
                else:
                    st.write(task_result)
            except Exception as e:
                st.error(f"Error displaying task results: {e}\nERROR TRACEBACK:{traceback.format_exc()}")
                st.json(task_result)

        # Display all visualizations (from EDA, clustering, etc.)
        other_figures = []
        for agent_name, figs in st.session_state.figures_by_agent.items():
            if agent_name not in ["business_insight", "visualization"]:
                other_figures.extend(figs)

        if other_figures:
            st.markdown("---")
            st.markdown("### 📈 Additional Visualizations")
            st.markdown("*Charts from exploratory analysis, clustering, and other agents*")

            # Display in expandable sections by agent
            for agent_name, figs in st.session_state.figures_by_agent.items():
                if agent_name not in ["business_insight", "visualization"] and figs:
                    with st.expander(f"🤖 {agent_name.replace('_', ' ').title()} ({len(figs)} chart{'s' if len(figs) > 1 else ''})"):
                        for i, fig in enumerate(figs):
                            st.pyplot(fig)

                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                f"📥 Download",
                                buf,
                                f"{agent_name}_chart_{i+1}.png",
                                "image/png",
                                key=f'download-{agent_name}-{i}'
                            )

        # Quick summary at bottom
        st.markdown("---")
        st.markdown("### 📋 Quick Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            datasets = result.get("datasets", {})
            st.metric("Datasets Analyzed", len(datasets))
        with col2:
            workflow_plan = result.get("workflow_plan", [])
            st.metric("Agents Executed", len(workflow_plan))
        with col3:
            total_viz = len(st.session_state.figures)
            st.metric("Visualizations Created", total_viz)

    # ============================================================================
    # TAB 2: PROGRESS LOG
    # ============================================================================
    with tab2:
        st.subheader("🔧 Process Execution Log")
        st.markdown("*Complete execution trace showing all agent activities, outputs, and results*")

        if st.session_state.process_logs:
            # Display logs in a text area
            log_text = "\n".join(st.session_state.process_logs)
            st.text_area(
                "Execution Log (contains all results and outputs)",
                log_text,
                height=500,
                help="This log shows real-time execution from all agents including prints, results, and status updates"
            )

            # Download logs
            st.download_button(
                "📥 Download Execution Log",
                log_text,
                "execution_log.txt",
                "text/plain",
                key='download-logs'
            )
        else:
            st.info("No execution logs captured")

        # Show generated code if available
        if result.get("task_code"):
            st.markdown("---")
            st.markdown("### 💻 Generated Code")
            st.code(result["task_code"], language="python")

            st.download_button(
                "📥 Download Generated Code",
                result["task_code"],
                "generated_code.py",
                "text/plain",
                key='download-code'
            )

    # ============================================================================
    # TAB 3: TECHNICAL DETAILS
    # ============================================================================
    with tab3:
        st.subheader("⚙️ Technical Analysis Details")

        # 1. Workflow Execution
        st.markdown("### 🔄 Workflow Execution")
        if "workflow_plan" in result and result["workflow_plan"]:
            workflow_df = pd.DataFrame(result["workflow_plan"])
            if not workflow_df.empty:
                st.dataframe(workflow_df, use_container_width=True)

            # Execution flow diagram
            st.markdown("**Execution Flow:**")
            flow_text = " → ".join([step.get("agent", "Unknown") for step in result["workflow_plan"]])
            st.code(flow_text)
        else:
            st.info("No workflow information available")

        st.markdown("---")

        # 2. Quality Metrics
        st.markdown("### 📊 Quality Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Optimization Cycles",
                result.get("opt_cnt", 0),
                help="Number of quality improvement iterations performed"
            )

        with col2:
            approval = result.get("opt_approval", None)
            if approval is not None:
                st.metric(
                    "Quality Approval",
                    "✅ Approved" if approval else "⚠️ Needs Work"
                )
            else:
                st.metric("Quality Approval", "N/A")

        with col3:
            retry_count = result.get("retry_count", 0)
            st.metric(
                "Retry Attempts",
                retry_count,
                help="Number of error recovery attempts"
            )

        # Optimization suggestions
        if result.get("opt_suggestions"):
            st.markdown("**Optimization Suggestions:**")
            st.text(result.get("opt_suggestions"))

        st.markdown("---")

        # 3. Expected Deliverables
        st.markdown("### 📋 Expected Deliverables")
        if result.get("deliverables"):
            for i, deliverable in enumerate(result["deliverables"], 1):
                st.write(f"{i}. {deliverable}")
        else:
            st.info("No deliverables listed")

        st.markdown("---")

        # 4. Agent Performance Summary
        st.markdown("### 🤖 Agent Performance Summary")
        agent_summary = []
        for step in result.get("workflow_plan", []):
            agent_name = step.get("agent", "Unknown")
            reason = step.get("reason", "N/A")
            viz_count = len(st.session_state.figures_by_agent.get(agent_name, []))
            agent_summary.append({
                "Agent": agent_name,
                "Purpose": reason,
                "Visualizations": viz_count
            })

        if agent_summary:
            summary_df = pd.DataFrame(agent_summary)
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No agent performance data available")

        st.markdown("---")

        # 5. Data Statistics
        st.markdown("### 📊 Data Statistics")
        datasets = result.get("datasets", {})
        if datasets:
            data_stats = []
            for name, df in datasets.items():
                data_stats.append({
                    "Dataset": name,
                    "Rows": df.shape[0],
                    "Columns": df.shape[1],
                    "Memory (MB)": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}"
                })

            stats_df = pd.DataFrame(data_stats)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No dataset statistics available")

        st.markdown("---")

        # 6. Error Information (if any)
        if result.get("has_error"):
            st.markdown("### ❌ Error Information")
            st.error(f"**Error:** {result.get('error_message', 'Unknown error')}")

            if result.get("debug_decision"):
                st.write(f"**Debug Decision:** {result.get('debug_decision')}")

            if result.get("can_retry"):
                st.info("✅ Error is recoverable - retry is possible")
            else:
                st.warning("⚠️ Error is not recoverable")

        st.markdown("---")

        # 7. Raw State (for debugging)
        with st.expander("🔍 View Raw State (JSON)", expanded=False):
            displayable_result = {}
            for key, value in result.items():
                if key == "datasets":
                    displayable_result[key] = {
                        name: f"DataFrame({df.shape[0]} rows × {df.shape[1]} cols)" 
                        for name, df in value.items()
                    }
                elif key == "messages":
                    displayable_result[key] = f"[{len(value)} messages]"
                else:
                    try:
                        import json
                        json.dumps(value)
                        displayable_result[key] = value
                    except:
                        displayable_result[key] = str(value)

            st.json(displayable_result)

# Footer
st.markdown("---")
st.markdown("**Multi-Agent Data Science System** - Powered by LangGraph & OpenAI")

# Sidebar - Usage instructions
with st.sidebar:
    st.markdown("---")
    with st.expander("📚 How to use"):
        st.markdown("""
**Quick Start:**

1. Enter OpenAI API key
2. Set dataset path
3. Enter business question
4. Click "Run Analysis"
5. Review 3 comprehensive tabs

**Tab Structure:**

🔹 **Business Insights & Data**
- Key findings and recommendations
- Supporting visualizations inline
- Data results in tables
- All charts with downloads

🔹 **Progress Log**
- Complete execution trace
- All agent activities
- Results and outputs
- Generated code

🔹 **Technical Details**
- Workflow execution
- Quality metrics
- Agent performance
- Data statistics
- Error information (if any)

**Example Queries:**
- "Perform exploratory data analysis"
- "Segment customers into groups"
- "What products are frequently bought together?"
- "Show sales trends with visualizations"
""")

    st.markdown("---")
    st.markdown("### 🔍 System Info")
    st.markdown(f"**Python:** {sys.version.split()[0]}")
    st.markdown(f"**Working Dir:** `{os.getcwd()}`")

if __name__ == "__main__":
    pass