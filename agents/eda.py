"""
EDA Agent
Performs exploratory data analysis with visualizations
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import traceback

from state import AgentState
from config import get_llm
from utils import generate_schema_summary, extract_python_code, execute_generated_code
from graph_utils import get_next_agent

def eda_agent(state: AgentState) -> AgentState:
    """Comprehensive exploratory data analysis, statistical summaries, and visualization"""

    print("\n\n" + "=" * 80)
    print("\n📊 Starting EDA analysis...")

    datasets = state["datasets"]
    user_query = state["user_prompt"]
    error_context = state.get("error_message", "")
    optimization_suggestions = state.get("opt_suggestions", "no suggestions yet")

    system_message = SystemMessage(content=f"""
You are an expert data analyst specializing in exploratory data analysis. Generate Python code for comprehensive data exploration and visualization.

CRITICAL: Respond ONLY with valid, runnable and ERROR-FREE Python code inside ```python ``` blocks.

Implement clean, concise EDA, covering some of the followings when needed for user query:

1. DATA QUALITY ASSESSMENT:
   - Missing values analysis and visualization
   - Data types verification
   - Duplicate detection
   - Basic statistics (describe)

2. UNIVARIATE ANALYSIS:
   - Distribution plots for numerical features (histograms, boxplots, KDE)
   - Frequency analysis for categorical features (bar charts, pie charts)
   - Outlier detection using IQR and statistical methods

3. MULTIVARIATE ANALYSIS:
   - Correlation matrix and heatmap
   - Scatter plots for numerical relationships
   - Cross-tabulations for categorical relationships
   - Pair plots for feature interactions

4. ADVANCED VISUALIZATION:
   - Interactive plots using Plotly (if appropriate)
   - Subplot grids for comprehensive overview
   - Time series decomposition (if datetime present)

TECHNICAL GUIDELINES:
- Handle missing values appropriately (visualize, don't drop without reason)
- Use appropriate visualization for each data type, and choose best arguments to generate clean and insightful visualization
- Include both statistical summaries and visual insights
- Automatically detect datetime columns and perform time-based analysis
- Keep all print() output minimal and clean — show only essential summaries, insights, and visuals.
- **Ensure all outputs are user readable, labeled, and self-explanatory**.
- ERROR FIX NEEDED: {error_context}

Code structure:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Pull data from datasets - AVAILABLE DataFrames
table1 = datasets['table1_name'].copy()
table2 ...

# 1. Data Quality Assessment if needed
# 2. Statistical Summary if needed
# 3. Visualizations if needed

# Add specific visualizations based on data characteristics
# - Distribution plots
# - Correlation heatmap
# - Missing values heatmap
# - Categorical analysis
# - Outlier detection

plt.tight_layout()
plt.show()

final_df = df # relevant, concise and comprehensive df
```
""")

    human_message = HumanMessage(content=f"""USER REQUEST: {user_query}

AVAILABLE DataFrames: {list(datasets.keys())}

DATASET SCHEMAS:
{generate_schema_summary(datasets)}

SPECIFIC TASK: Perform comprehensive exploratory data analysis relevant to user request.
""")

    try:
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        llm = get_llm(model="gpt-4o")
        llm_chain = prompt | llm

        response = llm_chain.invoke({})
        code = extract_python_code(response.content)

        state["task_code"] = code

        # Execute code
        final_df = execute_generated_code(code, datasets)

        state["task_result"] = {
            "summary_stats": final_df.describe(include="all").to_dict() if hasattr(final_df, 'describe') else {}
        }
        state["current_agent"] = get_next_agent(state, "eda_agent")

        print("✅ EDA analysis completed.")
        return state

    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"EDA failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}"
        state["can_retry"] = state.get("retry_count", 0) < 3

        return state
