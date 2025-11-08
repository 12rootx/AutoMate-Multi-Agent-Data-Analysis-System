"""
Clustering Agent
Performs customer segmentation and pattern discovery
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm
from utils import generate_schema_summary, extract_python_code, execute_generated_code
from graph_utils import get_next_agent

def clustering_agent(state: AgentState) -> AgentState:
    """Specialized in customer segmentation, product grouping, and pattern discovery using clustering algorithms"""

    print("\n\n" + "=" * 80)
    print("\n🤖 Starting clustering analysis...")

    datasets = state["datasets"]
    user_query = state["user_prompt"]
    error_context = state.get("error_message", "")
    optimization_suggestions = state.get("opt_suggestions", "no suggestions yet")

    system_message = SystemMessage(content=f"""
You are a clustering and segmentation expert. Generate Python code for customer segmentation, product clustering, and pattern discovery.

CRITICAL: Respond ONLY with valid, runnable and ERROR-FREE Python code inside ```python ``` blocks.

Based on the task, implement appropriate techniques:

1. CUSTOMER SEGMENTATION:
   - Use K-means, DBSCAN, or Gaussian Mixture Models
   - Preprocess data: handle missing values, scale numerical features, encode categorical variables
   - Determine optimal number of clusters using elbow method, silhouette score, or gap statistic
   - Profile clusters with summary statistics and characteristics

2. PRODUCT CLUSTERING:
   - Cluster products based on attributes, sales patterns, or customer behavior
   - Use hierarchical clustering for product hierarchies when needed
   - Apply PCA for dimensionality reduction and visualization when needed

3. PATTERN DISCOVERY:
   - Anomaly detection in customer behavior
   - Seasonal pattern clustering
   - Behavioral segment identification

TECHNICAL GUIDELINES:
- Always scale numerical features using StandardScaler
- Handle categorical variables with appropriate encoding (One-Hot, Label)
- Include cluster visualization (2D/3D scatter plots using PCA)
- Provide concise cluster profiling (summarize it if data>20) and business interpretation.
- Keep all print() output minimal and clean — show only essential summaries, insights, and visuals.
- Ensure all visualizations are readable, labeled, and self-explanatory.
- Never use DataFrames in if statements. Use `if not df.empty:` instead of `if df:`.
- ERROR FIX NEEDED: {error_context}
- OPTIMIZATION NEEDED: {optimization_suggestions}

Code structure:
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Pull data from datasets - AVAILABLE DataFrames
table1 = datasets['table1_name'].copy()
table2 ...

# Preprocessing, scaling, encoding
# Determine optimal clusters
# Apply clustering algorithm
# Analyze and profile clusters

Your code MUST a dataframe, end with one of these:
1. For DataFrames (small result data, such as 100 rows x 5 cols as maximum):
final_df = your_dataframe_variable
2. For large result data (use profile summary) 
final_df = profile_summary  # filtered user query relevant data for key findings 
"""
)

    human_message = HumanMessage(content=f"""
                                 USER REQUEST: {user_query}
                                 AVAILABLE DataFrames: {list(datasets.keys())}
                                 DATASET SCHEMAS: {generate_schema_summary(datasets)}
""")

    try:
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        llm = get_llm(model="gpt-4.1")
        llm_chain = prompt | llm

        response = llm_chain.invoke({})
        code = extract_python_code(response.content)

        state["task_code"] = code

        # Execute code
        final_df = execute_generated_code(code, datasets)

        state["task_result"] = final_df.to_dict(orient="records") 
        state["current_agent"] = get_next_agent(state, "clustering_agent")

        print("✅ Clustering analysis completed.")
        return state

    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"Clustering analysis failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}"
        state["can_retry"] = state.get("retry_count", 0) < 3

        return state
