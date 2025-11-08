"""
Recommendation Agent
Handles product recommendations and association analysis
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm
from utils import generate_schema_summary, extract_python_code, execute_generated_code
from graph_utils import get_next_agent

def recommendation_agent(state: AgentState) -> AgentState:
    """Handles product recommendations, frequently bought together, and association rules"""

    print("\n\n" + "=" * 80)
    print("\n🤖 Starting recommendation analysis...")

    datasets = state["datasets"]
    user_query = state["user_prompt"]
    error_context = state.get("error_message", "")
    optimization_suggestions = state.get("opt_suggestions", "no suggestions yet")

    system_message = SystemMessage(content=f"""
You are a recommendation systems expert. Generate Python code for product recommendations and association analysis.

CRITICAL: Respond ONLY with valid, runnable and ERROR-FREE Python code inside ```python ``` blocks.

Based on the task, implement appropriate techniques:

1. FREQUENTLY BOUGHT TOGETHER:
   - Use market basket analysis (Apriori algorithm)
   - Dynamically adjust/fine-tune important parameters based on dataset characteristics (e.g., min_support should be small for large and sparse dataset)
   - Optionally filter results by minimum occurrence count only when needed
   - Calculate support, confidence, lift metrics
   - Return top N rules sorted by lift/confidence

   - **Fallback rule**: If no frequent itemsets or no 'support' column is produced, retry Apriori with lower min_support (e.g., /10, /100).

2. PRODUCT RECOMMENDATIONS:
   - Collaborative filtering (user-item interactions)
   - Content-based filtering (product attributes)
   - Popularity-based recommendations for new users

3. CUSTOMER RECOMMENDATIONS:
   - Segment-based recommendations
   - Demographic-based suggestions
   - Hybrid approaches

ADDITIONAL RULES:
- ERROR FIX NEEDED: {error_context}.
- OPTIMIZATION NEEDED: {optimization_suggestions}.
- Always handle missing or empty results gracefully. Never assume a column (like 'support', 'lift') exists without checking.
- If the final dataframe is large (>1000 rows), return SUMMARY/KEY RESULTS based on user query.
- Always prioritize optimization suggestions and error messages if present.
- Perform DATA SANITIZATION when needed.

Code structure:
```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
# Pull data from datasets - AVAILABLE DataFrames
df1 = datasets["xx"].copy()
...
# Analysis based on request type
    # Market basket analysis
    # Recommendation logic
    # Collaborative filtering or popularity-based
    
Your code MUST a dataframe, end with one of these:
1. For DataFrames (small result data, such as 100 rows x 5 cols as maximum):
final_df = your_dataframe_variable
2. For large result data (use profile summary) 
final_df = profile_summary  # filtered user query relevant data for key findings 
""")

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
        state["current_agent"] = get_next_agent(state, "recommendation_agent")

        print("✅ Recommendation analysis completed.")
        return state

    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"Recommendation analysis failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}"
        state["can_retry"] = state.get("retry_count", 0) < 3

        return state
