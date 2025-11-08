"""
Data Query Agent
Generates and executes Python code for data processing
"""
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm
from utils import generate_schema_summary, extract_python_code, validate_and_execute_code
from graph_utils import get_next_agent

def data_query_agent(state: AgentState) -> AgentState:
    """LLM generates executable Python code for data processing and extract the result of data query"""

    print("\n\n" + "=" * 80)
    print("\n🤖 Starting data querying...")

    datasets = state["datasets"]
    user_query = state["user_prompt"]
    error_context = state.get("error_message", "")

    schema_summary = generate_schema_summary(datasets)

    system_message = SystemMessage(content=f"""You are a senior data engineer. Generate Python code to process the tables based on the user's analysis request.

CRITICAL: Respond ONLY with valid Python code inside ```python ``` blocks. No explanations.

Your code must:
1. Import necessary libraries (pandas, numpy already available)
2. Load tables from the `datasets` dictionary
3. Perform joins, filters, aggregations as needed
4. Sort the result for readability if applicable
5. Return the final business-ready DataFrame as `final_df`

Code structure:
```python
import pandas as pd
...

# Load datasets
table1 = datasets['table1_name'].copy()
table2 ...

# Data processing logic
# Joins, filters, aggregations, sorted etc.

# Final result
final_df = ...
```
- Write efficient, production-ready code.
- Present the result in a business-friendly way, such as sorted df.
- Additional Context:{error_context}.
- If the final dataframe is large (>1000rows), You should return SUMMARY/KEY RESULTS based on user query.
""")

    human_message = HumanMessage(content=f"""
User Analysis Request: {user_query}
Available Tables: {list(datasets.keys())}
Table Schemas: {json.dumps(schema_summary, indent=2, default=str)}""")

    try:
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        llm = get_llm()
        llm_chain = prompt | llm

        response = llm_chain.invoke({})

        code = extract_python_code(response.content)

        # Execute code and get result
        final_df = validate_and_execute_code(code, datasets)

        print(f"final dataframe:\n {final_df}")

        state["query_result"] = final_df.to_dict(orient="records")
        state["current_agent"] = get_next_agent(state, "data_query_agent")

        print("\n✅ Data query completed.")
        return state

    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"Data query failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}"
        state["can_retry"] = state.get("retry_count", 0) < 3

        return state
