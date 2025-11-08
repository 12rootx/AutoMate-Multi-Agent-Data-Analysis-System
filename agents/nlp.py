"""
NLP Agent
Processes text data for sentiment analysis and topic modeling
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm
from utils import generate_schema_summary, extract_python_code, execute_generated_code
from graph_utils import get_next_agent

def nlp_agent(state: AgentState) -> AgentState:
    """Processes text data for sentiment analysis, topic modeling, and natural language understanding"""

    print("\n\n" + "=" * 80)
    print("\n🤖 Starting NLP analysis...")

    datasets = state["datasets"]
    user_query = state["user_prompt"]
    error_context = state.get("error_message", "")
    optimization_suggestions = state.get("opt_suggestions", "no suggestions yet")

    system_message = SystemMessage(content=f"""
You are an NLP expert specializing in text analysis and natural language processing. Generate Python code for text mining and NLP tasks.

CRITICAL: Respond ONLY with valid, runnable and ERROR-FREE Python code inside ```python ``` blocks.

Implement appropriate NLP techniques:

1. TEXT PREPROCESSING:
   - Lowercasing, punctuation removal, tokenization
   - Stopword removal, lemmatization, stemming
   - Custom cleaning for specific domains

2. SENTIMENT ANALYSIS:
   - VADER sentiment analysis for social media/text
   - TextBlob for general sentiment
   - Custom sentiment dictionaries

3. TOPIC MODELING:
   - LDA (Latent Dirichlet Allocation)
   - NMF (Non-negative Matrix Factorization)
   - Key phrase extraction using RAKE or YAKE

4. TEXT CLASSIFICATION:
   - TF-IDF vectorization
   - Word embeddings (Word2Vec, GloVe)
   - Basic text classification models


5. TEXT VISUALIZATION:
   - Word clouds
   - Frequency distributions
   - Topic visualization with pyLDAvis

TECHNICAL GUIDELINES:
- Always handle missing text data gracefully
- Use progress bars for large text processing
- Implement efficient text preprocessing pipelines
- Include interpretability for NLP results
- Never use DataFrames in if statements. Use `if not df.empty:` instead of `if df:`.

- ERROR FIX NEEDED: {error_context}
- OPTIMIZATION NEEDED: {optimization_suggestions}

Code structure:
```python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
...

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Data preparation
# Pull data from datasets - AVAILABLE DataFrames
table1 = datasets['table1_name'].copy()
table2 ...
# Text preprocessing function

# Apply preprocessing

# Sentiment Analysis

# Topic Modeling
...

# Generate key results or summary, and visualizations
Your code MUST be a dataframe, end with one of these:
1. For DataFrames (small result data, such as 100 rows x 5 cols as maximum):
final_df = your_dataframe_variable
2. For large result data (use profile summary) 
final_df = profile_summary  # filtered user query relevant data for key findings 
""")

    human_message = HumanMessage(content=f"""
                                 USER REQUEST: {user_query}
                                 AVAILABLE DataFrames: {list(datasets.keys())}
                                 DATASET SCHEMAS:{generate_schema_summary(datasets)}
SPECIFIC TASK: Perform natural language processing and text analysis.
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
        state["current_agent"] = get_next_agent(state, "nlp_agent")

        print("✅ NLP analysis completed.")
        return state

    except Exception as e:
        state["has_error"] = True
        state["error_message"] = f"NLP analysis failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}"
        state["can_retry"] = state.get("retry_count", 0) < 3

        return state
