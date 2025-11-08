"""
Configuration module for Multi-Agent System
Contains all configuration variables, agent descriptions, and LLM setup
"""
import os
from langchain_openai import ChatOpenAI

# ============================================================================
# API CONFIGURATION
# ============================================================================

# OpenAI API Key - Load from environment variable or set directly
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Default data path - can be overridden
DATA_PATH = os.getenv("DATA_PATH", "/Users/rootx/DataScience_Std/iLab/dataset")

# ============================================================================
# AGENT DESCRIPTIONS
# ============================================================================

# Agent descriptions - Edit to customize behavior
AGENT_DESCRIPTIONS = {
    "data_acquisition": "Loads, validates, and initializes datasets from multiple file formats - the entry point for all data processing",
    "data_query_agent": "Specialized in data extraction, joining, filtering, and preparing business-ready datasets - focuses on data manipulation and data query",
    "eda_agent": "Performs data overview, exploratory data analysis, statistical summaries, visualization, and pattern discovery - understands data distributions and relationships",
    "recommendation_agent": "Handles product associations, market basket analysis, collaborative filtering, and recommendation systems",
    "clustering_agent": "Specialized in customer segmentation, product grouping, and pattern discovery using clustering algorithms",
    "nlp_agent": "Processes text data for sentiment analysis, topic modeling, and natural language understanding",
    "optimization_agent": "Reviews analytical outputs, suggests methodological improvements, parameter tuning, and quality enhancement strategies",
    "business_insight_agent": "Translates analytical findings into actionable business strategies, ROI calculations, and strategic recommendations",
    "visualization_agent": "Provide visualization support for business insight translations",
    "debugger_agent": "Handles pipeline errors, implements retry logic, and ensures system stability and graceful failure recovery"
}

# Functional nodes that perform core analytical tasks
FUNCTIONAL_NODES = [
    "data_query_agent",
    "eda_agent",
    "recommendation_agent",
    "clustering_agent",
    "nlp_agent"
]

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Default LLM model and settings
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0

def get_llm(model: str = None, temperature: float = None, api_key: str = None):
    """
    Get configured LLM instance
    Args:
        model: Model name (default: LLM_MODEL)
        temperature: Temperature setting (default: LLM_TEMPERATURE)
        api_key: API key (default: OPENAI_API_KEY)
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model or LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        api_key=api_key or OPENAI_API_KEY
    )

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Maximum optimization cycles
MAX_OPTIMIZATION_CYCLES = 3

# Maximum retry attempts for error recovery
MAX_RETRY_ATTEMPTS = 3

# Supported file patterns for data discovery
SUPPORTED_FILE_PATTERNS = ['*.csv', '*.parquet', '*.xlsx', '*.json']
