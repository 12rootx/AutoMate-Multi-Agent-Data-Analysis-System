"""
Agent modules for Multi-Agent System
Each agent is a specialized function that performs a specific task
"""
from .orchestrator import agent_orchestrator
from .data_acquisition import data_acquisition_agent
from .data_query import data_query_agent
from .eda import eda_agent
from .recommendation import recommendation_agent
from .clustering import clustering_agent
from .nlp import nlp_agent
from .optimization import optimization_agent
from .business_insight import business_insight_agent
from .visualization import visualization_agent
from .debugger import debugger_agent

__all__ = [
    'agent_orchestrator',
    'data_acquisition_agent',
    'data_query_agent',
    'eda_agent',
    'recommendation_agent',
    'clustering_agent',
    'nlp_agent',
    'optimization_agent',
    'business_insight_agent',
    'visualization_agent',
    'debugger_agent'
]
