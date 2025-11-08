"""
Optimization Agent
Reviews analytical outputs and suggests improvements
"""
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import traceback

from state import AgentState
from config import get_llm, MAX_OPTIMIZATION_CYCLES
from utils import generate_schema_summary
from graph_utils import get_next_agent

# Agent-specific optimization focus areas
specialization_prompt = { 
    'recommendation_agent': """
    Focus on these optimization areas for recommendation systems, only when needed:
    1. ALGORITHM_SELECTION: Compare Apriori vs FP-Growth vs ECLAT, collaborative filtering, content-based filtering
    2. HYPERPARAMETER_TUNING: Optimal min_support, min_confidence, min_lift, similarity metrics
    3. FEATURE_ENGINEERING: Temporal features, user context, product attributes, embedding strategies
    4. EVALUATION_METRICS: Precision@K, Recall@K, NDCG, MAP, diversity metrics
    5. SCALABILITY: Handling sparse data, distributed computing approaches
    6. ALTERNATIVE_METHODS: Matrix factorization and more

    Provide specific parameter ranges and algorithm alternatives.
    """,
    
    'clustering_agent': """
    Focus on these optimization areas for clustering algorithms, only when needed:
    1. ALGORITHM_SELECTION: K-Means vs DBSCAN vs Hierarchical vs Gaussian Mixture Models or more
    2. HYPERPARAMETER_TUNING: Optimal k, epsilon, min_samples, linkage criteria
    3. FEATURE_PREPROCESSING: Scaling, normalization, dimensionality reduction (PCA, t-SNE)
    4. DISTANCE_METRICS: Euclidean vs Manhattan vs Cosine similarity, custom distance functions
    5. CLUSTER_VALIDATION: Silhouette score, Davies-Bouldin, Calinski-Harabasz, elbow method
    6. OUTLIER_DETECTION: Handling noise points, robust clustering methods

    Suggest specific methods for determining optimal cluster count.
    """,
    
    'nlp_agent': """
    Focus on these optimization areas for natural language processing, only when needed:
    1. ALGORITHM_SELECTION: TF-IDF vs Word2Vec vs BERT, traditional vs transformer approaches
    2. TEXT_PREPROCESSING: Tokenization strategies, stopword removal, lemmatization vs stemming
    3. FEATURE_EXTRACTION: N-gram ranges, vectorization parameters, embedding dimensions
    4. TOPIC_MODELING_OPTIONS: LDA vs NMF vs BERTopic, optimal topic count determination
    5. SENTIMENT_ANALYSIS: Rule-based vs ML-based, ensemble sentiment approaches
    6. PERFORMANCE_OPTIMIZATION: Batch processing, efficient text cleaning, parallelization

    Suggest vocabulary size optimization and model complexity trade-offs.
    """
}
    

def optimization_agent(state: AgentState):
    """
    Provides specific technical optimizations for different agents.
    Focuses on methods, algorithms, features, processes, and hyperparameters and more.
    """
    current_cnt = state.get("opt_cnt", 0) + 1
    
    print("\n\n"+"="*80)
    print(f"\n🤖 Running optimization cycle {current_cnt}/{MAX_OPTIMIZATION_CYCLES}")
    
    # Check max optimization cycles
    if current_cnt > MAX_OPTIMIZATION_CYCLES:
        print(f"⚠️ Max optimization cycles ({MAX_OPTIMIZATION_CYCLES}) reached. Skipping.")
        state["opt_approval"] = True
        state["current_agent"] = get_next_agent(state, "optimization_agent")
        return state

    user_query = state["user_prompt"]
    datasets = state["datasets"]
    workflow = state["workflow_plan"]
    code = state["task_code"]
    result = state["task_result"]
    
    spec_prompt = ""
    if workflow:
        for i in range(len(workflow)):
            if workflow[i]["agent"] in specialization_prompt.keys():
                spec_prompt = specialization_prompt[workflow[i]["agent"]]
    
    base_prompt = """
You are an expert Data Scientist specializing in optimizing data science approaches.
Review the current approach used by the agent and decide if it should be approved as-is, or if technical optimizations are required.
    
    # Respond only with valid JSON like this, with NO markdown and extra text:
    {
  "opt_approval": true | false, 
  "suggestions": "Use ..., tune ... (<80 words)"
      }
    - Return opt_approval=True if the results correctly address the user query, are relevant, and are technically sound.
    - Provides suggestions olny when needed and they are essential and practical. Skipping is better than guessing.
    - Don't include phrases like "error" or "issue" in the JSON;
    
    """
    system_message = SystemMessage(content = base_prompt + spec_prompt)
    human_message = HumanMessage(content =f"""     
    PROBLEM CONTEXT:
    - USER REQUEST: {user_query}
    - DATASET SCHEMAS: {generate_schema_summary(datasets)}

    - CURRENT APPROACH (Python code):
    {code}
    
    - APPROACH RESULT:
    {result}
    """
    )
    
    try:
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        llm = get_llm()
        llm_chain = prompt | llm
    
        response = llm_chain.invoke({})
    
        if response.content.startswith("```"):
            response_content = response.content.strip("`").split("\n", 1)[-1]
        else:
            response_content = response.content

        result = json.loads(response_content)
        
        opt_approval = result.get("opt_approval", False)
        optimization_text = result.get("suggestions", "")
    
        state["opt_cnt"] = current_cnt
        state["opt_approval"] = opt_approval
        state["opt_suggestions"] = optimization_text
        state["current_agent"] = get_next_agent(state, "optimization_agent")
    
        # PRINTOUT
        print(f"✓ Approval: {opt_approval}")
        if optimization_text:
            print(f"🔧 Suggestions: {optimization_text}")
    
        print("\n✅ Optimization completed.")
    
    except Exception as e:
        # If optimization fails, approve and continue
        print(f"⚠️ Optimization review failed: {str(e)}\nERROR TRACEBACK:{traceback.format_exc()}")
        state["opt_approval"] = True
        state["current_agent"] = get_next_agent(state, "optimization_agent")
            
    return state