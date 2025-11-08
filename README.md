# Multi-Agent Data Analysis System

An intelligent data analysis system powered by LangGraph and OpenAI that automatically orchestrates multiple specialized agents to provide comprehensive business insights.

## 🌟 Features

- **Dynamic Workflows**: Automatically designs optimal workflow based on user query
- **Optimization Loops**: Self-optimizing with continuous quality improvements
- **Error Handling & Recovery**: Graceful failure handling with retry logic
- **Business Ready**: Generates executive-level insights and recommendations
- **Extensible**: Easy to add new agents and capabilities
- **Multiple Interfaces**: CLI, Python API, and Streamlit web UI

## 🏗️ Architecture

The system consists of specialized agents orchestrated by LangGraph:

### Core Agents
- **Orchestrator Agent**: Plans optimal workflow based on user query
- **Data Acquisition Agent**: Loads and validates datasets
- **Data Query Agent**: Generates and executes data queries
- **EDA Agent**: Performs exploratory data analysis
- **Recommendation Agent**: Builds recommendation systems
- **Clustering Agent**: Performs segmentation and pattern discovery
- **NLP Agent**: Processes text data for insights
- **Optimization Agent**: Reviews and improves analytical outputs
- **Business Insight Agent**: Translates findings into business strategies
- **Visualization Agent**: Creates compelling visualizations
- **Debugger Agent**: Handles errors and implements retry logic

## 📁 Project Structure

```
multi-agent-system/
├── agents/                 # Agent modules
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── data_acquisition.py
│   ├── data_query.py
│   ├── eda.py
│   ├── recommendation.py
│   ├── clustering.py
│   ├── nlp.py
│   ├── optimization.py
│   ├── business_insight.py
│   ├── visualization.py
│   └── debugger.py
├── config.py              # Configuration and settings
├── state.py               # State definition (AgentState)
├── graph.py               # Graph construction logic
├── graph_utils.py         # Graph utility functions
├── utils.py               # Data utilities
├── main.py                # CLI entry point
├── streamlit_app.py       # Streamlit web UI
├── requirements.txt       # Dependencies
├── .env.example          # Environment variable template
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Usage

#### Streamlit Web UI

```bash
# Launch Streamlit app
streamlit run streamlit_app.py
```

Then open browser to `http://localhost:8501`

## 💡 Example Queries

- "How many customers are there by state?"
- "What product categories have the highest review scores?"
- "What categories are frequently bought together with health_beauty?"
- "Show customer distribution by city"
- "Perform clustering analysis on customer segments"
- "What are the sentiment patterns in customer reviews?"

## 🔧 Configuration

### config.py

Main configuration file containing:
- API keys and model settings
- Agent descriptions
- System parameters (max optimization cycles, retry attempts)
- File patterns for data discovery

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `DATA_PATH`: Path to your dataset directory

## 📊 Supported Data Formats

- CSV (.csv)
- Parquet (.parquet)
- Excel (.xlsx, .xls)
- JSON (.json)

## 🛠️ Customization

### Adding New Agents

1. Create new agent module in `agents/` directory
2. Import in `agents/__init__.py`
3. Add to `AGENT_MAPPINGS` in `graph.py`
4. Update `AGENT_DESCRIPTIONS` in `config.py`


### Modifying Workflows

Edit the orchestrator prompt in `agents/orchestrator.py` to customize how workflows are planned.

## 🧪 Testing

```bash
# Run with a simple query
python main.py "Show me the data overview"

# Run with complex analysis
python main.py "Perform customer segmentation and provide business insights"
```

## 📝 Development

The codebase follows a modular, functional approach:
- Each agent is a standalone function
- State is passed between agents via `AgentState` TypedDict
- No class-based architecture (easy to modify individual functions)
- Clear separation of concerns


## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/)
- UI with [Streamlit](https://streamlit.io/)

## 📧 Contact

For questions or support, please open an issue on GitHub.
