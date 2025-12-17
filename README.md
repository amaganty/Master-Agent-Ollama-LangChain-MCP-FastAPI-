# Master Agent (Ollama + LangChain + MCP + FastAPI)

A fully local, free **Master Agent** that routes across MCP servers and returns structured BI dashboard JSON.

## Features
- Local LLM via Ollama (llama3.2)
- LangChain agent orchestration
- MCP servers over stdio (Auth, Data, Analytics)
- Role-based access (viewer vs admin)
- Deterministic dashboard JSON output
- FastAPI backend with Swagger UI

## Stack
- Ollama
- LangChain / LangGraph
- MCP (stdio)
- FastAPI
- Python 3.11+

## Run locally

### 1) Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
