# Master Agent (Ollama + LangChain + MCP + FastAPI)

A fully local, free **Master Agent** that routes across MCP servers and returns structured BI dashboard JSON.

---

## Features
- Local LLM via Ollama (llama3.2)
- LangChain agent orchestration
- MCP servers over stdio (Auth, Data, Analytics)
- Role-based access (viewer vs admin)
- Deterministic dashboard JSON output
- FastAPI backend with Swagger UI

---

## Tech Stack
- Ollama
- LangChain / LangGraph
- MCP (stdio)
- FastAPI
- Python 3.11+

---

## Run Locally

### Setup
```bash
python -m venv venv
```

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

---

### Start Ollama
```bash
ollama pull llama3.2
```

---

### Run API
```bash
uvicorn app:app --reload
```

Swagger UI:  
http://127.0.0.1:8000/docs

---

## Example Request
```json
{
  "user_id": "admin",
  "question": "What was total revenue from 2025-12-01 to 2025-12-07? Create a dashboard tile."
}
```

## Example Response
```json
{
  "answer": "ANSWER: ...",
  "role": "role=admin",
  "dashboard": {}
}
```

---

## Notes
- Viewer role is denied revenue and forecast access
- Admin role can query revenue and forecasts
- MCP outputs are normalized from stdio text blocks into JSON
