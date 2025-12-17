import json
import re
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent

from mcp_client import get_mcp_client


# ----------------------------
# Helpers
# ----------------------------

def extract_dashboard_json(text: str):
    """
    Extract JSON object from:
      DASHBOARD_JSON: {...}
    Returns dict or None.
    """
    m = re.search(r"DASHBOARD_JSON:\s*(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def normalize_mcp_text(result) -> str:
    """
    MCP tools may return:
      - plain string
      - list of content blocks: [{"type":"text","text":"..."}]
    Convert to usable string.
    """
    if isinstance(result, str):
        return result

    if isinstance(result, list) and result:
        parts = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join([p for p in parts if p]).strip()

    return str(result)


def mcp_json(result) -> dict:
    """
    Normalize MCP tool output into a dict:
      - dict -> dict
      - str (JSON) -> dict
      - list-of-text blocks -> parse text JSON -> dict
    """
    if isinstance(result, dict):
        return result

    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {}

    if isinstance(result, list) and result:
        text = normalize_mcp_text(result)
        try:
            return json.loads(text) if text else {}
        except Exception:
            return {}

    return {}


def is_forecast_intent(q: str) -> bool:
    ql = q.lower()
    keywords = [
        "forecast", "predict", "projection", "project", "estimate",
        "next 7 days", "next week", "next month", "future", "upcoming"
    ]
    return any(k in ql for k in keywords)


def filter_tools_for_role(all_tools, role: str):
    """
    Role-based allowlist:
    - viewer cannot call data/analytics MCP tools
    - admin can call auth/data/analytics MCP tools
    """
    role = (role or "").strip()

    ROLE_ALLOWLIST = {
        "role=admin": {
            "allow_prefixes": ("auth.", "data.", "analytics."),
            "allow_locals": {"authenticate_user", "predict_sales", "build_dashboard_payload_value"},
        },
        "role=viewer": {
            "allow_prefixes": ("auth.",),
            "allow_locals": {"authenticate_user", "predict_sales", "build_dashboard_payload_value"},
        },
    }

    policy = ROLE_ALLOWLIST.get(role, ROLE_ALLOWLIST["role=viewer"])
    allow_prefixes = policy["allow_prefixes"]
    allow_locals = policy["allow_locals"]

    filtered = []
    for t in all_tools:
        tname = getattr(t, "name", "") or ""
        if tname.startswith(allow_prefixes):
            filtered.append(t)
            continue
        if tname in allow_locals:
            filtered.append(t)
            continue
    return filtered


# ----------------------------
# LLM + Local Tools
# ----------------------------

def get_llm():
    return ChatOllama(model="llama3.2", temperature=0.01)


@tool(description="Authenticate a user_id and return their role.")
def authenticate_user(user_id: str) -> str:
    if user_id.lower() in ["admin", "john", "manager"]:
        return "role=admin"
    return "role=viewer"


@tool(description="Predict sales between two dates using a demo forecasting model.")
def predict_sales(start_date: str, end_date: str) -> str:
    return f"Predicted sales from {start_date} to {end_date}: 1,250,000"


@tool(description="Build a JSON payload for BI dashboards using an explicit numeric value")
def build_dashboard_payload_value(metric_name: str, value: float, unit: str = "INR", trend: str = "up") -> dict:
    return {"metric": metric_name, "value": value, "unit": unit, "trend": trend}


def get_master_agent(tools):
    llm = get_llm()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful master agent for business operations. "
            "Use tools when needed. Always output:\n"
            "ANSWER: <text>\n"
            "DASHBOARD_JSON: <valid JSON object or null>\n"
        ),
    )
    return agent


# ----------------------------
# FastAPI
# ----------------------------

app = FastAPI(title="Master Agent Demo")


class Query(BaseModel):
    user_id: str
    question: str


@app.post("/ask")
async def ask(q: Query) -> Dict[str, Any]:
    # --- MCP tools ---
    mcp_client = get_mcp_client()
    mcp_tools = await mcp_client.get_tools()

    # --- role lookup via MCP auth tool ---
    auth_tool = next(t for t in mcp_tools if t.name.endswith("get_role"))
    role_raw = await auth_tool.ainvoke({"user_id": q.user_id})
    role = normalize_mcp_text(role_raw).strip()

    ql = q.question.lower()
    forecast_requested = is_forecast_intent(q.question)

    # --- Deterministic deny for viewer on financial questions ---
    if role.startswith("role=viewer") and (("revenue" in ql) or ("sales" in ql) or forecast_requested):
        dashboard = {
            "metric": "permission_denied",
            "value": 0,
            "unit": "INR",
            "trend": "flat",
            "reason": "viewer_role_no_financial_access",
        }
        final_text = (
            "ANSWER: You do not have permission to access revenue or forecasting data.\n"
            f"DASHBOARD_JSON: {json.dumps(dashboard)}"
        )
        return {"answer": final_text, "role": role, "dashboard": dashboard}

    # --- date extraction (deterministic BI flows require two ISO dates) ---
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", q.question)
    start_date = dates[0] if len(dates) >= 1 else None
    end_date = dates[1] if len(dates) >= 2 else None

    # --- Forecast path (deterministic) ---
    if forecast_requested and start_date and end_date:
        analytics_tool = next((t for t in mcp_tools if t.name.endswith("forecast_sales")), None)
        if analytics_tool is None:
            dashboard = {"metric": "error", "value": 0, "unit": "INR", "trend": "flat", "reason": "analytics_tool_missing"}
            final_text = "ANSWER: Analytics service is unavailable.\nDASHBOARD_JSON: " + json.dumps(dashboard)
            return {"answer": final_text, "role": role, "dashboard": dashboard}

        forecast_raw = await analytics_tool.ainvoke({"start_date": start_date, "end_date": end_date, "method": "naive"})
        forecast = mcp_json(forecast_raw)

        value = forecast.get("value")
        dashboard = {
            "metric": forecast.get("metric", "forecast_revenue"),
            "start_date": forecast.get("start_date", start_date),
            "end_date": forecast.get("end_date", end_date),
            "method": forecast.get("method", "naive"),
            "value": value,
            "unit": forecast.get("unit", "INR"),
            "trend": forecast.get("trend", "up"),
        }

        if value is None:
            final_text = (
                f"ANSWER: Forecast revenue from {start_date} to {end_date} is unavailable.\n"
                f"DASHBOARD_JSON: {json.dumps(dashboard)}"
            )
            return {"answer": final_text, "role": role, "dashboard": dashboard}

        final_text = (
            f"ANSWER: The forecast revenue from {start_date} to {end_date} is ₹{int(value):,} {dashboard['unit']}.\n"
            f"DASHBOARD_JSON: {json.dumps(dashboard)}"
        )
        return {"answer": final_text, "role": role, "dashboard": dashboard}

    # --- Historical revenue path (deterministic) ---
    if (("revenue" in ql) or ("sales" in ql)) and start_date and end_date:
        data_tool = next((t for t in mcp_tools if t.name.endswith("query_sales")), None)
        if data_tool is None:
            dashboard = {"metric": "error", "value": 0, "unit": "INR", "trend": "flat", "reason": "data_tool_missing"}
            final_text = "ANSWER: Data service is unavailable.\nDASHBOARD_JSON: " + json.dumps(dashboard)
            return {"answer": final_text, "role": role, "dashboard": dashboard}

        data_raw = await data_tool.ainvoke({"start_date": start_date, "end_date": end_date, "role": role})
        data_result = mcp_json(data_raw)

        total_revenue = None

        # rows mode: sum per-day revenues
        if isinstance(data_result, dict) and data_result.get("mode") == "rows" and isinstance(data_result.get("data"), list):
            total = 0
            for r in data_result["data"]:
                if not isinstance(r, dict):
                    continue
                rev = r.get("revenue", 0)
                try:
                    total += int(float(rev))
                except Exception:
                    continue
            total_revenue = total

        # aggregate mode: direct total
        elif isinstance(data_result, dict) and isinstance(data_result.get("data"), dict):
            total_revenue = data_result["data"].get("total_revenue")

        dashboard = {
            "metric": f"total_revenue_{start_date}_to_{end_date}",
            "value": total_revenue,
            "unit": "INR",
            "trend": "up" if isinstance(total_revenue, int) else "flat",
        }

        if not isinstance(total_revenue, int):
            final_text = (
                f"ANSWER: Revenue data from {start_date} to {end_date} is unavailable.\n"
                f"DASHBOARD_JSON: {json.dumps(dashboard)}"
            )
            return {"answer": final_text, "role": role, "dashboard": dashboard}

        final_text = (
            f"ANSWER: The total revenue from {start_date} to {end_date} is ₹{total_revenue:,}.\n"
            f"DASHBOARD_JSON: {json.dumps(dashboard)}"
        )
        return {"answer": final_text, "role": role, "dashboard": dashboard}

    # --- Agent fallback (general questions) ---
    local_tools = [authenticate_user, build_dashboard_payload_value, predict_sales]
    all_tools = mcp_tools + local_tools
    allowed_tools = filter_tools_for_role(all_tools, role)

    agent = get_master_agent(allowed_tools)

    prompt = (
        f"User role info: {role}\n"
        f"User question: {q.question}\n\n"
        f"Always output EXACTLY:\n"
        f"ANSWER: <one sentence>\n"
        f"DASHBOARD_JSON: <valid JSON object or null>\n"
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})

    if isinstance(result, dict) and "messages" in result and result["messages"]:
        final_text = result["messages"][-1].content
    elif isinstance(result, dict) and "output" in result:
        final_text = result["output"]
    else:
        final_text = str(result)

    dashboard = extract_dashboard_json(final_text)
    return {"answer": final_text, "role": role, "dashboard": dashboard}
