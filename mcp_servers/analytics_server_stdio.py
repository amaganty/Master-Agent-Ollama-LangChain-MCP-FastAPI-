from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("analytics")

def _parse_yyyy_mm_dd(s:str) -> datetime:
    # Keep parsing strict to avoid ambiguous formats
    return datetime.strptime(s, "%Y-%m-%d")

@mcp.tool()

def forecast_sales(start_date: str, end_date: str, method: str = "naive") -> dict:
    '''
    Return a forecast payload for the requested window.
    method: naive | trend (can extend later)
    '''
    sd = _parse_yyyy_mm_dd(start_date)
    ed = _parse_yyyy_mm_dd(end_date)
    days = max(1, (ed - sd).days)

    # Simple placeholder logic (replace l8r with real model/ stats)
    base = 1250000
    if method == "trend":
        value = int(base * (1.0 + min(0.2, days * 3650))) # gentle upward drift
    else:
        value = base
    
    return {
        "metric": "forecast_revenue",
        "start_date" : start_date,
        "end_date" : end_date,
        "method" : method,
        "value" : value,
        "unit": "INR",
        "trend" : "up" if value >= base else "down",
    }
if __name__ == "__main__":
    mcp.run()

