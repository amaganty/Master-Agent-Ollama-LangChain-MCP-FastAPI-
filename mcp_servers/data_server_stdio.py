from mcp.server.fastmcp import FastMCP

mcp = FastMCP("data")

SALES = [
    {"date": "2025-12-01", "revenue": 120000, "region": "IN"},
    {"date": "2025-12-02", "revenue": 98000,  "region": "IN"},
    {"date": "2025-12-03", "revenue": 143000, "region": "IN"},
    {"date": "2025-12-04", "revenue": 110000, "region": "IN"},
]

@mcp.tool()
def query_sales(start_date: str, end_date: str, role: str) -> dict:
    '''
    Return sales data between start_date and end_date.
    Security rule:
    - admin/manager/analyst get raw rows
    - viewers get aggregated only
    '''

    rows = [r for r in SALES if start_date <=r["date"] <= end_date]

    if role in ("role=admin", "role=manager", "role=analyst"):
        return {"mode": "rows", "data": rows}
    
    total = sum(r["revenue"] for r in rows)
    return {"mode": "aggregate", "data": {"total_revenue": total, "days": len(rows)}}

if __name__ == "__main__":
    mcp.run()