import sys
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_mcp_client():
    auth_path = os.path.join(BASE_DIR, "mcp_servers", "auth_server_stdio.py")
    data_path = os.path.join(BASE_DIR, "mcp_servers", "data_server_stdio.py")
    analytics_path = os.path.join(BASE_DIR, "mcp_servers", "analytics_server_stdio.py")

    return MultiServerMCPClient(
        {
            "auth": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [auth_path],
            },
            "data": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [data_path],
            },
            "analytics": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [analytics_path],
            }
        }
    )