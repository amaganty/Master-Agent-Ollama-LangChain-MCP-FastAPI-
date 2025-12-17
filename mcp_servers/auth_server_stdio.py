from mcp.server.fastmcp import FastMCP

mcp = FastMCP("auth")

USER_ROLES = {
    "admin": "role=admin",
    "john": "role=admin",
    "manager": "role=admin",
}

@mcp.tool()
def get_role(user_id: str) -> str:
    '''Return role for given user_id'''
    return USER_ROLES.get(user_id.lower(),"role=viewer")

if __name__ == "__main__":
    mcp.run()