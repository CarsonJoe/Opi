from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("Secret Message Server")

SECRET_MESSAGES = [
    "The secret ingredient is always love!",
    "42 is the answer to life, the universe, and everything.",
    "The cake is a lie, but the cookies are real."
]

@mcp.tool()
def get_secret_message() -> str:
    return f"🔐 Secret: {random.choice(SECRET_MESSAGES)}"

@mcp.tool()
def count_secrets() -> int:
    return len(SECRET_MESSAGES)

if __name__ == "__main__":
    print("🔐 Starting HTTP MCP Server on port 7777, path '/mcp'...")
    # `transport="streamable-http"` is positional; specify host, port, and path in correct order
    mcp.run("streamable-http", 7777)
    print("[DEBUG] FastMCP run completed")

