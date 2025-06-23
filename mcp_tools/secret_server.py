# mcp_tools/secret_server.py
"""
Updated MCP Server using FastMCP 2.x
Provides secret messages and system status tools
"""

from fastmcp import FastMCP
import random
import datetime
import os
import psutil
import sys

# Create the FastMCP server instance
mcp = FastMCP("Secret Message Server")

SECRET_MESSAGES = [
    "The secret ingredient is always love! 💝",
    "42 is the answer to life, the universe, and everything. 🌌",
    "The cake is a lie, but the cookies are real. 🍪",
    "In a world of algorithms, be the poetry. 🎭",
    "The best debugging happens at 3 AM with coffee. ☕",
    "Code is poetry that compiles. 📝"
]

@mcp.tool()
def get_secret_message() -> str:
    """Get a random secret message."""
    return f"🔐 Secret: {random.choice(SECRET_MESSAGES)}"

@mcp.tool()
def count_secrets() -> int:
    """Count the total number of available secret messages."""
    return len(SECRET_MESSAGES)

@mcp.tool()
def get_secret_by_number(number: int) -> str:
    """Get a specific secret message by its number (1-based index)."""
    if 1 <= number <= len(SECRET_MESSAGES):
        return f"🔐 Secret #{number}: {SECRET_MESSAGES[number - 1]}"
    else:
        return f"❌ Secret #{number} doesn't exist. Available: 1-{len(SECRET_MESSAGES)}"

@mcp.tool()
def get_system_status() -> dict:
    """Get current system status information."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "stressed"
        }
    except Exception as e:
        return {"error": f"Failed to get system status: {str(e)}"}

@mcp.tool()
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@mcp.tool()
def echo_message(message: str) -> str:
    """Echo back a message (useful for testing)."""
    return f"🔄 Echo: {message}"

# Add a resource for server information
@mcp.resource("server://info")
def get_server_info() -> str:
    """Get information about this MCP server."""
    return f"""
# Secret Message Server Info

**Server Name:** Secret Message Server
**Version:** 2.0 (FastMCP 2.x)
**Available Tools:** {len([get_secret_message, count_secrets, get_secret_by_number, get_system_status, get_current_time, echo_message])}
**Total Secrets:** {len(SECRET_MESSAGES)}
**Runtime:** FastMCP 2.x
**Status:** 🟢 Active
"""

# Add a prompt for getting help
@mcp.prompt()
def help_with_secrets() -> str:
    """Get help on using the secret message tools."""
    return """I need help using the secret message tools. What can I do with this server?

Please explain:
1. How to get random secret messages
2. How to get specific secrets by number
3. What other tools are available
4. How to check system status

Provide examples of how to use each tool."""

if __name__ == "__main__":
    print("🔐 Starting Secret Message MCP Server...")
    
    # Check for command line arguments
    transport = "http"  # Default
    
    if len(sys.argv) > 1:
        if "--http" in sys.argv:
            transport = "http"
        elif "--sse" in sys.argv:
            transport = "sse"
        elif "--stdio" in sys.argv:
            transport = "stdio"
    
    print("Available transports:")
    print("  - STDIO (default): python secret_server.py --stdio")
    print("  - HTTP: python secret_server.py --http")
    print("  - SSE: python secret_server.py --sse")
    print()
    
    if transport == "http":
        print("🌐 Starting HTTP transport on 127.0.0.1:8000...")
        print("📍 Server URL: http://127.0.0.1:8000")
        print("💡 Update your config.json to use this HTTP URL")
        mcp.run('streamable-http', host='127.0.0.1', port=8000)
    
    elif transport == "sse":
        print("📡 Starting SSE transport on 127.0.0.1:8001...")
        print("📍 Server URL: http://127.0.0.1:8001")
        mcp.run('sse', host='127.0.0.1', port=8001)
    
    else:
        print("🔌 Starting STDIO transport (for local MCP clients)...")
        print("💡 Use this with your Opi config.json stdio transport")
        # Default STDIO transport for local development
        mcp.run()
