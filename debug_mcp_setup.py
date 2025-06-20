#!/usr/bin/env python3
import json
import asyncio
import os

try:
    from fastmcp import Client
except ImportError:
    print("[ERROR] ❌ Could not import fastmcp.Client. Run 'pip install fastmcp'")
    exit(1)

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

async def test_http_mcp_connection():
    print("[INFO] 🔍 Testing HTTP MCP connection…")
    config = load_config()
    server = config.get("mcp", {}).get("servers", {}).get("secret_server", {})
    url = server.get("url")
    if not url or not url.startswith("http://"):
        print(f"[ERROR] ❌ Invalid or missing URL in config: {url}")
        return

    print(f"[INFO] Connecting to {url}…")
    try:
        async with Client(url) as client:
            tools = await client.list_tools()
            tool_names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
            print("✅ Connected! Available tools:", ", ".join(tool_names))
            # optional: call a tool
            # result = await client.call_tool("get_secret_message", {})
            # print("Tool call result:", result)
        print("[SUCCESS] 🎉 MCP HTTP test passed!")
    except Exception as e:
        print("[ERROR] ❌ Connection failed:", e)

if __name__ == "__main__":
    asyncio.run(test_http_mcp_connection())

