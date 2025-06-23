import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000") as client:
        tools = await client.list_tools()
        print(f"\n🛠️ Tools available: {[tool.name for tool in tools]}")

        result = await client.call_tool("get_secret_message", {})
        print("\n✅ Result:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

