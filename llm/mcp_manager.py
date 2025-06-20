from fastmcp import Client
import asyncio


class MCPManager:
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.tools = []
        self.failed = []

    @property
    def connected_servers(self):
        return list(self.clients.keys())

    async def connect_all(self):
        servers = self.config or {}
        self.failed = []

        for name, server_cfg in servers.items():
            if not server_cfg.get("enabled", True):
                continue
            url = server_cfg.get("url")
            if not url:
                print(f"[WARN] Skipping MCP server '{name}': no URL")
                continue

            try:
                print(f"[INFO] Connecting to MCP server '{name}' at {url}")
                client = Client(url)
                await client.__aenter__()  # enter async context manually
                self.clients[name] = client

                tools = await client.list_tools()
                self.tools.extend(tools)

                for tool in tools:
                    tname = tool['name'] if isinstance(tool, dict) else tool.name
                    print(f"[MCP] ✅ Registered tool from '{name}': {tname}")

            except Exception as e:
                print(f"[ERROR] ❌ Failed to connect to MCP server '{name}': {e}")
                self.failed.append(name)

        print(f"[MCP] ✅ Connected to {len(self.clients)} MCP servers")

    def get_tools(self):
        return self.tools

    def get_server_status(self):
        return {
            "connected_servers": len(self.clients),
            "total_servers": len(self.config),
            "failed_servers": self.failed,
        }

    async def close(self):
        for name, client in self.clients.items():
            try:
                await client.__aexit__(None, None, None)
                print(f"[MCP] 🔌 Closed MCP client: {name}")
            except Exception as e:
                print(f"[WARN] Failed to close MCP client '{name}': {e}")
        self.clients = {}
        self.tools = []
        print("[MCP] ✅ All MCP clients closed")

