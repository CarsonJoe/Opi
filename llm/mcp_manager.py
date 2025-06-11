from typing import List, Dict
from langchain_core.tools import BaseTool, tool

class MCPManager:
    def __init__(self, server_configs=None, cache_dir=None, force_refresh=False):
        self.tools = []
    
    async def initialize(self):
        print('[MCP] Initializing simplified MCP manager...')
        self.tools = self._get_system_tools()
        print(f'[MCP] ✅ Initialized with {len(self.tools)} basic tools')
    
    def get_tools(self) -> List[BaseTool]:
        return self.tools
    
    def _get_system_tools(self) -> List[BaseTool]:
        @tool
        def get_system_status() -> str:
            """Get current system status."""
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                return f'CPU: {cpu:.1f}%, Memory: {mem.percent:.1f}%'
            except:
                return 'System status unavailable'
        
        @tool
        def get_current_time() -> str:
            """Get current time."""
            from datetime import datetime
            return datetime.now().strftime('Current time: %Y-%m-%d %H:%M:%S')
        
        return [get_system_status, get_current_time]
    
    async def close(self):
        print('[MCP] ✅ MCP manager closed')
