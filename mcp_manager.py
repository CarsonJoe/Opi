# llm/mcp_manager.py
"""
MCP Manager for Opi Voice Assistant
Handles initialization and management of MCP servers and tools
"""

import asyncio
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

from mcp import StdioServerParameters, types, ClientSession
from mcp.client.stdio import stdio_client
from langchain_core.tools import BaseTool
import aiosqlite

from config.settings import MCPServerConfig


class MCPToolCache:
    """Cache for MCP tools to avoid re-initialization."""
    
    def __init__(self, cache_dir: str, expiry_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_hours = expiry_hours
    
    def _get_cache_key(self, server_config: MCPServerConfig) -> str:
        """Generate cache key for server config."""
        key_parts = [server_config.command] + server_config.args
        return "-".join(key_parts).replace("/", "-").replace(" ", "_")
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"mcp_tools_{cache_key}.json"
    
    def get_cached_tools(self, server_config: MCPServerConfig) -> Optional[List[types.Tool]]:
        """Get cached tools if available and not expired."""
        cache_key = self._get_cache_key(server_config)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_time > timedelta(hours=self.expiry_hours):
                return None
            
            return [types.Tool(**tool) for tool in cache_data["tools"]]
        except Exception:
            return None
    
    def save_tools_cache(self, server_config: MCPServerConfig, tools: List[types.Tool]):
        """Save tools to cache."""
        cache_key = self._get_cache_key(server_config)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "tools": [tool.model_dump() for tool in tools]
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)


class MCPTool(BaseTool):
    """LangChain-compatible MCP tool wrapper."""
    
    def __init__(self, mcp_tool: types.Tool, session: ClientSession, server_name: str):
        self.mcp_tool = mcp_tool
        self.session = session
        self.server_name = server_name
        
        super().__init__(
            name=f"{server_name}_{mcp_tool.name}",
            description=mcp_tool.description,
            handle_tool_error=True
        )
    
    def _run(self, **kwargs) -> str:
        """Synchronous run - not supported for MCP tools."""
        raise NotImplementedError("MCP tools only support async execution")
    
    async def _arun(self, **kwargs) -> str:
        """Execute the MCP tool asynchronously."""
        try:
            result = await self.session.call_tool(
                self.mcp_tool.name, 
                arguments=kwargs
            )
            
            if result.isError:
                return f"Error: {result.content}"
            
            # Format the response content
            if isinstance(result.content, list):
                formatted_content = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        formatted_content.append(item.text)
                    else:
                        formatted_content.append(str(item))
                return "\n".join(formatted_content)
            else:
                return str(result.content)
                
        except Exception as e:
            return f"Tool execution error: {str(e)}"


class MCPServerManager:
    """Manages a single MCPserver connection."""
    
    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.client = None
        self.tools: List[MCPTool] = []
        self._init_lock = asyncio.Lock()
    
    async def initialize(self, tool_cache: MCPToolCache, force_refresh: bool = False):
        """Initialize the MCP server connection and tools."""
        if not self.config.enabled:
            return
        
        async with self._init_lock:
            if self.session and not force_refresh:
                return
            
            # Try to get cached tools first
            if not force_refresh:
                cached_tools = tool_cache.get_cached_tools(self.config)
                if cached_tools:
                    await self._connect()
                    for tool in cached_tools:
                        if tool.name not in self.config.exclude_tools:
                            mcp_tool = MCPTool(tool, self.session, self.name)
                            self.tools.append(mcp_tool)
                    return
            
            # Connect and fetch tools
            await self._connect()
            await self._fetch_tools(tool_cache)
    
    async def _connect(self):
        """Establish connection to MCP server."""
        if self.session:
            return
        
        # Prepare environment
        env = {**os.environ, **self.config.env}
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env
        )
        
        try:
            self.client = stdio_client(server_params)
            read, write = await self.client.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            await self.session.initialize()
            
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server {self.name}: {e}")
    
    async def _fetch_tools(self, tool_cache: MCPToolCache):
        """Fetch tools from the MCP server."""
        try:
            tools_result = await self.session.list_tools()
            
            # Cache the tools
            tool_cache.save_tools_cache(self.config, tools_result.tools)
            
            # Create LangChain tools
            for tool in tools_result.tools:
                if tool.name not in self.config.exclude_tools:
                    mcp_tool = MCPTool(tool, self.session, self.name)
                    self.tools.append(mcp_tool)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tools from {self.name}: {e}")
    
    async def close(self):
        """Close the MCP server connection."""
        try:
            if self.session:
                await asyncio.wait_for(
                    self.session.__aexit__(None, None, None),
                    timeout=2.0
                )
        except (asyncio.TimeoutError, Exception):
            pass
        
        try:
            if self.client:
                await asyncio.wait_for(
                    self.client.__aexit__(None, None, None),
                    timeout=2.0
                )
        except (asyncio.TimeoutError, Exception):
            pass
        
        self.session = None
        self.client = None


class MCPManager:
    """Main MCP manager that handles all MCP servers and tools."""
    
    def __init__(self, server_configs: Dict[str, MCPServerConfig], 
                 cache_dir: str = "~/.cache/opi", force_refresh: bool = False):
        self.server_configs = server_configs
        self.cache_dir = Path(cache_dir).expanduser()
        self.force_refresh = force_refresh
        
        self.tool_cache = MCPToolCache(str(self.cache_dir), expiry_hours=24)
        self.servers: Dict[str, MCPServerManager] = {}
        self.all_tools: List[BaseTool] = []
        
    async def initialize(self):
        """Initialize all MCP servers and their tools."""
        print(f"[MCP] Initializing {len(self.server_configs)} MCP servers...")
        
        # Create server managers
        for name, config in self.server_configs.items():
            if config.enabled:
                self.servers[name] = MCPServerManager(name, config)
        
        # Initialize servers concurrently
        init_tasks = []
        for server in self.servers.values():
            task = asyncio.create_task(
                server.initialize(self.tool_cache, self.force_refresh)
            )
            init_tasks.append(task)
        
        # Wait for all servers to initialize
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Collect tools and report results
        successful_servers = 0
        total_tools = 0
        
        for i, (server_name, result) in enumerate(zip(self.servers.keys(), results)):
            if isinstance(result, Exception):
                print(f"[MCP] ❌ Failed to initialize {server_name}: {result}")
            else:
                server = self.servers[server_name]
                tool_count = len(server.tools)
                self.all_tools.extend(server.tools)
                total_tools += tool_count
                successful_servers += 1
                print(f"[MCP] ✅ {server_name}: {tool_count} tools loaded")
        
        print(f"[MCP] Initialized {successful_servers}/{len(self.servers)} servers, {total_tools} tools total")
    
    def get_tools(self) -> List[BaseTool]:
        """Get all available MCP tools."""
        return self.all_tools
    
    def get_tools_by_server(self, server_name: str) -> List[BaseTool]:
        """Get tools from a specific server."""
        if server_name in self.servers:
            return self.servers[server_name].tools
        return []
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
        return [tool.name for tool in self.all_tools]
    
    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a specific tool by name."""
        for tool in self.all_tools:
            if tool.name == tool_name:
                return await tool._arun(**kwargs)
        
        raise ValueError(f"Tool '{tool_name}' not found")
    
    async def close(self):
        """Close all MCP server connections."""
        print("[MCP] Closing MCP server connections...")
        
        close_tasks = []
        for server in self.servers.values():
            task = asyncio.create_task(server.close())
            close_tasks.append(task)
        
        await asyncio.gather(*close_tasks, return_exceptions=True)
        print("[MCP] ✅ All MCP servers closed")


# Built-in Opi system tools
class OpiSystemTools:
    """Built-in system tools for Opi voice assistant."""
    
    @staticmethod
    def get_tools() -> List[BaseTool]:
        """Get list of built-in Opi system tools."""
        from langchain_core.tools import tool
        
        @tool
        async def get_system_status() -> str:
            """Get current system status including CPU, memory, and temperature."""
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Try to get temperature (Raspberry Pi / Orange Pi)
                temp = "N/A"
                temp_paths = [
                    "/sys/class/thermal/thermal_zone0/temp",
                    "/sys/class/thermal/thermal_zone1/temp"
                ]
                
                for path in temp_paths:
                    try:
                        with open(path, 'r') as f:
                            temp_raw = int(f.read().strip())
                            temp = f"{temp_raw / 1000:.1f}°C"
                            break
                    except:
                        continue
                
                return f"""System Status:
- CPU Usage: {cpu_percent:.1f}%
- Memory Usage: {memory.percent:.1f}% ({memory.used // 1024 // 1024}MB / {memory.total // 1024 // 1024}MB)
- Temperature: {temp}
- Available Memory: {memory.available // 1024 // 1024}MB"""
            
            except Exception as e:
                return f"Error getting system status: {e}"
        
        @tool
        async def list_audio_devices() -> str:
            """List available audio input and output devices."""
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                
                output = "Audio Devices:\n"
                for i, device in enumerate(devices):
                    device_type = []
                    if device['max_input_channels'] > 0:
                        device_type.append("Input")
                    if device['max_output_channels'] > 0:
                        device_type.append("Output")
                    
                    output += f"  {i}: {device['name']} ({', '.join(device_type)}) - {device['default_samplerate']}Hz\n"
                
                return output
            except Exception as e:
                return f"Error listing audio devices: {e}"
        
        @tool
        async def get_network_info() -> str:
            """Get network interface information."""
            try:
                import subprocess
                import json
                
                # Get IP addresses
                result = subprocess.run(['ip', '-j', 'addr'], capture_output=True, text=True)
                if result.returncode == 0:
                    interfaces = json.loads(result.stdout)
                    
                    output = "Network Interfaces:\n"
                    for iface in interfaces:
                        if iface['operstate'] == 'UP' and iface['ifname'] != 'lo':
                            output += f"  {iface['ifname']}: "
                            
                            addrs = []
                            for addr_info in iface.get('addr_info', []):
                                if addr_info['family'] in ['inet', 'inet6']:
                                    addrs.append(f"{addr_info['local']}/{addr_info['prefixlen']}")
                            
                            output += ", ".join(addrs) if addrs else "No IP"
                            output += "\n"
                    
                    return output
                else:
                    return "Could not get network information"
                    
            except Exception as e:
                return f"Error getting network info: {e}"
        
        return [get_system_status, list_audio_devices, get_network_info] 
