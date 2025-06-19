# llm/mcp_manager.py
"""
Enhanced MCP Manager for Opi Voice Assistant
Proper MCP host implementation with client support for multiple servers
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

from langchain_core.tools import BaseTool, tool
from termcolor import cprint

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[MCP] Warning: MCP Python SDK not installed. Install with: pip install mcp")

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: str  # 'stdio' or 'sse'
    
    # For stdio servers
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    
    # For SSE servers
    url: Optional[str] = None
    
    # Common options
    enabled: bool = True
    description: Optional[str] = None
    tool_prefix: Optional[str] = None  # Prefix for tool names to avoid conflicts

class MCPClient:
    """Wrapper for an individual MCP client connection."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.connected = False
        self.tools: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        self.prompts: List[Dict[str, Any]] = []
        
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            if self.config.transport == 'stdio':
                await self._connect_stdio()
            elif self.config.transport == 'sse':
                await self._connect_sse()
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")
            
            if self.session:
                await self.session.initialize()
                await self._discover_capabilities()
                self.connected = True
                return True
                
        except Exception as e:
            cprint(f"[MCP] Failed to connect to {self.config.name}: {e}", "red")
            return False
        
        return False
    
    async def _connect_stdio(self):
        """Connect via stdio transport."""
        if not self.config.command:
            raise ValueError("Command required for stdio transport")
        
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env
        )
        
        # This creates the connection but we need to manage it differently
        # We'll store the connection info for later use
        self._stdio_params = server_params
    
    async def _connect_sse(self):
        """Connect via SSE transport."""
        if not self.config.url:
            raise ValueError("URL required for SSE transport")
        
        # SSE connection will be established when session is created
        self._sse_url = self.config.url
    
    async def _discover_capabilities(self):
        """Discover server capabilities."""
        if not self.session:
            return
        
        try:
            # List tools
            tools_result = await self.session.list_tools()
            self.tools = [
                {
                    'name': self._add_prefix(tool.name),
                    'original_name': tool.name,
                    'description': tool.description,
                    'inputSchema': tool.inputSchema
                }
                for tool in tools_result.tools
            ]
            
            # List resources
            resources_result = await self.session.list_resources()
            self.resources = [
                {
                   'uri': resource.uri,
                    'name': resource.name,
                    'description': resource.description,
                    'mimeType': getattr(resource, 'mimeType', None)
                }
                for resource in resources_result.resources
            ]
            
            # List prompts
            prompts_result = await self.session.list_prompts()
            self.prompts = [
                {
                    'name': self._add_prefix(prompt.name),
                    'original_name': prompt.name,
                    'description': prompt.description,
                    'arguments': prompt.arguments
                }
                for prompt in prompts_result.prompts
            ]
            
            cprint(f"[MCP] {self.config.name}: {len(self.tools)} tools, {len(self.resources)} resources, {len(self.prompts)} prompts", "green")
            
        except Exception as e:
            cprint(f"[MCP] Error discovering capabilities for {self.config.name}: {e}", "yellow")
    
    def _add_prefix(self, name: str) -> str:
        """Add prefix to tool/prompt name if configured."""
        if self.config.tool_prefix:
            return f"{self.config.tool_prefix}_{name}"
        return name
    
    def _remove_prefix(self, name: str) -> str:
        """Remove prefix from tool/prompt name."""
        if self.config.tool_prefix and name.startswith(f"{self.config.tool_prefix}_"):
            return name[len(self.config.tool_prefix) + 1:]
        return name
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on this server."""
        if not self.session or not self.connected:
            raise RuntimeError(f"Not connected to {self.config.name}")
        
        # Remove prefix for actual call
        original_name = self._remove_prefix(tool_name)
        
        try:
            result = await self.session.call_tool(original_name, arguments)
            return result
        except Exception as e:
            cprint(f"[MCP] Error calling tool {tool_name} on {self.config.name}: {e}", "red")
            raise
    
    async def read_resource(self, uri: str) -> tuple[str, Optional[str]]:
        """Read a resource from this server."""
        if not self.session or not self.connected:
            raise RuntimeError(f"Not connected to {self.config.name}")
        
        try:
            content, mime_type = await self.session.read_resource(uri)
            return content, mime_type
        except Exception as e:
            cprint(f"[MCP] Error reading resource {uri} from {self.config.name}: {e}", "red")
            raise
    
    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, str]] = None) -> Any:
        """Get a prompt from this server."""
        if not self.session or not self.connected:
            raise RuntimeError(f"Not connected to {self.config.name}")
        
        # Remove prefix for actual call
        original_name = self._remove_prefix(prompt_name)
        
        try:
            result = await self.session.get_prompt(original_name, arguments)
            return result
        except Exception as e:
            cprint(f"[MCP] Error getting prompt {prompt_name} from {self.config.name}: {e}", "red")
            raise
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass
        self.connected = False
        self.session = None

class MCPManager:
    """Enhanced MCP Manager with proper host implementation."""
    
    def __init__(self, server_configs: Optional[Dict[str, Any]] = None, verbose: bool = False):
        self.server_configs = server_configs or {}
        self.verbose = verbose
        self.clients: Dict[str, MCPClient] = {}
        self.langchain_tools: List[BaseTool] = []
        
    async def initialize(self):
        """Initialize the MCP manager and connect to all configured servers."""
        if not MCP_AVAILABLE:
            cprint("[MCP] MCP SDK not available, using fallback mode", "yellow")
            self.langchain_tools = self._get_fallback_tools()
            return
        
        cprint("[MCP] Initializing MCP manager with proper host support...", "cyan")
        
        # Parse server configurations
        for server_name, server_config in self.server_configs.items():
            try:
                config = self._parse_server_config(server_name, server_config)
                if config.enabled:
                    client = MCPClient(config)
                    self.clients[server_name] = client
                    
                    if self.verbose:
                        cprint(f"[MCP] Configured server: {server_name} ({config.transport})", "white")
                        
            except Exception as e:
                cprint(f"[MCP] Error configuring server {server_name}: {e}", "red")
        
        # Connect to all servers
        connected_count = 0
        for name, client in self.clients.items():
            if await self._connect_client(client):
                connected_count += 1
        
        # Generate LangChain tools from all connected servers
        self._generate_langchain_tools()
        
        cprint(f"[MCP] ✅ Connected to {connected_count}/{len(self.clients)} MCP servers", "green")
        cprint(f"[MCP] Generated {len(self.langchain_tools)} LangChain tools", "green")
    
    def _parse_server_config(self, name: str, config: Dict[str, Any]) -> MCPServerConfig:
        """Parse a server configuration into MCPServerConfig."""
        if 'command' in config:
            # Stdio server
            return MCPServerConfig(
                name=name,
                transport='stdio',
                command=config['command'],
                args=config.get('args', []),
                env=config.get('env'),
                enabled=config.get('enabled', True),
                description=config.get('description'),
                tool_prefix=config.get('tool_prefix')
            )
        elif 'url' in config:
            # SSE server
            return MCPServerConfig(
                name=name,
                transport='sse',
                url=config['url'],
                enabled=config.get('enabled', True),
                description=config.get('description'),
                tool_prefix=config.get('tool_prefix')
            )
        else:
            raise ValueError(f"Invalid server config for {name}: must have 'command' or 'url'")
    
    async def _connect_client(self, client: MCPClient) -> bool:
        """Connect an individual client with proper session management."""
        try:
            if client.config.transport == 'stdio':
                # For stdio, we need to manage the connection context
                server_params = StdioServerParameters(
                    command=client.config.command,
                    args=client.config.args or [],
                    env=client.config.env
                )
                
                # Create a persistent connection
                read_stream, write_stream = await stdio_client(server_params).__aenter__()
                client.session = ClientSession(read_stream, write_stream)
                await client.session.__aenter__()
                
                # Store connection info for cleanup
                client._connection_context = stdio_client(server_params)
                
            elif client.config.transport == 'sse':
                # For SSE connections
                read_stream, write_stream = await sse_client(client.config.url).__aenter__()
                client.session = ClientSession(read_stream, write_stream)
                await client.session.__aenter__()
                
                # Store connection info for cleanup
                client._connection_context = sse_client(client.config.url)
            
            # Initialize and discover capabilities
            await client.session.initialize()
            await client._discover_capabilities()
            client.connected = True
            
            if self.verbose:
                cprint(f"[MCP] ✅ Connected to {client.config.name}", "green")
            
            return True
            
        except Exception as e:
            cprint(f"[MCP] ❌ Failed to connect to {client.config.name}: {e}", "red")
            return False
    
    def _generate_langchain_tools(self):
        """Generate LangChain tools from all connected MCP servers."""
        self.langchain_tools = []
        
        # Add MCP tools as LangChain tools
        for client_name, client in self.clients.items():
            if not client.connected:
                continue
                
            for tool_info in client.tools:
                langchain_tool = self._create_langchain_tool(client, tool_info)
                self.langchain_tools.append(langchain_tool)
        
        # Add fallback tools
        self.langchain_tools.extend(self._get_fallback_tools())
    
    def _create_langchain_tool(self, client: MCPClient, tool_info: Dict[str, Any]) -> BaseTool:
        """Create a LangChain tool from MCP tool info."""
        
        async def call_mcp_tool(**kwargs):
            """Async wrapper for MCP tool calls."""
            try:
                result = await client.call_tool(tool_info['original_name'], kwargs)
                
                # Extract text content from MCP result
                if hasattr(result, 'content') and result.content:
                    if isinstance(result.content, list) and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            return content_item.text
                        elif hasattr(content_item, 'data'):
                            return str(content_item.data)
                    return str(result.content)
                
                return str(result)
                
            except Exception as e:
                return f"Error calling {tool_info['name']}: {str(e)}"
        
        # Create a proper LangChain tool
        tool_name = tool_info['name']
        tool_description = tool_info['description'] or f"MCP tool from {client.config.name}"
        
        @tool(name=tool_name, description=tool_description)
        def mcp_tool_wrapper(*args, **kwargs):
            """Sync wrapper that runs the async MCP tool."""
            try:
                # Get current event loop or create new one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, we need to use a different approach
                        import threading
                        result = None
                        exception = None
                        
                        def run_async():
                            nonlocal result, exception
                            try:
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                result = new_loop.run_until_complete(call_mcp_tool(**kwargs))
                                new_loop.close()
                            except Exception as e:
                                exception = e
                        
                        thread = threading.Thread(target=run_async)
                        thread.start()
                        thread.join()
                        
                        if exception:
                            raise exception
                        return result
                    else:
                        return loop.run_until_complete(call_mcp_tool(**kwargs))
                except RuntimeError:
                    # No event loop
                    return asyncio.run(call_mcp_tool(**kwargs))
                    
            except Exception as e:
                return f"Error calling {tool_name}: {str(e)}"
        
        return mcp_tool_wrapper
    
    def _get_fallback_tools(self) -> List[BaseTool]:
        """Get fallback tools when MCP is not available."""
        
        @tool
        def get_system_status() -> str:
            """Get current system status."""
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                return f'System Status: CPU: {cpu:.1f}%, Memory: {mem.percent:.1f}%'
            except:
                return 'System status unavailable'
        
        @tool
        def get_current_time() -> str:
            """Get current time."""
            from datetime import datetime
            return f'Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        
        @tool
        def echo_message(message: str) -> str:
            """Echo a message back."""
            return f'Echo: {message}'
        
        return [get_system_status, get_current_time, echo_message]
    
    def get_tools(self) -> List[BaseTool]:
        """Get all available LangChain tools."""
        return self.langchain_tools
    
    async def call_tool_by_name(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool by name directly."""
        # Find which client has this tool
        for client in self.clients.values():
            if not client.connected:
                continue
            
            for tool_info in client.tools:
                if tool_info['name'] == tool_name:
                    return await client.call_tool(tool_name, arguments)
        
        raise ValueError(f"Tool {tool_name} not found")
    
    def list_servers(self) -> Dict[str, Dict[str, Any]]:
        """List all configured servers and their status."""
        server_info = {}
        
        for name, client in self.clients.items():
            server_info[name] = {
                'name': client.config.name,
                'transport': client.config.transport,
                'connected': client.connected,
                'tools_count': len(client.tools),
                'resources_count': len(client.resources),
                'prompts_count': len(client.prompts),
                'description': client.config.description
            }
        
        return server_info
    
    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tools for a specific server."""
        if server_name in self.clients:
            return self.clients[server_name].tools
        return []
    
    async def close(self):
        """Close all MCP connections."""
        cprint("[MCP] Closing MCP connections...", "yellow")
        
        for client in self.clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                if self.verbose:
                    cprint(f"[MCP] Error disconnecting {client.config.name}: {e}", "yellow")
        
        self.clients.clear()
        cprint("[MCP] ✅ MCP manager closed", "green")

# Configuration helpers
def create_simple_mcp_config(servers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to create a simple MCP configuration."""
    return servers

def create_stdio_server_config(command: str, args: Optional[List[str]] = None, 
                              env: Optional[Dict[str, str]] = None,
                              tool_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Helper to create stdio server config."""
    config = {
        'command': command,
        'transport': 'stdio'
    }
    
    if args:
        config['args'] = args
    if env:
        config['env'] = env
    if tool_prefix:
        config['tool_prefix'] = tool_prefix
    
    return config

def create_sse_server_config(url: str, tool_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Helper to create SSE server config."""
    config = {
        'url': url,
        'transport': 'sse'
    }
    
    if tool_prefix:
        config['tool_prefix'] = tool_prefix
    
    return config 
