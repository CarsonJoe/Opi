# llm/mcp_manager.py
"""
Simplified MCP Manager with FastMCP 2.x support and persistent connections
Optimized for speed and reliability
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from termcolor import cprint

try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    print("[MCP] ⚠️  FastMCP not available. Install with: pip install fastmcp")


@dataclass
class MCPTool:
    """Represents a complete MCP tool with schema information."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    annotations: Optional[Dict[str, Any]] = None
    
    def to_gemini_function(self) -> Dict[str, Any]:
        """Convert MCP tool to Gemini function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }
    
    def get_human_readable_schema(self) -> str:
        """Get human-readable description of the tool and its parameters."""
        desc = f"{self.name}: {self.description}"
        
        if "properties" in self.input_schema:
            props = self.input_schema["properties"]
            required = self.input_schema.get("required", [])
            
            params = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "")
                required_mark = " (required)" if param_name in required else ""
                
                param_str = f"  - {param_name} ({param_type}){required_mark}"
                if param_desc:
                    param_str += f": {param_desc}"
                params.append(param_str)
            
            if params:
                desc += "\n  Parameters:\n" + "\n".join(params)
        
        return desc


@dataclass
class ToolCallResult:
    """Result of a tool call execution."""
    success: bool
    content: List[Dict[str, Any]]
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_text_content(self) -> str:
        """Extract text content from the result."""
        text_parts = []
        for item in self.content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                # Handle FastMCP result objects
                if hasattr(item, 'text'):
                    text_parts.append(str(item.text))
                else:
                    text_parts.append(str(item))
        return "\n".join(text_parts)


class MCPManager:
    """Simplified MCP Manager optimized for speed with persistent connections."""
    
    def __init__(self, config):
        self.config = config
        self.client: Optional[Client] = None
        self.tools = []
        self.tool_to_server = {}
        self.failed = []
        self.verbose = False
        self.connected_servers = []
        self._client_lock = asyncio.Lock()

    async def connect_all(self, verbose=False):
        """Connect to MCP servers with simplified, fast connection logic."""
        self.verbose = verbose
        
        if not FASTMCP_AVAILABLE:
            cprint("[MCP] ❌ FastMCP not available. Cannot connect to servers.", "red")
            return
            
        if not self.config:
            cprint("[MCP] ⚠️  No MCP servers configured", "yellow")
            return

        try:
            # Find the first HTTP server (simplified approach)
            server_config = None
            server_name = None
            
            for name, cfg in self.config.items():
                if cfg.get("enabled", True) and "url" in cfg:
                    server_config = cfg
                    server_name = name
                    break
            
            if not server_config:
                cprint("[MCP] ❌ No HTTP server found in configuration", "red")
                return
            
            url = server_config["url"]
            
            if verbose:
                cprint(f"[MCP] Connecting to {server_name}: {url}", "yellow")
            
            # Create and connect FastMCP client directly
            self.client = Client(url)
            await self.client.__aenter__()

            
            # Fetch tools
            await self._fetch_tools_simple(server_name)
            
            self.connected_servers = [server_name]
            
            cprint(f"[MCP] ✅ Connected to 1 MCP server", "green")
            if verbose:
                cprint(f"[MCP] 📋 Total tools available: {len(self.tools)}", "green")
                for tool in self.tools:
                    cprint(f"[MCP]   - {tool.name}: {tool.description}", "cyan")

        except Exception as e:
            cprint(f"[MCP] ❌ Failed to connect to MCP servers: {e}", "red")
            if verbose:
                import traceback
                traceback.print_exc()
            self.failed = list(self.config.keys()) if self.config else []

    async def _fetch_tools_simple(self, server_name: str):
        """Simplified tool fetching."""
        if not self.client:
            return
            
        try:
            tools_response = await self.client.list_tools()
            
            self.tools = []
            self.tool_to_server = {}
            
            for tool_data in tools_response:
                # Handle both object and dict formats
                if hasattr(tool_data, 'name'):
                    name = tool_data.name
                    description = getattr(tool_data, 'description', '')
                    input_schema = getattr(tool_data, 'inputSchema', {})
                else:
                    name = tool_data.get('name')
                    description = tool_data.get('description', '')
                    input_schema = tool_data.get('inputSchema', {})
                
                if name:
                    mcp_tool = MCPTool(
                        name=name,
                        description=description,
                        input_schema=input_schema,
                        server_name=server_name
                    )
                    
                    self.tools.append(mcp_tool)
                    self.tool_to_server[name] = server_name
                    
                    if self.verbose:
                        cprint(f"[MCP] ✅ Registered tool: {name}", "green")
                
        except Exception as e:
            cprint(f"[MCP] ❌ Error fetching tools: {e}", "red")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def get_tools(self) -> List[MCPTool]:
        """Get all available MCP tools."""
        return self.tools

    def get_tool_by_name(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_tools_for_llm_prompt(self) -> str:
        """Generate a formatted string of tools for LLM prompt inclusion."""
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(tool.get_human_readable_schema())
        
        return "Available tools:\n\n" + "\n\n".join(tool_descriptions)

    def get_gemini_functions(self) -> List[Dict[str, Any]]:
        """Get tools formatted for Gemini function calling."""
        return [tool.to_gemini_function() for tool in self.tools]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Execute a tool call using the persistent connection."""
        # Use lock for thread safety, but this should be very fast
        async with self._client_lock:
            if not self.client:
                return ToolCallResult(
                    success=False,
                    content=[{"type": "text", "text": "MCP client not connected"}],
                    error_message="MCP client not available"
                )
            
            try:
                # Validate tool exists
                tool = self.get_tool_by_name(tool_name)
                if not tool:
                    return ToolCallResult(
                        success=False,
                        content=[{"type": "text", "text": f"Tool '{tool_name}' not found"}],
                        error_message=f"Unknown tool: {tool_name}"
                    )
                
                # Validate arguments (simplified)
                validation_error = self._validate_arguments_simple(tool, arguments)
                if validation_error:
                    return ToolCallResult(
                        success=False,
                        content=[{"type": "text", "text": f"Invalid arguments: {validation_error}"}],
                        error_message=validation_error
                    )
                
                if self.verbose:
                    cprint(f"[MCP] 🔧 Calling tool '{tool_name}' with args: {arguments}", "cyan")
                
                # Make the tool call - this should be very fast with persistent connection
                result = await self.client.call_tool(tool_name, arguments)
                
                # Process result
                if isinstance(result, list):
                    content = []
                    for item in result:
                        if hasattr(item, '__dict__'):
                            content.append(item.__dict__)
                        else:
                            content.append(item)
                elif isinstance(result, str):
                    content = [{"type": "text", "text": result}]
                elif isinstance(result, dict):
                    if "error" in result:
                        return ToolCallResult(
                            success=False,
                            content=[{"type": "text", "text": result.get("error", "Unknown error")}],
                            error_message=result.get("error", "Unknown error")
                        )
                    else:
                        content = [{"type": "text", "text": str(result)}]
                else:
                    content = [{"type": "text", "text": str(result)}]
                
                if self.verbose:
                    cprint(f"[MCP] ✅ Tool call completed. Success: True", "green")
                    cprint(f"[MCP] 📄 Result: {content}", "white")
                
                return ToolCallResult(
                    success=True,
                    content=content,
                    error_message=None
                )
                
            except Exception as e:
                error_msg = f"Tool call failed: {str(e)}"
                cprint(f"[MCP] ❌ {error_msg}", "red")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                
                return ToolCallResult(
                    success=False,
                    content=[{"type": "text", "text": error_msg}],
                    error_message=error_msg
                )

    def _validate_arguments_simple(self, tool: MCPTool, arguments: Dict[str, Any]) -> Optional[str]:
        """Simplified argument validation."""
        schema = tool.input_schema
        
        # Check required parameters
        required = schema.get("required", [])
        for req_param in required:
            if req_param not in arguments:
                return f"Missing required parameter: {req_param}"
        
        return None

    def get_server_status(self):
        """Get status information about connected servers."""
        return {
            "connected_servers": len(self.connected_servers),
            "total_servers": len(self.config) if self.config else 0,
            "failed_servers": self.failed,
            "available_tools": len(self.tools),
            "tools_by_server": {
                server: [tool.name for tool in self.tools if tool.server_name == server]
                for server in self.connected_servers
            }
        }

    async def list_tools(self):
        """List all available tools."""
        if not self.client:
            return []
        
        try:
            return await self.client.list_tools()
        except Exception as e:
            cprint(f"[MCP] Error listing tools: {e}", "red")
            return []

    async def close(self):
        """Close MCP client connection."""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
                cprint("[MCP] 🔌 Closed MCP client", "green")
            except Exception as e:
                cprint(f"[MCP] ⚠️  Error closing MCP client: {e}", "yellow")
        
        self.client = None
        self.tools = []
        self.tool_to_server = {}
        self.connected_servers = []
        cprint("[MCP] ✅ MCP manager closed", "green")
