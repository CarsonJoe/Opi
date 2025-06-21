# llm/mcp_manager.py
"""
Enhanced MCP Manager with FastMCP 2.x support
Compatible with the latest MCP ecosystem (2024-2025)
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
                text_parts.append(str(item))
        return "\n".join(text_parts)


class MCPManager:
    """Enhanced MCP Manager with FastMCP 2.x support."""
    
    def __init__(self, config):
        self.config = config
        self.client: Optional[Client] = None
        self.tools = []
        self.tool_to_server = {}  # Maps tool names to server names
        self.failed = []
        self.verbose = False
        self.connected_servers = []

    async def connect_all(self, verbose=False):
        """Connect to all configured MCP servers using FastMCP 2.x."""
        self.verbose = verbose
        
        if not FASTMCP_AVAILABLE:
            cprint("[MCP] ❌ FastMCP not available. Cannot connect to servers.", "red")
            return
            
        if not self.config:
            cprint("[MCP] ⚠️  No MCP servers configured", "yellow")
            return

        # Convert your config format to FastMCP client config
        fastmcp_config = self._convert_config_to_fastmcp_format()
        
        if not fastmcp_config.get("mcpServers"):
            cprint("[MCP] ⚠️  No valid MCP servers in config", "yellow")
            return

        try:
            if verbose:
                cprint(f"[MCP] Connecting to {len(fastmcp_config['mcpServers'])} MCP servers...", "yellow")
                cprint(f"[MCP] Config: {json.dumps(fastmcp_config, indent=2)}", "cyan")
            
            # Create FastMCP client with multi-server config
            self.client = Client(fastmcp_config)
            
            # Connect to all servers
            await self.client.__aenter__()
            
            # Fetch tools from all connected servers
            await self._fetch_all_tools()
            
            self.connected_servers = list(fastmcp_config["mcpServers"].keys())
            
            cprint(f"[MCP] ✅ Connected to {len(self.connected_servers)} MCP servers", "green")
            if verbose:
                cprint(f"[MCP] 📋 Total tools available: {len(self.tools)}", "green")
                for tool in self.tools:
                    cprint(f"[MCP]   - {tool.name} (from {tool.server_name})", "cyan")

        except Exception as e:
            cprint(f"[MCP] ❌ Failed to connect to MCP servers: {e}", "red")
            if verbose:
                import traceback
                traceback.print_exc()
            self.failed = list(self.config.keys()) if self.config else []

    def _convert_config_to_fastmcp_format(self) -> Dict[str, Any]:
        """Convert your config format to FastMCP client config format."""
        fastmcp_config = {"mcpServers": {}}
        
        for server_name, server_cfg in self.config.items():
            if not server_cfg.get("enabled", True):
                continue
                
            server_config = {}
            
            # Handle HTTP/HTTPS URLs
            if "url" in server_cfg:
                url = server_cfg["url"]
                server_config["url"] = url
                
                # Determine transport type
                if url.startswith("http://") or url.startswith("https://"):
                    server_config["transport"] = "streamable-http"
                else:
                    cprint(f"[MCP] ⚠️  Unknown URL format for server '{server_name}': {url}", "yellow")
                    continue
            
            # Handle command-based servers (stdio)
            elif "command" in server_cfg:
                server_config["command"] = server_cfg["command"]
                server_config["args"] = server_cfg.get("args", [])
                if "env" in server_cfg:
                    server_config["env"] = server_cfg["env"]
            
            # Handle script paths
            elif "script" in server_cfg:
                script_path = server_cfg["script"]
                if script_path.endswith(".py"):
                    server_config["command"] = "python"
                    server_config["args"] = [script_path]
                elif script_path.endswith(".js"):
                    server_config["command"] = "node"
                    server_config["args"] = [script_path]
                else:
                    cprint(f"[MCP] ⚠️  Unknown script type for server '{server_name}': {script_path}", "yellow")
                    continue
            
            else:
                cprint(f"[MCP] ⚠️  Invalid server config for '{server_name}': missing url, command, or script", "yellow")
                continue
            
            fastmcp_config["mcpServers"][server_name] = server_config
        
        return fastmcp_config

    async def _fetch_all_tools(self):
        """Fetch tools from all connected servers."""
        if not self.client:
            return
            
        try:
            # Get tools from the client
            # Note: FastMCP 2.x client handles multi-server tool fetching automatically
            tools_response = await self.client.list_tools()
            
            self.tools = []
            self.tool_to_server = {}
            
            # FastMCP 2.x returns tools with server prefixes for multi-server setups
            for tool_data in tools_response:
                # Handle both dict and object formats
                if hasattr(tool_data, 'name'):
                    # Object format
                    name = tool_data.name
                    description = getattr(tool_data, 'description', '')
                    input_schema = getattr(tool_data, 'inputSchema', {})
                    
                    # Extract server name from tool name prefix (if present)
                    server_name = "unknown"
                    if "_" in name:
                        potential_server = name.split("_")[0]
                        if potential_server in self.connected_servers:
                            server_name = potential_server
                    
                elif isinstance(tool_data, dict):
                    # Dict format
                    name = tool_data.get('name')
                    description = tool_data.get('description', '')
                    input_schema = tool_data.get('inputSchema', {})
                    server_name = tool_data.get('server', 'unknown')
                else:
                    continue
                
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
        """Get all available MCP tools with complete schema information."""
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
        """Execute a tool call via MCP protocol using FastMCP 2.x client."""
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
            
            # Validate arguments against tool schema
            validation_error = self._validate_arguments(tool, arguments)
            if validation_error:
                return ToolCallResult(
                    success=False,
                    content=[{"type": "text", "text": f"Invalid arguments: {validation_error}"}],
                    error_message=validation_error
                )
            
            if self.verbose:
                cprint(f"[MCP] 🔧 Calling tool '{tool_name}' with args: {arguments}", "cyan")
            
            # Make the actual tool call via FastMCP client
            result = await self.client.call_tool(tool_name, arguments)
            
            # Handle FastMCP 2.x response format
            if isinstance(result, str):
                # Simple string response
                content = [{"type": "text", "text": result}]
                success = True
            elif isinstance(result, dict):
                # Dict response
                if "error" in result:
                    content = [{"type": "text", "text": result.get("error", "Unknown error")}]
                    success = False
                else:
                    content = [{"type": "text", "text": str(result)}]
                    success = True
            elif isinstance(result, list):
                # List of content items
                content = result
                success = True
            else:
                # Other format, convert to string
                content = [{"type": "text", "text": str(result)}]
                success = True
            
            if self.verbose:
                cprint(f"[MCP] ✅ Tool call completed. Success: {success}", "green")
                cprint(f"[MCP] 📄 Result: {content}", "white")
            
            return ToolCallResult(
                success=success,
                content=content,
                error_message=None if success else "Tool execution failed"
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

    def _validate_arguments(self, tool: MCPTool, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against tool schema. Returns error message if invalid."""
        schema = tool.input_schema
        
        # Check required parameters
        required = schema.get("required", [])
        for req_param in required:
            if req_param not in arguments:
                return f"Missing required parameter: {req_param}"
        
        # Basic type checking for properties
        properties = schema.get("properties", {})
        for param_name, param_value in arguments.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type:
                    if not self._check_type(param_value, expected_type):
                        return f"Parameter '{param_name}' should be of type {expected_type}"
        
        return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Basic type checking for JSON schema types."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid

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
        """List all available tools - convenience method for debugging."""
        if not self.client:
            return []
        
        try:
            return await self.client.list_tools()
        except Exception as e:
            cprint(f"[MCP] Error listing tools: {e}", "red")
            return []

    async def close(self):
        """Close all MCP client connections."""
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
