# llm/mcp_manager.py
"""
Enhanced MCP Manager with full tool calling support
Implements complete MCP protocol for tool discovery and execution
"""

from fastmcp import Client
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from termcolor import cprint


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
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)


class MCPManager:
    """Enhanced MCP Manager with full tool calling support."""
    
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.tools = []
        self.tool_to_server = {}  # Maps tool names to server names
        self.failed = []
        self.verbose = False

    @property
    def connected_servers(self):
        return list(self.clients.keys())

    async def connect_all(self, verbose=False):
        """Connect to all configured MCP servers."""
        self.verbose = verbose
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
                if verbose:
                    print(f"[MCP] Connecting to server '{name}' at {url}")
                
                client = Client(url)
                await client.__aenter__()
                self.clients[name] = client

                # Fetch complete tool information
                await self._fetch_tools_from_server(name, client)

            except Exception as e:
                print(f"[ERROR] ❌ Failed to connect to MCP server '{name}': {e}")
                self.failed.append(name)

        print(f"[MCP] ✅ Connected to {len(self.clients)} MCP servers")
        if verbose:
            print(f"[MCP] 📋 Total tools available: {len(self.tools)}")
            for tool in self.tools:
                print(f"[MCP]   - {tool.name} (from {tool.server_name})")

    async def _fetch_tools_from_server(self, server_name: str, client: Client):
        """Fetch complete tool information from a server."""
        try:
            # Get tools list with full schema information
            tools_response = await client.list_tools()
            
            for tool_data in tools_response:
                # Handle both dict and object formats
                if isinstance(tool_data, dict):
                    name = tool_data.get("name")
                    description = tool_data.get("description", "")
                    input_schema = tool_data.get("inputSchema", {})
                    annotations = tool_data.get("annotations")
                else:
                    # Handle object format
                    name = getattr(tool_data, "name", None)
                    description = getattr(tool_data, "description", "")
                    input_schema = getattr(tool_data, "inputSchema", {})
                    annotations = getattr(tool_data, "annotations", None)
                
                if name:
                    mcp_tool = MCPTool(
                        name=name,
                        description=description,
                        input_schema=input_schema,
                        server_name=server_name,
                        annotations=annotations
                    )
                    
                    self.tools.append(mcp_tool)
                    self.tool_to_server[name] = server_name
                    
                    if self.verbose:
                        print(f"[MCP] ✅ Registered tool: {name}")
                        print(f"[MCP]   Description: {description}")
                        if input_schema.get("properties"):
                            print(f"[MCP]   Parameters: {list(input_schema['properties'].keys())}")
                
        except Exception as e:
            print(f"[MCP] ❌ Error fetching tools from server '{server_name}': {e}")

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
        """Execute a tool call via MCP protocol."""
        try:
            # Find the server for this tool
            server_name = self.tool_to_server.get(tool_name)
            if not server_name:
                return ToolCallResult(
                    success=False,
                    content=[{"type": "text", "text": f"Tool '{tool_name}' not found"}],
                    error_message=f"Unknown tool: {tool_name}"
                )
            
            client = self.clients.get(server_name)
            if not client:
                return ToolCallResult(
                    success=False,
                    content=[{"type": "text", "text": f"Server '{server_name}' not connected"}],
                    error_message=f"Server not available: {server_name}"
                )
            
            # Validate arguments against tool schema
            tool = self.get_tool_by_name(tool_name)
            if tool:
                validation_error = self._validate_arguments(tool, arguments)
                if validation_error:
                    return ToolCallResult(
                        success=False,
                        content=[{"type": "text", "text": f"Invalid arguments: {validation_error}"}],
                        error_message=validation_error
                    )
            
            if self.verbose:
                print(f"[MCP] 🔧 Calling tool '{tool_name}' with args: {arguments}")
            
            # Make the actual tool call via MCP client
            result = await client.call_tool(tool_name, arguments)
            
            # Handle the response
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict) and 'content' in result:
                content = result['content']
            else:
                # Fallback for unexpected response format
                content = [{"type": "text", "text": str(result)}]
            
            # Check if it's an error result
            is_error = getattr(result, 'isError', False) or (isinstance(result, dict) and result.get('isError', False))
            
            if self.verbose:
                print(f"[MCP] ✅ Tool call completed. Error: {is_error}")
                print(f"[MCP] 📄 Result content: {content}")
            
            return ToolCallResult(
                success=not is_error,
                content=content if isinstance(content, list) else [content],
                error_message=None if not is_error else "Tool execution failed",
                metadata=getattr(result, 'metadata', None) if hasattr(result, 'metadata') else None
            )
            
        except Exception as e:
            error_msg = f"Tool call failed: {str(e)}"
            print(f"[MCP] ❌ {error_msg}")
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
            "connected_servers": len(self.clients),
            "total_servers": len(self.config) if self.config else 0,
            "failed_servers": self.failed,
            "available_tools": len(self.tools),
            "tools_by_server": {
                server: [tool.name for tool in self.tools if tool.server_name == server]
                for server in self.connected_servers
            }
        }

    async def close(self):
        """Close all MCP client connections."""
        for name, client in self.clients.items():
            try:
                await client.__aexit__(None, None, None)
                print(f"[MCP] 🔌 Closed MCP client: {name}")
            except Exception as e:
                print(f"[WARN] Failed to close MCP client '{name}': {e}")
        
        self.clients = {}
        self.tools = []
        self.tool_to_server = {}
        print("[MCP] ✅ All MCP clients closed")
