
"""
Synchronous MCP Manager that avoids async conflicts completely
Uses blocking subprocess communication
"""

import json
import subprocess
import threading
import time
import queue
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
    transport_type: str = "http"
    annotations: Optional[Dict[str, Any]] = None
    
    def to_gemini_function(self) -> Dict[str, Any]:
        """Convert MCP tool to Gemini function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }


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


class SyncMCPClient:
    """Synchronous MCP client using blocking subprocess communication."""
    
    def __init__(self, command: str, args: List[str], name: str):
        self.command = command
        self.args = args
        self.name = name
        self.process = None
        self.message_id = 0
        self.lock = threading.Lock()  # Thread safety for message IDs
        
    def start(self, verbose: bool = False) -> bool:
        """Start the MCP server subprocess."""
        try:
            if verbose:
                cprint(f"[MCP] Starting subprocess: {self.command} {' '.join(self.args)}", "yellow")
            
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered
            )
            
            # Send initialize message
            init_msg = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "opi-voice-assistant",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = self._send_message(init_msg, timeout=10.0)
            if not response or "error" in response:
                raise Exception(f"Initialization failed: {response}")
            
            # Send initialized notification
            initialized_msg = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            self._send_notification(initialized_msg)
            
            if verbose:
                cprint(f"[MCP] ✅ Subprocess {self.name} initialized", "green")
            
            return True
            
        except Exception as e:
            if verbose:
                cprint(f"[MCP] ❌ Failed to start {self.name}: {e}", "red")
            self.close()
            return False
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.process:
            return []
        
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list"
        }
        
        response = self._send_message(msg)
        if response and "result" in response and "tools" in response["result"]:
            return response["result"]["tools"]
        
        return []
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool."""
        if not self.process:
            raise Exception("Process not started")
        
        msg = {
            "jsonrpc": "2.0", 
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        response = self._send_message(msg, timeout=10.0)
        
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Tool call failed: {response['error']}")
        else:
            raise Exception("No response from tool call")
    
    def _send_message(self, message: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a message and wait for response."""
        if not self.process or not self.process.stdin:
            return None
        
        with self.lock:  # Ensure thread safety
            try:
                # Send message
                msg_str = json.dumps(message) + '\n'
                self.process.stdin.write(msg_str)
                self.process.stdin.flush()
                
                # Read response with timeout
                start_time = time.time()
                message_id = message.get("id")
                
                while time.time() - start_time < timeout:
                    # Check if process is still alive
                    if self.process.poll() is not None:
                        raise Exception("Process terminated")
                    
                    # Try to read a line
                    try:
                        # Use a very short timeout for non-blocking read
                        import select
                        import sys
                        
                        if hasattr(select, 'select'):
                            ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                            if ready:
                                response_line = self.process.stdout.readline()
                            else:
                                continue
                        else:
                            # Fallback for Windows
                            response_line = self.process.stdout.readline()
                        
                        if response_line:
                            response = json.loads(response_line.strip())
                            
                            # Check if this response matches our request ID
                            if message_id is not None and response.get("id") == message_id:
                                return response
                            elif message_id is None:
                                # For notifications, return any response
                                return response
                            else:
                                # Wrong ID - continue reading
                                continue
                        else:
                            time.sleep(0.01)  # Small delay to prevent busy waiting
                            
                    except json.JSONDecodeError as e:
                        continue
                    except Exception as e:
                        time.sleep(0.01)
                        continue
                
                raise Exception(f"Timeout waiting for response to {message.get('method', 'unknown')}")
                
            except Exception as e:
                raise Exception(f"Communication error: {e}")
    
    def _send_notification(self, message: Dict[str, Any]):
        """Send a notification (no response expected)."""
        if not self.process or not self.process.stdin:
            return
        
        msg_str = json.dumps(message) + '\n'
        self.process.stdin.write(msg_str)
        self.process.stdin.flush()
    
    def _next_id(self) -> int:
        """Get next message ID."""
        self.message_id += 1
        return self.message_id
    
    def close(self):
        """Close the subprocess."""
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                
                # Wait for process to terminate
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        
            except Exception:
                pass
            finally:
                self.process = None


class MCPManager:
    """Synchronous MCP Manager using blocking subprocess communication."""
    
    def __init__(self, config):
        self.config = config
        self.clients: Dict[str, SyncMCPClient] = {}
        self.tools = []
        self.tool_to_server = {}
        self.failed = []
        self.verbose = False

    async def connect_all(self, verbose=False):
        """Connect to all configured MCP servers."""
        self.verbose = verbose
        
        if not self.config:
            cprint("[MCP] ⚠️  No MCP servers configured", "yellow")
            return

        connected_count = 0
        total_tools = 0

        for server_name, server_config in self.config.items():
            if not server_config.get("enabled", True):
                if verbose:
                    cprint(f"[MCP] Skipping disabled server: {server_name}", "yellow")
                continue

            try:
                if verbose:
                    cprint(f"[MCP] Connecting to server: {server_name}", "yellow")

                transport_type = server_config.get("transport", "stdio")
                
                if transport_type == "stdio":
                    command = server_config.get("command")
                    args = server_config.get("args", [])
                    
                    if not command:
                        raise Exception("No command specified for stdio transport")
                    
                    client = SyncMCPClient(command, args, server_name)
                    
                    if client.start(verbose):
                        self.clients[server_name] = client
                        
                        # Get tools
                        tools_list = client.list_tools()
                        server_tools = self._process_tools(tools_list, server_name)
                        
                        self.tools.extend(server_tools)
                        total_tools += len(server_tools)
                        connected_count += 1
                        
                        if verbose:
                            cprint(f"[MCP] ✅ Server {server_name} connected with {len(server_tools)} tools", "green")
                            for tool in server_tools:
                                cprint(f"[MCP]   - {tool.name}: {tool.description}", "cyan")
                    else:
                        raise Exception("Failed to start MCP client")
                
                else:
                    raise Exception(f"Unsupported transport type: {transport_type}")

            except Exception as e:
                cprint(f"[MCP] ❌ Failed to connect to {server_name}: {e}", "red")
                if verbose:
                    import traceback
                    traceback.print_exc()
                self.failed.append(server_name)

        cprint(f"[MCP] ✅ Connected to {connected_count}/{len(self.config)} MCP servers", "green")
        if total_tools > 0:
            cprint(f"[MCP] 🛠️ Total tools available: {total_tools}", "green")

    def _process_tools(self, tools_list: List[Dict[str, Any]], server_name: str) -> List[MCPTool]:
        """Process tools from server response."""
        processed_tools = []
        
        # Known problematic tools that cause Gemini schema errors
        PROBLEMATIC_TOOLS = {
            'browser_file_upload',      # Has array without items
            'browser_select_option',    # Has array without items  
            'browser_generate_playwright_test'  # Has array without items
        }
        
        for i, tool_data in enumerate(tools_list):
            try:
                name = tool_data.get('name')
                description = tool_data.get('description', '')
                input_schema = tool_data.get('inputSchema', {})
                
                if name:
                    # Skip known problematic tools
                    if name in PROBLEMATIC_TOOLS:
                        if self.verbose:
                            cprint(f"[MCP] ⚠️  Skipping problematic tool: {name} (index {i})", "yellow")
                        continue
                    
                    # Validate schema
                    if not self._is_schema_valid_for_gemini(input_schema, name):
                        if self.verbose:
                            cprint(f"[MCP] ⚠️  Skipping tool {name} (index {i}) due to invalid schema", "yellow")
                        continue
                    
                    # Fix any remaining schema issues
                    fixed_schema = self._fix_schema_for_gemini(input_schema)
                    
                    mcp_tool = MCPTool(
                        name=name,
                        description=description,
                        input_schema=fixed_schema,
                        server_name=server_name,
                        transport_type="stdio"
                    )
                    
                    processed_tools.append(mcp_tool)
                    self.tool_to_server[name] = server_name
                    
                    if self.verbose:
                        cprint(f"[MCP] ✅ Registered tool: {name}", "green")
                        
            except Exception as e:
                if self.verbose:
                    cprint(f"[MCP] ⚠️  Error processing tool {name} (index {i}): {e}", "yellow")
        
        if self.verbose:
            cprint(f"[MCP] Processed {len(processed_tools)} valid tools out of {len(tools_list)} total", "cyan")
        
        return processed_tools

    def _is_schema_valid_for_gemini(self, schema: Dict[str, Any], tool_name: str = "") -> bool:
        """Check if schema is valid for Gemini."""
        if not isinstance(schema, dict):
            return True
        
        try:
            # Check for problematic array properties recursively
            return self._check_schema_recursive(schema, tool_name)
        except Exception as e:
            if self.verbose:
                cprint(f"[MCP] Schema validation error for {tool_name}: {e}", "red")
            return False

    def _check_schema_recursive(self, obj: Any, tool_name: str = "", path: str = "") -> bool:
        """Recursively check schema for Gemini compatibility."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                if key == "properties" and isinstance(value, dict):
                    for prop_name, prop_def in value.items():
                        prop_path = f"{current_path}.{prop_name}"
                        
                        if isinstance(prop_def, dict):
                            # Check for array without items
                            if prop_def.get("type") == "array" and "items" not in prop_def:
                                if self.verbose:
                                    cprint(f"[MCP] Invalid array schema in {tool_name} at {prop_path}: missing 'items'", "red")
                                return False
                            
                            # Recursively check nested properties
                            if not self._check_schema_recursive(prop_def, tool_name, prop_path):
                                return False
                
                elif isinstance(value, (dict, list)):
                    if not self._check_schema_recursive(value, tool_name, current_path):
                        return False
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if not self._check_schema_recursive(item, tool_name, f"{path}[{i}]"):
                    return False
        
        return True

    def _fix_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fix JSON schema issues for Gemini compatibility."""
        if not isinstance(schema, dict):
            return schema
        
        return self._fix_schema_recursive(schema)
    
    def _fix_schema_recursive(self, obj: Any) -> Any:
        """Recursively fix schema issues."""
        if isinstance(obj, dict):
            fixed_obj = {}
            
            for key, value in obj.items():
                if key == "properties" and isinstance(value, dict):
                    fixed_obj[key] = {}
                    for prop_name, prop_def in value.items():
                        if isinstance(prop_def, dict):
                            fixed_prop = prop_def.copy()
                            
                            # Fix array without items
                            if fixed_prop.get("type") == "array" and "items" not in fixed_prop:
                                fixed_prop["items"] = {"type": "string"}
                                if self.verbose:
                                    cprint(f"[MCP] Fixed array schema for property: {prop_name}", "yellow")
                            
                            # Recursively fix nested objects
                            fixed_obj[key][prop_name] = self._fix_schema_recursive(fixed_prop)
                        else:
                            fixed_obj[key][prop_name] = value
                else:
                    fixed_obj[key] = self._fix_schema_recursive(value)
            
            return fixed_obj
        
        elif isinstance(obj, list):
            return [self._fix_schema_recursive(item) for item in obj]
        
        else:
            return obj

    def get_tools(self) -> List[MCPTool]:
        """Get all available MCP tools."""
        return self.tools

    def get_tool_by_name(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Execute a tool call synchronously."""
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
            
            if self.verbose:
                cprint(f"[MCP] 🔧 Calling tool '{tool_name}' on server '{server_name}' with args: {arguments}", "cyan")
            
            # Make the tool call
            result = client.call_tool(tool_name, arguments)
            
            # Process result
            content = []
            if "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict):
                        content.append(item)
                    else:
                        content.append({"type": "text", "text": str(item)})
            else:
                content = [{"type": "text", "text": str(result)}]
            
            if self.verbose:
                cprint(f"[MCP] ✅ Tool call completed successfully", "green")
            
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

    # Keep async version for backward compatibility
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Execute a tool call (async wrapper around sync call)."""
        # Just call the sync version - no async needed
        return self.call_tool_sync(tool_name, arguments)

    def get_server_status(self):
        """Get status information about connected servers."""
        return {
            "connected_servers": len(self.clients),
            "total_servers": len(self.config) if self.config else 0,
            "failed_servers": self.failed,
            "available_tools": len(self.tools),
            "tools_by_server": {
                server_name: [tool.name for tool in self.tools if tool.server_name == server_name]
                for server_name in self.clients.keys()
            }
        }

    async def close(self):
        """Close all MCP server connections."""
        for server_name, client in list(self.clients.items()):
            try:
                if self.verbose:
                    cprint(f"[MCP] Closing server: {server_name}", "yellow")
                
                client.close()
                
                if self.verbose:
                    cprint(f"[MCP] 🔌 Closed server: {server_name}", "green")
                    
            except Exception as e:
                cprint(f"[MCP] ⚠️  Error closing server {server_name}: {e}", "yellow")
        
        self.clients.clear()
        self.tools.clear()
        self.tool_to_server.clear()
        cprint("[MCP] ✅ All MCP servers closed", "green")
