"""
LLM Agent for Opi Voice Assistant.
Integrates with MCP tools and manages conversations.
Simplified version to get basic functionality working.
"""

import asyncio
import logging
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

# For now, use OpenAI directly - can be replaced with langchain later
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available - using mock responses")

from core.config import LLMConfig


class OpiAgent:
    """Main LLM agent for Opi Voice Assistant."""
    
    def __init__(self, config: LLMConfig, display_manager=None):
        self.config = config
        self.display_manager = display_manager
        self.logger = logging.getLogger("OpiAgent")
        
        # OpenAI client
        self.client = None
        
        # Conversation state
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
        
        # Tool registry
        self.tools = {}
        self.tool_functions = {}
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize the LLM agent."""
        if self.initialized:
            return
        
        self.logger.info("Initializing LLM agent...")
        
        if OPENAI_AVAILABLE and self.config.api_key:
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            self.logger.info(f"Using OpenAI model: {self.config.model}")
        else:
            self.logger.warning("OpenAI not available - using mock responses")
        
        # Register built-in tools
        self._register_builtin_tools()
        
        self.initialized = True
        self.logger.info("LLM agent initialized")
    
    def _register_builtin_tools(self):
        """Register built-in tools for Opi."""
        
        # Display control tools
        self.tools["switch_display_mode"] = {
            "type": "function",
            "function": {
                "name": "switch_display_mode",
                "description": "Switch the display mode between passthrough, overlay, and desktop",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["passthrough", "overlay", "desktop"],
                            "description": "The display mode to switch to"
                        }
                    },
                    "required": ["mode"]
                }
            }
        }
        
        self.tools["show_overlay"] = {
            "type": "function",
            "function": {
                "name": "show_overlay",
                "description": "Show text overlay on the display",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to display on overlay"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (optional)"
                        }
                    },
                    "required": ["text"]
                }
            }
        }
        
        self.tools["get_display_info"] = {
            "type": "function",
            "function": {
                "name": "get_display_info",
                "description": "Get current display information and status",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Register tool functions
        self.tool_functions["switch_display_mode"] = self._switch_display_mode
        self.tool_function["show_overlay"] = self._show_overlay
        self.tool_functions["get_display_info"] = self._get_display_info
    
    async def process_message(self, user_input: str) -> Optional[str]:
        """Process a user message and return response."""
        if not self.initialized:
            await self.initialize()
        
        self.logger.info(f"Processing message: {user_input}")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            if self.client:
                response = await self._get_llm_response()
            else:
                response = self._get_mock_response(user_input)
            
            # Add assistant response to history
            if response:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "Sorry, I encountered an error processing your request."
    
    async def _get_llm_response(self) -> Optional[str]:
        """Get response from LLM with tool support."""
        try:
            # Prepare messages with system prompt
            messages = [
                {"role": "system", "content": self.config.system_prompt}
            ] + self.conversation_history
            
            # Get response from OpenAI with tools
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=list(self.tools.values()) if self.tools else None,
                tool_choice="auto" if self.tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            message = response.choices[0].message
            
            # Handle tool calls
            if message.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in message.tool_calls:
                    result = await self._execute_tool_call(tool_call)
                    tool_results.append(result)
                
                # Add tool call message to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                })
                
                # Add tool results to history
                for i, tool_call in enumerate(message.tool_calls):
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_results[i])
                    })
                
                # Get final response after tool execution
                messages = [
                    {"role": "system", "content": self.config.system_prompt}
                ] + self.conversation_history
                
                final_response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                return final_response.choices[0].message.content
            
            return message.content
            
        except Exception as e:
            self.logger.error(f"LLM response error: {e}")
            return None
    
    async def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a tool call and return result."""
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            self.logger.info(f"Executing tool: {function_name} with args: {arguments}")
            
            if function_name in self.tool_functions:
                result = await self.tool_functions[function_name](**arguments)
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": f"Unknown tool: {function_name}"}
                
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_mock_response(self, user_input: str) -> str:
        """Generate mock response when OpenAI is not available."""
        user_lower = user_input.lower()
        
        if "display" in user_lower or "show" in user_lower or "overlay" in user_lower:
            return "I would control the display for you, but I need an API key to function properly."
        elif "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm Opi, your voice assistant. I'm running in demo mode without an API key."
        elif "weather" in user_lower:
            return "I'd check the weather for you, but I need to be properly configured first."
        elif "help" in user_lower:
            return "I can help with display control, system information, and more. Please configure an API key for full functionality."
        else:
            return "I heard you, but I need an OpenAI API key to provide intelligent responses."
    
    # Tool function implementations
    async def _switch_display_mode(self, mode: str) -> str:
        """Switch display mode."""
        if self.display_manager:
            try:
                await self.display_manager.switch_mode(mode)
                return f"Successfully switched to {mode} mode"
            except Exception as e:
                return f"Failed to switch to {mode} mode: {e}"
        else:
            return "Display manager not available"
    
    async def _show_overlay(self, text: str, timeout: int = None) -> str:
        """Show text overlay."""
        if self.display_manager:
            try:
                await self.display_manager.show_overlay(text, timeout)
                return f"Overlay displayed: {text}"
            except Exception as e:
                return f"Failed to show overlay: {e}"
        else:
            return "Display manager not available"
    
    async def _get_display_info(self) -> Dict[str, Any]:
        """Get display information."""
        if self.display_manager:
            try:
                return await self.display_manager.get_display_info()
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "Display manager not available"}
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary = f"Conversation with {len(self.conversation_history)} messages:\n"
        for msg in self.conversation_history[-4:]:  # Last 4 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                summary += f"{role}: {content[:100]}...\n"
        
        return summary
    
    async def shutdown(self):
        """Shutdown the agent."""
        self.logger.info("Shutting down LLM agent...")
        self.initialized = Falses
