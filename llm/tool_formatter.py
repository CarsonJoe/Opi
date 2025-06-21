# llm/tool_formatter.py
"""
Tool formatting utilities for LLM integration
Converts MCP tools to various formats needed by different LLM providers
"""

import json
import re
from typing import Dict, List, Any, Optional
from .mcp_manager import MCPTool


class ToolFormatter:
    """Formats MCP tools for different LLM providers and use cases."""
    
    @staticmethod
    def format_for_gemini_system_prompt(tools: List[MCPTool]) -> str:
        """Format tools for inclusion in Gemini system prompt."""
        if not tools:
            return "No tools are currently available."
        
        prompt_parts = ["Available tools:"]
        
        for tool in tools:
            # Basic tool information
            tool_desc = f"\n{tool.name}:"
            tool_desc += f"\n  Description: {tool.description}"
            
            # Format parameters
            if tool.input_schema.get("properties"):
                tool_desc += f"\n  Parameters:"
                properties = tool.input_schema["properties"]
                required = tool.input_schema.get("required", [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    
                    requirement = " (required)" if is_required else " (optional)"
                    tool_desc += f"\n    - {param_name} ({param_type}){requirement}"
                    
                    if param_desc:
                        tool_desc += f": {param_desc}"
                    
                    # Add enum values if present
                    if "enum" in param_info:
                        enum_values = ", ".join(str(v) for v in param_info["enum"])
                        tool_desc += f" [values: {enum_values}]"
            else:
                tool_desc += f"\n  Parameters: None required"
            
            # Add usage hints from annotations
            if tool.annotations:
                if tool.annotations.get("readOnlyHint"):
                    tool_desc += f"\n  Note: This tool only reads data, does not modify anything"
                if tool.annotations.get("destructive"):
                    tool_desc += f"\n  Warning: This tool may modify or delete data"
            
            prompt_parts.append(tool_desc)
        
        prompt_parts.append(f"\nTo use a tool, mention its name and specify the required parameters in your response. The system will automatically detect and execute the tool call.")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def format_for_function_calling(tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """Format tools for Gemini function calling API."""
        functions = []
        
        for tool in tools:
            function_def = {
                "name": tool.name,
                "description": tool.description
            }
            
            # Add parameters if they exist
            if tool.input_schema:
                # Clean up the schema for function calling
                cleaned_schema = ToolFormatter._clean_schema_for_function_calling(tool.input_schema)
                if cleaned_schema:
                    function_def["parameters"] = cleaned_schema
            
            functions.append(function_def)
        
        return functions
    
    @staticmethod
    def _clean_schema_for_function_calling(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up JSON schema for function calling compatibility."""
        cleaned = {}
        
        # Copy basic schema structure
        if "type" in schema:
            cleaned["type"] = schema["type"]
        
        if "properties" in schema:
            cleaned["properties"] = {}
            for prop_name, prop_def in schema["properties"].items():
                cleaned_prop = {}
                
                # Copy supported fields
                for field in ["type", "description", "enum", "items"]:
                    if field in prop_def:
                        cleaned_prop[field] = prop_def[field]
                
                # Handle nested objects
                if prop_def.get("type") == "object" and "properties" in prop_def:
                    cleaned_prop["properties"] = prop_def["properties"]
                
                cleaned["properties"][prop_name] = cleaned_prop
        
        if "required" in schema:
            cleaned["required"] = schema["required"]
        
        return cleaned
    
    @staticmethod
    def detect_tool_calls_in_text(text: str, available_tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """Detect tool call intentions in LLM response text."""
        tool_calls = []
        text_lower = text.lower()
        
        # Pattern matching for tool calls
        for tool in available_tools:
            tool_name_lower = tool.name.lower()
            
            # Look for explicit mentions of tool names
            if tool_name_lower in text_lower:
                # Try to extract parameters from context
                extracted_call = ToolFormatter._extract_tool_call_from_context(
                    text, tool, text_lower.find(tool_name_lower)
                )
                if extracted_call:
                    tool_calls.append(extracted_call)
        
        return tool_calls
    
    @staticmethod
    def _extract_tool_call_from_context(text: str, tool: MCPTool, tool_mention_pos: int) -> Optional[Dict[str, Any]]:
        """Extract tool call parameters from surrounding context."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated NLP parsing
        
        # Look for parameter patterns around the tool mention
        context_start = max(0, tool_mention_pos - 100)
        context_end = min(len(text), tool_mention_pos + 200)
        context = text[context_start:context_end]
        
        # Basic parameter extraction
        arguments = {}
        
        # For tools with simple parameters, try to extract from context
        if tool.input_schema.get("properties"):
            for param_name, param_info in tool.input_schema["properties"].items():
                param_type = param_info.get("type", "string")
                
                # Look for parameter values in context
                if param_type == "string":
                    # Look for quoted strings or specific patterns
                    value = ToolFormatter._extract_string_parameter(context, param_name)
                    if value:
                        arguments[param_name] = value
                elif param_type in ["integer", "number"]:
                    # Look for numbers
                    value = ToolFormatter._extract_number_parameter(context, param_name)
                    if value is not None:
                        arguments[param_name] = value
        
        # Only return if we have required parameters
        required = tool.input_schema.get("required", [])
        if all(req_param in arguments for req_param in required):
            return {
                "name": tool.name,
                "arguments": arguments
            }
        
        # If no parameters required, just call the tool
        if not required:
            return {
                "name": tool.name,
                "arguments": {}
            }
        
        return None
    
    @staticmethod
    def _extract_string_parameter(context: str, param_name: str) -> Optional[str]:
        """Extract string parameter from context."""
        # Look for quoted strings
        quoted_pattern = r'["\']([^"\']*)["\']'
        matches = re.findall(quoted_pattern, context)
        
        # For now, return the first quoted string found
        # More sophisticated parsing could match parameter names
        if matches:
            return matches[0]
        
        return None
    
    @staticmethod
    def _extract_number_parameter(context: str, param_name: str) -> Optional[float]:
        """Extract number parameter from context."""
        # Look for numbers in the context
        number_pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, context)
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def format_tool_result_for_conversation(result, tool_name: str) -> str:
        """Format tool execution result for inclusion in conversation."""
        if not result.success:
            return f"❌ Tool '{tool_name}' failed: {result.error_message}"
        
        content_text = result.get_text_content()
        if content_text:
            return f"🔧 Tool '{tool_name}' result:\n{content_text}"
        else:
            return f"✅ Tool '{tool_name}' executed successfully"
    
    @staticmethod
    def create_tool_usage_examples(tools: List[MCPTool]) -> str:
        """Create usage examples for tools to include in prompts."""
        if not tools:
            return ""
        
        examples = ["Tool usage examples:"]
        
        for tool in tools[:3]:  # Limit to first 3 tools to avoid prompt bloat
            example = f"\nTo use {tool.name}:"
            
            if tool.input_schema.get("properties"):
                required = tool.input_schema.get("required", [])
                props = tool.input_schema["properties"]
                
                example_params = []
                for param_name in required[:2]:  # Show first 2 required params
                    if param_name in props:
                        param_type = props[param_name].get("type", "string")
                        if param_type == "string":
                            example_params.append(f'{param_name}="example_value"')
                        elif param_type in ["integer", "number"]:
                            example_params.append(f'{param_name}=123')
                        elif param_type == "boolean":
                            example_params.append(f'{param_name}=true')
                
                if example_params:
                    example += f" Just mention '{tool.name}' and specify: {', '.join(example_params)}"
                else:
                    example += f" Just mention '{tool.name}' in your response"
            else:
                example += f" Just mention '{tool.name}' in your response"
            
            examples.append(example)
        
        return "\n".join(examples)
