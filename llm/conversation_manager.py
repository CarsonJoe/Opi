# llm/conversation_manager.py
"""
Conversation Manager
"""

from __future__ import annotations

import asyncio, inspect, os, re, time
from datetime import datetime
from typing import Any, Dict, List, Optional

from termcolor import cprint
try:
    from .phrase_stream import PhraseStreamer, UltraLowLatencyTTSPipeline
    from .mcp_manager import MCPManager, ToolCallResult
except ImportError:
    # Handle relative import when run directly
    from llm.phrase_stream import PhraseStreamer, UltraLowLatencyTTSPipeline
    from llm.mcp_manager import MCPManager, ToolCallResult

# ─── Lazy wrappers ───────────────────────────────────────────────────────────
LLM_WRAPPERS: Dict[str, Any] = {}
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_WRAPPERS["google-genai"] = ChatGoogleGenerativeAI
except ImportError:
    pass
try:
    from langchain_anthropic import ChatAnthropic  # >=0.1.0
    LLM_WRAPPERS["anthropic"] = ChatAnthropic
except ImportError:
    pass
try:
    from langchain_openai import ChatOpenAI
    LLM_WRAPPERS["openai"] = ChatOpenAI
except ImportError:
    pass

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

ENV_KEY = {
    "google-genai": "GOOGLE_API_KEY",
    "anthropic":    "ANTHROPIC_API_KEY",
    "openai":       "OPENAI_API_KEY",
}

# ─── Tool‑aware agent (LangChain) ─────────────────────────────────────────────
class ToolAwareAgent:
    def __init__(self, provider: str, mcp: MCPManager,
                 model: str, temp: float, api_key: str, verbose=False):
        if provider not in LLM_WRAPPERS:
            raise RuntimeError(f"Provider '{provider}' not installed")
        self.verbose = verbose 
        self.mcp = mcp
        
        # Store MCP server config for creating new connections
        self.mcp_config = mcp.config

        llm_cls = LLM_WRAPPERS[provider]
        if provider == "google-genai":
            self.llm = llm_cls(model=model, temperature=temp, google_api_key=api_key)
        elif provider == "anthropic":
            # ChatAnthropic uses model param for newer versions
            self.llm = llm_cls(model=model, temperature=temp, anthropic_api_key=api_key)
        else:
            self.llm = llm_cls(model_name=model, temperature=temp, openai_api_key=api_key)

        self.agent = self._build_agent()

    def _wrap_tool(self, tool) -> StructuredTool:
        def _run(**kwargs):  # Sync function for LangChain
            if self.verbose:
                cprint(f"[Agent] 🔧 Calling MCP tool: {tool.name} with {kwargs}", "green")
            
            # Clean kwargs - remove any LangChain-specific parameters
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['config', 'configurable']}
            
            # FINAL FIX: Use thread pool with new FastMCP connection
            # This avoids the client context issue entirely
            def run_tool_in_thread():
                """Run the tool in a separate thread with its own FastMCP connection"""
                try:
                    # Create new event loop for this thread
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def call_tool():
                        # Import here to avoid circular imports
                        try:
                            from fastmcp import Client
                        except ImportError:
                            return "FastMCP not available"
                        
                        # Get the server URL from MCP config
                        server_url = None
                        for server_name, server_config in self.mcp_config.items():
                            if server_config.get('enabled', True) and 'url' in server_config:
                                server_url = server_config['url']
                                break
                        
                        if not server_url:
                            return "No MCP server URL found"
                        
                        # Create new client connection for this tool call
                        async with Client(server_url) as client:
                            result = await client.call_tool(tool.name, clean_kwargs)
                            
                            # Format result like MCPManager does - but cleaner
                            if isinstance(result, list):
                                text_parts = []
                                for item in result:
                                    if hasattr(item, '__dict__'):
                                        item_dict = item.__dict__
                                    else:
                                        item_dict = item
                                    
                                    if isinstance(item_dict, dict):
                                        if item_dict.get('type') == 'text':
                                            text_parts.append(item_dict.get('text', ''))
                                        elif 'text' in item_dict:
                                            text_parts.append(str(item_dict['text']))
                                    elif isinstance(item_dict, str):
                                        text_parts.append(item_dict)
                                    else:
                                        text_parts.append(str(item_dict))
                                
                                clean_result = '\n'.join(text_parts)
                            elif isinstance(result, str):
                                clean_result = result
                            else:
                                clean_result = str(result)
                            
                            # Remove duplicate prefix if present
                            if clean_result.startswith('🔐 Secret: '):
                                clean_result = clean_result[11:]  # Remove "🔐 Secret: " prefix
                            
                            return clean_result
                    
                    # Run the async call
                    return loop.run_until_complete(call_tool())
                    
                except Exception as e:
                    if self.verbose:
                        cprint(f"[Agent] Tool thread error: {e}", "red")
                    return f"Tool execution failed: {str(e)}"
                finally:
                    try:
                        loop.close()
                    except:
                        pass
            
            # Use thread pool executor to run the tool
            try:
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_tool_in_thread)
                    
                    try:
                        result = future.result(timeout=10.0)  # 10 second timeout
                        
                        if self.verbose:
                            result_preview = result[:200] + "..." if len(str(result)) > 200 else str(result)
                            cprint(f"[Agent] 📄 Tool result: {result_preview}", "green")
                        
                        return str(result)
                        
                    except concurrent.futures.TimeoutError:
                        cprint(f"[Agent] ⏰ Tool '{tool.name}' timed out after 10 seconds", "yellow")
                        return f"Tool '{tool.name}' timed out"
                    
            except Exception as e:
                if self.verbose:
                    cprint(f"[Agent] ❌ Tool execution error: {e}", "red")
                    import traceback
                    traceback.print_exc()
                return f"Tool execution failed: {str(e)}"
        
        # Create proper parameter signature for LangChain compatibility
        from typing import Any
        
        # Map JSON schema types to Python types
        TYPE_MAPPING = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "any": Any
        }
        
        # Create parameter signature from tool schema
        sig_params = []
        properties = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])
        
        if properties:
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")
                python_type = TYPE_MAPPING.get(param_type, str)
                
                # Handle Union types for optional parameters
                if param_name not in required:
                    from typing import Optional
                    python_type = Optional[python_type]
                    default = None
                else:
                    default = inspect.Parameter.empty
                
                sig_params.append(
                    inspect.Parameter(
                        param_name, 
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default,
                        annotation=python_type
                    )
                )
        
        # Set the function signature
        _run.__signature__ = inspect.Signature(sig_params)
        
        _run.__annotations__ = {
            param.name: param.annotation
            for param in sig_params
            if param.annotation is not inspect.Parameter.empty
        }
            
        # Create the StructuredTool
        return StructuredTool.from_function(
            _run,
            name=tool.name,
            description=tool.description,
        )

    def _build_agent(self):
        tools = [self._wrap_tool(t) for t in self.mcp.get_tools()]
        
        # Use create_tool_calling_agent for better Anthropic support
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are Opi, an on‑device assistant running on an Orange Pi. "
             "You have access to tools that can provide better information or perform actions. "
             "ALWAYS use tools when they can answer the user's question better than your knowledge. "
             "For example, if someone asks for a secret message, use the get_secret_message tool. "
             "If someone asks how many secrets you know, use the count_secrets tool. "
             "When you get a result from a tool, provide that result to the user immediately. "
             "Do not call the same tool multiple times unless specifically asked. "
             "Be concise but helpful in your responses."),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        if self.verbose:
            cprint("\n=== UPDATED TOOL PROMPT ===", "cyan", attrs=["bold"])
            cprint(str(prompt), "cyan")
            cprint(f"Available tools: {len(tools)}", "yellow")
            for t in tools:
                cprint(f" • {t.name}: {t.description}", "yellow")
            cprint("===========================\n", "cyan", attrs=["bold"])
        
        # Use create_tool_calling_agent instead of create_structured_chat_agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=self.verbose,
            max_iterations=10,  # Allow 2 iterations max
            early_stopping_method="force"  # Use supported method
        )

    async def run(self, text: str, chat_history: List[BaseMessage] = None) -> str:
        try:
            if self.verbose:
                cprint(f"[Agent] 🎯 Processing: '{text}'", "cyan")
            
            out = await self.agent.ainvoke({
                "input": text,
                "chat_history": chat_history or []  # Use provided chat history
            })
            
            # Handle both string and list outputs from agent
            if isinstance(out, dict):
                result = out.get("output", "")
            else:
                result = out
            
            # If result is a list (like from Anthropic), extract text content
            if isinstance(result, list):
                text_parts = []
                for item in result:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                result = " ".join(text_parts)
            
            # Now ensure it's a string
            result = str(result).strip()
            
            return result
            
        except Exception as e:
            if self.verbose:
                cprint(f"[Agent] ❌ Error: {e}", "red")
                import traceback
                traceback.print_exc()
            raise e

# ─── ConversationManager ────────────────────────────────────────────────────
class ConversationManager:
    def __init__(self, llm_config, system_prompt: str,
                 mcp_manager: MCPManager, db_path: str):
        self.cfg = llm_config
        self.system_prompt = system_prompt
        self.mcp = mcp_manager

        self.agent: Optional[ToolAwareAgent] = None
        self.fast_llm = None
        
        # NEW: Proper conversation memory using LangChain messages
        self.conversation_history: List[BaseMessage] = []
        self.max_history = 20  # Keep last 20 messages (10 exchanges)
        
        # Legacy history for fast_llm fallback
        self.history: List[Dict[str, Any]] = []

    # ── Init ────────────────────────────────────────────────────────────────
    async def initialize(self):
        prov = getattr(self.cfg, "provider", "google-genai")
        model = getattr(self.cfg, "model", "gemini-pro")
        temp  = getattr(self.cfg, "temperature", 0.7)
        key   = os.getenv(ENV_KEY.get(prov, ""), "")

        # Agent (tool‑aware) --------------------------------------------------
        if key and prov in LLM_WRAPPERS:
            try:
                self.agent = ToolAwareAgent(prov, self.mcp, model, temp, key, verbose=True)
                cprint(f"[LLM] ✅ {prov} agent ready", "green")
            except Exception as e:
                cprint(f"[LLM] ⚠️  agent init failed: {e}", "yellow")
                import traceback
                traceback.print_exc()

        # Fast model (no tools) ---------------------------------------------
        try:
            if prov == "google-genai" and key:
                import google.generativeai as genai
                genai.configure(api_key=key)
                self.fast_llm = genai.GenerativeModel(
                    model_name=model,
                    system_instruction=self.system_prompt,
                    generation_config={"temperature": temp},
                )
            elif prov == "anthropic" and key:
                import anthropic
                self.fast_llm = anthropic.Anthropic(api_key=key)
            elif prov == "openai" and key:
                import openai
                openai.api_key = key
                self.fast_llm = openai
            if self.fast_llm:
                cprint(f"[LLM] ✅ Fast model ready ({prov})", "green")
        except Exception as e:
            cprint(f"[LLM] ⚠️  fast model init failed: {e}", "yellow")

    # ── NEW: Memory management methods ──────────────────────────────────────
    def _add_to_conversation(self, role: str, content: str):
        """Add message to conversation history in LangChain format"""
        if role == "user":
            self.conversation_history.append(HumanMessage(content=content))
        else:
            self.conversation_history.append(AIMessage(content=content))
        
        # Keep only recent messages to prevent context overflow
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Also update legacy history for fast_llm fallback
        self._add_history(role, content)

    def get_conversation_context(self) -> List[BaseMessage]:
        """Get conversation history excluding the current message"""
        return self.conversation_history[:-1] if self.conversation_history else []

    def clear_conversation(self):
        """Clear conversation memory"""
        self.conversation_history = []
        self.history = []
        cprint("[Memory] 🔄 Conversation memory cleared", "yellow")

    # ── Main entry ──────────────────────────────────────────────────────────
    async def process_user_input_streaming(self, text: str, *_,
                                           tts_worker=None, audio_worker=None,
                                           debug=False):
        # Add user message to conversation history first
        self._add_to_conversation("user", text)
        
        # ALWAYS try the agent first if available
        if self.agent:
            try:
                if debug:
                    cprint(f"[LLM] 🎯 Using agent for: '{text}'", "cyan")
                    cprint(f"[Memory] 📚 Using {len(self.conversation_history)-1} previous messages", "cyan")
                
                # Pass conversation history to agent (excluding current user message)
                reply = await self.agent.run(text, self.get_conversation_context())
                
                if reply and reply.strip():
                    # Add assistant response to conversation history
                    self._add_to_conversation("assistant", reply)
                    
                    if debug:
                        cprint(f"[LLM] ✅ Agent replied: '{reply[:100]}...'", "green")
                    return await self._speak(reply, tts_worker, audio_worker)
                else:
                    if debug:
                        cprint("[LLM] ⚠️  Agent returned empty response", "yellow")
                    
            except Exception as e:
                cprint(f"[LLM] ❌ Agent error: {e}", "red")
                if debug:
                    import traceback
                    traceback.print_exc()
                # Don't fall back to fast stream, try to fix the issue
                error_response = "Sorry, I encountered an error with my tools. Let me try a different approach."
                self._add_to_conversation("assistant", error_response)
                return await self._speak(error_response, tts_worker, audio_worker)
        
        # Only use fast stream if no agent available
        if debug:
            cprint("[LLM] 🚀 Using fast stream (no agent available)", "yellow")
        return await self._fast_stream(text, tts_worker, audio_worker)

    # ── Fast path (provider‑specific) ───────────────────────────────────────
    async def _fast_stream(self, text: str, tts_worker, audio_worker):
        if not self.fast_llm:
            error_response = "Sorry, my language model is offline."
            self._add_to_conversation("assistant", error_response)
            return await self._speak(error_response, tts_worker, audio_worker)

        # Build conversation context from LangChain messages
        context_messages = []
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            if isinstance(msg, HumanMessage):
                context_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_messages.append(f"Assistant: {msg.content}")
        
        context = "\n".join(context_messages[:-1])  # Exclude current user message
        prompt = f"{context}\nUser: {text}\nAssistant:" if context else f"User: {text}\nAssistant:"

        def _call() -> str:
            # Gemini ---------------------------------------------------------
            if hasattr(self.fast_llm, "generate_content"):
                return self.fast_llm.generate_content(prompt).text
            # OpenAI ---------------------------------------------------------
            if hasattr(self.fast_llm, "ChatCompletion"):
                r = self.fast_llm.ChatCompletion.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                )
                return r["choices"][0]["message"]["content"]
            # Anthropic (Messages API) --------------------------------------
            if hasattr(self.fast_llm, "messages"):
                r = self.fast_llm.messages.create(
                    model=self.cfg.model,
                    max_tokens=800,
                    temperature=self.cfg.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(block.text for block in r.content)
            raise RuntimeError("Unknown LLM client")

        full = await asyncio.get_event_loop().run_in_executor(None, _call)
        self._add_to_conversation("assistant", full)
        return await self._speak(full, tts_worker, audio_worker)

    # ── Text‑only generator (CLI) ───────────────────────────────────────────
    async def _stream_llm_response(self, text: str):
        # Add user message to conversation history
        self._add_to_conversation("user", text)
        
        # Always try agent first if available
        if self.agent:
            try:
                reply = await self.agent.run(text, self.get_conversation_context())
                if reply and reply.strip():
                    self._add_to_conversation("assistant", reply)
                    # Yield response in chunks for text-only mode
                    for sent in re.split(r"(?<=[.!?])\s+", reply):
                        if sent:
                            yield sent + " "
                            await asyncio.sleep(0)
                    return
            except Exception as e:
                cprint(f"[LLM] Agent error in text mode: {e}", "red")
                error_msg = f"(Agent error: {e}) "
                self._add_to_conversation("assistant", error_msg)
                yield error_msg
        
        # Fallback to fast LLM with conversation context
        if not self.fast_llm:
            error_msg = "(LLM unavailable)"
            self._add_to_conversation("assistant", error_msg)
            yield error_msg
            return

        # Build conversation context
        context_messages = []
        for msg in self.conversation_history[-10:]:
            if isinstance(msg, HumanMessage):
                context_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_messages.append(f"Assistant: {msg.content}")
        
        context = "\n".join(context_messages[:-1])  # Exclude current user message
        prompt = f"{context}\nUser: {text}\nAssistant:" if context else f"User: {text}\nAssistant:"

        def _call() -> str:
            if hasattr(self.fast_llm, "generate_content"):
                return self.fast_llm.generate_content(prompt).text
            if hasattr(self.fast_llm, "ChatCompletion"):
                return self.fast_llm.ChatCompletion.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                )["choices"][0]["message"]["content"]
            if hasattr(self.fast_llm, "messages"):
                r = self.fast_llm.messages.create(
                    model=self.cfg.model,
                    max_tokens=800,
                    temperature=self.cfg.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(block.text for block in r.content)
            raise RuntimeError("Unknown LLM client")

        full = await asyncio.get_event_loop().run_in_executor(None, _call)
        self._add_to_conversation("assistant", full)
        for sent in re.split(r"(?<=[.!?])\s+", full):
            if sent:
                yield sent + " "
                await asyncio.sleep(0)

    # ── Helpers ─────────────────────────────────────────────────────────═══
    def _history_prompt(self, user_text: str) -> str:
        """Legacy method for backward compatibility"""
        hist = "\n".join(
            ("User: " if h["role"] == "user" else "Assistant: ") + h["content"]
            for h in self.history[-10:]
        )
        self._add_history("user", user_text)
        return f"{hist}\nUser: {user_text}\nAssistant:"

    async def _speak(self, text: str, tts_worker, audio_worker):
        if tts_worker and audio_worker:
            try:
                pipe = UltraLowLatencyTTSPipeline(tts_worker, audio_worker)
                pipe.start_pipeline()
                pipe.add_phrase(text)
                return pipe.finish_pipeline() or time.time()
            except Exception as e:
                cprint(f"[LLM] TTS pipeline error: {e}", "red")
                return time.time()
        else:
            # Text-only mode or missing components
            return time.time()

    def _add_history(self, role: str, content: str):
        """Legacy history method for backward compatibility"""
        self.history.append({"role": role, "content": content, "ts": datetime.now()})
        if len(self.history) > 20:
            self.history = self.history[-10:]

    _add_to_history = _add_history  # legacy alias

    async def close(self):
        cprint("[LLM] Conversation manager closed", "green")
