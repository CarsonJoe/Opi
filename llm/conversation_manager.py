# llm/conversation_manager.py
"""
Conversation Manager 4.5 – FIXED tool calling with proper error handling
--------------------------------------------------------------------
Edit `config.json → llm.provider` to switch backend:
  • google-genai   – Gemini      (env GOOGLE_API_KEY)
  • anthropic      – Claude 3/4  (env ANTHROPIC_API_KEY)
  • openai         – GPT‑4/3.5   (env OPENAI_API_KEY)
"""

from __future__ import annotations

import asyncio, inspect, os, re, time
from datetime import datetime
from typing import Any, Dict, List, Optional

from termcolor import cprint
from .phrase_stream import PhraseStreamer, UltraLowLatencyTTSPipeline
from .mcp_manager import MCPManager, ToolCallResult

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
        self.verbose = verbose or os.getenv("OPI_DEBUG", "0") == "1"
        self.mcp = mcp

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
                cprint(f"[Agent] 🔧 Calling MCP tool: {tool.name} with {kwargs}", "cyan")
            
            # Clean kwargs - remove any LangChain-specific parameters
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['config', 'configurable']}
            
            try:
                import concurrent.futures
                import threading
                import signal
                
                def run_async_tool():
                    """Run the async tool in a separate thread with its own event loop."""
                    try:
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        
                        # Create the task with timeout
                        async def call_with_timeout():
                            return await asyncio.wait_for(
                                self.mcp.call_tool(tool.name, clean_kwargs),
                                timeout=2.0  # 2 second timeout (should be ~0.01s)
                            )
                        
                        try:
                            return new_loop.run_until_complete(call_with_timeout())
                        except asyncio.TimeoutError:
                            cprint(f"[Agent] ⏰ Tool '{tool.name}' timed out after 2 seconds", "yellow")
                            
                            class MockTimeoutResult:
                                def __init__(self, tool_name):
                                    self.success = False
                                    self.error_message = f"Tool '{tool_name}' timed out"
                                    self.tool_name = tool_name
                                
                                def get_text_content(self):
                                    return f"Tool '{self.tool_name}' timed out after 2 seconds"
                            
                            return MockTimeoutResult(tool.name)
                        finally:
                            new_loop.close()
                            
                    except Exception as e:
                        cprint(f"[Agent] 💥 Tool thread error: {e}", "red")
                        
                        class MockErrorResult:
                            def __init__(self, error_msg):
                                self.success = False
                                self.error_message = error_msg
                            
                            def get_text_content(self):
                                return f"Tool execution failed: {self.error_message}"
                        
                        return MockErrorResult(str(e))
                
                # Execute in thread pool with timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async_tool)
                    try:
                        res = future.result(timeout=3.0)  # 3 second total timeout
                    except concurrent.futures.TimeoutError:
                        cprint(f"[Agent] ⏰ Tool '{tool.name}' execution timed out", "red")
                        return f"[TIMEOUT] Tool '{tool.name}' took too long to execute"
                    
            except Exception as e:
                if self.verbose:
                    cprint(f"[Agent] ❌ Tool execution error: {e}", "red")
                    import traceback
                    traceback.print_exc()
                return f"[TOOL ERROR] Tool execution failed: {str(e)}"
            
            # Process result
            try:
                if self.verbose:
                    success = getattr(res, 'success', False)
                    cprint(f"[Agent] 📄 Tool result: success={success}", "cyan")
                    
                    if hasattr(res, 'get_text_content'):
                        content = res.get_text_content()
                        content_preview = content[:200] + "..." if len(content) > 200 else content
                        cprint(f"[Agent] 📄 Content: {content_preview}", "white")
                
                if hasattr(res, 'success') and res.success:
                    return res.get_text_content() if hasattr(res, 'get_text_content') else str(res)
                else:
                    error_msg = getattr(res, 'error_message', 'Unknown error')
                    return f"[TOOL ERROR] {error_msg}"
                    
            except Exception as e:
                cprint(f"[Agent] ❌ Error processing tool result: {e}", "red")
                return f"[TOOL ERROR] Failed to process result: {str(e)}"
        
        # Create proper parameter signature for Gemini compatibility
        sig_params = []
        properties = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])
        
        # If no properties, create a simple function
        if not properties:
            sig_params = []
        else:
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")
                # Convert JSON schema types to Python types
                python_type = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": list,
                    "object": dict
                }.get(param_type, str)
                
                default = inspect.Parameter.empty if param_name in required else None
                sig_params.append(
                    inspect.Parameter(
                        param_name, 
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default,
                        annotation=python_type
                    )
                )
        
        _run.__signature__ = inspect.Signature(sig_params)
        
        # Create a clean schema for LangChain/Gemini
        clean_schema = {}
        if properties:
            clean_schema = {
                "type": "object",
                "properties": {},
                "required": required
            }
            
            for prop_name, prop_info in properties.items():
                clean_prop = {"type": prop_info.get("type", "string")}
                if "description" in prop_info:
                    clean_prop["description"] = prop_info["description"]
                # Handle array types properly for Gemini
                if prop_info.get("type") == "array" and "items" in prop_info:
                    clean_prop["items"] = prop_info["items"]
                clean_schema["properties"][prop_name] = clean_prop
        
        return StructuredTool.from_function(
            _run,
            name=tool.name,
            description=tool.description,
            args_schema=None if not clean_schema else None,  # Let LangChain infer from signature
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
            max_iterations=3,  # Limit iterations to prevent loops
            early_stopping_method="force"  # Stop early if no more tools needed
        )

    async def run(self, text: str) -> str:
        try:
            if self.verbose:
                cprint(f"[Agent] 🎯 Processing: '{text}'", "cyan")
            
            out = await self.agent.ainvoke({
                "input": text,
                "chat_history": []  # Add proper chat history if needed
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
            
            if self.verbose:
                cprint(f"[Agent] ✅ Result: '{result}'", "green")
            
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
        self.history: List[Dict[str, Any]] = []
        self.max_history = 10

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

    # ── Main entry ──────────────────────────────────────────────────────────
    async def process_user_input_streaming(self, text: str, *_,
                                           tts_worker=None, audio_worker=None,
                                           debug=False):
        # ALWAYS try the agent first if available
        if self.agent:
            try:
                if debug:
                    cprint(f"[LLM] 🎯 Using agent for: '{text}'", "cyan")
                
                reply = await self.agent.run(text)
                
                if reply and reply.strip():
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
                return await self._speak("Sorry, I encountered an error with my tools. Let me try a different approach.", tts_worker, audio_worker)
        
        # Only use fast stream if no agent available
        if debug:
            cprint("[LLM] 🚀 Using fast stream (no agent available)", "yellow")
        return await self._fast_stream(text, tts_worker, audio_worker)

    # ── Fast path (provider‑specific) ───────────────────────────────────────
    async def _fast_stream(self, text: str, tts_worker, audio_worker):
        if not self.fast_llm:
            return await self._speak("Sorry, my language model is offline.",
                                     tts_worker, audio_worker)

        prompt = self._history_prompt(text)

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
        self._add_history("assistant", full)
        return await self._speak(full, tts_worker, audio_worker)

    # ── Text‑only generator (CLI) ───────────────────────────────────────────
    async def _stream_llm_response(self, text: str):
        # Always try agent first if available
        if self.agent:
            try:
                reply = await self.agent.run(text)
                if reply and reply.strip():
                    self._add_history("assistant", reply)
                    # Yield response in chunks for text-only mode
                    for sent in re.split(r"(?<=[.!?])\s+", reply):
                        if sent:
                            yield sent + " "
                            await asyncio.sleep(0)
                    return
            except Exception as e:
                cprint(f"[LLM] Agent error in text mode: {e}", "red")
                yield f"(Agent error: {e}) "
        
        # Fallback to fast LLM
        if not self.fast_llm:
            yield "(LLM unavailable)"; return

        prompt = self._history_prompt(text)

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
        self._add_history("assistant", full)
        for sent in re.split(r"(?<=[.!?])\s+", full):
            if sent:
                yield sent + " "
                await asyncio.sleep(0)

    # ── Helpers ─────────────────────────────────────────────────────────═══
    def _history_prompt(self, user_text: str) -> str:
        hist = "\n".join(
            ("User: " if h["role"] == "user" else "Assistant: ") + h["content"]
            for h in self.history[-self.max_history:]
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
        self.history.append({"role": role, "content": content, "ts": datetime.now()})
        if len(self.history) > 2 * self.max_history:
            self.history = self.history[-self.max_history:]

    _add_to_history = _add_history  # legacy alias

    async def close(self):
        cprint("[LLM] Conversation manager closed", "green")
