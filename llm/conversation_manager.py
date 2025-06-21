# llm/conversation_manager.py
"""
Conversation Manager 4.4 – provider‑agnostic (Gemini, Claude, GPT‑4o)
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
from langchain.agents import AgentExecutor, create_structured_chat_agent

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
            # ChatAnthropic uses model_name param like ChatOpenAI
            self.llm = llm_cls(model_name=model, temperature=temp, anthropic_api_key=api_key)
        else:
            self.llm = llm_cls(model_name=model, temperature=temp, openai_api_key=api_key)

        self.agent = self._build_agent()

    def _wrap_tool(self, tool) -> StructuredTool:
        async def _run(**kwargs):
            res: ToolCallResult = await self.mcp.call_tool(tool.name, kwargs)
            return res.get_text_content() if res.success else f"[TOOL ERROR] {res.error_message}"
        sig = [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
               for n in tool.input_schema.get("properties", {})]
        _run.__signature__ = inspect.signature(lambda **_: None).replace(parameters=sig)
        return StructuredTool.from_function(
            _run,
            name=tool.name,
            description=tool.description,
            infer_schema=False,
        )

    def _build_agent(self):
        tools = [self._wrap_tool(t) for t in self.mcp.get_tools()]
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are Opi, an on‑device assistant running on an Orange Pi. "
             "Call a tool whenever it can answer better.\n\n"
             "Tools: {tool_names}\n\nSpecs:\n{tools}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        if self.verbose:
            cprint("\n=== TOOL PROMPT ===", "cyan", attrs=["bold"])
            cprint(str(prompt), "cyan")
            for t in tools:
                cprint(f" • {t.name}", "yellow")
            cprint("===================\n", "cyan", attrs=["bold"])
        chain = create_structured_chat_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=chain, tools=tools, verbose=self.verbose)

    async def run(self, text: str) -> str:
        out = await self.agent.ainvoke({"input": text})
        return out["output"].strip()

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
                self.agent = ToolAwareAgent(prov, self.mcp, model, temp, key)
                cprint(f"[LLM] ✅ {prov} agent ready", "green")
            except Exception as e:
                cprint(f"[LLM] ⚠️  agent init failed: {e}", "yellow")

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
        if self.agent:
            print(await mcp_manager.list_tools())

            try:
                reply = await self.agent.run(text)
                if reply:
                    return await self._speak(reply, tts_worker, audio_worker)
            except Exception as e:
                if debug:
                    cprint(f"[LLM] agent error: {e}", "yellow")
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
        pipe = UltraLowLatencyTTSPipeline(tts_worker, audio_worker)
        pipe.start_pipeline()
        pipe.add_phrase(text)
        return pipe.finish_pipeline() or time.time()

    def _add_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content, "ts": datetime.now()})
        if len(self.history) > 2 * self.max_history:
            self.history = self.history[-self.max_history:]

    _add_to_history = _add_history  # legacy alias

    async def close(self):
        cprint("[LLM] Conversation manager closed", "green")

