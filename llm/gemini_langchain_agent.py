# llm/gemini_langchain_agent.py
from __future__ import annotations
import asyncio, json, re, inspect
from typing import Dict, Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_structured_chat_agent

from .mcp_manager import MCPManager, ToolCallResult

class GeminiMCPAgent:
    """
    A super-thin LangChain agent that
      • presents every MCP tool as a StructuredTool
      • lets Gemini decide whether to call a tool
      • calls the tool through MCPManager
      • streams / yields any normal text back to caller
    """

    def __init__(self, mcp: MCPManager, api_key: str,
                 model_name: str = "gemini-pro", temperature: float = 0.7):
        self.mcp = mcp
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            # we want function-calling JSON back, LangChain handles that
            convert_system_message_to_openai=True,
        )
        self.agent: AgentExecutor = self._build_agent()

    # ------------------------------------------------------------------ utils
    def _mcp_tool_to_lc_tool(self, tool) -> StructuredTool:
        """Return a LangChain StructuredTool that simply forwards to MCPManager."""
        async def _run(**kwargs):
            res: ToolCallResult = await self.mcp.call_tool(tool.name, kwargs)
            if res.success:
                # Pass only TEXT back to the model; keep binary / image blobs out
                return res.get_text_content() or "(Tool returned no text output.)"
            return f"[TOOL ERROR] {res.error_message or 'unknown error'}"

        sig = inspect.signature(
            lambda **_: None)           # we only need *names* / *types*
        sig = sig.replace(parameters=[
            inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation=Any, default=inspect._empty)
            for k in tool.input_schema.get("properties", {}).keys()
        ])
        _run.__signature__ = sig       # let LC know about kw-args

        return StructuredTool.from_function(
            name=tool.name,
            description=tool.description,
            func=_run,
            coroutine=_run,            # identical async version
        )

    def _build_agent(self) -> AgentExecutor:
        lc_tools = [self._mcp_tool_to_lc_tool(t) for t in self.mcp.get_tools()]
        prompt = ChatGoogleGenerativeAI.get_default_prompt()  # Gem-pro default
        agent = create_structured_chat_agent(self.model, lc_tools, prompt)
        return AgentExecutor(agent=agent, tools=lc_tools, verbose=False)

    # ------------------------------------------------------------------ public
    async def run(self, user_text: str) -> str:
        """
        Executes a single turn.
        If Gemini chooses to call a tool, LangChain will call _run() above,
        wait for the MCP result, and then send the *tool output* back to the
        model so it can produce a natural language answer.
        The final answer we return here is always plain text – perfect for
        your existing phrase-streaming → TTS pipeline.
        """
        result = await self.agent.ainvoke({"input": user_text})
        return result["output"]

