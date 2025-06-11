# llm/conversation_manager.py
"""
Conversation Manager for Opi Voice Assistant
Handles LLM interactions, conversation state, and response generation
"""

import asyncio
import uuid
import time
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore

from config.settings import LLMConfig
from llm.mcp_manager import MCPManager, OpiSystemTools
from llm.memory_store import SqliteMemoryStore


class ConversationState:
    """Maintains conversation state and context."""
    
    def __init__(self):
        self.thread_id: str = uuid.uuid4().hex
        self.started_at: datetime = datetime.now()
        self.message_count: int = 0
        self.last_interaction: Optional[datetime] = None
        self.context: Dict[str, Any] = {}
        
    def new_interaction(self):
        """Mark a new interaction."""
        self.message_count += 1
        self.last_interaction = datetime.now()
        
    def is_stale(self, max_idle_minutes: int = 30) -> bool:
        """Check if conversation is stale and should be reset."""
        if not self.last_interaction:
            return False
        
        idle_time = datetime.now() - self.last_interaction
        return idle_time.total_seconds() > (max_idle_minutes * 60)


class ConversationManager:
    """Manages LLM conversations with MCP tool integration."""
    
    def __init__(self, 
                 llm_config: LLMConfig,
                 system_prompt: str,
                 mcp_manager: MCPManager,
                 db_path: str):
        self.llm_config = llm_config
        self.system_prompt = system_prompt
        self.mcp_manager = mcp_manager
        self.db_path = Path(db_path)
        
        # Components to be initialized
        self.llm: Optional[BaseChatModel] = None
        self.agent = None
        self.checkpointer: Optional[AsyncSqliteSaver] = None
        self.memory_store: Optional[SqliteMemoryStore] = None
        
        # Conversation state
        self.current_conversation: Optional[ConversationState] = None
        self.conversation_active = False
        
    async def initialize(self):
        """Initialize the conversation manager."""
        print("[LLM] Initializing conversation manager...")
        
        # Initialize LLM
        await self._init_llm()
        
        # Initialize storage
        await self._init_storage()
        
        # Initialize agent with tools
        await self._init_agent()
        
        print("[LLM] ✅ Conversation manager ready")
    
    async def _init_llm(self):
        """Initialize the language model."""
        try:
            # Configure extra parameters for specific providers
            extra_body = {}
            default_headers = {
                "User-Agent": "Opi-Voice-Assistant/1.0",
            }
            
            if self.llm_config.base_url and "openrouter" in self.llm_config.base_url:
                extra_body = {"transforms": ["middle-out"]}
                default_headers.update({
                    "HTTP-Referer": "https://github.com/your-repo/opi-voice-assistant",
                    "X-Title": "Opi Voice Assistant"
                })
            
            self.llm = init_chat_model(
                model=self.llm_config.model,
                model_provider=self.llm_config.provider,
                api_key=self.llm_config.api_key,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                base_url=self.llm_config.base_url,
                timeout=self.llm_config.timeout,
                default_headers=default_headers,
                extrabody=extra_body
            )
            
            print(f"[LLM] ✅ {self.llm_config.provider}/{self.llm_config.model} initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}")
    
    async def _init_storage(self):
        """Initialize conversation storage."""
        # Create database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpointer for conversation state
        self.checkpointer = AsyncSqliteSaver.from_conn_string(str(self.db_path))
        await self.checkpointer.__aenter__()
        
        # Initialize memory store
        self.memory_store = SqliteMemoryStore(str(self.db_path))
        await self.memory_store.initialize()
        
        print("[LLM] ✅ Storage initialized")
    
    async def _init_agent(self):
        """Initialize the LangGraph agent with tools."""
        # Collect all tools
        all_tools = []
        
        # Add MCP tools
        mcp_tools = self.mcp_manager.get_tools()
        all_tools.extend(mcp_tools)
        
        # Add built-in system tools
        system_tools = OpiSystemTools.get_tools()
        all_tools.extend(system_tools)
        
        # Add memory tools
        memory_tools = self.memory_store.get_tools()
        all_tools.extend(memory_tools)
        
        print(f"[LLM] Agent has {len(all_tools)} tools available")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_enhanced_system_prompt()),
            ("placeholder", "{messages}")
        ])
        
        # Create agent
        self.agent = create_react_agent(
            self.llm,
            all_tools,
            state_modifier=prompt,
            checkpointer=self.checkpointer,
            store=self.memory_store
        )
        
        print("[LLM] ✅ Agent created with tools")
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get system prompt enhanced with current context."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        enhanced_prompt = f"""{self.system_prompt}

Current Information:
- Current time: {current_time}
- You are running on an Orange Pi single-board computer
- You have access to various tools for system information, file operations, and web search
- Keep responses conversational and concise since they will be spoken aloud
- When users ask about system status, use the get_system_status tool
- You can save important information to memory for future reference

Available Tool Categories:
- System tools: Check status, audio devices, network info
- File operations: Read, write, and manage files (via MCP)
- Web search: Search for current information (if configured)
- Memory: Save and recall important information

Remember: You are Opi, a helpful voice assistant. Be friendly, concise, and helpful!"""
        
        return enhanced_prompt
    
    async def process_user_input(self, 
                                user_text: str, 
                                speech_end_time: float) -> AsyncGenerator[str, None]:
        """Process user input and generate streaming response."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        # Start new conversation if needed
        if not self.current_conversation or self.current_conversation.is_stale():
            self.current_conversation = ConversationState()
            print(f"[LLM] Started new conversation: {self.current_conversation.thread_id}")
        
        self.current_conversation.new_interaction()
        self.conversation_active = True
        
        try:
            # Create user message
            user_message = HumanMessage(content=user_text)
            
            # Prepare agent input
            agent_input = {
                "messages": [user_message],
                "today_datetime": datetime.now().isoformat(),
                "speech_end_time": speech_end_time
            }
            
            # Configure agent execution
            config = {
                "configurable": {
                    "thread_id": self.current_conversation.thread_id,
                    "user_id": "opi_user"
                },
                "recursion_limit": 10
            }
            
            # Stream agent response
            response_buffer = ""
            async for chunk in self.agent.astream(
                agent_input,
                config=config,
                stream_mode=["messages"]
            ):
                if "messages" in chunk:
                    message = chunk["messages"][-1]
                    
                    if isinstance(message, AIMessage) and message.content:
                        # Handle tool calls
                        if message.tool_calls:
                            yield f"[Using tools: {', '.join([tc['name'] for tc in message.tool_calls])}] "
                        
                        # Stream content
                        if isinstance(message.content, str):
                            new_content = message.content[len(response_buffer):]
                            response_buffer = message.content
                            
                            if new_content:
                                yield new_content
                        
                        elif isinstance(message.content, list):
                            # Handle complex content (text + other types)
                            for content_part in message.content:
                                if isinstance(content_part, dict) and content_part.get("type") == "text":
                                    text = content_part.get("text", "")
                                    new_content = text[len(response_buffer):]
                                    response_buffer = text
                                    
                                    if new_content:
                                        yield new_content
            
            # Save conversation context
            await self._save_conversation_context(user_text, response_buffer)
            
        except Exception as e:
            error_message = f"I encountered an error: {str(e)}"
            print(f"[LLM] Error processing input: {e}")
            yield error_message
            
        finally:
            self.conversation_active = False
    
    async def _save_conversation_context(self, user_input: str, ai_response: str):
        """Save important conversation context to memory."""
        try:
            # Extract key information that might be worth remembering
            context_items = []
            
            # Save user preferences or important statements
            if any(keyword in user_input.lower() for keyword in [
                "my name is", "i am", "i like", "i prefer", "remember", "important"
            ]):
                context_items.append(f"User said: {user_input}")
            
            # Save successful tool interactions
            if "[Using tools:" in ai_response:
                context_items.append(f"Successfully helped with: {user_input}")
            
            # Save to memory store
            for item in context_items:
                await self.memory_store.save_memory([item], user_id="opi_user")
                
        except Exception as e:
            print(f"[LLM] Warning: Could not save conversation context: {e}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get current conversation statistics."""
        if not self.current_conversation:
            return {"status": "no_active_conversation"}
        
        return {
            "thread_id": self.current_conversation.thread_id,
            "started_at": self.current_conversation.started_at.isoformat(),
            "message_count": self.current_conversation.message_count,
            "last_interaction": self.current_conversation.last_interaction.isoformat() 
                               if self.current_conversation.last_interaction else None,
            "conversation_active": self.conversation_active
        }
    
    async def reset_conversation(self):
        """Reset the current conversation."""
        self.current_conversation = None
        self.conversation_active = False
        print("[LLM] Conversation reset")
    
    async def close(self):
        """Close the conversation manager and cleanup resources."""
        print("[LLM] Closing conversation manager...")
        
        self.conversation_active = False
        
        if self.checkpointer:
            try:
                await self.checkpointer.__aexit__(None, None, None)
            except Exception as e:
                print(f"[LLM] Warning: Error closing checkpointer: {e}")
        
        if self.memory_store:
            await self.memory_store.close()
        
        print("[LLM] ✅ Conversation manager closed")_
