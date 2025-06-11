# llm/memory_store.py
"""
Memory Store for Opi Voice Assistant
Provides persistent memory storage and retrieval with semantic search
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import aiosqlite

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore


class SqliteMemoryStore(BaseStore):
    """SQLite-based memory store for conversation context and user preferences."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize the memory store database."""
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await db.commit()
        
        print("[Memory] ✅ Memory store initialized")
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create database tables for memory storage."""
        # Memories table for storing user memories and context
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'general',
                importance INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Index for faster queries
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id 
            ON memories(user_id)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type 
            ON memories(memory_type)
        """)
        
        # Conversation summaries table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    async def save_memory(self, memories: List[str], user_id: str = "opi_user", 
                         memory_type: str = "general", importance: int = 1) -> str:
        """Save memories to the store."""
        async with aiosqlite.connect(self.db_path) as db:
            saved_count = 0
            for memory_content in memories:
                # Check if similar memory already exists
                if not await self._memory_exists(db, user_id, memory_content):
                    memory_id = uuid.uuid4().hex
                    await db.execute("""
                        INSERT INTO memories (id, user_id, content, memory_type, importance)
                        VALUES (?, ?, ?, ?, ?)
                    """, (memory_id, user_id, memory_content, memory_type, importance))
                    saved_count += 1
            
            await db.commit()
            return f"Saved {saved_count} new memories"
    
    async def _memory_exists(self, db: aiosqlite.Connection, user_id: str, content: str) -> bool:
        """Check if a similar memory already exists."""
        # Simple check for exact content match
        async with db.execute("""
            SELECT COUNT(*) FROM memories 
            WHERE user_id = ? AND content = ?
        """, (user_id, content)) as cursor:
            result = await cursor.fetchone()
            return result[0] > 0
    
    async def get_memories(self, user_id: str = "opi_user", 
                          memory_type: Optional[str] = None,
                          limit: int = 50) -> List[str]
        """Retrieve memories for a user."""
        async with aiosqlite.connect(self.db_path) as db:
            if memory_type:
                query = """
                    SELECT content FROM memories 
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """
                params = (user_id, memory_type, limit)
            else:
                query = """
                    SELECT content FROM memories 
                    WHERE user_id = ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                """
                params = (user_id, limit)
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
    
    async def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[str]:
        """Search memories using simple text matching."""
        async with aiosqlite.connect(self.db_path) as db:
            # Simple text search using LIKE
            search_query = f"%{query.lower()}%"
            async with db.execute("""
                SELECT content FROM memories 
                WHERE user_id = ? AND LOWER(content) LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (user_id, search_query, limit)) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
    
    async def update_memory_importance(self, user_id: str, content: str, importance: int):
        """Update the importance of a memory."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE memories 
                SET importance = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND content = ?
            """, (importance, user_id, content))
            await db.commit()
    
    async def delete_memory(self, user_id: str, content: str) -> bool:
        """Delete a specific memory."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                DELETE FROM memories 
                WHERE user_id = ? AND content = ?
            """, (user_id, content))
            await db.commit()
            return cursor.rowcount > 0
    
    async def save_conversation_summary(self, thread_id: str, summary: str, 
                                      message_count: int, user_id: str = "opi_user"):
        """Save a conversation summary."""
        async with aiosqlite.connect(self.db_path) as db:
            summary_id = uuid.uuid4().hex
            await db.execute("""
                INSERT INTO conversation_summaries 
                (id, user_id, thread_id, summary, message_count, ended_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (summary_id, user_id, thread_id, summary, message_count))
            await db.commit()
    
    async def get_recent_summaries(self, user_id: str = "opi_user", limit: int = 5) -> List[str]:
        """Get recent conversation summaries."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT summary FROM conversation_summaries 
                WHERE user_id = ?
                ORDER BY ended_at DESC
                LIMIT ?
            """, (user_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
    
    def get_tools(self) -> List:
        """Get LangChain tools for memory operations."""
        
        @tool
        async def save_memory_tool(memories: List[str], *, 
                                  config: RunnableConfig = None,
                                  memory_type: str = "general",
                                  importance: int = 1) -> str:
            """Save important information to memory for future reference.
            
            Args:
                memories: List of memory items to save
                memory_type: Type of memory (general, preference, fact, etc.)
                importance: Importance level 1-5 (5 being most important)
            """
            user_id = "opi_user"
            if config and "configurable" in config:
                user_id = config["configurable"].get("user_id", "opi_user")
            
            return await self.save_memory(memories, user_id, memory_type, importance)
        
        @tool
        async def recall_memories(query: str = "", *, 
                                 config: RunnableConfig = None,
                                 memory_type: str = None,
                                 limit: int = 10) -> str:
            """Recall saved memories, optionally filtered by query or type.
            
            Args:
                query: Search query to find relevant memories
                memory_type: Type of memory to filter by
                limit: Maximum number of memories to return
            """
            user_id = "opi_user"
            if config and "configurable" in config:
                user_id = config["configurable"].get("user_id", "opi_user")
            
            if query:
                memories = await self.search_memories(user_id, query, limit)
            else:
                memories = await self.get_memories(user_id, memory_type, limit)
            
            if memories:
                return f"Found {len(memories)} memories:\n" + "\n".join(f"- {m}" for m in memories)
            else:
                return "No memories found matching your criteria."
        
        @tool
        async def delete_memory_tool(memory_content: str, *,
                                   config: RunnableConfig = None) -> str:
            """Delete a specific memory.
            
            Args:
                memory_content: The exact content of the memory to delete
            """
            user_id = "opi_user"
            if config and "configurable" in config:
                user_id = config["configurable"].get("user_id", "opi_user")
            
            deleted = await self.delete_memory(user_id, memory_content)
            if deleted:
                return f"Memory deleted: {memory_content}"
            else:
                return f"Memory not found: {memory_content}"
        
        @tool
        async def get_conversation_context() -> str:
            """Get recent conversation context and summaries."""
            summaries = await self.get_recent_summaries(limit=3)
            memories = await self.get_memories(memory_type="preference", limit=5)
            
            context = "Recent Context:\n"
            if summaries:
                context += "Recent conversations:\n"
                for i, summary in enumerate(summaries, 1):
                    context += f"{i}. {summary}\n"
            
            if memories:
                context += "\nUser preferences/info:\n"
                for memory in memories:
                    context += f"- {memory}\n"
            
            return context if summaries or memories else "No recent context available."
        
        return [save_memory_tool, recall_memories, delete_memory_tool, get_conversation_context]
    
    # BaseStore interface implementation (simplified)
    def batch(self, ops) -> List:
        """Synchronous batch operations - not implemented."""
        raise NotImplementedError("Use abatch for async operations")
    
    async def abatch(self, ops) -> List:
        """Asynchronous batch operations - simplified implementation."""
        # For now, just return empty results
        # This can be expanded if needed for LangGraph integration
        return [None] * len(ops)
    
    async def close(self):
        """Close the memory store."""
        print("[Memory] ✅ Memory store closed")


# Utility functions for memory management
class MemoryManager:
    """High-level memory management utilities."""
    
    def __init__(self, memory_store: SqliteMemoryStore):
        self.store = memory_store
    
    async def extract_and_save_important_info(self, conversation_text: str, 
                                            user_id: str = "opi_user") -> List[str]:
        """Extract and save important information from conversation text."""
        important_patterns = [
            ("my name is", "preference"),
            ("i am", "preference"), 
            ("i like", "preference"),
            ("i prefer", "preference"),
            ("remember", "important"),
            ("don't forget", "important"),
            ("important", "important"),
            ("always", "preference"),
            ("never", "preference")
        ]
        
        saved_memories = []
        text_lower = conversation_text.lower()
        
        for pattern, memory_type in important_patterns:
            if pattern in text_lower:
                # Extract the relevant sentence
                sentences = conversation_text.split('.')
                for sentence in sentences:
                    if pattern in sentence.lower():
                        clean_sentence = sentence.strip()
                        if clean_sentence:
                            await self.store.save_memory(
                                [clean_sentence], 
                                user_id, 
                                memory_type, 
                                importance=3
                            )
                            saved_memories.append(clean_sentence)
        
        return saved_memories
    
    async def get_relevant_context(self, user_input: str, 
                                 user_id: str = "opi_user") -> str:
        """Get relevant context for the current user input."""
        # Search for relevant memories
        relevant_memories = await self.store.search_memories(user_id, user_input, limit=5)
        
        # Get recent conversation summaries
        recent_summaries = await self.store.get_recent_summaries(user_id, limit=2)
        
        context_parts = []
        
        if relevant_memories:
            context_parts.append("Relevant memories:")
            for memory in relevant_memories:
                context_parts.append(f"- {memory}")
        
        if recent_summaries:
            context_parts.append("\nRecent conversations:")
            for summary in recent_summaries:
                context_parts.append(f"- {summary}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    async def cleanup_old_memories(self, days_old: int = 90, 
                                 user_id: str = "opi_user") -> int:
        """Clean up old, low-importance memories."""
        async with aiosqlite.connect(self.store.db_path) as db:
            cursor = await db.execute("""
                DELETE FROM memories 
                WHERE user_id = ? 
                AND importance < 3 
                AND created_at < datetime('now', '-{} days')
            """.format(days_old), (user_id,))
            
            await db.commit()
            return cursor.rowcount:
