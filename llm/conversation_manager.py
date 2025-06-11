from typing import AsyncGenerator
from datetime import datetime

class ConversationManager:
    def __init__(self, llm_config, system_prompt, mcp_manager, db_path):
        self.llm_config = llm_config
        self.system_prompt = system_prompt
        self.mcp_manager = mcp_manager
        
    async def initialize(self):
        print('[LLM] Initializing conversation manager...')
        print('[LLM] ✅ Conversation manager ready')
    
    async def process_user_input(self, user_text: str, speech_end_time: float) -> AsyncGenerator[str, None]:
        # Simple responses for testing
        responses = {
            'system status': 'CPU: 45%, Memory: 60%',
            'hello': 'Hello! I am Opi, your voice assistant.',
            'time': f'The current time is {datetime.now().strftime("%H:%M:%S")}',
            'help': 'I can help with system status, time, and basic questions.',
        }
        
        user_lower = user_text.lower()
        for keyword, response in responses.items():
            if keyword in user_lower:
                yield response
                return
        
        yield f'I heard you say: {user_text}'
    
    async def close(self):
        print('[LLM] ✅ Conversation manager closed')
