# llm/conversation_manager.py
"""
Simple Google Gemini Conversation Manager for Opi Voice Assistant
Uses Google's Gemini API directly without heavy dependencies
"""

import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("[LLM] Warning: google-generativeai not installed")
    print("      Install with: pip install google-generativeai")


class ConversationManager:
    """Simple conversation manager using Google Gemini."""
    
    def __init__(self, llm_config, system_prompt, mcp_manager, db_path):
        self.llm_config = llm_config
        self.system_prompt = system_prompt
        self.mcp_manager = mcp_manager
        self.db_path = db_path
        
        # Gemini model
        self.model = None
        self.chat_session = None
        
        # Conversation history (simple in-memory storage)
        self.conversation_history = []
        self.max_history = 10
        
    async def initialize(self):
        """Initialize the conversation manager."""
        print('[LLM] Initializing Google Gemini conversation manager...')
        
        if not GOOGLE_AI_AVAILABLE:
            print('[LLM] ❌ Google AI library not available')
            print('[LLM] Using fallback responses only')
            return
        
        if not self.llm_config.api_key:
            print('[LLM] ❌ No Google API key found')
            print('[LLM] Using fallback responses only')
            return
        
        try:
            # Configure Gemini
            genai.configure(api_key=self.llm_config.api_key)
            
            # Initialize model
            model_name = self.llm_config.model if hasattr(self.llm_config, 'model') else 'gemini-pro'
            
            # Generation config for voice assistant
            generation_config = {
                "temperature": getattr(self.llm_config, 'temperature', 0.7),
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": getattr(self.llm_config, 'max_tokens', 1000),
            }
            
            # Safety settings (relaxed for general conversation)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=self._get_enhanced_system_prompt()
            )
            
            print(f'[LLM] ✅ Google Gemini initialized: {model_name}')
            
        except Exception as e:
            print(f'[LLM] ❌ Failed to initialize Gemini: {e}')
            self.model = None
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt for voice assistant."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tools_info = ""
        if self.mcp_manager:
            tools = self.mcp_manager.get_tools()
            if tools:
                tool_names = [tool.name for tool in tools]
                tools_info = f"\n\nAvailable tools: {', '.join(tool_names)}"
        
        enhanced_prompt = f"""{self.system_prompt}

Current Information:
- Current time: {current_time}
- You are running on an Orange Pi single-board computer
- This is a voice conversation - keep responses conversational and concise
- Responses will be spoken aloud, so avoid long lists or complex formatting
- Be helpful, friendly, and engaging{tools_info}

Important: Keep responses uder 100 words unless specifically asked for detailed information."""
        
        return enhanced_prompt
    
    async def process_user_input(self, user_text: str, speech_end_time: float) -> AsyncGenerator[str, None]:
        """Process user input and generate streaming response."""
        
        # Try Gemini first
        if self.model:
            try:
                async for chunk in self._process_with_gemini(user_text):
                    yield chunk
                return
            except Exception as e:
                print(f"[LLM] Gemini error: {e}")
                # Fall through to fallback responses
        
        # Fallback responses when Gemini is not available
        async for chunk in self._get_fallback_response(user_text):
            yield chunk
    
    async def _process_with_gemini(self, user_text: str) -> AsyncGenerator[str, None]:
        """Process input using Google Gemini."""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user", 
            "content": user_text,
            "timestamp": datetime.now()
        })
        
        # Prepare conversation context
        messages = []
        for msg in self.conversation_history[-self.max_history:]:
            if msg["role"] == "user":
                messages.append(f"User: {msg['content']}")
            else:
                messages.append(f"Assistant: {msg['content']}")
        
        # Add current user input
        prompt = f"""Previous conversation:
{chr(10).join(messages[-6:]) if len(messages) > 1 else ""}

Current user input: {user_text}

Please respond as Opi, the helpful voice assistant. Keep it conversational and concise since this will be spoken aloud."""
        
        try:
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            
            if response.text:
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.text,
                    "timestamp": datetime.now()
                })
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                
                # Stream the response
                yield response.text
            else:
                yield "I'm having trouble generating a response right now."
                
        except Exception as e:
            print(f"[LLM] Gemini generation error: {e}")
            yield f"I encountered an error: {str(e)}"
    
    async def _get_fallback_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """Generate fallback responses when Gemini is not available."""
        user_lower = user_text.lower()
        
        # Enhanced fallback responses
        responses = {
            'hello': 'Hello! I am Opi, your voice assistant.',
            'hi': 'Hi there! How can I help you?',
            'how are you': 'I am doing well, thank you for asking!',
            'time': f'The current time is {datetime.now().strftime("%I:%M %p")}',
            'date': f'Today is {datetime.now().strftime("%A, %B %d, %Y")}',
            'system status': 'System is running normally. CPU and memory usage are within normal ranges.',
            'help': 'I can help with basic questions, time, date, and system information. I need a proper API connection for more advanced features.',
            'weather': 'I would check the weather for you, but I need my API connection working for that.',
            'what can you do': 'I can tell you the time, date, system status, and answer basic questions. With a proper API connection, I could do much more!',
            'airplane food': 'Airplane food is quite a mystery! It never tastes quite right, does it?',
            'imagine': 'Imagine all the people living life in peace - great song by John Lennon!',
            'thank you': 'You are very welcome!',
            'thanks': 'My pleasure!',
            'goodbye': 'Goodbye! Have a great day!',
            'test': 'Test successful! I can hear you clearly.',
            'orange pi': 'I am running on an Orange Pi single-board computer. It is a great little device!',
            'who are you': 'I am Opi, your voice assistant running on an Orange Pi computer.',
            'what is your name': 'My name is Opi. I am your voice assistant.',
        }
        
        # Check for keyword matches
        for keyword, response in responses.items():
            if keyword in user_lower:
                yield response
                return
        
        # Check for questions
        if any(word in user_lower for word in ['what', 'how', 'when', 'where', 'why', 'who']):
            yield f"That's an interesting question about '{user_text}'. I would need my full AI capabilities to give you a proper answer."
        elif any(word in user_lower for word in ['please', 'can you', 'could you']):
            yield f"I'd be happy to help with that, but I need my API connection working to handle that request."
        else:
            yield f"I heard you say '{user_text}'. I am still learning to respond to new things without my full AI connection!"
    
    async def close(self):
        """Close the conversation manager."""
        print('[LLM] ✅ Conversation manager closed')n
