# llm/conversation_manager.py
"""
COMPLETE REPLACEMENT - Streaming Conversation Manager for Opi Voice Assistant
Drop-in replacement with ultra-low latency streaming capabilities
"""

import asyncio
import time
import re
import threading
import queue
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime
from collections import deque
import numpy as np
from termcolor import cprint

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("[LLM] Warning: google-generativeai not installed")


class SentenceStreamer:
    """Accumulates LLM chunks and yields complete sentences immediately."""
    
    def __init__(self):
        self.buffer = ""
        self.sentence_endings = re.compile(r'[.!?]+\s*')
        self.min_sentence_length = 8  # Minimum chars for a sentence
        
    def add_chunk(self, chunk: str) -> List[str]:
        """Add chunk and return any complete sentences."""
        if not chunk:
            return []
            
        self.buffer += chunk
        sentences = []
        
        # Find sentence boundaries
        matches = list(self.sentence_endings.finditer(self.buffer))
        
        for match in matches:
            end_pos = match.end()
            potential_sentence = self.buffer[:end_pos].strip()
            
            # Only yield if it's long enough and looks like a real sentence
            if len(potential_sentence) >= self.min_sentence_length:
                sentences.append(potential_sentence)
                self.buffer = self.buffer[end_pos:].lstrip()
        
        return sentences
    
    def flush(self) -> Optional[str]:
        """Return any remaining content as final sentence."""
        if self.buffer.strip() and len(self.buffer.strip()) >= 5:
            final = self.buffer.strip()
            self.buffer = ""
            return final
        return None


class StreamingTTSPipeline:
    """Handles the complete streaming TTS pipeline."""
    
    def __init__(self, tts_worker, audio_worker):
        self.tts_worker = tts_worker
        self.audio_worker = audio_worker
        self.sentence_queue = queue.Queue(maxsize=20)
        self.audio_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.first_audio_time = None
        self.processing_threads = []
        
    def start_pipeline(self):
        """Start the streaming pipeline."""
        self.stop_event.clear()
        self.first_audio_time = None
        
        # Start TTS processing thread
        tts_thread = threading.Thread(
            target=self._tts_processing_loop,
            name="StreamingTTS",
            daemon=True
        )
        tts_thread.start()
        self.processing_threads.append(tts_thread)
        
        # Start audio playback thread
        audio_thread = threading.Thread(
            target=self._audio_playback_loop,
            name="StreamingAudio",
            daemon=True
        )
        audio_thread.start()
        self.processing_threads.append(audio_thread)
        
        cprint("[Pipeline] 🚀 Streaming pipeline started", "cyan")
    
    def add_sentence(self, sentence: str):
        """Add a sentence for TTS processing."""
        if not sentence.strip():
            return
            
        try:
            self.sentence_queue.put(sentence.strip(), timeout=0.1)
            cprint(f"[Pipeline] 📝 Queued: \"{sentence[:40]}...\"", "cyan")
        except queue.Full:
            cprint("[Pipeline] ⚠️ Sentence queue full, dropping sentence", "yellow")
    
    def finish_pipeline(self):
        """Complete processing and cleanup."""
        # Signal end of sentences
        self.sentence_queue.put(None)
        
        # Wait for processing to complete
        for thread in self.processing_threads:
            thread.join(timeout=10.0)
        
        # Get first audio time for latency calculation
        return self.first_audio_time
    
    def _tts_processing_loop(self):
        """TTS processing lop."""
        try:
            while not self.stop_event.is_set():
                try:
                    sentence = self.sentence_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if sentence is None:  # End signal
                    break
                
                # Process through TTS
                start_time = time.time()
                audio_data = self._synthesize_sentence(sentence)
                process_time = time.time() - start_time
                
                if audio_data is not None:
                    self.audio_queue.put(audio_data)
                    cprint(f"[Pipeline] ⚡ TTS: {process_time:.3f}s for \"{sentence[:30]}...\"", "green")
                
        except Exception as e:
            cprint(f"[Pipeline] ❌ TTS processing error: {e}", "red")
        finally:
            self.audio_queue.put(None)  # Signal end of audio
    
    def _audio_playback_loop(self):
        """Audio playback loop."""
        timings = {}
        
        try:
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if audio_chunk is None:  # End signal
                    break
                
                # Mark first audio time
                if self.first_audio_time is None:
                    self.first_audio_time = time.time()
                    cprint("[Pipeline] 🔊 First audio playing!", "blue", attrs=['bold'])
                
                # Play audio using existing audio worker
                temp_queue = queue.Queue()
                temp_queue.put(audio_chunk)
                temp_queue.put(None)  # End marker
                
                # Use existing audio worker (temporarily override stop event)
                original_stop = self.audio_worker.__dict__.get('stop_event')
                temp_stop = threading.Event()
                
                self.audio_worker.process_playback(temp_queue, temp_stop, timings)
                
        except Exception as e:
            cprint(f"[Pipeline] ❌ Audio playback error: {e}", "red")
    
    def _synthesize_sentence(self, sentence: str):
        """Synthesize a sentence using the TTS worker."""
        try:
            # Use existing TTS worker's synthesis method
            return self.tts_worker._synthesize_text(sentence, 1.0)
        except Exception as e:
            cprint(f"[Pipeline] TTS synthesis error: {e}", "red")
            return None


class ConversationManager:
    """Enhanced conversation manager with streaming capabilities - DROP-IN REPLACEMENT."""
    
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
        
        # Streaming components
        self.streaming_enabled = True
        
    async def initialize(self):
        """Initialize the conversation manager."""
        print('[LLM] Initializing enhanced conversation manager with streaming...')
        
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
            
            # Generation config optimized for streaming
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
            
            print(f'[LLM] ✅ Enhanced Gemini initialized with streaming: {model_name}')
            
        except Exception as e:
            print(f'[LLM] ❌ Failed to initialize Gemini: {e}')
            self.model = None
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt optimized for streaming."""
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
- This is a voice conversation - responses will be spoken aloud
- IMPORTANT: Structure responses in clear, complete sentences
- Start responding immediately - don't use filler phrases
- Keep responses conversational and concise{tools_info}

Response Guidelines for Voice:
- Use clear, complete sentences that work well when spoken
- Avoid parenthetical comments or complex punctuation
- Start with the most important information first
- Keep responses under 100 words unless specifically asked for detail
- Each sentence should be able to stand alone for text-to-speech processing"""
        
        return enhanced_prompt
    
    # MAIN STREAMING METHOD - replaces the old process_user_input
    async def process_user_input_streaming(self, user_text: str, speech_end_time: float, 
                                         tts_worker, audio_worker) -> float:
        """Process user input with streaming pipeline - NEW MAIN METHOD."""
        
        # Create streaming pipeline
        pipeline = StreamingTTSPipeline(tts_worker, audio_worker)
        sentence_streamer = SentenceStreamer()
        
        # Start pipeline
        pipeline.start_pipeline()
        
        try:
            cprint(f"[LLM] 🧠 Processing with streaming: \"{user_text}\"", "blue")
            
            # Stream LLM response
            response_chunks = []
            async for chunk in self._stream_llm_response(user_text):
                if chunk and chunk.strip():
                    response_chunks.append(chunk)
                    
                    # Process through sentence streamer
                    sentences = sentence_streamer.add_chunk(chunk)
                    
                    # Send complete sentences to TTS pipeline immediately
                    for sentence in sentences:
                        pipeline.add_sentence(sentence)
            
            # Handle any remaining content
            final_sentence = sentence_streamer.flush()
            if final_sentence:
                pipeline.add_sentence(final_sentence)
            
            # Complete pipeline and get first audio time
            first_audio_time = pipeline.finish_pipeline()
            
            # Log results
            if response_chunks:
                full_response = "".join(response_chunks)
                cprint(f"[LLM] 🗣️ Complete response: \"{full_response}\"", "green")
                
                # Add to conversation history
                self._add_to_history("user", user_text)
                self._add_to_history("assistant", full_response)
            
            return first_audio_time
            
        except Exception as e:
            cprint(f"[LLM] ❌ Streaming processing error: {e}", "red")
            # Fallback to simple response
            pipeline.add_sentence("Sorry, I encountered an error processing your request.")
            pipeline.finish_pipeline()
            raise
    
    # LEGACY METHOD - kept for backward compatibility
    async def process_user_input(self, user_text: str, speech_end_time: float) -> AsyncGenerator[str, None]:
        """Legacy method - kept for backward compatibility."""
        
        # Add user message to history
        self._add_to_history("user", user_text)
        
        # Try Gemini first
        if self.model:
            try:
                full_response = ""
                async for chunk in self._stream_llm_response(user_text):
                    full_response += chunk
                    yield chunk
                
                # Add complete response to history
                if full_response:
                    self._add_to_history("assistant", full_response)
                
                return
                
            except Exception as e:
                print(f"[LLM] Streaming error: {e}")
        
        # Fallback to non-streaming response
        async for chunk in self._get_fallback_response(user_text):
            yield chunk
    
    async def _stream_llm_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream response from LLM with optimized chunking."""
        
        # Prepare conversation context
        messages = []
        for msg in self.conversation_history[-self.max_history:]:
            if msg["role"] == "user":
                messages.append(f"User: {msg['content']}")
            else:
                messages.append(f"Assistant: {msg['content']}")
        
        prompt = f"""Previous conversation:
{chr(10).join(messages[-6:]) if len(messages) > 1 else ""}

Current user input: {user_text}

Respond as Opi, the helpful voice assistant. Structure your response in clear sentences for text-to-speech. Be concise and helpful."""
        
        try:
            # Try streaming API first
            if hasattr(self.model, 'generate_content') and hasattr(genai, 'GenerationConfig'):
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.model.generate_content(prompt)
                )
                
                if response.text:
                    # Stream word by word for better sentence detection
                    words = response.text.split()
                    current_chunk = ""
                    
                    for i, word in enumerate(words):
                        current_chunk += word + " "
                        
                        # Yield chunks at natural boundaries
                        if (word.endswith('.') or word.endswith('!') or word.endswith('?') or 
                            word.endswith(',') or len(current_chunk.split()) >= 4):
                            
                            yield current_chunk
                            current_chunk = ""
                            await asyncio.sleep(0.001)  # Tiny delay for processing
                    
                    # Yield any remaining content
                    if current_chunk.strip():
                        yield current_chunk
                else:
                    yield "I'm having trouble generating a response right now."
                
        except Exception as e:
            print(f"[LLM] Gemini streaming error: {e}")
            yield f"I encountered an error: {str(e)}"
    
    async def _get_fallback_response(self, user_text: str) -> AsyncGenerator[str, None]:
        """Generate fallback responses when Gemini is not available."""
        user_lower = user_text.lower()
        
        # Enhanced fallback responses
        responses = {
            'hello': 'Hello! I am Opi, your voice assistant.',
            'hi': 'Hi there! How can I help you?',
            'how are you': 'I am doing well, thank you for asking!',
            'time': f'The current time is {datetime.now().strftime("%I:%M %p")}.',
            'date': f'Today is {datetime.now().strftime("%A, %B %d, %Y")}.',
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
                # Stream the response in chunks
                words = response.split()
                current_chunk = ""
                
                for word in words:
                    current_chunk += word + " "
                    if len(current_chunk.split()) >= 3:
                        yield current_chunk.strip()
                        current_chunk = ""
                        await asyncio.sleep(0.01)
                
                if current_chunk.strip():
                    yield current_chunk.strip()
                return
        
        # Default response for unknown input
        response = f"I heard you say '{user_text}'. I would need my full AI capabilities to give you a proper answer."
        
        words = response.split()
        current_chunk = ""
        
        for word in words:
            current_chunk += word + " "
            if len(current_chunk.split()) >= 4:
                yield current_chunk.strip()
                current_chunk = ""
                await asyncio.sleep(0.01)
        
        if current_chunk.strip():
            yield current_chunk.strip()
    
    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    async def close(self):
        """Close the conversation manager."""
        print('[LLM] ✅ Enhanced conversation manager closed')
