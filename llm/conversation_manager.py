# llm/conversation_manager.py
"""
Ultra-Low Latency Streaming Conversation Manager for Opi Voice Assistant
with phrase-level streaming for dramatically reduced first audio latency
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


class PhraseStreamer:
    """Enhanced streamer that yields phrases for ultra-low latency TTS."""
    
    def __init__(self):
        self.buffer = ""
        
        # Phrase boundaries (more aggressive than sentences)
        self.phrase_endings = re.compile(r'[.!?]+\s*|[,;]\s*|(?:\s+(?:and|or|but|so|then|however|also|therefore|meanwhile|furthermore)\s+)')
        
        # Sentence endings for final cleanup
        self.sentence_endings = re.compile(r'[.!?]+\s*')
        
        # Length thresholds
        self.min_phrase_length = 4   # Much shorter than sentences
        self.max_phrase_length = 40  # Prevent overly long phrases
        self.min_sentence_length = 8 # For complete sentences
        
        # Timing for aggressive streaming
        self.last_chunk_time = time.time()
        self.max_wait_time = 0.3  # Max 300ms before forcing output
        
        # Common phrase starters that indicate good break points
        self.natural_breaks = re.compile(r'\b(?:well|so|now|then|also|however|but|and|or|actually|basically|essentially|meanwhile|furthermore|additionally|moreover|therefore|thus|consequently)\b', re.IGNORECASE)
        
    def add_chunk(self, chunk: str) -> List[str]:
        """Add chunk and return phrases immediately when possible."""
        if not chunk:
            return []
            
        self.buffer += chunk
        phrases = []
        current_time = time.time()
        
        # Strategy 1: Look for natural phrase boundaries
        phrase_matches = list(self.phrase_endings.finditer(self.buffer))
        
        for match in phrase_matches:
            end_pos = match.end()
            potential_phrase = self.buffer[:end_pos].strip()
            
            # Yield if it's long enough and meaningful
            if len(potential_phrase) >= self.min_phrase_length:
                phrases.append(potential_phrase)
                self.buffer = self.buffer[end_pos:].lstrip()
                self.last_chunk_time = current_time
        
        # Strategy 2: Aggressive timeout-based chunking
        if not phrases and self.buffer.strip():
            time_since_last = current_time - self.last_chunk_time
            
            if time_since_last > self.max_wait_time:
                # Force output if we've waited too long
                if len(self.buffer.strip()) >= self.min_phrase_length:
                    # Try to break at natural points
                    natural_break_pos = self._find_natural_break()
                    if natural_break_pos > 0:
                        phrase = self.buffer[:natural_break_pos].strip()
                        phrases.append(phrase)
                        self.buffer = self.buffer[natural_break_pos:].lstrip()
                        self.last_chunk_time = current_time
        
        # Strategy 3: Prevent overly long phrases
        if not phrases and len(self.buffer.strip()) > self.max_phrase_length:
            break_pos = self._find_natural_break() or self.max_phrase_length
            phrase = self.buffer[:break_pos].strip()
            if len(phrase) >= self.min_phrase_length:
                phrases.append(phrase)
                self.buffer = self.buffer[break_pos:].lstrip()
                self.last_chunk_time = current_time
        
        return phrases
    
    def _find_natural_break(self) -> int:
        """Find the best natural breaking point in current buffer."""
        if not self.buffer:
            return 0
        
        # Look for natural break words
        natural_matches = list(self.natural_breaks.finditer(self.buffer))
        if natural_matches:
            # Take the last natural break that's not too close to the end
            for match in reversed(natural_matches):
                if match.start() >= self.min_phrase_length:
                    return match.start()
        
        # Fall back to word boundaries
        words = self.buffer.split()
        if len(words) >= 3:  # At least 3 words for a phrase
            # Take about 2/3 of available words
            break_word = min(len(words) - 1, max(2, len(words) * 2 // 3))
            break_pos = len(' '.join(words[:break_word])) + 1
            return break_pos
        
        return 0
    
    def flush(self) -> Optional[str]:
        """Return any remaining content as final phrase."""
        if self.buffer.strip() and len(self.buffer.strip()) >= self.min_phrase_length:
            final = self.buffer.strip()
            self.buffer = ""
            return final
        return None


class UltraLowLatencyTTSPipeline:
    """TTS pipeline optimized for phrase-level streaming."""
    
    def __init__(self, tts_worker, audio_worker):
        self.tts_worker = tts_worker
        self.audio_worker = audio_worker
        self.phrase_queue = queue.Queue(maxsize=50)  # More capacity for phrases
        self.audio_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.first_audio_time = None
        self.processing_threads = []
        self.debug = False
        
        # Performance tracking
        self.phrases_processed = 0
        self.total_phrases = 0
        
        # Audio sequencing to maintain order
        self.audio_sequence = {}
        self.next_audio_sequence = 0
        self.sequence_lock = threading.Lock()
        
    def start_pipeline(self, debug=False):
        """Start the ultra-low latency pipeline."""
        self.stop_event.clear()
        self.first_audio_time = None
        self.phrases_processed = 0
        self.next_audio_sequence = 0
        self.audio_sequence.clear()
        self.debug = debug
        
        # Start multiple TTS threads for parallel processing
        num_tts_threads = 2  # Process 2 phrases in parallel
        
        for i in range(num_tts_threads):
            tts_thread = threading.Thread(
                target=self._tts_processing_loop,
                name=f"UltraFastTTS-{i}",
                daemon=True
            )
            tts_thread.start()
            self.processing_threads.append(tts_thread)
        
        # Start audio playback thread
        audio_thread = threading.Thread(
            target=self._audio_playback_loop,
            name="UltraFastAudio",
            daemon=True
        )
        audio_thread.start()
        self.processing_threads.append(audio_thread)
        
        if debug:
            cprint("[Pipeline] 🚀 Ultra-low latency phrase pipeline started", "cyan", attrs=['bold'])
    
    def add_phrase(self, phrase: str, debug=False):
        """Add a phrase for immediate TTS processing."""
        if not phrase.strip():
            return
            
        try:
            phrase_data = {
                'text': phrase.strip(),
                'sequence': self.total_phrases
            }
            self.phrase_queue.put(phrase_data, timeout=0.05)
            self.total_phrases += 1
            cprint(f"{phrase.strip()}", "blue", attrs=['bold'])

            if debug:
                cprint(f"[Pipeline] ⚡ Queued phrase: \"{phrase[:25]}...\"", "cyan")
        except queue.Full:
            cprint("[Pipeline] ⚠️ Phrase queue full, dropping phrase", "yellow")
    
    def finish_pipeline(self):
        """Complete processing and cleanup."""
        # Signal end of phrases to all TTS threads
        for _ in range(len(self.processing_threads) - 1):  # All except audio thread
            self.phrase_queue.put(None)
        
        # Wait for processing to complete
        for thread in self.processing_threads:
            thread.join(timeout=8.0)
        
        if hasattr(self, 'debug') and self.debug:
            cprint(f"[Pipeline] ✅ Processed {self.phrases_processed}/{self.total_phrases} phrases", "green")
        
        return self.first_audio_time
    
    def _tts_processing_loop(self):
        """Ultra-fast TTS processing loop."""
        thread_name = threading.current_thread().name
        debug = getattr(self, 'debug', False)
        
        try:
            while not self.stop_event.is_set():
                try:
                    phrase_data = self.phrase_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if phrase_data is None:  # End signal
                    break
                
                phrase_text = phrase_data['text']
                sequence_num = phrase_data['sequence']
                
                # Process through TTS immediately
                start_time = time.time()
                audio_data = self._synthesize_phrase(phrase_text)
                process_time = time.time() - start_time
                
                if audio_data is not None:
                    # Store with sequence number to maintain order
                    with self.sequence_lock:
                        self.audio_sequence[sequence_num] = audio_data
                        self._check_and_queue_audio()
                    
                    self.phrases_processed += 1
                    if debug:
                        cprint(f"[{thread_name}] ⚡ TTS: {process_time:.3f}s for \"{phrase_text[:20]}...\"", "green")
                
        except Exception as e:
            cprint(f"[Pipeline] ❌ TTS processing error in {thread_name}: {e}", "red")
        finally:
            # Signal end of audio processing
            self.audio_queue.put(None)
    
    def _check_and_queue_audio(self):
        """Check if we have the next audio sequence ready and queue it."""
        while self.next_audio_sequence in self.audio_sequence:
            audio_data = self.audio_sequence.pop(self.next_audio_sequence)
            self.audio_queue.put(audio_data)
            self.next_audio_sequence += 1
    
    def _synthesize_phrase(self, phrase: str):
        """Synthesize a phrase with optimizations for speed."""
        try:
            # Use existing TTS worker but with speed optimizations
            return self.tts_worker._synthesize_text(phrase, 1.1)  # Slightly faster speech
        except Exception as e:
            cprint(f"[Pipeline] Phrase synthesis error: {e}", "red")
            return None
    
    def _audio_playback_loop(self):
        """Audio playback loop with sequence management."""
        debug = getattr(self, 'debug', False)
        
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
                    if debug:
                        cprint("[Pipeline] 🔊 First audio playing!", "blue", attrs=['bold'])
                
                # Play audio using existing audio worker
                temp_queue = queue.Queue()
                temp_queue.put(audio_chunk)
                temp_queue.put(None)  # End marker
                
                # Use existing audio worker
                temp_stop = threading.Event()
                timings = {}
                
                self.audio_worker.process_playback(temp_queue, temp_stop, timings)
                
        except Exception as e:
            cprint(f"[Pipeline] ❌ Audio playback error: {e}", "red")


class ConversationManager:
    """Enhanced conversation manager with ultra-low latency phrase streaming - DROP-IN REPLACEMENT."""
    
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
        print('[LLM] Initializing enhanced conversation manager...')
        
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
            
            print(f'[LLM] ✅ Enhanced Gemini ready with phrase streaming: {model_name}')
            
        except Exception as e:
            print(f'[LLM] ❌ Failed to initialize Gemini: {e}')
            self.model = None
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt optimized for phrase streaming."""
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
- IMPORTANT: Structure responses in short, natural phrases for immediate speech output
- Start responding immediately - avoid lengthy introductions
- Keep responses conversational and concise{tools_info}

Response Guidelines for Ultra-Low Latency Voice:
- Use short, punchy phrases that work well when spoken immediately
- Avoid complex sentence structures that delay the start of speech
- Begin with the most important point first
- Each phrase should be meaningful on its own
- Keep total responses under 80 words unless specifically asked for detail
- Use natural speech patterns with appropriate pauses"""
        
        return enhanced_prompt
    
    # MAIN STREAMING METHOD - now with phrase-level streaming
    async def process_user_input_streaming(self, user_text: str, speech_end_time: float, 
                                         tts_worker, audio_worker, debug=False) -> float:
        """Process user input with ultra-low latency phrase streaming - ENHANCED MAIN METHOD."""
        
        # Create ultra-low latency pipeline
        pipeline = UltraLowLatencyTTSPipeline(tts_worker, audio_worker)
        phrase_streamer = PhraseStreamer()
        
        # Start pipeline
        pipeline.start_pipeline(debug)
        
        try:
            if debug:
                cprint(f"[LLM] 🧠 Processing with phrase streaming: \"{user_text}\"", "blue")
            
            # Stream LLM response with phrase-level chunking
            response_chunks = []
            async for chunk in self._stream_llm_response(user_text):
                if chunk and chunk.strip():
                    response_chunks.append(chunk)
                    
                    # Process through phrase streamer (much more aggressive than sentences)
                    phrases = phrase_streamer.add_chunk(chunk)
                    
                    # Send phrases to TTS pipeline immediately
                    for phrase in phrases:
                        pipeline.add_phrase(phrase, debug)
            
            # Handle any remaining content
            final_phrase = phrase_streamer.flush()
            if final_phrase:
                pipeline.add_phrase(final_phrase, debug)
            
            # Complete pipeline and get first audio time
            first_audio_time = pipeline.finish_pipeline()
            
            # Log results
            if response_chunks:
                full_response = "".join(response_chunks)
                if debug:
                    cprint(f"[LLM] 🗣️ Complete response: \"{full_response}\"", "green")
                
                # Add to conversation history
                self._add_to_history("user", user_text)
                self._add_to_history("assistant", full_response)
            
            return first_audio_time
            
        except Exception as e:
            cprint(f"[LLM] ❌ Phrase streaming processing error: {e}", "red")
            # Fallback to simple response
            pipeline.add_phrase("Sorry, I encountered an error processing your request.", debug)
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
        """Stream response from LLM with optimized chunking for phrases."""
        
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

Respond as Opi, the helpful voice assistant. Use short, natural phrases that can be spoken immediately. Be concise and conversational."""
        
        try:
            # Try streaming API first
            if hasattr(self.model, 'generate_content') and hasattr(genai, 'GenerationConfig'):
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.model.generate_content(prompt)
                )
                
                if response.text:
                    # Stream in smaller chunks for better phrase detection
                    words = response.text.split()
                    current_chunk = ""
                    
                    for i, word in enumerate(words):
                        current_chunk += word + " "
                        
                        # Yield chunks more frequently for phrase detection
                        if (word.endswith('.') or word.endswith('!') or word.endswith('?') or 
                            word.endswith(',') or word.endswith(';') or 
                            len(current_chunk.split()) >= 2):  # Smaller chunks
                            
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
                # Stream the response in smaller chunks for phrase detection
                words = response.split()
                current_chunk = ""
                
                for word in words:
                    current_chunk += word + " "
                    if len(current_chunk.split()) >= 2:  # Smaller chunks for phrases
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
            if len(current_chunk.split()) >= 3:  # Smaller chunks
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
        print('[LLM] ✅ Conversation manager closed')
