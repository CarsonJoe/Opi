# llm/conversation_manager.py
"""
Low Latency Streaming Conversation Manager for Opi Voice Assistant
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
    """Enhanced streamer that yields natural phrases for ultra-low latency TTS."""
    
    def __init__(self):
        self.buffer = ""
        self.word_count = 0  # Track words processed
        
        # Natural phrase boundaries (prioritized by naturalness)
        self.strong_boundaries = re.compile(r'[.!?]+\s*')  # Sentence endings
        self.medium_boundaries = re.compile(r'[,;:]\s+')   # Clause separators  
        self.weak_boundaries = re.compile(r'(?:\s+(?:and|or|but|so|then|however|also|therefore|meanwhile|furthermore|moreover|additionally|consequently|thus|actually|basically|essentially|well|now|okay|alright)\s+)', re.IGNORECASE)
        
        # Conjunction/transition words that are good natural break points
        self.natural_breaks = re.compile(r'\b(?:and|or|but|so|then|however|also|therefore|meanwhile|furthermore|moreover|additionally|consequently|thus|actually|basically|essentially|well|now|okay|alright)\b', re.IGNORECASE)
        
        # Length thresholds - more conservative for first part
        self.early_phrase_max_words = 8    # First few phrases shorter for speed
        self.normal_phrase_max_words = 12  # Later phrases can be longer
        self.min_phrase_words = 3          # Minimum meaningful phrase
        self.speed_critical_word_limit = 15  # Be very aggressive only for first 15 words
        
        # Timing for aggressive streaming (only early on)
        self.last_chunk_time = time.time()
        self.max_wait_time_early = 0.2     # Very fast for first words
        self.max_wait_time_normal = 0.4    # More patient later
        
        # Track if we're still in the speed-critical phase
        self.in_speed_critical_phase = True
        
    def add_chunk(self, chunk: str) -> List[str]:
        """Add chunk and return phrases with improved natural breaking."""
        if not chunk:
            return []
            
        self.buffer += chunk
        phrases = []
        current_time = time.time()
        
        # Update speed-critical phase status
        current_word_count = len(self.buffer.split())
        if current_word_count > self.speed_critical_word_limit:
            self.in_speed_critical_phase = False
        
        # Strategy 1: Look for strong natural boundaries (sentences)
        strong_matches = list(self.strong_boundaries.finditer(self.buffer))
        for match in strong_matches:
            end_pos = match.end()
            potential_phrase = self.buffer[:end_pos].strip()
            word_count = len(potential_phrase.split())
            
            if word_count >= self.min_phrase_words:
                phrases.append(potential_phrase)
                self.buffer = self.buffer[end_pos:].lstrip()
                self.word_count += word_count
                self.last_chunk_time = current_time
                continue
        
        # Strategy 2: Look for medium boundaries (commas, colons, semicolons)
        if not phrases:
            medium_matches = list(self.medium_boundaries.finditer(self.buffer))
            for match in medium_matches:
                end_pos = match.end()
                potential_phrase = self.buffer[:end_pos].strip()
                word_count = len(potential_phrase.split())
                
                # Be more selective with comma breaks
                if (word_count >= self.min_phrase_words and 
                    word_count <= (self.early_phrase_max_words if self.in_speed_critical_phase else self.normal_phrase_max_words)):
                    phrases.append(potential_phrase)
                    self.buffer = self.buffer[end_pos:].lstrip()
                    self.word_count += word_count
                    self.last_chunk_time = current_time
                    break
        
        # Strategy 3: Look for weak boundaries (conjunctions/transitions) - only if needed
        if not phrases and self.in_speed_critical_phase:
            weak_matches = list(self.weak_boundaries.finditer(self.buffer))
            for match in weak_matches:
                # Break before the conjuncton, not after
                break_pos = match.start()
                if break_pos > 0:
                    potential_phrase = self.buffer[:break_pos].strip()
                    word_count = len(potential_phrase.split())
                    
                    if word_count >= self.min_phrase_words:
                        phrases.append(potential_phrase)
                        self.buffer = self.buffer[break_pos:].lstrip()
                        self.word_count += word_count
                        self.last_chunk_time = current_time
                        break
        
        # Strategy 4: Length-based breaking (more conservative)
        if not phrases:
            current_words = self.buffer.split()
            max_words = self.early_phrase_max_words if self.in_speed_critical_phase else self.normal_phrase_max_words
            
            if len(current_words) > max_words:
                # Try to find a good break point within the max length
                break_pos = self._find_best_break_point(current_words, max_words)
                if break_pos > self.min_phrase_words:
                    phrase_words = current_words[:break_pos]
                    phrase = ' '.join(phrase_words)
                    phrases.append(phrase)
                    
                    remaining_words = current_words[break_pos:]
                    self.buffer = ' '.join(remaining_words)
                    self.word_count += len(phrase_words)
                    self.last_chunk_time = current_time
        
        # Strategy 5: Timeout-based chunking (only in speed-critical phase)
        if not phrases and self.in_speed_critical_phase:
            time_since_last = current_time - self.last_chunk_time
            max_wait = self.max_wait_time_early if self.in_speed_critical_phase else self.max_wait_time_normal
            
            if time_since_last > max_wait:
                current_words = self.buffer.split()
                if len(current_words) >= self.min_phrase_words:
                    # Find the best natural break point available
                    break_pos = self._find_emergency_break_point(current_words)
                    if break_pos >= self.min_phrase_words:
                        phrase_words = current_words[:break_pos]
                        phrase = ' '.join(phrase_words)
                        phrases.append(phrase)
                        
                        remaining_words = current_words[break_pos:]
                        self.buffer = ' '.join(remaining_words)
                        self.word_count += len(phrase_words)
                        self.last_chunk_time = current_time
        
        return phrases
    
    def _find_best_break_point(self, words, max_words):
        """Find the best natural break point within max_words limit."""
        # Look for natural break words within the limit
        for i in range(max_words - 1, self.min_phrase_words - 1, -1):
            if i < len(words):
                word = words[i].lower().strip('.,!?;:')
                if word in ['and', 'or', 'but', 'so', 'then', 'however', 'also', 'therefore']:
                    return i  # Break before the conjunction
                elif i > 0 and words[i-1].rstrip('.,!?;:').lower() in ['well', 'now', 'okay', 'alright']:
                    return i  # Break after transition words
        
        # Look for words that end with punctuation
        for i in range(max_words - 1, self.min_phrase_words - 1, -1):
            if i < len(words) and any(words[i].endswith(p) for p in '.,;:'):
                return i + 1
        
        # Default to about 2/3 of max_words
        return min(max_words * 2 // 3, len(words))
    
    def _find_emergency_break_point(self, words):
        """Find an emergency break point when timeout forces a break."""
        # Try to break at any natural word boundary
        natural_break_words = {'and', 'or', 'but', 'so', 'then', 'well', 'now', 'also'}
        
        # Look backwards from a reasonable point
        max_emergency_length = min(len(words), 6)
        for i in range(max_emergency_length, self.min_phrase_words - 1, -1):
            if words[i-1].lower().strip('.,!?;:') in natural_break_words:
                return i
        
        # If no natural break, just break at min length + 1
        return min(self.min_phrase_words + 1, len(words))
    
    def flush(self) -> Optional[str]:
        """Return any remaining content as final phrase."""
        if self.buffer.strip():
            word_count = len(self.buffer.split())
            if word_count >= 1:  # Even single words at the end are worth saying
                final = self.buffer.strip()
                self.buffer = ""
                return final
        return None

class UltraLowLatencyTTSPipeline:
    """TTS pipeline optimized for phrase-level streaming - FIXED VERSION."""
    
    def __init__(self, tts_worker, audio_worker):
        self.tts_worker = tts_worker
        self.audio_worker = audio_worker
        self.phrase_queue = queue.Queue(maxsize=50)
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
        
        # CRITICAL FIX: Track pipeline completion state
        self.tts_threads_finished = 0
        self.tts_threads_total = 0
        self.tts_finish_lock = threading.Lock()
        
    def start_pipeline(self, debug=False):
        """Start the ultra-low latency pipeline."""
        self.stop_event.clear()
        self.first_audio_time = None
        self.phrases_processed = 0
        self.next_audio_sequence = 0
        self.audio_sequence.clear()
        self.debug = debug
        
        # Reset completion tracking
        self.tts_threads_finished = 0
        self.tts_threads_total = 2  # We'll start 2 TTS threads
        
        # Start multiple TTS threads for parallel processing
        for i in range(self.tts_threads_total):
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
        """Complete processing and cleanup - FIXED VERSION."""
        if self.debug:
            cprint("[Pipeline] 🏁 Starting pipeline shutdown...", "yellow")
        
        # Step 1: Signal end of phrases to all TTS threads
        for i in range(self.tts_threads_total):
            self.phrase_queue.put(None)
            if self.debug:
                cprint(f"[Pipeline] Sent shutdown signal to TTS thread {i}", "cyan")
        
        # Step 2: Wait for all TTS threads to complete phrase processing
        tts_timeout = 10.0  # Give plenty of time for TTS processing
        for thread in self.processing_threads[:-1]:  # All except audio thread
            thread.join(timeout=tts_timeout)
            if thread.is_alive():
                cprint(f"[Pipeline] ⚠️ TTS thread {thread.name} didn't finish in time", "yellow")
        
        # Step 3: Wait a bit more for any remaining audio to queue up
        time.sleep(0.2)
        
        # Step 4: Now signal audio thread to finish (but only after TTS is done)
        self.audio_queue.put(None)
        if self.debug:
            cprint("[Pipeline] Sent shutdown signa to audio thread", "cyan")
        
        # Step 5: Wait for audio thread to finish playing everything
        audio_timeout = 8.0
        audio_thread = self.processing_threads[-1]  # Last thread is audio
        audio_thread.join(timeout=audio_timeout)
        
        if audio_thread.is_alive():
            cprint("[Pipeline] ⚠️ Audio thread didn't finish in time", "yellow")
        
        if self.debug:
            cprint(f"[Pipeline] ✅ Processed {self.phrases_processed}/{self.total_phrases} phrases", "green")
        
        return self.first_audio_time
    
    def _tts_processing_loop(self):
        """Ultra-fast TTS processing loop - FIXED VERSION."""
        thread_name = threading.current_thread().name
        debug = getattr(self, 'debug', False)
        
        try:
            while not self.stop_event.is_set():
                try:
                    phrase_data = self.phrase_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if phrase_data is None:  # End signal
                    if debug:
                        cprint(f"[{thread_name}] Received shutdown signal", "yellow")
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
            # CRITICAL FIX: Track when TTS threads finish
            with self.tts_finish_lock:
                self.tts_threads_finished += 1
                if debug:
                    cprint(f"[{thread_name}] TTS thread finished ({self.tts_threads_finished}/{self.tts_threads_total})", "cyan")
                
                # Only signal audio end when ALL TTS threads are done
                if self.tts_threads_finished >= self.tts_threads_total:
                    if debug:
                        cprint("[Pipeline] All TTS threads finished - checking for remaining audio", "green")
                    
                    # Queue any remaining audio that was processed
                    with self.sequence_lock:
                        self._check_and_queue_audio()
                    
                    # DON'T signal audio end here - let finish_pipeline() do it
                    # This prevents the race condition!
    
    def _check_and_queue_audio(self):
        """Check if we have the next audio sequence ready and queue it."""
        queued_count = 0
        while self.next_audio_sequence in self.audio_sequence:
            audio_data = self.audio_sequence.pop(self.next_audio_sequence)
            self.audio_queue.put(audio_data)
            self.next_audio_sequence += 1
            queued_count += 1
        
        if queued_count > 0 and self.debug:
            cprint(f"[Pipeline] Queued {queued_count} audio chunks", "green")
    
    def _synthesize_phrase(self, phrase: str):
        """Synthesize a phrase with optimizations for speed."""
        try:
            # Use existing TTS worker but with speed optimizations
            return self.tts_worker._synthesize_text(phrase, 1.1)  # Slightly faster speech
        except Exception as e:
            cprint(f"[Pipeline] Phrase synthesis error: {e}", "red")
            return None
    
    def _audio_playback_loop(self):
        """Audio playback loop with sequence management - FIXED VERSION."""
        debug = getattr(self, 'debug', False)
        chunks_played = 0
        
        try:
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if audio_chunk is None:  # End signal
                    if debug:
                        cprint(f"[Pipeline] Audio thread received end signal after playing {chunks_played} chunks", "yellow")
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
                chunks_played += 1
                
                if debug:
                    cprint(f"[Pipeline] 🔊 Played audio chunk {chunks_played}", "blue")
                
        except Exception as e:
            cprint(f"[Pipeline] ❌ Audio playback error: {e}", "red")
        finally:
            if debug:
                cprint(f"[Pipeline] Audio thread finished after playing {chunks_played} chunks", "green")

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
        
        print("\n" + "="*80)
        print("🔍 FULL GEMINI API REQUEST:")
        print("="*80)
        print("SYSTEM PROMPT:")
        print(self._get_enhanced_system_prompt())
        print("\nUSER PROMPT:")
        print(prompt)
        print("="*80 + "\n")
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
