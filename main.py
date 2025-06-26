#!/usr/bin/env python3
"""
Opi Voice Assistant - Enhanced with LangSmith Tracing
Main application with voice + text input modes and comprehensive observability
"""

import anyio
import asyncio
import argparse
import os
import sys
import threading
import queue
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from termcolor import cprint
from dotenv import load_dotenv

# LangSmith setup - Import early for tracing
from langchain_core.tracers.langchain import wait_for_all_tracers

# Core components
from voice.speech_worker import SpeechWorker
from voice.tts_worker import TTSWorker
from voice.audio_worker import AudioWorker
from voice.text_input_worker import TextInputWorker, AsyncTextInputWorker
from llm.mcp_manager import MCPManager
from config.settings import OpiConfig
from utils.timing import TimingTracker


class LangSmithConfig:
    """LangSmith configuration manager."""
    
    def __init__(self, config_dict=None):
        # Load from config.json with environment variable fallbacks for API key only
        config_dict = config_dict or {}
        
        self.enabled = config_dict.get('enabled', True)
        self.project_name = config_dict.get('project_name', 'opi-voice-assistant')
        self.endpoint = config_dict.get('endpoint', 'https://api.smith.langchain.com')
        self.tags = config_dict.get('tags', ['voice-assistant'])
        self.metadata = config_dict.get('metadata', {})
        self.background_callbacks = config_dict.get('background_callbacks', True)
        self.session_tracking = config_dict.get('session_tracking', True)
        
        # Only API key comes from environment variable
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        
        # Enable only if we have an API key
        self.enabled = self.enabled and bool(self.api_key)
        
        # Set environment variables for automatic LangChain integration
        if self.enabled:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.api_key
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            os.environ["LANGSMITH_ENDPOINT"] = self.endpoint
            
            # Configure background callbacks
            if not os.getenv("LANGCHAIN_CALLBACKS_BACKGROUND"):
                os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = str(self.background_callbacks).lower()
    
    def print_status(self, verbose=False):
        """Print LangSmith status."""
        if self.enabled:
            cprint(f"[LangSmith] ✅ Tracing enabled", "green")
            if verbose:
                cprint(f"[LangSmith] 📊 Project: {self.project_name}", "cyan")
                cprint(f"[LangSmith] 🔗 Endpoint: {self.endpoint}", "cyan")
                cprint(f"[LangSmith] 🔑 API Key: {self.api_key[:8]}..." if self.api_key else "No API Key", "cyan")
        else:
            cprint("[LangSmith] ❌ Tracing disabled", "yellow")
            if verbose:
                cprint("           Set LANGSMITH_API_KEY in .env file to enable", "yellow")


class EnhancedOpiVoiceAssistant:
    """Enhanced Opi Voice Assistant with LangSmith integration."""

    def __init__(self, config_path: Optional[str] = None):
        # Load environment variables first
        load_dotenv()
        
        self.config = OpiConfig.load(config_path)
        self.running = False
        self.stop_event = threading.Event()

        # Verbose flag from config or environment
        self.verbose = getattr(self.config, 'verbose', False) or os.getenv('OPI_DEBUG', '0') == '1'

        # Initialize LangSmith configuration from config.json
        langsmith_config_dict = getattr(self.config, 'langsmith', {})
        if hasattr(langsmith_config_dict, '__dict__'):
            langsmith_config_dict = langsmith_config_dict.__dict__
        
        self.langsmith_config = LangSmithConfig(langsmith_config_dict)

        # Voice components
        self.speech_worker: Optional[SpeechWorker] = None
        self.tts_worker: Optional[TTSWorker] = None
        self.audio_worker: Optional[AudioWorker] = None

        # Text input components
        self.text_worker: Optional[TextInputWorker] = None
        self.async_text_worker: Optional[AsyncTextInputWorker] = None

        # LLM components
        self.mcp_manager: Optional[MCPManager] = None
        self.conversation_manager = None

        # Queues for inter-component communication
        self.speech_queue = queue.Queue(maxsize=10)
        self.text_queue = queue.Queue(maxsize=50)
        self.input_queue = queue.Queue(maxsize=100)
        self.audio_queue = queue.Queue(maxsize=100)

        # Timing and monitoring
        self.timing_tracker = TimingTracker()

        # Mode settings
        self.text_only_mode = False
        self.hybrid_mode = False

        # Enhanced metrics with LangSmith integration
        self.streaming_metrics = {
            'total_interactions': 0,
            'first_audio_times': [],
            'total_response_times': [],
            'text_inputs': 0,
            'voice_inputs': 0,
            'langsmith_traces': 0,
            'tool_calls': 0,
            'llm_calls': 0
        }

        # Session tracking for LangSmith
        import uuid
        self.session_id = str(uuid.uuid4())

    async def initialize(self, text_only_mode=False, hybrid_mode=False):
        """Initialize all components with LangSmith integration."""
        self.text_only_mode = text_only_mode
        self.hybrid_mode = hybrid_mode

        # Print LangSmith status
        self.langsmith_config.print_status(verbose=self.verbose)

        if text_only_mode:
            cprint("[Opi] Initializing in TEXT-ONLY mode with LangSmith tracing...", "cyan", attrs=['bold'])
        elif hybrid_mode:
            cprint("[Opi] Initializing in HYBRID mode (voice + text) with LangSmith...", "cyan", attrs=['bold'])
        else:
            if self.verbose:
                cprint("[Opi] Initializing voice assistant with LangSmith (VERBOSE MODE)...", "cyan", attrs=['bold'])
            else:
                cprint("[Opi] Initializing voice assistant with LangSmith...", "cyan", attrs=['bold'])

        # Initialize voice components (skip in text-only mode)
        if not text_only_mode:
            await self._init_voice_components()

        # Initialize text components
        await self._init_text_components()

        # Initialize LLM components with LangSmith
        await self._init_llm_components()

        # Test LLM processing in verbose mode only
        if self.verbose and not text_only_mode:
            await self._test_streaming_processing()

        cprint("[Opi] ✅ All components initialized with LangSmith tracing", "green", attrs=['bold'])

    async def _init_text_components(self):
        """Initialize text input components."""
        cprint("[Opi] Loading text input components...", "yellow")

        self.text_worker = TextInputWorker(verbose=self.verbose)

        if self.hybrid_mode:
            self.async_text_worker = AsyncTextInputWorker(verbose=self.verbose)

        cprint("[Opi] ✅ Text input components ready", "green")

    async def _init_voice_components(self):
        """Initialize speech recognition, TTS, and audio components."""
        cprint("[Opi] Loading voice components...", "yellow")

        self.speech_worker = SpeechWorker(
            model_size=self.config.voice.whisper_model,
            compute_type=self.config.voice.whisper_compute_type,
            verbose=self.verbose
        )

        self.tts_worker = TTSWorker(
            model_path=self.config.voice.tts_model_path,
            config_path=self.config.voice.tts_config_path,
            speaker_id=self.config.voice.speaker_id,
            speech_speed=self.config.voice.speech_speed,
            verbose=self.verbose
        )
        await self.tts_worker.initialize()

        self.audio_worker = AudioWorker(
            device_index=self.config.voice.audio_device,
            sample_rate=self.config.voice.sample_rate,
            chunk_pause_ms=self.config.voice.chunk_pause_ms,
            verbose=self.verbose
        )

        cprint("[Opi] ✅ Voice components ready", "green")

    async def _init_llm_components(self):
        """Initialize LLM and MCP components with LangSmith integration."""
        cprint("[Opi] Initializing LLM components with LangSmith...", "yellow")

        # Validate LLM configuration
        if not self.config.llm.api_key:
            cprint("[Opi] ❌ No LLM API key configured!", "red")
            cprint("       Set GOOGLE_API_KEY environment variable", "yellow")
            cprint("       The assistant will use fallback responses only", "yellow")
        else:
            if self.verbose:
                cprint(f"[Opi] LLM API key configured: {self.config.llm.api_key[:10]}...", "green")

        # Initialize MCP manager
        self.mcp_manager = MCPManager(
            config=self.config.mcp.servers,
        )
        await self.mcp_manager.connect_all()

        # Import and initialize the enhanced conversation manager
        try:
            from llm.conversation_manager import ConversationManager
            
            # Create enhanced conversation manager with LangSmith metadata
            self.conversation_manager = ConversationManager(
                llm_config=self.config.llm,
                system_prompt=self.config.prompts.system_prompt,
                mcp_manager=self.mcp_manager,
                db_path=self.config.storage.conversation_db
            )
            
            # Add session metadata to the conversation manager
            self.conversation_manager.session_id = self.session_id
            self.conversation_manager.langsmith_enabled = self.langsmith_config.enabled
            
            await self.conversation_manager.initialize()
            
            if self.langsmith_config.enabled:
                cprint("[Opi] ✅ Enhanced conversation manager with LangSmith ready", "green")
            else:
                cprint("[Opi] ✅ Standard conversation manager ready", "green")
                
        except Exception as e:
            cprint(f"[Opi] ❌ Failed to initialize enhanced conversation manager: {e}", "red")
            raise e

        tool_count = len(self.mcp_manager.get_tools())
        server_status = self.mcp_manager.get_server_status()
        connected_servers = server_status['connected_servers']
        total_servers = server_status['total_servers']

        cprint(f"[Opi] ✅ LLM ready with {connected_servers}/{total_servers} MCP servers", "green")
        cprint(f"[Opi] 🛠️ Available tools: {tool_count}", "green")

    async def _test_streaming_processing(self):
        """Test streaming processing with LangSmith tracing."""
        cprint("[Opi] Testing streaming pipeline with LangSmith...", "yellow")
        try:
            test_start = time.time()

            # Test with LangSmith tracing context
            first_audio_time = await self._process_with_enhanced_tracing(
                "hello", test_start, source="test"
            )

            test_duration = time.time() - test_start

            if first_audio_time:
                first_audio_latency = first_audio_time - test_start
                cprint(f"[Opi] ✅ LangSmith streaming test successful!", "green")
                cprint(f"[Opi] ⚡ First audio latency: {first_audio_latency:.3f}s", "green", attrs=['bold'])
                cprint(f"[Opi] 📊 Total test time: {test_duration:.3f}s", "white")
                if self.langsmith_config.enabled:
                    cprint(f"[Opi] 🔍 Check LangSmith project: {self.langsmith_config.project_name}", "cyan")
            else:
                cprint("[Opi] ⚠️ Streaming test completed but no first audio time recorded", "yellow")

        except Exception as e:
            cprint(f"[Opi] ❌ LangSmith streaming test failed: {e}", "red")
            if self.verbose:
                import traceback
                traceback.print_exc()

    async def start_listening(self):
        """Start the main interaction loop with LangSmith tracing."""
        if not self.conversation_manager:
            raise RuntimeError("Components not initialized. Call initialize() first.")

        self.running = True

        if self.text_only_mode:
            cprint("[Opi] 💬 Starting TEXT-ONLY mode with LangSmith... Type to chat!", "blue", attrs=['bold'])
            await self._text_only_loop()
        elif self.hybrid_mode:
            cprint("[Opi] 🎤💬 Starting HYBRID mode with LangSmith... Voice + Text!", "blue", attrs=['bold'])
            await self._hybrid_loop()
        else:
            if self.verbose:
                cprint("[Opi] 🚀 Starting voice assistant with LangSmith (VERBOSE MODE)... Listening!", "blue", attrs=['bold'])
            else:
                cprint("[Opi] 🚀 Starting voice assistant with LangSmith... Listening!", "blue", attrs=['bold'])
            await self._voice_only_loop()

    async def _text_only_loop(self):
        """Main loop for text-only mode."""
        text_thread = threading.Thread(
            target=self.text_worker.process_text_input,
            args=(self.input_queue, self.stop_event, {}, True),
            name="TextOnlyWorker",
            daemon=True
        )
        text_thread.start()

        try:
            await self._unified_processing_loop()
        except KeyboardInterrupt:
            cprint("\n[Opi] Shutting down...", "yellow")
        finally:
            await self.shutdown()

    async def _hybrid_loop(self):
        """Main loop for hybrid voice + text mode."""
        if self.speech_worker:
            speech_thread = threading.Thread(
                target=self._run_speech_worker,
                name="SpeechWorker",
                daemon=True
            )
            speech_thread.start()

        if self.async_text_worker:
            self.async_text_worker.start_async_input(self.input_queue, self.stop_event)

        try:
            await self._unified_processing_loop()
        except KeyboardInterrupt:
            cprint("\n[Opi] Shutting down...", "yellow")
        finally:
            await self.shutdown()

    async def _voice_only_loop(self):
        """Main loop for voice-only mode."""
        speech_thread = threading.Thread(
            target=self._run_speech_worker,
            name="SpeechWorker",
            daemon=True
        )
        speech_thread.start()

        try:
            await self._streaming_main_loop()
        except KeyboardInterrupt:
            cprint("\n[Opi] Shutting down...", "yellow")
        finally:
            await self.shutdown()

    async def _unified_processing_loop(self):
        """Unified processing loop with LangSmith tracing."""
        interaction_count = 0

        if self.text_only_mode:
            cprint("[Opi] 💬 Ready with LangSmith! Type your messages...", "green", attrs=['bold'])
        else:
            cprint("[Opi] 🎤💬 Ready with LangSmith! Listening for speech and text...", "green", attrs=['bold'])

        while self.running and not self.stop_event.is_set():
            try:
                input_data = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_unified_input, 1.0
                )

                if not input_data:
                    continue

                user_text = input_data['text'].strip()
                input_end_time = input_data['user_speech_end_time']
                input_source = input_data.get('source', 'voice')

                if self.verbose:
                    cprint(f"[DEBUG] Received {input_source} input: {input_data}", "cyan")

                if not user_text:
                    if self.verbose:
                        cprint("[DEBUG] Empty user text, skipping", "yellow")
                    continue

                # Update metrics
                interaction_count += 1
                self.streaming_metrics['total_interactions'] = interaction_count
                if input_source == 'text':
                    self.streaming_metrics['text_inputs'] += 1
                else:
                    self.streaming_metrics['voice_inputs'] += 1

                # Show input with source indicator
                source_icon = "💬" if input_source == 'text' else "👂"
                cprint(f"[Opi] {source_icon} {input_source.title()}: \"{user_text}\"", "white")

                # Check for exit commands
                if self._is_exit_command(user_text):
                    cprint("[Opi] 👋 Goodbye command detected", "yellow")
                    await self._handle_goodbye()
                    break

                # Process with enhanced LangSmith tracing
                await self._process_with_enhanced_tracing(user_text, input_end_time, input_source)

            except Exception as e:
                cprint(f"[Opi] ❌ Processing loop error: {e}", "red")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.1)

    async def _process_with_enhanced_tracing(self, user_text: str, input_end_time: float, source: str = "voice"):
        """Process user input with enhanced LangSmith tracing and metadata."""
        interaction_start = time.time()

        try:
            if self.verbose:
                cprint(f"[DEBUG] Starting enhanced processing for: '{user_text}'", "cyan")

            # Create enhanced metadata for LangSmith
            interaction_metadata = {
                "session_id": self.session_id,
                "interaction_count": self.streaming_metrics['total_interactions'],
                "input_source": source,
                "text_length": len(user_text),
                "has_audio_output": bool(self.tts_worker and self.audio_worker and not self.text_only_mode),
                "conversation_turn": self.streaming_metrics['total_interactions'],
                "timestamp": interaction_start
            }

            # Use conversation manager with metadata
            if self.text_only_mode:
                first_audio_time = await self._process_text_only_response_enhanced(
                    user_text, input_end_time, interaction_metadata
                )
            else:
                # Enhanced streaming with LangSmith
                first_audio_time = await self.conversation_manager.process_user_input_streaming(
                    user_text,
                    input_end_time,
                    self.tts_worker,
                    self.audio_worker,
                    debug=self.verbose
                )

            # Update LangSmith metrics
            if self.langsmith_config.enabled:
                self.streaming_metrics['langsmith_traces'] += 1

            # Calculate performance metrics
            total_time = time.time() - interaction_start

            if first_audio_time:
                first_audio_latency = first_audio_time - input_end_time
                self.streaming_metrics['first_audio_times'].append(first_audio_latency)
                self.timing_tracker.add_timing('first_audio_latency', first_audio_latency)

                if self.verbose:
                    cprint(f"[Opi] ⚡ FIRST AUDIO: {first_audio_latency:.3f}s", "green", attrs=['bold'])

            self.streaming_metrics['total_response_times'].append(total_time)
            self.timing_tracker.add_timing('streaming_response', total_time)

            if self.verbose:
                cprint(f"[Opi] ✅ Complete enhanced response in {total_time:.3f}s", "green")
                if self.langsmith_config.enabled:
                    cprint(f"[Opi] 🔍 Trace logged to LangSmith project: {self.langsmith_config.project_name}", "cyan")

        except Exception as e:
            cprint(f"[Opi] ❌ Enhanced processing error: {e}", "red")
            if self.verbose:
                import traceback
                traceback.print_exc()

            # Fallback response
            if self.text_only_mode:
                cprint("[Opi] 🤖 Sorry, I encountered an error processing your request.", "red")
            else:
                await self._stream_simple_response("Sorry, I encountered an error processing your request.")

    async def _process_text_only_response_enhanced(self, user_text: str, input_end_time: float, metadata: Dict[str, Any]):
        """Process text-only response with LangSmith tracing."""
        try:
            # Use conversation manager's streaming method even for text-only
            response_chunks = []
            async for chunk in self.conversation_manager._stream_llm_response(user_text):
                if chunk and chunk.strip():
                    response_chunks.append(chunk)
                    print(chunk, end='', flush=True)

            if response_chunks:
                full_response = "".join(response_chunks)
                print()  # New line after response
                
                # Update metrics for LangSmith
                self.streaming_metrics['llm_calls'] += 1

            return time.time()

        except Exception as e:
            cprint(f"[Opi] ❌ Enhanced text processing error: {e}", "red")
            cprint("[Opi] 🤖 Sorry, I encountered an error processing your request.", "red")
            return time.time()

    async def _streaming_main_loop(self):
        """Original main loop with enhanced LangSmith tracing."""
        interaction_count = 0

        cprint("[Opi] 🎤 Ready with LangSmith! Listening for speech...", "green", attrs=['bold'])

        while self.running and not self.stop_event.is_set():
            try:
                speech_data = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_speech_input, 1.0
                )

                if not speech_data:
                    continue

                user_text = speech_data['text'].strip()
                speech_end_time = speech_data['user_speech_end_time']

                if self.verbose:
                    cprint(f"[DEBUG] Received speech data: {speech_data}", "cyan")

                if not user_text:
                    if self.verbose:
                        cprint("[DEBUG] Empty user text, skipping", "yellow")
                    continue

                interaction_count += 1
                self.streaming_metrics['total_interactions'] = interaction_count
                self.streaming_metrics['voice_inputs'] += 1

                cprint(f"[Opi] 👂 Heard: \"{user_text}\"", "white")

                if self._is_exit_command(user_text):
                    cprint("[Opi] 👋 Goodbye command detected", "yellow")
                    await self._handle_goodbye()
                    break

                await self._process_with_enhanced_tracing(user_text, speech_end_time, "voice")

            except Exception as e:
                cprint(f"[Opi] ❌ Main loop error: {e}", "red")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.1)

    def _run_speech_worker(self):
        """Run speech recognition in separate thread."""
        try:
            while self.running and not self.stop_event.is_set():
                timings = {}
                
                target_queue = self.input_queue if self.hybrid_mode else self.speech_queue
                
                self.speech_worker.process_speech_input(
                    target_queue,
                    self.stop_event,
                    timings
                )
                if timings.get('transcription_duration'):
                    self.timing_tracker.add_timing('stt', timings['transcription_duration'])
        except Exception as e:
            cprint(f"[Opi] Speech worker error: {e}", "red")
            if self.verbose:
                import traceback
                traceback.print_exc()
            self.stop_event.set()

    async def _handle_goodbye(self):
        """Handle goodbye command with LangSmith tracing."""
        try:
            if self.text_only_mode:
                cprint("[Opi] 🤖 Goodbye!", "blue")
            else:
                await self._stream_simple_response("Goodbye!")
                await asyncio.sleep(3)
        except Exception as e:
            cprint(f"[Opi] Error in goodbye: {e}", "red")

    async def _stream_simple_response(self, text: str):
        """Stream a simple text response."""
        try:
            if self.text_only_mode:
                cprint(f"[Opi] 🤖 {text}", "blue")
            else:
                await self.conversation_manager.process_user_input_streaming(
                    text,
                    time.time(),
                    self.tts_worker,
                    self.audio_worker
                )
        except Exception as e:
            cprint(f"[Opi] Error streaming simple response: {e}", "red")
            cprint(f"[Opi] 🗣️ {text}", "blue")

    def _get_unified_input(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Get input from unified queue."""
        try:
            if not self.text_only_mode and not self.speech_queue.empty():
                voice_data = self.speech_queue.get_nowait()
                if voice_data:
                    voice_data['source'] = 'voice'
                    return voice_data

            if not self.input_queue.empty():
                return self.input_queue.get_nowait()

            try:
                return self.input_queue.get(timeout=timeout)
            except queue.Empty:
                return None

        except queue.Empty:
            return None

    def _get_speech_input(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Get speech input from queue with timeout."""
        try:
            result = self.speech_queue.get(timeout=timeout)
            if self.verbose and result:
                cprint(f"[DEBUG] Retrieved from speech_queue: text='{result.get('text', '')}', time={result.get('user_speech_end_time', 0)}", "cyan")
            return result
        except queue.Empty:
            return None

    def _is_exit_command(self, text: str) -> bool:
        """Check if text contains exit command."""
        text_lower = text.lower().strip()
        exit_phrases = ["goodbye", "bye bye", "exit", "quit", "stop listening", "see you later", "that's all"]

        is_exit = any(phrase in text_lower for phrase in exit_phrases)

        if self.verbose and is_exit:
            cprint(f"[DEBUG] Exit command detected: '{text}'", "yellow")

        return is_exit

    def print_enhanced_performance_summary(self):
        """Print enhanced performance summary with LangSmith metrics."""
        metrics = self.streaming_metrics

        if metrics['total_interactions'] == 0:
            return

        if self.verbose:
            # Detailed summary
            cprint("\n" + "="*70, "cyan", attrs=['bold'])
            cprint("🚀 ENHANCED PERFORMANCE SUMMARY WITH LANGSMITH", "cyan", attrs=['bold'])
            cprint("="*70, "cyan", attrs=['bold'])

            cprint(f"📊 Total Interactions: {metrics['total_interactions']}", "white")
            
            # Input method breakdown
            if metrics.get('text_inputs', 0) > 0 or metrics.get('voice_inputs', 0) > 0:
                cprint(f"\n📝 Input Methods:", "yellow", attrs=['bold'])
                cprint(f"   Voice: {metrics.get('voice_inputs', 0)}", "white")
                cprint(f"   Text:  {metrics.get('text_inputs', 0)}", "white")

            # LangSmith metrics
            if self.langsmith_config.enabled:
                cprint(f"\n🔍 LangSmith Observability:", "yellow", attrs=['bold'])
                cprint(f"   Traces Logged: {metrics.get('langsmith_traces', 0)}", "green")
                cprint(f"   LLM Calls: {metrics.get('llm_calls', 0)}", "green")
                cprint(f"   Tool Calls: {metrics.get('tool_calls', 0)}", "green")
                cprint(f"   Project: {self.langsmith_config.project_name}", "cyan")
                cprint(f"   View at: https://smith.langchain.com/", "blue")

            if metrics['first_audio_times']:
                avg_first_audio = sum(metrics['first_audio_times']) / len(metrics['first_audio_times'])
                min_first_audio = min(metrics['first_audio_times'])
                max_first_audio = max(metrics['first_audio_times'])

                cprint(f"\n⚡ FIRST AUDIO LATENCY:", "yellow", attrs=['bold'])
                cprint(f"   Average: {avg_first_audio:.3f}s", "white")
                cprint(f"   Best:    {min_first_audio:.3f}s", "green")
                cprint(f"   Worst:   {max_first_audio:.3f}s", "red" if max_first_audio > 1.0 else "white")

                if avg_first_audio < 0.5:
                    cprint("   Rating:  ⭐⭐⭐ EXCELLENT - Under 500ms!", "green", attrs=['bold'])
                elif avg_first_audio < 1.0:
                    cprint("   Rating:  ⭐⭐ GOOD - Under 1 second", "green")
                else:
                    cprint("   Rating:  ⭐ Could be improved", "yellow")

            cprint("="*70, "cyan", attrs=['bold'])
        else:
            # Simple summary
            cprint(f"\n[Opi] Enhanced Session Summary:", "cyan")
            cprint(f"  Interactions: {metrics['total_interactions']}", "white")
            
            if metrics.get('text_inputs', 0) > 0 or metrics.get('voice_inputs', 0) > 0:
                cprint(f"  Voice: {metrics.get('voice_inputs', 0)}, Text: {metrics.get('text_inputs', 0)}", "white")

            if self.langsmith_config.enabled:
                cprint(f"  LangSmith Traces: {metrics.get('langsmith_traces', 0)}", "green")

            if metrics['first_audio_times']:
                avg_latency = sum(metrics['first_audio_times']) / len(metrics['first_audio_times'])
                cprint(f"  Avg Response Time: {avg_latency:.2f}s", "white")

                if avg_latency < 0.5:
                    cprint("  Performance: Excellent! ⭐⭐⭐", "green")
                elif avg_latency < 1.0:
                    cprint("  Performance: Good ⭐⭐", "green")
                else:
                    cprint("  Performance: Could be improved ⭐", "yellow")

    async def shutdown(self):
        """Gracefully shutdown all components with LangSmith trace submission."""
        cprint("[Opi] Shutting down with LangSmith trace finalization...", "yellow")

        self.running = False
        self.stop_event.set()

        # Stop async text worker if running
        if self.async_text_worker:
            self.async_text_worker.stop()

        # Signal queues to stop
        try:
            self.speech_queue.put(None)
            self.text_queue.put(None)
            self.input_queue.put(None)
            self.audio_queue.put(None)
        except:
            pass

        # Shutdown LLM components
        if self.conversation_manager:
            await self.conversation_manager.close()

        if self.mcp_manager:
            await self.mcp_manager.close()

        # Wait for all LangSmith traces to be submitted
        if self.langsmith_config.enabled:
            try:
                cprint("[LangSmith] 📤 Waiting for all traces to be submitted...", "cyan")
                wait_for_all_tracers()
                cprint("[LangSmith] ✅ All traces submitted successfully", "green")
            except Exception as e:
                cprint(f"[LangSmith] ⚠️  Error waiting for traces: {e}", "yellow")

        # Print enhanced performance summary
        self.print_enhanced_performance_summary()

        # Print detailed timing summary only in verbose mode
        if self.verbose:
            self.timing_tracker.print_summary()

        cprint("[Opi] ✅ Enhanced shutdown complete", "green")


def setup_signal_handlers(opi: EnhancedOpiVoiceAssistant):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        cprint(f"\n[Opi] Received signal {signum}, shutting down with LangSmith...", "yellow")
        opi.stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point with enhanced LangSmith integration."""
    parser = argparse.ArgumentParser(description="Opi Voice Assistant with LangSmith Observability")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode (same as verbose)")
    
    # Text input options
    parser.add_argument("--text", action="store_true", help="Text-only mode (no voice)")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid mode (voice + text)")
    
    # LangSmith options
    parser.add_argument("--no-langsmith", action="store_true", help="Disable LangSmith tracing")
    parser.add_argument("--langsmith-project", help="Override LangSmith project name")
    
    # Testing options
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--test-voice", action="store_true", help="Test voice components and exit")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming pipeline and exit")
    parser.add_argument("--test-mcp", action="store_true", help="Test MCP server connection and exit")
    parser.add_argument("--test-langsmith", action="store_true", help="Test LangSmith integration and exit")

    args = parser.parse_args()

    # Set debug/verbose mode
    if args.debug or args.verbose:
        os.environ['OPI_DEBUG'] = '1'

    # Override LangSmith settings from command line
    if args.no_langsmith:
        os.environ['LANGSMITH_TRACING'] = 'false'
    
    if args.langsmith_project:
        os.environ['LANGSMITH_PROJECT'] = args.langsmith_project

    if args.list_devices:
        from voice.audio_utils import list_audio_devices
        list_audio_devices()
        return

    # Load environment variables early
    load_dotenv()

    # Determine mode
    text_only_mode = args.text
    hybrid_mode = args.hybrid

    if text_only_mode and hybrid_mode:
        cprint("[Opi] ❌ Cannot use both --text and --hybrid modes", "red")
        sys.exit(1)

    # Initialize Enhanced Opi with LangSmith
    opi = EnhancedOpiVoiceAssistant(config_path=args.config)
    setup_signal_handlers(opi)

    # Show mode and verbose status
    if text_only_mode:
        cprint("[Opi] 💬 TEXT-ONLY MODE WITH LANGSMITH", "cyan", attrs=['bold'])
    elif hybrid_mode:
        cprint("[Opi] 🎤💬 HYBRID MODE WITH LANGSMITH - Voice + Text", "cyan", attrs=['bold'])
    
    if opi.verbose:
        cprint("[Opi] VERBOSE MODE ENABLED - Detailed logging with LangSmith active", "cyan", attrs=['bold'])

    try:
        await opi.initialize(text_only_mode=text_only_mode, hybrid_mode=hybrid_mode)

        # Enhanced testing options
        if args.test_langsmith:
            cprint("[Opi] Testing LangSmith integration...", "cyan")
            if opi.langsmith_config.enabled:
                cprint("[LangSmith] ✅ LangSmith is properly configured", "green")
                cprint(f"[LangSmith] 📊 Project: {opi.langsmith_config.project_name}", "cyan")
                cprint(f"[LangSmith] 🔗 Endpoint: {opi.langsmith_config.endpoint}", "cyan")
                
                # Test a simple LLM call
                test_inputs = ["hello langsmith", "what is 2+2", "test tracing"]
                for test_input in test_inputs:
                    cprint(f"[LangSmith Test] Testing: {test_input}", "yellow")
                    await opi._process_with_enhanced_tracing(test_input, time.time(), "test")
                    await asyncio.sleep(1)
                
                cprint("[LangSmith] ✅ Test complete - check your LangSmith dashboard", "green")
                cprint(f"[LangSmith] 🔗 View at: https://smith.langchain.com/", "blue")
            else:
                cprint("[LangSmith] ❌ LangSmith not configured - set LANGSMITH_API_KEY", "red")
            return

        if args.test_voice and not text_only_mode:
            cprint("[Opi] Testing voice components with LangSmith...", "cyan")
            await opi._stream_simple_response("Voice test successful with LangSmith tracing!")
            await asyncio.sleep(3)
            return

        if args.test_streaming and not text_only_mode:
            cprint("[Opi] Testing streaming pipeline with LangSmith...", "cyan")
            test_inputs = ["hello", "what time is it", "system status", "goodbye"]
            for test_input in test_inputs:
                cprint(f"[Test] Input: {test_input}", "yellow")
                await opi._process_with_enhanced_tracing(test_input, time.time(), "test")
                await asyncio.sleep(1)
            return

        if args.test_mcp:
            cprint("[Opi] Testing MCP server connections with LangSmith...", "cyan")
            if opi.mcp_manager:
                server_status = opi.mcp_manager.get_server_status()
                cprint(f"[MCP Test] Total servers: {server_status['total_servers']}", "white")
                cprint(f"[MCP Test] Connected: {server_status['connected_servers']}", "green")
                cprint(f"[MCP Test] Failed: {server_status['failed_servers']}", "red")

                tools = opi.mcp_manager.get_tools()
                cprint(f"[MCP Test] Available tools: {len(tools)}", "white")
                for tool in tools:
                    cprint(f"  - {tool.name}: {tool.description}", "cyan")

                # Test tools with LangSmith tracing
                if tools and any("secret" in tool.name.lower() for tool in tools):
                    cprint("[MCP Test] Testing secret message tool with LangSmith...", "yellow")
                    try:
                        test_inputs = ["get me a secret message", "tell me secret number 1"]
                        for test_input in test_inputs:
                            cprint(f"[MCP Test] Testing: {test_input}", "yellow")
                            await opi._process_with_enhanced_tracing(test_input, time.time(), "test")
                            await asyncio.sleep(2)
                    except Exception as e:
                        cprint(f"[MCP Test] Error testing tools: {e}", "red")

                cprint("[MCP Test] ✅ MCP test with LangSmith complete", "green")
            else:
                cprint("[MCP Test] ❌ No MCP manager available", "red")
            return

        # Start the enhanced assistant
        await opi.start_listening()

    except Exception as e:
        cprint(f"[Opi] Fatal error: {e}", "red")
        if opi.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    anyio.run(main)
