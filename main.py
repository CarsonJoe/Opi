#!/usr/bin/env python3
"""
Opi Voice Assistant - FIXED MCP VERSION
Main application with fixed MCP manager
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

# Core components
from voice.speech_worker import SpeechWorker
from voice.tts_worker import TTSWorker
from voice.audio_worker import AudioWorker
from llm.mcp_manager import MCPManager  # Use fixed version
from llm.conversation_manager import ConversationManager
from config.settings import OpiConfig
from utils.timing import TimingTracker

class OpiVoiceAssistant:
    """Main Opi Voice Assistant application with fixed MCP support."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = OpiConfig.load(config_path)
        self.running = False
        self.stop_event = threading.Event()

        # Simple verbose flag from config or environment
        self.verbose = getattr(self.config, 'verbose', False) or os.getenv('OPI_DEBUG', '0') == '1'

        # Voice components
        self.speech_worker: Optional[SpeechWorker] = None
        self.tts_worker: Optional[TTSWorker] = None
        self.audio_worker: Optional[AudioWorker] = None

        # LLM components
        self.mcp_manager: Optional[MCPManager] = None
        self.conversation_manager: Optional[ConversationManager] = None

        # Queues for inter-component communication
        self.speech_queue = queue.Queue(maxsize=10)
        self.text_queue = queue.Queue(maxsize=50)
        self.audio_queue = queue.Queue(maxsize=100)

        # Timing and monitoring
        self.timing_tracker = TimingTracker()

        # Streaming performance metrics
        self.streaming_metrics = {
            'total_interactions': 0,
            'first_audio_times': [],
            'total_response_times': [],
        }

    async def initialize(self):
        """Initialize all components."""
        if self.verbose:
            cprint("[Opi] Initializing voice assistant with streaming (VERBOSE MODE)...", "cyan", attrs=['bold'])
        else:
            cprint("[Opi] Initializing voice assistant...", "cyan", attrs=['bold'])

        # Initialize voice components
        await self._init_voice_components()

        # Initialize LLM components
        await self._init_llm_components()

        # Test LLM processing in verbose mode only
        if self.verbose:
            await self._test_streaming_processing()

        cprint("[Opi] ✅ All components initialized", "green", attrs=['bold'])

    async def _init_voice_components(self):
        """Initialize speech recognition, TTS, and audio components."""
        cprint("[Opi] Loading voice components...", "yellow")

        # Speech recognition with verbose flag
        self.speech_worker = SpeechWorker(
            model_size=self.config.voice.whisper_model,
            compute_type=self.config.voice.whisper_compute_type,
            verbose=self.verbose
        )

        # Text-to-speech with verbose flag
        self.tts_worker = TTSWorker(
            model_path=self.config.voice.tts_model_path,
            config_path=self.config.voice.tts_config_path,
            speaker_id=self.config.voice.speaker_id,
            speech_speed=self.config.voice.speech_speed,
            verbose=self.verbose
        )
        await self.tts_worker.initialize()

        # Audio output with verbose flag
        self.audio_worker = AudioWorker(
            device_index=self.config.voice.audio_device,
            sample_rate=self.config.voice.sample_rate,
            chunk_pause_ms=self.config.voice.chunk_pause_ms,
            verbose=self.verbose
        )

        cprint("[Opi] ✅ Voice components ready", "green")

    async def _init_llm_components(self):
        """Initialize LLM and MCP components."""
        cprint("[Opi] Initializing LLM components...", "yellow")

        # Validate LLM configuration
        if not self.config.llm.api_key:
            cprint("[Opi] ❌ No LLM API key configured!", "red")
            cprint("       Set GOOGLE_API_KEY environment variable", "yellow")
            cprint("       The assistant will use fallback responses only", "yellow")
        else:
            if self.verbose:
                cprint(f"[Opi] LLM API key configured: {self.config.llm.api_key[:10]}...", "green")

        self.mcp_manager = MCPManager(
            config=self.config.mcp.servers,
        )
        await self.mcp_manager.connect_all()

        # Conversation manager with streaming support
        self.conversation_manager = ConversationManager(
            llm_config=self.config.llm,
            system_prompt=self.config.prompts.system_prompt,
            mcp_manager=self.mcp_manager,
            db_path=self.config.storage.conversation_db
        )
        await self.conversation_manager.initialize()

        tool_count = len(self.mcp_manager.get_tools())
        server_status = self.mcp_manager.get_server_status()  # This should work now
        connected_servers = server_status['connected_servers']
        total_servers = server_status['total_servers']

        cprint(f"[Opi] ✅ LLM ready with {connected_servers}/{total_servers} MCP servers", "green")
        cprint(f"[Opi] 🛠️ Available tools: {tool_count}", "green")

    async def _test_streaming_processing(self):
        """Test streaming processing functionality (verbose mode only)."""
        cprint("[Opi] Testing streaming pipeline...", "yellow")
        try:
            test_start = time.time()

            # Test the streaming method
            first_audio_time = await self.conversation_manager.process_user_input_streaming(
                "hello",
                time.time(),
                self.tts_worker,
                self.audio_worker
            )

            test_duration = time.time() - test_start

            if first_audio_time:
                first_audio_latency = first_audio_time - test_start
                cprint(f"[Opi] ✅ Streaming test successful!", "green")
                cprint(f"[Opi] ⚡ First audio latency: {first_audio_latency:.3f}s", "green", attrs=['bold'])
                cprint(f"[Opi] 📊 Total test time: {test_duration:.3f}s", "white")
            else:
                cprint("[Opi] ⚠️ Streaming test completed but no first audio time recorded", "yellow")

        except Exception as e:
            cprint(f"[Opi] ❌ Streaming test failed: {e}", "red")
            if self.verbose:
                import traceback
                traceback.print_exc()

    async def start_listening(self):
        """Start the main voice interaction loop."""
        if not self.speech_worker or not self.conversation_manager:
            raise RuntimeError("Components not initialized. Call initialize() first.")

        self.running = True

        if self.verbose:
            cprint("[Opi] 🚀 Starting voice assistant (VERBOSE MODE)... Listening!", "blue", attrs=['bold'])
        else:
            cprint("[Opi] 🚀 Starting voice assistant... Listening!", "blue", attrs=['bold'])

        # Start worker threads
        speech_thread = threading.Thread(
            target=self._run_speech_worker,
            name="SpeechWorker",
            daemon=True
        )

        speech_thread.start()

        # Main interaction loop (no separate TTS/audio threads - handled by streaming pipeline)
        try:
            await self._streaming_main_loop()
        except KeyboardInterrupt:
            cprint("\n[Opi] Shutting down...", "yellow")
        finally:
            await self.shutdown()

    def _run_speech_worker(self):
        """Run speech recognition in separate thread."""
        try:
            while self.running and not self.stop_event.is_set():
                timings = {}
                self.speech_worker.process_speech_input(
                    self.speech_queue,
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

    async def _streaming_main_loop(self):
        """Main loop with streaming pipeline."""
        interaction_count = 0

        cprint("[Opi] 🎤 Ready! Listening for speech...", "green", attrs=['bold'])

        while self.running and not self.stop_event.is_set():
            try:
                # Wait for speech input
                speech_data = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_speech_input, 1.0  # 1 second timeout
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

                # Process with streaming pipeline
                interaction_count += 1
                self.streaming_metrics['total_interactions'] = interaction_count

                cprint(f"[Opi] 👂 Heard: \"{user_text}\"", "white")

                # Check for exit commands
                if self._is_exit_command(user_text):
                    cprint("[Opi] 👋 Goodbye command detected", "yellow")
                    await self._handle_goodbye()
                    break

                # MAIN STREAMING PROCESSING
                await self._process_with_streaming_pipeline(user_text, speech_end_time)

            except Exception as e:
                cprint(f"[Opi] ❌ Main loop error: {e}", "red")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.1)

    async def _process_with_streaming_pipeline(self, user_text: str, speech_end_time: float):
        """Process user input with ultra-low latency streaming."""

        interaction_start = time.time()

        try:
            if self.verbose:
                cprint(f"[DEBUG] Starting streaming processing for: '{user_text}'", "cyan")

            # Use the enhanced conversation manager's streaming method
            first_audio_time = await self.conversation_manager.process_user_input_streaming(
                user_text,
                speech_end_time,
                self.tts_worker,
                self.audio_worker
            )

            # Calculate and log performance metrics
            total_time = time.time() - interaction_start

            if first_audio_time:
                first_audio_latency = first_audio_time - speech_end_time
                self.streaming_metrics['first_audio_times'].append(first_audio_latency)
                self.timing_tracker.add_timing('first_audio_latency', first_audio_latency)

                if self.verbose:
                    cprint(f"[Opi] ⚡ FIRST AUDIO: {first_audio_latency:.3f}s", "green", attrs=['bold'])

            self.streaming_metrics['total_response_times'].append(total_time)
            self.timing_tracker.add_timing('streaming_response', total_time)

            if self.verbose:
                cprint(f"[Opi] ✅ Complete streaming response in {total_time:.3f}s", "green")

                # Show running average in verbose mode
                if self.streaming_metrics['first_audio_times']:
                    avg_first_audio = sum(self.streaming_metrics['first_audio_times']) / len(self.streaming_metrics['first_audio_times'])
                    cprint(f"[DEBUG] Average first audio latency: {avg_first_audio:.3f}s", "cyan")

        except Exception as e:
            cprint(f"[Opi] ❌ Streaming pipeline error: {e}", "red")
            if self.verbose:
                import traceback
                traceback.print_exc()

            # Fallback to simple error response
            await self._stream_simple_response("Sorry, I encountered an error processing your request.")

    async def _stream_simple_response(self, text: str):
        """Stream a simple text response using the pipeline."""
        try:
            # Use the streaming method for simple responses too
            await self.conversation_manager.process_user_input_streaming(
                "error_response",
                time.time(),
                self.tts_worker,
                self.audio_worker
            )
        except Exception as e:
            cprint(f"[Opi] Error streaming simple response: {e}", "red")
            # Ultimate fallback - just print
            cprint(f"[Opi] 🗣️ {text}", "blue")

    async def _handle_goodbye(self):
        """Handle goodbye command with streaming."""
        try:
            await self._stream_simple_response("Goodbye!")
            await asyncio.sleep(3)  # Wait for goodbye to play
        except Exception as e:
            cprint(f"[Opi] Error in goodbye: {e}", "red")

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

    def print_streaming_performance_summary(self):
        """Print performance summary."""
        metrics = self.streaming_metrics

        if metrics['total_interactions'] == 0:
            return

        if self.verbose:
            # Detailed summary in verbose mode
            cprint("\n" + "="*60, "cyan", attrs=['bold'])
            cprint("🚀 STREAMING PERFORMANCE SUMMARY", "cyan", attrs=['bold'])
            cprint("="*60, "cyan", attrs=['bold'])

            cprint(f"📊 Total Interactions: {metrics['total_interactions']}", "white")

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

            cprint("="*60, "cyan", attrs=['bold'])
        else:
            # Simple summary in normal mode
            cprint(f"\n[Opi] Session Summary:", "cyan")
            cprint(f"  Interactions: {metrics['total_interactions']}", "white")

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
        """Gracefully shutdown all components."""
        cprint("[Opi] Shutting down...", "yellow")

        self.running = False
        self.stop_event.set()

        # Signal queues to stop
        try:
            self.speech_queue.put(None)
            self.text_queue.put(None)
            self.audio_queue.put(None)
        except:
            pass

        # Shutdown LLM components
        if self.conversation_manager:
            await self.conversation_manager.close()

        if self.mcp_manager:
            await self.mcp_manager.close()

        # Print performance summary
        self.print_streaming_performance_summary()

        # Print detailed timing summary only in verbose mode
        if self.verbose:
            self.timing_tracker.print_summary()

        cprint("[Opi] ✅ Shutdown complete", "green")


def setup_signal_handlers(opi: OpiVoiceAssistant):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        cprint(f"\n[Opi] Received signal {signum}, shutting down...", "yellow")
        opi.stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Opi Voice Assistant with MCP Support")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode (same as verbose)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--test-voice", action="store_true", help="Test voice components and exit")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming pipeline and exit")
    parser.add_argument("--test-mcp", action="store_true", help="Test MCP server connection and exit")

    args = parser.parse_args()

    # Set debug/verbose mode
    if args.debug or args.verbose:
        os.environ['OPI_DEBUG'] = '1'

    if args.list_devices:
        from voice.audio_utils import list_audio_devices
        list_audio_devices()
        return

    # Load environment variables
    load_dotenv()

    # Initialize Opi with streaming
    opi = OpiVoiceAssistant(config_path=args.config)
    setup_signal_handlers(opi)

    # Show verbose mode status
    if opi.verbose:
        cprint("[Opi] VERBOSE MODE ENABLED - Detailed logging active", "cyan", attrs=['bold'])

    try:
        await opi.initialize()

        if args.test_voice:
            cprint("[Opi] Testing voice components...", "cyan")
            await opi._stream_simple_response("Voice test successful! Streaming pipeline is working.")
            await asyncio.sleep(3)
            return

        if args.test_streaming:
            cprint("[Opi] Testing streaming pipeline...", "cyan")
            test_inputs = ["hello", "what time is it", "system status", "goodbye"]
            for test_input in test_inputs:
                cprint(f"[Test] Input: {test_input}", "yellow")
                await opi._process_with_streaming_pipeline(test_input, time.time())
                await asyncio.sleep(1)
            return

        if args.test_mcp:
            cprint("[Opi] Testing MCP server connections...", "cyan")
            if opi.mcp_manager:
                server_status = opi.mcp_manager.get_server_status()
                cprint(f"[MCP Test] Total servers: {server_status['total_servers']}", "white")
                cprint(f"[MCP Test] Connected: {server_status['connected_servers']}", "green")
                cprint(f"[MCP Test] Failed: {server_status['failed_servers']}", "red")

                tools = opi.mcp_manager.get_tools()
                cprint(f"[MCP Test] Available tools: {len(tools)}", "white")
                for tool in tools:
                    cprint(f"  - {tool.name}: {tool.description}", "cyan")

                # Test a tool if available
                if tools and any("secret" in tool.name.lower() for tool in tools):
                    cprint("[MCP Test] Testing secret message tool...", "yellow")
                    try:
                        test_inputs = ["get me a secret message", "tell me secret number 1"]
                        for test_input in test_inputs:
                            cprint(f"[MCP Test] Testing: {test_input}", "yellow")
                            await opi._process_with_streaming_pipeline(test_input, time.time())
                            await asyncio.sleep(2)
                    except Exception as e:
                        cprint(f"[MCP Test] Error testing tools: {e}", "red")

                cprint("[MCP Test] ✅ MCP test complete", "green")
            else:
                cprint("[MCP Test] ❌ No MCP manager available", "red")
            return

        await opi.start_listening()

    except Exception as e:
        cprint(f"[Opi] Fatal error: {e}", "red")
        if opi.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    anyio.run(main)
