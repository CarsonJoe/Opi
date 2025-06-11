#!/usr/bin/env python3
"""
Opi Voice Assistant - Orange Pi Voice Assistant with MCP Integration
Main application that combines voice I/O with MCP tool system
"""

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
from llm.mcp_manager import MCPManager
from llm.conversation_manager import ConversationManager
from config.settings import OpiConfig
from utils.timing import TimingTracker

class OpiVoiceAssistant:
    """Main Opi Voice Assistant application."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = OpiConfig.load(config_path)
        self.running = False
        self.stop_event = threading.Event()
        
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
        
        # Debug mode
        self.debug = os.getenv('OPI_DEBUG', '0') == '1'
        
    async def initialize(self):
        """Initialize all components."""
        cprint("[Opi] Initializing voice assistant...", "cyan")
        
        # Initialize voice components
        await self._init_voice_components()
        
        # Initialize LLM components
        await self._init_llm_components()
        
        # Test LLM processing
        if self.debug:
            await self._test_llm_processing()
        
        cprint("[Opi] ✅ All components initialized successfully", "green")
        
    async def _init_voice_components(self):
        """Initialize speech recognition, TTS, and audio components."""
        cprint("[Opi] Loading voice models...", "yellow")
        
        # Speech recognition
        self.speech_worker = SpeechWorker(
            model_size=self.config.voice.whisper_model,
            compute_type=self.config.voice.whisper_compute_type
        )
        
        # Text-to-speech
        self.tts_worker = TTSWorker(
            model_path=self.config.voice.tts_model_path,
            config_path=self.config.voice.tts_config_path,
            speaker_id=self.config.voice.speaker_id,
            speech_speed=self.config.voice.speech_speed
        )
        await self.tts_worker.initialize()
        
        # Audio output
        self.audio_worker = AudioWorker(
            device_index=self.config.voice.audio_device,
            sample_rate=self.config.voice.sample_rate,
            chunk_pause_ms=self.config.voice.chunk_pause_ms
        )
        
        cprint("[Opi] ✅ Voice components ready", "green")
        
    async def _init_llm_components(self):
        """Initialize LLM and MCP components."""
        cprint("[Opi] Initializing LLM and MCP tools...", "yellow")
        
        # Validate LLM configuration
        if not self.config.llm.api_key:
            cprint("[Opi] ❌ No LLM API key configured!", "red")
            cprint("       Set OPENAI_API_KEY environment variable", "yellow")
            cprint("       The assistant will use fallback responses only", "yellow")
        else:
            if self.debug:
                cprint(f"[Opi] LLM API key configured: {self.config.llm.api_key[:10]}...", "green")
        
        # MCP manager for tool integration
        self.mcp_manager = MCPManager(
            server_configs=self.config.mcp.servers,
            force_refresh=False
        )
        await self.mcp_manager.initialize()
        
        # Conversation manager
        self.conversation_manager = ConversationManager(
            llm_config=self.config.llm,
            system_prompt=self.config.prompts.system_prompt,
            mcp_manager=self.mcp_manager,
            db_path=self.config.storage.conversation_db
        )
        await self.conversation_manager.initialize()
        
        if self.debug:
            cprint(f"[Opi] ConversationManager type: {type(self.conversation_manager)}", "cyan")
            cprint(f"[Opi] ConversationManager module: {self.conversation_manager.__class__.__module__}", "cyan")
        
        cprint(f"[Opi] ✅ LLM ready with {len(self.mcp_manager.get_tools())} MCP tools", "green")
        
    async def _test_llm_processing(self):
        """Test LLM processing functionality."""
        cprint("[Opi] Testing LLM processing...", "yellow")
        try:
            test_responses = []
            async for chunk in self.conversation_manager.process_user_input("hello", time.time()):
                test_responses.append(chunk)
            
            if test_responses:
                full_response = "".join(test_responses)
                cprint(f"[Opi] ✅ LLM test successful: '{full_response[:50]}...'", "green")
            else:
                cprint("[Opi] ⚠️ LLM test returned no response", "yellow")
        except Exception as e:
            cprint(f"[Opi] ❌ LLM test failed: {e}", "red")
            if self.debug:
                import traceback
                traceback.print_exc()
        
    async def start_listening(self):
        """Start the main voice interaction loop."""
        if not self.speech_worker or not self.conversation_manager:
            raise RuntimeError("Components not initialized. Call initialize() first.")
            
        self.running = True
        cprint("[Opi] 🎤 Starting voice assistant... Listening for any speech!", "blue")
        
        # Start worker threads
        speech_thread = threading.Thread(
            target=self._run_speech_worker,
            name="SpeechWorker",
            daemon=True
        )
        
        tts_thread = threading.Thread(
            target=self._run_tts_worker,
            name="TTSWorker", 
            daemon=True
        )
        
        audio_thread = threading.Thread(
            target=self._run_audio_worker,
            name="AudioWorker",
            daemon=True
        )
        
        speech_thread.start()
        tts_thread.start()
        audio_thread.start()
        
        # Main interaction loop
        try:
            await self._main_loop()
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
            if self.debug:
                import traceback
                traceback.print_exc()
            self.stop_event.set()
            
    def _run_tts_worker(self):
        """Run TTS synthesis in separate thread."""
        try:
            while self.running and not self.stop_event.is_set():
                timings = {}
                self.tts_worker.process_synthesis(
                    self.text_queue,
                    self.audio_queue,
                    self.stop_event,
                    timings,
                    self.config.voice.speech_speed,
                    self.config.storage.output_dir
                )
                if timings.get('tts_generation_duration'):
                    self.timing_tracker.add_timing('tts', timings['tts_generation_duration'])
        except Exception as e:
            cprint(f"[Opi] TTS worker error: {e}", "red")
            if self.debug:
                import traceback
                traceback.print_exc()
            self.stop_event.set()
            
    def _run_audio_worker(self):
        """Run audio playback in separate thread."""
        try:
            while self.running and not self.stop_event.is_set():
                timings = {}
                self.audio_worker.process_playback(
                    self.audio_queue,
                    self.stop_event,
                    timings
                )
        except Exception as e:
            cprint(f"[Opi] Audio worker error: {e}", "red")
            if self.debug:
                import traceback
                traceback.print_exc()
            self.stop_event.set()
            
    async def _main_loop(self):
        """Main interaction loop that processes voice input and generates responses."""
        interaction_count = 0
        
        cprint("[Opi] 🎤 Ready! Listening for any speech...", "green")
        
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
                
                if self.debug:
                    cprint(f"[DEBUG] Received speech data: {speech_data}", "cyan")
                
                if not user_text:
                    if self.debug:
                        cprint("[DEBUG] Empty user text, skipping", "yellow")
                    continue
                    
                # Process any speech input immediately
                interaction_count += 1
                cprint(f"[Opi] 👂 Heard: \"{user_text}\"", "white")
                cprint(f"[Opi] 🤖 Processing request #{interaction_count}...", "blue")
                
                # Check for exit commands
                if self._is_exit_command(user_text):
                    cprint("[Opi] 👋 Goodbye command detected", "yellow")
                    self._queue_response("Goodbye!")
                    await asyncio.sleep(2)  # Wait for response to play
                    break
                    
                # Generate response using LLM + MCP tools
                response_start_time = time.time()
                
                try:
                    if self.debug:
                        cprint(f"[DEBUG] Calling conversation_manager.process_user_input with: '{user_text}'", "cyan")
                    
                    response_chunks = []
                    chunk_count = 0
                    
                    async for response_chunk in self.conversation_manager.process_user_input(
                        user_text, 
                        speech_end_time
                    ):
                        if response_chunk and response_chunk.strip():
                            chunk_count += 1
                            response_chunks.append(response_chunk)
                            
                            if self.debug:
                                cprint(f"[DEBUG] Got response chunk #{chunk_count}: '{response_chunk[:50]}...'", "green")
                            
                            # Queue for TTS
                            self.text_queue.put(response_chunk)
                            
                    # Mark end of response
                    self.text_queue.put(None)
                    
                    # Log complete response
                    if response_chunks:
                        full_response = "".join(response_chunks)
                        cprint(f"[Opi] 🗣️ Response: \"{full_response}\"", "green")
                        
                        response_time = time.time() - response_start_time
                        self.timing_tracker.add_timing('llm_response', response_time)
                        
                        if self.debug:
                            cprint(f"[DEBUG] LLM response completed in {response_time:.2f}s with {chunk_count} chunks", "cyan")
                    else:
                        cprint("[Opi] ⚠️ No response received from LLM", "yellow")
                        self._queue_response("I'm sorry, I didn't understand that. Could you try again?")
                    
                except Exception as e:
                    cprint(f"[Opi] ❌ Error processing input: {e}", "red")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    self._queue_response("Sorry, I encountered an error processing your request.")
                    
            except Exception as e:
                cprint(f"[Opi] ❌ Main loop error: {e}", "red")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.1)
                
    def _get_speech_input(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Get speech input from queue with timeout."""
        try:
            result = self.speech_queue.get(timeout=timeout)
            if self.debug and result:
                cprint(f"[DEBUG] Retrieved from speech_queue: text='{result.get('text', '')}', time={result.get('user_speech_end_time', 0)}", "cyan")
            return result
        except queue.Empty:
            return None
            
    def _detect_wake_word(self, text: str) -> bool:
        """Detect wake word in speech text. (DISABLED - will be handled at audio level)"""
        # This function is kept for compatibility but always returns False
        # Wake word detection will be implemented at the audio capture level
        return False
        
    def _is_exit_command(self, text: str) -> bool:
        """Check if text contains exit command."""
        text_lower = text.lower().strip()
        exit_phrases = ["goodbye", "bye bye", "exit", "quit", "stop listening", "see you later", "that's all"]
        
        is_exit = any(phrase in text_lower for phrase in exit_phrases)
        
        if self.debug and is_exit:
            cprint(f"[DEBUG] Exit command detected: '{text}'", "yellow")
        
        return is_exit
        
    def _queue_response(self, text: str):
        """Queue text response for TTS."""
        if self.debug:
            cprint(f"[DEBUG] Queueing response: '{text}'", "green")
        
        self.text_queue.put(text)
        self.text_queue.put(None)  # End marker
        
    async def shutdown(self):
        """Gracefully shutdown all components."""
        cprint("[Opi] Shutting down components...", "yellow")
        
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
            
        # Print timing summary
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
    parser = argparse.ArgumentParser(description="Opi Voice Assistant")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--test-voice", action="store_true", help="Test voice components and exit")
    parser.add_argument("--test-llm", action="store_true", help="Test LLM integration and exit")
    
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        os.environ['OPI_DEBUG'] = '1'
    
    if args.list_devices:
        from voice.audio_utils import list_audio_devices
        list_audio_devices()
        return
        
    # Load environment variables
    load_dotenv()
    
    # Initialize Opi
    opi = OpiVoiceAssistant(config_path=args.config)
    setup_signal_handlers(opi)
    
    try:
        await opi.initialize()
        
        if args.test_voice:
            cprint("[Opi] Testing voice components...", "cyan")
            opi._queue_response("Voice test successful! All components are working.")
            await asyncio.sleep(3)
            return
            
        if args.test_llm:
            cprint("[Opi] Testing LLM integration...", "cyan")
            test_inputs = ["hello", "what time is it", "system status"]
            for test_input in test_inputs:
                cprint(f"[Test] Input: {test_input}", "yellow")
                async for chunk in opi.conversation_manager.process_user_input(test_input, time.time()):
                    cprint(f"[Test] Output: {chunk}", "green")
            return
            
        await opi.start_listening()
        
    except Exception as e:
        cprint(f"[Opi] Fatal error: {e}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
