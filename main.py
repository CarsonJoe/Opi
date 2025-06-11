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
        
    async def initialize(self):
        """Initialize all components."""
        cprint("[Opi] Initializing voice assistant...", "cyan")
        
        # Initialize voice components
        await self._init_voice_components()
        
        # Initialize LLM components
        await self._init_llm_components()
        
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
        
        cprint(f"[Opi] ✅ LLM ready with {len(self.mcp_manager.get_tools())} MCP tools", "green")
        
    async def start_listening(self):
        """Start the main voice interaction loop."""
        if not self.speech_worker or not self.conversation_manager:
            raise RuntimeError("Components not initialized. Call initialize() first.")
            
        self.running = True
        cprint("[Opi] 🎤 Starting voice assistant... Say 'Hey Opi' to begin!", "blue")
        
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
            self.stop_event.set()
            
    async def _main_loop(self):
        """Main interaction loop that processes voice input and generates responses."""
        wake_word_detected = False
        
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
                
                if not user_text:
                    continue
                    
                # Check for wake word or if already activated
                if not wake_word_detected:
                    if self._detect_wake_word(user_text):
                        wake_word_detected = True
                        cprint("[Opi] 👋 Wake word detected! Listening...", "green")
                        self._queue_response("Hello! How can I help you?")
                        continue
                    else:
                        continue  # Ignore speech until wake word
                        
                # Process user input
                cprint(f"[Opi] User: {user_text}", "white")
                
                # Check for exit commands
                if self._is_exit_command(user_text):
                    self._queue_response("Goodbye!")
                    await asyncio.sleep(2)  # Wait for response to play
                    break
                    
                # Generate response using LLM + MCP tools
                response_start_time = time.time()
                
                try:
                    async for response_chunk in self.conversation_manager.process_user_input(
                        user_text, 
                        speech_end_time
                    ):
                        if response_chunk.strip():
                            self.text_queue.put(response_chunk)
                            
                    # Mark end of response
                    self.text_queue.put(None)
                    
                    response_time = time.time() - response_start_time
                    self.timing_tracker.add_timing('llm_response', response_time)
                    
                except Exception as e:
                    cprint(f"[Opi] Error processing input: {e}", "red")
                    self._queue_response("Sorry, I encountered an error processing your request.")
                    
                # Reset wake word after processing (require wake word for next interaction)
                if self.config.voice.require_wake_word_each_time:
                    wake_word_detected = False
                    
            except Exception as e:
                cprint(f"[Opi] Main loop error: {e}", "red")
                await asyncio.sleep(0.1)
                
    def _get_speech_input(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Get speech input from queue with timeout."""
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _detect_wake_word(self, text: str) -> bool:
        """Detect wake word in speech text."""
        text_lower = text.lower().strip()
        wake_words = self.config.voice.wake_words
        
        return any(wake_word.lower() in text_lower for wake_word in wake_words)
        
    def _is_exit_command(self, text: str) -> bool:
        """Check if text contains exit command."""
        text_lower = text.lower().strip()
        exit_phrases = ["goodbye", "bye bye", "exit", "quit", "stop listening"]
        
        return any(phrase in text_lower for phrase in exit_phrases)
        
    def _queue_response(self, text: str):
        """Queue text response for TTS."""
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
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--test-voice", action="store_true", help="Test voice components and exit")
    
    args = parser.parse_args()
    
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
            
        await opi.start_listening()
        
    except Exception as e:
        cprint(f"[Opi] Fatal error: {e}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 
