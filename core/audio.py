"""
Audio system for Opi Voice Assistant.
Combines STT, TTS, and wake word detection.
Adapted from the chat application codebase.
"""

import asyncio
import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional

# Speech recognition
from faster_whisper import WhisperModel

# TTS - we'll use a simple TTS for now, can be replaced with Piper later
import subprocess
import tempfile

from core.config import AudioConfig, SpeechConfig, TTSConfig


class AudioManager:
    """Manages all audio operations: wake word detection, STT, and TTS."""
    
    def __init__(self, audio_config: AudioConfig, speech_config: SpeechConfig, tts_config: TTSConfig):
        self.audio_config = audio_config
        self.speech_config = speech_config
        self.tts_config = tts_config
        self.logger = logging.getLogger("AudioManager")
        
        # Components
        self.wake_word_detector = None
        self.speech_recognizer = None
        self.tts_engine = None
        
        # State
        self.initialized = False
        self.recording = False
        self.speaking = False
        
        # Audio settings
        self.sample_rate = audio_config.sample_rate
        self.chunk_size = audio_config.chunk_size
        self.device_index = audio_config.device_index
    
    async def initialize(self):
        """Initialize all audio components."""
        if self.initialized:
            return
        
        self.logger.info("Initializing audio system...")
        
        # Test audio devices
        self._check_audio_devices()
        
        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector(
            self.audio_config.wake_word,
            self.audio_config.wake_sensitivity
        )
        
        # Initialize speech recognizer
        self.speech_recognizer = SpeechRecognizer(self.speech_config)
        await self.speech_recognizer.initialize()
        
        # Initialize TTS engine
        self.tts_engine = TTSEngine(self.tts_config)
        await self.tts_engine.initialize()
        
        self.initialized = True
        self.logger.info("Audio system initialized successfully")
    
    def _check_audio_devices(self):
        """Check and log available audio devices."""
        try:
            devices = sd.query_devices()
            self.logger.info("Available audio devices:")
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 or device['max_output_channels'] > 0:
                    self.logger.info(f"  {i}: {device['name']} "
                                   f"(In: {device['max_input_channels']}, "
                                   f"Out: {device['max_output_channels']})")
            
            # Set default device if not specified
            if self.device_index is None:
                default_device = sd.default.device
                self.logger.info(f"Using default audio device: {default_device}")
        except Exception as e:
            self.logger.warning(f"Could not query audio devices: {e}")
    
    async def wait_for_wake_word(self) -> bool:
        """Wait for wake word detection."""
        if not self.initialized:
            await self.initialize()
        
        return await self.wake_word_detector.wait_for_wake_word()
    
    async def listen_for_speech(self) -> Optional[str]:
        """Listen for user speech and return transcription."""
        if not self.initialized:
            await self.initialize()
        
        if self.recording:
            self.logger.warning("Already recording")
            return None
        
        try:
            self.recording = True
            return await self.speech_recognizer.listen_and_transcribe()
        finally:
            self.recording = False
    
    async def speak(self, text: str):
        """Speak the given text using TTS."""
        if not self.initialized:
            await self.initialize()
        
        if self.speaking:
           self.logger.warning("Already speaking")
            return
        
        try:
            self.speaking = True
            await self.tts_engine.speak(text)
        finally:
            self.speaking = False
    
    async def shutdown(self):
        """Shutdown audio system."""
        self.logger.info("Shutting down audio system...")
        
        if self.wake_word_detector:
            await self.wake_word_detector.shutdown()
        
        if self.speech_recognizer:
            await self.speech_recognizer.shutdown()
        
        if self.tts_engine:
            await self.tts_engine.shutdown()
        
        self.initialized = False


class WakeWordDetector:
    """Simple wake word detection using keyword matching in transcription."""
    
    def __init__(self, wake_word: str, sensitivity: float = 0.7):
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.logger = logging.getLogger("WakeWordDetector")
        self.running = False
    
    async def wait_for_wake_word(self) -> bool:
        """Wait for wake word detection."""
        self.logger.debug(f"Listening for wake word: '{self.wake_word}'")
        
        # Simple implementation: listen for audio and check for wake word
        # In a production system, you'd use a dedicated wake word engine
        try:
            self.running = True
            
            # Record a short audio snippet
            duration = 3.0  # seconds
            audio_data = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            
            # Quick transcription to check for wake word
            # This is a simplified approach - in reality you'd use a more efficient method
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(temp_file.name, 16000, audio_data)
                
                # Use a very basic check - just see if we can detect speech
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                if rms > 0.01:  # Basic voice activity detection
                    self.logger.debug("Voice activity detected, assuming wake word")
                    return True
                    
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Wake word detection error: {e}")
            return False
        finally:
            self.running = False
    
    async def shutdown(self):
        """Shutdown wake word detector."""
        self.running = False


class SpeechRecognizer:
    """Speech recognition using Whisper."""
    
    def __init__(self, config: SpeechConfig):
        self.config = config
        self.logger = logging.getLogger("SpeechRecognizer")
        self.model = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Whisper model."""
        if self.initialized:
            return
        
        self.logger.info(f"Loading Whisper model: {self.config.model}")
        
        # Load model in a separate thread to avoid blocking
        def load_model():
            return WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type
            )
        
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(None, load_model)
        
        self.initialized = True
        self.logger.info("Speech recognizer initialized")
    
    async def listen_and_transcribe(self) -> Optional[str]:
        """Listen for speech and return transcription."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Record audio
            self.logger.debug("Recording speech...")
            duration = 5.0  # Maximum recording duration
            
            audio_data = sd.rec(
                int(duration * self.config.device_sample_rate if hasattr(self.config, 'device_sample_rate') else duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='int16'
            )
            sd.wait()
            
            # Save to temporary file for Whisper
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(temp_file.name, 16000, audio_data)
                
                # Transcribe
                self.logger.debug("Transcribing speech...")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._transcribe_file, 
                    temp_file.name
                )
                
                return result
                
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
        
        except Exception as e:
            self.logger.error(f"Speech recognition error: {e}")
            return None
    
    def _transcribe_file(self, file_path: str) -> Optional[str]:
        """Transcribe audio file using Whisper."""
        try:
            segments, info = self.model.transcribe(
                file_path,
                language=self.config.language if self.config.language != "auto" else None
            )
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments])
            return text.strip() if text.strip() else None
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
    
    async def shutdown(self):
        """Shutdown speech recognizer."""
        self.initialized = False
        self.model = None


class TTSEngine:
    """Text-to-speech engine. Simple implementation using system TTS for now."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.logger = logging.getLogger("TTSEngine")
        self.initialized = False
    
    async def initialize(self):
        """Initialize TTS engine."""
        if self.initialized:
            return
        
        self.logger.info("Initializing TTS engine...")
        
        # For now, use system TTS. Later we can integrate Piper TTS
        # Check if espeak is available
        try:
            result = subprocess.run(['which', 'espeak'], capture_output=True)
            if result.returncode == 0:
                self.logger.info("Using espeak for TTS")
            else:
                self.logger.warning("espeak not found, TTS may not work")
        except Exception as e:
            self.logger.warning(f"Could not check for espeak: {e}")
        
        self.initialized = True
    
    async def speak(self, text: str):
        """Speak the given text."""
        if not self.initialized:
            await self.initialize()
        
        try:
            self.logger.debug(f"Speaking: {text}")
            
            # Use espeak for simple TTS
            # Adjust speed based on config
            speed = int(175 * self.config.speed)  # espeak speed in words per minute
            
            cmd = [
                'espeak',
                '-s', str(speed),
                '-v', 'en',
                text
            ]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                subprocess.run,
                cmd
            )
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
    
    async def shutdown(self):
        """Shutdown TTS engine."""
        self.initialized = False 
