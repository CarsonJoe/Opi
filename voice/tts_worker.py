# voice/tts_worker.py
"""
Text-to-speech worker for Opi Voice Assistant
Simplified version that works without Piper initially
"""

import time
import traceback
import numpy as np
from termcolor import cprint
import queue
import threading
from pathlib import Path
import subprocess
import tempfile
import os


class TTSWorker:
    """Text-to-speech worker that can use multiple backends."""
    
    def __init__(self, model_path=None, config_path=None, speaker_id=None, speech_speed=1.0):
        self.model_path = model_path
        self.config_path = config_path  
        self.speaker_id = speaker_id
        self.speech_speed = speech_speed
        self.sample_rate = 22050
        self.tts_backend = None
        self.audio_file_counter = 0
        
    async def initialize(self):
        """Initialize the TTS backend."""
        print("[TTS] Initializing TTS worker...")
        
        # Try to determine which TTS backend to use
        if self.model_path and Path(self.model_path).exists():
            try:
                await self._init_piper()
                self.tts_backend = "piper"
                print("[TTS] ✅ Piper TTS initialized")
            except Exception as e:
                print(f"[TTS] ⚠️  Piper failed to initialize: {e}")
                await self._init_fallback()
        else:
            print("[TTS] No TTS model specified, using fallback")
            await self._init_fallback()
    
    async def _init_piper(self):
        """Initialize Piper TTS."""
        try:
            # Check if piper command is available
            result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Piper command not found")
            
            # Test piper with model
            test_result = subprocess.run([
                'piper', '--model', self.model_path, '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if test_result.returncode != 0:
                raise Exception(f"Piper model test failed: {test_result.stderr}")
                
        except Exception as e:
            raise Exception(f"Piper initialization failed: {e}")
    
    async def _init_fallback(self):
        """Initialize fallback TTS (espeak or festival)."""
        # Try espeak first
        try:
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode == 0:
                self.tts_backend = "espeak"
                print("[TTS] ✅ Using espeak TTS (fallback)")
                return
        except:
            pass
        
        # Try festival
        try:
            result = subprocess.run(['which', 'festival'], capture_output=True, text=True)
            if result.returncode == 0:
                self.tts_backend = "festival"
                print("[TTS] ✅ Using festival TTS (fallback)")
                return
        except:
            pass
        
        # Use system say command (macOS) or Windows SAPI
        import platform
        system = platform.system()
        
        if system == "Darwin":  # macOS
            self.tts_backend = "say"
            print("[TTS] ✅ Using macOS say command (fallback)")
        elif system == "Windows":
            self.tts_backend = "sapi"
            print("[TTS] ✅ Using Windows SAPI (fallback)")
        else:
            # Last resort - no TTS, just print
            self.tts_backend = "print"
            print("[TTS] ⚠️  No TTS backend available, will print text only")
    
    def process_synthesis(self, text_queue, audio_queue, stop_event, timings, speech_speed_factor, output_dir):
        """Process text synthesis and queue audio chunks."""
        total_words_synthesized = 0
        first_audio_chunk_generated = False
        
        try:
            while not stop_event.is_set():
                try:
                    text_to_speak = text_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if text_to_speak is None:
                    break
                
                if not text_to_speak.strip():
                    text_queue.task_done()
                    continue
                
                cleaned_text = self._clean_text_for_tts(text_to_speak)
                word_count = len(cleaned_text.strip().split())
                total_words_synthesized += word_count
                
                # Generate audio
                start_time = time.time()
                audio_data = self._synthesize_text(cleaned_text, speech_speed_factor)
                generation_time = time.time() - start_time
                
                if not first_audio_chunk_generated:
                    timings['tts_first_audio_chunk_generated_time'] = time.time()
                    timings['tts_first_chunk_word_count'] = word_count
                    first_audio_chunk_generated = True
                
                if audio_data is not None:
                    # Queue audio for playback
                    audio_queue.put(audio_data)
                    
                    # Save to file if requested
                    if output_dir:
                        self._save_audio_file(audio_data, output_dir)
                
                print(f"[TTS] Synthesized: \"{cleaned_text[:50]}{'...' if len(cleaned_text) > 50 else ''}\" ({word_count} words, {generation_time:.2f}s)")
                text_queue.task_done()
            
            timings['audio_words_synthesized'] = total_words_synthesized
            timings['tts_generation_duration'] = generation_time if 'generation_time' in locals() else 0
            
        except Exception as e:
            cprint(f"[TTS] Error: {e}", "red")
            traceback.print_exc()
            stop_event.set()
        finally:
            audio_queue.put(None)  # Signal end of audio
            if 'tts_last_audio_chunk_generated_time' not in timings:
                timings['tts_last_audio_chunk_generated_time'] = time.time()
    
    def _clean_text_for_tts(self, text):
        """Clean text for TTS synthesis."""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*+([^*]*?)\*+', r'\1', text)  # **bold**, ***italic***
        text = re.sub(r'_+([^_]*?)_+', r'\1', text)    # _italic_, __bold__
        text = re.sub(r'`+([^`]*?)`+', r'\1', text)    # `code`
        text = re.sub(r'#{1,6}\s*', '', text)          # Headers
        text = re.sub(r'\[([^\]]*?)\]\([^)]*?\)', r'\1', text)  # [text](url)
        
        # Clean up whitespace
        text = re.sub(r'\n{2,}', '. ', text)           # Multiple newlines
        text = re.sub(r'\s{2,}', ' ', text)            # Multiple spaces
        
        return text.strip()
    
    def _synthesize_text(self, text, speed_factor=1.0):
        """Synthesize text using the configured backend."""
        try:
            if self.tts_backend == "piper":
                return self._synthesize_piper(text, speed_factor)
            elif self.tts_backend == "espeak":
                return self._synthesize_espeak(text, speed_factor)
            elif self.tts_backend == "festival":
                return self._synthesize_festival(text, speed_factor)
            elif self.tts_backend == "say":
                return self._synthesize_say(text, speed_factor)
            elif self.tts_backend == "sapi":
                return self._synthesize_sapi(text, speed_factor)
            else:
                # Print fallback
                cprint(f"[TTS] 🔊 {text}", "blue")
                return self._generate_silence(len(text.split()) * 0.3)  # Fake audio duration
                
        except Exception as e:
            cprint(f"[TTS] Synthesis error: {e}", "red")
            return self._generate_silence(1.0)  # 1 second of silence as fallback
    
    def _synthesize_piper(self, text, speed_factor):
        """Synthesize using Piper TTS."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            cmd = ['piper', '--model', self.model_path, '--output_file', temp_path]
            
            if self.speaker_id:
                cmd.extend(['--speaker', str(self.speaker_id)])
            
            # Piper uses length_scale for speed (inverse of speed)
            if speed_factor != 1.0:
                length_scale = 1.0 / speed_factor
                cmd.extend(['--length_scale', str(length_scale)])
            
            # Run piper
            process = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if process.returncode != 0:
                raise Exception(f"Piper failed: {process.stderr}")
            
            # Load generated audio
            return self._load_wav_file(temp_path)
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _synthesize_espeak(self, text, speed_factor):
        """Synthesize using espeak."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Calculate espeak speed (words per minute)
            base_speed = 175  # Default espeak speed
            espeak_speed = int(base_speed * speed_factor)
            
            cmd = [
                'espeak',
                '-w', temp_path,
                '-s', str(espeak_speed),
                '-a', '100',  # Amplitude
                text
            ]
            
            process = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if process.returncode != 0:
                raise Exception(f"espeak failed: {process.stderr}")
            
            return self._load_wav_file(temp_path)
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _synthesize_festival(self, text, speed_factor):
        """Synthesize using Festival TTS."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create festival script
            festival_script = f'''
(voice_default)
(Parameter.set 'Duration_Stretch {1.0 / speed_factor})
(utt.save.wave (utt.synth (Utterance Text "{text}")) "{temp_path}")
'''
            
            process = subprocess.run(
                ['festival'],
                input=festival_script,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if process.returncode != 0:
                raise Exception(f"Festival failed: {process.stderr}")
            
            return self._load_wav_file(temp_path)
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _synthesize_say(self, text, speed_factor):
        """Synthesize using macOS say command."""
        with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Calculate say rate (words per minute)
            base_rate = 200
            say_rate = int(base_rate * speed_factor)
            
            cmd = ['say', '-o', temp_path, '-r', str(say_rate), text]
            
            process = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if process.returncode != 0:
                raise Exception(f"say command failed: {process.stderr}")
            
            # Convert AIFF to WAV if needed
            wav_path = temp_path.replace('.aiff', '.wav')
            subprocess.run(['ffmpeg', '-i', temp_path, wav_path], 
                         capture_output=True, check=True)
            
            return self._load_wav_file(wav_path)
            
        finally:
            for path in [temp_path, temp_path.replace('.aiff', '.wav')]:
                try:
                    os.unlink(path)
                except:
                    pass
    
    def _synthesize_sapi(self, text, speed_factor):
        """Synthesize using Windows SAPI."""
        try:
            import win32com.client
            
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            file_stream.Open(temp_path, 3)
            speaker.AudioOutputStream = file_stream
            
            # Set rate (range -10 to 10, default 0)
            rate = max(-10, min(10, int((speed_factor - 1.0) * 5)))
            speaker.Rate = rate
            
            speaker.Speak(text)
            file_stream.Close()
            
            return self._load_wav_file(temp_path)
            
        except ImportError:
            raise Exception("Windows SAPI requires pywin32: pip install pywin32")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _load_wav_file(self, file_path):
        """Load a WAV file and return audio data."""
        try:
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(file_path)
            
            # Convert to int16 if needed
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            return audio_data
            
        except Exception as e:
            print(f"[TTS] Error loading WAV file: {e}")
            return None
    
    def _generate_silence(self, duration_seconds):
        """Generate silence audio data."""
        samples = int(self.sample_rate * duration_seconds)
        return np.zeros(samples, dtype=np.int16)
    
    def _save_audio_file(self, audio_data, output_dir):
        """Save audio data to file."""
        try:
            from scipy.io import wavfile
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"tts_output_{self.audio_file_counter:03d}.wav"
            file_path = output_path / filename
            
            wavfile.write(str(file_path), self.sample_rate, audio_data)
            self.audio_file_counter += 1
            
        except Exception as e:
            print(f"[TTS] Error saving audio file: {e}") 
