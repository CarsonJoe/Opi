# voice/audio_worker.py
"""
Audio playback worker for Opi Voice Assistant - SIMPLIFIED LOGGING
"""

import sounddevice as sd
import time
import traceback
import queue
from termcolor import cprint
import numpy as np
from scipy.signal import resample


class AudioWorker:
    """Handles audio playback from queue with simplified logging."""
    
    def __init__(self, device_index=None, sample_rate=22050, chunk_pause_ms=0, verbose=False):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_pause_ms = chunk_pause_ms
        self.verbose = verbose
        
        # Get device info for validation
        self.device_info = None
        if device_index is not None:
            try:
                self.device_info = sd.query_devices(device_index)
                cprint(f"[Audio] Using device [{device_index}]: {self.device_info['name']}", "green")
            except Exception as e:
                cprint(f"[Audio] Warning: Could not query device {device_index}: {e}", "yellow")
        
    def process_playback(self, audio_queue, stop_event, timings):
        """Process audio playback from queue."""
        audio_stream = None
        total_duration_played = 0.0
        first_chunk_played = False
        chunk_count = 0
        
        try:
            # Determine optimal sample rate
            target_sample_rate = self.sample_rate
            if self.device_info:
                device_sr = int(self.device_info['default_samplerate'])
                if device_sr in [44100, 48000] and self.sample_rate == 22050:
                    target_sample_rate = device_sr
            
            # Initialize audio stream
            stream_params = {
                'samplerate': target_sample_rate,
                'channels': 1,
                'dtype': 'int16'
            }
            
            if self.device_index is not None:
                stream_params['device'] = self.device_index
            
            audio_stream = sd.OutputStream(**stream_params)
            audio_stream.start()
            
            if self.verbose:
                cprint(f"[Audio] Audio stream started", "green")
            
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if audio_chunk is None:
                    break
                
                if not first_chunk_played:
                    timings['playback_first_audio_chunk_played_time'] = time.time()
                    first_chunk_played = True
                    cprint("[Audio] 🔊 Playing audio", "blue")
                
                # Add pause between chunks
                if chunk_count > 0 and self.chunk_pause_ms > 0:
                    pause_seconds = self.chunk_pause_ms / 1000.0
                    if not stop_event.is_set():
                        time.sleep(pause_seconds)
                
                if stop_event.is_set():
                    break
                
                # Process audio chunk
                try:
                    if isinstance(audio_chunk, np.ndarray):
                        processed_chunk = self._process_audio_chunk(
                            audio_chunk, 
                            self.sample_rate, 
                            target_sample_rate
                        )
                        
                        if processed_chunk is not None:
                            processed_chunk = self._normalize_volume(processed_chunk)
                            audio_stream.write(processed_chunk)
                            
                            duration = len(processed_chunk) / target_sample_rate
                            total_duration_played += duration
                            chunk_count += 1
                            
                            if self.verbose:
                                cprint(f"[Audio] Played chunk {chunk_count} ({duration:.2f}s)", "white")
                        elif self.verbose:
                            cprint(f"[Audio] Could not process audio chunk", "yellow")
                    elif self.verbose:
                        cprint(f"[Audio] Unexpected audio data type: {type(audio_chunk)}", "yellow")
                        
                except Exception as e:
                    if self.verbose:
                        cprint(f"[Audio] Error playing chunk: {e}", "red")
                
                try:
                    audio_queue.task_done()
                except:
                    pass
            
            timings['audio_duration_played'] = total_duration_played
            
        except sd.PortAudioError as pae:
            self._handle_audio_error(pae, stop_event)
        except Exception as e:
            cprint(f"[Audio] Error: {e}", "red")
            if self.verbose:
                traceback.print_exc()
            stop_event.set()
        finally:
            self._cleanup_stream(audio_stream, stop_event, total_duration_played, target_sample_rate)
            timings['playback_last_audio_chunk_played_time'] = time.time()
            if first_chunk_played:
                if self.verbose:
                    cprint("[Audio] ✅ Audio complete", "green")
    
    def _process_audio_chunk(self, audio_chunk, source_sr, target_sr):
        """Process audio chunk with sample rate conversion if needed."""
        try:
            # Ensure audio is int16
            if audio_chunk.dtype != np.int16:
                if audio_chunk.dtype in [np.float32, np.float64]:
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_chunk = audio_chunk.astype(np.int16)
            
            # Handle sample rate conversion
            if source_sr != target_sr:
                audio_float = audio_chunk.astype(np.float32) / 32767.0
                resampled = resample(audio_float, int(len(audio_float) * target_sr / source_sr))
                audio_chunk = (resampled * 32767).astype(np.int16)
            
            # Ensure mono
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk[:, 0]
            
            return audio_chunk
            
        except Exception as e:
            if self.verbose:
                cprint(f"[Audio] Error processing audio chunk: {e}", "red")
            return None
    
    def _normalize_volume(self, audio_chunk, target_volume=0.7):
        """Normalize audio volume to ensure it's audible."""
        try:
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            if rms > 0:
                target_rms = target_volume * 32767 * 0.3
                gain = target_rms / rms
                gained_audio = audio_chunk.astype(np.float32) * gain
                gained_audio = np.clip(gained_audio, -32767, 32767)
                return gained_audio.astype(np.int16)
            else:
                return audio_chunk
                
        except Exception as e:
            if self.verbose:
                cprint(f"[Audio] Error normalizing volume: {e}", "red")
            return audio_chunk
    
    def _handle_audio_error(self, error, stop_event):
        """Handle audio device errors."""
        cprint(f"[Audio] PortAudio Error: {error}", "red")
        cprint("💡 Try: python main.py --list-devices", "cyan")
        stop_event.set()
    
    def _cleanup_stream(self, stream, stop_event, duration_played, sample_rate):
        """Clean up audio stream properly."""
        if not stream:
            return
        
        try:
            if not stream.closed:
                if not stop_event.is_set() and duration_played > 0:
                    latency = getattr(stream, 'latency', 0.1)
                    buffer_time = max(latency, 0.1)
                    time.sleep(buffer_time)
                
                stream.stop()
                stream.close()
                
        except Exception as e:
            if self.verbose:
                cprint(f"[Audio] Error closing stream: {e}", "red") 
