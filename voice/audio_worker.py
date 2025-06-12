# voice/audio_worker.py
"""
Audio playback worker for Opi Voice Assistant - FIXED for USB headphones
"""

import sounddevice as sd
import time
import traceback
import queue
from termcolor import cprint
import numpy as np
from scipy.signal import resample


class AudioWorker:
    """Handles audio playback from queue - FIXED for USB headphones."""
    
    def __init__(self, device_index=None, sample_rate=22050, chunk_pause_ms=0):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_pause_ms = chunk_pause_ms
        
        # Get device info for validation
        self.device_info = None
        if device_index is not None:
            try:
                self.device_info = sd.query_devices(device_index)
                cprint(f"[Audio] Using device [{device_index}]: {self.device_info['name']}", "green")
                cprint(f"[Audio] Device sample rate: {self.device_info['default_samplerate']}Hz", "white")
                cprint(f"[Audio] Device channels: {self.device_info['max_output_channels']}", "white")
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
                # Use device's preferred sample rate if significantly different
                if device_sr in [44100, 48000] and self.sample_rate == 22050:
                    target_sample_rate = device_sr
                    cprint(f"[Audio] Using device sample rate: {target_sample_rate}Hz", "cyan")
            
            # Initialize audio stream with FIXED device parameter
            stream_params = {
                'samplerate': target_sample_rate,
                'channels': 1,
                'dtype': 'int16'
            }
            
            # CRITICAL FIX: Actually use the device_index!
            if self.device_index is not None:
                stream_params['device'] = self.device_index
            
            audio_stream = sd.OutputStream(**stream_params)
            audio_stream.start()
            
            cprint(f"[Audio] ✅ Audio stream started on device {self.device_index or 'default'}", "green")
            cprint(f"[Audio] Stream config: {target_sample_rate}Hz, 1ch, int16", "white")
            
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if audio_chunk is None:
                    cprint("[Audio] Received end-of-stream signal", "yellow")
                    break
                
                if not first_chunk_played:
                    timings['playback_first_audio_chunk_played_time'] = time.time()
                    first_chunk_played = True
                    cprint("[Audio] 🔊 Starting audio playback", "blue")
                
                # Add pause between chunks (but not before the first chunk)
                if chunk_count > 0 and self.chunk_pause_ms > 0:
                    pause_seconds = self.chunk_pause_ms / 1000.0
                    if not stop_event.is_set():
                        time.sleep(pause_seconds)
                
                if stop_event.is_set():
                    break
                
                # Process audio chunk
                try:
                    if isinstance(audio_chunk, np.ndarray):
                        # CRITICAL FIX: Handle sample rate conversion
                        processed_chunk = self._process_audio_chunk(
                            audio_chunk, 
                           self.sample_rate, 
                            target_sample_rate
                        )
                        
                        if processed_chunk is not None:
                            # CRITICAL FIX: Ensure audio is loud enough
                            processed_chunk = self._normalize_volume(processed_chunk)
                            
                            # Write to stream
                            audio_stream.write(processed_chunk)
                            
                            # Calculate duration
                            duration = len(processed_chunk) / target_sample_rate
                            total_duration_played += duration
                            chunk_count += 1
                            
                            cprint(f"[Audio] Played chunk {chunk_count} ({duration:.2f}s)", "white")
                        else:
                            cprint(f"[Audio] Warning: Could not process audio chunk", "yellow")
                    else:
                        cprint(f"[Audio] Warning: Unexpected audio data type: {type(audio_chunk)}", "yellow")
                        
                except Exception as e:
                    cprint(f"[Audio] Error playing chunk: {e}", "red")
                
                try:
                    audio_queue.task_done()
                except:
                    pass  # task_done() might not be available on all queue types
            
            timings['audio_duration_played'] = total_duration_played
            
        except sd.PortAudioError as pae:
            self._handle_audio_error(pae, stop_event)
        except Exception as e:
            cprint(f"[Audio] Error: {e}", "red")
            traceback.print_exc()
            stop_event.set()
        finally:
            self._cleanup_stream(audio_stream, stop_event, total_duration_played, target_sample_rate)
            timings['playback_last_audio_chunk_played_time'] = time.time()
            if first_chunk_played:
                cprint("[Audio] ✅ Audio playback complete", "green")
    
    def _process_audio_chunk(self, audio_chunk, source_sr, target_sr):
        """Process audio chunk with sample rate conversion if needed."""
        try:
            # Ensure audio is int16
            if audio_chunk.dtype != np.int16:
                if audio_chunk.dtype in [np.float32, np.float64]:
                    # Convert float to int16
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_chunk = audio_chunk.astype(np.int16)
            
            # Handle sample rate conversion
            if source_sr != target_sr:
                # Convert to float for resampling
                audio_float = audio_chunk.astype(np.float32) / 32767.0
                
                # Resample
                resampled = resample(audio_float, int(len(audio_float) * target_sr / source_sr))
                
                # Convert back to int16
                audio_chunk = (resampled * 32767).astype(np.int16)
                
                cprint(f"[Audio] Resampled {source_sr}Hz -> {target_sr}Hz", "cyan")
            
            # Ensure mono
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk[:, 0]
            
            return audio_chunk
            
        except Exception as e:
            cprint(f"[Audio] Error processing audio chunk: {e}", "red")
            return None
    
    def _normalize_volume(self, audio_chunk, target_volume=0.7):
        """Normalize audio volume to ensure it's audible."""
        try:
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            if rms > 0:
                # Target RMS for good volume (not too loud, not too quiet)
                target_rms = target_volume * 32767 * 0.3  # 30% of max volume
                
                # Calculate gain needed
                gain = target_rms / rms
                
                # Apply gain but prevent clipping
                gained_audio = audio_chunk.astype(np.float32) * gain
                
                # Clip to prevent distortion
                gained_audio = np.clip(gained_audio, -32767, 32767)
                
                return gained_audio.astype(np.int16)
            else:
                return audio_chunk
                
        except Exception as e:
            cprint(f"[Audio] Error normalizing volume: {e}", "red")
            return audio_chunk
    
    def _handle_audio_error(self, error, stop_event):
        """Handle audio device errors."""
        cprint(f"[Audio] PortAudio Error: {error}", "red")
        cprint("Available audio devices:", "yellow")
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    marker = " ← YOUR DEVICE" if i == self.device_index else ""
                    cprint(f"  [{i:2d}] {device['name']} "
                          f"(Channels: {device['max_output_channels']}, "
                          f"Sample Rate: {device['default_samplerate']}Hz){marker}", "yellow")
        except Exception as e:
            cprint(f"  Could not query devices: {e}", "red")
        
        cprint("💡 Try: python main.py --list-devices", "cyan")
        cprint("💡 Fix: Set correct audio device in config.json", "cyan")
        cprint("💡 Check: USB headphones volume and connection", "cyan")
        stop_event.set()
    
    def _cleanup_stream(self, stream, stop_event, duration_played, sample_rate):
        """Clean up audio stream properly."""
        if not stream:
            return
        
        try:
            if not stream.closed:
                # CRITICAL FIX: Wait for audio buffer to empty
                if not stop_event.is_set() and duration_played > 0:
                    # Calculate proper wait time based on stream latency
                    latency = getattr(stream, 'latency', 0.1)
                    buffer_time = max(latency, 0.1)  # At least 100ms
                    
                    cprint(f"[Audio] Waiting {buffer_time:.1f}s for audio buffer to empty...", "cyan")
                    time.sleep(buffer_time)
                
                stream.stop()
                stream.close()
                cprint("[Audio] ✅ Audio stream closed cleanly", "green")
                
        except Exception as e:
            cprint(f"[Audio] Error closing stream: {e}", "red")


def test_fixed_audio_playback():
    """Test the fixed audio playback functionality."""
    cprint("[Audio] Testing FIXED audio playback on device 4...", "cyan", attrs=['bold'])
    
    try:
        import threading
        
        # Generate test audio
        duration = 3.0  # seconds
        sample_rate = 22050
        frequency = 440  # Hz (A4 note)
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        # Create fixed audio worker
        worker = AudioWorker(device_index=4, sample_rate=sample_rate)
        
        # Create test queue and events
        test_queue = queue.Queue()
        stop_event = threading.Event()
        timings = {}
        
        # Split audio into chunks to simulate real usage
        chunk_size = 4410  # 0.2 seconds per chunk
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            test_queue.put(chunk)
        
        test_queue.put(None)  # End signal
        
        cprint("[Audio] Starting playback test...", "yellow")
        
        # Start playback in separate thread
        playback_thread = threading.Thread(
            target=worker.process_playback,
            args=(test_queue, stop_event, timings)
        )
        playback_thread.start()
        
        # Wait for completion
        playback_thread.join(timeout=10.0)
        
        if playback_thread.is_alive():
            cprint("[Audio] ❌ Test timeout", "red")
            stop_event.set()
            return False
        else:
            cprint("[Audio] ✅ Test completed successfully!", "green")
            if 'audio_duration_played' in timings:
                cprint(f"[Audio] Total audio played: {timings['audio_duration_played']:.2f}s", "white")
            return True
            
    except Exception as e:
        cprint(f"[Audio] ❌ Test failed: {e}", "red")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run fixed audio worker test if called directly."""
    cprint("🔊 Testing Fixed AudioWorker", "cyan", attrs=['bold'])
    cprint("=" * 50, "cyan")
    
    # Run the test
    test_fixed_audio_playback()
    
    cprint("\n🔧 If you still don't hear anything:", "yellow")
    cprint("1. Check USB headphones are plugged in and powered on", "white")
    cprint("2. Run: alsamixer (check volumes and mute status)", "white") 
    cprint("3. Run: pavucontrol (check PulseAudio settings)", "white")
    cprint("4. Test system audio: speaker-test -D hw:4 -c2", "white")
    cprint("5. Check USB device: lsusb | grep -i audio", "white") 
