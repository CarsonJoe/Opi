# voice/audio_worker.py
"""
Audio playback worker for Opi Voice Assistant
Fixed version with proper device selection and error handling
"""

import sounddevice as sd
import time
import traceback
import queue
import threading
from termcolor import cprint
import numpy as np


class AudioWorker:
    """Handles audio playback from queue with proper device selection."""
    
    def __init__(self, device_index=None, sample_rate=22050, chunk_pause_ms=0):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_pause_ms = chunk_pause_ms
        
        # Validate and log device selection
        if self.device_index is not None:
            try:
                device_info = sd.query_devices(self.device_index)
                if device_info['max_output_channels'] == 0:
                    cprint(f"⚠️  Device {self.device_index} has no output channels, using default", "yellow")
                    self.device_index = None
                else:
                    cprint(f"🔊 Using audio device {self.device_index}: {device_info['name']}", "green")
            except Exception as e:
                cprint(f"⚠️  Error with device {self.device_index}, using default: {e}", "yellow")
                self.device_index = None
        else:
            cprint("🔊 Using default audio device", "cyan")
        
    def process_playback(self, audio_queue, stop_event, timings):
        """Process audio playback from queue."""
        audio_stream = None
        total_duration_played = 0.0
        first_chunk_played = False
        chunk_count = 0
        
        try:
            # Initialize audio stream with explicit device
            stream_kwargs = {
                'samplerate': self.sample_rate,
                'channels': 1,
                'dtype': 'int16'
            }
            
            if self.device_index is not None:
                stream_kwargs['device'] = self.device_index
                
            audio_stream = sd.OutputStream(**stream_kwargs)
            audio_stream.start()
            
            # Log which device we're actually using
            device_name = "default"
            if self.device_index is not None:
                try:
                    device_info = sd.query_devices(self.device_index)
                    device_name = f"device {self.device_index} ({device_info['name']})"
                except:
                    device_name = f"device {self.device_index}"
                    
            print(f"[Audio] Audio stream started on {device_name}")
            
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if audio_chunk is None:
                    print("[Audio] Received end-of-stream signal")
                    break
                
                if not first_chunk_played:
                    timings['playback_first_audio_chunk_played_time'] = time.time()
                    first_chunk_played = True
                    print("[Audio] 🔊 Starting audio playback")
                
                # Add pause between chunks (but not before the first chunk)
                if chunk_count > 0 and self.chunk_pause_ms > 0:
                    pause_seconds = self.chunk_pause_ms / 1000.0
                    if not stop_event.is_set():
                        time.sleep(pause_seconds)
                
                if stop_event.is_set():
                    break
                
                # Play audio chunk with proper data conversion
                try:
                    if isinstance(audio_chunk, np.ndarray):
                        # Ensure correct data type
                        if audio_chunk.dtype != np.int16:
                            if audio_chunk.dtype in [np.float32, np.float64]:
                                # Convert float to int16 (assuming range -1.0 to 1.0)
                                audio_chunk = (np.clip(audio_hunk, -1.0, 1.0) * 32767).astype(np.int16)
                            else:
                                audio_chunk = audio_chunk.astype(np.int16)
                        
                        # Ensure 1D array (mono)
                        if len(audio_chunk.shape) > 1:
                            audio_chunk = audio_chunk.flatten()
                        
                        # Write to audio stream
                        audio_stream.write(audio_chunk)
                        total_duration_played += len(audio_chunk) / self.sample_rate
                        chunk_count += 1
                        
                    else:
                        print(f"[Audio] Warning: Unexpected audio data type: {type(audio_chunk)}")
                        
                except Exception as e:
                    print(f"[Audio] Error playing chunk: {e}")
                
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
            self._cleanup_stream(audio_stream, stop_event, total_duration_played)
            timings['playback_last_audio_chunk_played_time'] = time.time()
            if first_chunk_played:
                print("[Audio] ✅ Audio playback complete")
    
    def _handle_audio_error(self, error, stop_event):
        """Handle audio device errors with helpful information."""
        cprint(f"[Audio] PortAudio Error: {error}", "red")
        cprint("Available audio output devices:", "yellow")
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    marker = " ← YOUR HEADPHONES" if i == 4 else ""
                    default_marker = " (DEFAULT)" if i == sd.default.device[1] else ""
                    cprint(f"  [{i:2d}] {device['name']}"
                          f"{default_marker}{marker}", "yellow")
                    cprint(f"       Channels: {device['max_output_channels']}, "
                          f"Sample Rate: {device['default_samplerate']}Hz", "white")
        except Exception as e:
            cprint(f"  Could not query devices: {e}", "red")
        
        cprint("\n💡 Troubleshooting tips:", "cyan")
        cprint("  1. Check headphone connection", "white")
        cprint("  2. Try: python test_audio_device.py", "white")
        cprint("  3. Update audio_device in config.json", "white")
        stop_event.set()
    
    def _cleanup_stream(self, stream, stop_event, duration_played):
        """Clean up audio stream properly."""
        if not stream:
            return
        
        try:
            if not stream.closed:
                # Wait for audio to finish if normal completion
                if not stop_event.is_set() and duration_played > 0:
                    # Add small delay to ensure audio completes
                    wait_time_ms = max(100, int(stream.latency * 1000) + 100)
                    sd.sleep(wait_time_ms)
                
                stream.stop()
                stream.close()
                print("[Audio] Audio stream closed")
        except Exception as e:
            cprint(f"[Audio] Error closing stream: {e}", "red")


def test_audio_playback_with_device(device_id=4):
    """Test audio playback functionality with specific device."""
    print(f"[Audio] Testing audio playback on device {device_id}...")
    
    try:
        # Check if device exists and is suitable
        try:
            device_info = sd.query_devices(device_id)
            if device_info['max_output_channels'] == 0:
                print(f"[Audio] ❌ Device {device_id} has no output channels")
                return False
            print(f"[Audio] Testing device {device_id}: {device_info['name']}")
        except Exception as e:
            print(f"[Audio] ❌ Device {device_id} not available: {e}")
            return False
        
        # Generate a test tone
        duration = 2.0  # seconds
        frequency = 440  # Hz (A4 note)
        sample_rate = 22050
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        # Create audio worker with specific device
        worker = AudioWorker(device_index=device_id, sample_rate=sample_rate)
        
        # Create test queue and events
        test_queue = queue.Queue()
        stop_event = threading.Event()
        timings = {}
        
        # Add test audio to queue
        test_queue.put(test_audio)
        test_queue.put(None)  # End signal
        
        print(f"[Audio] Playing test tone on device {device_id}...")
        
        # Start playback in separate thread
        playback_thread = threading.Thread(
            target=worker.process_playback,
            args=(test_queue, stop_event, timings)
        )
        playback_thread.start()
        
        # Wait for completion
        playback_thread.join(timeout=10.0)
        
        if playback_thread.is_alive():
            print("[Audio] ❌ Test timeout")
            stop_event.set()
            playback_thread.join(timeout=2.0)
            return False
        else:
            print("[Audio] ✅ Test completed successfully")
            if 'audio_duration_played' in timings:
                print(f"[Audio] Played {timings['audio_duration_played']:.2f}s of audio")
            return True
            
    except Exception as e:
        print(f"[Audio] ❌ Test failed: {e}")
        traceback.print_exc()
        return False


def test_audio_playback():
    """Test audio playback functionality with default device."""
    return test_audio_playback_with_device(None)


if __name__ == "__main__":
    """Run audio worker test if called directly."""
    cprint("🎧 Audio Worker Test", "cyan", attrs=['bold'])
    cprint("=" * 30, "cyan")
    
    # Test device 4 first
    print("\n1. Testing device 4 (your headphones):")
    success_4 = test_audio_playback_with_device(4)
    
    if not success_4:
        print("\n2. Testing default device:")
        success_default = test_audio_playback()
        
        if not success_default:
            cprint("\n❌ All audio tests failed!", "red")
            cprint("💡 Check your audio setup and device connections", "yellow")
        else:
            cprint("\n⚠️  Default device works, but device 4 doesn't", "yellow")
            cprint("💡 Update config.json to use default device (remove audio_device setting)", "yellow")
    else:
        cprint(f"\n🎉 Device 4 is working perfectly!", "green")
        cprint("💡 Your headphones should work with Opi now", "cyan")
