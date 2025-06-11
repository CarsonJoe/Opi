# voice/audio_worker.py
"""
Audio playback worker for Opi Voice Assistant
"""

import sounddevice as sd
import time
import traceback
import queue
from termcolor import cprint
import numpy as np


class AudioWorker:
    """Handles audio playback from queue."""
    
    def __init__(self, device_index=None, sample_rate=22050, chunk_pause_ms=0):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_pause_ms = chunk_pause_ms
        
    def process_playback(self, audio_queue, stop_event, timings):
        """Process audio playback from queue."""
        audio_stream = None
        total_duration_played = 0.0
        first_chunk_played = False
        chunk_count = 0
        
        try:
            # Initialize audio stream
            audio_stream = sd.OutputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            audio_stream.start()
            print("[Audio] Audio stream started")
            
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
                
                # Play audio chunk
                try:
                    if isinstance(audio_chunk, np.ndarray):
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
        """Handle audio device errors."""
        cprint(f"[Audio] PortAudio Error: {error}", "red")
        cprint("Available audio devices:", "yellow")
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    cprint(f"  [{i:2d}] {device['name']} "
                          f"(Channels: {device['max_output_channels']}, "
                          f"Sample Rate: {device['default_samplerate']}Hz)", "yellow")
        except Exception as e:
            cprint(f"  Could not query devics: {e}", "red")
        
        cprint("💡 Try: python main.py --list-devices", "cyan")
        cprint("💡 Fix: Set correct audio device in config.json", "cyan")
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
                    wait_time_ms = int(stream.latency * 1000) + 200
                    sd.sleep(wait_time_ms)
                
                stream.stop()
                stream.close()
                print("[Audio] Audio stream closed")
        except Exception as e:
            cprint(f"[Audio] Error closing stream: {e}", "red")


def test_audio_playback():
    """Test audio playback functionality."""
    print("[Audio] Testing audio playback...")
    
    try:
        # Generate a test tone
        duration = 2.0  # seconds
        frequency = 440  # Hz (A4 note)
        sample_rate = 22050
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        # Create audio worker
        worker = AudioWorker()
        
        # Create test queue and events
        test_queue = queue.Queue()
        stop_event = threading.Event()
        timings = {}
        
        # Add test audio to queue
        test_queue.put(test_audio)
        test_queue.put(None)  # End signal
        
        # Start playback in separate thread
        playback_thread = threading.Thread(
            target=worker.process_playback,
            args=(test_queue, stop_event, timings)
        )
        playback_thread.start()
        
        # Wait for completion
        playback_thread.join(timeout=5.0)
        
        if playback_thread.is_alive():
            print("[Audio] ❌ Test timeout")
            stop_event.set()
            return False
        else:
            print("[Audio] ✅ Test completed successfully")
            if 'audio_duration_played' in timings:
                print(f"[Audio] Played {timings['audio_duration_played']:.2f}s of audio")
            return True
            
    except Exception as e:
        print(f"[Audio] ❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run audio worker test if called directly."""
    test_audio_playback()
