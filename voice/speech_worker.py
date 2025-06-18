# voice/speech_worker.py
"""
Speech recognition worker for Opi Voice Assistant
Adapted from the original chat/workers/speech_worker.py
"""

import os
import time
import queue
import threading
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly
import numpy as np
from faster_whisper import WhisperModel
from collections import deque
from termcolor import cprint
import traceback
from pathlib import Path


class SpeechWorker:
    """Worker for speech-to-text transcription with smart chunking and utterance aggregation."""

    def __init__(self, model_size="tiny.en", compute_type="int8", verbose=False):
        # Audio parameters
        self.target_sample_rate = 16000
        self.min_chunk_duration = 0.0
        self.max_chunk_duration = 8.0
        self.channels = 1
        self.verbose = verbose

        # VAD Parameters
        self.silence_threshold = 200.0
        self.silence_duration = 0.4  # For individual chunk segmentation
        self.buffer_size = 0.05  # 100ms buffers
        self.pre_record_duration = 0.5

        # Utterance aggregation parameter
        self.end_of_utterance_silence_duration = 1.0

        # Speech detection for filtering silent chunks
        self.min_speech_ratio = 0.1

        # Initialize model
        self.model = WhisperModel(model_size, compute_type=compute_type)
        print(f"[Speech] ✅ Whisper model loaded: {model_size}")

        self.audio_queue = queue.Queue()
        self.pending_transcripts = []
        self.last_transcript_time = None
        self.transcript_agg_lock = threading.Lock()
        self.current_utterance_actual_speech_end_time = None

        self.mic_sample_rate = self._get_input_sample_rate()

        # Create audio chunks directory
        self.audio_dir = Path("data/audio_chunks")
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _get_input_sample_rate(self):
        """Get the default input sample rate."""
        try:
            default_device = sd.query_devices(sd.default.device[0], 'input')
            sr = int(default_device['default_samplerate'])
            return sr
        except Exception as e:
            print(f"[Speech] Could not detect input sample rate: {e}. Defaulting to 44100 Hz.")
            return 44100

    def _calculate_rms(self, audio_chunk):
        """Calculate RMS (Root Mean Square) of audio chunk."""
        if len(audio_chunk) == 0:
            return 0
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    def _detect_speech_gap(self, rms_buffer, threshold, min_duration_buffers):
        """Detect if there's a speech gap (silence) in recent buffers."""
        if len(rms_buffer) < min_duration_buffers:
            return False
        recent_rms = list(rms_buffer)[-min_duration_buffers:]
        return all(rms < threshold for rms in recent_rms)

    def _has_sufficient_speech(self, rms_buffer, threshold, min_ratio):
        """Check if the buffer has sufficient speech content."""
        if len(rms_buffer) == 0:
            return False
        speech_buffers = sum(1 for rms in rms_buffer if rms > threshold)
        return (speech_buffers / len(rms_buffer)) >= min_ratio

    def process_speech_input(self, final_speech_queue, stop_event, timings):
        """Main method to process speech input."""
        try:
            with self.transcript_agg_lock:
                self.pending_transcripts.clear()
                self.last_transcript_time = None
                self.current_utterance_actual_speech_end_time = None

            # Clear any existing audio queue items
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Start transcription aggregator thread
            transcription_aggregator_thread = threading.Thread(
                target=self._transcription_aggregator_loop,
                args=(final_speech_queue, stop_event, timings),
                daemon=True,
               name="TranscriptionAggregatorThread"
            )
            transcription_aggregator_thread.start()

            # Start recording loop
            self._record_loop(stop_event, timings)

            print("[Speech] Record loop finished. Waiting for aggregator thread to complete.")
            join_timeout = self.end_of_utterance_silence_duration + self.max_chunk_duration + 2.0
            transcription_aggregator_thread.join(timeout=join_timeout)
            
            if transcription_aggregator_thread.is_alive():
                print("[Speech] Aggregator thread did not exit cleanly after record loop.")

        except Exception as e:
            print(f"[Speech] Error in process_speech_input: {e}")
            traceback.print_exc()
            stop_event.set()
        finally:
            if not stop_event.is_set():
                stop_event.set()
            print("[Speech] process_speech_input finished.")

    def _record_loop(self, stop_event, timings):
        """Main recording loop that captures and processes audio."""
        chunk_id = 0
        audio_buffer = deque()
        rms_buffer = deque()
        pre_record_buffers = int(self.pre_record_duration / self.buffer_size)
        pre_audio_buffer = deque(maxlen=pre_record_buffers)
        pre_rms_buffer = deque(maxlen=pre_record_buffers)
        chunk_start_time = None
        buffer_samples = int(self.buffer_size * self.mic_sample_rate)
        min_silence_buffers = int(self.silence_duration / self.buffer_size)
        is_recording = False

        try:
            while not stop_event.is_set():
                try:
                    # Record audio chunk
                    audio_chunk = sd.rec(
                        buffer_samples,
                        samplerate=self.mic_sample_rate,
                        channels=self.channels,
                        dtype='int16'
                    )
                    sd.wait()
                    audio_chunk = audio_chunk.flatten()
                    rms = self._calculate_rms(audio_chunk)

                    # Add to pre-recording buffer
                    pre_audio_buffer.append(audio_chunk)
                    pre_rms_buffer.append(rms)
                    has_speech = rms > self.silence_threshold

                    # Start recording if speech detected
                    if not is_recording and has_speech:
                        is_recording = True
                        chunk_start_time = time.time() - (len(pre_audio_buffer) * self.buffer_size)
                        audio_buffer = deque(pre_audio_buffer)
                        rms_buffer = deque(pre_rms_buffer)
                        print(f"[Speech] 🎤 Listening")

                    # Continue recording
                    elif is_recording:
                        audio_buffer.append(audio_chunk)
                        rms_buffer.append(rms)
                        
                        current_cycle_time = time.time()
                        current_duration = current_cycle_time - chunk_start_time

                        should_finalize = False
                        reason = None

                        # Check finalization conditions
                        if current_duration >= self.max_chunk_duration:
                            should_finalize = True
                            reason = "max_duration"
                        elif current_duration >= self.min_chunk_duration:
                            if self._detect_speech_gap(rms_buffer, self.silence_threshold, min_silence_buffers):
                                should_finalize = True
                                reason = "speech_gap"

                        if should_finalize:
                            if not self._has_sufficient_speech(rms_buffer, self.silence_threshold, self.min_speech_ratio):
                                print(f"[Speech] No speech detected")
                            else:
                                # Process and save audio chunk
                                full_audio = np.concatenate(list(audio_buffer))
                                resampled = resample_poly(full_audio, self.target_sample_rate, self.mic_sample_rate)
                                resampled = np.round(resampled).astype(np.int16)
                                filename = self.audio_dir / f"chunk_{chunk_id}.wav"
                                write(str(filename), self.target_sample_rate, resampled)
                                
                                chunk_info = {
                                    'filename': str(filename),
                                    'duration': current_duration,
                                    'reason': reason,
                                    'chunk_id': chunk_id,
                                    'record_end_time': current_cycle_time
                                }
                                self.audio_queue.put(chunk_info)

                            # Reset for next chunk
                            is_recording = False
                            chunk_start_time = None
                            audio_buffer.clear()
                            rms_buffer.clear()
                            chunk_id += 1

                except sd.PortAudioError as pae:
                    print(f"[Speech] Audio error: {pae}")
                    if "Input overflowed" in str(pae):
                        time.sleep(0.05)
                    continue
                except Exception as e:
                    print(f"[Speech] Recording error: {e}")
                    if stop_event.is_set():
                        break
                    continue

        except Exception as e:
            print(f"[Speech] Record loop error: {e}")
            traceback.print_exc()
        finally:
            print("[Speech] Record loop stopping. Signaling transcription aggregator.")
            self.audio_queue.put(None)

    def _transcription_aggregator_loop(self, final_speech_queue, stop_event, timings):
        """Aggregates transcriptions and sends complete utterances."""
        print("[Speech] Transcription Aggregator loop started.")
        queue_check_timeout = min(0.25, self.end_of_utterance_silence_duration / 2.0)
        active = True

        try:
            while active:
                chunk_info = None
                processed_chunk_this_iteration = False
                
                try:
                    current_timeout = 0.01 if (stop_event.is_set() and self.audio_queue.empty()) else queue_check_timeout
                    chunk_info = self.audio_queue.get(timeout=current_timeout)

                    if chunk_info is None:
                        active = False
                        print("[Speech] Aggregator: Received None (end of audio stream).")
                    elif chunk_info:
                        processed_chunk_this_iteration = True
                        transcript = self._transcribe_chunk(chunk_info, timings)
                        
                        if transcript and transcript.strip():
                            with self.transcript_agg_lock:
                                self.pending_transcripts.append(transcript.strip())
                                self.last_transcript_time = time.time()

                                # Calculate actual speech end time
                                record_end_time = chunk_info['record_end_time']
                                reason = chunk_info['reason']
                                
                                actual_chunk_speech_end_time = record_end_time
                                if reason == "speech_gap":
                                    actual_chunk_speech_end_time -= self.silence_duration
                                
                                self.current_utterance_actual_speech_end_time = actual_chunk_speech_end_time

                except queue.Empty:
                    pass

                if not processed_chunk_this_iteration and stop_event.is_set() and self.audio_queue.empty():
                    active = False

                # Check if we should flush the utterance
                with self.transcript_agg_lock:
                    should_flush_utterance = not active
                    flush_reason = ""

                    if self.pending_transcripts:
                        if not should_flush_utterance and self.last_transcript_time:
                            elapsed_since_last_transcript = time.time() - self.last_transcript_time
                            if elapsed_since_last_transcript > self.end_of_utterance_silence_duration:
                                should_flush_utterance = True
                                flush_reason = "timeout"
                        elif should_flush_utterance and not flush_reason:
                            flush_reason = "end of stream"

                        if should_flush_utterance:
                            full_utterance = " ".join(self.pending_transcripts)
                            user_speech_end_timestamp = self.current_utterance_actual_speech_end_time

                            if user_speech_end_timestamp is None:
                                user_speech_end_timestamp = self.last_transcript_time if self.last_transcript_time else time.time()

                            final_speech_queue.put({
                                'text': full_utterance,
                                'user_speech_end_time': user_speech_end_timestamp
                            })

                            # Reset for next utterance
                            self.pending_transcripts.clear()
                            self.last_transcript_time = None
                            self.current_utterance_actual_speech_end_time = None

                if not active and self.audio_queue.empty():
                    break

        except Exception as e:
            print(f"[Speech] Error in transcription aggregator loop: {e}")
            traceback.print_exc()
        finally:
            print("[Speech] Transcription Aggregator loop finished.")

    def _transcribe_chunk(self, chunk_info, timings):
        """Transcribe a single audio chunk."""
        filename = chunk_info['filename']
        duration = chunk_info['duration']
        reason = chunk_info['reason']
        chunk_id = chunk_info['chunk_id']

        transcription_start = time.perf_counter()
        try:
            segments, _ = self.model.transcribe(filename)
            transcript = " ".join([segment.text for segment in segments])
            transcription_end = time.perf_counter()
            transcription_duration = transcription_end - transcription_start
            timings['transcription_duration'] = transcription_duration

            reason_symbol = "⏰" if reason == "max_duration" else "🔇"

            if transcript.strip():
                return transcript
            else:
                print(f"[Speech] No speech detected")
                return None
                
        except Exception as e:
            print(f"[Speech] Transcription error for {filename}: {e}")
            return None 
