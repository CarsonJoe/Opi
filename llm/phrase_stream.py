# llm/phrase_stream.py
"""
Ultra-low-latency speech helpers
================================
* PhraseStreamer — converts a raw text stream into human-friendly phrases.
* UltraLowLatencyTTSPipeline — parallel TTS + ordered playback.

Both classes are self-contained and have **no third-party dependencies**
beyond the `tts_worker` / `audio_worker` you already created elsewhere.
"""

from __future__ import annotations
import queue, threading, time
from typing import List, Dict, Optional
from termcolor import cprint


# ---------------------------------------------------------------------------
# PhraseStreamer
# ---------------------------------------------------------------------------
class PhraseStreamer:
    """
    Collects text chunks and yields natural breakpoints (phrases) quickly.
    The goal is to start speaking after only ~5-8 words instead of a full
    sentence.
    """

    _SENTENCE_END = ".!?●"  # last char triggers a hard break

    def __init__(self):
        self.buffer = ""

    def add_chunk(self, text: str) -> List[str]:
        """
        Feed partial text (e.g. a token or a short chunk).  Returns zero or
        more *phrases* that are ready to be spoken.
        """
        if not text:
            return []

        self.buffer += text
        phrases: List[str] = []

        # 1. hard break at sentence-ending punctuation
        while self.buffer and any(self.buffer.endswith(p) for p in self._SENTENCE_END):
            phrases.append(self.buffer.strip())
            self.buffer = ""

        # 2. soft break every ~10 words
        words = self.buffer.split()
        if len(words) > 10:
            cut = " ".join(words[:-3])  # leave a few words in buffer
            phrases.append(cut.strip())
            self.buffer = " ".join(words[-3:])

        return phrases

    def flush(self) -> Optional[str]:
        """Return whatever text is left (called after the model stops streaming)."""
        leftover = self.buffer.strip()
        self.buffer = ""
        return leftover if leftover else None


# ---------------------------------------------------------------------------
# UltraLowLatencyTTSPipeline
# ---------------------------------------------------------------------------
class UltraLowLatencyTTSPipeline:
    """
    * TTS threads work in parallel for speed.
    * Audio thread keeps output in the correct order.
    """

    def __init__(self, tts_worker, audio_worker, *, num_tts_threads: int = 2):
        self.tts_worker = tts_worker
        self.audio_worker = audio_worker
        self.num_tts_threads = num_tts_threads

        self.phrase_q: queue.Queue[Dict] = queue.Queue(maxsize=100)
        self.audio_q: queue.Queue[bytes] = queue.Queue(maxsize=100)

        self._stop = threading.Event()
        self.first_audio_time: Optional[float] = None

        # sequencing
        self._seq_lock = threading.Lock()
        self._next_seq_to_play = 0
        self._audio_buffer: Dict[int, bytes] = {}

    # ------------------------------- public API
    def start_pipeline(self, debug=False):
        self.debug = debug
        self._stop.clear()

        # TTS workers
        for i in range(self.num_tts_threads):
            threading.Thread(
                target=self._tts_loop,
                name=f"TTS-{i}",
                daemon=True,
            ).start()

        # Audio playback
        threading.Thread(
            target=self._audio_loop,
            name="AudioPlayback",
            daemon=True,
        ).start()

    def add_phrase(self, phrase: str, debug=False):
        """Queue a phrase for immediate TTS processing."""
        if not phrase.strip():
            return
        seq = getattr(self, "_seq_counter", 0)
        setattr(self, "_seq_counter", seq + 1)
        self.phrase_q.put({"seq": seq, "text": phrase.strip()})
        if debug or self.debug:
            cprint(f"[Pipeline] → queued phrase {seq}: {phrase[:40]}...", "cyan")

    def finish_pipeline(self) -> Optional[float]:
        """Block until all audio is spoken.  Returns first-audio timestamp."""
        # send poison pill for each TTS thread
        for _ in range(self.num_tts_threads):
            self.phrase_q.put(None)

        # wait for queues to empty
        self.phrase_q.join()
        self.audio_q.put(None)  # signal audio thread to finish
        self.audio_q.join()
        self._stop.set()
        return self.first_audio_time

    # ------------------------------- internal loops
    def _tts_loop(self):
        while not self._stop.is_set():
            item = self.phrase_q.get()
            if item is None:  # shutdown signal
                self.phrase_q.task_done()
                break

            seq = item["seq"]
            text = item["text"]
            audio = self.tts_worker._synthesize_text(text)  # blocking synth
            self.audio_q.put({"seq": seq, "audio": audio})
            self.phrase_q.task_done()

    def _audio_loop(self):
        while True:
            chunk = self.audio_q.get()
            if chunk is None:
                self.audio_q.task_done()
                break

            seq = chunk["seq"]
            with self._seq_lock:
                self._audio_buffer[seq] = chunk["audio"]
                # play any ready-to-go chunks in order
                while self._next_seq_to_play in self._audio_buffer:
                    audio = self._audio_buffer.pop(self._next_seq_to_play)
                    if self.first_audio_time is None:
                        self.first_audio_time = time.time()
                    self.audio_worker.play_audio(audio)
                    self._next_seq_to_play += 1

            self.audio_q.task_done()

