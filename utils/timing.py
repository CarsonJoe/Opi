# utils/timing.py
"""
Timing utilities for Opi Voice Assistant
Tracks performance metrics across all components
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from termcolor import cprint


@dataclass
class TimingMetric:
    """Individual timing metric."""
    name: str
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = None


class TimingTracker:
    """Tracks and analyzes performance metrics."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.session_start = time.time()
        
    def add_timing(self, category: str, duration: float, metadata: Dict[str, Any] = None):
        """Add a timing measurement."""
        metric = TimingMetric(
            name=category,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.timings[category].append(metric)
    
    def start_timer(self, category: str) -> 'Timer':
        """Start a context manager timer."""
        return Timer(self, category)
    
    def get_average(self, category: str) -> Optional[float]:
        """Get average duration for a category."""
        if category not in self.timings or not self.timings[category]:
            return None
        
        durations = [metric.duration for metric in self.timings[category]]
        return sum(durations) / len(durations)
    
    def get_latest(self, category: str) -> Optional[float]:
        """Get latest timing for a category."""
        if category not in self.timings or not self.timings[category]:
            return None
        
        return self.timings[category][-1].duration
    
    def get_stats(self, category: str) -> Dict[str, float]:
        """Get comprehensive stats for a category."""
        if category not in self.timings or not self.timings[category]:
           return {}
        
        durations = [metric.duration for metric in self.timings[category]]
        
        return {
            "count": len(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "latest": durations[-1],
            "total": sum(durations)
        }
    
    def print_summary(self):
        """Print a comprehensive timing summary."""
        if not self.timings:
            cprint("[Timing] No timing data collected", "yellow")
            return
        
        session_duration = time.time() - self.session_start
        cprint(f"\n=== Opi Performance Summary (Session: {session_duration:.1f}s) ===", "cyan")
        
        # Calculate key metrics
        categories = ["stt", "llm_response", "tts", "end_to_end"]
        
        for category in categories:
            if category in self.timings:
                stats = self.get_stats(category)
                self._print_category_stats(category, stats)
        
        # Print other categories
        other_categories = set(self.timings.keys()) - set(categories)
        if other_categories:
            cprint("\n[Other Metrics]", "cyan")
            for category in sorted(other_categories):
                stats = self.get_stats(category)
                self._print_category_stats(category, stats)
        
        # Performance analysis
        self._print_performance_analysis()
        
        cprint("=" * 50, "cyan")
    
    def _print_category_stats(self, category: str, stats: Dict[str, float]):
        """Print stats for a specific category."""
        if not stats:
            return
        
        name = category.replace("_", " ").title()
        cprint(f"\n[{name}]", "yellow")
        cprint(f"  Count: {stats['count']}", "white")
        cprint(f"  Average: {stats['average']:.3f}s", "white")
        cprint(f"  Range: {stats['min']:.3f}s - {stats['max']:.3f}s", "white")
        cprint(f"  Latest: {stats['latest']:.3f}s", "white")
    
    def _print_performance_analysis(self):
        """Print performance analysis and recommendations."""
        cprint("\n[Performance Analysis]", "cyan")
        
        # STT Performance
        stt_avg = self.get_average("stt")
        if stt_avg:
            if stt_avg > 2.0:
                cprint("  🐌 STT: Consider using smaller Whisper model", "yellow")
            elif stt_avg < 0.5:
                cprint("  ⚡ STT: Excellent performance", "green")
            else:
                cprint("  ✅ STT: Good performance", "green")
        
        # LLM Response Performance
        llm_avg = self.get_average("llm_response")
        if llm_avg:
            if llm_avg > 3.0:
                cprint("  🐌 LLM: Consider reducing context or using faster model", "yellow")
            elif llm_avg < 1.0:
                cprint("  ⚡ LLM: Excellent response time", "green")
            else:
                cprint("  ✅ LLM: Good response time", "green")
        
        # TTS Performance
        tts_avg = self.get_average("tts")
        if tts_avg:
            if tts_avg > 1.5:
                cprint("  🐌 TTS: Consider using smaller model or reduce quality", "yellow")
            elif tts_avg < 0.3:
                cprint("  ⚡ TTS: Excellent synthesis speed", "green")
            else:
                cprint("  ✅ TTS: Good synthesis speed", "green")
        
        # End-to-end latency
        e2e_avg = self.get_average("end_to_end")
        if e2e_avg:
            if e2e_avg > 5.0:
                cprint("  🐌 Overall: High latency - check network and models", "red")
            elif e2e_avg < 2.0:
                cprint("  ⚡ Overall: Excellent responsiveness", "green")
            else:
                cprint("  ✅ Overall: Good responsiveness", "green")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a structured performance report."""
        report = {
            "session_duration": time.time() - self.session_start,
            "categories": {},
            "summary": {}
        }
        
        for category in self.timings:
            report["categories"][category] = self.get_stats(category)
        
        # Calculate summary metrics
        if "stt" in self.timings and "llm_response" in self.timings and "tts" in self.timings:
            stt_avg = self.get_average("stt") or 0
            llm_avg = self.get_average("llm_response") or 0
            tts_avg = self.get_average("tts") or 0
            
            report["summary"] = {
                "average_total_latency": stt_avg + llm_avg + tts_avg,
                "stt_percentage": (stt_avg / (stt_avg + llm_avg + tts_avg)) * 100 if (stt_avg + llm_avg + tts_avg) > 0 else 0,
                "llm_percentage": (llm_avg / (stt_avg + llm_avg + tts_avg)) * 100 if (stt_avg + llm_avg + tts_avg) > 0 else 0,
                "tts_percentage": (tts_avg / (stt_avg + llm_avg + tts_avg)) * 100 if (stt_avg + llm_avg + tts_avg) > 0 else 0
            }
        
        return report


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: TimingTracker, category: str, metadata: Dict[str, Any] = None):
        self.tracker = tracker
        self.category = category
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.tracker.add_timing(self.category, duration, self.metadata)


# Decorator for automatic timing
def timed(category: str, tracker: TimingTracker):
    """Decorator to automatically time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracker.start_timer(category):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def timed_async(category: str, tracker: TimingTracker):
    """Decorator for async function timing."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with tracker.start_timer(category):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global timing utilities
def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"


def calculate_latency_breakdown(timings: Dict[str, float]) -> Dict[str, float]:
    """Calculate percentage breakdown of latency components."""
    total = sum(timings.values())
    if total == 0:
        return {}
    
    return {
        category: (duration / total) * 100 
        for category, duration in timings.items()
    }


# Voice-specific timing utilities
class VoiceLatencyTracker:
    """Specialized tracker for voice interaction latency."""
    
    def __init__(self):
        self.interaction_start = None
        self.speech_end_time = None
        self.first_response_time = None
        self.response_complete_time = None
    
    def start_interaction(self):
        """Mark start of voice interaction."""
        self.interaction_start = time.time()
    
    def mark_speech_end(self):
        """Mark end of user speech."""
        self.speech_end_time = time.time()
    
    def mark_first_response(self):
        """Mark first audio output."""
        self.first_response_time = time.time()
    
    def mark_response_complete(self):
        """Mark end of response playback."""
        self.response_complete_time = time.time()
    
    def get_latency_metrics(self) -> Dict[str, float]:
        """Get comprehensive latency metrics."""
        if not all([self.interaction_start, self.speech_end_time, 
                   self.first_response_time, self.response_complete_time]):
            return {}
        
        return {
            "speech_to_first_response": self.first_response_time - self.speech_end_time,
            "total_interaction_time": self.response_complete_time - self.interaction_start,
            "response_duration": self.response_complete_time - self.first_response_time,
            "processing_time": self.first_response_time - self.speech_end_time
        }
    
    def reset(self):
        """Reset for next interaction."""
        self.interaction_start = None
        self.speech_end_time = None
        self.first_response_time = None
        self.response_complete_time = None 
