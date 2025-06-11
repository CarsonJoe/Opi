#!/bin/bash
# setup.sh - Quick setup script for Opi Voice Assistant

echo "🚀 Setting up Opi Voice Assistant..."

# Install missing Python packages
echo "📦 Installing missing Python packages..."
pip install google-generativeai
pip install python-dotenv
pip install commentjson

# Fix the config.json (replace the one you have)
echo "🔧 Fixing configuration file..."
cat > config.json << 'EOF'
{
  "llm": {
    "provider": "google-genai",
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.7,
    "timeout": 30
  },
  "voice": {
    "whisper_model": "tiny.en",
    "whisper_compute_type": "int8",
    "speech_speed": 1.0,
    "sample_rate": 22050,
    "chunk_pause_ms": 50,
    "wake_words": ["hey opi", "opi"],
    "require_wake_word_each_time": false,
    "silence_threshold": 0.01,
    "silence_duration": 1.0,
    "min_speech_ratio": 0.1,
    "end_of_utterance_silence": 2.0
  },
  "mcp": {
    "servers": {},
    "cache_expiry_hours": 24,
    "max_concurrent_calls": 5
  },
  "prompts": {
    "system_prompt": "You are Opi, a helpful voice assistant running on an Orange Pi. Keep responses conversational and concise since they will be spoken aloud. Always be friendly and helpful.",
    "wake_response": "Hello! How can I help you?",
    "error_response": "Sorry, I encountered an error. Please try again.",
    "goodbye_response": "Goodbye! Have a great day!"
  },
  "storage": {
    "base_dir": "./data",
    "conversation_db": "./data/conversations.db",
    "cache_dir": "./cache",
    "output_dir": "./output",
    "log_dir": "./logs"
  },
  "display": {
    "enabled": false
  },
  "verbose": false,
  "debug_timing": false
}
EOF

# Fix the voice/__init__.py file (remove the bad import)
echo "🔧 Fixing voice package imports..."
cat > voice/__init__.py << 'EOF'
# voice/__init__.py
"""
Voice processing package for Opi Voice Assistant
"""

from .audio_utils import list_audio_devices, test_audio_recording, test_audio_playback
from .speech_worker import SpeechWorker

__all__ = [
    'list_audio_devices', 
    'test_audio_recording', 
    'test_audio_playback',
    'SpeechWorker'
]
EOF

# Check if API key is set
if grep -q "GOOGLE_API_KEY=" .env 2>/dev/null; then
    echo "✅ API key found in .env file"
else
    echo "⚠️  API key not found. Make sure .env contains:"
fi

echo "✅ Setup complete! Now run:"
echo "   python test.py"
