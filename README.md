# Opi Voice Assistant

A sophisticated voice assistant for Orange Pi with MCP integration, HDMI display control, and real-time overlays.

## Quick Start

1. **Setup environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure:**
   ```bash
   cp config.sample.json config.json
   # Edit config.json with your settings
   export OPENAI_API_KEY="your_api_key"
   ```

3. **Download TTS models:**
   ```bash
   mkdir -p models
   # Download Piper TTS models (see setup guide)
   ```

4. **Run:**
   ```bash
   python main.py
   ```

## Features

- 🎤 Wake word detection and speech-to-text
- 🧠 LLM with MCP tool integration
- 📺 HDMI display control with real-time overlays
- 🗣️ High-quality text-to-speech
- 💾 Conversation memory and history
- 🔧 Extensible tool system

## Documentation

See the setup guide and documentation in the project files.
