# Opi Voice Assistant

A sophisticated voice assistant designed for Orange Pi and Linux systems with advanced AI capabilities, multi-modal input support, and comprehensive observability.

## 🚀 Features

### Core Capabilities
- 🎤 **Multi-Modal Input**: Voice recognition, text input, or hybrid mode
- 🧠 **Multiple LLM Support**: Google Gemini, OpenAI, Anthropic Claude
- 🛠️ **MCP Tool Integration**: Extensible tool system via Model Context Protocol
- 🗣️ **High-Quality TTS**: Piper TTS with configurable voices and speeds
- ⚡ **Ultra-Low Latency**: Optimized for real-time conversation (<1s response)
- 📊 **Advanced Observability**: LangSmith integration for debugging and analytics

### Voice Processing
- **Speech-to-Text**: Faster Whisper with configurable models
- **Wake Word Detection**: Intelligent utterance boundary detection  
- **Audio Pipeline**: Real-time streaming with smart chunking
- **Device Support**: Automatic audio device detection and configuration

### AI & Tools
- **Agent Framework**: LangChain-based tool-aware agents
- **Tool Calling**: Synchronous MCP integration with error handling
- **Conversation Memory**: Context-aware multi-turn conversations
- **Performance Monitoring**: Comprehensive metrics and timing analysis

## 📋 Requirements

### System Requirements
- **OS**: Linux (Ubuntu/Debian recommended), macOS, Windows
- **Python**: 3.11 or higher
- **Audio**: Microphone and speakers (for voice mode)
- **Memory**: 2GB+ RAM (depending on models)

### Hardware Recommendations
- **Primary Target**: Orange Pi devices
- **Audio**: USB microphone for better quality
- **Storage**: 1GB+ for models and cache

## 🛠️ Installation

### 1. Clone and Setup Environment
```bash
git clone <repository_url>
cd Opi
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate on Windows
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system audio dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-dev portaudio19-dev espeak espeak-data

# Add user to audio group (Linux)
sudo usermod -a -G audio $USER
# Log out and back in for this to take effect
```

### 3. Configure API Keys
```bash
# Create .env file for API keys
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
echo "LANGSMITH_API_KEY=your_langsmith_key_here" >> .env  # Optional
```

### 4. Download TTS Models (Optional)
```bash
# Create models directory
mkdir -p models

# Download Piper TTS models from:
# https://github.com/rhasspy/piper/releases
# Place .onnx and .onnx.json files in models/ directory
```

### 5. Test Installation
```bash
# Test audio devices
python main.py --list-devices

# Test voice components
python main.py --test-voice

# Test in text-only mode (no audio required)
python main.py --text
```

## 🎯 Usage

### Basic Usage
```bash
# Default voice mode
python main.py

# Text-only mode (no audio hardware needed)
python main.py --text

# Hybrid mode (voice + text input)
python main.py --hybrid

# Verbose mode for debugging
python main.py --verbose
```

### Advanced Options
```bash
# Custom configuration
python main.py --config /path/to/config.json

# Test different components
python main.py --test-streaming    # Test full pipeline
python main.py --test-mcp         # Test tool integration
python main.py --test-langsmith   # Test observability

# Audio diagnostics
python main.py --list-devices
python voice/audio_utils.py
```

## ⚙️ Configuration

### Basic Configuration
Create a `config.json` file in the project root:

```json
{
  "llm": {
    "provider": "google-genai",
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.7
  },
  "voice": {
    "whisper_model": "tiny.en",
    "audio_device": null,
    "sample_rate": 22050,
    "wake_words": ["hey opi", "opi"]
  },
  "mcp": {
    "servers": {
      "example_server": {
        "command": "path/to/mcp/server",
        "args": ["--arg1", "value1"],
        "enabled": true
      }
    }
  },
  "prompts": {
    "system_prompt": "You are Opi, a helpful voice assistant."
  },
  "langsmith": {
    "enabled": true,
    "project_name": "opi-voice-assistant"
  }
}
```

### Environment Variables
```bash
# Required for LLM
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_key    # Optional
OPENAI_API_KEY=your_openai_key          # Optional

# Optional for observability
LANGSMITH_API_KEY=your_langsmith_key

# Debug mode
OPI_DEBUG=1
```

## 🏗️ Architecture

Opi follows a modular, worker-based architecture:

- **Voice Workers**: Handle audio input/output and speech processing
- **LLM Integration**: Multi-provider LLM support with tool calling
- **MCP Manager**: Tool integration via Model Context Protocol
- **Configuration System**: JSON-based config with env var overrides
- **Observability**: LangSmith tracing and performance monitoring

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🧪 Testing

### Component Tests
```bash
# Test individual components
python main.py --test-voice       # Voice pipeline
python main.py --test-streaming   # Full streaming pipeline  
python main.py --test-mcp         # MCP tool integration
python main.py --test-langsmith   # Observability tracing

# Audio diagnostics
python voice/audio_utils.py
```

### Manual Testing
```bash
# Test different modes
python main.py --text --verbose
python main.py --hybrid --verbose
python main.py --verbose

# Test with custom config
python main.py --config test_config.json
```

## 🐛 Troubleshooting

### Audio Issues
```bash
# Check audio devices
python main.py --list-devices

# Test audio permissions
python voice/audio_utils.py

# Fix permissions (Linux)
sudo usermod -a -G audio $USER
```

### LLM Issues
```bash
# Verify API keys
python -c "import os; print('GOOGLE_API_KEY:', bool(os.getenv('GOOGLE_API_KEY')))"

# Test LLM connection
python main.py --test-streaming
```

### MCP Issues
```bash
# Test MCP servers
python main.py --test-mcp

# Check server logs
python main.py --verbose --test-mcp
```

## 📊 Performance

### Typical Latencies
- **Speech Recognition**: 0.1-0.5s (depending on model)
- **LLM Response**: 0.5-2.0s (depending on provider and tools)
- **Speech Synthesis**: 0.1-0.3s
- **Total Response Time**: <1.0s (target for first audio)

### Optimization Tips
- Use smaller Whisper models for faster STT
- Configure audio devices for optimal sample rates
- Use local TTS models for lower latency
- Monitor performance with LangSmith integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the code style guidelines
4. Add tests for new features
5. Update documentation
6. Submit a pull request

See development guidelines in project memory files.

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech
- [LangChain](https://github.com/langchain-ai/langchain) for LLM orchestration
- [Model Context Protocol](https://modelcontextprotocol.io/) for tool integration
