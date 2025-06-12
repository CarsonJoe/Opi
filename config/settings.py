"""
Configuration management for Opi Voice Assistant
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

@dataclass
class VoiceConfig:
    whisper_model: str = "tiny.en"
    whisper_compute_type: str = "int8"
    tts_model_path: Optional[str] = None
    tts_config_path: Optional[str] = None
    speaker_id: Optional[str] = None
    speech_speed: float = 1.0
    audio_device: Optional[int] = None
    sample_rate: int = 22050
    chunk_pause_ms: int = 50
    wake_words: List[str] = field(default_factory=lambda: ["hey opi", "opi"])
    require_wake_word_each_time: bool = False

@dataclass
class LLMConfig:
    provider: str = "google-genai"
    model: str = "gemini-2.0-flash-exp"
    api_key: Optional[str] = None
    temperature: float = 0.7

@dataclass
class MCPConfig:
    servers: Dict = field(default_factory=dict)

@dataclass
class PromptsConfig:
    system_prompt: str = "You are Opi, a helpful voice assistant."

@dataclass
class StorageConfig:
    base_dir: str = "./data"
    conversation_db: str = "./data/conversations.db"
    cache_dir: str = "./cache"
    output_dir: str = "./output"
    log_dir: str = "./logs"

@dataclass
class OpiConfig:
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    verbose: bool = False
    
    @classmethod
    def load(cls, config_path=None):
        """FIXED: Actually load the JSON configuration file!"""
        load_dotenv()
        
        # Determine config file path
        if config_path is None:
            config_path = "config.json"
        
        # Load JSON config if it exists
        config_data = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            print(f"[Config] Loaded configuration from {config_path}")
        else:
            print(f"[Config] No config file found at {config_path}, using defaults")
        
        # Create config with loaded data
        config = cls()
        
        # Load voice config
        if 'voice' in config_data:
            voice_data = config_data['voice']
            for key, value in voice_data.items():
                if hasattr(config.voice, key):
                    setattr(config.voice, key, value)
        
        # Load LLM config
        if 'llm' in config_data:
            llm_data = config_data['llm']
            for key, value in llm_data.items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)
        
        # Load other configs
        if 'mcp' in config_data:
            config.mcp.servers = config_data['mcp'].get('servers', {})
        
        if 'prompts' in config_data:
            prompts_data = config_data['prompts']
            for key, value in prompts_data.items():
                if hasattr(config.prompts, key):
                    setattr(config.prompts, key, value)
        
        if 'storage' in config_data:
            storage_data = config_data['storage']
            for key, value in storage_data.items():
                if hasattr(config.storage, key):
                    setattr(config.storage, key, value)
        
        # Set API key from environment if not in config
        if not config.llm.api_key:
            config.llm.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Debug: Print loaded audio device
        print(f"[Config] Audio device loaded: {config.voice.audio_device}")
        
        return config
    
    def validate(self):
        issues = []
        if not self.llm.api_key:
            issues.append("LLM API key not set")
        return issues
