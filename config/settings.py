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
        load_dotenv()
        config = cls()
        config.llm.api_key = os.getenv("GOOGLE_API_KEY")
        return config
    
    def validate(self):
        issues = []
        if not self.llm.api_key:
            issues.append("LLM API key not set")
        return issues
