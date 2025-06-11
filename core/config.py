"""
Configuration management for Opi Voice Assistant.
Loads from YAML with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging


@dataclass
class AudioConfig:
    """Audio system configuration."""
    sample_rate: int = 16000
    chunk_size: int = 1024
    device_index: Optional[int] = None
    wake_word: str = "hey opi"
    wake_sensitivity: float = 0.7


@dataclass
class SpeechConfig:
    """Speech recognition configuration."""
    model: str = "tiny.en"
    language: str = "en"
    compute_type: str = "int8"
    device: str = "cpu"
    
    # VAD parameters
    silence_threshold: float = 0.01
    silence_duration: float = 1.0
    min_chunk_duration: float = 0.5
    max_chunk_duration: float = 5.0


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0
    model_path: Optional[str] = None
    config_path: Optional[str] = None


@dataclass
class DisplayConfig:
    """Display system configuration."""
    connector_id: int = 208
    plane_id: int = 143
    video_device: str = "/dev/video0"
    width: int = 1920
    height: int = 1080
    default_mode: str = "passthrough"
    overlay_timeout: int = 30
    font_size: int = 24
    font_family: str = "Monospace"


@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = "You are Opi, a helpful voice assistant running on an Orange Pi."


@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Config:
    """Main application configuration."""
    # Component configs
    audio: AudioConfig = field(default_factory=AudioConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # MCP servers
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    
    # General settings
    log_level: str = "INFO"
    data_dir: str = "./data"
    
    @classmethod
    def load(cls, config_path: str) -> "Config":
        """Load configuration from YAML file with environment overrides."""
        logger = logging.getLogger("Config")
        
        # Load YAML file
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            config_data = {}
        else:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        config_data = cls._apply_env_overrides(config_data)
        
        # Create config objects
        config = cls()
        
        # Audio config
        if 'audio' in config_data:
            config.audio = AudioConfig(**config_data['audio'])
        
        # Speech config
        if 'speech' in config_data:
            config.speech = SpeechConfig(**config_data['speech'])
        
        # TTS config
        if 'tts' in config_data:
            config.tts = TTSConfig(**config_data['tts'])
        
        # Display config
        if 'display' in config_data:
            config.display = DisplayConfig(**config_data['display'])
        
        # LLM config
        if 'llm' in config_data:
            llm_data = config_data['llm'].copy()
            # Handle API key from environment
            if not llm_data.get('api_key'):
                llm_data['api_key'] = os.getenv('OPENAI_API_KEY')
            config.llm = LLMConfig(**llm_data)
        
        # MCP server
        if 'mcp_servers' in config_data:
            for name, server_data in config_data['mcp_servers'].items():
                config.mcp_servers[name] = MCPServerConfig(**server_data)
        
        # General settings
        config.log_level = config_data.get('log_level', config.log_level)
        config.data_dir = config_data.get('data_dir', config.data_dir)
        
        # Validate configuration
        config._validate()
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    @staticmethod
    def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config data."""
        # Simple environment variable mapping
        env_mappings = {
            'OPENAI_API_KEY': ['llm', 'api_key'],
            'OPI_LOG_LEVEL': ['log_level'],
            'OPI_DATA_DIR': ['data_dir'],
            'OPI_WAKE_WORD': ['audio', 'wake_word'],
            'OPI_TTS_VOICE': ['tts', 'voice'],
            'OPI_LLM_MODEL': ['llm', 'model'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Navigate to the nested config location
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = env_value
        
        return config_data
    
    def _validate(self):
        """Validate configuration values."""
        logger = logging.getLogger("Config")
        
        # Validate LLM API key
        if not self.llm.api_key:
            logger.warning("No LLM API key configured - set OPENAI_API_KEY environment variable")
        
        # Validate data directory
        data_path = Path(self.data_dir)
        try:
            data_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create data directory {self.data_dir}: {e}")
        
        # Validate audio settings
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            logger.warning(f"Unusual sample rate: {self.audio.sample_rate}")
        
        # Validate TTS voice model paths if specified
        if self.tts.model_path and not Path(self.tts.model_path).exists():
            logger.warning(f"TTS model path not found: {self.tts.model_path}")
    
    def save(self, config_path: str):
        """Save current configuration to YAML file."""
        config_data = {
            'audio': self.audio.__dict__,
            'speech': self.speech.__dict__,
            'tts': self.tts.__dict__,
            'display': self.display.__dict__,
            'llm': {k: v for k, v in self.llm.__dict__.items() if k != 'api_key'},  # Don't save API key
            'mcp_servers': {name: server.__dict__ for name, server in self.mcp_servers.items()},
            'log_level': self.log_level,
            'data_dir': self.data_dir,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    def get_model_paths(self) -> tuple[Optional[str], Optional[str]]:
        """Get TTS model paths, with fallbacks to default locations."""
        model_path = self.tts.model_path
        config_path = self.tts.config_path
        
        if not model_path or not config_path:
            # Try to find default Piper models
            possible_paths = [
                Path.home() / ".local/share/piper",
                Path("/usr/share/piper"),
                Path("./models/piper"),
            ]
            
            for base_path in possible_paths:
                if base_path.exists():
                    # Look for voice model
                    voice_files = list(base_path.glob(f"*{self.tts.voice}*.onnx"))
                    if voice_files:
                        model_path = str(voice_files[0])
                        config_path = str(voice_files[0].with_suffix('.onnx.json'))
                        break
        
        return model_path, config_paths
