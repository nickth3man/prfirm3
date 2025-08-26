"""
Configuration management module for the Virtual PR Firm system.

This module provides centralized configuration management with support for:
- Environment variables
- Configuration files (YAML, JSON)
- Runtime configuration updates
- Default values and validation
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class NodeConfig:
    """Configuration for individual nodes."""
    max_retries: int = 3
    wait_time: float = 1.0
    timeout: Optional[float] = None
    fallback_enabled: bool = True
    cache_enabled: bool = False
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class LLMConfig:
    """Configuration for LLM services."""
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    rate_limit: Optional[int] = None  # requests per minute
    cost_limit: Optional[float] = None  # maximum cost per request


@dataclass
class PlatformConfig:
    """Configuration for platform-specific settings."""
    twitter_char_limit: int = 280
    linkedin_char_limit: int = 3000
    instagram_caption_limit: int = 2200
    reddit_title_limit: int = 300
    email_subject_limit: int = 100
    blog_min_words: int = 300
    blog_max_words: int = 2000
    hashtag_limit: Dict[str, int] = field(default_factory=lambda: {
        "twitter": 3,
        "instagram": 30,
        "linkedin": 5
    })


@dataclass
class StyleConfig:
    """Configuration for style compliance."""
    forbidden_phrases: List[str] = field(default_factory=lambda: [
        "em dash",
        "—",
        "not just",
        "not X, but Y",
        "rhetorical contrast"
    ])
    max_revision_attempts: int = 5
    style_check_enabled: bool = True
    brand_voice_enforcement: bool = True
    tone_analysis_enabled: bool = True


@dataclass
class StreamingConfig:
    """Configuration for streaming and real-time updates."""
    enabled: bool = True
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    buffer_size: int = 1024
    reconnect_attempts: int = 3
    reconnect_delay: float = 5.0
    heartbeat_interval: float = 30.0


@dataclass
class SystemConfig:
    """Main system configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    temp_dir: Path = Path("./tmp")
    max_concurrent_requests: int = 10
    request_timeout: float = 300.0  # 5 minutes
    enable_metrics: bool = True
    enable_tracing: bool = False


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_defaults()
        self._load_environment()
        if config_path:
            self._load_file(config_path)
        self._validate_config()
        
        # Create configuration objects
        self.system = self._create_system_config()
        self.nodes = self._create_node_configs()
        self.llm = self._create_llm_config()
        self.platforms = self._create_platform_config()
        self.style = self._create_style_config()
        self.streaming = self._create_streaming_config()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.config = {
            "system": {
                "environment": "development",
                "debug_mode": False,
                "log_level": "INFO",
                "data_dir": "./data",
                "cache_dir": "./cache",
                "temp_dir": "./tmp"
            },
            "nodes": {
                "default": {
                    "max_retries": 3,
                    "wait_time": 1.0,
                    "fallback_enabled": True
                }
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "platforms": {
                "twitter_char_limit": 280,
                "linkedin_char_limit": 3000,
                "instagram_caption_limit": 2200
            },
            "style": {
                "forbidden_phrases": ["em dash", "—"],
                "max_revision_attempts": 5
            },
            "streaming": {
                "enabled": True,
                "websocket_host": "localhost",
                "websocket_port": 8765
            }
        }
    
    def _load_environment(self):
        """Load configuration from environment variables."""
        # System environment
        env = os.getenv("APP_ENVIRONMENT", "development")
        self.config["system"]["environment"] = env
        self.config["system"]["debug_mode"] = os.getenv("DEBUG", "false").lower() == "true"
        self.config["system"]["log_level"] = os.getenv("LOG_LEVEL", "INFO")
        
        # LLM configuration
        self.config["llm"]["provider"] = os.getenv("LLM_PROVIDER", self.config["llm"]["provider"])
        self.config["llm"]["model"] = os.getenv("LLM_MODEL", self.config["llm"]["model"])
        self.config["llm"]["api_key"] = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
        
        # Node configuration
        if os.getenv("NODE_MAX_RETRIES"):
            self.config["nodes"]["default"]["max_retries"] = int(os.getenv("NODE_MAX_RETRIES"))
        if os.getenv("NODE_WAIT_TIME"):
            self.config["nodes"]["default"]["wait_time"] = float(os.getenv("NODE_WAIT_TIME"))
        
        # Streaming configuration
        if os.getenv("STREAMING_ENABLED"):
            self.config["streaming"]["enabled"] = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        if os.getenv("WEBSOCKET_HOST"):
            self.config["streaming"]["websocket_host"] = os.getenv("WEBSOCKET_HOST")
        if os.getenv("WEBSOCKET_PORT"):
            self.config["streaming"]["websocket_port"] = int(os.getenv("WEBSOCKET_PORT"))
    
    def _load_file(self, path: str):
        """Load configuration from file."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Configuration file not found: {path}")
            return
        
        try:
            with open(path_obj, 'r') as f:
                if path_obj.suffix == '.json':
                    file_config = json.load(f)
                elif path_obj.suffix in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {path_obj.suffix}")
                    return
            
            # Deep merge with existing config
            self._deep_merge(self.config, file_config)
            logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate environment
        try:
            Environment(self.config["system"]["environment"])
        except ValueError:
            logger.warning(f"Invalid environment: {self.config['system']['environment']}, using development")
            self.config["system"]["environment"] = "development"
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config["system"]["log_level"] not in valid_log_levels:
            logger.warning(f"Invalid log level: {self.config['system']['log_level']}, using INFO")
            self.config["system"]["log_level"] = "INFO"
        
        # Validate numeric values
        if self.config["nodes"]["default"]["max_retries"] < 0:
            self.config["nodes"]["default"]["max_retries"] = 0
        if self.config["nodes"]["default"]["wait_time"] < 0:
            self.config["nodes"]["default"]["wait_time"] = 0
        
        # Create directories if they don't exist
        for dir_key in ["data_dir", "cache_dir", "temp_dir"]:
            dir_path = Path(self.config["system"][dir_key])
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True)
                    logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
    
    def _create_system_config(self) -> SystemConfig:
        """Create system configuration object."""
        sys_cfg = self.config["system"]
        return SystemConfig(
            environment=Environment(sys_cfg["environment"]),
            debug_mode=sys_cfg["debug_mode"],
            log_level=sys_cfg["log_level"],
            data_dir=Path(sys_cfg["data_dir"]),
            cache_dir=Path(sys_cfg["cache_dir"]),
            temp_dir=Path(sys_cfg["temp_dir"])
        )
    
    def _create_node_configs(self) -> Dict[str, NodeConfig]:
        """Create node configuration objects."""
        configs = {}
        for node_name, node_cfg in self.config["nodes"].items():
            configs[node_name] = NodeConfig(
                max_retries=node_cfg.get("max_retries", 3),
                wait_time=node_cfg.get("wait_time", 1.0),
                fallback_enabled=node_cfg.get("fallback_enabled", True)
            )
        return configs
    
    def _create_llm_config(self) -> LLMConfig:
        """Create LLM configuration object."""
        llm_cfg = self.config["llm"]
        return LLMConfig(
            provider=llm_cfg["provider"],
            model=llm_cfg["model"],
            api_key=llm_cfg.get("api_key"),
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 1000)
        )
    
    def _create_platform_config(self) -> PlatformConfig:
        """Create platform configuration object."""
        plat_cfg = self.config["platforms"]
        return PlatformConfig(
            twitter_char_limit=plat_cfg.get("twitter_char_limit", 280),
            linkedin_char_limit=plat_cfg.get("linkedin_char_limit", 3000),
            instagram_caption_limit=plat_cfg.get("instagram_caption_limit", 2200)
        )
    
    def _create_style_config(self) -> StyleConfig:
        """Create style configuration object."""
        style_cfg = self.config["style"]
        return StyleConfig(
            forbidden_phrases=style_cfg.get("forbidden_phrases", []),
            max_revision_attempts=style_cfg.get("max_revision_attempts", 5)
        )
    
    def _create_streaming_config(self) -> StreamingConfig:
        """Create streaming configuration object."""
        stream_cfg = self.config["streaming"]
        return StreamingConfig(
            enabled=stream_cfg["enabled"],
            websocket_host=stream_cfg["websocket_host"],
            websocket_port=stream_cfg["websocket_port"]
        )
    
    def get_node_config(self, node_name: str) -> NodeConfig:
        """Get configuration for a specific node."""
        return self.nodes.get(node_name, self.nodes.get("default", NodeConfig()))
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration at runtime."""
        self._deep_merge(self.config, updates)
        self._validate_config()
        # Recreate configuration objects
        self.system = self._create_system_config()
        self.nodes = self._create_node_configs()
        self.llm = self._create_llm_config()
        self.platforms = self._create_platform_config()
        self.style = self._create_style_config()
        self.streaming = self._create_streaming_config()
        logger.info("Configuration updated")
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            logger.error("No path specified for saving configuration")
            return
        
        path_obj = Path(save_path)
        try:
            with open(path_obj, 'w') as f:
                if path_obj.suffix == '.json':
                    json.dump(self.config, f, indent=2)
                elif path_obj.suffix in ['.yml', '.yaml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    logger.error(f"Unsupported configuration file format: {path_obj.suffix}")
                    return
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self.config.copy()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def initialize_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager