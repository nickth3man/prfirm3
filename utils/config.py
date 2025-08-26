"""Configuration management utilities for the Virtual PR Firm.

This module provides centralized configuration management for the Virtual PR Firm
application, including environment-based settings, default values, and validation.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration with sensible defaults."""
    
    # LLM Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")))
    llm_timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "60")))
    
    # Application Settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    cache_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))
    
    # Gradio Interface Settings
    gradio_port: int = field(default_factory=lambda: int(os.getenv("GRADIO_PORT", "7860")))
    gradio_share: bool = field(default_factory=lambda: os.getenv("GRADIO_SHARE", "false").lower() == "true")
    gradio_debug: bool = field(default_factory=lambda: os.getenv("GRADIO_DEBUG", "false").lower() == "true")
    
    # Content Generation Settings
    max_content_length: int = field(default_factory=lambda: int(os.getenv("MAX_CONTENT_LENGTH", "10000")))
    supported_platforms: list = field(default_factory=lambda: [
        "twitter", "linkedin", "facebook", "instagram", "tiktok", "youtube"
    ])
    
    # File Paths
    brand_bible_path: Optional[str] = field(default_factory=lambda: os.getenv("BRAND_BIBLE_PATH"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./output"))
    
    # Rate Limiting
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "3600")))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        if not self.llm_api_key:
            logger.warning("LLM_API_KEY not set - some features may not work")
        
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be non-negative")
        
        if self.llm_timeout < 1:
            raise ValueError("LLM_TIMEOUT must be at least 1 second")
        
        if self.gradio_port < 1 or self.gradio_port > 65535:
            raise ValueError("GRADIO_PORT must be between 1 and 65535")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_max_retries": self.llm_max_retries,
            "llm_timeout": self.llm_timeout,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "gradio_port": self.gradio_port,
            "gradio_share": self.gradio_share,
            "gradio_debug": self.gradio_debug,
            "max_content_length": self.max_content_length,
            "supported_platforms": self.supported_platforms,
            "brand_bible_path": self.brand_bible_path,
            "output_dir": self.output_dir,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
        }
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AppConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config_from_file(config_path: Union[str, Path]) -> AppConfig:
    """Load configuration from file and set as global config."""
    config = AppConfig.from_file(config_path)
    set_config(config)
    return config


def create_default_config_file(config_path: Union[str, Path] = "config.json") -> None:
    """Create a default configuration file."""
    config = AppConfig()
    config.save_to_file(config_path)
    logger.info(f"Created default configuration file: {config_path}")