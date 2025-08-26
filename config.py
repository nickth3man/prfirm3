"""Configuration management for the Virtual PR Firm application.

This module provides centralized configuration management with support for
environment variables, configuration files, and sensible defaults. It implements
a hierarchical configuration system that prioritizes CLI args > environment
variables > config file > defaults.

Example Usage:
    >>> from config import get_config
    >>> config = get_config()
    >>> print(config.gradio.port)
    7860
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "json"  # "json" or "human"
    file: Optional[str] = None
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class GradioConfig:
    """Gradio interface configuration."""
    port: int = 7860
    host: str = "0.0.0.0"
    share: bool = False
    auth: Optional[str] = None  # username:password format
    ssl_verify: bool = True
    show_error: bool = True


@dataclass
class LLMConfig:
    """LLM API configuration."""
    provider: str = "openai"  # "openai", "anthropic", "google"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1


@dataclass
class SecurityConfig:
    """Security and authentication settings."""
    enable_auth: bool = False
    session_timeout: int = 3600  # 1 hour
    rate_limit_requests: int = 60  # requests per minute
    rate_limit_window: int = 60  # seconds
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = field(default_factory=lambda: [".xml", ".json", ".txt"])


@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_cache: bool = True
    ttl: int = 3600  # 1 hour
    max_size: int = 1000
    redis_url: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    config_file: Optional[str] = None
    
    # Sub-configurations
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gradio: GradioConfig = field(default_factory=GradioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    def __post_init__(self):
        """Post-initialization processing."""
        self._load_from_env()
        self._load_from_file()
        self._validate()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Logging
        if env_val := os.getenv("LOG_LEVEL"):
            self.logging.level = env_val
        if env_val := os.getenv("LOG_FORMAT"):
            self.logging.format = env_val
        if env_val := os.getenv("LOG_FILE"):
            self.logging.file = env_val
            
        # Gradio
        if env_val := os.getenv("GRADIO_PORT"):
            self.gradio.port = int(env_val)
        if env_val := os.getenv("GRADIO_HOST"):
            self.gradio.host = env_val
        if env_val := os.getenv("GRADIO_SHARE"):
            self.gradio.share = env_val.lower() in ("true", "1", "yes")
        if env_val := os.getenv("DEMO_PASSWORD"):
            self.gradio.auth = f"admin:{env_val}"
            
        # LLM
        if env_val := os.getenv("LLM_PROVIDER"):
            self.llm.provider = env_val
        if env_val := os.getenv("LLM_MODEL"):
            self.llm.model = env_val
        if env_val := os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(env_val)
        if env_val := os.getenv("LLM_MAX_TOKENS"):
            self.llm.max_tokens = int(env_val)
            
        # Security
        if env_val := os.getenv("ENABLE_AUTH"):
            self.security.enable_auth = env_val.lower() in ("true", "1", "yes")
        if env_val := os.getenv("RATE_LIMIT_REQUESTS"):
            self.security.rate_limit_requests = int(env_val)
            
        # Cache
        if env_val := os.getenv("REDIS_URL"):
            self.cache.redis_url = env_val
        if env_val := os.getenv("ENABLE_CACHE"):
            self.cache.enable_cache = env_val.lower() in ("true", "1", "yes")
            
        # Debug mode
        if env_val := os.getenv("DEBUG"):
            self.debug = env_val.lower() in ("true", "1", "yes")
    
    def _load_from_file(self):
        """Load configuration from YAML file if specified."""
        config_file = self.config_file or os.getenv("CONFIG_FILE")
        if not config_file:
            return
            
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # Apply file configuration (lower priority than env vars)
            self._apply_dict_config(config_data)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_file}: {e}")
    
    def _apply_dict_config(self, config_data: Dict[str, Any]):
        """Apply configuration from dictionary."""
        if not isinstance(config_data, dict):
            return
            
        # Apply to sub-configurations
        for section, data in config_data.items():
            if hasattr(self, section) and isinstance(data, dict):
                section_obj = getattr(self, section)
                for key, value in data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def _validate(self):
        """Validate configuration values."""
        # Validate logging level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_log_levels:
            logger.warning(f"Invalid log level: {self.logging.level}, using INFO")
            self.logging.level = "INFO"
        
        # Validate port range
        if not (1 <= self.gradio.port <= 65535):
            logger.warning(f"Invalid port: {self.gradio.port}, using 7860")
            self.gradio.port = 7860
        
        # Validate LLM settings
        if not (0 <= self.llm.temperature <= 2):
            logger.warning(f"Invalid temperature: {self.llm.temperature}, using 0.7")
            self.llm.temperature = 0.7
        
        if self.llm.max_tokens <= 0:
            logger.warning(f"Invalid max_tokens: {self.llm.max_tokens}, using 2000")
            self.llm.max_tokens = 2000


def get_config(config_file: Optional[str] = None) -> AppConfig:
    """Get application configuration.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        AppConfig: Configured application settings
    """
    config = AppConfig(config_file=config_file)
    return config


def create_config_template(output_path: str = "config_template.yaml"):
    """Create a configuration template file.
    
    Args:
        output_path: Path where to save the template
    """
    config = AppConfig()
    template_data = {
        "logging": {
            "level": config.logging.level,
            "format": config.logging.format,
            "file": config.logging.file,
        },
        "gradio": {
            "port": config.gradio.port,
            "host": config.gradio.host,
            "share": config.gradio.share,
        },
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        "security": {
            "enable_auth": config.security.enable_auth,
            "rate_limit_requests": config.security.rate_limit_requests,
        },
        "cache": {
            "enable_cache": config.cache.enable_cache,
            "ttl": config.cache.ttl,
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template_data, f, default_flow_style=False, indent=2)
    
    print(f"Configuration template created: {output_path}")


if __name__ == "__main__":
    # Create configuration template when run directly
    create_config_template()