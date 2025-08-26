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
        """
        Run configuration loading and validation after dataclass initialization.
        
        Called automatically after AppConfig is constructed. Loads overrides from environment variables, then attempts to load and apply a YAML configuration file (if configured), and finally validates and normalizes resulting settings. This mutates the AppConfig instance in place to reflect environment/file values and validated fallbacks.
        """
        self._load_from_env()
        self._load_from_file()
        self._validate()
    
    def _load_from_env(self):
        """
        Load configuration values from environment variables and apply them to this AppConfig instance.
        
        One-line summary:
        Populates relevant sub-configs (logging, gradio, llm, security, cache) and top-level debug flag from environment variables when present.
        
        Details:
        - Updates only when an environment variable is set; existing values are preserved otherwise.
        - Boolean-like variables accept "true", "1", or "yes" (case-insensitive).
        - Numeric environment variables are parsed with int() or float() as appropriate.
        - Environment variables handled:
          - Logging: LOG_LEVEL, LOG_FORMAT, LOG_FILE
          - Gradio: GRADIO_PORT, GRADIO_HOST, GRADIO_SHARE, DEMO_PASSWORD (sets gradio.auth as "admin:<password>")
          - LLM: LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
          - Security: ENABLE_AUTH, RATE_LIMIT_REQUESTS
          - Cache: REDIS_URL, ENABLE_CACHE
          - Debug: DEBUG
        
        Side effects:
        Mutates attributes on self and its nested config objects (self.logging, self.gradio, self.llm, self.security, self.cache, self.debug).
        """
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
        """
        Load configuration from a YAML file (if specified) and apply it to this AppConfig.
        
        This method checks self.config_file first, falling back to the CONFIG_FILE environment variable. If a path is provided and the file exists, the file is parsed with yaml.safe_load and the resulting mapping is applied via self._apply_dict_config. File-based values are treated as lower priority than environment-variable overrides. If the file is missing the method returns silently after logging a warning; any parsing or IO errors are caught and logged (the method does not raise).
        """
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
        """
        Apply configuration values from a dictionary to this AppConfig's sub-configuration objects.
        
        Only dictionary entries whose top-level key matches an attribute on this AppConfig instance and whose value is a dict are considered. For each matching section, keys that correspond to attributes on the section object are assigned the provided values. Unknown sections, non-dict section values, and unknown keys on section objects are ignored. Mutates the existing sub-configuration objects in place and does not return a value.
        """
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
        """
        Validate and normalize AppConfig fields, mutating the instance in-place.
        
        Checks and enforces sane values for several sub-configurations, replacing invalid values with safe defaults and emitting warnings:
        - logging.level: must be one of ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]; defaults to "INFO".
        - gradio.port: must be in range 1–65535; defaults to 7860.
        - llm.temperature: must be between 0 and 2 (inclusive); defaults to 0.7.
        - llm.max_tokens: must be > 0; defaults to 2000.
        
        Side effects:
        - Mutates self to apply fallback values for invalid settings.
        - Emits warning-level log messages when replacements occur.
        """
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
    """
    Return an AppConfig populated from defaults, environment variables, and an optional YAML file.
    
    If provided, config_file is used to load additional settings (lower precedence than environment variables). The returned AppConfig has validation applied and contains nested sub-configurations (logging, gradio, llm, security, cache).
    
    Parameters:
        config_file (Optional[str]): Path to a YAML config file to load (optional).
    
    Returns:
        AppConfig: A fully initialized and validated application configuration object.
    """
    config = AppConfig(config_file=config_file)
    return config


def create_config_template(output_path: str = "config_template.yaml"):
    """
    Create a YAML configuration template file containing a subset of AppConfig fields.
    
    The template is populated from a default AppConfig instance (logging, gradio, llm, security, cache)
    and written to the given output_path as YAML. If a file already exists at output_path it will be
    overwritten. A short confirmation is printed after successful write.
    
    Parameters:
        output_path (str): Path where the YAML template will be written (default: "config_template.yaml").
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