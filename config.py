"""Configuration management for the Virtual PR Firm application.

This module provides a centralized configuration system that supports:
- Environment variable overrides
- Configuration file loading (YAML/JSON)
- Sensible defaults
- Type validation and error handling
- Hierarchical configuration management
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "json"  # "json" or "human"
    file: Optional[str] = None
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    correlation_id: bool = True


@dataclass
class GradioConfig:
    """Configuration for Gradio web interface."""
    port: int = 7860
    host: str = "0.0.0.0"
    share: bool = False
    auth: Optional[str] = None  # username:password
    auth_message: str = "Please enter your credentials"
    ssl_verify: bool = True
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    favicon_path: Optional[str] = None
    app_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "openai"  # "openai", "anthropic", "openrouter"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 1
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class FlowConfig:
    """Configuration for flow execution."""
    timeout: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: int = 5
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_size: int = 1000


@dataclass
class SecurityConfig:
    """Configuration for security features."""
    enable_auth: bool = False
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 60  # requests per minute
    rate_limit_window: int = 60  # seconds
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_origins: list = field(default_factory=lambda: ["*"])
    session_timeout: int = 3600  # 1 hour


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = "human"
    log_file: Optional[str] = None
    
    # Sub-configurations
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gradio: GradioConfig = field(default_factory=GradioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ConfigManager:
    """Manages application configuration with hierarchical loading."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize a ConfigManager.
        
        Parameters:
            config_file (Optional[str]): Path to a YAML/JSON configuration file to load. If None, only defaults and environment overrides are used.
        
        Description:
            Stores the optional config file path and prepares an internal cached AppConfig (initially None) that will be populated when load() is called.
        """
        self.config_file = config_file
        self._config: Optional[AppConfig] = None
    
    def load(self) -> AppConfig:
        """
        Load and return the application configuration, merged from defaults, an optional config file, and environment variables.
        
        Performs a hierarchical merge in this order:
        1. default values
        2. values from the configured YAML/JSON file (if provided)
        3. environment variable overrides
        
        The result is validated, cached, and returned.
        
        Returns:
            AppConfig: The merged and validated application configuration.
        
        Raises:
            ConfigurationError: If validation of the merged configuration fails.
        """
        if self._config is not None:
            return self._config
        
        # Start with defaults
        config = self._get_defaults()
        
        # Load from config file if specified
        if self.config_file:
            config = self._merge_config(config, self._load_config_file())
        
        # Override with environment variables
        config = self._merge_config(config, self._load_from_env())
        
        # Validate configuration
        self._validate_config(config)
        
        self._config = config
        return config
    
    def _get_defaults(self) -> AppConfig:
        """
        Return a new AppConfig populated with module defaults.
        
        Returns:
            AppConfig: A fresh AppConfig instance with default values for all sub-configurations.
        """
        return AppConfig()
    
    def _load_config_file(self) -> Dict[str, Any]:
        """
        Load configuration from the configured file path.
        
        Attempts to read and parse the file at self.config_file. Supports YAML (".yaml", ".yml") via yaml.safe_load and JSON (".json") via json.load. If the file path is not set or the file does not exist, or if parsing fails, an empty dict is returned. If the file has an unsupported extension, raises ConfigurationError.
        
        Returns:
            dict: The parsed configuration as a dictionary, or an empty dict on missing file or parse error.
        
        Raises:
            ConfigurationError: If the file has an unsupported extension.
        """
        if not self.config_file or not Path(self.config_file).exists():
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                elif self.config_file.endswith('.json'):
                    return json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_file}: {e}")
            return {}
    
    def _load_from_env(self) -> Dict[str, Any]:
        """
        Builds a nested override dictionary from environment variables to apply on top of defaults.
        
        Reads specific environment variables and converts them into typed values placed into sections matching the AppConfig structure (top-level keys: debug, environment, logging, gradio, llm, security). Parsing behavior and mappings:
        - Booleans: parsed case-insensitively from ('true', '1', 'yes').
        - Integers: parsed via int() for port and rate-limit values.
        - DEMO_PASSWORD → gradio.auth: stored as "admin:<password>".
        - OPENAI_API_KEY and ANTHROPIC_API_KEY → llm.api_key (the last one present wins).
        - Supported environment variables:
          - App: DEBUG (bool), ENVIRONMENT (str)
          - Logging: LOG_LEVEL (str), LOG_FORMAT (str), LOG_FILE (str)
          - Gradio: GRADIO_PORT (int), GRADIO_HOST (str), GRADIO_SHARE (bool), DEMO_PASSWORD (str)
          - LLM: LLM_PROVIDER (str), LLM_MODEL (str), OPENAI_API_KEY (str), ANTHROPIC_API_KEY (str), LLM_BASE_URL (str)
          - Security: ENABLE_AUTH (bool), RATE_LIMIT_REQUESTS (int)
        
        Returns:
            Dict[str, Any]: A nested dict of overrides suitable for deep-merging into the default configuration.
        """
        config = {}
        
        # App-level settings
        if os.getenv('DEBUG'):
            config['debug'] = os.getenv('DEBUG').lower() in ('true', '1', 'yes')
        if os.getenv('ENVIRONMENT'):
            config['environment'] = os.getenv('ENVIRONMENT')
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FORMAT'):
            config.setdefault('logging', {})['format'] = os.getenv('LOG_FORMAT')
        if os.getenv('LOG_FILE'):
            config.setdefault('logging', {})['file'] = os.getenv('LOG_FILE')
        
        # Gradio
        if os.getenv('GRADIO_PORT'):
            config.setdefault('gradio', {})['port'] = int(os.getenv('GRADIO_PORT'))
        if os.getenv('GRADIO_HOST'):
            config.setdefault('gradio', {})['host'] = os.getenv('GRADIO_HOST')
        if os.getenv('GRADIO_SHARE'):
            config.setdefault('gradio', {})['share'] = os.getenv('GRADIO_SHARE').lower() in ('true', '1', 'yes')
        if os.getenv('DEMO_PASSWORD'):
            config.setdefault('gradio', {})['auth'] = f"admin:{os.getenv('DEMO_PASSWORD')}"
        
        # LLM
        if os.getenv('LLM_PROVIDER'):
            config.setdefault('llm', {})['provider'] = os.getenv('LLM_PROVIDER')
        if os.getenv('LLM_MODEL'):
            config.setdefault('llm', {})['model'] = os.getenv('LLM_MODEL')
        if os.getenv('OPENAI_API_KEY'):
            config.setdefault('llm', {})['api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            config.setdefault('llm', {})['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        if os.getenv('LLM_BASE_URL'):
            config.setdefault('llm', {})['base_url'] = os.getenv('LLM_BASE_URL')
        
        # Security
        if os.getenv('ENABLE_AUTH'):
            config.setdefault('security', {})['enable_auth'] = os.getenv('ENABLE_AUTH').lower() in ('true', '1', 'yes')
        if os.getenv('RATE_LIMIT_REQUESTS'):
            config.setdefault('security', {})['rate_limit_requests'] = int(os.getenv('RATE_LIMIT_REQUESTS'))
        
        return config
    
    def _merge_config(self, base: AppConfig, override: Dict[str, Any]) -> AppConfig:
        """
        Return a new AppConfig created by deep-merging a nested override dictionary into a base AppConfig.
        
        Parameters:
            base (AppConfig): The source configuration to use as defaults.
            override (Dict[str, Any]): A nested dictionary of overrides to apply on top of `base`. Keys map to AppConfig fields and nested sub-config sections.
        
        Returns:
            AppConfig: A new AppConfig instance representing the result of merging `override` on top of `base`. The original `base` is not modified.
        """
        if not override:
            return base
        
        # Convert base config to dict for merging
        config_dict = self._config_to_dict(base)
        
        # Recursively merge
        merged = self._deep_merge(config_dict, override)
        
        # Convert back to AppConfig
        return self._dict_to_config(merged)
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """
        Convert an AppConfig (or nested config dataclass) into a plain dictionary.
        
        Recursively traverses the given config object's attributes and converts any nested objects
        that expose a __dict__ into dictionaries so the entire configuration tree becomes
        serializable as standard Python dictionaries and primitive values.
        
        Parameters:
            config (AppConfig): Top-level configuration object (or any nested config object)
                to convert.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the configuration with nested
            sections converted to dictionaries.
        """
        result = {}
        for field_name, field_value in config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = self._config_to_dict(field_value)
            else:
                result[field_name] = field_value
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """
        Convert a nested plain dictionary into an AppConfig instance.
        
        Takes a mapping of configuration values (typically produced by merging defaults,
        file, and environment overrides) and constructs typed sub-config dataclasses
        (LoggingConfig, GradioConfig, LLMConfig, FlowConfig, SecurityConfig) which are
        assembled into the returned AppConfig. Missing top-level or sub-config keys
        will be filled with the dataclasses' defaults; values present in the dictionary
        are passed to the corresponding dataclass constructors.
        
        Parameters:
            config_dict (Dict[str, Any]): Nested configuration dictionary with optional
                top-level keys like "debug", "environment", "log_level", "log_format",
                "log_file", "logging", "gradio", "llm", "flow", "security", and "custom".
        
        Returns:
            AppConfig: Fully constructed application configuration object reflecting
            the provided overrides merged with dataclass defaults.
        """
        # Handle sub-configurations
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        gradio_config = GradioConfig(**config_dict.get('gradio', {}))
        llm_config = LLMConfig(**config_dict.get('llm', {}))
        flow_config = FlowConfig(**config_dict.get('flow', {}))
        security_config = SecurityConfig(**config_dict.get('security', {}))
        
        # Create main config
        return AppConfig(
            debug=config_dict.get('debug', False),
            environment=config_dict.get('environment', 'development'),
            log_level=config_dict.get('log_level', 'INFO'),
            log_format=config_dict.get('log_format', 'human'),
            log_file=config_dict.get('log_file'),
            logging=logging_config,
            gradio=gradio_config,
            llm=llm_config,
            flow=flow_config,
            security=security_config,
            custom=config_dict.get('custom', {})
        )
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep-merge two mapping objects and return a new dictionary.
        
        Performs a recursive merge of `override` into `base`: when a key exists in both and both
        values are dictionaries, their contents are merged recursively; otherwise the value from
        `override` replaces the one from `base`. The inputs are not mutated — a shallow copy of
        `base` is created and returned with overrides applied.
        
        Parameters:
            base (Dict[str, Any]): The base mapping to merge into.
            override (Dict[str, Any]): Mapping whose values override or extend `base`.
        
        Returns:
            Dict[str, Any]: A new dictionary with `override` applied on top of `base`.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self, config: AppConfig) -> None:
        """
        Validate critical AppConfig fields and raise ConfigurationError on invalid values.
        
        Performs the following checks:
        - logging.level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive).
        - gradio.port must be an integer in the range 1–65535.
        - llm.provider must be one of: 'openai', 'anthropic', 'openrouter'.
        - llm.temperature must be within 0.0–2.0 inclusive.
        - flow.timeout and llm.timeout must be positive.
        
        Parameters:
            config (AppConfig): The configuration instance to validate.
        
        Raises:
            ConfigurationError: If any validation rule is violated (message describes the invalid field).
        """
        # Validate logging level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level.upper() not in valid_log_levels:
            raise ConfigurationError(f"Invalid log level: {config.logging.level}")
        
        # Validate port range
        if not (1 <= config.gradio.port <= 65535):
            raise ConfigurationError(f"Invalid port number: {config.gradio.port}")
        
        # Validate LLM provider
        valid_providers = ['openai', 'anthropic', 'openrouter']
        if config.llm.provider not in valid_providers:
            raise ConfigurationError(f"Invalid LLM provider: {config.llm.provider}")
        
        # Validate temperature range
        if not (0.0 <= config.llm.temperature <= 2.0):
            raise ConfigurationError(f"Invalid temperature: {config.llm.temperature}")
        
        # Validate timeouts
        if config.flow.timeout <= 0:
            raise ConfigurationError(f"Invalid flow timeout: {config.flow.timeout}")
        if config.llm.timeout <= 0:
            raise ConfigurationError(f"Invalid LLM timeout: {config.llm.timeout}")


@lru_cache(maxsize=1)
def get_config(config_file: Optional[str] = None) -> AppConfig:
    """
    Load and return the application's AppConfig, using an internal cache to avoid repeated parsing.
    
    If a config_file path is provided, its settings (YAML or JSON) are merged over built-in defaults and then environment variable overrides are applied. The resulting configuration is validated before being returned and cached for subsequent calls.
    
    Parameters:
        config_file (Optional[str]): Path to a YAML or JSON config file to merge over defaults. If None, only defaults and environment overrides are used.
    
    Returns:
        AppConfig: The merged, validated application configuration.
    
    Raises:
        ConfigurationError: If the configuration file is an unsupported format or the final configuration fails validation.
    """
    config_manager = ConfigManager(config_file)
    return config_manager.load()


def configure_logging(config: AppConfig) -> None:
    """
    Configure the Python logging system according to the provided AppConfig.
    
    This sets the root logger level, installs a console handler (and a rotating file handler if
    config.logging.file is set), selects either a JSON-style or human-readable formatter based
    on config.logging.format, and forces the logging configuration. If config.logging.correlation_id
    is enabled, initializes a module-level placeholder (logging.correlation_id) for per-request use.
    
    Parameters:
        config (AppConfig): Application configuration containing the `logging` sub-config.
    """
    import logging.config
    
    # Determine log level
    log_level = getattr(logging, config.logging.level.upper())
    
    # Configure log format
    if config.logging.format == 'json':
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"correlation_id": "%(correlation_id)s" if hasattr(logging, "correlation_id") else ""}'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if config.logging.file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.logging.file,
            maxBytes=config.logging.max_size,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Set correlation ID if enabled
    if config.logging.correlation_id:
        logging.correlation_id = None  # Will be set per request
    
    logger.info(f"Logging configured with level={config.logging.level}, format={config.logging.format}")


def create_config_template(output_path: str = "config.yaml") -> None:
    """
    Write a YAML configuration template containing all top-level and sub-config defaults.
    
    Creates a YAML file at `output_path` with the canonical configuration structure (debug, environment,
    logging, gradio, llm, flow, security, and custom) populated with sensible defaults. If the file
    already exists it will be overwritten.
    
    Parameters:
        output_path (str): Filesystem path where the YAML template will be written (default: "config.yaml").
    
    Returns:
        None
    """
    template = {
        "debug": False,
        "environment": "development",
        "log_level": "INFO",
        "log_format": "human",
        "log_file": None,
        "logging": {
            "level": "INFO",
            "format": "human",
            "file": None,
            "max_size": 10485760,
            "backup_count": 5,
            "correlation_id": True
        },
        "gradio": {
            "port": 7860,
            "host": "0.0.0.0",
            "share": False,
            "auth": None,
            "auth_message": "Please enter your credentials",
            "ssl_verify": True,
            "ssl_keyfile": None,
            "ssl_certfile": None,
            "favicon_path": None,
            "app_kwargs": {}
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4000,
            "timeout": 60,
            "retries": 3,
            "retry_delay": 1,
            "api_key": None,
            "base_url": None
        },
        "flow": {
            "timeout": 300,
            "max_retries": 3,
            "retry_delay": 5,
            "enable_streaming": True,
            "enable_caching": True,
            "cache_ttl": 3600,
            "cache_size": 1000
        },
        "security": {
            "enable_auth": False,
            "enable_rate_limiting": True,
            "rate_limit_requests": 60,
            "rate_limit_window": 60,
            "max_request_size": 10485760,
            "allowed_origins": ["*"],
            "session_timeout": 3600
        },
        "custom": {}
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration template created at {output_path}")