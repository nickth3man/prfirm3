"""
Configuration management for the Virtual PR Firm.

This module provides centralized configuration management with environment variable
support, validation, and type safety for all application settings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM services."""
    model: str = "gpt-4o"
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    def __post_init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')

@dataclass
class ValidationConfig:
    """Configuration for validation settings."""
    max_topic_length: int = 1000
    max_brand_bible_size: int = 100000
    max_revision_count: int = 5
    strict_style_enforcement: bool = True
    
    # Platform limits
    platform_limits: Dict[str, int] = field(default_factory=lambda: {
        "twitter": 280,
        "linkedin": 3000,
        "facebook": 63206,
        "instagram": 2200,
        "reddit": 40000,
        "email": 100000,
        "blog": 100000
    })

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    enable_monitoring: bool = True
    metrics_history_size: int = 1000
    collection_interval: int = 60
    enable_system_metrics: bool = True

@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""
    enable_retries: bool = True
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    enable_fallbacks: bool = True

@dataclass
class StreamingConfig:
    """Configuration for streaming features."""
    enable_streaming: bool = True
    stream_milestones: bool = True
    stream_progress: bool = True
    stream_errors: bool = True
    max_stream_history: int = 100

@dataclass
class AppConfig:
    """Main application configuration."""
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Core components
    llm: LLMConfig = field(default_factory=LLMConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    
    # File paths
    config_file: Optional[str] = None
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        # Load from environment variables
        self._load_from_env()
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Environment
        self.environment = os.getenv('ENVIRONMENT', self.environment)
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        
        # LLM settings
        self.llm.model = os.getenv('LLM_MODEL', self.llm.model)
        self.llm.max_retries = int(os.getenv('LLM_MAX_RETRIES', self.llm.max_retries))
        self.llm.timeout = int(os.getenv('LLM_TIMEOUT', self.llm.timeout))
        self.llm.temperature = float(os.getenv('LLM_TEMPERATURE', self.llm.temperature))
        self.llm.max_tokens = int(os.getenv('LLM_MAX_TOKENS', self.llm.max_tokens))
        
        # Validation settings
        self.validation.max_topic_length = int(os.getenv('MAX_TOPIC_LENGTH', self.validation.max_topic_length))
        self.validation.max_brand_bible_size = int(os.getenv('MAX_BRAND_BIBLE_SIZE', self.validation.max_brand_bible_size))
        self.validation.max_revision_count = int(os.getenv('MAX_REVISION_COUNT', self.validation.max_revision_count))
        self.validation.strict_style_enforcement = os.getenv('STRICT_STYLE_ENFORCEMENT', 'true').lower() == 'true'
        
        # Performance settings
        self.performance.enable_monitoring = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        self.performance.metrics_history_size = int(os.getenv('METRICS_HISTORY_SIZE', self.performance.metrics_history_size))
        self.performance.collection_interval = int(os.getenv('COLLECTION_INTERVAL', self.performance.collection_interval))
        self.performance.enable_system_metrics = os.getenv('ENABLE_SYSTEM_METRICS', 'true').lower() == 'true'
        
        # Error handling settings
        self.error_handling.enable_retries = os.getenv('ENABLE_RETRIES', 'true').lower() == 'true'
        self.error_handling.max_retries = int(os.getenv('MAX_RETRIES', self.error_handling.max_retries))
        self.error_handling.base_delay = float(os.getenv('BASE_DELAY', self.error_handling.base_delay))
        self.error_handling.max_delay = float(os.getenv('MAX_DELAY', self.error_handling.max_delay))
        self.error_handling.exponential_backoff = os.getenv('EXPONENTIAL_BACKOFF', 'true').lower() == 'true'
        self.error_handling.enable_fallbacks = os.getenv('ENABLE_FALLBACKS', 'true').lower() == 'true'
        
        # Streaming settings
        self.streaming.enable_streaming = os.getenv('ENABLE_STREAMING', 'true').lower() == 'true'
        self.streaming.stream_milestones = os.getenv('STREAM_MILESTONES', 'true').lower() == 'true'
        self.streaming.stream_progress = os.getenv('STREAM_PROGRESS', 'true').lower() == 'true'
        self.streaming.stream_errors = os.getenv('STREAM_ERRORS', 'true').lower() == 'true'
        self.streaming.max_stream_history = int(os.getenv('MAX_STREAM_HISTORY', self.streaming.max_stream_history))
        
        # File paths
        self.data_dir = os.getenv('DATA_DIR', self.data_dir)
        self.logs_dir = os.getenv('LOGS_DIR', self.logs_dir)
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "llm": {
                "model": self.llm.model,
                "max_retries": self.llm.max_retries,
                "timeout": self.llm.timeout,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "has_openai_key": bool(self.llm.openai_api_key),
                "has_anthropic_key": bool(self.llm.anthropic_api_key),
                "has_gemini_key": bool(self.llm.gemini_api_key)
            },
            "validation": {
                "max_topic_length": self.validation.max_topic_length,
                "max_brand_bible_size": self.validation.max_brand_bible_size,
                "max_revision_count": self.validation.max_revision_count,
                "strict_style_enforcement": self.validation.strict_style_enforcement,
                "platform_limits": self.validation.platform_limits
            },
            "performance": {
                "enable_monitoring": self.performance.enable_monitoring,
                "metrics_history_size": self.performance.metrics_history_size,
                "collection_interval": self.performance.collection_interval,
                "enable_system_metrics": self.performance.enable_system_metrics
            },
            "error_handling": {
                "enable_retries": self.error_handling.enable_retries,
                "max_retries": self.error_handling.max_retries,
                "base_delay": self.error_handling.base_delay,
                "max_delay": self.error_handling.max_delay,
                "exponential_backoff": self.error_handling.exponential_backoff,
                "enable_fallbacks": self.error_handling.enable_fallbacks
            },
            "streaming": {
                "enable_streaming": self.streaming.enable_streaming,
                "stream_milestones": self.streaming.stream_milestones,
                "stream_progress": self.streaming.stream_progress,
                "stream_errors": self.streaming.stream_errors,
                "max_stream_history": self.streaming.max_stream_history
            },
            "paths": {
                "data_dir": self.data_dir,
                "logs_dir": self.logs_dir
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'AppConfig':
        """Load configuration from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            config = cls()
            
            # Update configuration from file
            if 'llm' in data:
                for key, value in data['llm'].items():
                    if hasattr(config.llm, key):
                        setattr(config.llm, key, value)
            
            if 'validation' in data:
                for key, value in data['validation'].items():
                    if hasattr(config.validation, key):
                        setattr(config.validation, key, value)
            
            if 'performance' in data:
                for key, value in data['performance'].items():
                    if hasattr(config.performance, key):
                        setattr(config.performance, key, value)
            
            if 'error_handling' in data:
                for key, value in data['error_handling'].items():
                    if hasattr(config.error_handling, key):
                        setattr(config.error_handling, key, value)
            
            if 'streaming' in data:
                for key, value in data['streaming'].items():
                    if hasattr(config.streaming, key):
                        setattr(config.streaming, key, value)
            
            logger.info(f"Configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            return cls()

class ConfigManager:
    """Manages application configuration with validation and updates."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate LLM settings
        if self.config.llm.max_retries < 0:
            raise ValueError("LLM max_retries must be non-negative")
        
        if not (0 <= self.config.llm.temperature <= 2):
            raise ValueError("LLM temperature must be between 0 and 2")
        
        if self.config.llm.max_tokens <= 0:
            raise ValueError("LLM max_tokens must be positive")
        
        # Validate validation settings
        if self.config.validation.max_topic_length <= 0:
            raise ValueError("max_topic_length must be positive")
        
        if self.config.validation.max_brand_bible_size <= 0:
            raise ValueError("max_brand_bible_size must be positive")
        
        if self.config.validation.max_revision_count <= 0:
            raise ValueError("max_revision_count must be positive")
        
        # Validate performance settings
        if self.config.performance.metrics_history_size <= 0:
            raise ValueError("metrics_history_size must be positive")
        
        if self.config.performance.collection_interval <= 0:
            raise ValueError("collection_interval must be positive")
        
        # Validate error handling settings
        if self.config.error_handling.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.config.error_handling.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        
        if self.config.error_handling.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        
        if self.config.error_handling.max_delay < self.config.error_handling.base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        
        # Validate streaming settings
        if self.config.streaming.max_stream_history <= 0:
            raise ValueError("max_stream_history must be positive")
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        logger.info(f"Updating configuration with: {updates}")
        
        # Update nested configuration objects
        if 'llm' in updates and isinstance(updates['llm'], dict):
            for key, value in updates['llm'].items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)
        
        if 'validation' in updates and isinstance(updates['validation'], dict):
            for key, value in updates['validation'].items():
                if hasattr(self.config.validation, key):
                    setattr(self.config.validation, key, value)
        
        if 'performance' in updates and isinstance(updates['performance'], dict):
            for key, value in updates['performance'].items():
                if hasattr(self.config.performance, key):
                    setattr(self.config.performance, key, value)
        
        if 'error_handling' in updates and isinstance(updates['error_handling'], dict):
            for key, value in updates['error_handling'].items():
                if hasattr(self.config.error_handling, key):
                    setattr(self.config.error_handling, key, value)
        
        if 'streaming' in updates and isinstance(updates['streaming'], dict):
            for key, value in updates['streaming'].items():
                if hasattr(self.config.streaming, key):
                    setattr(self.config.streaming, key, value)
        
        # Update top-level settings
        for key, value in updates.items():
            if hasattr(self.config, key) and not isinstance(value, dict):
                setattr(self.config, key, value)
        
        # Re-validate
        self._validate_config()
    
    def reload_from_env(self):
        """Reload configuration from environment variables."""
        self.config._load_from_env()
        self._validate_config()
        logger.info("Configuration reloaded from environment variables")

# Global configuration instance
_global_config = AppConfig()
_global_config_manager = ConfigManager(_global_config)

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return _global_config

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return _global_config_manager

def update_config(updates: Dict[str, Any]):
    """Update the global configuration."""
    _global_config_manager.update_config(updates)

def reload_config():
    """Reload the global configuration from environment variables."""
    _global_config_manager.reload_from_env()

# Test function for development
if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration:", config.to_dict())
    
    # Test configuration manager
    manager = get_config_manager()
    
    # Test validation
    try:
        manager.update_config({"llm": {"max_retries": -1}})
    except ValueError as e:
        print(f"Validation error (expected): {e}")
    
    # Test valid update
    manager.update_config({"debug": True})
    print("Updated config:", config.to_dict())