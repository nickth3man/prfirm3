"""Unit tests for configuration management."""

import pytest
import tempfile
import os
import yaml
import json
from pathlib import Path

from config import (
    get_config, create_config_template, ConfigManager, 
    AppConfig, LoggingConfig, GradioConfig, LLMConfig, 
    FlowConfig, SecurityConfig, ConfigurationError
)


class TestConfigClasses:
    """Test configuration dataclasses."""
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file is None
        assert config.max_size == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.correlation_id is True
    
    def test_gradio_config_defaults(self):
        """Test GradioConfig default values."""
        config = GradioConfig()
        assert config.port == 7860
        assert config.host == "0.0.0.0"
        assert config.share is False
        assert config.auth is None
        assert config.auth_message == "Please enter your credentials"
        assert config.ssl_verify is True
    
    def test_llm_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
        assert config.timeout == 60
        assert config.retries == 3
        assert config.retry_delay == 1
    
    def test_flow_config_defaults(self):
        """Test FlowConfig default values."""
        config = FlowConfig()
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.retry_delay == 5
        assert config.enable_streaming is True
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.cache_size == 1000
    
    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig()
        assert config.enable_auth is False
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 60
        assert config.rate_limit_window == 60
        assert config.max_request_size == 10 * 1024 * 1024
        assert config.allowed_origins == ["*"]
        assert config.session_timeout == 3600
    
    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        assert config.debug is False
        assert config.environment == "development"
        assert config.log_level == "INFO"
        assert config.log_format == "human"
        assert config.log_file is None
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.gradio, GradioConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.flow, FlowConfig)
        assert isinstance(config.security, SecurityConfig)
        assert config.custom == {}


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_get_defaults(self):
        """Test getting default configuration."""
        manager = ConfigManager()
        config = manager._get_defaults()
        assert isinstance(config, AppConfig)
        assert config.debug is False
        assert config.environment == "development"
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ['DEBUG'] = 'true'
        os.environ['ENVIRONMENT'] = 'production'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        os.environ['GRADIO_PORT'] = '8080'
        os.environ['LLM_PROVIDER'] = 'anthropic'
        
        try:
            manager = ConfigManager()
            env_config = manager._load_from_env()
            
            assert env_config['debug'] is True
            assert env_config['environment'] == 'production'
            assert env_config['logging']['level'] == 'DEBUG'
            assert env_config['gradio']['port'] == 8080
            assert env_config['llm']['provider'] == 'anthropic'
        finally:
            # Clean up environment variables
            for key in ['DEBUG', 'ENVIRONMENT', 'LOG_LEVEL', 'GRADIO_PORT', 'LLM_PROVIDER']:
                os.environ.pop(key, None)
    
    def test_load_config_file_yaml(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'debug': True,
            'environment': 'test',
            'logging': {
                'level': 'DEBUG',
                'format': 'human'
            },
            'gradio': {
                'port': 9000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigManager(config_file)
            file_config = manager._load_config_file()
            
            assert file_config['debug'] is True
            assert file_config['environment'] == 'test'
            assert file_config['logging']['level'] == 'DEBUG'
            assert file_config['gradio']['port'] == 9000
        finally:
            os.unlink(config_file)
    
    def test_load_config_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            'debug': True,
            'environment': 'test',
            'logging': {
                'level': 'DEBUG',
                'format': 'human'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigManager(config_file)
            file_config = manager._load_config_file()
            
            assert file_config['debug'] is True
            assert file_config['environment'] == 'test'
            assert file_config['logging']['level'] == 'DEBUG'
        finally:
            os.unlink(config_file)
    
    def test_merge_config(self):
        """Test configuration merging."""
        base_config = AppConfig()
        override = {
            'debug': True,
            'logging': {
                'level': 'DEBUG'
            }
        }
        
        manager = ConfigManager()
        merged = manager._merge_config(base_config, override)
        
        assert merged.debug is True
        assert merged.logging.level == 'DEBUG'
        # Other values should remain unchanged
        assert merged.environment == 'development'
        assert merged.gradio.port == 7860
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid values."""
        config = AppConfig()
        manager = ConfigManager()
        
        # Should not raise an exception
        manager._validate_config(config)
    
    def test_validate_config_invalid_log_level(self):
        """Test configuration validation with invalid log level."""
        config = AppConfig()
        config.logging.level = "INVALID"
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Invalid log level"):
            manager._validate_config(config)
    
    def test_validate_config_invalid_port(self):
        """Test configuration validation with invalid port."""
        config = AppConfig()
        config.gradio.port = 70000  # Invalid port
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Invalid port number"):
            manager._validate_config(config)
    
    def test_validate_config_invalid_provider(self):
        """Test configuration validation with invalid LLM provider."""
        config = AppConfig()
        config.llm.provider = "invalid_provider"
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Invalid LLM provider"):
            manager._validate_config(config)
    
    def test_validate_config_invalid_temperature(self):
        """Test configuration validation with invalid temperature."""
        config = AppConfig()
        config.llm.temperature = 3.0  # Invalid temperature
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Invalid temperature"):
            manager._validate_config(config)
    
    def test_validate_config_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        config = AppConfig()
        config.flow.timeout = -1  # Invalid timeout
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Invalid flow timeout"):
            manager._validate_config(config)


class TestConfigFunctions:
    """Test configuration utility functions."""
    
    def test_get_config_default(self):
        """Test getting default configuration."""
        config = get_config()
        assert isinstance(config, AppConfig)
        assert config.debug is False
        assert config.environment == 'development'
    
    def test_get_config_with_file(self):
        """Test getting configuration with file."""
        config_data = {
            'debug': True,
            'environment': 'test'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config = get_config(config_file)
            assert config.debug is True
            assert config.environment == 'test'
        finally:
            os.unlink(config_file)
    
    def test_create_config_template(self):
        """Test creating configuration template."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            template_file = f.name
        
        try:
            create_config_template(template_file)
            
            # Verify file was created
            assert Path(template_file).exists()
            
            # Verify content is valid YAML
            with open(template_file, 'r') as f:
                template_data = yaml.safe_load(f)
            
            assert 'debug' in template_data
            assert 'logging' in template_data
            assert 'gradio' in template_data
            assert 'llm' in template_data
            assert 'flow' in template_data
            assert 'security' in template_data
            assert 'custom' in template_data
        finally:
            os.unlink(template_file)
    
    def test_get_config_caching(self):
        """Test that get_config uses caching."""
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same object due to caching
        assert config1 is config2
    
    def test_config_manager_load(self):
        """Test ConfigManager load method."""
        manager = ConfigManager()
        config = manager.load()
        
        assert isinstance(config, AppConfig)
        assert config.debug is False
        
        # Second load should return cached config
        config2 = manager.load()
        assert config is config2


class TestConfigurationError:
    """Test ConfigurationError exception."""
    
    def test_configuration_error_message(self):
        """Test ConfigurationError message."""
        error = ConfigurationError("Test error message")
        assert str(error) == "Test error message"
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Test")
        assert isinstance(error, Exception)


if __name__ == '__main__':
    pytest.main([__file__])