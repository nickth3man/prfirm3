"""Unit tests for the main module.

This module contains comprehensive unit tests for the Virtual PR Firm main module,
covering all major functionality including CLI demo, Gradio interface, and validation.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules to test
from main import run_demo, create_gradio_interface, validate_shared_store
from utils.config import AppConfig, get_config, set_config
from utils.validation import (
    validate_topic, validate_platforms, validate_shared_store as validate_shared_store_util,
    ValidationResult, ValidationError
)
from utils.error_handling import VirtualPRError, ValidationError as ValidationErrorClass
from utils.caching import CacheManager, get_cache_manager


class TestRunDemo:
    """Test cases for the run_demo function."""
    
    def test_run_demo_success(self):
        """Test that run_demo runs successfully with valid inputs."""
        with patch('main.create_main_flow') as mock_create_flow:
            mock_flow = Mock()
            mock_create_flow.return_value = mock_flow
            
            # Mock the flow.run method
            mock_flow.run = Mock()
            
            # Test that run_demo doesn't raise an exception
            run_demo()
            
            # Verify flow was created and run
            mock_create_flow.assert_called_once()
            mock_flow.run.assert_called_once()
    
    def test_run_demo_with_invalid_shared_store(self):
        """Test that run_demo handles invalid shared store gracefully."""
        with patch('main.validate_shared_store') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid shared store")
            
            with pytest.raises(ValueError, match="Invalid shared store"):
                run_demo()
    
    def test_run_demo_flow_execution_error(self):
        """Test that run_demo handles flow execution errors."""
        with patch('main.create_main_flow') as mock_create_flow:
            mock_flow = Mock()
            mock_flow.run.side_effect = Exception("Flow execution failed")
            mock_create_flow.return_value = mock_flow
            
            with pytest.raises(Exception, match="Flow execution failed"):
                run_demo()


class TestValidateSharedStore:
    """Test cases for the validate_shared_store function."""
    
    def test_validate_shared_store_valid(self):
        """Test validation with valid shared store."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None
        }
        
        # Should not raise an exception
        validate_shared_store(shared)
    
    def test_validate_shared_store_not_dict(self):
        """Test validation with non-dict input."""
        with pytest.raises(TypeError, match="shared must be a dict"):
            validate_shared_store("not a dict")
    
    def test_validate_shared_store_missing_task_requirements(self):
        """Test validation with missing task_requirements."""
        shared = {"brand_bible": {"xml_raw": ""}}
        
        with pytest.raises(ValueError, match="task_requirements is required"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_invalid_task_requirements(self):
        """Test validation with invalid task_requirements."""
        shared = {
            "task_requirements": "not a dict",
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValueError, match="task_requirements must be a dictionary"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_missing_platforms(self):
        """Test validation with missing platforms."""
        shared = {
            "task_requirements": {
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValueError, match="platforms is required"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_invalid_platforms_type(self):
        """Test validation with invalid platforms type."""
        shared = {
            "task_requirements": {
                "platforms": "not a list",
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(TypeError, match="platforms must be a list"):
            validate_shared_store(shared)


class TestCreateGradioInterface:
    """Test cases for the create_gradio_interface function."""
    
    @patch('main.gr')
    def test_create_gradio_interface_success(self, mock_gr):
        """Test successful Gradio interface creation."""
        # Mock Gradio components
        mock_blocks = Mock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks
        mock_gr.Blocks.return_value.__exit__.return_value = None
        
        mock_textbox = Mock()
        mock_gr.Textbox.return_value = mock_textbox
        
        mock_json = Mock()
        mock_gr.JSON.return_value = mock_json
        
        mock_button = Mock()
        mock_gr.Button.return_value = mock_button
        
        mock_markdown = Mock()
        mock_gr.Markdown.return_value = mock_markdown
        
        # Test interface creation
        result = create_gradio_interface()
        
        # Verify Gradio components were created
        mock_gr.Blocks.assert_called_once()
        mock_gr.Markdown.assert_called_once()
        mock_gr.Textbox.assert_called()
        mock_gr.JSON.assert_called_once()
        mock_gr.Button.assert_called_once()
        
        assert result == mock_gr.Blocks.return_value
    
    @patch('main.gr', None)
    def test_create_gradio_interface_gradio_not_installed(self):
        """Test Gradio interface creation when Gradio is not installed."""
        with pytest.raises(RuntimeError, match="Gradio not installed"):
            create_gradio_interface()


class TestValidationUtilities:
    """Test cases for validation utility functions."""
    
    def test_validate_topic_valid(self):
        """Test topic validation with valid input."""
        result = validate_topic("Valid topic for testing")
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_topic_empty(self):
        """Test topic validation with empty input."""
        result = validate_topic("")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].field == "topic"
        assert "cannot be empty" in result.errors[0].message
    
    def test_validate_topic_too_short(self):
        """Test topic validation with too short input."""
        result = validate_topic("ab")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "at least 3 characters" in result.errors[0].message
    
    def test_validate_topic_inappropriate_content(self):
        """Test topic validation with inappropriate content."""
        result = validate_topic("This is a hack attempt")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "inappropriate content" in result.errors[0].message
    
    def test_validate_platforms_valid_list(self):
        """Test platform validation with valid list."""
        result = validate_platforms(["twitter", "linkedin"])
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_platforms_valid_string(self):
        """Test platform validation with valid string."""
        result = validate_platforms("twitter, linkedin")
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_platforms_empty(self):
        """Test platform validation with empty input."""
        result = validate_platforms([])
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "At least one platform" in result.errors[0].message
    
    def test_validate_platforms_unsupported(self):
        """Test platform validation with unsupported platform."""
        result = validate_platforms(["unsupported_platform"])
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Unsupported platform" in result.errors[0].message
    
    def test_validate_shared_store_util_valid(self):
        """Test shared store validation utility with valid input."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": "Test topic"
            }
        }
        
        result = validate_shared_store_util(shared)
        assert result.is_valid
        assert len(result.errors) == 0


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_app_config_defaults(self):
        """Test AppConfig with default values."""
        config = AppConfig()
        
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o"
        assert config.debug_mode is False
        assert config.log_level == "INFO"
        assert "twitter" in config.supported_platforms
    
    def test_app_config_validation(self):
        """Test AppConfig validation."""
        # Test invalid max_retries
        with pytest.raises(ValueError, match="must be non-negative"):
            AppConfig(llm_max_retries=-1)
        
        # Test invalid timeout
        with pytest.raises(ValueError, match="must be at least 1 second"):
            AppConfig(llm_timeout=0)
        
        # Test invalid port
        with pytest.raises(ValueError, match="must be between 1 and 65535"):
            AppConfig(gradio_port=0)
    
    def test_app_config_to_dict(self):
        """Test AppConfig serialization to dictionary."""
        config = AppConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["llm_provider"] == "openai"
        assert config_dict["debug_mode"] is False
    
    def test_app_config_from_file(self):
        """Test AppConfig loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "llm_provider": "anthropic",
                "llm_model": "claude-3",
                "debug_mode": True
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = AppConfig.from_file(config_path)
            assert config.llm_provider == "anthropic"
            assert config.llm_model == "claude-3"
            assert config.debug_mode is True
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_get_config_singleton(self):
        """Test that get_config returns a singleton instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_set_config(self):
        """Test setting custom configuration."""
        custom_config = AppConfig(llm_provider="custom")
        set_config(custom_config)
        
        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.llm_provider == "custom"


class TestCaching:
    """Test cases for caching functionality."""
    
    def test_cache_manager_creation(self):
        """Test CacheManager creation and basic operations."""
        cache_manager = CacheManager()
        
        # Test setting and getting values
        cache_manager.set("test_key", "test_value", ttl=60)
        result = cache_manager.get("test_key")
        assert result == "test_value"
        
        # Test deletion
        deleted = cache_manager.delete("test_key")
        assert deleted is True
        
        # Test getting non-existent key
        result = cache_manager.get("non_existent")
        assert result is None
    
    def test_cache_manager_stats(self):
        """Test CacheManager statistics."""
        cache_manager = CacheManager()
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        stats = cache_manager.get_stats()
        assert stats["memory_cache_enabled"] is True
        assert stats["file_cache_enabled"] is True
        assert stats["memory_cache_size"] == 2
    
    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns a singleton instance."""
        cache_manager1 = get_cache_manager()
        cache_manager2 = get_cache_manager()
        assert cache_manager1 is cache_manager2


class TestErrorHandling:
    """Test cases for error handling utilities."""
    
    def test_virtual_pr_error_creation(self):
        """Test VirtualPRError creation and properties."""
        error = VirtualPRError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.severity.value == "medium"
        assert error.timestamp > 0
    
    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        error = ValidationErrorClass("Validation failed", field="test_field", value="test_value")
        
        assert error.field == "test_field"
        assert error.value == "test_value"
        assert error.severity.value == "medium"


if __name__ == "__main__":
    pytest.main([__file__])