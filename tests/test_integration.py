"""Integration tests for the Virtual PR Firm.

This module contains integration tests that verify the complete flow execution
and end-to-end functionality of the Virtual PR Firm application.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

# Import the modules to test
from main import run_demo, create_gradio_interface
from flow import create_main_flow
from utils.config import AppConfig, get_config
from utils.validation import validate_and_sanitize_inputs
from utils.caching import CacheManager
from utils.error_handling import setup_error_handling


class TestCompleteFlowExecution:
    """Integration tests for complete flow execution."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Set up error handling
        setup_error_handling(log_level="WARNING")
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('flow.create_main_flow')
    def test_complete_flow_with_valid_inputs(self, mock_create_flow):
        """Test complete flow execution with valid inputs."""
        # Mock the flow
        mock_flow = Mock()
        mock_create_flow.return_value = mock_flow
        
        # Mock the flow.run method to simulate successful execution
        def mock_run(shared):
            shared["content_pieces"] = {
                "twitter": "Exciting news! Our new product is here...",
                "linkedin": "We are pleased to announce the launch of our latest innovation..."
            }
            shared["workflow_state"] = "completed"
        
        mock_flow.run.side_effect = mock_run
        
        # Test the complete flow
        run_demo()
        
        # Verify flow was created and executed
        mock_create_flow.assert_called_once()
        mock_flow.run.assert_called_once()
        
        # Verify the shared store was properly structured
        call_args = mock_flow.run.call_args[0][0]
        assert "task_requirements" in call_args
        assert "platforms" in call_args["task_requirements"]
        assert "topic_or_goal" in call_args["task_requirements"]
    
    def test_flow_with_real_validation(self):
        """Test flow with real validation utilities."""
        # Test input validation and sanitization
        topic = "Announce new product launch"
        platforms = "twitter, linkedin"
        
        validation_result, sanitized_shared = validate_and_sanitize_inputs(topic, platforms)
        
        # Verify validation passed
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        
        # Verify sanitized shared store structure
        assert "task_requirements" in sanitized_shared
        assert sanitized_shared["task_requirements"]["platforms"] == ["twitter", "linkedin"]
        assert sanitized_shared["task_requirements"]["topic_or_goal"] == topic
    
    def test_flow_with_invalid_inputs(self):
        """Test flow behavior with invalid inputs."""
        # Test with empty topic
        topic = ""
        platforms = "twitter"
        
        validation_result, sanitized_shared = validate_and_sanitize_inputs(topic, platforms)
        
        # Verify validation failed
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        
        # Check that the error is about empty topic
        topic_errors = [e for e in validation_result.errors if e.field == "topic"]
        assert len(topic_errors) > 0
        assert "cannot be empty" in topic_errors[0].message
    
    def test_flow_with_unsupported_platforms(self):
        """Test flow behavior with unsupported platforms."""
        topic = "Valid topic"
        platforms = "unsupported_platform"
        
        validation_result, sanitized_shared = validate_and_sanitize_inputs(topic, platforms)
        
        # Verify validation failed
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        
        # Check that the error is about unsupported platform
        platform_errors = [e for e in validation_result.errors if "platform" in e.field.lower()]
        assert len(platform_errors) > 0
        assert "Unsupported platform" in platform_errors[0].message


class TestGradioInterfaceIntegration:
    """Integration tests for Gradio interface."""
    
    @patch('main.gr')
    def test_gradio_interface_creation_and_execution(self, mock_gr):
        """Test Gradio interface creation and flow execution."""
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
        
        # Create the interface
        interface = create_gradio_interface()
        
        # Verify interface was created
        assert interface == mock_gr.Blocks.return_value
        
        # Test the run_flow function (extract it from the interface)
        # This is a bit tricky since it's nested, but we can test the logic
        with patch('main.create_main_flow') as mock_create_flow:
            mock_flow = Mock()
            mock_create_flow.return_value = mock_flow
            
            def mock_run(shared):
                shared["content_pieces"] = {
                    "twitter": "Test content for twitter",
                    "linkedin": "Test content for linkedin"
                }
            
            mock_flow.run.side_effect = mock_run
            
            # Test the run_flow logic directly
            from main import run_flow
            result = run_flow("Test topic", "twitter, linkedin")
            
            # Verify the result
            assert "twitter" in result
            assert "linkedin" in result
            assert "Test content for twitter" in result["twitter"]
            assert "Test content for linkedin" in result["linkedin"]


class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    def test_configuration_with_environment_variables(self):
        """Test configuration loading with environment variables."""
        import os
        
        # Set environment variables
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["LLM_MODEL"] = "claude-3"
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        try:
            # Create new config instance
            config = AppConfig()
            
            # Verify environment variables were loaded
            assert config.llm_provider == "anthropic"
            assert config.llm_model == "claude-3"
            assert config.debug_mode is True
            assert config.log_level == "DEBUG"
        finally:
            # Clean up environment variables
            for key in ["LLM_PROVIDER", "LLM_MODEL", "DEBUG", "LOG_LEVEL"]:
                os.environ.pop(key, None)
    
    def test_configuration_file_operations(self):
        """Test configuration file save and load operations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Create configuration
            config = AppConfig(
                llm_provider="test_provider",
                llm_model="test_model",
                debug_mode=True
            )
            
            # Save to file
            config.save_to_file(config_path)
            
            # Verify file was created
            assert Path(config_path).exists()
            
            # Load from file
            loaded_config = AppConfig.from_file(config_path)
            
            # Verify configuration was loaded correctly
            assert loaded_config.llm_provider == "test_provider"
            assert loaded_config.llm_model == "test_model"
            assert loaded_config.debug_mode is True
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestCachingIntegration:
    """Integration tests for caching functionality."""
    
    def test_caching_with_flow_execution(self):
        """Test caching with flow execution."""
        # Create cache manager
        cache_manager = CacheManager()
        
        # Test caching of flow results
        test_key = "flow_result_test"
        test_value = {
            "twitter": "Cached content for twitter",
            "linkedin": "Cached content for linkedin"
        }
        
        # Set cache
        cache_manager.set(test_key, test_value, ttl=60)
        
        # Get from cache
        cached_value = cache_manager.get(test_key)
        
        # Verify cached value
        assert cached_value == test_value
        assert "twitter" in cached_value
        assert "linkedin" in cached_value
        
        # Test cache statistics
        stats = cache_manager.get_stats()
        assert stats["memory_cache_enabled"] is True
        assert stats["file_cache_enabled"] is True
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        cache_manager = CacheManager()
        
        # Add some test entries
        cache_manager.set("key1", "value1", ttl=1)  # Short TTL
        cache_manager.set("key2", "value2", ttl=3600)  # Long TTL
        
        # Wait for first entry to expire
        import time
        time.sleep(2)
        
        # Clean up expired entries
        removed_count = cache_manager.cleanup()
        
        # Verify cleanup worked
        assert removed_count >= 1
        
        # Verify expired entry is gone
        assert cache_manager.get("key1") is None
        
        # Verify non-expired entry is still there
        assert cache_manager.get("key2") == "value2"


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""
    
    def test_error_handling_with_invalid_flow(self):
        """Test error handling with invalid flow execution."""
        from utils.error_handling import VirtualPRError, safe_execute
        
        # Test safe execution with function that raises an exception
        def failing_function():
            raise ValueError("Test error")
        
        # Execute safely
        result = safe_execute(failing_function, default_return="fallback_value")
        
        # Verify fallback value was returned
        assert result == "fallback_value"
    
    def test_error_handling_with_validation_errors(self):
        """Test error handling with validation errors."""
        from utils.validation import ValidationResult
        
        # Test validation with multiple errors
        result = ValidationResult()
        result.add_error("topic", "Topic is too short")
        result.add_error("platforms", "No platforms specified")
        
        # Verify error collection
        assert not result.is_valid
        assert len(result.errors) == 2
        assert result.errors[0].field == "topic"
        assert result.errors[1].field == "platforms"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @patch('flow.create_main_flow')
    def test_complete_workflow_from_input_to_output(self, mock_create_flow):
        """Test complete workflow from user input to final output."""
        # Mock the flow
        mock_flow = Mock()
        mock_create_flow.return_value = mock_flow
        
        # Mock successful flow execution
        def mock_run(shared):
            shared["content_pieces"] = {
                "twitter": "🚀 Exciting news! Our revolutionary new product is now available...",
                "linkedin": "We are thrilled to announce the launch of our latest innovation..."
            }
            shared["workflow_state"] = "completed"
            shared["metadata"] = {
                "generated_at": "2024-01-01T12:00:00Z",
                "platforms_processed": 2,
                "total_characters": 150
            }
        
        mock_flow.run.side_effect = mock_run
        
        # Test the complete workflow
        run_demo()
        
        # Verify the complete flow execution
        mock_flow.run.assert_called_once()
        
        # Get the shared store that was passed to the flow
        shared_store = mock_flow.run.call_args[0][0]
        
        # Verify input structure
        assert "task_requirements" in shared_store
        assert "platforms" in shared_store["task_requirements"]
        assert "topic_or_goal" in shared_store["task_requirements"]
        
        # Verify output structure (after flow execution)
        assert "content_pieces" in shared_store
        assert "twitter" in shared_store["content_pieces"]
        assert "linkedin" in shared_store["content_pieces"]
        assert "workflow_state" in shared_store
        assert shared_store["workflow_state"] == "completed"
    
    def test_workflow_with_configuration_integration(self):
        """Test workflow with configuration integration."""
        # Set up custom configuration
        config = AppConfig(
            llm_provider="test_provider",
            llm_model="test_model",
            debug_mode=True,
            supported_platforms=["twitter", "linkedin", "facebook"]
        )
        
        # Test that configuration affects validation
        topic = "Test topic"
        platforms = "facebook"  # Should be supported with custom config
        
        validation_result, sanitized_shared = validate_and_sanitize_inputs(
            topic, platforms, supported_platforms=config.supported_platforms
        )
        
        # Verify validation passes with custom supported platforms
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        assert "facebook" in sanitized_shared["task_requirements"]["platforms"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])