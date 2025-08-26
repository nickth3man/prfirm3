"""Comprehensive test suite for the main module.

This module provides extensive testing for the Virtual PR Firm's main functionality,
including CLI operations, Gradio interface, configuration management, and error handling.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from main import (
    setup_logging,
    handle_errors,
    validate_shared_store,
    sanitize_input,
    ConfigurationManager,
    run_demo,
    create_gradio_interface,
    parse_arguments,
    run_cli_demo,
    run_web_interface,
    main
)


class TestLogging:
    """Test logging configuration and setup."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            setup_logging()
            mock_get_logger.assert_called()
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.close()
            try:
                setup_logging(level="DEBUG", log_file=temp_file.name)
                # Verify file was created
                assert os.path.exists(temp_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level."""
        setup_logging(level="INVALID_LEVEL")  # Should not raise error


class TestErrorHandling:
    """Test error handling decorators and utilities."""
    
    def test_handle_errors_success(self):
        """Test error handling decorator with successful function."""
        @handle_errors
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_handle_errors_exception(self):
        """Test error handling decorator with exception."""
        @handle_errors
        def test_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            test_func()


class TestValidation:
    """Test input validation functions."""
    
    def test_validate_shared_store_valid(self):
        """Test shared store validation with valid input."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "test topic"
            }
        }
        validate_shared_store(shared)  # Should not raise
    
    def test_validate_shared_store_invalid_type(self):
        """Test shared store validation with invalid type."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_store("not a dict")
    
    def test_validate_shared_store_missing_key(self):
        """Test shared store validation with missing required key."""
        shared = {"other_key": "value"}
        with pytest.raises(ValueError, match="Missing required key"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_invalid_task_requirements(self):
        """Test shared store validation with invalid task_requirements."""
        shared = {"task_requirements": "not a dict"}
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_store(shared)
    
    def test_sanitize_input_valid(self):
        """Test input sanitization with valid input."""
        result = sanitize_input("test input")
        assert result == "test input"
    
    def test_sanitize_input_too_long(self):
        """Test input sanitization with input too long."""
        long_input = "x" * 10001
        with pytest.raises(ValueError, match="too long"):
            sanitize_input(long_input, max_length=10000)
    
    def test_sanitize_input_dangerous_chars(self):
        """Test input sanitization with dangerous characters."""
        dangerous_input = "test<script>alert('xss')</script>"
        result = sanitize_input(dangerous_input)
        assert "<script>" not in result
        assert "alert('xss')" not in result
    
    def test_sanitize_input_not_string(self):
        """Test input sanitization with non-string input."""
        with pytest.raises(ValueError, match="must be a string"):
            sanitize_input(123)


class TestConfigurationManager:
    """Test configuration management."""
    
    def test_configuration_manager_default(self):
        """Test configuration manager with default configuration."""
        config = ConfigurationManager()
        assert config.get("logging.level") == "INFO"
        assert config.get("demo.default_platforms") == ["twitter", "linkedin"]
    
    def test_configuration_manager_with_file(self):
        """Test configuration manager with configuration file."""
        config_data = {
            "logging": {"level": "DEBUG"},
            "demo": {"default_topic": "Custom topic"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(config_data, temp_file)
            temp_file.close()
            
            try:
                config = ConfigurationManager(temp_file.name)
                assert config.get("logging.level") == "DEBUG"
                assert config.get("demo.default_topic") == "Custom topic"
            finally:
                os.unlink(temp_file.name)
    
    def test_configuration_manager_invalid_file(self):
        """Test configuration manager with invalid file."""
        config = ConfigurationManager("nonexistent_file.json")
        # Should not raise error, should use defaults
        assert config.get("logging.level") == "INFO"
    
    def test_configuration_manager_get_nested(self):
        """Test configuration manager get method with nested keys."""
        config = ConfigurationManager()
        assert config.get("logging.level") == "INFO"
        assert config.get("nonexistent.key", "default") == "default"


class TestRunDemo:
    """Test demo execution."""
    
    @patch('main.create_main_flow')
    @patch('main.logger')
    def test_run_demo_success(self, mock_logger, mock_create_flow):
        """Test successful demo execution."""
        mock_flow = MagicMock()
        mock_create_flow.return_value = mock_flow
        
        run_demo()
        
        mock_create_flow.assert_called_once()
        mock_flow.run.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.create_main_flow')
    @patch('main.logger')
    def test_run_demo_flow_creation_failure(self, mock_logger, mock_create_flow):
        """Test demo execution with flow creation failure."""
        mock_create_flow.side_effect = Exception("Flow creation failed")
        
        with pytest.raises(Exception, match="Flow creation failed"):
            run_demo()
        
        mock_logger.error.assert_called()


class TestGradioInterface:
    """Test Gradio interface creation."""
    
    @patch('main.gr')
    def test_create_gradio_interface_success(self, mock_gr):
        """Test successful Gradio interface creation."""
        mock_blocks = MagicMock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks
        
        result = create_gradio_interface()
        
        assert result == mock_blocks
        mock_gr.Blocks.assert_called_once()
    
    @patch('main.gr', None)
    def test_create_gradio_interface_no_gradio(self):
        """Test Gradio interface creation when Gradio is not available."""
        with pytest.raises(RuntimeError, match="Gradio is not installed"):
            create_gradio_interface()


class TestCLIArgumentParsing:
    """Test command line argument parsing."""
    
    def test_parse_arguments_demo(self):
        """Test argument parsing for demo mode."""
        with patch('sys.argv', ['main.py', '--demo']):
            args = parse_arguments()
            assert args.demo is True
            assert args.web is False
            assert args.api is False
    
    def test_parse_arguments_web(self):
        """Test argument parsing for web mode."""
        with patch('sys.argv', ['main.py', '--web']):
            args = parse_arguments()
            assert args.demo is False
            assert args.web is True
            assert args.api is False
    
    def test_parse_arguments_with_config(self):
        """Test argument parsing with configuration file."""
        with patch('sys.argv', ['main.py', '--demo', '--config', 'test.json']):
            args = parse_arguments()
            assert args.config == 'test.json'
    
    def test_parse_arguments_with_log_level(self):
        """Test argument parsing with log level."""
        with patch('sys.argv', ['main.py', '--demo', '--log-level', 'DEBUG']):
            args = parse_arguments()
            assert args.log_level == 'DEBUG'
    
    def test_parse_arguments_no_mode(self):
        """Test argument parsing without mode specification."""
        with patch('sys.argv', ['main.py']):
            with pytest.raises(SystemExit):
                parse_arguments()
    
    def test_parse_arguments_multiple_modes(self):
        """Test argument parsing with multiple modes (should fail)."""
        with patch('sys.argv', ['main.py', '--demo', '--web']):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestCLIExecution:
    """Test CLI execution functions."""
    
    @patch('main.run_demo')
    @patch('main.logger')
    def test_run_cli_demo_success(self, mock_logger, mock_run_demo):
        """Test successful CLI demo execution."""
        args = MagicMock()
        args.topic = None
        args.platforms = None
        
        run_cli_demo(args)
        
        mock_run_demo.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.run_demo')
    @patch('main.logger')
    def test_run_cli_demo_with_custom_params(self, mock_logger, mock_run_demo):
        """Test CLI demo execution with custom parameters."""
        args = MagicMock()
        args.topic = "Custom topic"
        args.platforms = "twitter,instagram"
        
        run_cli_demo(args)
        
        mock_run_demo.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('main.create_gradio_interface')
    @patch('main.logger')
    def test_run_web_interface_success(self, mock_logger, mock_create_interface):
        """Test successful web interface execution."""
        args = MagicMock()
        args.host = "127.0.0.1"
        args.port = 7860
        args.share = False
        
        mock_app = MagicMock()
        mock_create_interface.return_value = mock_app
        
        run_web_interface(args)
        
        mock_create_interface.assert_called_once()
        mock_app.launch.assert_called_once()
        mock_logger.info.assert_called()


class TestMainFunction:
    """Test main function execution."""
    
    @patch('main.parse_arguments')
    @patch('main.setup_logging')
    @patch('main.run_cli_demo')
    @patch('main.logger')
    def test_main_demo_mode(self, mock_logger, mock_run_demo, mock_setup_logging, mock_parse_args):
        """Test main function in demo mode."""
        args = MagicMock()
        args.demo = True
        args.web = False
        args.api = False
        args.config = None
        args.log_level = "INFO"
        args.log_file = None
        mock_parse_args.return_value = args
        
        with patch('sys.exit') as mock_exit:
            main()
            
            mock_parse_args.assert_called_once()
            mock_setup_logging.assert_called_once()
            mock_run_demo.assert_called_once()
            mock_exit.assert_not_called()
    
    @patch('main.parse_arguments')
    @patch('main.setup_logging')
    @patch('main.run_web_interface')
    @patch('main.logger')
    def test_main_web_mode(self, mock_logger, mock_run_web, mock_setup_logging, mock_parse_args):
        """Test main function in web mode."""
        args = MagicMock()
        args.demo = False
        args.web = True
        args.api = False
        args.config = None
        args.log_level = "INFO"
        args.log_file = None
        mock_parse_args.return_value = args
        
        with patch('sys.exit') as mock_exit:
            main()
            
            mock_parse_args.assert_called_once()
            mock_setup_logging.assert_called_once()
            mock_run_web.assert_called_once()
            mock_exit.assert_not_called()
    
    @patch('main.parse_arguments')
    @patch('main.logger')
    def test_main_api_mode_not_implemented(self, mock_logger, mock_parse_args):
        """Test main function with API mode (not implemented)."""
        args = MagicMock()
        args.demo = False
        args.web = False
        args.api = True
        mock_parse_args.return_value = args
        
        with patch('sys.exit') as mock_exit:
            main()
            
            mock_logger.error.assert_called_with("API mode not yet implemented")
            mock_exit.assert_called_with(1)
    
    @patch('main.parse_arguments')
    @patch('main.logger')
    def test_main_exception_handling(self, mock_logger, mock_parse_args):
        """Test main function exception handling."""
        mock_parse_args.side_effect = Exception("Test exception")
        
        with patch('sys.exit') as mock_exit:
            main()
            
            mock_logger.error.assert_called_with("Application error: Test exception")
            mock_exit.assert_called_with(1)


class TestIntegration:
    """Integration tests for the main module."""
    
    def test_full_pipeline_integration(self):
        """Test full pipeline integration with sample data."""
        # This test would require actual flow execution
        # For now, we'll test the configuration and validation parts
        config = ConfigurationManager()
        
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test integration"
            }
        }
        
        validate_shared_store(shared)
        assert config.get("demo.default_platforms") == ["twitter", "linkedin"]
    
    def test_error_recovery_integration(self):
        """Test error recovery in integration scenarios."""
        # Test that the system can handle various error conditions gracefully
        config = ConfigurationManager("nonexistent_file.json")
        
        # Should not crash, should use defaults
        assert config.get("logging.level") == "INFO"
        
        # Test with invalid shared store
        with pytest.raises(ValueError):
            validate_shared_store("invalid")


if __name__ == "__main__":
    pytest.main([__file__])