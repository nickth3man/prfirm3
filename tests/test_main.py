"""Unit tests for the main.py module.

This module provides comprehensive testing for the Virtual PR Firm's main interface,
including CLI functionality, Gradio interface, input validation, and error handling.
"""

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from main import (
    Config, 
    ValidationError, 
    SecurityError, 
    InputValidator,
    run_demo, 
    validate_shared_store, 
    create_gradio_interface,
    main
)


class TestConfig:
    """Test the Config class for configuration management."""
    
    def test_config_defaults(self):
        """Test that Config has correct default values."""
        config = Config()
        
        assert config.DEFAULT_TOPIC == "Announce product"
        assert config.DEFAULT_PLATFORMS == ["twitter", "linkedin"]
        assert config.MAX_TOPIC_LENGTH == 500
        assert config.MAX_PLATFORMS == 10
        assert "twitter" in config.SUPPORTED_PLATFORMS
        assert "linkedin" in config.SUPPORTED_PLATFORMS
        assert config.REQUEST_TIMEOUT == 300
        assert config.MAX_RETRIES == 3
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'PR_FIRM_DEFAULT_TOPIC': 'Test Topic',
            'PR_FIRM_DEFAULT_PLATFORMS': 'facebook,instagram',
            'PR_FIRM_MAX_TOPIC_LENGTH': '1000',
            'PR_FIRM_REQUEST_TIMEOUT': '600'
        }):
            config = Config.from_env()
            
            assert config.DEFAULT_TOPIC == 'Test Topic'
            assert config.DEFAULT_PLATFORMS == ['facebook', 'instagram']
            assert config.MAX_TOPIC_LENGTH == 1000
            assert config.REQUEST_TIMEOUT == 600
    
    def test_config_from_env_invalid_values(self):
        """Test handling of invalid environment variable values."""
        with patch.dict(os.environ, {
            'PR_FIRM_MAX_TOPIC_LENGTH': 'invalid',
            'PR_FIRM_REQUEST_TIMEOUT': 'not_a_number'
        }):
            # Should not raise an exception, should use defaults
            config = Config.from_env()
            assert config.MAX_TOPIC_LENGTH == 500
            assert config.REQUEST_TIMEOUT == 300


class TestInputValidator:
    """Test the InputValidator class for input validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
    
    def test_validate_platforms_valid(self):
        """Test validation of valid platform lists."""
        platforms = ["twitter", "linkedin", "facebook"]
        result = self.validator.validate_platforms(platforms)
        
        assert result == ["twitter", "linkedin", "facebook"]
    
    def test_validate_platforms_normalize(self):
        """Test platform name normalization."""
        platforms = ["Twitter", " LINKEDIN ", "Facebook"]
        result = self.validator.validate_platforms(platforms)
        
        assert result == ["twitter", "linkedin", "facebook"]
    
    def test_validate_platforms_duplicates(self):
        """Test removal of duplicate platforms."""
        platforms = ["twitter", "linkedin", "twitter", "facebook"]
        result = self.validator.validate_platforms(platforms)
        
        assert result == ["twitter", "linkedin", "facebook"]
    
    def test_validate_platforms_empty(self):
        """Test validation of empty platform list."""
        with pytest.raises(ValidationError, match="At least 1 platform must be specified"):
            self.validator.validate_platforms([])
    
    def test_validate_platforms_too_many(self):
        """Test validation when too many platforms are specified."""
        platforms = ["platform" + str(i) for i in range(15)]
        with pytest.raises(ValidationError, match="Maximum 10 platforms allowed"):
            self.validator.validate_platforms(platforms)
    
    def test_validate_platforms_unsupported(self):
        """Test validation of unsupported platforms."""
        platforms = ["twitter", "unsupported_platform"]
        with pytest.raises(ValidationError, match="Unsupported platform: unsupported_platform"):
            self.validator.validate_platforms(platforms)
    
    def test_validate_platforms_invalid_type(self):
        """Test validation with invalid platform list type."""
        with pytest.raises(ValidationError, match="Platforms must be a list"):
            self.validator.validate_platforms("not_a_list")
    
    def test_validate_topic_valid(self):
        """Test validation of valid topics."""
        topic = "This is a valid topic"
        result = self.validator.validate_topic(topic)
        
        assert result == "This is a valid topic"
    
    def test_validate_topic_too_short(self):
        """Test validation of topics that are too short."""
        with pytest.raises(ValidationError, match="Topic must be at least 3 characters"):
            self.validator.validate_topic("ab")
    
    def test_validate_topic_too_long(self):
        """Test validation of topics that are too long."""
        long_topic = "a" * 501
        with pytest.raises(ValidationError, match="Topic must be no more than 500 characters"):
            self.validator.validate_topic(long_topic)
    
    def test_validate_topic_security_script(self):
        """Test security validation against script injection."""
        malicious_topic = "Normal topic <script>alert('xss')</script>"
        with pytest.raises(SecurityError, match="Text contains forbidden pattern"):
            self.validator.validate_topic(malicious_topic)
    
    def test_validate_topic_security_javascript(self):
        """Test security validation against JavaScript protocol."""
        malicious_topic = "Normal topic javascript:alert('xss')"
        with pytest.raises(SecurityError, match="Text contains forbidden pattern"):
            self.validator.validate_topic(malicious_topic)
    
    def test_validate_intents_valid(self):
        """Test validation of valid intents."""
        intents = {
            "twitter": {"value": "engagement"},
            "linkedin": {"value": "thought_leadership"}
        }
        result = self.validator.validate_intents(intents)
        
        assert result == {
            "twitter": {"value": "engagement"},
            "linkedin": {"value": "thought_leadership"}
        }
    
    def test_validate_intents_invalid_type(self):
        """Test validation with invalid intents type."""
        with pytest.raises(ValidationError, match="Intents must be a dictionary"):
            self.validator.validate_intents("not_a_dict")
    
    def test_validate_intents_invalid_platform_data(self):
        """Test validation with invalid platform data."""
        intents = {"twitter": "not_a_dict"}
        with pytest.raises(ValidationError, match="Intent data for twitter must be a dictionary"):
            self.validator.validate_intents(intents)
    
    def test_validate_intents_invalid_value_type(self):
        """Test validation with invalid intent value type."""
        intents = {"twitter": {"value": 123}}
        with pytest.raises(ValidationError, match="Intent value for twitter must be a string"):
            self.validator.validate_intents(intents)
    
    def test_validate_intents_too_long(self):
        """Test validation of intents that are too long."""
        long_intent = "a" * 101
        intents = {"twitter": {"value": long_intent}}
        with pytest.raises(ValidationError, match="Intent value for twitter too long"):
            self.validator.validate_intents(intents)


class TestValidateSharedStore:
    """Test the validate_shared_store function."""
    
    def test_validate_shared_store_valid(self):
        """Test validation of valid shared store."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        # Should not raise an exception
        validate_shared_store(shared)
    
    def test_validate_shared_store_not_dict(self):
        """Test validation when shared is not a dictionary."""
        with pytest.raises(TypeError, match="shared must be a dict"):
            validate_shared_store("not_a_dict")
    
    def test_validate_shared_store_missing_task_requirements(self):
        """Test validation when task_requirements is missing."""
        shared = {"brand_bible": {"xml_raw": ""}}
        with pytest.raises(ValueError, match="shared\\['task_requirements'\\] must be a dict"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_task_requirements_not_dict(self):
        """Test validation when task_requirements is not a dictionary."""
        shared = {
            "task_requirements": "not_a_dict",
            "brand_bible": {"xml_raw": ""}
        }
        with pytest.raises(ValueError, match="shared\\['task_requirements'\\] must be a dict"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_missing_platforms(self):
        """Test validation when platforms is missing."""
        shared = {
            "task_requirements": {
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        with pytest.raises(ValueError, match="task_requirements must include 'platforms'"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_platforms_not_list(self):
        """Test validation when platforms is not a list."""
        shared = {
            "task_requirements": {
                "platforms": "not_a_list",
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        with pytest.raises(TypeError, match="task_requirements\\['platforms'\\] must be a list"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_missing_topic(self):
        """Test validation when topic_or_goal is missing."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"]
            },
            "brand_bible": {"xml_raw": ""}
        }
        with pytest.raises(ValueError, match="task_requirements must include 'topic_or_goal'"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_topic_not_string(self):
        """Test validation when topic_or_goal is not a string."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": 123
            },
            "brand_bible": {"xml_raw": ""}
        }
        with pytest.raises(TypeError, match="task_requirements\\['topic_or_goal'\\] must be a string"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_missing_brand_bible(self):
        """Test validation when brand_bible is missing."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            }
        }
        with pytest.raises(ValueError, match="shared\\['brand_bible'\\] must be a dict"):
            validate_shared_store(shared)
    
    def test_validate_shared_store_brand_bible_not_dict(self):
        """Test validation when brand_bible is not a dictionary."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": "not_a_dict"
        }
        with pytest.raises(ValueError, match="shared\\['brand_bible'\\] must be a dict"):
            validate_shared_store(shared)


class TestRunDemo:
    """Test the run_demo function."""
    
    @patch('main.create_main_flow')
    def test_run_demo_success(self, mock_create_flow):
        """Test successful execution of run_demo."""
        # Mock the flow
        mock_flow = Mock()
        mock_create_flow.return_value = mock_flow
        
        # Mock the shared state
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None
        }
        
        # Mock the flow.run method to update shared state
        def mock_run(shared_state):
            shared_state["content_pieces"] = {
                "twitter": "Generated content for Twitter",
                "linkedin": "Generated content for LinkedIn"
            }
        
        mock_flow.run.side_effect = mock_run
        
        # Test that run_demo doesn't raise an exception
        run_demo()
        
        # Verify the flow was created and run
        mock_create_flow.assert_called_once()
        mock_flow.run.assert_called_once()
    
    @patch('main.create_main_flow')
    def test_run_demo_import_error(self, mock_create_flow):
        """Test run_demo when flow module import fails."""
        mock_create_flow.side_effect = ImportError("Module not found")
        
        with pytest.raises(ImportError, match="Module not found"):
            run_demo()
    
    @patch('main.create_main_flow')
    def test_run_demo_flow_execution_error(self, mock_create_flow):
        """Test run_demo when flow execution fails."""
        mock_flow = Mock()
        mock_create_flow.return_value = mock_flow
        mock_flow.run.side_effect = Exception("Flow execution failed")
        
        with pytest.raises(Exception, match="Flow execution failed"):
            run_demo()


class TestCreateGradioInterface:
    """Test the create_gradio_interface function."""
    
    @patch('main.gr')
    def test_create_gradio_interface_success(self, mock_gr):
        """Test successful creation of Gradio interface."""
        # Mock Gradio components
        mock_blocks = Mock()
        mock_markdown = Mock()
        mock_textbox = Mock()
        mock_button = Mock()
        mock_json = Mock()
        mock_row = Mock()
        mock_column = Mock()
        
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks
        mock_gr.Markdown.return_value = mock_markdown
        mock_gr.Textbox.return_value = mock_textbox
        mock_gr.Button.return_value = mock_button
        mock_gr.JSON.return_value = mock_json
        mock_gr.Row.return_value.__enter__.return_value = mock_row
        mock_gr.Column.return_value.__enter__.return_value = mock_column
        
        # Test interface creation
        result = create_gradio_interface()
        
        assert result == mock_blocks
        mock_gr.Blocks.assert_called_once()
    
    def test_create_gradio_interface_not_installed(self):
        """Test create_gradio_interface when Gradio is not installed."""
        with patch('main.gr', None):
            with pytest.raises(RuntimeError, match="Gradio not installed"):
                create_gradio_interface()


class TestMain:
    """Test the main function."""
    
    @patch('main.run_demo')
    @patch('main.create_gradio_interface')
    @patch('main.sys.exit')
    def test_main_cli_mode(self, mock_exit, mock_create_interface, mock_run_demo):
        """Test main function in CLI mode."""
        with patch('sys.argv', ['main.py', '--cli']):
            main()
            
            mock_run_demo.assert_called_once()
            mock_exit.assert_not_called()
    
    @patch('main.create_gradio_interface')
    @patch('main.sys.exit')
    def test_main_web_mode(self, mock_exit, mock_create_interface):
        """Test main function in web mode."""
        mock_app = Mock()
        mock_create_interface.return_value = mock_app
        
        with patch('sys.argv', ['main.py', '--web']):
            with patch('main.gr', Mock()):
                main()
                
                mock_create_interface.assert_called_once()
                mock_app.launch.assert_called_once()
                mock_exit.assert_not_called()
    
    @patch('main.sys.exit')
    def test_main_web_mode_gradio_not_installed(self, mock_exit):
        """Test main function in web mode when Gradio is not installed."""
        with patch('sys.argv', ['main.py', '--web']):
            with patch('main.gr', None):
                main()
                
                mock_exit.assert_called_once_with(1)
    
    @patch('main.run_demo')
    @patch('main.sys.exit')
    def test_main_keyboard_interrupt(self, mock_exit, mock_run_demo):
        """Test main function handling keyboard interrupt."""
        mock_run_demo.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['main.py']):
            main()
            
            mock_exit.assert_called_once_with(0)
    
    @patch('main.run_demo')
    @patch('main.sys.exit')
    def test_main_general_exception(self, mock_exit, mock_run_demo):
        """Test main function handling general exceptions."""
        mock_run_demo.side_effect = Exception("Test error")
        
        with patch('sys.argv', ['main.py']):
            main()
            
            mock_exit.assert_called_once_with(1)


class TestIntegration:
    """Integration tests for the main module."""
    
    def test_config_and_validator_integration(self):
        """Test integration between Config and InputValidator."""
        config = Config()
        config.MAX_TOPIC_LENGTH = 100
        config.SUPPORTED_PLATFORMS = {"twitter", "linkedin"}
        
        validator = InputValidator(config)
        
        # Test that validator uses config values
        with pytest.raises(ValidationError, match="Topic must be no more than 100 characters"):
            validator.validate_topic("a" * 101)
        
        with pytest.raises(ValidationError, match="Unsupported platform: facebook"):
            validator.validate_platforms(["twitter", "facebook"])
    
    def test_validation_error_hierarchy(self):
        """Test that validation errors are properly categorized."""
        validator = InputValidator()
        
        # Security errors should be SecurityError, not ValidationError
        with pytest.raises(SecurityError):
            validator.validate_topic("Normal topic <script>alert('xss')</script>")
        
        # Regular validation errors should be ValidationError
        with pytest.raises(ValidationError):
            validator.validate_topic("ab")
    
    def test_shared_store_validation_integration(self):
        """Test integration between InputValidator and validate_shared_store."""
        validator = InputValidator()
        
        # Create valid shared store
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        # Validate shared store
        validate_shared_store(shared)
        
        # Validate individual components
        platforms = validator.validate_platforms(shared["task_requirements"]["platforms"])
        topic = validator.validate_topic(shared["task_requirements"]["topic_or_goal"])
        
        assert platforms == ["twitter", "linkedin"]
        assert topic == "Test topic"


if __name__ == "__main__":
    pytest.main([__file__])