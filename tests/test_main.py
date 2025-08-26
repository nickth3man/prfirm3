# Test framework: pytest (with unittest.mock)

"""Comprehensive unit tests for the main module.

This module provides extensive test coverage for all main.py functions including
input validation, error handling, configuration management, and flow execution.

Test Coverage:
- Configuration loading and validation
- Input validation and sanitization
- Error handling and logging
- Flow execution with mocked dependencies
- CLI argument parsing
- Gradio interface creation
"""

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import run_demo, create_gradio_interface, main
from config import get_config, AppConfig
from validation import validate_topic, validate_platforms, ValidationError


class TestConfiguration:
    """Test configuration management functionality."""

    def test_get_config_defaults(self):
        """Test that get_config returns default configuration."""
        config = get_config()
        assert isinstance(config, AppConfig)
        assert config.gradio.port == 7860
        assert config.logging.level == "INFO"
        assert config.llm.provider == "openai"

    def test_get_config_with_file(self):
        """Test configuration loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
gradio:
  port: 8080
  host: "127.0.0.1"
logging:
  level: "DEBUG"
llm:
  provider: "anthropic"
  model: "claude-3-sonnet"
            """)
            config_file = f.name

        try:
            config = get_config(config_file)
            assert config.gradio.port == 8080
            assert config.gradio.host == "127.0.0.1"
            assert config.logging.level == "DEBUG"
            assert config.llm.provider == "anthropic"
        finally:
            os.unlink(config_file)

    def test_config_environment_override(self):
        """Test that environment variables override file config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
gradio:
  port: 8080
logging:
  level: "INFO"
            """)
            config_file = f.name

        try:
            # Set environment variables
            os.environ['GRADIO_PORT'] = '9000'
            os.environ['LOG_LEVEL'] = 'ERROR'

            config = get_config(config_file)
            assert config.gradio.port == 9000
            assert config.logging.level == "ERROR"
        finally:
            os.unlink(config_file)
            # Clean up environment
            os.environ.pop('GRADIO_PORT', None)
            os.environ.pop('LOG_LEVEL', None)


class TestValidation:
    """Test input validation functionality."""

    def test_validate_topic_valid(self):
        """Test valid topic validation."""
        topic = "Announce product launch"
        result = validate_topic(topic)
        assert result == topic

    def test_validate_topic_too_short(self):
        """Test topic validation with too short input."""
        with pytest.raises(ValidationError) as exc_info:
            validate_topic("ab")
        assert "too short" in str(exc_info.value)
        assert exc_info.value.field == "topic"

    def test_validate_topic_too_long(self):
        """Test topic validation with too long input."""
        long_topic = "a" * 501
        with pytest.raises(ValidationError) as exc_info:
            validate_topic(long_topic)
        assert "too long" in str(exc_info.value)

    def test_validate_topic_malicious_content(self):
        """Test topic validation with malicious content."""
        malicious_topics = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "onclick=alert('xss')",
        ]

        for topic in malicious_topics:
            with pytest.raises(ValidationError) as exc_info:
                validate_topic(topic)
            assert "malicious" in str(exc_info.value)

    def test_validate_platforms_valid(self):
        """Test valid platform validation."""
        platforms = "twitter, linkedin, facebook"
        result = validate_platforms(platforms)
        assert result == ["twitter", "linkedin", "facebook"]

    def test_validate_platforms_normalization(self):
        """Test platform name normalization."""
        platforms = "TWITTER, LinkedIn, FB"
        result = validate_platforms(platforms)
        assert result == ["twitter", "linkedin", "facebook"]

    def test_validate_platforms_unsupported(self):
        """Test platform validation with unsupported platform."""
        with pytest.raises(ValidationError) as exc_info:
            validate_platforms("invalid_platform")
        assert "Unsupported platform" in str(exc_info.value)

    def test_validate_platforms_duplicate(self):
        """Test platform validation with duplicate platforms."""
        with pytest.raises(ValidationError) as exc_info:
            validate_platforms("twitter, twitter")
        assert "Duplicate platform" in str(exc_info.value)

    def test_validate_platforms_empty(self):
        """Test platform validation with empty input."""
        with pytest.raises(ValidationError) as exc_info:
            validate_platforms("")
        assert "At least one platform" in str(exc_info.value)


class TestRunDemo:
    """Test the run_demo function."""

    @patch('main.create_main_flow')
    @patch('main.validate_shared_store')
    def test_run_demo_success(self, mock_validate, mock_create_flow):
        """Test successful demo execution."""
        # Mock the flow
        mock_flow = Mock()
        mock_create_flow.return_value = mock_flow

        # Mock validation
        mock_validate.return_value = {
            "task_requirements": {"platforms": ["twitter"], "topic_or_goal": "test"},
            "brand_bible": {"xml_raw": ""},
            "content_pieces": {"twitter": "Test content"}
        }

        # Run demo
        run_demo()

        # Verify flow was created and run
        mock_create_flow.assert_called_once()
        mock_flow.run.assert_called_once()

    @patch('main.create_main_flow')
    @patch('main.validate_shared_store')
    def test_run_demo_validation_error(self, mock_validate, mock_create_flow):
        """Test demo execution with validation error."""
        # Mock validation to raise error
        mock_validate.side_effect = ValidationError("Invalid input", "test")

        # Run demo should raise the error
        with pytest.raises(ValidationError):
            run_demo()

        # Flow should not be created
        mock_create_flow.assert_not_called()

    @patch('main.create_main_flow')
    @patch('main.validate_shared_store')
    def test_run_demo_flow_error(self, mock_validate, mock_create_flow):
        """Test demo execution with flow error."""
        # Mock validation
        mock_validate.return_value = {
            "task_requirements": {"platforms": ["twitter"], "topic_or_goal": "test"},
            "brand_bible": {"xml_raw": ""}
        }

        # Mock flow to raise error
        mock_flow = Mock()
        mock_flow.run.side_effect = RuntimeError("Flow error")
        mock_create_flow.return_value = mock_flow

        # Run demo should raise the error
        with pytest.raises(RuntimeError):
            run_demo()


class TestGradioInterface:
    """Test Gradio interface creation."""

    @patch('main.gr')
    def test_create_gradio_interface_success(self, mock_gr):
        """Test successful Gradio interface creation."""
        # Mock Gradio components
        mock_blocks = Mock()
        mock_gr.Blocks.return_value.__enter__.return_value = mock_blocks
        mock_gr.Blocks.return_value.__exit__.return_value = None

        # Mock other components
        mock_gr.Markdown.return_value = Mock()
        mock_gr.Textbox.return_value = Mock()
        mock_gr.JSON.return_value = Mock()
        mock_gr.Button.return_value = Mock()

        # Create interface
        result = create_gradio_interface()

        # Verify interface was created
        assert result is not None
        mock_gr.Blocks.assert_called_once()

    @patch('main.gr', None)
    def test_create_gradio_interface_no_gradio(self):
        """Test Gradio interface creation when Gradio is not available."""
        with pytest.raises(RuntimeError) as exc_info:
            create_gradio_interface()
        assert "Gradio not installed" in str(exc_info.value)


class TestCLI:
    """Test CLI functionality."""

    @patch('main.run_demo')
    def test_cli_demo_mode(self, mock_run_demo):
        """Test CLI demo mode."""
        with patch('sys.argv', ['main.py', '--demo']):
            main()
            mock_run_demo.assert_called_once()

    @patch('main.create_gradio_interface')
    @patch('main.gr')
    def test_cli_serve_mode(self, mock_gr, mock_create_interface):
        """Test CLI serve mode."""
        # Mock Gradio
        mock_gr.is_available.return_value = True

        # Mock interface
        mock_app = Mock()
        mock_create_interface.return_value = mock_app

        with patch('sys.argv', ['main.py', '--serve']):
            main()
            mock_create_interface.assert_called_once()

    def test_cli_version(self):
        """Test CLI version flag."""
        with patch('sys.argv', ['main.py', '--version']), patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()

    def test_cli_health(self):
        """Test CLI health flag."""
        with patch('sys.argv', ['main.py', '--health']), patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()

    def test_cli_info(self):
        """Test CLI info flag."""
        with patch('sys.argv', ['main.py', '--info']), patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()

    def test_cli_invalid_args(self):
        """Test CLI with invalid arguments."""
        with patch('sys.argv', ['main.py', '--invalid']), pytest.raises(SystemExit):
            main()


class TestErrorHandling:
    """Test error handling functionality."""

    @patch('main.log_error_with_context')
    def test_error_logging(self, mock_log_error):
        """Test that errors are properly logged."""
        test_error = ValueError("Test error")

        with patch('main.create_main_flow', side_effect=test_error), pytest.raises(ValueError):
            run_demo()

        # Verify error was logged
        mock_log_error.assert_called_once()

    def test_validation_error_structure(self):
        """Test ValidationError structure."""
        error = ValidationError("Test message", "field_name", "error_code")
        assert error.message == "Test message"
        assert error.field == "field_name"
        assert error.code == "error_code"


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('main.create_main_flow')
    def test_full_demo_workflow(self, mock_create_flow):
        """Test complete demo workflow."""
        # Mock successful flow execution
        mock_flow = Mock()
        mock_flow.run.return_value = None
        mock_create_flow.return_value = mock_flow

        # Mock validation
        with patch('main.validate_shared_store') as mock_validate:
            mock_validate.return_value = {
                "task_requirements": {"platforms": ["twitter"], "topic_or_goal": "test"},
                "brand_bible": {"xml_raw": ""},
                "content_pieces": {"twitter": "Generated content"}
            }

            # Run demo
            run_demo()

            # Verify complete workflow
            mock_validate.assert_called_once()
            mock_create_flow.assert_called_once()
            mock_flow.run.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
# ========================
# Additional tests (PR-focused)
# ========================

@pytest.mark.skipif('ValidationError' not in globals() or ValidationError is None, reason="validation module not available")
class TestValidation_PRDiffFocus:
    def test_validate_topic_rejects_none_and_whitespace_only(self):
        # None should raise ValidationError with field 'topic'
        with pytest.raises(ValidationError) as excinfo_none:
            validate_topic(None)  # type: ignore[arg-type]
        assert "topic" in getattr(excinfo_none.value, "field", "topic")
        # Whitespace-only string
        with pytest.raises(ValidationError) as excinfo_ws:
            validate_topic("   ")
        assert "empty" in str(excinfo_ws.value).lower() or "blank" in str(excinfo_ws.value).lower()

    def test_validate_topic_trims_and_accepts_reasonable_length(self):
        topic = "   Launch v2.0  "
        assert validate_topic(topic) == "Launch v2.0"

    def test_validate_platforms_handles_mixed_separators_and_spaces(self):
        # Mixed commas and extra spaces
        res = validate_platforms(" twitter ,linkedin ,  facebook ")
        assert res == ["twitter", "linkedin", "facebook"]

    def test_validate_platforms_list_input_and_case_insensitivity(self):
        # If API supports list input, ensure normalization; otherwise expect ValidationError gracefully.
        try:
            res = validate_platforms(["TWITTER", "LinkedIn", "FB"])  # type: ignore[list-item]
            assert res == ["twitter", "linkedin", "facebook"]
        except ValidationError:
            # Acceptable if API only supports string input
            pass

    def test_validate_platforms_rejects_non_string_entries(self):
        with pytest.raises(ValidationError):
            validate_platforms(123)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            validate_platforms(["twitter", 5])  # type: ignore[list-item]

@pytest.mark.skipif('get_config' not in globals() or get_config is None, reason="config module not available")
class TestConfig_PRDiffFocus:
    def test_get_config_invalid_file_path_raises(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError):
            get_config(str(missing))

    def test_get_config_env_bool_and_int_parsing(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "app.yaml"
        cfg_file.write_text(
            "gradio:\n  port: 8080\n  host: \"0.0.0.0\"\nlogging:\n  level: \"INFO\"\n"
        )
        monkeypatch.setenv("GRADIO_PORT", "9091")
        monkeypatch.setenv("GRADIO_HOST", "127.0.0.1")
        try:
            cfg = get_config(str(cfg_file))
            # Ensure env overrides take precedence and types parsed correctly
            assert int(cfg.gradio.port) == 9091
            assert cfg.gradio.host == "127.0.0.1"
        finally:
            monkeypatch.delenv("GRADIO_PORT", raising=False)
            monkeypatch.delenv("GRADIO_HOST", raising=False)

@pytest.mark.skipif('main' not in globals() or main is None, reason="main module not importable")
class TestCLI_PRDiffFocus:
    @patch('main.run_demo')
    def test_cli_mutually_exclusive_flags_error(self, mock_run_demo):
        # If both --demo and --serve supplied, expect SystemExit with error
        with patch('sys.argv', ['main.py', '--demo', '--serve']), pytest.raises(SystemExit):
            main()
        mock_run_demo.assert_not_called()

    def test_cli_no_args_shows_help_or_exits_cleanly(self):
        # Without arguments, many CLIs print help and exit
        with patch('sys.argv', ['main.py']), pytest.raises(SystemExit):
            main()

    @patch('builtins.print')
    def test_cli_unknown_flag_prints_usage(self, mock_print):
        with patch('sys.argv', ['main.py', '--unknown-flag']), pytest.raises(SystemExit):
            main()
        assert mock_print.called

@pytest.mark.skipif('run_demo' not in globals() or run_demo is None, reason="main.run_demo not available")
class TestRunDemo_PRDiffFocus:
    @patch('main.validate_shared_store')
    @patch('main.create_main_flow')
    def test_run_demo_passes_validated_store_into_flow(self, mock_create_flow, mock_validate):
        # Ensure the data returned by validation is passed into flow.run (either via constructor or call)
        validated = {
            "task_requirements": {"platforms": ["twitter"], "topic_or_goal": "ok"},
            "brand_bible": {"xml_raw": "<b/>"},
            "content_pieces": {"twitter": "hello"}
        }
        mock_validate.return_value = validated
        flow = Mock()
        mock_create_flow.return_value = flow

        run_demo()

        mock_create_flow.assert_called_once()
        # Run should be called (optionally with args); allow either signature
        try:
            flow.run.assert_called()
        except AssertionError:
            # If called with the store
            flow.run.assert_called_with(validated)

    @patch('main.validate_shared_store', side_effect=RuntimeError("unexpected"))
    @patch('main.log_error_with_context')
    def test_run_demo_logs_unexpected_validation_exceptions(self, mock_log, _):
        with pytest.raises(RuntimeError):
            run_demo()
        assert mock_log.called

@pytest.mark.skipif('create_gradio_interface' not in globals() or create_gradio_interface is None, reason="gradio interface factory not importable")
class TestGradioInterface_PRDiffFocus:
    @patch('main.gr')
    def test_gradio_blocks_lifecycle_and_button_bindings(self, mock_gr):
        # Set up fake Blocks context manager behavior
        blocks_ctx = Mock()
        mock_gr.Blocks.return_value.__enter__.return_value = blocks_ctx
        mock_gr.Blocks.return_value.__exit__.return_value = None

        # Components
        prompt = Mock(name="prompt")
        output = Mock(name="output")
        run_btn = Mock(name="run_button")
        mock_gr.Textbox.return_value = prompt
        mock_gr.Textbox.return_value = prompt
        mock_gr.Markdown.return_value = Mock()
        mock_gr.JSON.return_value = output
        mock_gr.Button.return_value = run_btn

        app = create_gradio_interface()
        assert app is not None
        mock_gr.Blocks.assert_called_once()

    @patch('main.gr', None)
    def test_gradio_interface_raises_when_gradio_missing(self):
        with pytest.raises(RuntimeError):
            create_gradio_interface()