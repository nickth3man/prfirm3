"""Unit tests for the nodes.py module.

This module provides comprehensive testing for the Virtual PR Firm's node implementations,
including input validation, error handling, and node execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from nodes import (
    ValidationConfig,
    ValidationError,
    SecurityError,
    InputValidator,
    EngagementManagerNode
)


class TestValidationConfig:
    """Test the ValidationConfig class."""
    
    def test_validation_config_defaults(self):
        """Test that ValidationConfig has correct default values."""
        config = ValidationConfig()
        
        assert config.MAX_PLATFORMS == 10
        assert config.MIN_PLATFORMS == 1
        assert config.MAX_TOPIC_LENGTH == 500
        assert config.MIN_TOPIC_LENGTH == 3
        assert config.MAX_INTENT_LENGTH == 100
        assert "twitter" in config.SUPPORTED_PLATFORMS
        assert "linkedin" in config.SUPPORTED_PLATFORMS
        assert len(config.FORBIDDEN_PATTERNS) > 0
    
    def test_validation_config_custom_values(self):
        """Test ValidationConfig with custom values."""
        config = ValidationConfig()
        config.MAX_PLATFORMS = 5
        config.SUPPORTED_PLATFORMS = {"custom_platform"}
        config.FORBIDDEN_PATTERNS = ["custom_pattern"]
        
        assert config.MAX_PLATFORMS == 5
        assert config.SUPPORTED_PLATFORMS == {"custom_platform"}
        assert config.FORBIDDEN_PATTERNS == ["custom_pattern"]


class TestInputValidator:
    """Test the InputValidator class."""
    
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
    
    def test_validate_topic_security_data_url(self):
        """Test security validation against data URLs."""
        malicious_topic = "Normal topic data:text/html,<script>alert('xss')</script>"
        with pytest.raises(SecurityError, match="Text contains forbidden pattern"):
            self.validator.validate_topic(malicious_topic)
    
    def test_validate_topic_security_event_handler(self):
        """Test security validation against event handlers."""
        malicious_topic = "Normal topic <img src=x onerror=alert('xss')>"
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
    
    def test_validate_security_valid_text(self):
        """Test security validation with valid text."""
        valid_text = "This is normal text with allowed characters: 123, !@#$%"
        # Should not raise an exception
        self.validator._validate_security(valid_text)
    
    def test_validate_security_suspicious_characters(self):
        """Test security validation with suspicious characters."""
        suspicious_text = "Normal text with suspicious chars: \x00\x01\x02"
        with pytest.raises(SecurityError, match="Text contains suspicious characters"):
            self.validator._validate_security(suspicious_text)
    
    def test_validate_security_invalid_html_entities(self):
        """Test security validation with invalid HTML entities."""
        invalid_entities = "Text with invalid entities: &invalid; &another;"
        with pytest.raises(SecurityError, match="Invalid HTML entities detected"):
            self.validator._validate_security(invalid_entities)
    
    def test_validate_security_valid_html_entities(self):
        """Test security validation with valid HTML entities."""
        valid_entities = "Text with valid entities: &amp; &lt; &gt; &quot;"
        # Should not raise an exception
        self.validator._validate_security(valid_entities)


class TestEngagementManagerNode:
    """Test the EngagementManagerNode class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = EngagementManagerNode()
    
    def test_node_initialization(self):
        """Test node initialization with default parameters."""
        node = EngagementManagerNode()
        assert node.max_retries == 2
        assert hasattr(node, 'validator')
        assert hasattr(node, 'validation_warnings')
        assert hasattr(node, 'validation_errors')
    
    def test_node_initialization_with_config(self):
        """Test node initialization with custom configuration."""
        config = ValidationConfig()
        config.MAX_TOPIC_LENGTH = 100
        
        node = EngagementManagerNode(validation_config=config)
        assert node.validator.config.MAX_TOPIC_LENGTH == 100
    
    def test_prep_valid_shared(self):
        """Test prep method with valid shared state."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "intents_by_platform": {
                    "twitter": {"value": "engagement"}
                },
                "topic_or_goal": "Test topic"
            }
        }
        
        result = self.node.prep(shared)
        
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Test topic"
        assert result["intents_by_platform"]["twitter"]["value"] == "engagement"
    
    def test_prep_missing_task_requirements(self):
        """Test prep method when task_requirements is missing."""
        shared = {}
        
        result = self.node.prep(shared)
        
        assert result["platforms"] == ["twitter", "linkedin"]  # Defaults
        assert result["topic_or_goal"] == "Announce product"  # Defaults
        assert result["intents_by_platform"] == {}
        
        # Check that warnings were added
        assert "No platforms specified, using defaults: twitter, linkedin" in self.node.validation_warnings
        assert "No topic specified, using default: Announce product" in self.node.validation_warnings
    
    def test_prep_invalid_task_requirements_type(self):
        """Test prep method when task_requirements is not a dict."""
        shared = {"task_requirements": "not_a_dict"}
        
        result = self.node.prep(shared)
        
        # Should use defaults
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Announce product"
    
    def test_prep_validation_errors(self):
        """Test prep method when validation fails."""
        shared = {
            "task_requirements": {
                "platforms": ["unsupported_platform"],
                "topic_or_goal": "ab"  # Too short
            }
        }
        
        result = self.node.prep(shared)
        
        # Should use defaults due to validation errors
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Announce product"
        
        # Check that errors were recorded
        assert len(self.node.validation_errors) > 0
    
    def test_prep_security_errors(self):
        """Test prep method when security validation fails."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": "Normal topic <script>alert('xss')</script>"
            }
        }
        
        result = self.node.prep(shared)
        
        # Should use defaults due to security errors
        assert result["topic_or_goal"] == "Announce product"
        
        # Check that errors were recorded
        assert len(self.node.validation_errors) > 0
    
    def test_exec_success(self):
        """Test exec method with valid input."""
        prep_res = {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        
        result = self.node.exec(prep_res)
        
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Test topic"
        assert "validation_metadata" in result
        assert result["validation_metadata"]["platforms_count"] == 2
        assert result["validation_metadata"]["has_intents"] is False
        assert result["validation_metadata"]["topic_length"] == 10
    
    def test_exec_with_warnings(self):
        """Test exec method when there are validation warnings."""
        self.node.validation_warnings = ["Warning 1", "Warning 2"]
        
        prep_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        
        result = self.node.exec(prep_res)
        
        assert result["validation_metadata"]["validation_warnings"] == ["Warning 1", "Warning 2"]
        assert result["validation_metadata"]["quality_score"] < 1.0  # Should be reduced due to warnings
    
    def test_exec_with_errors(self):
        """Test exec method when there are validation errors."""
        self.node.validation_errors = ["Error 1", "Error 2"]
        
        prep_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        
        result = self.node.exec(prep_res)
        
        assert result["validation_metadata"]["validation_errors"] == ["Error 1", "Error 2"]
        assert result["validation_metadata"]["quality_score"] < 0.5  # Should be significantly reduced due to errors
    
    def test_calculate_quality_score_perfect(self):
        """Test quality score calculation for perfect input."""
        requirements = {
            "topic_or_goal": "A perfect topic with good content",
            "intents_by_platform": {"twitter": {"value": "engagement"}},
            "platforms": ["twitter", "linkedin", "facebook"]
        }
        
        score = self.node._calculate_quality_score(requirements)
        assert score == 1.0
    
    def test_calculate_quality_score_defaults(self):
        """Test quality score calculation with default values."""
        requirements = {
            "topic_or_goal": "Announce product",  # Default
            "intents_by_platform": {},  # Empty
            "platforms": ["twitter"]  # Only one platform
        }
        
        score = self.node._calculate_quality_score(requirements)
        assert score < 1.0  # Should be reduced due to defaults
    
    def test_calculate_quality_score_with_issues(self):
        """Test quality score calculation with validation issues."""
        self.node.validation_errors = ["Error 1", "Error 2"]
        self.node.validation_warnings = ["Warning 1"]
        
        requirements = {
            "topic_or_goal": "Good topic",
            "intents_by_platform": {"twitter": {"value": "engagement"}},
            "platforms": ["twitter", "linkedin"]
        }
        
        score = self.node._calculate_quality_score(requirements)
        assert score < 0.5  # Should be significantly reduced due to errors
    
    def test_post_success(self):
        """Test post method with successful execution."""
        shared = {}
        prep_res = {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        exec_res = {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {},
            "validation_metadata": {
                "platforms_count": 2,
                "has_intents": False,
                "topic_length": 10,
                "validation_warnings": [],
                "validation_errors": [],
                "quality_score": 1.0
            }
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert shared["task_requirements"]["platforms"] == ["twitter", "linkedin"]
        assert shared["task_requirements"]["topic_or_goal"] == "Test topic"
        assert "validation_results" in shared
        assert shared["validation_results"]["platforms_count"] == 2
    
    def test_post_with_streaming(self):
        """Test post method with streaming enabled."""
        mock_stream = Mock()
        shared = {"stream": mock_stream}
        
        prep_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        exec_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {},
            "validation_metadata": {
                "platforms_count": 1,
                "has_intents": False,
                "topic_length": 10,
                "validation_warnings": [],
                "validation_errors": [],
                "quality_score": 1.0
            }
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        # Verify streaming events were emitted
        assert mock_stream.emit.call_count == 3  # engagement_manager_started, validation_completed, requirements_normalized
    
    def test_post_streaming_failure(self):
        """Test post method when streaming fails."""
        mock_stream = Mock()
        mock_stream.emit.side_effect = Exception("Streaming failed")
        shared = {"stream": mock_stream}
        
        prep_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        exec_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {},
            "validation_metadata": {
                "platforms_count": 1,
                "has_intents": False,
                "topic_length": 10,
                "validation_warnings": [],
                "validation_errors": [],
                "quality_score": 1.0
            }
        }
        
        # Should not raise an exception, should handle streaming failure gracefully
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert shared["task_requirements"]["platforms"] == ["twitter"]
    
    def test_exec_fallback(self):
        """Test exec_fallback method."""
        prep_res = {
            "platforms": ["twitter"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        exc = Exception("Test error")
        
        result = self.node.exec_fallback(prep_res, exc)
        
        assert result["platforms"] == ["twitter"]
        assert result["topic_or_goal"] == "Announce product"
        assert result["validation_metadata"]["validation_warnings"] == ["Using fallback due to error: Test error"]
        assert result["validation_metadata"]["validation_errors"] == ["Test error"]
        assert result["validation_metadata"]["quality_score"] == 0.5


class TestIntegration:
    """Integration tests for the nodes module."""
    
    def test_validator_and_node_integration(self):
        """Test integration between InputValidator and EngagementManagerNode."""
        config = ValidationConfig()
        config.MAX_TOPIC_LENGTH = 100
        config.SUPPORTED_PLATFORMS = {"twitter", "linkedin"}
        
        validator = InputValidator(config)
        node = EngagementManagerNode(validation_config=config)
        
        # Test that node uses the same config as validator
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
    
    def test_node_validation_integration(self):
        """Test integration of validation in node execution."""
        node = EngagementManagerNode()
        
        # Test with valid input
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Valid topic",
                "intents_by_platform": {
                    "twitter": {"value": "engagement"}
                }
            }
        }
        
        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        
        assert exec_res["validation_metadata"]["quality_score"] > 0.8  # Should be high for valid input
        assert len(exec_res["validation_metadata"]["validation_errors"]) == 0
    
    def test_node_validation_integration_with_issues(self):
        """Test integration of validation in node execution with issues."""
        node = EngagementManagerNode()
        
        # Test with problematic input
        shared = {
            "task_requirements": {
                "platforms": ["unsupported_platform"],
                "topic_or_goal": "ab",  # Too short
                "intents_by_platform": {}
            }
        }
        
        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        
        assert exec_res["validation_metadata"]["quality_score"] < 0.8  # Should be lower due to issues
        assert len(exec_res["validation_metadata"]["validation_errors"]) > 0
    
    def test_node_streaming_integration(self):
        """Test integration of streaming in node execution."""
        mock_stream = Mock()
        node = EngagementManagerNode()
        
        shared = {
            "stream": mock_stream,
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": "Test topic",
                "intents_by_platform": {}
            }
        }
        
        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        result = node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        # Verify streaming events were emitted
        assert mock_stream.emit.call_count == 3
    
    def test_node_error_handling_integration(self):
        """Test integration of error handling in node execution."""
        node = EngagementManagerNode()
        
        # Test with invalid input that should trigger fallback behavior
        shared = {
            "task_requirements": {
                "platforms": ["unsupported_platform"],
                "topic_or_goal": "ab",
                "intents_by_platform": {}
            }
        }
        
        prep_res = node.prep(shared)
        
        # The node should handle validation errors gracefully and use defaults
        assert prep_res["platforms"] == ["twitter", "linkedin"]  # Defaults
        assert prep_res["topic_or_goal"] == "Announce product"  # Defaults
        
        exec_res = node.exec(prep_res)
        
        # Should have validation metadata indicating issues
        assert "validation_metadata" in exec_res
        assert len(exec_res["validation_metadata"]["validation_errors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])