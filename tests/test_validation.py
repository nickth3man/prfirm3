"""Unit tests for input validation and sanitization."""

import pytest
import tempfile
import os
from pathlib import Path

from validation import (
    validate_shared_store, ValidationError, InputValidator,
    RateLimitValidator, sanitize_text, normalize_platforms
)


class TestInputValidator:
    """Test InputValidator functionality."""
    
    def test_validate_shared_store_valid(self):
        """Test validation with valid shared store."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        # Should not raise an exception
        validate_shared_store(shared)
        
        # Check that platforms were normalized
        assert shared["task_requirements"]["platforms"] == ["twitter", "linkedin"]
    
    def test_validate_shared_store_missing_task_requirements(self):
        """Test validation with missing task_requirements."""
        shared = {"brand_bible": {"xml_raw": ""}}
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements"
        assert "required" in errors[0].message.lower()
    
    def test_validate_shared_store_missing_platforms(self):
        """Test validation with missing platforms."""
        shared = {
            "task_requirements": {
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.platforms"
        assert "required" in errors[0].message.lower()
    
    def test_validate_shared_store_missing_topic(self):
        """Test validation with missing topic."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"]
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.topic_or_goal"
        assert "required" in errors[0].message.lower()
    
    def test_validate_shared_store_empty_platforms(self):
        """Test validation with empty platforms list."""
        shared = {
            "task_requirements": {
                "platforms": [],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.platforms"
        assert "at least one" in errors[0].message.lower()
    
    def test_validate_shared_store_unsupported_platform(self):
        """Test validation with unsupported platform."""
        shared = {
            "task_requirements": {
                "platforms": ["unsupported_platform"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.platforms[0]"
        assert "unsupported" in errors[0].message.lower()
    
    def test_validate_shared_store_topic_too_short(self):
        """Test validation with topic that's too short."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": ""
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.topic_or_goal"
        assert "at least" in errors[0].message.lower()
    
    def test_validate_shared_store_topic_too_long(self):
        """Test validation with topic that's too long."""
        long_topic = "x" * 501  # Exceeds 500 character limit
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": long_topic
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.topic_or_goal"
        assert "no more than" in errors[0].message.lower()
    
    def test_validate_shared_store_spam_content(self):
        """Test validation with spam content."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": "Buy now! Limited time offer! Click here!"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.topic_or_goal"
        assert "spam" in errors[0].message.lower()
    
    def test_validate_shared_store_excessive_special_chars(self):
        """Test validation with excessive special characters."""
        special_chars_topic = "Test!!!@@@###$$$%%%^^^&&&***((()))___+++---==="
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": special_chars_topic
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "task_requirements.topic_or_goal"
        assert "special characters" in errors[0].message.lower()
    
    def test_validate_shared_store_platform_aliases(self):
        """Test validation with platform aliases."""
        shared = {
            "task_requirements": {
                "platforms": ["x", "li", "fb", "ig"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        # Should not raise an exception
        validate_shared_store(shared)
        
        # Check that aliases were normalized
        assert shared["task_requirements"]["platforms"] == ["twitter", "linkedin", "facebook", "instagram"]
    
    def test_validate_brand_bible_file_path(self):
        """Test validation with brand bible file path."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write("<brand>Test brand</brand>")
            file_path = f.name
        
        try:
            shared = {
                "task_requirements": {
                    "platforms": ["twitter"],
                    "topic_or_goal": "Test topic"
                },
                "brand_bible": {"file_path": file_path}
            }
            
            # Should not raise an exception
            validate_shared_store(shared)
        finally:
            os.unlink(file_path)
    
    def test_validate_brand_bible_file_not_found(self):
        """Test validation with non-existent brand bible file."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"file_path": "/nonexistent/file.xml"}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "brand_bible.file_path"
        assert "does not exist" in errors[0].message.lower()
    
    def test_validate_brand_bible_invalid_extension(self):
        """Test validation with invalid file extension."""
        # Create a temporary file with invalid extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
            f.write("test content")
            file_path = f.name
        
        try:
            shared = {
                "task_requirements": {
                    "platforms": ["twitter"],
                    "topic_or_goal": "Test topic"
                },
                "brand_bible": {"file_path": file_path}
            }
            
            with pytest.raises(ValidationError) as exc_info:
                validate_shared_store(shared)
            
            errors = exc_info.value.errors
            assert len(errors) == 1
            assert errors[0].field == "brand_bible.file_path"
            assert "not allowed" in errors[0].message.lower()
        finally:
            os.unlink(file_path)
    
    def test_validate_brand_bible_dangerous_xml(self):
        """Test validation with dangerous XML content."""
        dangerous_xml = "<script>alert('xss')</script><brand>Test</brand>"
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": dangerous_xml}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_shared_store(shared)
        
        errors = exc_info.value.errors
        assert len(errors) == 1
        assert errors[0].field == "brand_bible.xml_raw"
        assert "dangerous" in errors[0].message.lower()
    
    def test_validate_content_type_valid(self):
        """Test validation with valid content type."""
        validator = InputValidator()
        errors = validator._validate_content_type("press_release")
        assert len(errors) == 0
    
    def test_validate_content_type_invalid(self):
        """Test validation with invalid content type."""
        validator = InputValidator()
        errors = validator._validate_content_type("invalid_type")
        assert len(errors) == 1
        assert "invalid" in errors[0].message.lower()
    
    def test_validate_target_audience_valid(self):
        """Test validation with valid target audience."""
        validator = InputValidator()
        errors = validator._validate_target_audience("Professional developers")
        assert len(errors) == 0
    
    def test_validate_target_audience_too_short(self):
        """Test validation with target audience that's too short."""
        validator = InputValidator()
        errors = validator._validate_target_audience("ab")
        assert len(errors) == 1
        assert "at least 3" in errors[0].message.lower()
    
    def test_validate_target_audience_too_long(self):
        """Test validation with target audience that's too long."""
        validator = InputValidator()
        long_audience = "x" * 201  # Exceeds 200 character limit
        errors = validator._validate_target_audience(long_audience)
        assert len(errors) == 1
        assert "no more than 200" in errors[0].message.lower()


class TestSanitization:
    """Test sanitization functions."""
    
    def test_sanitize_text_normal(self):
        """Test sanitizing normal text."""
        text = "  Hello   World  "
        sanitized = sanitize_text(text)
        assert sanitized == "Hello World"
    
    def test_sanitize_text_control_chars(self):
        """Test sanitizing text with control characters."""
        text = "Hello\x00World\x01\x02"
        sanitized = sanitize_text(text)
        assert sanitized == "HelloWorld"
    
    def test_sanitize_text_too_long(self):
        """Test sanitizing text that's too long."""
        long_text = "x" * 600  # Exceeds 500 character limit
        sanitized = sanitize_text(long_text)
        assert len(sanitized) == 500
    
    def test_sanitize_text_non_string(self):
        """Test sanitizing non-string input."""
        sanitized = sanitize_text(123)
        assert sanitized == "123"
    
    def test_normalize_platforms(self):
        """Test platform normalization."""
        platforms = ["TWITTER", "LinkedIn", "fb", "instagram"]
        normalized = normalize_platforms(platforms)
        assert normalized == ["twitter", "linkedin", "facebook", "instagram"]
    
    def test_normalize_platforms_unsupported(self):
        """Test platform normalization with unsupported platforms."""
        platforms = ["twitter", "unsupported", "linkedin"]
        normalized = normalize_platforms(platforms)
        assert normalized == ["twitter", "linkedin"]  # Unsupported platforms are filtered out


class TestRateLimitValidator:
    """Test RateLimitValidator functionality."""
    
    def test_rate_limit_valid(self):
        """Test rate limiting with valid requests."""
        validator = RateLimitValidator(max_requests=2, window_seconds=60)
        
        # First request should be allowed
        allowed, info = validator.validate_rate_limit("client1")
        assert allowed is True
        assert info["requests_remaining"] == 1
        
        # Second request should be allowed
        allowed, info = validator.validate_rate_limit("client1")
        assert allowed is True
        assert info["requests_remaining"] == 0
        
        # Third request should be rate limited
        allowed, info = validator.validate_rate_limit("client1")
        assert allowed is False
        assert info["rate_limited"] is True
        assert info["requests_remaining"] == 0
    
    def test_rate_limit_different_clients(self):
        """Test rate limiting with different clients."""
        validator = RateLimitValidator(max_requests=1, window_seconds=60)
        
        # Client 1 should be allowed
        allowed, _ = validator.validate_rate_limit("client1")
        assert allowed is True
        
        # Client 2 should also be allowed
        allowed, _ = validator.validate_rate_limit("client2")
        assert allowed is True
        
        # Client 1 should be rate limited
        allowed, _ = validator.validate_rate_limit("client1")
        assert allowed is False
    
    def test_rate_limit_cleanup(self):
        """Test rate limit cleanup of old entries."""
        validator = RateLimitValidator(max_requests=1, window_seconds=1)
        
        # Make a request
        validator.validate_rate_limit("client1")
        
        # Wait for window to expire
        import time
        time.sleep(1.1)
        
        # Should be allowed again after cleanup
        allowed, _ = validator.validate_rate_limit("client1")
        assert allowed is True


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_message(self):
        """Test ValidationError message."""
        errors = [
            ValidationError("field1", "Error 1"),
            ValidationError("field2", "Error 2")
        ]
        error = ValidationError(errors)
        assert str(error) == "Validation failed: 2 errors"
    
    def test_validation_error_errors_attribute(self):
        """Test ValidationError errors attribute."""
        errors = [ValidationError("field1", "Error 1")]
        error = ValidationError(errors)
        assert error.errors == errors
        assert len(error.errors) == 1


if __name__ == '__main__':
    pytest.main([__file__])