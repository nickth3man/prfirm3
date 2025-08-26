"""Validation utilities for the Virtual PR Firm.

This module provides comprehensive input validation and sanitization functions
for the Virtual PR Firm application, ensuring data integrity and security.
"""

import re
import html
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error with field and message."""
    field: str
    message: str
    value: Any = None


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[ValidationError]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, field: str, message: str, value: Any = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field, message, value))
        self.is_valid = False
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def __str__(self) -> str:
        if self.is_valid:
            return "Validation passed"
        return f"Validation failed: {len(self.errors)} errors"


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """Sanitize text input by removing potentially dangerous content.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (truncates if exceeded)
    
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove HTML tags and entities
    sanitized = html.escape(text)
    
    # Remove control characters except newlines and tabs
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Truncate if max_length specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip()
        logger.warning(f"Text truncated to {max_length} characters")
    
    return sanitized


def validate_topic(topic: str, min_length: int = 3, max_length: int = 500) -> ValidationResult:
    """Validate topic/goal input.
    
    Args:
        topic: Topic string to validate
        min_length: Minimum required length
        max_length: Maximum allowed length
    
    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()
    
    if not topic:
        result.add_error("topic", "Topic cannot be empty")
        return result
    
    if not isinstance(topic, str):
        result.add_error("topic", "Topic must be a string", topic)
        return result
    
    sanitized_topic = sanitize_text(topic, max_length)
    
    if len(sanitized_topic) < min_length:
        result.add_error("topic", f"Topic must be at least {min_length} characters long", topic)
    
    # Check for potentially inappropriate content
    inappropriate_patterns = [
        r'\b(hack|crack|steal|illegal|fraud)\b',
        r'<script',
        r'javascript:',
        r'data:text/html',
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, sanitized_topic, re.IGNORECASE):
            result.add_error("topic", "Topic contains inappropriate content", topic)
            break
    
    return result


def validate_platforms(platforms: Union[str, List[str]], 
                      supported_platforms: Optional[List[str]] = None) -> ValidationResult:
    """Validate platform input.
    
    Args:
        platforms: Platform string or list to validate
        supported_platforms: List of supported platform names
    
    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()
    
    if supported_platforms is None:
        supported_platforms = ["twitter", "linkedin", "facebook", "instagram", "tiktok", "youtube"]
    
    # Convert string to list if needed
    if isinstance(platforms, str):
        platform_list = [p.strip() for p in platforms.split(",") if p.strip()]
    elif isinstance(platforms, list):
        platform_list = platforms
    else:
        result.add_error("platforms", "Platforms must be a string or list", platforms)
        return result
    
    if not platform_list:
        result.add_error("platforms", "At least one platform must be specified")
        return result
    
    # Validate each platform
    for platform in platform_list:
        if not isinstance(platform, str):
            result.add_error("platforms", f"Platform must be a string: {platform}", platform)
            continue
        
        sanitized_platform = platform.lower().strip()
        
        if not sanitized_platform:
            result.add_error("platforms", "Platform name cannot be empty", platform)
            continue
        
        if sanitized_platform not in supported_platforms:
            result.add_error("platforms", f"Unsupported platform: {sanitized_platform}", platform)
    
    return result


def validate_shared_store(shared: Dict[str, Any]) -> ValidationResult:
    """Validate shared store structure.
    
    Args:
        shared: Shared store dictionary to validate
    
    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()
    
    if not isinstance(shared, dict):
        result.add_error("shared", "Shared store must be a dictionary", shared)
        return result
    
    # Validate task_requirements
    task_reqs = shared.get("task_requirements")
    if task_reqs is None:
        result.add_error("task_requirements", "task_requirements is required")
    elif not isinstance(task_reqs, dict):
        result.add_error("task_requirements", "task_requirements must be a dictionary", task_reqs)
    else:
        # Validate platforms within task_requirements
        platforms = task_reqs.get("platforms")
        if platforms is None:
            result.add_error("task_requirements.platforms", "platforms is required")
        else:
            platform_result = validate_platforms(platforms)
            if not platform_result:
                for error in platform_result.errors:
                    result.add_error(f"task_requirements.{error.field}", error.message, error.value)
        
        # Validate topic_or_goal
        topic = task_reqs.get("topic_or_goal")
        if topic is None:
            result.add_error("task_requirements.topic_or_goal", "topic_or_goal is required")
        else:
            topic_result = validate_topic(topic)
            if not topic_result:
                for error in topic_result.errors:
                    result.add_error(f"task_requirements.{error.field}", error.message, error.value)
    
    # Validate brand_bible (optional but if present should be dict)
    brand_bible = shared.get("brand_bible")
    if brand_bible is not None and not isinstance(brand_bible, dict):
        result.add_error("brand_bible", "brand_bible must be a dictionary", brand_bible)
    
    return result


def validate_file_upload(file_path: str, allowed_extensions: Optional[List[str]] = None, 
                        max_size_mb: int = 10) -> ValidationResult:
    """Validate file upload.
    
    Args:
        file_path: Path to the file to validate
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB
    
    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult()
    
    if allowed_extensions is None:
        allowed_extensions = [".txt", ".md", ".json", ".xml", ".yaml", ".yml"]
    
    try:
        import os
        from pathlib import Path
        
        file_path_obj = Path(file_path)
        
        # Check if file exists
        if not file_path_obj.exists():
            result.add_error("file_path", f"File does not exist: {file_path}")
            return result
        
        # Check file extension
        file_ext = file_path_obj.suffix.lower()
        if file_ext not in allowed_extensions:
            result.add_error("file_extension", f"File extension not allowed: {file_ext}", file_ext)
        
        # Check file size
        file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            result.add_error("file_size", f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)", file_size_mb)
        
        # Check if file is readable
        if not os.access(file_path_obj, os.R_OK):
            result.add_error("file_permissions", "File is not readable", file_path)
    
    except Exception as e:
        result.add_error("file_validation", f"Error validating file: {str(e)}", file_path)
    
    return result


def sanitize_shared_store(shared: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize shared store data.
    
    Args:
        shared: Shared store dictionary to sanitize
    
    Returns:
        Sanitized shared store dictionary
    """
    sanitized = {}
    
    for key, value in shared.items():
        if key == "task_requirements" and isinstance(value, dict):
            sanitized[key] = {}
            for req_key, req_value in value.items():
                if req_key == "platforms":
                    if isinstance(req_value, str):
                        platforms = [p.strip() for p in req_value.split(",") if p.strip()]
                    else:
                        platforms = req_value
                    sanitized[key][req_key] = [p.lower() for p in platforms]
                elif req_key == "topic_or_goal":
                    sanitized[key][req_key] = sanitize_text(req_value, 500)
                else:
                    sanitized[key][req_key] = req_value
        elif key == "brand_bible" and isinstance(value, dict):
            sanitized[key] = {}
            for bible_key, bible_value in value.items():
                if isinstance(bible_value, str):
                    sanitized[key][bible_key] = sanitize_text(bible_value, 10000)
                else:
                    sanitized[key][bible_key] = bible_value
        else:
            sanitized[key] = value
    
    return sanitized


def validate_and_sanitize_inputs(topic: str, platforms: Union[str, List[str]], 
                                brand_bible_path: Optional[str] = None) -> Tuple[ValidationResult, Dict[str, Any]]:
    """Validate and sanitize all inputs.
    
    Args:
        topic: Topic/goal string
        platforms: Platform string or list
        brand_bible_path: Optional path to brand bible file
    
    Returns:
        Tuple of (ValidationResult, sanitized_shared_store)
    """
    result = ValidationResult()
    
    # Validate individual inputs
    topic_result = validate_topic(topic)
    if not topic_result:
        for error in topic_result.errors:
            result.add_error(error.field, error.message, error.value)
    
    platform_result = validate_platforms(platforms)
    if not platform_result:
        for error in platform_result.errors:
            result.add_error(error.field, error.message, error.value)
    
    # Validate brand bible file if provided
    if brand_bible_path:
        file_result = validate_file_upload(brand_bible_path)
        if not file_result:
            for error in file_result.errors:
                result.add_error(f"brand_bible_file.{error.field}", error.message, error.value)
    
    # Create sanitized shared store
    if isinstance(platforms, str):
        platform_list = [p.strip() for p in platforms.split(",") if p.strip()]
    else:
        platform_list = platforms
    
    sanitized_shared = {
        "task_requirements": {
            "platforms": [p.lower() for p in platform_list],
            "topic_or_goal": sanitize_text(topic, 500)
        },
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }
    
    return result, sanitized_shared