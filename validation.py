"""Comprehensive input validation and sanitization for the Virtual PR Firm application.

This module provides robust validation for all user inputs including topic content,
platform names, brand bible content, and file uploads. It implements security
measures to prevent injection attacks and ensure data integrity.

Features:
- Input sanitization and validation
- Platform name normalization
- Content length and format validation
- Security checks for malicious content
- Rate limiting validation
- File upload validation

Example Usage:
    >>> from validation import validate_topic, validate_platforms
    >>> validate_topic("Announce product launch")
    >>> platforms = validate_platforms("twitter, linkedin")
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, code: str = None):
        """
        Initialize a ValidationError.
        
        Parameters:
            message (str): Human-readable error message describing the validation failure.
            field (str, optional): Name of the field related to the error, if applicable.
            code (str, optional): Machine-friendly error code for programmatic handling.
        
        The instance will expose .message, .field, and .code attributes and behaves like a standard Exception.
        """
        self.message = message
        self.field = field
        self.code = code
        super().__init__(self.message)


class RateLimitError(ValidationError):
    """Exception for rate limiting violations."""
    pass


# Supported platforms with their characteristics
SUPPORTED_PLATFORMS = {
    "twitter": {
        "max_length": 280,
        "supports_hashtags": True,
        "supports_mentions": True,
        "supports_links": True,
    },
    "linkedin": {
        "max_length": 3000,
        "supports_hashtags": True,
        "supports_mentions": True,
        "supports_links": True,
    },
    "facebook": {
        "max_length": 63206,
        "supports_hashtags": True,
        "supports_mentions": True,
        "supports_links": True,
    },
    "instagram": {
        "max_length": 2200,
        "supports_hashtags": True,
        "supports_mentions": True,
        "supports_links": False,
    },
    "tiktok": {
        "max_length": 150,
        "supports_hashtags": True,
        "supports_mentions": True,
        "supports_links": False,
    },
    "youtube": {
        "max_length": 5000,
        "supports_hashtags": True,
        "supports_mentions": False,
        "supports_links": True,
    }
}

# Rate limiting storage (in production, use Redis or database)
_rate_limit_store = defaultdict(list)


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """
    Return a sanitized, normalized string suitable for further validation.
    
    Removes null bytes and non-printable control characters, collapses and trims whitespace, and enforces length constraints.
    
    Parameters:
        max_length (int): Maximum allowed character length for the returned string (default 1000).
    
    Returns:
        str: The sanitized string.
    
    Raises:
        ValidationError: If input is not a string, is empty after sanitization, or exceeds max_length.
    """
    if not isinstance(text, str):
        raise ValidationError("Input must be a string", "input", "TYPE_ERROR")
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Check length
    if len(text) > max_length:
        raise ValidationError(
            f"Input too long. Maximum {max_length} characters allowed.",
            "input", "LENGTH_ERROR"
        )
    
    if len(text) < 1:
        raise ValidationError("Input cannot be empty", "input", "EMPTY_ERROR")
    
    return text


def validate_topic(topic: str, min_length: int = 3, max_length: int = 500) -> str:
    """
    Validate and sanitize a topic string for safe use.
    
    Performs string sanitization (whitespace/control-char normalization and length trimming), enforces minimum and maximum length, rejects common injection and embedded-content patterns (script, iframe, javascript:, data URLs, event handlers, etc.), and rejects topics with excessive repeated words.
    
    Parameters:
        topic (str): Input topic to validate.
        min_length (int): Minimum allowed character length (default 3).
        max_length (int): Maximum allowed character length; also passed to the underlying sanitizer (default 500).
    
    Returns:
        str: The sanitized, validated topic.
    
    Raises:
        ValidationError: If the input is not a valid topic. Error codes used include:
            - "LENGTH_ERROR" when shorter than min_length,
            - "SECURITY_ERROR" when suspicious/malicious patterns are detected,
            - "CONTENT_ERROR" when excessive word repetition is found.
    """
    # Basic sanitization
    topic = sanitize_string(topic, max_length)
    
    # Check minimum length
    if len(topic) < min_length:
        raise ValidationError(
            f"Topic too short. Minimum {min_length} characters required.",
            "topic", "LENGTH_ERROR"
        )
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'data:text/html',  # Data URLs
        r'vbscript:',  # VBScript
        r'on\w+\s*=',  # Event handlers
        r'<iframe[^>]*>',  # Iframe tags
        r'<object[^>]*>',  # Object tags
        r'<embed[^>]*>',  # Embed tags
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ValidationError(
                "Topic contains potentially malicious content",
                "topic", "SECURITY_ERROR"
            )
    
    # Check for excessive repetition
    words = topic.split()
    if len(words) > 2:
        word_counts = {}
        for word in words:
            word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
            if word_counts[word.lower()] > 3:
                raise ValidationError(
                    "Topic contains excessive word repetition",
                    "topic", "CONTENT_ERROR"
                )
    
    return topic


def normalize_platform_name(platform: str) -> str:
    """
    Normalize a platform identifier to the canonical platform name.
    
    Converts the input to lowercase, trims surrounding whitespace, and maps common short forms
    (e.g., "x" -> "twitter", "fb" -> "facebook", "ig" -> "instagram", "yt" -> "youtube",
    "tt" -> "tiktok", "li" -> "linkedin"). If the input is already a known canonical name it
    is returned in lowercase trimmed form; otherwise the normalized input is returned as-is.
    
    Raises:
        ValidationError: if `platform` is not a string.
    """
    if not isinstance(platform, str):
        raise ValidationError("Platform must be a string", "platform", "TYPE_ERROR")
    
    # Normalize to lowercase and remove whitespace
    normalized = platform.lower().strip()
    
    # Handle common variations
    platform_mappings = {
        "x": "twitter",
        "fb": "facebook",
        "ig": "instagram",
        "yt": "youtube",
        "tt": "tiktok",
        "li": "linkedin",
    }
    
    return platform_mappings.get(normalized, normalized)


def validate_platforms(platforms_text: str, max_platforms: int = 10) -> List[str]:
    """
    Validate and normalize a comma-separated list of platform names.
    
    Takes a comma-separated string, normalizes each entry via normalize_platform_name, enforces a maximum count,
    ensures each platform is supported (per SUPPORTED_PLATFORMS), and rejects duplicates. Returns a list of
    canonical platform names suitable for downstream use.
    
    Parameters:
        platforms_text (str): Comma-separated platform names (e.g., "twitter, facebook, linkedin").
        max_platforms (int): Maximum allowed platforms; exceeding this raises a validation error.
    
    Returns:
        List[str]: Canonical, validated platform names.
    
    Raises:
        ValidationError: If input is not a string, no platforms are provided, the count exceeds max_platforms,
                         any platform is unsupported, or a duplicate platform is present.
    """
    if not isinstance(platforms_text, str):
        raise ValidationError("Platforms must be a string", "platforms", "TYPE_ERROR")
    
    # Split and normalize platforms
    raw_platforms = [p.strip() for p in platforms_text.split(",") if p.strip()]
    
    if not raw_platforms:
        raise ValidationError("At least one platform must be specified", "platforms", "EMPTY_ERROR")
    
    if len(raw_platforms) > max_platforms:
        raise ValidationError(
            f"Too many platforms. Maximum {max_platforms} allowed.",
            "platforms", "COUNT_ERROR"
        )
    
    # Normalize and validate each platform
    validated_platforms = []
    for platform in raw_platforms:
        normalized = normalize_platform_name(platform)
        
        if normalized not in SUPPORTED_PLATFORMS:
            raise ValidationError(
                f"Unsupported platform: {platform}. Supported: {', '.join(SUPPORTED_PLATFORMS.keys())}",
                "platforms", "UNSUPPORTED_PLATFORM"
            )
        
        if normalized in validated_platforms:
            raise ValidationError(
                f"Duplicate platform: {platform}",
                "platforms", "DUPLICATE_PLATFORM"
            )
        
        validated_platforms.append(normalized)
    
    return validated_platforms


def validate_brand_bible_content(content: str, max_size: int = 10 * 1024 * 1024) -> str:
    """
    Validate and sanitize brand bible content.
    
    Performs type and size checks, runs general string sanitization, and rejects content containing common
    embedded/malicious constructs (scripts, iframes, javascript/data/vbscript URIs, inline event handlers, forms/inputs/objects/embeds).
    Returns the sanitized content when valid.
    
    Parameters:
        content (str): Raw brand bible text to validate.
        max_size (int): Maximum allowed size in bytes (default 10 MiB).
    
    Returns:
        str: Sanitized brand bible content.
    
    Raises:
        ValidationError: If the input is not a string, exceeds max_size, or contains potentially malicious elements.
    """
    if not isinstance(content, str):
        raise ValidationError("Brand bible content must be a string", "brand_bible", "TYPE_ERROR")
    
    # Check size
    content_size = len(content.encode('utf-8'))
    if content_size > max_size:
        raise ValidationError(
            f"Brand bible content too large. Maximum {max_size} bytes allowed.",
            "brand_bible", "SIZE_ERROR"
        )
    
    # Basic sanitization
    content = sanitize_string(content, max_length=content_size)
    
    # Check for malicious content
    malicious_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'<form[^>]*>',
        r'<input[^>]*>',
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            raise ValidationError(
                "Brand bible content contains potentially malicious elements",
                "brand_bible", "SECURITY_ERROR"
            )
    
    return content


def validate_file_upload(file_path: str, allowed_extensions: List[str] = None, max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
    """
    Validate a file upload path and return basic file metadata.
    
    Validates that the path exists, the file size does not exceed max_size, the file extension is in allowed_extensions,
    and the filename does not contain suspicious/traversal characters. Returns a metadata dict with keys: `path`,
    `name`, `size`, `extension`, and `modified`.
    
    Parameters:
        file_path (str): Path to the uploaded file.
        allowed_extensions (List[str], optional): Allowed extensions (including leading dot). Defaults to [".xml", ".json", ".txt", ".md"].
        max_size (int, optional): Maximum allowed file size in bytes.
    
    Returns:
        Dict[str, Any]: File metadata containing `path`, `name`, `size`, `extension`, and `modified` (datetime).
    
    Raises:
        ValidationError: If the file is not found, exceeds max_size, has a disallowed extension, or its name matches suspicious patterns.
    """
    if allowed_extensions is None:
        allowed_extensions = [".xml", ".json", ".txt", ".md"]
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise ValidationError(f"File not found: {file_path}", "file", "NOT_FOUND")
    
    # Check file size
    file_size = path.stat().st_size
    if file_size > max_size:
        raise ValidationError(
            f"File too large. Maximum {max_size} bytes allowed.",
            "file", "SIZE_ERROR"
        )
    
    # Check file extension
    if path.suffix.lower() not in allowed_extensions:
        raise ValidationError(
            f"File type not allowed. Allowed: {', '.join(allowed_extensions)}",
            "file", "TYPE_ERROR"
        )
    
    # Check for suspicious file names
    suspicious_patterns = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'[<>:"|?*]',  # Invalid characters
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, str(path)):
            raise ValidationError(
                "File name contains invalid characters",
                "file", "SECURITY_ERROR"
            )
    
    return {
        "path": str(path),
        "name": path.name,
        "size": file_size,
        "extension": path.suffix.lower(),
        "modified": datetime.fromtimestamp(path.stat().st_mtime)
    }


def check_rate_limit(identifier: str, max_requests: int = 60, window_seconds: int = 60) -> bool:
    """
    Return True if the identifier is allowed to make a request under the specified rate limits; otherwise return False.
    
    Tracks request timestamps in an in-memory store and enforces at most `max_requests` within the past `window_seconds`. If allowed, the current timestamp is recorded (mutating the module's rate-limit store). Note: this is an in-memory, process-local limiter and is not persistent or distributed.
     
    Parameters:
        identifier (str): Unique key for rate limiting (e.g., IP address or user ID).
        max_requests (int): Maximum number of requests permitted within the rolling window.
        window_seconds (int): Length of the rolling time window in seconds.
    
    Returns:
        bool: True when the request is allowed (and recorded); False when the rate limit has been exceeded.
    """
    now = datetime.now()
    window_start = now - timedelta(seconds=window_seconds)
    
    # Clean old entries
    _rate_limit_store[identifier] = [
        timestamp for timestamp in _rate_limit_store[identifier]
        if timestamp > window_start
    ]
    
    # Check if limit exceeded
    if len(_rate_limit_store[identifier]) >= max_requests:
        return False
    
    # Add current request
    _rate_limit_store[identifier].append(now)
    return True


def validate_shared_store(shared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a complete shared store structure and return a sanitized, normalized copy.
    
    Validates the top-level `shared` dict, ensuring required fields are present and correctly typed:
    - `task_requirements` (dict) with a required `topic_or_goal` (string) and `platforms` (string or list).
      - `topic_or_goal` is sanitized and validated via `validate_topic`.
      - `platforms` may be a comma-separated string or a list of platform names; it is validated and normalized via `validate_platforms`.
    - `brand_bible` (optional dict); if it contains `xml_raw`, that content is validated via `validate_brand_bible_content`.
    
    Returns a new dict containing the validated `task_requirements`, `brand_bible`, the original `stream` value (if any), and a `validation_timestamp` (ISO 8601 string).
    
    Parameters:
        shared (Dict[str, Any]): The shared store to validate.
    
    Returns:
        Dict[str, Any]: A validated and normalized shared store with a `validation_timestamp`.
    
    Raises:
        ValidationError: If `shared` or any required subfield is missing, has the wrong type, or fails validation.
    """
    if not isinstance(shared, dict):
        raise ValidationError("Shared store must be a dictionary", "shared", "TYPE_ERROR")
    
    # Validate task_requirements
    task_requirements = shared.get("task_requirements")
    if not task_requirements:
        raise ValidationError("task_requirements is required", "task_requirements", "MISSING")
    
    if not isinstance(task_requirements, dict):
        raise ValidationError("task_requirements must be a dictionary", "task_requirements", "TYPE_ERROR")
    
    # Validate topic
    topic = task_requirements.get("topic_or_goal")
    if not topic:
        raise ValidationError("topic_or_goal is required", "topic_or_goal", "MISSING")
    
    validated_topic = validate_topic(topic)
    task_requirements["topic_or_goal"] = validated_topic
    
    # Validate platforms
    platforms = task_requirements.get("platforms")
    if not platforms:
        raise ValidationError("platforms is required", "platforms", "MISSING")
    
    if isinstance(platforms, str):
        validated_platforms = validate_platforms(platforms)
    elif isinstance(platforms, list):
        validated_platforms = validate_platforms(",".join(platforms))
    else:
        raise ValidationError("platforms must be a string or list", "platforms", "TYPE_ERROR")
    
    task_requirements["platforms"] = validated_platforms
    
    # Validate brand_bible
    brand_bible = shared.get("brand_bible", {})
    if not isinstance(brand_bible, dict):
        raise ValidationError("brand_bible must be a dictionary", "brand_bible", "TYPE_ERROR")
    
    xml_raw = brand_bible.get("xml_raw", "")
    if xml_raw:
        validated_content = validate_brand_bible_content(xml_raw)
        brand_bible["xml_raw"] = validated_content
    
    # Create validated shared store
    validated_shared = {
        "task_requirements": task_requirements,
        "brand_bible": brand_bible,
        "stream": shared.get("stream"),
        "validation_timestamp": datetime.now().isoformat()
    }
    
    return validated_shared


def sanitize_for_llm_prompt(text: str) -> str:
    """
    Prepare and sanitize a string for safe inclusion in LLM prompts.
    
    Returns a sanitized version of `text` suitable for embedding in prompt templates:
    - Non-string inputs return an empty string.
    - Removes ASCII control characters and null bytes.
    - Escapes backslashes, quotes, and common whitespace characters (newline, carriage return, tab) into their literal escape sequences.
    - Strips HTML-like tags and truncates output to 10,000 characters, appending an ellipsis if truncated.
    
    This function performs lightweight, conservative sanitization to reduce prompt-injection risk; it is not a substitute for context-specific escaping or a full HTML/XML sanitizer.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove or escape potentially dangerous characters
    # This is a basic implementation - consider using a proper HTML/XML parser for production
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Escape special characters that might interfere with prompt formatting
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    text = text.replace("'", "\\'")
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    
    # Remove any remaining HTML-like tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Limit length to prevent prompt injection
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def validate_and_sanitize_inputs(topic: str, platforms_text: str, brand_bible_content: str = "") -> Dict[str, Any]:
    """
    Validate and sanitize topic, platforms, and optional brand bible content and return a normalized "shared" payload.
    
    Performs high-level orchestration: sanitizes the topic, normalizes and validates comma-separated platform names, validates optional brand bible content, assembles a canonical shared dictionary and runs final validation via validate_shared_store.
    
    Parameters:
        topic: The user-provided topic or goal string (will be sanitized and length-checked).
        platforms_text: Comma-separated platform names (case/abbreviation-normalized and deduplicated).
        brand_bible_content: Optional raw brand bible text (validated and size-checked); pass empty string if none.
    
    Returns:
        Dict[str, Any]: A validated shared store containing at minimum:
            - task_requirements: {"platforms": List[str], "topic_or_goal": str}
            - brand_bible: {"xml_raw": str}
            - stream: None (reserved)
    
    Raises:
        ValidationError: If any component (topic, platforms, or brand bible) fails validation.
    """
    try:
        # Validate and sanitize topic
        validated_topic = validate_topic(topic)
        
        # Validate and normalize platforms
        validated_platforms = validate_platforms(platforms_text)
        
        # Validate brand bible content
        validated_brand_bible = validate_brand_bible_content(brand_bible_content) if brand_bible_content else ""
        
        # Create validated shared store
        shared = {
            "task_requirements": {
                "platforms": validated_platforms,
                "topic_or_goal": validated_topic,
            },
            "brand_bible": {
                "xml_raw": validated_brand_bible
            },
            "stream": None,
        }
        
        # Final validation
        validated_shared = validate_shared_store(shared)
        
        logger.info(
            "Inputs validated successfully",
            extra={
                "topic_length": len(validated_topic),
                "platform_count": len(validated_platforms),
                "brand_bible_size": len(validated_brand_bible),
                "platforms": validated_platforms
            }
        )
        
        return validated_shared
        
    except ValidationError as e:
        logger.warning(
            "Input validation failed",
            extra={
                "error": str(e),
                "field": e.field,
                "code": e.code,
                "topic_length": len(topic) if topic else 0,
                "platforms_text": platforms_text
            }
        )
        raise


if __name__ == "__main__":
    # Test validation functions
    try:
        # Test valid inputs
        result = validate_and_sanitize_inputs(
            "Announce product launch",
            "twitter, linkedin",
            "<brand>Test brand</brand>"
        )
        print("Validation successful:", result)
        
        # Test invalid inputs
        try:
            validate_topic("")
        except ValidationError as e:
            print(f"Expected error: {e}")
        
        try:
            validate_platforms("invalid_platform")
        except ValidationError as e:
            print(f"Expected error: {e}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")