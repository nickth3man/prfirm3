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
        Initialize the ValidationError.
        
        Parameters:
            message (str): Human-readable error message.
            field (str, optional): Name of the field related to the error, if applicable.
            code (str, optional): Machine-readable error code describing the error.
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
    Sanitize and normalize a text string by removing control characters, collapsing whitespace, and enforcing length limits.
    
    Removes null bytes and other ASCII control characters, trims and collapses consecutive whitespace to single spaces, and ensures the result is non-empty and does not exceed max_length.
    
    Parameters:
        text (str): Input string to sanitize.
        max_length (int): Maximum allowed length of the sanitized string (default 1000).
    
    Returns:
        str: The sanitized, normalized string.
    
    Raises:
        ValidationError: If text is not a string (code "TYPE_ERROR"), is empty after sanitization ("EMPTY_ERROR"),
                         or exceeds max_length ("LENGTH_ERROR").
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
    Validate and sanitize a topic string for use (ensuring length, safety, and reasonable repetition).
    
    Performs string sanitization, enforces minimum/maximum length, scans for common malicious patterns (scripts, data/JS/VBScript protocols, iframes/embeds, and inline event handlers), and rejects topics with excessive repetition of the same word.
    
    Args:
        topic (str): Input topic to validate and sanitize.
        min_length (int): Minimum allowed character length (inclusive). Defaults to 3.
        max_length (int): Maximum allowed character length (inclusive). Passed to the sanitizer and enforced. Defaults to 500.
    
    Returns:
        str: The sanitized topic string.
    
    Raises:
        ValidationError: If the input is not a valid string after sanitization, is too short, contains potentially malicious content, or has excessive word repetition.
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
    Normalize a platform name into the canonical lowercase form, expanding common abbreviations.
    
    This returns a normalized platform identifier suitable for lookup against SUPPORTED_PLATFORMS.
    Common shorthand mappings are expanded (e.g., "x" -> "twitter", "fb" -> "facebook", "ig" -> "instagram",
    "yt" -> "youtube", "tt" -> "tiktok", "li" -> "linkedin").
    
    Parameters:
        platform (str): Platform name or common abbreviation to normalize.
    
    Returns:
        str: The normalized platform name (lowercased and trimmed, with common abbreviations expanded).
    
    Raises:
        ValidationError: If `platform` is not a string (code "TYPE_ERROR").
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
    
    Parses `platforms_text` into individual platform identifiers, normalizes each via
    `normalize_platform_name`, enforces a maximum count, ensures each platform is
    supported (per `SUPPORTED_PLATFORMS`), and prevents duplicates.
    
    Parameters:
        platforms_text (str): Comma-separated platform names (e.g. "Twitter, LinkedIn").
        max_platforms (int): Maximum allowed platforms; defaults to 10.
    
    Returns:
        List[str]: A list of normalized, validated platform names.
    
    Raises:
        ValidationError: For non-string input (code "TYPE_ERROR"), empty input
            ("EMPTY_ERROR"), too many platforms ("COUNT_ERROR"), unsupported
            platforms ("UNSUPPORTED_PLATFORM"), or duplicate entries
            ("DUPLICATE_PLATFORM").
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
    Validate and sanitize brand bible content, enforcing size and security checks.
    
    Performs type and size validation (bytes, via UTF-8), runs general string sanitization, and scans for common potentially malicious constructs (scripts, iframes, inline event handlers, data/JS/VBScript URIs, and embedded objects).
    
    Parameters:
        content (str): Raw brand bible content to validate.
        max_size (int): Maximum allowed content size in bytes (measured after UTF-8 encoding).
    
    Returns:
        str: The sanitized, validated content.
    
    Raises:
        ValidationError: If `content` is not a string (code "TYPE_ERROR"), exceeds `max_size` (code "SIZE_ERROR"), or contains detected malicious elements (code "SECURITY_ERROR").
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
    Validate a file path for upload and return basic metadata.
    
    Performs existence, size, extension, and filename-safety checks and returns a metadata dictionary on success.
    
    Parameters:
        file_path (str): Path to the uploaded file.
        allowed_extensions (List[str], optional): Allowed file suffixes (including the leading dot). Defaults to [".xml", ".json", ".txt", ".md"].
        max_size (int, optional): Maximum allowed file size in bytes. Defaults to 10 * 1024 * 1024 (10 MiB).
    
    Returns:
        Dict[str, Any]: Metadata for the validated file with keys:
            - "path": str absolute or given path
            - "name": str file name
            - "size": int file size in bytes
            - "extension": str file extension (lowercased, leading dot)
            - "modified": datetime last modification timestamp
    
    Raises:
        ValidationError: If the file does not exist ("NOT_FOUND"), exceeds max_size ("SIZE_ERROR"),
                         has an unsupported extension ("TYPE_ERROR"), or contains suspicious/unsafe
                         filename/path patterns ("SECURITY_ERROR").
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
    Determine whether an identifier may make a request under the configured rate limit window.
    
    Checks and updates an in-memory timestamp store for the given identifier, removing timestamps older than the sliding window. If the number of requests in the window is already at or above max_requests the function returns False and does not record the current request; otherwise it records the current timestamp and returns True.
    
    Parameters:
        identifier (str): Key to track request timestamps (e.g., IP address or user ID).
        max_requests (int): Maximum allowed requests within the time window.
        window_seconds (int): Length of the sliding time window in seconds.
    
    Returns:
        bool: True if the request is allowed (and recorded); False if the rate limit would be exceeded.
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
    Validate and normalize a complete shared store.
    
    Performs deep validation of the top-level shared dictionary:
    - Ensures `task_requirements` exists and is a dict.
    - Validates and normalizes `task_requirements["topic_or_goal"]` via `validate_topic`.
    - Validates and normalizes `task_requirements["platforms"]` (accepts a comma-separated string or list) via `validate_platforms`.
    - If present, validates `brand_bible["xml_raw"]` via `validate_brand_bible_content`.
    - Preserves `stream` if present and adds a `validation_timestamp` (ISO 8601).
    
    Parameters:
        shared (Dict[str, Any]): The shared store to validate.
    
    Returns:
        Dict[str, Any]: A validated and possibly normalized shared store containing keys
        "task_requirements", "brand_bible", "stream" (if present), and "validation_timestamp".
    
    Raises:
        ValidationError: On type errors, missing required fields, or validation failures from nested validators.
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
    Sanitize a string for safe inclusion in LLM prompts by removing control characters, stripping HTML-like tags, escaping delimiter characters, and truncating overly long input.
    
    This function returns an empty string for non-str inputs. It:
    - Removes null bytes and ASCII control characters.
    - Escapes backslashes, double quotes, single quotes, and common whitespace characters (`\n`, `\r`, `\t`) so they appear as literals in prompts.
    - Strips any remaining HTML-like tags.
    - Truncates the result to 10,000 characters and appends "..." when truncation occurs.
    
    Parameters:
        text: Input to sanitize. If not a string, the function returns an empty string.
    
    Returns:
        A sanitized string safe for embedding in LLM prompts.
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
    Validate and sanitize topic, platforms, and optional brand bible content, returning a validated shared store.
    
    Performs full validation/sanitization of:
    - topic: enforces length, content and security checks via validate_topic.
    - platforms_text: comma-separated platform names normalized and validated via validate_platforms.
    - brand_bible_content: optional large-text validation via validate_brand_bible_content.
    
    Parameters:
        topic (str): User-provided topic string.
        platforms_text (str): Comma-separated platform names.
        brand_bible_content (str, optional): Raw brand bible text; empty string disables validation.
    
    Returns:
        Dict[str, Any]: A validated shared store (contains "task_requirements" with "topic_or_goal" and "platforms", "brand_bible" with "xml_raw", "stream", plus any fields added by validate_shared_store such as a validation timestamp).
    
    Raises:
        ValidationError: If any input fails validation (propagates errors from underlying validators).
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