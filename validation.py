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
        Initialize a ValidationError with message and optional field and error code.
        
        Parameters:
            message (str): Human-readable error message describing the validation failure.
            field (str, optional): Name of the field associated with the error, if applicable.
            code (str, optional): Machine-readable error code for programmatic handling.
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
    Sanitize and normalize a text string for safe downstream use.
    
    Removes null bytes and other ASCII control characters, collapses and trims whitespace, and enforces length limits.
    Returns the cleaned string.
    
    Parameters:
        text (str): Input text to sanitize.
        max_length (int): Maximum allowed length of the returned string; a longer input triggers a validation error.
    
    Returns:
        str: The sanitized, trimmed string.
    
    Raises:
        ValidationError: If text is not a string (code "TYPE_ERROR"), is empty after trimming (code "EMPTY_ERROR"), or exceeds max_length (code "LENGTH_ERROR").
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
    Validate and sanitize a user-provided topic string.
    
    Performs type- and content-safe sanitization, enforces minimum and maximum length,
    and rejects topics containing common injection or embedding vectors and excessive word repetition.
    
    Parameters:
        topic (str): Raw topic input to validate and sanitize.
        min_length (int): Minimum allowed length after sanitization (default 3).
        max_length (int): Maximum allowed length passed to the sanitizer (default 500).
    
    Returns:
        str: The sanitized topic string.
    
    Raises:
        ValidationError: If the topic is the wrong type/length or contains disallowed content.
            - "LENGTH_ERROR" when shorter than min_length.
            - "SECURITY_ERROR" when suspicious patterns (script, data:, javascript:, vbscript:, event handlers, iframe/object/embed, etc.) are found.
            - "CONTENT_ERROR" when excessive word repetition is detected.
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
    Normalize a platform name to the canonical platform identifier.
    
    Converts common short forms and abbreviations (e.g., "x" -> "twitter", "fb" -> "facebook",
    "ig" -> "instagram", "yt" -> "youtube", "tt" -> "tiktok", "li" -> "linkedin") and
    returns the lowercased, trimmed name for other inputs.
    
    Parameters:
        platform (str): Platform name or abbreviation to normalize.
    
    Returns:
        str: Canonical platform name.
    
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
    
    Takes a comma-separated string of platform identifiers, normalizes each entry
    (e.g., "fb" -> "facebook", "x" -> "twitter"), enforces uniqueness and a
    maximum count, and verifies each platform is supported by SUPPORTED_PLATFORMS.
    
    Parameters:
        platforms_text (str): Comma-separated platform names (e.g., "twitter, facebook, ig").
        max_platforms (int): Maximum allowed platforms in the result (default 10).
    
    Returns:
        List[str]: Normalized, deduplicated list of supported platform keys.
    
    Raises:
        ValidationError: If input is not a string ("TYPE_ERROR"), is empty ("EMPTY_ERROR"),
                         exceeds max_platforms ("COUNT_ERROR"), contains an unsupported
                         platform ("UNSUPPORTED_PLATFORM"), or includes duplicates
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
    Validate and sanitize brand bible content, rejecting oversized or potentially malicious input.
    
    Parameters:
        content (str): Raw brand bible content (e.g., XML/HTML/text) to validate and sanitize.
        max_size (int): Maximum allowed size in bytes for the content (defaults to 10 MB).
    
    Returns:
        str: Sanitized content suitable for further processing.
    
    Raises:
        ValidationError: If `content` is not a string ("TYPE_ERROR"), exceeds `max_size` ("SIZE_ERROR"),
                         or contains potentially malicious elements such as scripts, inline handlers,
                         iframes/embeds, or data/JavaScript URI patterns ("SECURITY_ERROR").
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
    Validate a file upload path and return its metadata.
    
    Validates that the file exists, is within the allowed size, uses an allowed extension, and that the filename does not contain suspicious/traversal characters. On success returns a metadata dictionary describing the file.
    
    Parameters:
        file_path (str): Path to the uploaded file.
        allowed_extensions (List[str], optional): Allowed file extensions (including leading dot). Defaults to [".xml", ".json", ".txt", ".md"].
        max_size (int, optional): Maximum allowed file size in bytes. Defaults to 10 * 1024 * 1024 (10 MB).
    
    Returns:
        Dict[str, Any]: Metadata with keys:
            - path (str): Absolute or provided path string.
            - name (str): Filename.
            - size (int): File size in bytes.
            - extension (str): Lowercased file extension (including leading dot).
            - modified (datetime): Last modified timestamp as a datetime object.
    
    Raises:
        ValidationError: If the file does not exist ("NOT_FOUND"), is too large ("SIZE_ERROR"), has a disallowed extension ("TYPE_ERROR"), or the filename contains potentially dangerous characters ("SECURITY_ERROR").
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
    Return whether a request for the given identifier is within the configured rate limit.
    
    Performs a sliding-window check against an in-memory timestamp store and records the current request when allowed. The function prunes timestamps older than `window_seconds` for the identifier before evaluating the count.
    
    Parameters:
        identifier: Unique identifier to rate-limit (for example, an IP address or user ID).
        max_requests: Maximum allowed requests in the sliding window.
        window_seconds: Length of the sliding window in seconds.
    
    Returns:
        True if the request is allowed (timestamp recorded); False if the rate limit has been reached.
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
    Validate and normalize the complete `shared` store used by the application.
    
    Performs structural and semantic checks, normalizes/validates nested fields, and returns a cleaned copy suitable for downstream processing.
    
    Details:
    - Ensures `shared` is a dict.
    - Validates `task_requirements` exists and is a dict.
    - Validates and sanitizes `task_requirements["topic_or_goal"]` via `validate_topic`.
    - Validates `task_requirements["platforms"]` (accepts a comma-separated string or a list); normalizes via `validate_platforms` and replaces the original value with the validated list.
    - Validates `brand_bible` when present as a dict and, if `brand_bible["xml_raw"]` is provided, validates it via `validate_brand_bible_content`.
    - Returns a new dict containing `task_requirements`, `brand_bible`, the original `stream` (if any), and a `validation_timestamp` (ISO 8601 string).
    
    Parameters:
        shared (Dict[str, Any]): The input shared store to validate.
    
    Returns:
        Dict[str, Any]: Validated and normalized shared store with a `validation_timestamp`.
    
    Raises:
        ValidationError: On any structural or content validation failure. The exception includes a message, the related field name, and an error code.
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
    Sanitize a string for safe inclusion in LLM prompts by removing control characters, stripping HTML-like tags, escaping quotes/backslashes/newlines/tabs, and truncating to a safe maximum length.
    
    Parameters:
        text (str): Input text to sanitize. Non-string inputs return an empty string.
    
    Returns:
        str: Sanitized string suitable for embedding in prompts (control characters removed, quotes/backslashes escaped, HTML-like tags removed, and truncated to 10,000 characters with an appended ellipsis if truncated).
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
    Validate and sanitize topic, platform list, and optional brand-bible content, returning a validated "shared" structure.
    
    Performs full input validation:
    - Sanitizes and enforces constraints on `topic`.
    - Parses, normalizes and validates `platforms_text` (comma-separated).
    - Optionally validates `brand_bible_content` when provided.
    Returns the normalized shared store as accepted by the application (includes `task_requirements`, `brand_bible`, `stream`, and additional validation metadata).
    
    Parameters:
        topic (str): User-provided topic or goal text.
        platforms_text (str): Comma-separated platform names (e.g., "twitter, facebook").
        brand_bible_content (str): Optional raw brand bible content; validated when non-empty.
    
    Returns:
        Dict[str, Any]: The validated and sanitized shared structure ready for downstream use.
    
    Raises:
        ValidationError: If any input fails validation or security checks.
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