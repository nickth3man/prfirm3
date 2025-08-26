"""Input validation and sanitization for the Virtual PR Firm application.

This module provides comprehensive validation for all user inputs including:
- Topic content validation
- Platform name validation and normalization
- Brand bible content validation
- Input sanitization for security
- Rate limiting validation
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error with field and message."""
    field: str
    message: str
    code: str = "validation_error"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[ValidationError]
    sanitized_data: Optional[Dict[str, Any]] = None


class ValidationError(Exception):
    """Raised when validation fails."""
    
    def __init__(self, errors: List[ValidationError]):
        """
        Initialize the ValidationError exception that aggregates multiple field errors.
        
        Parameters:
            errors (List[ValidationError]): List of ValidationError dataclass instances to attach to this exception.
        
        The provided list is stored on the instance as `.errors`. The exception message is set to indicate the number of contained errors.
        """
        self.errors = errors
        super().__init__(f"Validation failed: {len(errors)} errors")


class InputValidator:
    """Comprehensive input validator for the Virtual PR Firm application."""
    
    # Supported platforms with aliases
    SUPPORTED_PLATFORMS = {
        'twitter': ['twitter', 'x', 'tweet'],
        'linkedin': ['linkedin', 'li'],
        'facebook': ['facebook', 'fb'],
        'instagram': ['instagram', 'ig'],
        'tiktok': ['tiktok', 'tt'],
        'youtube': ['youtube', 'yt'],
        'reddit': ['reddit', 'red'],
        'medium': ['medium', 'med'],
        'blog': ['blog', 'website', 'web'],
        'press_release': ['press_release', 'press', 'pr'],
        'email': ['email', 'mail'],
        'newsletter': ['newsletter', 'news']
    }
    
    # Content filtering patterns
    SPAM_PATTERNS = [
        r'\b(?:buy\s+now|click\s+here|limited\s+time|act\s+now|urgent|exclusive)\b',
        r'\b(?:free\s+offer|money\s+back|guaranteed|no\s+risk|100%\s+free)\b',
        r'\b(?:earn\s+\$\d+|make\s+\$\d+|profit\s+\$\d+)\b',
        r'\b(?:weight\s+loss|diet\s+pill|miracle\s+cure|anti\s+aging)\b',
        r'\b(?:casino|poker|bet|gambling|lottery)\b',
        r'\b(?:viagra|cialis|levitra|penis|enlargement)\b',
        r'\b(?:loan|credit|debt|mortgage|refinance)\b',
        r'\b(?:investment|trading|forex|bitcoin|crypto)\b',
        r'\b(?:dating|hookup|escort|adult|porn)\b',
        r'\b(?:pharmacy|medication|prescription|drug)\b'
    ]
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_TOPIC_LENGTH = 500
    MIN_TOPIC_LENGTH = 1
    
    def __init__(self):
        """
        Initialize the validator instance.
        
        Compiles the module-level SPAM_PATTERNS into case-insensitive regular expression objects
        and stores them on self.spam_patterns for reuse by spam-detection routines.
        """
        self.spam_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SPAM_PATTERNS]
    
    def validate_shared_store(self, shared: Dict[str, Any]) -> ValidationResult:
        """
        Validate the complete shared store structure and produce a sanitized copy when valid.
        
        Performs type check on `shared` (must be a dict), validates required fields under `task_requirements` (platforms, topic_or_goal, optional content_type and target_audience), validates optional `brand_bible` (xml content and file path), and validates the optional `stream` entry. Returns a ValidationResult where `is_valid` is True only if no validation errors were found; when valid `sanitized_data` contains a sanitized copy of the input (normalized platforms, cleaned text and XML). If `shared` is not a dict, returns a single `type_error` validation error.
        """
        errors = []
        
        # Basic structure validation
        if not isinstance(shared, dict):
            errors.append(ValidationError(
                field="shared",
                message="Shared store must be a dictionary",
                code="type_error"
            ))
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate task_requirements
        task_req_errors = self._validate_task_requirements(shared.get("task_requirements"))
        errors.extend(task_req_errors)
        
        # Validate brand_bible
        brand_bible_errors = self._validate_brand_bible(shared.get("brand_bible"))
        errors.extend(brand_bible_errors)
        
        # Validate stream (optional)
        if "stream" in shared and shared["stream"] is not None:
            stream_errors = self._validate_stream(shared["stream"])
            errors.extend(stream_errors)
        
        is_valid = len(errors) == 0
        sanitized_data = self._sanitize_shared_store(shared) if is_valid else None
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            sanitized_data=sanitized_data
        )
    
    def _validate_task_requirements(self, task_req: Any) -> List[ValidationError]:
        """
        Validate the `task_requirements` section of input and return any field-level validation errors.
        
        Expects `task_req` to be a dict containing at minimum:
        - "platforms": a non-empty list of platform identifiers (validated/normalized via _validate_platforms)
        - "topic_or_goal": a topic/goal string (validated via _validate_topic)
        
        Optionally validates:
        - "content_type" (validated via _validate_content_type)
        - "target_audience" (validated via _validate_target_audience)
        
        Returns a list of ValidationError objects describing each problem found. If `task_req` is None or not a dict, a single ValidationError describing the missing or wrong-typed top-level value is returned.
        """
        errors = []
        
        if task_req is None:
            errors.append(ValidationError(
                field="task_requirements",
                message="task_requirements is required",
                code="required_field"
            ))
            return errors
        
        if not isinstance(task_req, dict):
            errors.append(ValidationError(
                field="task_requirements",
                message="task_requirements must be a dictionary",
                code="type_error"
            ))
            return errors
        
        # Validate platforms
        platforms = task_req.get("platforms")
        if platforms is None:
            errors.append(ValidationError(
                field="task_requirements.platforms",
                message="platforms field is required",
                code="required_field"
            ))
        else:
            platform_errors = self._validate_platforms(platforms)
            errors.extend(platform_errors)
        
        # Validate topic_or_goal
        topic = task_req.get("topic_or_goal")
        if topic is None:
            errors.append(ValidationError(
                field="task_requirements.topic_or_goal",
                message="topic_or_goal field is required",
                code="required_field"
            ))
        else:
            topic_errors = self._validate_topic(topic)
            errors.extend(topic_errors)
        
        # Validate optional fields
        if "content_type" in task_req:
            content_type_errors = self._validate_content_type(task_req["content_type"])
            errors.extend(content_type_errors)
        
        if "target_audience" in task_req:
            audience_errors = self._validate_target_audience(task_req["target_audience"])
            errors.extend(audience_errors)
        
        return errors
    
    def _validate_platforms(self, platforms: Any) -> List[ValidationError]:
        """
        Validate the 'platforms' entry in task_requirements.
        
        Checks that `platforms` is a non-empty list of non-empty strings and that each entry maps to a supported canonical platform name (aliases are accepted). Does not modify the input; any unsupported, missing, or mistyped entries are reported.
        
        Returns:
            List[ValidationError]: A list of validation errors found. Empty list means the platforms value passed validation.
        """
        errors = []
        
        if not isinstance(platforms, list):
            errors.append(ValidationError(
                field="task_requirements.platforms",
                message="platforms must be a list",
                code="type_error"
            ))
            return errors
        
        if not platforms:
            errors.append(ValidationError(
                field="task_requirements.platforms",
                message="At least one platform must be specified",
                code="empty_list"
            ))
            return errors
        
        normalized_platforms = []
        for i, platform in enumerate(platforms):
            if not isinstance(platform, str):
                errors.append(ValidationError(
                    field=f"task_requirements.platforms[{i}]",
                    message="Platform must be a string",
                    code="type_error"
                ))
                continue
            
            platform = platform.strip().lower()
            if not platform:
                errors.append(ValidationError(
                    field=f"task_requirements.platforms[{i}]",
                    message="Platform cannot be empty",
                    code="empty_string"
                ))
                continue
            
            # Check if platform is supported
            supported = False
            for canonical, aliases in self.SUPPORTED_PLATFORMS.items():
                if platform in aliases:
                    normalized_platforms.append(canonical)
                    supported = True
                    break
            
            if not supported:
                errors.append(ValidationError(
                    field=f"task_requirements.platforms[{i}]",
                    message=f"Unsupported platform: {platform}. Supported platforms: {', '.join(self.SUPPORTED_PLATFORMS.keys())}",
                    code="unsupported_platform"
                ))
        
        return errors
    
    def _validate_topic(self, topic: Any) -> List[ValidationError]:
        """
        Validate the task_requirements.topic_or_goal value and return any validation issues.
        
        Performs these checks and returns a list of ValidationError instances for each violation:
        - Type: must be a string.
        - Whitespace trimming, then length: must be between MIN_TOPIC_LENGTH and MAX_TOPIC_LENGTH.
        - Content: rejects texts matching spam patterns.
        - Character composition: rejects topics with an excessive ratio of special characters (>30%).
        
        Errors use the field "task_requirements.topic_or_goal" and include codes such as "type_error", "too_short", "too_long", "spam_content", and "excessive_special_chars".
        """
        errors = []
        
        if not isinstance(topic, str):
            errors.append(ValidationError(
                field="task_requirements.topic_or_goal",
                message="Topic must be a string",
                code="type_error"
            ))
            return errors
        
        topic = topic.strip()
        
        # Length validation
        if len(topic) < self.MIN_TOPIC_LENGTH:
            errors.append(ValidationError(
                field="task_requirements.topic_or_goal",
                message=f"Topic must be at least {self.MIN_TOPIC_LENGTH} character(s) long",
                code="too_short"
            ))
        
        if len(topic) > self.MAX_TOPIC_LENGTH:
            errors.append(ValidationError(
                field="task_requirements.topic_or_goal",
                message=f"Topic must be no more than {self.MAX_TOPIC_LENGTH} characters long",
                code="too_long"
            ))
        
        # Content filtering
        if self._contains_spam_content(topic):
            errors.append(ValidationError(
                field="task_requirements.topic_or_goal",
                message="Topic contains inappropriate or spam-like content",
                code="spam_content"
            ))
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', topic)) / len(topic) if topic else 0
        if special_char_ratio > 0.3:
            errors.append(ValidationError(
                field="task_requirements.topic_or_goal",
                message="Topic contains too many special characters",
                code="excessive_special_chars"
            ))
        
        return errors
    
    def _validate_brand_bible(self, brand_bible: Any) -> List[ValidationError]:
        """
        Validate the optional brand bible structure and its contents.
        
        Accepts None or a dict. If provided, the dict may include:
        - "xml_raw": raw XML content (validated for type, size, and safety via _validate_xml_content).
        - "file_path": path to a brand bible file (validated for existence, size, extension via _validate_file_path).
        
        Returns a list of ValidationError objects describing any problems found; an empty list means the brand bible is valid or was not provided.
        """
        errors = []
        
        if brand_bible is None:
            # Brand bible is optional
            return errors
        
        if not isinstance(brand_bible, dict):
            errors.append(ValidationError(
                field="brand_bible",
                message="Brand bible must be a dictionary",
                code="type_error"
            ))
            return errors
        
        # Validate xml_raw field
        xml_raw = brand_bible.get("xml_raw")
        if xml_raw is not None:
            xml_errors = self._validate_xml_content(xml_raw)
            errors.extend(xml_errors)
        
        # Validate file_path field if present
        file_path = brand_bible.get("file_path")
        if file_path is not None:
            file_errors = self._validate_file_path(file_path)
            errors.extend(file_errors)
        
        return errors
    
    def _validate_xml_content(self, xml_raw: Any) -> List[ValidationError]:
        """
        Validate a raw brand-bible XML string and return any validation errors.
        
        Performs the following checks and returns a list of ValidationError for each failing rule:
        - Type: value must be a string.
        - Size: must not exceed MAX_FILE_SIZE bytes.
        - Basic structure: non-empty content must start with '<'.
        - Safety: rejects content that matches dangerous XML constructs (checked via _contains_dangerous_xml).
        
        Errors use the field name "brand_bible.xml_raw" and may include codes such as "type_error", "file_too_large", "invalid_xml", and "dangerous_xml".
        
        Parameters:
            xml_raw (Any): Raw XML content to validate (expected to be a string).
        
        Returns:
            List[ValidationError]: A list of validation errors; empty if the content is valid.
        """
        errors = []
        
        if not isinstance(xml_raw, str):
            errors.append(ValidationError(
                field="brand_bible.xml_raw",
                message="XML content must be a string",
                code="type_error"
            ))
            return errors
        
        # Check content length
        if len(xml_raw) > self.MAX_FILE_SIZE:
            errors.append(ValidationError(
                field="brand_bible.xml_raw",
                message=f"XML content exceeds maximum size of {self.MAX_FILE_SIZE} bytes",
                code="file_too_large"
            ))
        
        # Basic XML structure validation
        if xml_raw.strip() and not xml_raw.strip().startswith('<'):
            errors.append(ValidationError(
                field="brand_bible.xml_raw",
                message="XML content must start with '<' character",
                code="invalid_xml"
            ))
        
        # Check for potential XML injection
        if self._contains_dangerous_xml(xml_raw):
            errors.append(ValidationError(
                field="brand_bible.xml_raw",
                message="XML content contains potentially dangerous elements",
                code="dangerous_xml"
            ))
        
        return errors
    
    def _validate_file_path(self, file_path: Any) -> List[ValidationError]:
        """
        Validate a filesystem path intended for a brand bible file and return any validation issues.
        
        Checks performed:
        - Ensures `file_path` is a string representing a filesystem path.
        - Verifies the file exists and is accessible.
        - Validates the file size does not exceed InputValidator.MAX_FILE_SIZE.
        - Validates the file extension is one of the allowed types: .xml, .json, .txt, .yaml, .yml.
        
        Parameters:
            file_path: Expected to be a filesystem path provided as a string.
        
        Returns:
            List[ValidationError]: A list of ValidationError instances describing any problems found.
                Possible error codes included in the list:
                - "type_error": `file_path` was not a string.
                - "file_not_found": the path does not point to an existing file.
                - "file_access_error": filesystem access to the path failed (OS error).
                - "file_too_large": file size exceeds MAX_FILE_SIZE (size included in the message).
                - "invalid_file_extension": file extension is not one of the allowed set.
        """
        errors = []
        
        if not isinstance(file_path, str):
            errors.append(ValidationError(
                field="brand_bible.file_path",
                message="File path must be a string",
                code="type_error"
            ))
            return errors
        
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            errors.append(ValidationError(
                field="brand_bible.file_path",
                message=f"File does not exist: {file_path}",
                code="file_not_found"
            ))
            return errors
        
        # Check file size
        try:
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                errors.append(ValidationError(
                    field="brand_bible.file_path",
                    message=f"File size ({file_size} bytes) exceeds maximum allowed size ({self.MAX_FILE_SIZE} bytes)",
                    code="file_too_large"
                ))
        except OSError:
            errors.append(ValidationError(
                field="brand_bible.file_path",
                message=f"Cannot access file: {file_path}",
                code="file_access_error"
            ))
        
        # Check file extension
        allowed_extensions = {'.xml', '.json', '.txt', '.yaml', '.yml'}
        if path.suffix.lower() not in allowed_extensions:
            errors.append(ValidationError(
                field="brand_bible.file_path",
                message=f"File extension '{path.suffix}' not allowed. Allowed extensions: {', '.join(allowed_extensions)}",
                code="invalid_file_extension"
            ))
        
        return errors
    
    def _validate_stream(self, stream: Any) -> List[ValidationError]:
        """
        Validate a stream configuration value.
        
        Accepts either a boolean or a dict. If the value is invalid, returns a list of ValidationError entries describing the problems.
        
        Validation rules:
        - If not a bool or dict: returns a single ValidationError with field "stream" and code "type_error".
        - If a dict and contains an "enabled" key: "enabled" must be a boolean; otherwise a ValidationError with field "stream.enabled" and code "type_error" is returned.
        
        Returns:
            List[ValidationError]: Empty list when valid, otherwise one or more ValidationError instances.
        """
        errors = []
        
        if not isinstance(stream, (bool, dict)):
            errors.append(ValidationError(
                field="stream",
                message="Stream must be a boolean or dictionary",
                code="type_error"
            ))
            return errors
        
        if isinstance(stream, dict):
            # Validate stream configuration options
            if "enabled" in stream and not isinstance(stream["enabled"], bool):
                errors.append(ValidationError(
                    field="stream.enabled",
                    message="Stream enabled must be a boolean",
                    code="type_error"
                ))
        
        return errors
    
    def _validate_content_type(self, content_type: Any) -> List[ValidationError]:
        """
        Validate the task requirement content type and return any validation errors.
        
        Checks that `content_type` is a string and (case-insensitively) one of the allowed types:
        press_release, social_post, blog_post, email, newsletter, announcement, update, story.
        
        Returns a list of ValidationError objects describing problems; returns an empty list when valid.
        """
        errors = []
        
        if not isinstance(content_type, str):
            errors.append(ValidationError(
                field="task_requirements.content_type",
                message="Content type must be a string",
                code="type_error"
            ))
            return errors
        
        valid_types = {
            'press_release', 'social_post', 'blog_post', 'email', 
            'newsletter', 'announcement', 'update', 'story'
        }
        
        if content_type.lower() not in valid_types:
            errors.append(ValidationError(
                field="task_requirements.content_type",
                message=f"Invalid content type: {content_type}. Valid types: {', '.join(valid_types)}",
                code="invalid_content_type"
            ))
        
        return errors
    
    def _validate_target_audience(self, audience: Any) -> List[ValidationError]:
        """
        Validate the `target_audience` value in task requirements.
        
        Checks that `audience` is a string and enforces length constraints (minimum 3 trimmed characters, maximum 200 characters). Returns a list of ValidationError instances describing any problems; if the list is empty, the value passed validation.
        
        Parameters:
            audience: The value to validate (expected to be a string).
        
        Returns:
            List[ValidationError]: One or more validation errors; empty list when valid.
        """
        errors = []
        
        if not isinstance(audience, str):
            errors.append(ValidationError(
                field="task_requirements.target_audience",
                message="Target audience must be a string",
                code="type_error"
            ))
            return errors
        
        if len(audience.strip()) < 3:
            errors.append(ValidationError(
                field="task_requirements.target_audience",
                message="Target audience must be at least 3 characters long",
                code="too_short"
            ))
        
        if len(audience) > 200:
            errors.append(ValidationError(
                field="task_requirements.target_audience",
                message="Target audience must be no more than 200 characters long",
                code="too_long"
            ))
        
        return errors
    
    def _contains_spam_content(self, text: str) -> bool:
        """
        Return True if the given text matches any configured spam pattern.
        
        Performs a case-insensitive check against this validator's compiled spam patterns and returns True on the first match; otherwise returns False.
        """
        text_lower = text.lower()
        for pattern in self.spam_patterns:
            if pattern.search(text_lower):
                return True
        return False
    
    def _contains_dangerous_xml(self, xml_content: str) -> bool:
        """
        Return True if the provided XML-like content contains constructs considered potentially dangerous.
        
        Performs case-insensitive regex checks for common risky patterns including CDATA sections, DOCTYPE/ENTITY declarations, embedded script/iframe/object/embed tags, and dangerous URI schemes (e.g., `javascript:`, `vbscript:`, `data:`). Returns False when none of these patterns are found.
        """
        dangerous_patterns = [
            r'<!\[CDATA\[.*?\]\]>',  # CDATA sections
            r'<!DOCTYPE.*?>',        # DOCTYPE declarations
            r'<!ENTITY.*?>',         # Entity declarations
            r'<script.*?>',          # Script tags
            r'<iframe.*?>',          # Iframe tags
            r'<object.*?>',          # Object tags
            r'<embed.*?>',           # Embed tags
            r'javascript:',          # JavaScript protocol
            r'vbscript:',            # VBScript protocol
            r'data:',                # Data protocol
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, xml_content, re.IGNORECASE):
                return True
        return False
    
    def _sanitize_shared_store(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a sanitized shallow copy of the shared store suitable for downstream use.
        
        Creates and returns a copy of `shared` with targeted sanitization applied:
        - task_requirements.platforms: normalized to canonical platform names.
        - task_requirements.topic_or_goal: whitespace-collapsed, control characters removed, and truncated to the configured maximum length.
        - brand_bible.xml_raw: dangerous XML constructs removed.
        
        The original `shared` object is not modified; unspecified keys are preserved as-is in the returned dictionary.
        """
        sanitized = shared.copy()
        
        # Sanitize task_requirements
        if "task_requirements" in sanitized:
            task_req = sanitized["task_requirements"].copy()
            
            # Normalize platforms
            if "platforms" in task_req:
                task_req["platforms"] = self._normalize_platforms(task_req["platforms"])
            
            # Sanitize topic
            if "topic_or_goal" in task_req:
                task_req["topic_or_goal"] = self._sanitize_text(task_req["topic_or_goal"])
            
            sanitized["task_requirements"] = task_req
        
        # Sanitize brand_bible
        if "brand_bible" in sanitized and sanitized["brand_bible"]:
            brand_bible = sanitized["brand_bible"].copy()
            if "xml_raw" in brand_bible:
                brand_bible["xml_raw"] = self._sanitize_xml(brand_bible["xml_raw"])
            sanitized["brand_bible"] = brand_bible
        
        return sanitized
    
    def _normalize_platforms(self, platforms: List[str]) -> List[str]:
        """
        Normalize a list of platform identifiers to the module's canonical platform names.
        
        Leading/trailing whitespace is stripped and comparison is case-insensitive. Any input entries that do not match a known alias are omitted. The returned list preserves the order of matched inputs and may contain duplicates if the input contains duplicates or multiple aliases that map to the same canonical name.
        
        Parameters:
            platforms (List[str]): Platform identifiers (names or aliases) to normalize.
        
        Returns:
            List[str]: List of canonical platform names corresponding to recognized inputs.
        """
        normalized = []
        for platform in platforms:
            platform_lower = platform.strip().lower()
            for canonical, aliases in self.SUPPORTED_PLATFORMS.items():
                if platform_lower in aliases:
                    normalized.append(canonical)
                    break
        return normalized
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize and normalize an input text string.
        
        Converts non-string inputs to str, collapses repeated internal whitespace to single spaces and trims ends, removes control characters except for newline, carriage return, and tab, and truncates the result to MAX_TOPIC_LENGTH.
         
        Parameters:
            text (str | any): The input to sanitize; non-strings will be converted to str.
        
        Returns:
            str: The sanitized, normalized text suitable for downstream use.
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        if len(text) > self.MAX_TOPIC_LENGTH:
            text = text[:self.MAX_TOPIC_LENGTH]
        
        return text
    
    def _sanitize_xml(self, xml_content: str) -> str:
        """
        Sanitize XML-like content by removing dangerous constructs and risky protocols.
        
        If the input is not a string, it is converted to a string and returned. The sanitizer removes CDATA sections, DOCTYPE and ENTITY declarations, script/iframe/object/embed blocks, and strips risky protocols such as "javascript:", "vbscript:", and "data:". Returns the cleaned XML string suitable for safer parsing or storage.
        """
        if not isinstance(xml_content, str):
            return str(xml_content)
        
        # Remove dangerous patterns
        dangerous_patterns = [
            (r'<!\[CDATA\[.*?\]\]>', ''),  # Remove CDATA sections
            (r'<!DOCTYPE.*?>', ''),        # Remove DOCTYPE declarations
            (r'<!ENTITY.*?>', ''),         # Remove entity declarations
            (r'<script.*?>.*?</script>', '', re.DOTALL),  # Remove script tags
            (r'<iframe.*?>.*?</iframe>', '', re.DOTALL),  # Remove iframe tags
            (r'<object.*?>.*?</object>', '', re.DOTALL),  # Remove object tags
            (r'<embed.*?>', ''),           # Remove embed tags
            (r'javascript:', ''),          # Remove JavaScript protocol
            (r'vbscript:', ''),            # Remove VBScript protocol
            (r'data:', ''),                # Remove data protocol
        ]
        
        sanitized = xml_content
        for pattern, replacement, *flags in dangerous_patterns:
            if flags:
                sanitized = re.sub(pattern, replacement, sanitized, flags=flags[0])
            else:
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized


class RateLimitValidator:
    """Rate limiting validation for request throttling."""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize the rate limit validator.
        
        Parameters:
            max_requests (int): Maximum number of allowed requests per client within the time window. Defaults to 60.
            window_seconds (int): Time window duration in seconds for rate limiting. Defaults to 60.
        
        Notes:
            Uses an in-memory `request_history` dict mapping client IDs to lists of request timestamps; this is suitable for single-process usage and testing only — use a centralized store (e.g., Redis) in production.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_history = {}  # In production, use Redis or similar
    
    def validate_rate_limit(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check whether a client is within the configured rate limit and record the request when allowed.
        
        This method:
        - Cleans up expired entries from the internal request history.
        - Checks the number of requests the given client has made within the current rolling window.
        - If the client has reached the limit, returns a rate-limited response (no new entry is recorded).
        - If the client is under the limit, appends the current timestamp to the client's history and returns remaining allowance.
        
        Parameters:
            client_id (str): Identifier for the client making the request.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple where the first element is True if the request is allowed, False if rate-limited.
            The second element is a dict with:
                - rate_limited (bool): True when the client has exceeded the allowed requests.
                - time_until_reset (float): Seconds until the oldest recorded request falls out of the current window (0 when not provided or already expired). Present when rate_limited is True.
                - requests_remaining (int): Number of additional requests the client may make in the current window (0 when rate_limited is True).
                - limit (int): Configured maximum requests per window.
                - window (int): Window duration in seconds.
        
        Side effects:
            - Calls _cleanup_old_entries(current_time) to prune old timestamps.
            - When allowed, appends the current epoch timestamp (time.time()) to self.request_history[client_id].
        """
        import time
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Get client history
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        client_requests = self.request_history[client_id]
        
        # Check if within rate limit
        if len(client_requests) >= self.max_requests:
            oldest_request = min(client_requests)
            time_until_reset = self.window_seconds - (current_time - oldest_request)
            
            return False, {
                "rate_limited": True,
                "time_until_reset": max(0, time_until_reset),
                "requests_remaining": 0,
                "limit": self.max_requests,
                "window": self.window_seconds
            }
        
        # Add current request
        client_requests.append(current_time)
        
        return True, {
            "rate_limited": False,
            "requests_remaining": self.max_requests - len(client_requests),
            "limit": self.max_requests,
            "window": self.window_seconds
        }
    
    def _cleanup_old_entries(self, current_time: float) -> None:
        """
        Remove timestamps outside the current rate-limit window from the in-memory history.
        
        This mutates self.request_history in-place: for each client, timestamps older than
        (current_time - self.window_seconds) are discarded. Any client with no remaining
        timestamps is removed from self.request_history.
        
        Parameters:
            current_time (float): Current time in seconds since the epoch used to compute the cutoff.
        """
        cutoff_time = current_time - self.window_seconds
        
        for client_id in list(self.request_history.keys()):
            self.request_history[client_id] = [
                req_time for req_time in self.request_history[client_id]
                if req_time > cutoff_time
            ]
            
            # Remove empty client entries
            if not self.request_history[client_id]:
                del self.request_history[client_id]


# Global validator instance
validator = InputValidator()
rate_limit_validator = RateLimitValidator()


def validate_shared_store(shared: Dict[str, Any]) -> None:
    """
    Validate and sanitize the provided shared store in-place.
    
    Validates the structure and fields of the given `shared` dictionary using the module's InputValidator.
    If validation fails, raises ValidationError containing a list of field-specific errors.
    If validation succeeds and sanitized data is produced, replaces the contents of `shared` with the sanitized copy.
    
    Parameters:
        shared (dict): Mutable mapping representing the shared store to validate and sanitize.
    
    Raises:
        ValidationError: Aggregates one or more validation issues when the input is invalid.
    """
    result = validator.validate_shared_store(shared)
    
    if not result.is_valid:
        raise ValidationError(result.errors)
    
    # Update shared store with sanitized data
    if result.sanitized_data:
        shared.clear()
        shared.update(result.sanitized_data)


def validate_rate_limit(client_id: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Check whether the given client is within the configured rate limit window.
    
    Parameters:
        client_id (str): Identifier for the client (e.g., API key, user id) to check.
    
    Returns:
        Tuple[bool, dict]: A tuple where the first element is True if the client is allowed to proceed (within limit) or False if rate-limited.
            The second element is a details dictionary:
            - When allowed: contains at least "remaining" (int) — number of requests left in the current window.
            - When rate-limited: contains at least "time_until_reset" (float) — seconds until the rate window resets.
    """
    return rate_limit_validator.validate_rate_limit(client_id)


def sanitize_text(text: str) -> str:
    """
    Sanitize free-form text for safe downstream use.
    
    Collapses consecutive whitespace, removes control/control-like characters while preserving common whitespace (spaces, tabs, newlines), and truncates the result to the configured maximum topic length. Intended for user-provided topics/goals and other short text fields.
    
    Parameters:
        text (str): Input text to sanitize.
    
    Returns:
        str: Sanitized text.
    """
    return validator._sanitize_text(text)


def normalize_platforms(platforms: List[str]) -> List[str]:
    """Normalize platform names to canonical form."""
    return validator._normalize_platforms(platforms)