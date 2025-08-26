"""Comprehensive logging configuration for the Virtual PR Firm application.

This module provides structured logging with support for different environments,
request correlation, and security-aware logging that filters sensitive data.

Features:
- Structured JSON logging for production
- Human-readable logging for development
- Request correlation IDs for tracing
- Sensitive data filtering
- Log rotation and file management
- Environment-based configuration

Example Usage:
    >>> from logging_config import setup_logging
    >>> setup_logging(level="INFO", format="json")
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Application started", extra={"user_id": "123"})
"""

import logging
import logging.config
import json
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import os

# Global request ID for correlation
_request_id = None


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from log messages."""
    
    SENSITIVE_KEYS = {
        'api_key', 'password', 'token', 'secret', 'key', 'auth',
        'authorization', 'cookie', 'session', 'credential'
    }
    
    def filter(self, record):
        """
        Sanitize sensitive information in a LogRecord in-place and always allow the record to be emitted.
        
        If the record has a string `msg`, it is sanitized. If the record has `args` that are a dict, list, or tuple,
        their values are recursively sanitized and replaced on the record. This method mutates `record.msg` and
        `record.args` and returns True so the logging framework continues processing the record.
        """
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._sanitize_string(record.msg)
        
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = self._sanitize_dict(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(self._sanitize_value(arg) for arg in record.args)
        
        return True
    
    def _sanitize_string(self, text: str) -> str:
        """
        Sanitize sensitive data in a text string by redacting common secrets.
        
        Replaces:
        - long alphanumeric sequences (32+ chars) with `[API_KEY]`
        - email addresses with `[EMAIL]`
        - HTTP/HTTPS URLs with `[URL]`
        
        Parameters:
            text (str): Input string to sanitize.
        
        Returns:
            str: A copy of `text` with sensitive patterns redacted.
        """
        # Simple pattern matching for common sensitive data patterns
        import re
        
        # Replace API keys (common patterns)
        text = re.sub(r'[a-zA-Z0-9]{32,}', '[API_KEY]', text)
        
        # Replace email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Replace URLs with sensitive data
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        
        return text
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a new dictionary with sensitive values redacted.
        
        Keys containing any term from SENSITIVE_KEYS (case-insensitive) have their values replaced with the literal "[REDACTED]". All other values are recursively sanitized via _sanitize_value. The input dictionary is not mutated; a new dictionary is returned.
        """
        sanitized = {}
        for key, value in data.items():
            if isinstance(key, str) and any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = self._sanitize_value(value)
        return sanitized
    
    def _sanitize_value(self, value: Any) -> Any:
        """
        Sanitize a value of any type by redacting or masking sensitive information.
        
        Recursively processes the input and:
        - If a dict: returns a sanitized dict with sensitive keys redacted.
        - If a list or tuple: returns a new sequence of the same type with each element sanitized.
        - If a string: returns the string with identified sensitive patterns (e.g., API keys, emails, URLs) masked.
        - Otherwise: returns the value unchanged.
        
        Returns:
            The sanitized value with the same type as the input where applicable.
        """
        if isinstance(value, dict):
            return self._sanitize_dict(value)
        elif isinstance(value, (list, tuple)):
            return type(value)(self._sanitize_value(item) for item in value)
        elif isinstance(value, str):
            return self._sanitize_string(value)
        return value


class RequestCorrelationFilter(logging.Filter):
    """Add request correlation ID to log records."""
    
    def filter(self, record):
        """
        Attach a request correlation ID and UTC ISO8601 timestamp to a LogRecord.
        
        This filter mutates the provided log `record` by setting `record.request_id`
        to the current per-request ID (via get_request_id()) and `record.timestamp`
        to the current UTC time in ISO 8601 format. Always returns True so the
        record continues through the logging pipeline.
        
        Parameters:
            record: logging.LogRecord
                The log record to annotate; fields `request_id` and `timestamp` will
                be added or overwritten.
        
        Returns:
            bool: Always True.
        """
        record.request_id = get_request_id()
        record.timestamp = datetime.utcnow().isoformat()
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """
        Format a logging.LogRecord into a JSON string suitable for structured logging.
        
        The resulting JSON contains standard fields: `timestamp` (uses record.timestamp if present, otherwise current UTC ISO timestamp), `level`, `logger`, `message`, `request_id`, `module`, `function`, and `line`. If the record contains exception information, an `exception` field is included (formatted via the formatter's `formatException`). Any additional non-standard attributes on the record are copied into the JSON object.
        
        Parameters:
            record (logging.LogRecord): The log record to format.
        
        Returns:
            str: A JSON-formatted string representation of the log entry. Non-serializable values are converted to strings (uses `str()`).
        """
        log_entry = {
            'timestamp': getattr(record, 'timestamp', datetime.utcnow().isoformat()),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'request_id': getattr(record, 'request_id', None),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'timestamp', 'request_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    def format(self, record):
        """
        Format a LogRecord into a concise, human-readable string.
        
        Builds a single-line message that includes timestamp, level, logger name, request correlation id, and the formatted log message.
        - If the record provides a `timestamp` or `request_id`, those values are used; otherwise a UTC timestamp and 'N/A' are used for `timestamp` and `request_id` respectively.
        - For DEBUG (and more verbose) levels the module name and source line number are appended.
        - If exception information is present on the record, the formatted exception is appended on a new line.
        
        Parameters:
            record (logging.LogRecord): The log record to format.
        
        Returns:
            str: The formatted log message.
        """
        timestamp = getattr(record, 'timestamp', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        request_id = getattr(record, 'request_id', 'N/A')
        
        # Base format
        format_str = f'[{timestamp}] [{record.levelname:8}] [{record.name}] [{request_id}] {record.getMessage()}'
        
        # Add module info for debug level
        if record.levelno <= logging.DEBUG:
            format_str += f' ({record.module}:{record.lineno})'
        
        # Add exception info if present
        if record.exc_info:
            format_str += f'\n{self.formatException(record.exc_info)}'
        
        return format_str


def get_request_id() -> str:
    """Get current request ID or generate new one."""
    global _request_id
    if _request_id is None:
        _request_id = str(uuid.uuid4())
    return _request_id


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set the module-level request correlation ID.
    
    If `request_id` is None, a new UUID4 string is generated, stored in the module-level `_request_id`, and returned. Otherwise the provided `request_id` is stored and returned.
    
    Parameters:
        request_id (Optional[str]): The request correlation ID to set. If omitted, a new UUID4 is created.
    
    Returns:
        str: The request ID that was set.
    """
    global _request_id
    if request_id is None:
        request_id = str(uuid.uuid4())
    _request_id = request_id
    return request_id


def clear_request_id():
    """
    Clear the module-level request correlation ID.
    
    Resets the global request ID (used for request correlation) to None so that
    subsequent calls to get_request_id() will generate and store a new UUID.
    """
    global _request_id
    _request_id = None


def setup_logging(
    level: str = "INFO",
    format_type: str = "human",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure and install application-wide logging handlers, formatters, and filters.
    
    Sets up console (stdout) logging and an optional rotating file handler, choosing either a structured JSON formatter ("json") or a human-readable formatter ("human"). Applies SensitiveDataFilter (redacts/masks sensitive values) and RequestCorrelationFilter (attaches a request_id and timestamp) to all handlers, and sets sane log levels for noisy libraries (urllib3, requests, gradio). If a file path is provided, the containing directory is created and a RotatingFileHandler is added with the given max_size and backup_count.
    
    Parameters:
        level: Log level name (e.g., "DEBUG", "INFO"). Controls logger and handler levels.
        format_type: "json" to use StructuredFormatter (JSON output) or "human" for HumanReadableFormatter.
        log_file: Optional path to a log file; when provided, enables rotation and persistent logs.
        max_size: Maximum size in bytes for the rotating log file before rotation (default 10MB).
        backup_count: Number of rotated backup files to keep.
    
    Raises:
        ValueError: If `level` is not one of {"DEBUG","INFO","WARNING","ERROR","CRITICAL"} or if `format_type` is not "json" or "human".
    """
    # Validate inputs
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
    
    if format_type not in ["json", "human"]:
        raise ValueError(f"Invalid format type: {format_type}. Must be 'json' or 'human'")
    
    # Create formatter
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()
    
    # Create filters
    sensitive_filter = SensitiveDataFilter()
    correlation_filter = RequestCorrelationFilter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)
    console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('gradio').setLevel(logging.INFO)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "level": level,
            "format": format_type,
            "log_file": log_file,
            "max_size": max_size,
            "backup_count": backup_count
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator that logs entry, successful completion, and exceptions for the wrapped function.
    
    The decorator logs a DEBUG message when the function is called (includes function name, count of positional args, keys of keyword args, and current request_id), a DEBUG message on successful return (function name and request_id), and an ERROR message if the function raises (function name, error message, error type, request_id, and exception info). Exceptions are re-raised after being logged. The decorator preserves the wrapped function's signature and return value.
    """
    def wrapper(*args, **kwargs):
        """
        Decorator wrapper that invokes the wrapped function while logging its entry, successful completion, and any exception.
        
        On entry, logs a debug record with the wrapped function's name, number of positional args, names of keyword args, and the current request_id. On successful return, logs a debug record noting completion and the request_id. If the wrapped function raises an exception, logs an error with the exception message, its type, the request_id, and includes exception info, then re-raises the exception.
        
        Returns the wrapped function's return value unchanged and preserves exception propagation.
        """
        logger = logging.getLogger(func.__module__)
        request_id = get_request_id()
        
        # Log function entry
        logger.debug(
            f"Function {func.__name__} called",
            extra={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                "request_id": request_id
            }
        )
        
        try:
            result = func(*args, **kwargs)
            logger.debug(
                f"Function {func.__name__} completed successfully",
                extra={
                    "function": func.__name__,
                    "request_id": request_id
                }
            )
            return result
        except Exception as e:
            logger.error(
                f"Function {func.__name__} failed: {str(e)}",
                extra={
                    "function": func.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request_id": request_id
                },
                exc_info=True
            )
            raise
    
    return wrapper


def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """
    Log an exception with optional contextual data, attaching request correlation and structured extra fields.
    
    Logs an error-level entry containing the exception type and message, includes exception traceback (exc_info=True), and adds a `request_id` to the log record. If `context` is provided, its key/value pairs are merged into the structured `extra` fields that accompany the log record.
    
    Parameters:
        error (Exception): The exception instance to log.
        context (dict, optional): Additional context to include in the log's `extra` fields (merged into the structured payload).
    """
    logger = logging.getLogger(__name__)
    extra = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "request_id": get_request_id()
    }
    
    if context:
        extra.update(context)
    
    logger.error(
        f"Error occurred: {str(error)}",
        extra=extra,
        exc_info=True
    )


if __name__ == "__main__":
    # Test logging configuration
    setup_logging(level="DEBUG", format_type="human")
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test with sensitive data
    logger.info("API call with key", extra={"api_key": "secret123", "user": "test"})
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error_with_context(e, {"test_context": "value"})