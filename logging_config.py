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
        Sanitize sensitive information in a logging.LogRecord in-place and always allow the record.
        
        This filter inspects the record's message and formatting arguments and replaces or redacts sensitive values:
        - If record.msg is a string, it is sanitized via _sanitize_string.
        - If record.args is a dict, it is sanitized via _sanitize_dict.
        - If record.args is a list or tuple, each element is sanitized (tuples are returned).
        The method mutates the provided record and always returns True so it never prevents logging.
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
        Redact common sensitive data patterns from a text string.
        
        This method returns a sanitized copy of `text` where likely secrets are replaced with placeholders.
        Patterns redacted include:
        - long alphanumeric sequences (likely API keys),
        - email addresses,
        - HTTP/HTTPS URLs.
        
        Parameters:
            text (str): Input string to sanitize.
        
        Returns:
            str: Sanitized string with sensitive values replaced by placeholders like `[API_KEY]`, `[EMAIL]`, and `[URL]`.
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
        Sanitize a mapping by redacting sensitive values and recursively sanitizing non-sensitive values.
        
        For each key/value pair in `data`:
        - If the key is a string and contains any substring listed in `self.SENSITIVE_KEYS` (case-insensitive),
          the value is replaced with the literal '[REDACTED]'.
        - Otherwise the value is processed with `self._sanitize_value` (recursively sanitizes dicts, lists, tuples, and strings).
        
        Parameters:
            data (Dict[str, Any]): Input dictionary to sanitize. Keys are preserved; values may be redacted or transformed.
        
        Returns:
            Dict[str, Any]: A new dictionary with the same keys where sensitive values are replaced with '[REDACTED]' and other values sanitized.
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
        Recursively sanitize a value by redacting sensitive information.
        
        Handles dicts (by delegating to _sanitize_dict), lists and tuples (recursively sanitizes each element, preserving the original sequence type), and strings (by delegating to _sanitize_string). Other value types are returned unchanged.
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
        Attach a request correlation ID and an ISO-8601 UTC timestamp to the given log record.
        
        This filter sets record.request_id using get_request_id() (generating one if necessary)
        and sets record.timestamp to the current UTC time in ISO 8601 format. Always returns True
        so the record continues through the logging pipeline.
        """
        record.request_id = get_request_id()
        record.timestamp = datetime.utcnow().isoformat()
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
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
        Format a LogRecord into a human-readable string.
        
        The output includes a timestamp (from record.timestamp if present, otherwise current UTC time),
        the log level, logger name, request correlation id (record.request_id or "N/A"), and the formatted message.
        For DEBUG level and lower the module name and line number are appended. If exception information is present
        (record.exc_info) the formatted traceback is appended on a new line.
        
        Returns:
            str: The formatted log string.
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
    """
    Return the current request correlation ID, generating and storing a new UUID4 if none exists.
    
    Returns:
        str: The request correlation ID for use in log records and request tracing.
    """
    global _request_id
    if _request_id is None:
        _request_id = str(uuid.uuid4())
    return _request_id


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set the module-level request correlation ID.
    
    If `request_id` is provided, it becomes the current correlation ID; otherwise a new UUID4 string is generated, stored, and returned.
    
    Parameters:
        request_id (Optional[str]): Specific request ID to set. If omitted or None, a new UUID4 string is created.
    
    Returns:
        str: The request ID that was set.
    """
    global _request_id
    if request_id is None:
        request_id = str(uuid.uuid4())
    _request_id = request_id
    return request_id


def clear_request_id():
    """Clear current request ID."""
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
    Configure the application's root logger with console (and optional rotating file) handlers, formatters, and filters.
    
    This sets the root logger level, removes existing handlers, and attaches:
    - a console StreamHandler (stdout) and, if log_file is provided, a RotatingFileHandler.
    - a SensitiveDataFilter to redact sensitive values from messages/arguments.
    - a RequestCorrelationFilter to inject a per-request correlation ID and timestamp.
    - either StructuredFormatter (JSON) when format_type == "json" or HumanReadableFormatter when "human".
    
    Parameters:
        level: Logging level name; one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        format_type: Output format, either "json" for structured JSON logs or "human" for readable text.
        log_file: Optional filesystem path to write logs. If provided, its parent directory will be created and a RotatingFileHandler will be used.
        max_size: Maximum size in bytes for a single log file before rotation (applies only when log_file is set).
        backup_count: Number of rotated backup files to keep (applies only when log_file is set).
    
    Raises:
        ValueError: If `level` is not a supported level name or if `format_type` is not "json" or "human".
    
    Side effects:
        - Modifies the root logger and may change logging behaviour across the process.
        - May create directories on disk when a log_file is specified.
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
    Decorator that logs a function's entry, successful completion, and failures.
    
    On call, logs a DEBUG entry with the function name, positional-argument count,
    keyword-argument keys, and the current request correlation ID. On successful
    return, logs a DEBUG message indicating completion (including request ID).
    If the wrapped function raises an exception, logs an ERROR with the error
    message, type, request ID, and traceback, then re-raises the exception.
    
    Returns the wrapped function's result unchanged.
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper for a decorated function that logs function entry, successful exit, and errors.
        
        This wrapper records a debug log when the function is called (including argument count and keyword names), a debug log on successful completion, and an error log with traceback and error metadata if the function raises. It attaches the current request correlation ID to all logs and re-raises any exception from the wrapped function.
        
        Returns:
            The return value of the wrapped function.
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
    Log an exception with structured contextual information.
    
    Logs the provided exception at ERROR level, including its type, message, the current request correlation ID, and any additional key/value pairs supplied via `context`. The traceback is logged (exc_info=True) so the exception stack is captured for diagnostics.
    
    Parameters:
        error (Exception): The exception instance to log.
        context (dict, optional): Additional context to include in the log record; keys are merged into the structured `extra` fields.
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