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
        Redact sensitive information from a logging.LogRecord in-place.
        
        Inspects record.msg (if a string) and record.args (if present) and replaces sensitive values by calling the filter's sanitization helpers. Mutates record.msg and record.args to sanitized versions so downstream formatters/handlers receive redacted data.
        
        Parameters:
            record (logging.LogRecord): The log record to sanitize; modified in place.
        
        Returns:
            bool: Always returns True to allow the record to be processed by other filters and handlers.
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
        Sanitize sensitive data found in a text string by replacing common secret-like patterns with safe placeholders.
        
        Replaces:
        - long alphanumeric sequences (32+ chars) with `[API_KEY]` (covers many API keys/tokens),
        - email addresses with `[EMAIL]`,
        - HTTP/HTTPS URLs with `[URL]`.
        
        Parameters:
            text: The input string to sanitize.
        
        Returns:
            A copy of `text` with the above patterns redacted.
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
        Return a copy of `data` with sensitive values redacted and other values recursively sanitized.
        
        This method performs a case-insensitive substring match of dictionary keys against
        the filter's SENSITIVE_KEYS set; if a key matches, its value is replaced with
        "[REDACTED]". For non-matching keys the value is passed to `_sanitize_value`
        so nested dicts, lists, strings, and other supported types are sanitized
        recursively. The input dictionary is not mutated; a sanitized shallow copy is returned.
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
        Recursively sanitize a value by redacting sensitive data.
        
        - dict: delegated to _sanitize_dict and returns a sanitized dict.
        - list/tuple: returns a new sequence of the same type with each element sanitized recursively.
        - str: returns a sanitized string.
        - other types: returned unchanged.
        
        Returns:
            The sanitized value, preserving the input's type where applicable.
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
        Attach request correlation data to a LogRecord.
        
        Sets record.request_id to the current request ID (generating one if needed) and record.timestamp to the current UTC time in ISO 8601 format. Returns True to allow the record to be processed by other handlers/filters.
        """
        record.request_id = get_request_id()
        record.timestamp = datetime.utcnow().isoformat()
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """
        Format a logging.LogRecord into a JSON string with structured fields.
        
        The returned JSON contains standard fields: timestamp (uses record.timestamp if present, otherwise current UTC ISO timestamp), level, logger, message, request_id (if present on the record), module, function, and line. If the record contains exception information, an "exception" field is included using the formatter's exception formatting. Any non-standard LogRecord attributes are merged into the output as extra fields (standard LogRecord attributes are skipped).
        
        Parameters:
            record (logging.LogRecord): The log record to format.
        
        Returns:
            str: JSON-encoded string representing the structured log entry.
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
        Format a logging.LogRecord into a concise, human-readable string.
        
        The output includes timestamp, level, logger name, request ID (if present on the record), and the formatted message. For debug-level records the module name and line number are appended. If exception information is present on the record, the rendered traceback is appended on a new line.
        
        Parameters:
            record (logging.LogRecord): The log record to format.
        
        Returns:
            str: The formatted single-line (or multi-line when an exception is present) log message.
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
    Return the current request correlation ID, generating and storing a new UUIDv4 string if none exists.
    
    If a request ID has previously been set, that value is returned. Otherwise a new UUID v4 is generated, stored in the module-level _request_id, and returned.
    
    Returns:
        str: Request ID as a UUID v4 string.
    """
    global _request_id
    if _request_id is None:
        _request_id = str(uuid.uuid4())
    return _request_id


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set the current request correlation ID for logging and return it.
    
    If `request_id` is provided, stores that value in the module-level correlation ID; otherwise generates a new UUID4 string, stores it, and returns it.
    
    Parameters:
        request_id (Optional[str]): The request ID to set. If omitted or None, a new UUID4 string is generated.
    
    Returns:
        str: The request ID that was stored.
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
    Configure the root Python logger with structured or human-readable output, optional rotating file logging, sensitive-data filtering, and per-request correlation.
    
    This sets the root logger level, clears existing handlers, installs a console StreamHandler (stdout) with the chosen formatter, and optionally adds a RotatingFileHandler writing to `log_file` (creating parent directories if needed). Both handlers get a SensitiveDataFilter and RequestCorrelationFilter. The function also adjusts noisy third-party logger levels (urllib3, requests, gradio) and emits an informational log summarizing the active configuration.
    
    Parameters:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" (case-insensitive). Controls the root and handler log levels.
        format_type: "json" to use StructuredFormatter (JSON output) or "human" for HumanReadableFormatter.
        log_file: Optional path to a file for rotating file logging. If provided, parent directories will be created.
        max_size: Maximum size in bytes for the rotating log file before rotation occurs.
        backup_count: Number of rotated backup files to keep.
    
    Raises:
        ValueError: If `level` is not one of the accepted levels or if `format_type` is not "json" or "human".
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
    Decorator that logs a wrapped function's entry, successful completion, and failures.
    
    When applied, the wrapper logs a debug message on function entry (includes function name, number of positional arguments, keyword argument keys, and the current request_id), logs a debug message on successful return (includes function name and request_id), and logs an error with exception information on failure (includes function name, error message, error type, and request_id). Exceptions raised by the wrapped function are re-raised after being logged.
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper for a decorated function that logs entry, successful completion, and exceptions, including argument summary and request correlation.
        
        This inner wrapper obtains a module-level logger and the current request ID, logs a debug entry with the called function name, argument count, and keyword names, then executes the wrapped function. On success it logs completion and returns the function's result. On exception it logs an error with the exception message and type (including traceback) and re-raises the original exception.
        
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
    Log an exception with structured contextual metadata.
    
    Logs the provided exception at ERROR level and includes structured fields in the record's `extra` payload:
    `error_type` (exception class name), `error_message` (stringified exception), and the current `request_id`.
    If `context` is provided it is merged into the extra payload. The log entry is emitted with traceback information (`exc_info=True`).
    Parameters:
        error (Exception): The exception to log.
        context (Dict[str, Any], optional): Additional key/value context to include in the log record's `extra` payload.
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