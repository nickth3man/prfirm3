"""Comprehensive logging configuration for the Virtual PR Firm application.

This module provides structured logging with correlation ID tracking,
sensitive data filtering, and environment-based configuration.
"""

import logging
import logging.handlers
import json
import uuid
import time
import re
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from functools import wraps
import threading
from contextvars import ContextVar

# Thread-local storage for request context
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


@dataclass
class LogContext:
    """Context information for logging."""
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    platform: Optional[str] = None
    topic_hash: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from log messages."""
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS = [
        r'api_key["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'password["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'token["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'secret["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'key["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'authorization["\']?\s*[:=]\s*["\'][^"\']+["\']',
        r'bearer\s+[a-zA-Z0-9._-]+',
        r'sk-[a-zA-Z0-9]{48}',
        r'pk-[a-zA-Z0-9]{48}',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card pattern
    ]
    
    def __init__(self):
        """
        Initialize the SensitiveDataFilter.
        
        Compiles the module's SENSITIVE_PATTERNS into case-insensitive regular expression objects
        and stores them on self.patterns for use when sanitizing log messages and arguments.
        """
        super().__init__()
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SENSITIVE_PATTERNS]
    
    def filter(self, record):
        """
        Sanitize sensitive values on a logging.LogRecord in place.
        
        This filter inspects the LogRecord's `msg` and `args` and redacts any matched
        sensitive patterns (replacing them with "[REDACTED]"). Mutates `record.msg`
        (and, if present, `record.args`) so downstream handlers receive the sanitized
        values.
        
        Parameters:
            record (logging.LogRecord): The log record to sanitize.
        
        Returns:
            bool: Always returns True so logging continues processing the record.
        """
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)
        
        if hasattr(record, 'args') and record.args:
            record.args = tuple(self._sanitize_arg(arg) for arg in record.args)
        
        return True
    
    def _sanitize_message(self, message: str) -> str:
        """
        Replace substrings matching the filter's compiled sensitive patterns (self.patterns) with "[REDACTED]" and return the sanitized message.
        
        Returns:
            str: The message with sensitive data redacted.
        """
        for pattern in self.patterns:
            message = pattern.sub('[REDACTED]', message)
        return message
    
    def _sanitize_arg(self, arg: Any) -> Any:
        """
        Sanitize a log argument by redacting sensitive substrings while preserving structure and type.
        
        This method recursively processes the given argument:
        - If it's a string, returns a sanitized string with sensitive patterns replaced.
        - If it's a dict/list/tuple, returns a new object of the same type with all nested elements sanitized.
        - All other types are returned unchanged.
        
        Returns:
            The sanitized value with the same type as the input (strings, dicts, lists, and tuples are processed; other types are returned as-is).
        """
        if isinstance(arg, str):
            return self._sanitize_message(arg)
        elif isinstance(arg, dict):
            return {k: self._sanitize_arg(v) for k, v in arg.items()}
        elif isinstance(arg, (list, tuple)):
            return type(arg)(self._sanitize_arg(item) for item in arg)
        return arg


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def __init__(self, include_correlation_id: bool = True):
        """
        Initialize the formatter.
        
        Parameters:
            include_correlation_id (bool): If True, the formatter will include correlation_id and request_id
                from the module's context variables in formatted output; otherwise those fields are omitted.
        """
        super().__init__()
        self.include_correlation_id = include_correlation_id
    
    def format(self, record):
        """
        Format a LogRecord into a JSON string for structured logging.
        
        Converts the provided LogRecord into a JSON-serializable dict containing:
        - required fields: `timestamp`, `level`, `logger`, `message`, `module`, `function`, `line`.
        - optional context IDs: `correlation_id` and `request_id` when `include_correlation_id` is enabled and values are present in the module ContextVars.
        - exception information under `exception` when `record.exc_info` is set.
        - any non-standard LogRecord attributes (extra fields) are copied into the output.
        
        Parameters:
            record (logging.LogRecord): The log record to format. Extra attributes attached to the record (beyond standard logging attributes) will be included in the output.
        
        Returns:
            str: A JSON string representation of the structured log entry. Default JSON encoding for non-serializable objects is handled via str().
        """
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        if self.include_correlation_id:
            correlation_id = correlation_id_var.get()
            if correlation_id:
                log_entry['correlation_id'] = correlation_id
            
            request_id = request_id_var.get()
            if request_id:
                log_entry['request_id'] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    def __init__(self, include_correlation_id: bool = True):
        """
        Initialize the human-readable log formatter.
        
        Parameters:
            include_correlation_id (bool): If True (default), prefix formatted messages with correlation and request IDs from context when available.
        """
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.include_correlation_id = include_correlation_id
    
    def format(self, record):
        """
        Format a logging.LogRecord into a human-readable string, optionally prefixed with correlation and request IDs.
        
        Returns the base formatter output and, if include_correlation_id is True, prepends available correlation_id and request_id from the module-level ContextVars (correlation_id_var, request_id_var) in square brackets, newest first.
        """
        formatted = super().format(record)
        
        # Add correlation ID if available
        if self.include_correlation_id:
            correlation_id = correlation_id_var.get()
            if correlation_id:
                formatted = f"[{correlation_id}] {formatted}"
            
            request_id = request_id_var.get()
            if request_id:
                formatted = f"[{request_id}] {formatted}"
        
        return formatted


class RequestContextManager:
    """Manages request context for logging."""
    
    def __init__(self, request_id: Optional[str] = None, correlation_id: Optional[str] = None):
        """
        Initialize a RequestContextManager, creating or storing the request and correlation IDs and recording the start time.
        
        If request_id or correlation_id are not provided, new UUIDs are generated. Also prepares internal slots to save previous context values so they can be restored when the context manager exits.
        
        Parameters:
            request_id (Optional[str]): Existing request identifier to use for this context; if None a new UUID is generated.
            correlation_id (Optional[str]): Existing correlation identifier to use for this context; if None a new UUID is generated.
        """
        self.request_id = request_id or str(uuid.uuid4())
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.start_time = time.time()
        self._old_request_id = None
        self._old_correlation_id = None
    
    def __enter__(self):
        """
        Enter the request context: saves current ContextVar values, sets the manager's request_id and correlation_id into the context, and returns self.
        
        This method stores the previous values of the module-level ContextVars so they can be restored on exit, then sets request_id_var and correlation_id_var to the values provided when the RequestContextManager was created.
        
        Returns:
            self: The context manager instance.
        """
        self._old_request_id = request_id_var.get()
        self._old_correlation_id = correlation_id_var.get()
        
        request_id_var.set(self.request_id)
        correlation_id_var.set(self.correlation_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore the previous request and correlation IDs into the context variables upon exiting the request context.
        
        This is called by the context manager protocol; it resets the module-level ContextVars (request_id_var and correlation_id_var) to the values that were active before the context was entered. It does not suppress exceptions (returns None).
        """
        request_id_var.set(self._old_request_id)
        correlation_id_var.set(self._old_correlation_id)
    
    def get_duration_ms(self) -> float:
        """
        Return the elapsed wall-clock time in milliseconds since this RequestContextManager was created.
        
        Returns:
            float: Elapsed time in milliseconds (may be fractional, non-negative).
        """
        return (time.time() - self.start_time) * 1000


def configure_logging(
    level: str = "INFO",
    format_type: str = "human",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    include_correlation_id: bool = True
) -> None:
    """
    Configure the application's root logger with console (and optional rotating file) handlers.
    
    Sets the logging level from a string name, chooses either a human-readable or JSON structured formatter, attaches the SensitiveDataFilter to redact sensitive values, and replaces existing root handlers (force reconfigure). Supports an optional rotating file handler with size and backup limits and an option to include correlation/request IDs in formatted output.
    
    Parameters:
        level: Log level name (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        format_type: Formatter type; "human" for readable text or "json" for structured JSON output.
        log_file: Optional filesystem path to write logs to; when provided a RotatingFileHandler is added.
        max_size: Maximum size in bytes for a rotated log file before rollover.
        backup_count: Number of rotated log files to retain.
        include_correlation_id: If True, include correlation/request IDs in formatted log records.
    """
    # Determine log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if format_type.lower() == 'json':
        formatter = StructuredFormatter(include_correlation_id=include_correlation_id)
    else:
        formatter = HumanReadableFormatter(include_correlation_id=include_correlation_id)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    console_handler.addFilter(SensitiveDataFilter())
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        file_handler.addFilter(SensitiveDataFilter())
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured",
        extra={
            'level': level,
            'format': format_type,
            'log_file': log_file,
            'include_correlation_id': include_correlation_id
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with the application's logging settings.
    
    Parameters:
        name (str): Logger name (typically __name__ of the calling module).
    
    Returns:
        logging.Logger: Logger instance configured by this module's logging setup.
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator that logs a function's entry, successful completion (with execution time), and errors.
    
    Wraps a function to:
    - Log a DEBUG-level message when the function is entered.
    - Measure execution time and log a DEBUG-level completion message with duration in milliseconds on success.
    - Log an ERROR-level message with exception details and re-raise the exception if the function raises.
    - Attach request and correlation IDs taken from module-level ContextVars to all log entries.
    
    The decorator uses the module's logger (get_logger(func.__module__)), preserves the wrapped function's metadata, and sets structured `extra` fields including `function`, `duration_ms`, `error`/`error_type`, `request_id`, and `correlation_id`.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Decorator wrapper that logs entry, successful completion (with duration), and errors for the wrapped function.
        
        The wrapper forwards all positional and keyword arguments to the wrapped function, records execution time in milliseconds, and emits debug logs on entry and successful exit. If the wrapped function raises an exception, the wrapper logs an error including the exception type/message and duration, then re-raises the same exception. Logged records include the function name, duration_ms, and current request_id and correlation_id (when available).
         
        Returns:
            The return value of the wrapped function.
        
        Raises:
            Re-raises any exception raised by the wrapped function after logging it.
        """
        logger = get_logger(func.__module__)
        
        # Get request context
        request_id = request_id_var.get()
        correlation_id = correlation_id_var.get()
        
        start_time = time.time()
        
        # Log function entry
        logger.debug(
            f"Entering {func.__name__}",
            extra={
                'function': func.__name__,
                'module': func.__module__,
                'request_id': request_id,
                'correlation_id': correlation_id
            }
        )
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful completion
            logger.debug(
                f"Completed {func.__name__} in {duration_ms:.2f}ms",
                extra={
                    'function': func.__name__,
                    'duration_ms': duration_ms,
                    'request_id': request_id,
                    'correlation_id': correlation_id
                }
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"Error in {func.__name__}: {str(e)}",
                extra={
                    'function': func.__name__,
                    'duration_ms': duration_ms,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'request_id': request_id,
                    'correlation_id': correlation_id
                },
                exc_info=True
            )
            raise
    
    return wrapper


def log_flow_execution(flow_name: str, shared: Dict[str, Any]):
    """
    Log that a named flow has started, recording platforms and a privacy-preserving topic hash.
    
    Logs an INFO-level message indicating the start of `flow_name`. If `shared` contains a 'task_requirements' mapping with keys 'platforms' (list) and 'topic_or_goal' (string), those platforms are included and the topic is hashed (MD5, first 8 hex chars) to avoid storing raw topic text in logs. The log record's extra fields include: 'flow_name', 'platforms', 'topic_hash', 'request_id', and 'correlation_id'.
    
    Parameters:
        flow_name (str): Human-readable name of the flow being executed.
        shared (Dict[str, Any]): Shared execution state; expected to optionally contain
            'task_requirements' -> {'platforms': [...], 'topic_or_goal': '...'}.
    """
    logger = get_logger('flow')
    
    # Extract relevant information from shared store
    platforms = shared.get('task_requirements', {}).get('platforms', [])
    topic = shared.get('task_requirements', {}).get('topic_or_goal', '')
    
    # Create topic hash for privacy
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8] if topic else None
    
    logger.info(
        f"Starting flow execution: {flow_name}",
        extra={
            'flow_name': flow_name,
            'platforms': platforms,
            'topic_hash': topic_hash,
            'request_id': request_id_var.get(),
            'correlation_id': correlation_id_var.get()
        }
    )


def log_flow_completion(flow_name: str, shared: Dict[str, Any], duration_ms: float, success: bool):
    """
    Record the completion or failure of a flow run to the 'flow' logger.
    
    Logs an INFO on success or ERROR on failure and attaches structured metadata useful for tracing and analytics.
    - Extracts platforms and topic from shared['task_requirements'] (keys: 'platforms', 'topic_or_goal').
    - If a topic is present, a privacy-preserving topic_hash is generated (MD5, first 8 hex chars).
    - Determines platforms_completed from the keys of shared.get('content_pieces', {}).
    - Adds the following fields to the log `extra`: flow_name, platforms, platforms_completed, topic_hash, duration_ms, success, request_id, correlation_id.
    
    Parameters:
        flow_name (str): Human-readable identifier for the flow.
        shared (Dict[str, Any]): Shared flow state. Expected to optionally contain:
            - 'task_requirements': dict with optional 'platforms' (list) and 'topic_or_goal' (str).
            - 'content_pieces': dict whose keys represent completed platforms.
        duration_ms (float): Execution duration in milliseconds.
        success (bool): Whether the flow completed successfully; controls log level (INFO if True, ERROR if False).
    """
    logger = get_logger('flow')
    
    platforms = shared.get('task_requirements', {}).get('platforms', [])
    topic = shared.get('task_requirements', {}).get('topic_or_goal', '')
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8] if topic else None
    
    content_pieces = shared.get('content_pieces', {})
    platforms_completed = list(content_pieces.keys()) if content_pieces else []
    
    log_level = logging.INFO if success else logging.ERROR
    message = f"Flow completed: {flow_name}" if success else f"Flow failed: {flow_name}"
    
    logger.log(
        log_level,
        message,
        extra={
            'flow_name': flow_name,
            'platforms': platforms,
            'platforms_completed': platforms_completed,
            'topic_hash': topic_hash,
            'duration_ms': duration_ms,
            'success': success,
            'request_id': request_id_var.get(),
            'correlation_id': correlation_id_var.get()
        }
    )


def log_user_action(action: str, details: Dict[str, Any]):
    """
    Record a user action event for analytics.
    
    Logs an INFO-level message to the 'user_actions' logger with the provided action name and a dictionary of contextual details. The `details` dict is attached to the log record under the `details` key and the log also includes the current request and correlation IDs from context variables.
    
    Parameters:
        action (str): Short identifier or name of the user action (e.g., "signup", "click_publish").
        details (Dict[str, Any]): Arbitrary context about the action (metadata useful for analytics).
    """
    logger = get_logger('user_actions')
    
    logger.info(
        f"User action: {action}",
        extra={
            'action': action,
            'details': details,
            'request_id': request_id_var.get(),
            'correlation_id': correlation_id_var.get()
        }
    )


def log_performance_metric(metric_name: str, value: float, unit: str = "ms"):
    """
    Log a named performance metric at INFO level.
    
    Records a performance metric to the "performance" logger and includes the current request and correlation IDs from context variables. The metric is emitted as a structured log entry via the `extra` mapping with keys: `metric_name`, `value`, `unit`, `request_id`, and `correlation_id`.
    
    Parameters:
        metric_name (str): Human-readable name of the metric (e.g., "db.query.time").
        value (float): Numeric metric value.
        unit (str): Unit for the metric value (default "ms").
    """
    logger = get_logger('performance')
    
    logger.info(
        f"Performance metric: {metric_name}",
        extra={
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'request_id': request_id_var.get(),
            'correlation_id': correlation_id_var.get()
        }
    )


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "info"):
    """
    Log a security-related event to the 'security' logger.
    
    Parameters:
        event_type (str): Short identifier for the security event (e.g., "auth_failure", "suspicious_activity").
        details (Dict[str, Any]): Structured metadata about the event (will be attached to the log `extra` payload).
        severity (str): Log severity name (e.g., "debug", "info", "warning", "error", "critical"); defaults to "info".
        
    This function emits a log entry at the resolved logging level and includes the provided `event_type`,
    `severity`, and `details` in the record's `extra` fields. It also attaches the current `request_id`
    and `correlation_id` from the module's context variables for traceability.
    """
    logger = get_logger('security')
    
    log_level = getattr(logging, severity.upper(), logging.INFO)
    
    logger.log(
        log_level,
        f"Security event: {event_type}",
        extra={
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'request_id': request_id_var.get(),
            'correlation_id': correlation_id_var.get()
        }
    )


def create_request_context(request_id: Optional[str] = None) -> RequestContextManager:
    """
    Create a RequestContextManager for use as a context manager that sets request and correlation IDs for logging.
    
    If request_id is omitted, a new UUID will be generated. The returned RequestContextManager also generates a correlation_id (unless provided via its constructor) and records the start time; use it with `with create_request_context(...)` to ensure IDs are set for the duration of a request scope.
    
    Parameters:
        request_id (Optional[str]): Optional request identifier to use instead of generating a new UUID.
    
    Returns:
        RequestContextManager: A context manager that sets/restores request_id and correlation_id and tracks request duration.
    """
    return RequestContextManager(request_id=request_id)


# Convenience functions for common logging patterns
def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an exception with structured contextual fields to the 'errors' logger.
    
    Logs an ERROR-level entry that includes the exception type and message, the current request_id and correlation_id from context variables, and any additional key/value pairs provided via `context`. The exception traceback is included via `exc_info=True`. 
    
    Parameters:
        error (Exception): The exception to log.
        context (Optional[Dict[str, Any]]): Optional additional fields to merge into the log record's `extra` payload (e.g., transaction ids, user info). Omitted or None results in no extra fields beyond the default ones.
    """
    logger = get_logger('errors')
    
    extra = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'request_id': request_id_var.get(),
        'correlation_id': correlation_id_var.get()
    }
    
    if context:
        extra.update(context)
    
    logger.error(
        f"Error occurred: {str(error)}",
        extra=extra,
        exc_info=True
    )


def log_warning(message: str, context: Optional[Dict[str, Any]] = None):
    """
    Log a warning-level message, automatically attaching the current request and correlation IDs.
    
    The provided `context` dict, if any, is merged into the logger `extra` payload and will be included with the log record (useful for adding structured fields like user_id, session_id, or operation). This function does not return a value.
    Parameters:
        message (str): The warning message to log.
        context (Optional[Dict[str, Any]]): Optional mapping of additional fields to include in the log's `extra` payload.
    """
    logger = get_logger('warnings')
    
    extra = {
        'request_id': request_id_var.get(),
        'correlation_id': correlation_id_var.get()
    }
    
    if context:
        extra.update(context)
    
    logger.warning(message, extra=extra)


def log_info(message: str, context: Optional[Dict[str, Any]] = None):
    """
    Log an informational message enriched with the current request and correlation IDs.
    
    If provided, the `context` mapping is merged into the log's `extra` payload. The function automatically adds `request_id` and `correlation_id` from the module's ContextVars so those IDs are included in the emitted INFO-level log.
    Parameters:
        context (Optional[Dict[str, Any]]): Additional fields to include in the log record's `extra` data.
    """
    logger = get_logger('info')
    
    extra = {
        'request_id': request_id_var.get(),
        'correlation_id': correlation_id_var.get()
    }
    
    if context:
        extra.update(context)
    
    logger.info(message, extra=extra)


def log_debug(message: str, context: Optional[Dict[str, Any]] = None):
    """
    Log a debug-level message enriched with current request and correlation IDs.
    
    The provided `message` is logged at DEBUG level and the log record's `extra` data
    includes the current `request_id` and `correlation_id` from the module context.
    If `context` is supplied, its key/value pairs are merged into the `extra` payload
    and will be attached to the log record.
    
    Parameters:
        message (str): The message to log.
        context (Optional[Dict[str, Any]]): Additional fields to include in the log's `extra` data.
    """
    logger = get_logger('debug')
    
    extra = {
        'request_id': request_id_var.get(),
        'correlation_id': correlation_id_var.get()
    }
    
    if context:
        extra.update(context)
    
    logger.debug(message, extra=extra)


# Initialize logging when module is imported
import hashlib