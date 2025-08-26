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
        super().__init__()
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SENSITIVE_PATTERNS]
    
    def filter(self, record):
        """Filter sensitive data from log record."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)
        
        if hasattr(record, 'args') and record.args:
            record.args = tuple(self._sanitize_arg(arg) for arg in record.args)
        
        return True
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize log message."""
        for pattern in self.patterns:
            message = pattern.sub('[REDACTED]', message)
        return message
    
    def _sanitize_arg(self, arg: Any) -> Any:
        """Sanitize log arguments."""
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
        super().__init__()
        self.include_correlation_id = include_correlation_id
    
    def format(self, record):
        """Format log record as structured JSON."""
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
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.include_correlation_id = include_correlation_id
    
    def format(self, record):
        """Format log record for human reading."""
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
        self.request_id = request_id or str(uuid.uuid4())
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.start_time = time.time()
        self._old_request_id = None
        self._old_correlation_id = None
    
    def __enter__(self):
        """Set request context."""
        self._old_request_id = request_id_var.get()
        self._old_correlation_id = correlation_id_var.get()
        
        request_id_var.set(self.request_id)
        correlation_id_var.set(self.correlation_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context."""
        request_id_var.set(self._old_request_id)
        correlation_id_var.set(self._old_correlation_id)
    
    def get_duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        return (time.time() - self.start_time) * 1000


def configure_logging(
    level: str = "INFO",
    format_type: str = "human",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    include_correlation_id: bool = True
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type ('human' or 'json')
        log_file: Optional file path for logging
        max_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        include_correlation_id: Whether to include correlation IDs in logs
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
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function calls with timing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
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
    """Log flow execution details."""
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
    """Log flow completion details."""
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
    """Log user actions for analytics."""
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
    """Log performance metrics."""
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
    """Log security-related events."""
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
    """Create a new request context for logging."""
    return RequestContextManager(request_id=request_id)


# Convenience functions for common logging patterns
def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an error with context."""
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
    """Log a warning with context."""
    logger = get_logger('warnings')
    
    extra = {
        'request_id': request_id_var.get(),
        'correlation_id': correlation_id_var.get()
    }
    
    if context:
        extra.update(context)
    
    logger.warning(message, extra=extra)


def log_info(message: str, context: Optional[Dict[str, Any]] = None):
    """Log an info message with context."""
    logger = get_logger('info')
    
    extra = {
        'request_id': request_id_var.get(),
        'correlation_id': correlation_id_var.get()
    }
    
    if context:
        extra.update(context)
    
    logger.info(message, extra=extra)


def log_debug(message: str, context: Optional[Dict[str, Any]] = None):
    """Log a debug message with context."""
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