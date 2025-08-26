"""Error handling utilities for the Virtual PR Firm.

This module provides comprehensive error handling decorators and exception classes
for the Virtual PR Firm application, ensuring graceful degradation and proper logging.
"""

import functools
import logging
import traceback
import time
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    args: tuple
    kwargs: dict
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None


class VirtualPRError(Exception):
    """Base exception class for Virtual PR Firm errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context
        self.timestamp = time.time()


class ValidationError(VirtualPRError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message, ErrorSeverity.MEDIUM)
        self.field = field
        self.value = value


class ConfigurationError(VirtualPRError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, ErrorSeverity.HIGH)
        self.config_key = config_key


class FlowExecutionError(VirtualPRError):
    """Raised when flow execution fails."""
    
    def __init__(self, message: str, flow_name: Optional[str] = None, 
                 node_name: Optional[str] = None):
        super().__init__(message, ErrorSeverity.HIGH)
        self.flow_name = flow_name
        self.node_name = node_name


class LLMError(VirtualPRError):
    """Raised when LLM operations fail."""
    
    def __init__(self, message: str, provider: Optional[str] = None, 
                 model: Optional[str] = None):
        super().__init__(message, ErrorSeverity.HIGH)
        self.provider = provider
        self.model = model


class RateLimitError(VirtualPRError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, ErrorSeverity.MEDIUM)
        self.retry_after = retry_after


def handle_errors(func: Callable) -> Callable:
    """Decorator to handle errors with logging and graceful degradation.
    
    Args:
        func: Function to wrap with error handling
    
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        context = ErrorContext(
            function_name=func.__name__,
            args=args,
            kwargs=kwargs
        )
        
        try:
            return func(*args, **kwargs)
        except VirtualPRError as e:
            # Log Virtual PR specific errors
            logger.error(f"Virtual PR Error in {func.__name__}: {e.message}", 
                        extra={"severity": e.severity.value, "context": context})
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", 
                        extra={"traceback": traceback.format_exc(), "context": context})
            raise VirtualPRError(f"Unexpected error in {func.__name__}: {str(e)}", 
                               ErrorSeverity.HIGH, context)
    
    return wrapper


def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0,
                  exceptions: tuple = (Exception,)) -> Callable:
    """Decorator to retry functions on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to retry on
    
    Returns:
        Wrapped function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed in {func.__name__}: {str(e)}")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed in {func.__name__}: {str(e)}")
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float, default_return: Any = None) -> Callable:
    """Decorator to handle function timeouts.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        default_return: Value to return if timeout occurs
    
    Returns:
        Wrapped function with timeout handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler_signal(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set up signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler_signal)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutError:
                logger.warning(f"Function {func.__name__} timed out, returning default value")
                return default_return
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def validate_inputs(*validators: Callable) -> Callable:
    """Decorator to validate function inputs.
    
    Args:
        *validators: Validation functions to apply
    
    Returns:
        Wrapped function with input validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for validator in validators:
                try:
                    validator(*args, **kwargs)
                except Exception as e:
                    raise ValidationError(f"Input validation failed: {str(e)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time.
    
    Args:
        func: Function to wrap with timing
    
    Returns:
        Wrapped function with execution time logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


def safe_execute(func: Callable, *args, default_return: Any = None, 
                log_errors: bool = True, **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if execution fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or default_return if execution fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_return


def create_error_response(error: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """Create a standardized error response.
    
    Args:
        error: The exception that occurred
        include_traceback: Whether to include traceback information
    
    Returns:
        Dictionary with error information
    """
    response = {
        "error": True,
        "message": str(error),
        "type": type(error).__name__,
        "timestamp": time.time()
    }
    
    if isinstance(error, VirtualPRError):
        response["severity"] = error.severity.value
        if error.context:
            response["context"] = {
                "function": error.context.function_name,
                "retry_count": error.context.retry_count
            }
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response


def setup_error_handling(log_level: str = "INFO", 
                        log_file: Optional[str] = None) -> None:
    """Set up comprehensive error handling and logging.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Configure logging
    log_config = {
        "level": getattr(logging, log_level.upper()),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": [logging.StreamHandler()]
    }
    
    if log_file:
        log_config["handlers"].append(logging.FileHandler(log_file))
    
    logging.basicConfig(**log_config)
    
    # Set up exception hook for unhandled exceptions
    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    import sys
    sys.excepthook = handle_unhandled_exception