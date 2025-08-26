"""
Comprehensive error handling module for the Virtual PR Firm system.

This module provides custom exceptions, error handling decorators, and
recovery strategies to ensure robust operation across all nodes.
"""

import logging
import functools
import time
from typing import Any, Callable, Optional, Dict, Type, Union, List
from enum import Enum
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"          # Can be ignored or logged
    MEDIUM = "medium"    # Should be handled but not critical
    HIGH = "high"        # Critical error requiring immediate handling
    CRITICAL = "critical"  # System failure requiring shutdown


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"              # Retry the operation
    FALLBACK = "fallback"        # Use fallback implementation
    SKIP = "skip"                # Skip this operation and continue
    FAIL = "fail"                # Fail immediately
    PARTIAL = "partial"          # Return partial results
    DEFAULT = "default"          # Return default/safe value


@dataclass
class ErrorContext:
    """Context information for error handling."""
    node_name: str
    operation: str
    attempt: int
    max_attempts: int
    error_type: Type[Exception]
    error_message: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    additional_info: Dict[str, Any]


class PRFirmError(Exception):
    """Base exception for all Virtual PR Firm errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL):
        super().__init__(message)
        self.severity = severity
        self.recovery_strategy = recovery_strategy


class ValidationError(PRFirmError):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, RecoveryStrategy.FAIL)
        self.field = field


class ContentGenerationError(PRFirmError):
    """Raised when content generation fails."""
    def __init__(self, message: str, platform: Optional[str] = None):
        super().__init__(message, ErrorSeverity.HIGH, RecoveryStrategy.RETRY)
        self.platform = platform


class StyleComplianceError(PRFirmError):
    """Raised when content violates style guidelines."""
    def __init__(self, message: str, violations: List[str]):
        super().__init__(message, ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY)
        self.violations = violations


class ExternalServiceError(PRFirmError):
    """Raised when external service (LLM, API) fails."""
    def __init__(self, message: str, service: str, status_code: Optional[int] = None):
        super().__init__(message, ErrorSeverity.HIGH, RecoveryStrategy.RETRY)
        self.service = service
        self.status_code = status_code


class ConfigurationError(PRFirmError):
    """Raised when configuration is invalid or missing."""
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, RecoveryStrategy.FAIL)
        self.config_key = config_key


class ResourceError(PRFirmError):
    """Raised when resources (memory, disk) are exhausted."""
    def __init__(self, message: str, resource_type: str):
        super().__init__(message, ErrorSeverity.CRITICAL, RecoveryStrategy.FAIL)
        self.resource_type = resource_type


def handle_errors(
    fallback_value: Any = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    exceptions_to_catch: tuple = (Exception,),
    exceptions_to_ignore: tuple = (),
    log_errors: bool = True,
    reraise: bool = True
):
    """
    Decorator for comprehensive error handling with retry logic.
    
    Args:
        fallback_value: Value to return if all retries fail
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        exceptions_to_catch: Tuple of exceptions to handle
        exceptions_to_ignore: Tuple of exceptions to not retry
        log_errors: Whether to log errors
        reraise: Whether to re-raise exception after all retries
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions_to_ignore:
                    raise
                except exceptions_to_catch as e:
                    last_exception = e
                    
                    if log_errors:
                        logger.warning(
                            f"Error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}",
                            exc_info=True
                        )
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        if exponential_backoff:
                            delay *= 2
                    else:
                        if reraise:
                            raise
                        if fallback_value is not None:
                            logger.info(f"Using fallback value for {func.__name__}")
                            return fallback_value
                        raise
            
            # This should never be reached
            if last_exception and reraise:
                raise last_exception
                
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Type[Exception] = Exception
):
    """
    Circuit breaker pattern to prevent cascading failures.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to count as failure
    """
    def decorator(func: Callable) -> Callable:
        func._failures = 0
        func._last_failure_time = None
        func._circuit_open = False
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if circuit is open
            if func._circuit_open:
                if func._last_failure_time and \
                   time.time() - func._last_failure_time > recovery_timeout:
                    func._circuit_open = False
                    func._failures = 0
                    logger.info(f"Circuit breaker for {func.__name__} reset")
                else:
                    raise ExternalServiceError(
                        f"Circuit breaker open for {func.__name__}",
                        service=func.__name__
                    )
            
            try:
                result = func(*args, **kwargs)
                # Reset failures on success
                func._failures = 0
                return result
            except expected_exception as e:
                func._failures += 1
                func._last_failure_time = time.time()
                
                if func._failures >= failure_threshold:
                    func._circuit_open = True
                    logger.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {func._failures} failures"
                    )
                
                raise
                
        return wrapper
    return decorator


def validate_input(**validators):
    """
    Decorator for input validation.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validation functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValidationError(
                                f"Validation failed for parameter '{param_name}' with value: {value}",
                                field=param_name
                            )
                    except Exception as e:
                        if isinstance(e, ValidationError):
                            raise
                        raise ValidationError(
                            f"Validation error for parameter '{param_name}': {str(e)}",
                            field=param_name
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ErrorRecoveryManager:
    """Manages error recovery strategies for nodes."""
    
    def __init__(self):
        self.recovery_handlers: Dict[Type[Exception], Callable] = {}
        self.error_history: List[ErrorContext] = []
    
    def register_handler(
        self, 
        exception_type: Type[Exception], 
        handler: Callable[[ErrorContext], Any]
    ):
        """Register a recovery handler for a specific exception type."""
        self.recovery_handlers[exception_type] = handler
    
    def handle_error(
        self, 
        error: Exception, 
        context: ErrorContext
    ) -> Any:
        """Handle an error using registered recovery strategies."""
        self.error_history.append(context)
        
        # Find appropriate handler
        for exc_type, handler in self.recovery_handlers.items():
            if isinstance(error, exc_type):
                logger.info(f"Using recovery handler for {exc_type.__name__}")
                return handler(context)
        
        # Default handling based on recovery strategy
        if context.recovery_strategy == RecoveryStrategy.RETRY:
            if context.attempt < context.max_attempts:
                logger.info(f"Retrying operation {context.operation}")
                return None  # Signal retry
        elif context.recovery_strategy == RecoveryStrategy.FALLBACK:
            logger.info(f"Using fallback for {context.operation}")
            return None  # Signal fallback
        elif context.recovery_strategy == RecoveryStrategy.SKIP:
            logger.info(f"Skipping operation {context.operation}")
            return None  # Signal skip
        elif context.recovery_strategy == RecoveryStrategy.DEFAULT:
            logger.info(f"Returning default value for {context.operation}")
            return {}  # Return safe default
        
        # Re-raise if no recovery possible
        raise error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        summary = {
            "total_errors": len(self.error_history),
            "by_severity": {},
            "by_node": {},
            "by_type": {}
        }
        
        for context in self.error_history:
            # Count by severity
            severity = context.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Count by node
            node = context.node_name
            summary["by_node"][node] = summary["by_node"].get(node, 0) + 1
            
            # Count by error type
            error_type = context.error_type.__name__
            summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1
        
        return summary


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


# Validation helper functions
def validate_non_empty_string(value: Any) -> bool:
    """Validate that value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def validate_platform_list(value: Any) -> bool:
    """Validate that value is a list of valid platforms."""
    if not isinstance(value, list) or not value:
        return False
    
    try:
        from utils.schemas import PlatformEnum
        for platform in value:
            PlatformEnum(platform)
        return True
    except (ImportError, ValueError):
        return False


def validate_positive_int(value: Any) -> bool:
    """Validate that value is a positive integer."""
    return isinstance(value, int) and value > 0


def validate_dict(value: Any) -> bool:
    """Validate that value is a dictionary."""
    return isinstance(value, dict)