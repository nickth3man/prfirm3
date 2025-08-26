"""
Centralized error handling and retry configuration.

This module provides robust error handling, retry mechanisms, and error recovery
strategies for the Virtual PR Firm pipeline.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors."""
    VALIDATION = "validation"
    API = "api"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    LOGIC = "logic"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for an error."""
    node_name: Optional[str] = None
    operation: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay

class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.error_counts: Dict[str, int] = {}
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Register default error handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default error handlers."""
        # Rate limit errors
        self.register_error_handler(
            Exception,  # Will catch rate limit exceptions from llm_utils
            self._handle_rate_limit_error
        )
        
        # Network errors
        self.register_error_handler(
            (ConnectionError, TimeoutError),
            self._handle_network_error
        )
        
        # Validation errors
        self.register_error_handler(
            ValueError,
            self._handle_validation_error
        )
    
    def register_error_handler(self, 
                              exception_type: Union[Type[Exception], tuple],
                              handler: Callable[[Exception, ErrorContext], Any]):
        """Register an error handler for a specific exception type."""
        self.error_handlers[exception_type] = handler
    
    def register_recovery_strategy(self, 
                                  category: ErrorCategory,
                                  strategy: Callable[[Exception, ErrorContext], Any]):
        """Register a recovery strategy for an error category."""
        self.recovery_strategies[category] = strategy
    
    def handle_error(self, 
                    error: Exception, 
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Any:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            severity: Severity level of the error
            
        Returns:
            Result from error handler or recovery strategy
        """
        # Log the error
        self._log_error(error, context, severity)
        
        # Update error counts
        error_key = f"{type(error).__name__}_{context.node_name or 'unknown'}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Find appropriate error handler
        handler = self._find_error_handler(error)
        if handler:
            try:
                return handler(error, context)
            except Exception as handler_error:
                logger.error(f"Error in error handler: {handler_error}")
        
        # Try recovery strategy
        category = self._categorize_error(error)
        strategy = self.recovery_strategies.get(category)
        if strategy:
            try:
                return strategy(error, context)
            except Exception as strategy_error:
                logger.error(f"Error in recovery strategy: {strategy_error}")
        
        # Default fallback
        return self._default_error_handler(error, context)
    
    def _find_error_handler(self, error: Exception) -> Optional[Callable]:
        """Find the appropriate error handler for an exception."""
        for exception_type, handler in self.error_handlers.items():
            if isinstance(error, exception_type):
                return handler
        return None
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "quota" in error_str:
            return ErrorCategory.RATE_LIMIT
        elif "timeout" in error_str:
            return ErrorCategory.TIMEOUT
        elif "connection" in error_str or "network" in error_str:
            return ErrorCategory.NETWORK
        elif "authentication" in error_str or "unauthorized" in error_str:
            return ErrorCategory.AUTHENTICATION
        elif "forbidden" in error_str or "permission" in error_str:
            return ErrorCategory.AUTHORIZATION
        elif "validation" in error_str or "invalid" in error_str:
            return ErrorCategory.VALIDATION
        elif "resource" in error_str or "memory" in error_str:
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN
    
    def _log_error(self, error: Exception, context: ErrorContext, severity: ErrorSeverity):
        """Log error with appropriate level."""
        log_message = f"Error in {context.node_name or 'unknown'}: {str(error)}"
        if context.operation:
            log_message += f" (operation: {context.operation})"
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _handle_rate_limit_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle rate limit errors with exponential backoff."""
        if context.retry_count < self.retry_config.max_retries:
            delay = self.retry_config.get_delay(context.retry_count)
            logger.info(f"Rate limit hit, retrying in {delay:.2f}s (attempt {context.retry_count + 1})")
            time.sleep(delay)
            return {"action": "retry", "delay": delay}
        else:
            logger.error("Rate limit exceeded after all retries")
            return {"action": "fallback", "reason": "rate_limit_exceeded"}
    
    def _handle_network_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle network errors with retry."""
        if context.retry_count < self.retry_config.max_retries:
            delay = self.retry_config.get_delay(context.retry_count)
            logger.info(f"Network error, retrying in {delay:.2f}s (attempt {context.retry_count + 1})")
            time.sleep(delay)
            return {"action": "retry", "delay": delay}
        else:
            logger.error("Network error after all retries")
            return {"action": "fallback", "reason": "network_unavailable"}
    
    def _handle_validation_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle validation errors."""
        logger.error(f"Validation error: {error}")
        return {"action": "fail", "reason": "validation_error", "details": str(error)}
    
    def _default_error_handler(self, error: Exception, context: ErrorContext) -> Any:
        """Default error handler when no specific handler is found."""
        logger.error(f"Unhandled error: {error}")
        return {"action": "fail", "reason": "unhandled_error", "details": str(error)}
    
    def should_retry(self, error: Exception, context: ErrorContext) -> bool:
        """Determine if an error should be retried."""
        if context.retry_count >= self.retry_config.max_retries:
            return False
        
        category = self._categorize_error(error)
        retryable_categories = {
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RESOURCE
        }
        
        return category in retryable_categories
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts)
        }
    
    def reset_error_counts(self):
        """Reset error count statistics."""
        self.error_counts.clear()

def create_error_handler(retry_config: Optional[RetryConfig] = None) -> ErrorHandler:
    """Factory function to create an error handler."""
    return ErrorHandler(retry_config)

# Global error handler instance
_global_error_handler = ErrorHandler()

def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler

def handle_error(error: Exception, 
                context: ErrorContext,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Any:
    """Convenience function to handle errors using the global handler."""
    return _global_error_handler.handle_error(error, context, severity)

# Test function for development
if __name__ == "__main__":
    # Test error handler
    error_handler = create_error_handler(RetryConfig(max_retries=2))
    
    # Test rate limit error
    context = ErrorContext(node_name="TestNode", operation="test_operation")
    
    class RateLimitError(Exception):
        pass
    
    error = RateLimitError("Rate limit exceeded")
    result = error_handler.handle_error(error, context)
    print(f"Rate limit error result: {result}")
    
    # Test error stats
    stats = error_handler.get_error_stats()
    print(f"Error stats: {stats}")