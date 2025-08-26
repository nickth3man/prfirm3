# utils/circuit_breaker.py
"""Circuit Breaker utility for preventing cascading failures from external dependencies.

This module implements a configurable circuit breaker pattern that monitors external
service calls and prevents cascading failures by temporarily blocking calls when
failure rates exceed configured thresholds.

Circuit Breaker States:
    - CLOSED: Normal operation, calls are allowed
    - OPEN: Failing state, calls are blocked and fail fast
    - HALF_OPEN: Testing recovery, limited calls are allowed

Usage Examples:
    # As a decorator
    @circuit_breaker(failure_threshold=5, reset_timeout=60)
    def call_external_api():
        # Your external API call here
        pass
    
    # As a context manager
    with CircuitBreaker() as cb:
        result = cb.call(external_function, *args, **kwargs)
    
    # Direct usage
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30)
    result = breaker.call(risky_function, arg1, arg2)
"""

import logging
import time
import threading
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

log = logging.getLogger(__name__)

# Type variable for generic function returns
T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking calls due to failures
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker performance."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    blocked_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreaker:
    """Configurable circuit breaker for external dependency calls.
    
    Implements the circuit breaker pattern to prevent cascading failures
    by monitoring failure rates and temporarily blocking calls when
    thresholds are exceeded.
    
    Args:
        failure_threshold: Number of consecutive failures before opening circuit
        reset_timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN
        window_size: Number of recent calls to consider for failure rate calculation
        success_threshold: Number of successful calls needed in HALF_OPEN to close circuit
        name: Optional name for logging and identification
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        window_size: int = 10,
        success_threshold: int = 2,
        name: Optional[str] = None
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.name = name or f"CircuitBreaker-{id(self)}"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        
        # Recent call history for window-based calculations
        self._call_history: deque = deque(maxlen=window_size)
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        log.debug(f"Initialized {self.name} with threshold={failure_threshold}, "
                 f"timeout={reset_timeout}s, window={window_size}")
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function call through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result if call is allowed and successful
            
        Raises:
            CircuitBreakerError: If circuit is open and call is blocked
            Exception: Any exception raised by the wrapped function
        """
        with self._lock:
            # Check if we should allow the call
            if not self._should_allow_call():
                self.metrics.blocked_calls += 1
                raise CircuitBreakerError(
                    f"Circuit breaker {self.name} is {self._state.value}, "
                    f"blocking call to {func.__name__}"
                )
            
            # Attempt the call
            self.metrics.total_calls += 1
            call_start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                self._record_success(call_start_time)
                return result
                
            except Exception as e:
                self._record_failure(call_start_time, e)
                raise
    
    def _should_allow_call(self) -> bool:
        """Determine if a call should be allowed based on current state."""
        current_time = time.time()
        
        if self._state == CircuitBreakerState.CLOSED:
            return True
            
        elif self._state == CircuitBreakerState.OPEN:
            # Check if enough time has passed to transition to HALF_OPEN
            if (self._last_failure_time and 
                current_time - self._last_failure_time >= self.reset_timeout):
                self._transition_to_half_open()
                return True
            return False
            
        elif self._state == CircuitBreakerState.HALF_OPEN:
            # Allow limited calls to test recovery
            return True
            
        return False
    
    def _record_success(self, call_start_time: float) -> None:
        """Record a successful call and update state if necessary."""
        call_time = time.time() - call_start_time
        
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = time.time()
        self._call_history.append(True)
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        
        log.debug(f"{self.name}: Successful call (took {call_time:.3f}s)")
        
        # State transitions based on success
        if self._state == CircuitBreakerState.HALF_OPEN:
            if self._consecutive_successes >= self.success_threshold:
                self._transition_to_closed()
    
    def _record_failure(self, call_start_time: float, exception: Exception) -> None:
        """Record a failed call and update state if necessary."""
        call_time = time.time() - call_start_time
        
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()
        self._call_history.append(False)
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        self._last_failure_time = time.time()
        
        log.warning(f"{self.name}: Failed call (took {call_time:.3f}s): {exception}")
        
        # State transitions based on failure
        if (self._state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN] and
            self._consecutive_failures >= self.failure_threshold):
            self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.OPEN
        self.metrics.state_changes += 1
        
        log.warning(f"{self.name}: Circuit breaker OPENED - "
                   f"consecutive failures: {self._consecutive_failures}")
        
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.HALF_OPEN
        self._consecutive_successes = 0
        self.metrics.state_changes += 1
        
        log.info(f"{self.name}: Circuit breaker HALF_OPEN - testing recovery")
        
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        old_state = self._state
        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self.metrics.state_changes += 1
        
        log.info(f"{self.name}: Circuit breaker CLOSED - recovery successful")
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._last_failure_time = None
            self._call_history.clear()
            
            if old_state != CircuitBreakerState.CLOSED:
                self.metrics.state_changes += 1
                
            log.info(f"{self.name}: Circuit breaker manually reset to CLOSED")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status and metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "window_size": self.window_size,
                "success_threshold": self.success_threshold,
                "metrics": {
                    "total_calls": self.metrics.total_calls,
                    "successful_calls": self.metrics.successful_calls,
                    "failed_calls": self.metrics.failed_calls,
                    "blocked_calls": self.metrics.blocked_calls,
                    "failure_rate": self.metrics.failure_rate,
                    "success_rate": self.metrics.success_rate,
                    "state_changes": self.metrics.state_changes,
                    "last_failure_time": self.metrics.last_failure_time,
                    "last_success_time": self.metrics.last_success_time,
                }
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # No cleanup needed for basic usage
        pass


# Global registry for shared circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.RLock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    window_size: int = 10,
    success_threshold: int = 2
) -> CircuitBreaker:
    """Get or create a named circuit breaker from the global registry.
    
    This function provides a way to share circuit breakers across different
    parts of the application by name, ensuring consistent behavior for the
    same external dependency.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures before opening (only used for new breakers)
        reset_timeout: Timeout before testing recovery (only used for new breakers)
        window_size: Size of call history window (only used for new breakers)
        success_threshold: Successes needed to close (only used for new breakers)
    
    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                window_size=window_size,
                success_threshold=success_threshold,
                name=name
            )
        return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    window_size: int = 10,
    success_threshold: int = 2
) -> Callable:
    """Decorator to wrap functions with circuit breaker protection.
    
    Args:
        name: Optional name for the circuit breaker (defaults to function name)
        failure_threshold: Number of consecutive failures before opening circuit
        reset_timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN
        window_size: Number of recent calls to consider
        success_threshold: Number of successful calls needed in HALF_OPEN to close
    
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = get_circuit_breaker(
            breaker_name, failure_threshold, reset_timeout, 
            window_size, success_threshold
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, **kwargs)
            
        # Expose circuit breaker for testing/monitoring
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator


def list_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """List all registered circuit breakers and their status.
    
    Returns:
        Dictionary mapping breaker names to their status information
    """
    with _registry_lock:
        return {name: breaker.get_status() 
                for name, breaker in _circuit_breakers.items()}


def reset_all_circuit_breakers() -> None:
    """Reset all registered circuit breakers to closed state.
    
    Useful for testing or manual recovery operations.
    """
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()
        log.info("Reset all circuit breakers to CLOSED state")


def clear_circuit_breaker_registry() -> None:
    """Clear the global circuit breaker registry.
    
    Primarily used for testing to ensure clean state.
    """
    with _registry_lock:
        _circuit_breakers.clear()
        log.debug("Cleared circuit breaker registry")


# Test function for development
def test_circuit_breaker():
    """Test the circuit breaker functionality with simulated failures."""
    
    # Create a test function that fails intermittently
    call_count = 0
    
    @circuit_breaker(name="test_function", failure_threshold=3, reset_timeout=2)
    def unreliable_function(should_fail=False):
        nonlocal call_count
        call_count += 1
        if should_fail:
            raise Exception(f"Simulated failure #{call_count}")
        return f"Success #{call_count}"
    
    print("Testing Circuit Breaker functionality...")
    
    # Test normal operation
    print("\n1. Normal operation:")
    for i in range(3):
        try:
            result = unreliable_function(should_fail=False)
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: Error - {e}")
    
    # Test failure scenario
    print("\n2. Failure scenario (should open circuit):")
    for i in range(5):
        try:
            result = unreliable_function(should_fail=True)
            print(f"  Call {i+1}: {result}")
        except CircuitBreakerError as e:
            print(f"  Call {i+1}: Circuit breaker blocked - {e}")
        except Exception as e:
            print(f"  Call {i+1}: Function error - {e}")
    
    # Check circuit breaker status
    breaker = unreliable_function._circuit_breaker
    print(f"\n3. Circuit breaker status: {breaker.get_status()}")
    
    # Test recovery
    print("\n4. Testing recovery (waiting for reset timeout)...")
    time.sleep(2.5)  # Wait for reset timeout
    
    for i in range(3):
        try:
            result = unreliable_function(should_fail=False)
            print(f"  Recovery call {i+1}: {result}")
        except Exception as e:
            print(f"  Recovery call {i+1}: Error - {e}")
    
    print(f"\n5. Final status: {breaker.get_status()}")
    print("✅ Circuit breaker test completed")


if __name__ == "__main__":
    # Enable debug logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Run test
    test_circuit_breaker()