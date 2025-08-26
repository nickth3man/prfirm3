"""Tests for the Circuit Breaker utility.

This module provides comprehensive tests for the circuit breaker implementation,
including state transitions, failure scenarios, recovery testing, and edge cases.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerState, CircuitBreakerError,
    circuit_breaker, get_circuit_breaker, list_circuit_breakers,
    reset_all_circuit_breakers, clear_circuit_breaker_registry
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        clear_circuit_breaker_registry()
        
        # Create a fresh circuit breaker for testing
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=1.0,  # Short timeout for faster tests
            window_size=5,
            success_threshold=2,
            name="test_breaker"
        )
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        assert self.breaker.state == CircuitBreakerState.CLOSED
        assert self.breaker.is_closed
        assert not self.breaker.is_open
        assert not self.breaker.is_half_open
        
        status = self.breaker.get_status()
        assert status["state"] == "closed"
        assert status["consecutive_failures"] == 0
        assert status["metrics"]["total_calls"] == 0
    
    def test_successful_calls(self):
        """Test successful function calls through circuit breaker."""
        def successful_function(x, y=1):
            return x + y
        
        # Make several successful calls
        for i in range(5):
            result = self.breaker.call(successful_function, i, y=2)
            assert result == i + 2
        
        # Check metrics
        status = self.breaker.get_status()
        assert status["metrics"]["total_calls"] == 5
        assert status["metrics"]["successful_calls"] == 5
        assert status["metrics"]["failed_calls"] == 0
        assert status["state"] == "closed"
    
    def test_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        def failing_function():
            raise ValueError("Test failure")
        
        # Make calls up to threshold - 1 (should stay closed)
        for i in range(2):
            with pytest.raises(ValueError):
                self.breaker.call(failing_function)
            assert self.breaker.state == CircuitBreakerState.CLOSED
        
        # One more failure should open the circuit
        with pytest.raises(ValueError):
            self.breaker.call(failing_function)
        assert self.breaker.state == CircuitBreakerState.OPEN
        
        # Subsequent calls should be blocked
        with pytest.raises(CircuitBreakerError):
            self.breaker.call(failing_function)
    
    def test_circuit_breaker_blocking(self):
        """Test that open circuit breaker blocks calls."""
        def test_function():
            return "success"
        
        # Force circuit to open
        self.breaker._state = CircuitBreakerState.OPEN
        self.breaker._last_failure_time = time.time()
        
        # Call should be blocked
        with pytest.raises(CircuitBreakerError) as exc_info:
            self.breaker.call(test_function)
        
        assert "blocking call" in str(exc_info.value)
        assert self.breaker.metrics.blocked_calls == 1
    
    def test_half_open_recovery(self):
        """Test transition from OPEN to HALF_OPEN and recovery."""
        def test_function(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"
        
        # Force circuit to open
        for _ in range(3):
            with pytest.raises(Exception):
                self.breaker.call(test_function, should_fail=True)
        
        assert self.breaker.state == CircuitBreakerState.OPEN
        
        # Wait for reset timeout
        time.sleep(1.1)
        
        # Next call should transition to HALF_OPEN
        result = self.breaker.call(test_function, should_fail=False)
        assert result == "success"
        assert self.breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Another success should close the circuit
        result = self.breaker.call(test_function, should_fail=False)
        assert result == "success"
        assert self.breaker.state == CircuitBreakerState.CLOSED
    
    def test_half_open_failure(self):
        """Test circuit breaker reopening from HALF_OPEN on failure."""
        def test_function(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                self.breaker.call(test_function, should_fail=True)
        
        assert self.breaker.state == CircuitBreakerState.OPEN
        
        # Wait and transition to HALF_OPEN
        time.sleep(1.1)
        self.breaker.call(test_function, should_fail=False)
        assert self.breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Single failure in HALF_OPEN should reopen circuit when threshold=1 for half-open
        # But our breaker has failure_threshold=3, so we need 3 failures
        for _ in range(3):
            with pytest.raises(Exception):
                self.breaker.call(test_function, should_fail=True)
        
        assert self.breaker.state == CircuitBreakerState.OPEN
    
    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        def failing_function():
            raise ValueError("Test failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                self.breaker.call(failing_function)
        
        assert self.breaker.state == CircuitBreakerState.OPEN
        
        # Manual reset
        self.breaker.reset()
        assert self.breaker.state == CircuitBreakerState.CLOSED
        assert self.breaker._consecutive_failures == 0
        
        # Should work normally now
        def working_function():
            return "success"
        
        result = self.breaker.call(working_function)
        assert result == "success"
    
    def test_metrics_tracking(self):
        """Test comprehensive metrics tracking."""
        def mixed_function(fail_number):
            if fail_number in [1, 3]:
                raise Exception(f"Failure {fail_number}")
            return f"Success {fail_number}"
        
        # Make mixed calls
        call_results = []
        for i in range(5):
            try:
                result = self.breaker.call(mixed_function, i)
                call_results.append(("success", result))
            except Exception as e:
                call_results.append(("failure", str(e)))
        
        status = self.breaker.get_status()
        metrics = status["metrics"]
        
        assert metrics["total_calls"] >= 3  # Some may be blocked
        assert metrics["successful_calls"] == 3
        assert metrics["failed_calls"] == 2
        assert 0 <= metrics["failure_rate"] <= 1
        assert 0 <= metrics["success_rate"] <= 1
        assert metrics["last_failure_time"] is not None
        assert metrics["last_success_time"] is not None
    
    def test_context_manager(self):
        """Test circuit breaker as context manager."""
        def test_function():
            return "context_success"
        
        with self.breaker as cb:
            result = cb.call(test_function)
            assert result == "context_success"
        
        # Context manager should not affect state
        assert self.breaker.state == CircuitBreakerState.CLOSED
    
    def test_thread_safety(self):
        """Test circuit breaker thread safety."""
        results = []
        errors = []
        
        def worker_function(worker_id, should_fail=False):
            if should_fail:
                raise Exception(f"Worker {worker_id} failure")
            return f"Worker {worker_id} success"
        
        def worker_thread(worker_id, num_calls, fail_rate=0.3):
            for i in range(num_calls):
                try:
                    should_fail = (i % int(1/fail_rate)) == 0 if fail_rate > 0 else False
                    result = self.breaker.call(worker_function, worker_id, should_fail)
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(
                target=worker_thread, 
                args=(worker_id, 10, 0.2)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check that we got some results and the state is consistent
        assert len(results) > 0
        status = self.breaker.get_status()
        assert status["metrics"]["total_calls"] > 0


class TestCircuitBreakerDecorator:
    """Test suite for circuit breaker decorator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_circuit_breaker_registry()
    
    def test_decorator_basic_usage(self):
        """Test basic decorator usage."""
        @circuit_breaker(name="decorated_function", failure_threshold=2)
        def test_function(value):
            if value < 0:
                raise ValueError("Negative value")
            return value * 2
        
        # Successful calls
        assert test_function(5) == 10
        assert test_function(0) == 0
        
        # Check that circuit breaker is accessible
        assert hasattr(test_function, '_circuit_breaker')
        assert test_function._circuit_breaker.name == "decorated_function"
    
    def test_decorator_failure_handling(self):
        """Test decorator failure handling and circuit opening."""
        @circuit_breaker(failure_threshold=2, reset_timeout=0.5)
        def unreliable_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Simulated failure")
            return "success"
        
        # Successful calls
        assert unreliable_function(False) == "success"
        
        # Fail enough times to open circuit
        with pytest.raises(RuntimeError):
            unreliable_function(True)
        with pytest.raises(RuntimeError):
            unreliable_function(True)
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            unreliable_function(False)
        
        # Wait for reset and test recovery
        time.sleep(0.6)
        assert unreliable_function(False) == "success"
    
    def test_decorator_default_name(self):
        """Test decorator with default name generation."""
        @circuit_breaker()
        def my_function():
            return "test"
        
        breaker = my_function._circuit_breaker
        assert "my_function" in breaker.name


class TestCircuitBreakerRegistry:
    """Test suite for circuit breaker registry functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_circuit_breaker_registry()
    
    def test_get_circuit_breaker(self):
        """Test getting circuit breakers from registry."""
        # First call creates new breaker
        breaker1 = get_circuit_breaker("test_service")
        assert breaker1.name == "test_service"
        
        # Second call returns same breaker
        breaker2 = get_circuit_breaker("test_service")
        assert breaker1 is breaker2
        
        # Different name creates different breaker
        breaker3 = get_circuit_breaker("other_service")
        assert breaker3 is not breaker1
    
    def test_list_circuit_breakers(self):
        """Test listing all circuit breakers."""
        # Initially empty
        breakers = list_circuit_breakers()
        assert len(breakers) == 0
        
        # Add some breakers
        get_circuit_breaker("service1")
        get_circuit_breaker("service2")
        
        breakers = list_circuit_breakers()
        assert len(breakers) == 2
        assert "service1" in breakers
        assert "service2" in breakers
        
        # Check status structure
        assert "state" in breakers["service1"]
        assert "metrics" in breakers["service1"]
    
    def test_reset_all_circuit_breakers(self):
        """Test resetting all circuit breakers."""
        # Create and modify some breakers
        breaker1 = get_circuit_breaker("service1", failure_threshold=1)
        breaker2 = get_circuit_breaker("service2", failure_threshold=1)
        
        # Force them to open
        def failing_function():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            breaker1.call(failing_function)
        with pytest.raises(Exception):
            breaker2.call(failing_function)
        
        assert breaker1.state == CircuitBreakerState.OPEN
        assert breaker2.state == CircuitBreakerState.OPEN
        
        # Reset all
        reset_all_circuit_breakers()
        
        assert breaker1.state == CircuitBreakerState.CLOSED
        assert breaker2.state == CircuitBreakerState.CLOSED
    
    def test_clear_registry(self):
        """Test clearing the circuit breaker registry."""
        # Add some breakers
        get_circuit_breaker("service1")
        get_circuit_breaker("service2")
        
        assert len(list_circuit_breakers()) == 2
        
        # Clear registry
        clear_circuit_breaker_registry()
        
        assert len(list_circuit_breakers()) == 0


class TestCircuitBreakerEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_circuit_breaker_registry()
        self.breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.5)
    
    def test_zero_failure_threshold(self):
        """Test circuit breaker with zero failure threshold."""
        breaker = CircuitBreaker(failure_threshold=0)
        
        def failing_function():
            raise Exception("Immediate failure")
        
        # Should open immediately on first failure
        with pytest.raises(Exception):
            breaker.call(failing_function)
        
        # Should be open now
        with pytest.raises(CircuitBreakerError):
            breaker.call(lambda: "success")
    
    def test_exception_propagation(self):
        """Test that original exceptions are properly propagated."""
        def custom_exception_function():
            raise ValueError("Custom error message")
        
        # Exception should propagate through circuit breaker
        with pytest.raises(ValueError) as exc_info:
            self.breaker.call(custom_exception_function)
        
        assert "Custom error message" in str(exc_info.value)
    
    def test_function_with_return_value(self):
        """Test functions with various return types."""
        def return_dict():
            return {"key": "value", "number": 42}
        
        def return_list():
            return [1, 2, 3]
        
        def return_none():
            return None
        
        # Test different return types
        assert self.breaker.call(return_dict) == {"key": "value", "number": 42}
        assert self.breaker.call(return_list) == [1, 2, 3]
        assert self.breaker.call(return_none) is None
    
    def test_function_with_args_kwargs(self):
        """Test functions with complex argument patterns."""
        def complex_function(a, b, c=None, *args, **kwargs):
            return {
                "a": a,
                "b": b, 
                "c": c,
                "args": args,
                "kwargs": kwargs
            }
        
        result = self.breaker.call(
            complex_function, 
            1, 2,  # a=1, b=2
            c=3,   # c=3 (keyword arg)
            extra1="value1", extra2="value2"  # kwargs
        )
        
        expected = {
            "a": 1,
            "b": 2,
            "c": 3,
            "args": (),  # No additional positional args
            "kwargs": {"extra1": "value1", "extra2": "value2"}
        }
        
        assert result == expected


@pytest.mark.integration
class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with simulated external services."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_circuit_breaker_registry()
    
    def test_simulated_api_failure_recovery(self):
        """Test circuit breaker with simulated API that recovers."""
        call_count = 0
        
        @circuit_breaker(failure_threshold=2, reset_timeout=0.2)  # Faster test
        def simulated_api_call(should_fail=False):
            nonlocal call_count
            call_count += 1
            
            if should_fail:
                raise ConnectionError(f"API down (call #{call_count})")
            
            return f"API response #{call_count}"
        
        # Initial successful calls
        assert simulated_api_call(False) == "API response #1"
        assert simulated_api_call(False) == "API response #2"
        
        # Failure period - should open circuit after 2 failures
        with pytest.raises(ConnectionError):
            simulated_api_call(True)  # Call #3 - fails
        with pytest.raises(ConnectionError):
            simulated_api_call(True)  # Call #4 - fails, opens circuit
        
        # Circuit should be open - calls blocked
        with pytest.raises(CircuitBreakerError):
            simulated_api_call(False)  # Blocked
        
        # Wait for reset timeout
        time.sleep(0.3)
        
        # Should allow recovery attempts now
        assert simulated_api_call(False) == "API response #5"
        assert simulated_api_call(False) == "API response #6"
        
        # Circuit should be closed after successful recovery
        breaker = simulated_api_call._circuit_breaker
        assert breaker.state == CircuitBreakerState.CLOSED


def test_circuit_breaker_smoke():
    """Smoke test to ensure basic circuit breaker functionality works."""
    # Clear any existing state
    clear_circuit_breaker_registry()
    
    # Test basic functionality
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
    
    # Successful call
    result = breaker.call(lambda x: x * 2, 5)
    assert result == 10
    
    # Check status
    status = breaker.get_status()
    assert status["state"] == "closed"
    assert status["metrics"]["successful_calls"] == 1


if __name__ == "__main__":
    # Run basic smoke test
    test_circuit_breaker_smoke()
    print("✅ Circuit breaker smoke test passed")