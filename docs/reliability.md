# Reliability Guide: Circuit Breaker Pattern Implementation

This document describes the circuit breaker pattern implementation for preventing cascading failures in the Virtual PR Firm system.

## Overview

The circuit breaker pattern is a resilience pattern that prevents cascading failures by monitoring external service calls and temporarily blocking requests when failure rates exceed configured thresholds. Our implementation provides automatic protection for all external dependencies including LLM APIs and streaming services.

## Circuit Breaker States

The circuit breaker operates in three states:

### CLOSED (Normal Operation)
- All calls are allowed to pass through
- Failures are counted but don't block subsequent calls
- Transitions to OPEN when failure threshold is reached

### OPEN (Blocking Mode)
- All calls are immediately blocked with `CircuitBreakerError`
- No requests are sent to the failing service
- Transitions to HALF_OPEN after reset timeout expires

### HALF_OPEN (Recovery Testing)
- Limited calls are allowed to test service recovery
- Success leads to CLOSED state
- Failure leads back to OPEN state

## Configuration Options

### Core Settings

```python
CircuitBreaker(
    failure_threshold=5,    # Number of failures before opening circuit
    reset_timeout=60.0,     # Seconds to wait before testing recovery
    window_size=10,         # Number of recent calls to track
    success_threshold=2,    # Successes needed in HALF_OPEN to close circuit
    name="my_service"       # Optional name for identification
)
```

### Recommended Settings by Service Type

#### LLM API Calls
```python
failure_threshold=3      # Fail quickly due to high cost per call
reset_timeout=60.0       # 1 minute to allow for service recovery
window_size=10           # Track recent history for failure patterns
success_threshold=2      # Require 2 successes before trusting service
```

#### Streaming Services
```python
failure_threshold=5      # Allow more failures for streaming
reset_timeout=30.0       # Shorter recovery time for real-time services
window_size=20           # Larger window for stream fluctuations
success_threshold=3      # Higher confidence required for streams
```

#### Vector Database Operations
```python
failure_threshold=2      # DB failures should be treated seriously
reset_timeout=120.0      # Longer recovery time for database issues
window_size=15           # Medium window for DB call patterns
success_threshold=2      # Standard recovery requirement
```

## Usage Patterns

### 1. Decorator Pattern (Recommended)

The simplest way to protect functions:

```python
from utils.circuit_breaker import circuit_breaker

@circuit_breaker(
    name="openai_calls",
    failure_threshold=3,
    reset_timeout=60.0
)
def call_openai_api(prompt):
    # Your OpenAI API call here
    return openai_client.create(prompt=prompt)

# Use normally - circuit breaker is transparent
try:
    result = call_openai_api("Hello world")
except CircuitBreakerError:
    # Handle circuit breaker blocking the call
    result = fallback_response()
except Exception:
    # Handle actual API errors
    result = error_response()
```

### 2. Context Manager Pattern

For more control over individual calls:

```python
from utils.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60)

with breaker as cb:
    try:
        result = cb.call(risky_api_function, arg1, arg2)
    except CircuitBreakerError:
        result = fallback_function(arg1, arg2)
```

### 3. Direct Call Pattern

For maximum flexibility:

```python
breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60)

try:
    result = breaker.call(external_service, param1, param2)
except CircuitBreakerError:
    # Service is down, use cached result or alternative
    result = get_cached_result(param1, param2)
```

### 4. Shared Circuit Breakers

Use the registry for consistent behavior across modules:

```python
from utils.circuit_breaker import get_circuit_breaker

# Get or create a shared circuit breaker
llm_breaker = get_circuit_breaker(
    "llm_service",
    failure_threshold=3,
    reset_timeout=60
)

# Use across different modules
result = llm_breaker.call(api_function, args)
```

## Monitoring and Observability

### Status Checking

```python
# Get current circuit breaker status
status = breaker.get_status()
print(f"State: {status['state']}")
print(f"Total calls: {status['metrics']['total_calls']}")
print(f"Failure rate: {status['metrics']['failure_rate']:.2%}")
```

### Registry Management

```python
from utils.circuit_breaker import (
    list_circuit_breakers,
    reset_all_circuit_breakers,
    clear_circuit_breaker_registry
)

# List all circuit breakers and their status
all_breakers = list_circuit_breakers()
for name, status in all_breakers.items():
    print(f"{name}: {status['state']} ({status['metrics']['total_calls']} calls)")

# Reset all circuit breakers (emergency recovery)
reset_all_circuit_breakers()
```

### Metrics Collection

Key metrics tracked automatically:

- **total_calls**: Total number of attempted calls
- **successful_calls**: Number of successful calls
- **failed_calls**: Number of failed calls
- **blocked_calls**: Number of calls blocked by circuit breaker
- **failure_rate**: Current failure rate (0.0 to 1.0)
- **success_rate**: Current success rate (0.0 to 1.0)
- **state_changes**: Number of state transitions
- **last_failure_time**: Timestamp of last failure
- **last_success_time**: Timestamp of last success

## Integration with Existing Services

### LLM Services Protection

The circuit breaker is automatically applied to:

- `utils.call_llm._call_openai()` - OpenAI API calls
- `utils.call_llm._call_anthropic()` - Anthropic API calls  
- `utils.openrouter_client._make_request()` - OpenRouter API calls
- `utils.openrouter_client._stream_response()` - OpenRouter streaming

Circuit breaker errors are handled gracefully with fallback to mock responses.

### Node-Level Protection

Apply circuit breaker protection in nodes for external dependencies:

```python
from utils.circuit_breaker import circuit_breaker

class ContentCraftsmanNode(Node):
    
    @circuit_breaker(name="content_generation", failure_threshold=3)
    def _generate_content_with_llm(self, prompt, platform):
        """Protected content generation with circuit breaker."""
        return call_llm(prompt=prompt, temperature=0.7)
    
    def exec(self, inputs):
        try:
            content = self._generate_content_with_llm(prompt, platform)
        except CircuitBreakerError:
            # Use fallback content generation
            content = self._generate_fallback_content(inputs)
        return {"content": content}
```

## Best Practices

### 1. Choose Appropriate Thresholds

- **Low-cost operations**: Higher failure thresholds (5-10)
- **High-cost operations**: Lower failure thresholds (2-3)
- **Critical services**: Lower failure thresholds (1-2)
- **Best-effort services**: Higher failure thresholds (10+)

### 2. Set Reasonable Timeouts

- **Fast services**: 15-30 seconds
- **Normal services**: 60-120 seconds
- **Slow services**: 300+ seconds
- **Database services**: 120-300 seconds

### 3. Implement Proper Fallbacks

Always provide meaningful fallback behavior:

```python
@circuit_breaker(name="llm_service")
def generate_content(prompt):
    return expensive_llm_call(prompt)

def safe_generate_content(prompt):
    try:
        return generate_content(prompt)
    except CircuitBreakerError:
        # Provide meaningful fallback
        return template_based_generation(prompt)
    except Exception:
        # Handle other errors
        return error_fallback()
```

### 4. Monitor Circuit Breaker Health

Implement monitoring for circuit breaker state changes:

```python
import logging

def monitor_circuit_breakers():
    """Log circuit breaker status for monitoring."""
    breakers = list_circuit_breakers()
    for name, status in breakers.items():
        if status['state'] != 'closed':
            logging.warning(f"Circuit breaker {name} is {status['state']}")
        
        failure_rate = status['metrics']['failure_rate']
        if failure_rate > 0.1:  # 10% failure rate
            logging.warning(f"High failure rate for {name}: {failure_rate:.2%}")
```

### 5. Handle Circuit Breaker Errors Gracefully

```python
def robust_api_call(data):
    """Example of robust API calling with circuit breaker."""
    try:
        return protected_api_call(data)
    except CircuitBreakerError as e:
        # Circuit is open - service is down
        logging.warning(f"Service unavailable: {e}")
        return get_cached_result(data) or default_response()
    except Exception as e:
        # Actual service error
        logging.error(f"Service error: {e}")
        return error_response()
```

## Testing Circuit Breaker Behavior

### Unit Testing

```python
import pytest
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerError

def test_circuit_breaker_opens_on_failures():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
    
    def failing_function():
        raise Exception("Service error")
    
    # First failure
    with pytest.raises(Exception):
        breaker.call(failing_function)
    
    # Second failure - should open circuit
    with pytest.raises(Exception):
        breaker.call(failing_function)
    
    # Circuit should be open now
    with pytest.raises(CircuitBreakerError):
        breaker.call(lambda: "success")
```

### Integration Testing

```python
def test_llm_circuit_breaker_integration():
    """Test circuit breaker with actual LLM integration."""
    # Mock failing LLM calls
    with patch('utils.call_llm._call_openai') as mock_openai:
        mock_openai.side_effect = Exception("API Error")
        
        # Should eventually use fallback
        result = call_llm("test prompt")
        assert "mock response" in result.lower()
```

### Manual Testing

```python
# Test circuit breaker behavior manually
from utils.circuit_breaker import test_circuit_breaker
test_circuit_breaker()
```

## Troubleshooting

### Common Issues

#### Circuit Breaker Won't Open
- Check if failure threshold is too high
- Verify that exceptions are actually being raised
- Ensure the same circuit breaker instance is being used

#### Circuit Breaker Won't Close
- Check if success threshold is too high
- Verify that recovery calls are actually succeeding
- Check reset timeout is not too long

#### Fallbacks Not Working
- Ensure CircuitBreakerError is caught separately from other exceptions
- Verify fallback logic is implemented correctly
- Check that fallback doesn't depend on the same failing service

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger('utils.circuit_breaker').setLevel(logging.DEBUG)

# Check circuit breaker state
breaker = get_circuit_breaker("my_service")
print(breaker.get_status())

# Manual reset for testing
breaker.reset()
```

### Performance Considerations

- Circuit breakers add minimal overhead (< 1ms per call)
- Thread-safe operations use efficient RLock
- Metrics collection has minimal memory footprint
- Registry lookup is O(1) operation

## Security Considerations

### API Key Protection
- Circuit breakers help prevent API key exhaustion
- Blocked calls don't consume rate limits
- Automatic backoff reduces service load

### Error Information
- Circuit breaker logs don't expose sensitive data
- Error messages are generic and safe
- Metrics don't include request/response content

## Migration Guide

### Existing Code Integration

1. **Import the circuit breaker**:
   ```python
   from utils.circuit_breaker import circuit_breaker
   ```

2. **Add decorator to external calls**:
   ```python
   @circuit_breaker(name="my_service")
   def my_external_call():
       # existing code
   ```

3. **Add error handling**:
   ```python
   try:
       result = my_external_call()
   except CircuitBreakerError:
       result = fallback_function()
   ```

### Gradual Rollout Strategy

1. Start with non-critical services
2. Use high failure thresholds initially
3. Monitor and adjust thresholds based on observed behavior
4. Gradually apply to more critical services
5. Implement comprehensive fallback strategies

## Conclusion

The circuit breaker pattern provides essential resilience for external dependencies in the Virtual PR Firm system. By implementing proper configuration, monitoring, and fallback strategies, the system can gracefully handle service failures and maintain availability even when external services are experiencing issues.

Regular monitoring of circuit breaker metrics and proactive adjustment of thresholds ensures optimal protection against cascading failures while maintaining system responsiveness.