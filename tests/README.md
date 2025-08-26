# Integration Test Harness Documentation

This directory contains the integration test harness for the Virtual PR Firm content generation pipeline. The harness provides comprehensive testing capabilities for validating end-to-end flow execution without external dependencies.

## Quick Start

Run all integration tests:
```bash
pytest tests/test_integration_harness.py -v
```

Run a specific test:
```bash
pytest tests/test_integration_harness.py::test_core_flow_smoke -v
```

Run the harness directly (includes example execution):
```bash
python tests/test_integration_harness.py
```

## Test Harness Features

### 🔧 `IntegrationTestHarness` Class

The main harness provides utilities for:
- Creating minimal shared state configurations
- Running complete flows end-to-end in memory
- Mocking external dependencies for deterministic testing
- Validating results and extracting metrics

### 🎯 Core Flow Testing

Tests the essential pipeline: **Engagement → BrandBible → PlatformFormatting → ContentCraftsman**

```python
from tests.test_integration_harness import IntegrationTestHarness

# Initialize harness
harness = IntegrationTestHarness()

# Create test scenario
shared = harness.create_minimal_shared_state(
    platforms=["twitter", "linkedin"], 
    topic="AI innovation"
)

# Run flow
result = harness.run_flow(shared)

# Validate results
validation = harness.validate_flow_result(result, ["twitter", "linkedin"])
assert validation["has_content_pieces"]
```

### 🧪 Mock Utilities

The harness includes deterministic mocks for:
- **LLM calls** (`mock_call_llm`) - Returns platform-specific content based on prompt keywords
- **Brand bible parser** (`mock_brand_bible_parser`) - Provides structured brand data
- **Platform formatter** (`mock_format_platform`) - Returns platform-specific guidelines
- **Style checker** (`mock_check_style_violations`) - Validates content quality

## Available Tests

### `test_core_flow_smoke()`
Validates that the core content generation pipeline completes without errors.

**What it tests:**
- Flow execution from start to finish
- Basic shared state processing
- Platform requirement preservation
- Brand bible handling

### `test_multi_platform_flow()`
Tests flow execution with multiple social media platforms.

**What it tests:**
- Multi-platform processing
- Platform list preservation
- Flow completion with complex inputs

### `test_flow_with_brand_bible()`
Tests flow behavior with brand bible XML input.

**What it tests:**
- XML brand bible processing
- Brand constraint handling
- Content generation with brand guidelines

### `test_empty_input_handling()`
Validates graceful handling of minimal/empty inputs.

**What it tests:**
- Defensive programming in nodes
- Fallback behavior
- Error resilience

## Writing New Integration Tests

### Template for Custom Tests

```python
def test_custom_scenario():
    """Custom test for specific scenario."""
    harness = IntegrationTestHarness()
    try:
        # Step 1: Configure test scenario
        shared = harness.create_minimal_shared_state(
            platforms=["your_platforms"],
            topic="your_topic",
            brand_xml="<optional_brand_xml/>"
        )
        
        # Step 2: Customize shared state if needed
        shared["task_requirements"]["urgency_level"] = "high"
        shared["config"]["style_policy"] = "lenient"
        
        # Step 3: Run flow
        result = harness.run_flow(shared)
        
        # Step 4: Validate results
        assert isinstance(result, dict)
        assert "content_pieces" in result
        
        # Step 5: Custom validations
        task_req = result.get("task_requirements", {})
        assert task_req.get("urgency_level") == "high"
        
        print("✓ Custom test passed!")
        
    finally:
        # Step 6: Always clean up
        harness.cleanup()
```

### Advanced Test Patterns

#### Testing Error Scenarios
```python
def test_error_handling():
    harness = IntegrationTestHarness()
    try:
        # Create invalid shared state
        shared = {"invalid": "structure"}
        
        # Flow should handle gracefully
        result = harness.run_flow(shared)
        
        # Validate error handling
        assert isinstance(result, dict)
        # Add specific error handling validations
        
    finally:
        harness.cleanup()
```

#### Testing Specific Node Combinations
```python
def test_specific_nodes():
    harness = IntegrationTestHarness()
    try:
        # Test subset of pipeline
        result = harness.run_core_flow_subset(shared)
        
        # Validate specific node outputs
        assert "brand_bible" in result
        assert "platform_guidelines" in result
        
    finally:
        harness.cleanup()
```

#### Performance Testing
```python
def test_performance():
    harness = IntegrationTestHarness()
    try:
        import time
        
        start_time = time.time()
        result = harness.run_flow(shared)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        
    finally:
        harness.cleanup()
```

## Shared State Structure

The harness creates minimal shared states with this structure:

```python
{
    "task_requirements": {
        "platforms": ["twitter", "linkedin"],           # Target platforms
        "topic_or_goal": "AI innovation",               # Content topic
        "intents_by_platform": {...},                   # Platform-specific intents
        "urgency_level": "normal"                       # Content urgency
    },
    "brand_bible": {
        "xml_raw": "<brand_bible>...</brand_bible>",    # Raw XML input
        "parsed": {},                                   # Parsed brand data
        "persona_voice": {},                            # Voice guidelines
        "preset_name": None                             # Preset identifier
    },
    "house_style": {
        "email_signature": {"name": None, "content": ""}, # Email templates
        "blog_style": {"name": None, "content": ""}       # Blog templates
    },
    "platform_nuance": {},                             # Platform-specific settings
    "content_pieces": {},                               # Generated content
    "workflow_state": {
        "current_step": "engagement",                    # Pipeline progress
        "revision_count": 0,                            # Edit iterations
        "errors": []                                    # Error tracking
    },
    "stream": None,                                     # Real-time updates
    "config": {"style_policy": "strict"}               # Global configuration
}
```

## Test Validation Methods

### `validate_flow_result(result, expected_platforms)`

Returns comprehensive validation data:

```python
{
    "has_content_pieces": bool,                    # Content was generated
    "has_task_requirements": bool,                 # Requirements preserved
    "has_brand_bible": bool,                       # Brand data processed
    "platform_coverage": {                        # Per-platform results
        "twitter": bool,
        "linkedin": bool
    },
    "content_quality": {                           # Quality metrics
        "twitter": {
            "has_text": bool,
            "length": int,
            "not_empty": bool
        }
    },
    "errors": []                                   # Validation errors
}
```

## Best Practices

### ✅ Do
- Always use the harness context manager pattern with try/finally
- Call `harness.cleanup()` in the finally block
- Test realistic scenarios with valid platform combinations
- Use descriptive test names that explain the scenario
- Include both positive and negative test cases
- Validate both structure and content of results

### ❌ Don't
- Forget to call `cleanup()` - this can cause test interference
- Assume external services are available - the harness mocks them
- Test implementation details - focus on behavior and outcomes
- Create tests that depend on specific content generation (it's mocked)
- Ignore error conditions - test both success and failure paths

## Debugging Failed Tests

### Common Issues

1. **ImportError**: Ensure all project dependencies are in the Python path
2. **AssertionError**: Check that expectations match the fallback implementations
3. **AttributeError**: Verify shared state structure matches expected format
4. **Test hanging**: Ensure `cleanup()` is called properly

### Debug Mode

Run tests with verbose output:
```bash
pytest tests/test_integration_harness.py -v -s
```

Add debug prints to see intermediate state:
```python
print(f"Shared state keys: {list(shared.keys())}")
print(f"Result structure: {list(result.keys())}")
```

## Contributing

When adding new integration tests:

1. Follow the template pattern shown above
2. Add docstrings explaining what the test validates
3. Include both positive and edge case scenarios
4. Update this documentation with new test descriptions
5. Ensure tests are deterministic and don't rely on external services

## CI/CD Integration

These tests are designed to run in CI/CD pipelines without external dependencies:

```yaml
# Example GitHub Actions step
- name: Run Integration Tests
  run: |
    cd /path/to/project
    python -m pytest tests/test_integration_harness.py -v
```

The tests use mocked utilities and fallback implementations, making them suitable for automated testing environments.