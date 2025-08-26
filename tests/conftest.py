"""Pytest configuration and fixtures for the test suite.

This module provides common fixtures and configuration for all tests
in the project, ensuring consistent test setup and teardown.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List


@pytest.fixture
def sample_shared_state():
    """Provide a sample shared state for testing."""
    return {
        "task_requirements": {
            "platforms": ["twitter", "linkedin", "facebook"],
            "topic_or_goal": "Test content generation",
            "intents_by_platform": {
                "twitter": "engagement",
                "linkedin": "professional",
                "facebook": "community"
            }
        },
        "brand_bible": {
            "xml_raw": """
            <brand_bible>
                <voice>
                    <tone>Professional</tone>
                    <style>Formal</style>
                </voice>
                <guidelines>
                    <rule>Use clear language</rule>
                    <rule>Avoid jargon</rule>
                </guidelines>
            </brand_bible>
            """
        },
        "stream": None
    }


@pytest.fixture
def sample_brand_bible():
    """Provide a sample brand bible for testing."""
    return {
        "voice": {
            "tone": "Professional",
            "style": "Formal",
            "personality": "Innovative"
        },
        "guidelines": {
            "rule": [
                "Use clear language",
                "Avoid jargon",
                "Maintain professional tone"
            ]
        },
        "content_guidelines": {
            "do": [
                "Use active voice",
                "Keep it simple",
                "Be concise"
            ],
            "dont": [
                "Avoid technical jargon",
                "Don't be overly formal",
                "Don't use passive voice"
            ]
        }
    }


@pytest.fixture
def sample_style_guidelines():
    """Provide sample style guidelines for testing."""
    return {
        "tone": "Professional",
        "style": "Formal",
        "rules": [
            "Use clear language",
            "Avoid jargon",
            "Maintain professional tone",
            "Use active voice",
            "Keep sentences under 25 words"
        ],
        "avoid_words": ["jargon", "technical", "complex"],
        "target_audience": "professionals"
    }


@pytest.fixture
def sample_content():
    """Provide sample content for testing."""
    return {
        "original": "This is the original content that needs processing.",
        "formatted": {
            "twitter": "Original content for Twitter.",
            "linkedin": "Original content for LinkedIn.",
            "facebook": "Original content for Facebook."
        },
        "styled": "This is styled content that follows guidelines.",
        "violations": [
            "Content uses technical jargon",
            "Sentences are too long",
            "Tone is too informal"
        ]
    }


@pytest.fixture
def mock_llm_response():
    """Provide a mock LLM response for testing."""
    return "This is a mock response from the LLM that simulates content generation."


@pytest.fixture
def temp_config_file():
    """Provide a temporary configuration file for testing."""
    config_data = {
        "default_platforms": ["twitter", "linkedin"],
        "default_topic": "Test topic",
        "logging": {
            "level": "INFO",
            "file": None
        },
        "web_interface": {
            "port": 7860,
            "host": "0.0.0.0",
            "share": False
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    try:
        os.unlink(temp_file)
    except OSError:
        pass


@pytest.fixture
def mock_stream():
    """Provide a mock stream for testing streaming functionality."""
    stream = MagicMock()
    stream.emit = MagicMock()
    return stream


@pytest.fixture
def sample_platforms():
    """Provide sample platform lists for testing."""
    return {
        "valid": ["twitter", "linkedin", "facebook", "instagram"],
        "invalid": ["invalid@platform", "platform with spaces", "PLATFORM"],
        "mixed": ["twitter", "invalid@platform", "linkedin", "PLATFORM"],
        "empty": [],
        "none": None
    }


@pytest.fixture
def sample_topics():
    """Provide sample topics for testing."""
    return {
        "valid": "Test content generation for social media platforms",
        "empty": "",
        "long": "A" * 1000,
        "special_chars": "Topic with émojis 🚀 and special chars: @#$%^&*()",
        "none": None
    }


@pytest.fixture
def mock_openai_client():
    """Provide a mock OpenAI client for testing."""
    with patch('utils.call_llm.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mock OpenAI response"
        mock_client.chat.completions.create.return_value = mock_response
        
        yield mock_client


@pytest.fixture
def mock_openrouter_client():
    """Provide a mock OpenRouter client for testing."""
    with patch('utils.openrouter_client.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Mock OpenRouter response'}}]
        }
        mock_post.return_value = mock_response
        
        yield mock_post


@pytest.fixture
def sample_xml_content():
    """Provide sample XML content for testing."""
    return {
        "valid": """
        <brand_bible>
            <company_info>
                <name>Test Company</name>
                <industry>Technology</industry>
            </company_info>
            <voice>
                <tone>Professional</tone>
                <style>Formal</style>
                <personality>Innovative</personality>
            </voice>
            <content_guidelines>
                <do>
                    <item>Use active voice</item>
                    <item>Keep it simple</item>
                </do>
                <dont>
                    <item>Avoid technical jargon</item>
                    <item>Don't be overly formal</item>
                </dont>
            </content_guidelines>
        </brand_bible>
        """,
        "invalid": "<brand_bible><unclosed_tag>",
        "empty": "",
        "none": None,
        "non_xml": "This is not XML content"
    }


@pytest.fixture
def sample_constraints():
    """Provide sample constraints for testing."""
    return {
        "basic": {
            "tone": "Professional",
            "style": "Formal",
            "max_length": 100
        },
        "complex": {
            "tone": "Professional",
            "style": "Formal",
            "max_length": 150,
            "avoid_words": ["jargon", "technical"],
            "include_keywords": ["innovation", "quality"],
            "target_audience": "executives",
            "sentence_length": 25,
            "paragraph_length": 5
        },
        "empty": {},
        "none": None
    }


@pytest.fixture
def mock_flow():
    """Provide a mock flow for testing."""
    flow = MagicMock()
    flow.run = MagicMock()
    flow.start = MagicMock()
    return flow


@pytest.fixture
def mock_node():
    """Provide a mock node for testing."""
    node = MagicMock()
    node.prep = MagicMock()
    node.exec = MagicMock()
    node.post = MagicMock()
    node.run = MagicMock()
    return node


@pytest.fixture
def sample_error_messages():
    """Provide sample error messages for testing."""
    return {
        "validation": "Validation failed",
        "api": "API error occurred",
        "network": "Network connection failed",
        "timeout": "Request timed out",
        "rate_limit": "Rate limit exceeded",
        "authentication": "Authentication failed",
        "permission": "Permission denied",
        "not_found": "Resource not found",
        "server_error": "Internal server error",
        "unknown": "Unknown error occurred"
    }


@pytest.fixture
def sample_log_messages():
    """Provide sample log messages for testing."""
    return {
        "info": "Information message",
        "warning": "Warning message",
        "error": "Error message",
        "debug": "Debug message",
        "critical": "Critical message"
    }


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    logger.critical = MagicMock()
    return logger


@pytest.fixture
def sample_test_data():
    """Provide comprehensive test data for integration tests."""
    return {
        "shared_state": {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "AI in healthcare",
                "intents_by_platform": {
                    "twitter": "engagement",
                    "linkedin": "professional"
                }
            },
            "brand_bible": {
                "xml_raw": """
                <brand_bible>
                    <voice>
                        <tone>Professional</tone>
                        <style>Formal</style>
                    </voice>
                    <guidelines>
                        <rule>Use clear language</rule>
                        <rule>Avoid jargon</rule>
                    </guidelines>
                </brand_bible>
                """
            },
            "stream": None
        },
        "expected_output": {
            "formatted_content": {
                "twitter": "AI in healthcare: transforming patient care",
                "linkedin": "The impact of AI on healthcare delivery and patient outcomes"
            },
            "style_violations": [],
            "final_content": "AI in healthcare is revolutionizing patient care delivery."
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests with "test_" prefix as unit tests
        if item.name.startswith("test_"):
            item.add_marker(pytest.mark.unit)
        
        # Mark integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark API tests
        if "api" in item.name.lower() or "llm" in item.name.lower():
            item.add_marker(pytest.mark.api)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["slow", "timeout", "long"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    
    # Setup any test-specific configuration
    yield
    
    # Cleanup after each test
    # Remove test environment variables
    os.environ.pop('TESTING', None)


@pytest.fixture
def mock_file_system():
    """Provide a mock file system for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = {
            "config.json": json.dumps({"test": "config"}),
            "brand_bible.xml": "<brand_bible><voice><tone>Professional</tone></voice></brand_bible>",
            "content.txt": "Test content for file operations"
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        yield temp_dir


@pytest.fixture
def sample_error_scenarios():
    """Provide sample error scenarios for testing error handling."""
    return {
        "network_error": {
            "exception": ConnectionError("Network connection failed"),
            "expected_behavior": "should handle gracefully"
        },
        "api_error": {
            "exception": Exception("API rate limit exceeded"),
            "expected_behavior": "should retry with backoff"
        },
        "validation_error": {
            "exception": ValueError("Invalid input data"),
            "expected_behavior": "should return error message"
        },
        "timeout_error": {
            "exception": TimeoutError("Request timed out"),
            "expected_behavior": "should retry or fail gracefully"
        },
        "permission_error": {
            "exception": PermissionError("Access denied"),
            "expected_behavior": "should log and fail gracefully"
        }
    }


@pytest.fixture
def mock_async_context():
    """Provide mock async context for testing async functionality."""
    async def mock_async_function():
        return "async result"
    
    return mock_async_function


@pytest.fixture
def sample_performance_metrics():
    """Provide sample performance metrics for testing."""
    return {
        "response_time": 1.5,
        "memory_usage": 1024,
        "cpu_usage": 25.5,
        "throughput": 100,
        "error_rate": 0.01,
        "success_rate": 0.99
    }


# Custom pytest hooks for better test reporting
def pytest_runtest_logreport(report):
    """Custom test reporting hook."""
    if report.when == "call" and report.failed:
        # Log failed test details
        print(f"\nTest {report.nodeid} failed:")
        print(f"Duration: {report.duration:.2f}s")
        if hasattr(report, 'longrepr'):
            print(f"Error: {report.longrepr}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom terminal summary hook."""
    # Add custom summary information
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    # Count test results
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {passed + failed + skipped}")
    
    if failed > 0:
        print("\nFailed tests:")
        for report in terminalreporter.stats.get('failed', []):
            print(f"  - {report.nodeid}")
    
    print("="*50)