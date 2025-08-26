"""
Comprehensive tests for structured logging functionality.

Tests cover:
- Basic structured logging
- Node lifecycle logging (prep, exec, post)
- Retry logging
- Fallback logging
- Different log levels and formats
- Correlation ID tracking
"""
import os
import sys
import json
import logging
import subprocess
from io import StringIO

# Set environment for JSON structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocketflow import Node
from logging_mixin import LoggingMixin
from logging_config import get_node_logger, StructuredFormatter


class MockNode(LoggingMixin, Node):
    """Mock node for testing basic logging."""
    
    def exec(self, prep_res):
        self.node_logger.info("Mock exec operation")
        return "mock_result"
    
    def post(self, shared, prep_res, exec_res):
        shared["result"] = exec_res
        # Call super() to ensure logging happens
        return super().post(shared, prep_res, exec_res) or "default"


class RetryNode(LoggingMixin, Node):
    """Node that fails and retries for testing retry logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(max_retries=3, wait=0, *args, **kwargs)
        self.attempt_count = 0
    
    def exec(self, prep_res):
        self.attempt_count += 1
        if self.attempt_count <= 2:
            raise Exception(f"Simulated failure on attempt {self.attempt_count}")
        return f"Success on attempt {self.attempt_count}"


class FallbackNode(LoggingMixin, Node):
    """Node that always fails to test fallback logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(max_retries=2, wait=0, *args, **kwargs)
    
    def exec(self, prep_res):
        raise Exception("Always fails")
    
    def exec_fallback(self, prep_res, exc):
        return "fallback_result"


def capture_logs(func, *args, **kwargs):
    """Helper to capture logs from a function call."""
    log_capture = StringIO()
    pocketflow_logger = logging.getLogger("pocketflow")
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(StructuredFormatter())
    pocketflow_logger.addHandler(handler)
    
    try:
        result = func(*args, **kwargs)
        return result, log_capture.getvalue()
    finally:
        pocketflow_logger.removeHandler(handler)


def parse_log_lines(log_content):
    """Parse log content into structured log entries."""
    log_lines = [line.strip() for line in log_content.strip().split('\n') if line.strip()]
    parsed_logs = []
    
    for line in log_lines:
        try:
            parsed_logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    return parsed_logs


def test_basic_logging():
    """Test basic node logging functionality."""
    print("Testing basic logging...")
    
    def run_node():
        node = MockNode()
        node.set_correlation_id("test-basic-123")
        shared = {}
        return node.run(shared)
    
    action, log_content = capture_logs(run_node)
    logs = parse_log_lines(log_content)
    
    # Verify we got logs
    assert len(logs) > 0, "No logs were generated"
    
    # Verify structure
    for log in logs:
        assert "timestamp" in log
        assert "level" in log
        assert "message" in log
        assert "node" in log
        assert log["node"] == "MockNode"
        assert "correlation_id" in log
        assert log["correlation_id"] == "test-basic-123"
    
    # Verify lifecycle events
    actions = [log.get("action") for log in logs if log.get("action")]
    assert "prep" in actions, f"prep action not found in {actions}"
    assert "exec" in actions, f"exec action not found in {actions}"
    assert "post" in actions, f"post action not found in {actions}"
    
    print("✅ Basic logging test passed")


def test_retry_logging():
    """Test retry logging functionality."""
    print("Testing retry logging...")
    
    def run_retry_node():
        node = RetryNode()
        node.set_correlation_id("test-retry-456")
        shared = {}
        return node.run(shared)
    
    action, log_content = capture_logs(run_retry_node)
    logs = parse_log_lines(log_content)
    
    # Count retry logs
    retry_logs = [log for log in logs if log.get("action") == "exec_retry"]
    assert len(retry_logs) >= 2, f"Expected at least 2 retry logs, got {len(retry_logs)}"
    
    # Verify retry log structure
    for retry_log in retry_logs:
        assert "retry_count" in retry_log
        assert "error" in retry_log
        assert retry_log["level"] == "WARNING"
    
    print("✅ Retry logging test passed")


def test_fallback_logging():
    """Test fallback logging functionality.""" 
    print("Testing fallback logging...")
    
    def run_fallback_node():
        node = FallbackNode()
        node.set_correlation_id("test-fallback-789")
        shared = {}
        return node.run(shared)
    
    action, log_content = capture_logs(run_fallback_node)
    logs = parse_log_lines(log_content)
    
    # Find fallback logs
    fallback_logs = [log for log in logs if log.get("action") == "exec_fallback"]
    assert len(fallback_logs) >= 1, f"Expected at least 1 fallback log, got {len(fallback_logs)}"
    
    # Verify fallback log structure
    for fallback_log in fallback_logs:
        assert "retry_count" in fallback_log
        assert "error" in fallback_log
        assert fallback_log["level"] == "ERROR"
    
    print("✅ Fallback logging test passed")


def test_correlation_id_tracking():
    """Test correlation ID tracking across nodes."""
    print("Testing correlation ID tracking...")
    
    correlation_id = "test-correlation-tracking-999"
    
    def run_multiple_nodes():
        node1 = MockNode()
        node2 = MockNode()
        
        node1.set_correlation_id(correlation_id)
        node2.set_correlation_id(correlation_id)
        
        shared = {}
        node1.run(shared)
        node2.run(shared)
    
    _, log_content = capture_logs(run_multiple_nodes)
    logs = parse_log_lines(log_content)
    
    # Verify all logs have the same correlation ID
    correlation_ids = [log.get("correlation_id") for log in logs]
    assert all(cid == correlation_id for cid in correlation_ids), "Correlation ID not consistent"
    
    print("✅ Correlation ID tracking test passed")


def test_different_log_levels():
    """Test different log levels work correctly."""
    print("Testing different log levels...")
    
    # Test with a logger directly
    logger = get_node_logger("TestLevels", "test-levels-111")
    
    def test_with_level(level):
        # Temporarily change the log level
        old_level = logging.getLogger("pocketflow").level
        logging.getLogger("pocketflow").setLevel(getattr(logging, level))
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(StructuredFormatter())
        pocketflow_logger = logging.getLogger("pocketflow")
        pocketflow_logger.addHandler(handler)
        
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message") 
            logger.error("Error message")
            return log_capture.getvalue()
        finally:
            pocketflow_logger.removeHandler(handler)
            logging.getLogger("pocketflow").setLevel(old_level)
    
    # Test INFO level (should see INFO, WARNING, ERROR)
    info_logs = test_with_level("INFO")
    info_parsed = parse_log_lines(info_logs)
    info_levels = [log["level"] for log in info_parsed]
    assert "INFO" in info_levels
    assert "WARNING" in info_levels
    assert "ERROR" in info_levels
    assert "DEBUG" not in info_levels
    
    # Test ERROR level (should only see ERROR)
    error_logs = test_with_level("ERROR")
    error_parsed = parse_log_lines(error_logs)
    error_levels = [log["level"] for log in error_parsed]
    assert "ERROR" in error_levels
    assert "INFO" not in error_levels
    assert "WARNING" not in error_levels
    
    print("✅ Different log levels test passed")


def test_text_format():
    """Test text format logging by calling subprocess."""
    print("Testing text format...")
    
    # Run test_text_format.py and check output
    result = subprocess.run([
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "test_text_format.py")
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Text format test failed: {result.stderr}"
    
    # Verify it's not JSON format
    assert '{"timestamp"' not in result.stderr, "Output should not be JSON format"
    
    # Verify it contains expected text format elements
    assert "pocketflow.node.TestTextNode" in result.stderr, "Should contain logger name"
    assert "INFO" in result.stderr, "Should contain log level"
    
    print("✅ Text format test passed")


def run_all_tests():
    """Run all logging tests."""
    print("Running structured logging tests...\n")
    
    tests = [
        test_basic_logging,
        test_retry_logging,
        test_fallback_logging,
        test_correlation_id_tracking,
        test_different_log_levels,
        test_text_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 All tests passed!")


if __name__ == "__main__":
    run_all_tests()