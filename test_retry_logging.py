"""
Test retry logging functionality.
"""
import os
import sys
import logging
from io import StringIO

# Set environment for structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

# Add current directory to path
sys.path.insert(0, '/home/runner/work/prfirm3/prfirm3')

from pocketflow import Node
from logging_mixin import LoggingMixin
from logging_config import StructuredFormatter

class RetryTestNode(LoggingMixin, Node):
    def __init__(self, *args, **kwargs):
        super().__init__(max_retries=3, wait=0, *args, **kwargs)  # 3 retries, no wait
        self.attempt_count = 0
    
    def exec(self, prep_res):
        self.attempt_count += 1
        if self.attempt_count <= 2:  # Fail first 2 attempts
            raise Exception(f"Simulated failure on attempt {self.attempt_count}")
        return f"Success on attempt {self.attempt_count}"
    
    def exec_fallback(self, prep_res, exc):
        return f"Fallback result after {self.cur_retry + 1} attempts"

def test_retry_logging():
    """Test that retry events are logged correctly."""
    print("Testing retry logging functionality...")
    
    # Create a string buffer to capture logs
    log_capture = StringIO()
    
    # Get the pocketflow logger and add our handler
    pocketflow_logger = logging.getLogger("pocketflow")
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(StructuredFormatter())
    pocketflow_logger.addHandler(handler)
    
    try:
        # Create retry test node
        retry_node = RetryTestNode()
        retry_node.set_correlation_id("retry-test-correlation")
        
        # Run the node (should succeed on 3rd attempt)
        shared = {}
        result = retry_node.run(shared)
        print(f"RetryTestNode result: {result}")
        
    finally:
        # Remove our handler
        pocketflow_logger.removeHandler(handler)
    
    # Analyze log output
    log_content = log_capture.getvalue()
    log_lines = [line.strip() for line in log_content.strip().split('\n') if line.strip()]
    print(f"\nCaptured {len(log_lines)} log lines:")
    
    retry_logs = 0
    for i, line in enumerate(log_lines):
        try:
            log_entry = json.loads(line)
            print(f"Log {i+1}: {log_entry}")
            
            if log_entry.get("action") == "exec_retry":
                retry_logs += 1
                assert "retry_count" in log_entry
                # Error is stored directly in the log entry, not in context
                assert "error" in log_entry
                
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in log line {i+1}: {line}")
    
    print(f"\nFound {retry_logs} retry log entries")
    assert retry_logs > 0, "Expected retry logs but found none"
    print("✅ Retry logging test passed!")

if __name__ == "__main__":
    import json
    test_retry_logging()