"""
Test fallback logging functionality.
"""
import os
import sys
import logging
import json
from io import StringIO

# Set environment for structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

# Add current directory to path
sys.path.insert(0, '/home/runner/work/prfirm3/prfirm3')

from pocketflow import Node
from logging_mixin import LoggingMixin
from logging_config import StructuredFormatter

class FallbackTestNode(LoggingMixin, Node):
    def __init__(self, *args, **kwargs):
        super().__init__(max_retries=2, wait=0, *args, **kwargs)  # 2 retries, no wait
    
    def exec(self, prep_res):
        # Always fail to trigger fallback
        raise Exception("Simulated failure for fallback test")
    
    def exec_fallback(self, prep_res, exc):
        return "Fallback result"

def test_fallback_logging():
    """Test that fallback events are logged correctly."""
    print("Testing fallback logging functionality...")
    
    # Create a string buffer to capture logs
    log_capture = StringIO()
    
    # Get the pocketflow logger and add our handler
    pocketflow_logger = logging.getLogger("pocketflow")
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(StructuredFormatter())
    pocketflow_logger.addHandler(handler)
    
    try:
        # Create fallback test node
        fallback_node = FallbackTestNode()
        fallback_node.set_correlation_id("fallback-test-correlation")
        
        # Run the node (should trigger fallback)
        shared = {}
        result = fallback_node.run(shared)
        print(f"FallbackTestNode result: {result}")
        
    finally:
        # Remove our handler
        pocketflow_logger.removeHandler(handler)
    
    # Analyze log output
    log_content = log_capture.getvalue()
    log_lines = [line.strip() for line in log_content.strip().split('\n') if line.strip()]
    print(f"\nCaptured {len(log_lines)} log lines:")
    
    fallback_logs = 0
    retry_logs = 0
    for i, line in enumerate(log_lines):
        try:
            log_entry = json.loads(line)
            print(f"Log {i+1}: {log_entry}")
            
            if log_entry.get("action") == "exec_fallback":
                fallback_logs += 1
                assert "retry_count" in log_entry
                assert "error" in log_entry
            elif log_entry.get("action") == "exec_retry":
                retry_logs += 1
                
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in log line {i+1}: {line}")
    
    print(f"\nFound {retry_logs} retry log entries and {fallback_logs} fallback log entries")
    assert fallback_logs > 0, "Expected fallback logs but found none"
    assert retry_logs > 0, "Expected retry logs but found none"
    print("✅ Fallback logging test passed!")

if __name__ == "__main__":
    test_fallback_logging()