"""
Simple test to verify logging is working correctly.
"""
import os
import sys
import logging

# Set environment for structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

# Add current directory to path
sys.path.insert(0, '/home/runner/work/prfirm3/prfirm3')

from logging_config import get_node_logger

def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing basic logging functionality...")
    
    # Create a logger
    logger = get_node_logger("TestNode", "test-correlation-456")
    
    # Test different log levels
    logger.info("This is a test info message", action="test")
    logger.prep_start()
    logger.exec_start(retry_count=0)
    logger.exec_end(retry_count=0)
    logger.post_start(action_result="default")
    logger.post_end(action_result="default")
    
    print("Basic logging test completed. Check above for JSON log output.")

if __name__ == "__main__":
    test_basic_logging()