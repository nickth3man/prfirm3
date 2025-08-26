"""
Test text format logging.
"""
import os

# Set environment for text format logging BEFORE any imports
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "text"

import sys
sys.path.insert(0, '/home/runner/work/prfirm3/prfirm3')

from logging_config import get_node_logger

def test_text_format():
    """Test text format logging."""
    print("Testing text format logging...")
    
    # Create a logger
    logger = get_node_logger("TestTextNode", "test-text-correlation")
    
    # Test different log levels
    logger.info("This is a test info message", action="test")
    logger.prep_start()
    logger.exec_start(retry_count=0)
    logger.exec_end(retry_count=0)
    logger.post_start(action_result="default")
    logger.post_end(action_result="default")
    
    print("Text format logging test completed. Check above for text log output.")

if __name__ == "__main__":
    test_text_format()