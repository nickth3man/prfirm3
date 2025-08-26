"""
Test different log levels and formats.
"""
import os
import sys
import subprocess

# Add current directory to path
sys.path.insert(0, '/home/runner/work/prfirm3/prfirm3')

def test_log_level_configuration():
    """Test that log level configuration works correctly."""
    print("Testing log level configuration...")
    
    # Test DEBUG level
    env = os.environ.copy()
    env["LOG_LEVEL"] = "DEBUG"
    env["LOG_FORMAT"] = "json"
    
    result = subprocess.run([
        sys.executable, 
        "/home/runner/work/prfirm3/prfirm3/test_basic_logging.py"
    ], env=env, capture_output=True, text=True)
    
    print("DEBUG level output:")
    print(result.stderr)
    assert result.returncode == 0
    assert '"level": "INFO"' in result.stderr  # Should contain INFO logs
    
    # Test ERROR level (should suppress INFO)
    env["LOG_LEVEL"] = "ERROR"
    
    result = subprocess.run([
        sys.executable, 
        "/home/runner/work/prfirm3/prfirm3/test_basic_logging.py"
    ], env=env, capture_output=True, text=True)
    
    print("\nERROR level output (should be minimal):")
    print(result.stderr)
    assert result.returncode == 0
    # Should have very few or no logs since we're only logging at INFO level
    
    # Test text format
    env["LOG_LEVEL"] = "INFO"
    env["LOG_FORMAT"] = "text"
    
    result = subprocess.run([
        sys.executable, 
        "/home/runner/work/prfirm3/prfirm3/test_basic_logging.py"
    ], env=env, capture_output=True, text=True)
    
    print("\nText format output:")
    print(result.stderr)
    assert result.returncode == 0
    # Should not contain JSON format
    assert '{"timestamp"' not in result.stderr
    assert 'pocketflow.node.TestNode' in result.stderr  # Should contain text format
    
    print("✅ Log level configuration test passed!")

if __name__ == "__main__":
    test_log_level_configuration()