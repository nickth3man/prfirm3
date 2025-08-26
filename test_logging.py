"""
Test script to verify logging functionality without requiring actual LLM calls.
"""
import os
import sys
import json
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout

# Set environment for structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

# Add current directory to path
sys.path.insert(0, '/home/runner/work/prfirm3/prfirm3')

from pocketflow import Node
from logging_mixin import LoggingMixin
from logging_config import setup_logging

# Mock LLM call to avoid needing API key
def mock_call_llm(prompt):
    return f"Mock response to: {prompt[:50]}..."

# Create test nodes
class MockGetQuestionNode(LoggingMixin, Node):
    def exec(self, _):
        self.node_logger.info("Requesting user input for question")
        user_question = "What is 2+2?"  # Mocked input
        self.node_logger.info("User question received", context={"question_length": len(user_question)})
        return user_question
    
    def post(self, shared, prep_res, exec_res):
        shared["question"] = exec_res
        self.node_logger.info("Question stored in shared context")
        return "default"

class MockAnswerNode(LoggingMixin, Node):
    def prep(self, shared):
        question = shared["question"]
        self.node_logger.info("Retrieved question from shared context", 
                             context={"question_length": len(question) if question else 0})
        return question
    
    def exec(self, question):
        self.node_logger.info("Calling LLM for answer", context={"question": question[:100] + "..." if len(question) > 100 else question})
        answer = mock_call_llm(question)
        self.node_logger.info("LLM response received", context={"answer_length": len(answer)})
        return answer
    
    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res
        self.node_logger.info("Answer stored in shared context")

def test_logging():
    """Test that nodes produce structured logs."""
    print("Testing structured logging functionality...")
    
    # Create a temporary log handler to capture logs
    import logging
    from io import StringIO
    
    # Create a string buffer to capture logs
    log_capture = StringIO()
    
    # Get the pocketflow logger and add our handler
    pocketflow_logger = logging.getLogger("pocketflow")
    
    # Create handler with our formatter
    from logging_config import StructuredFormatter
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(StructuredFormatter())
    pocketflow_logger.addHandler(handler)
    
    try:
        # Create nodes
        get_question_node = MockGetQuestionNode()
        answer_node = MockAnswerNode()
        
        # Set correlation ID
        get_question_node.set_correlation_id("test-correlation-123")
        answer_node.set_correlation_id("test-correlation-123")
        
        # Run nodes
        shared = {}
        
        # Run get question node
        action1 = get_question_node.run(shared)
        print(f"GetQuestionNode returned action: {action1}")
        
        # Run answer node  
        action2 = answer_node.run(shared)
        print(f"AnswerNode returned action: {action2}")
        
        print(f"Final shared state: {shared}")
        
    finally:
        # Remove our handler
        pocketflow_logger.removeHandler(handler)
    
    # Analyze log output
    log_content = log_capture.getvalue()
    log_lines = [line.strip() for line in log_content.strip().split('\n') if line.strip()]
    print(f"\nCaptured {len(log_lines)} log lines:")
    
    valid_json_logs = 0
    for i, line in enumerate(log_lines):
        try:
            log_entry = json.loads(line)
            valid_json_logs += 1
            print(f"Log {i+1}: {json.dumps(log_entry, indent=2)}")
            
            # Verify expected fields
            required_fields = ["timestamp", "level", "message"]
            for field in required_fields:
                assert field in log_entry, f"Missing required field: {field}"
            
            # Check for structured fields when present
            if "node" in log_entry:
                assert log_entry["node"] in ["MockGetQuestionNode", "MockAnswerNode"]
            if "correlation_id" in log_entry:
                assert log_entry["correlation_id"] == "test-correlation-123"
            if "action" in log_entry:
                assert log_entry["action"] in ["prep", "exec", "post"]
                
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in log line {i+1}: {line}")
            print(f"Error: {e}")
    
    print(f"\nValidation complete: {valid_json_logs} valid JSON log entries found")
    
    # Verify we got some logs
    assert valid_json_logs > 0, "No valid JSON logs were produced"
    print("✅ Structured logging test passed!")

if __name__ == "__main__":
    test_logging()