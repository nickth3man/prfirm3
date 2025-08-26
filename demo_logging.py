"""
Demo of the complete Q&A application with structured logging enabled.
This demonstrates the logging functionality in a real flow scenario.
"""
import os
import uuid

# Set environment for structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

from flow import create_qa_flow
from logging_config import get_node_logger

def main():
    """Run the Q&A flow with structured logging."""
    
    # Generate a correlation ID for this request
    correlation_id = str(uuid.uuid4())
    
    # Create logger for the main flow
    main_logger = get_node_logger("MainFlow", correlation_id)
    main_logger.info("Starting Q&A flow demo")
    
    # Create shared state
    shared = {
        "question": "What is the meaning of life?",  # Pre-populated to avoid user input
        "answer": None
    }
    
    main_logger.info("Initial shared state", context={"question": shared["question"]})
    
    # Create the flow
    qa_flow = create_qa_flow()
    
    # Set correlation ID for all nodes in the flow
    # This would ideally be done automatically by the flow
    for node_name in ["get_question_node", "answer_node"]:
        if hasattr(qa_flow, node_name):
            getattr(qa_flow, node_name).set_correlation_id(correlation_id)
    
    # In practice, we'd traverse the flow graph to set correlation ID
    # For now, let's just run and see what happens
    main_logger.info("Running Q&A flow")
    
    try:
        # Run the flow
        result = qa_flow.run(shared)
        
        main_logger.info("Q&A flow completed", 
                        context={
                            "result": result,
                            "question": shared["question"],
                            "answer": shared.get("answer", "No answer")[:100] + "..." if len(shared.get("answer", "")) > 100 else shared.get("answer", "No answer")
                        })
        
        print(f"\n🎯 Final Results:")
        print(f"Question: {shared['question']}")
        print(f"Answer: {shared.get('answer', 'No answer generated')}")
        
    except Exception as e:
        main_logger.error(f"Q&A flow failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()