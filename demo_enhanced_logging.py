"""
Demo using enhanced flow with shared correlation ID.
"""
import os
import uuid

# Set environment for structured logging
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "json"

from flow_enhanced import create_qa_flow
from logging_config import get_node_logger

def main():
    """Run the Q&A flow with shared correlation ID."""
    
    # Generate a correlation ID for this request
    correlation_id = str(uuid.uuid4())
    
    # Create logger for the main flow
    main_logger = get_node_logger("MainFlow", correlation_id)
    main_logger.info("Starting enhanced Q&A flow demo with shared correlation ID")
    
    # Create shared state
    shared = {
        "question": "What is 2+2?",
        "answer": None
    }
    
    main_logger.info("Initial shared state", context={"question": shared["question"]})
    
    # Create the flow with shared correlation ID
    qa_flow = create_qa_flow(correlation_id)
    
    main_logger.info("Running Q&A flow with correlation tracking")
    
    try:
        # Run the flow
        result = qa_flow.run(shared)
        
        main_logger.info("Q&A flow completed successfully", 
                        context={
                            "result": result,
                            "question": shared["question"],
                            "answer": shared.get("answer", "No answer")
                        })
        
        print(f"\n🎯 Final Results (Correlation ID: {correlation_id}):")
        print(f"Question: {shared['question']}")
        print(f"Answer: {shared.get('answer', 'No answer generated')}")
        
    except Exception as e:
        main_logger.error(f"Q&A flow failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()