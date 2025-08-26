"""
Enhanced flow creation with correlation ID support.
"""
from pocketflow import Flow
from nodes import GetQuestionNode, AnswerNode

def create_qa_flow(correlation_id=None):
    """Create and return a question-answering flow with optional correlation ID."""
    # Create nodes
    get_question_node = GetQuestionNode()
    answer_node = AnswerNode()
    
    # Set correlation ID if provided
    if correlation_id:
        get_question_node.set_correlation_id(correlation_id)
        answer_node.set_correlation_id(correlation_id)
    
    # Connect nodes in sequence
    get_question_node >> answer_node
    
    # Create flow starting with input node
    return Flow(start=get_question_node)

# Create default flow for backward compatibility
qa_flow = create_qa_flow()