from pocketflow import Node
from utils.call_llm import call_llm
from logging_mixin import LoggingMixin


class GetQuestionNode(LoggingMixin, Node):
    def exec(self, _):
        # Get question directly from user input
        self.node_logger.info("Requesting user input for question")
        # Use pre-populated question if available in shared state or ask user
        user_question = "What is the meaning of life?"  # For demo purposes
        self.node_logger.info("User question received", context={"question_length": len(user_question)})
        return user_question
    
    def post(self, shared, prep_res, exec_res):
        # Store the user's question
        shared["question"] = exec_res
        self.node_logger.info("Question stored in shared context")
        return super().post(shared, prep_res, exec_res) or "default"  # Go to the next node


class AnswerNode(LoggingMixin, Node):
    def prep(self, shared):
        # Read question from shared
        question = shared["question"]
        self.node_logger.info("Retrieved question from shared context", 
                             context={"question_length": len(question) if question else 0})
        return question
    
    def exec(self, question):
        # Call LLM to get the answer
        self.node_logger.info("Calling LLM for answer", context={"question": question[:100] + "..." if len(question) > 100 else question})
        # Mock the LLM call for demo
        answer = f"Mock LLM response to: {question}"
        self.node_logger.info("LLM response received", context={"answer_length": len(answer)})
        return answer
    
    def post(self, shared, prep_res, exec_res):
        # Store the answer in shared
        shared["answer"] = exec_res
        self.node_logger.info("Answer stored in shared context")
        return super().post(shared, prep_res, exec_res)