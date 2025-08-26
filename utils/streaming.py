import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Generator, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

class MessageType(Enum):
    MILESTONE = "milestone"
    PROGRESS = "progress"
    AGENT_REASONING = "agent_reasoning"
    ERROR = "error"
    COMPLETION = "completion"
    FEEDBACK = "feedback"

@dataclass
class StreamMessage:
    """Structured message for streaming"""
    type: MessageType
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type.value,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }

class StreamingManager:
    """Manages real-time streaming of messages to UI"""
    
    def __init__(self):
        self.messages: List[StreamMessage] = []
        self.subscribers: List[Callable[[StreamMessage], None]] = []
        self.logger = logging.getLogger(__name__)
        
    def emit(self, role: str, content: str, 
             message_type: MessageType = MessageType.MILESTONE,
             metadata: Optional[Dict[str, Any]] = None):
        """Emit a message to all subscribers"""
        message = StreamMessage(
            type=message_type,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.messages.append(message)
        self.logger.info(f"Streaming: {role} - {content[:100]}...")
        
        # Notify all subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(message)
            except Exception as e:
                self.logger.error(f"Error in subscriber: {e}")
    
    def subscribe(self, callback: Callable[[StreamMessage], None]):
        """Subscribe to streaming messages"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[StreamMessage], None]):
        """Unsubscribe from streaming messages"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_messages(self, message_type: Optional[MessageType] = None, 
                    limit: Optional[int] = None) -> List[StreamMessage]:
        """Get messages with optional filtering"""
        messages = self.messages
        
        if message_type:
            messages = [msg for msg in messages if msg.type == message_type]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def clear_messages(self):
        """Clear all messages"""
        self.messages.clear()

class ProgressTracker:
    """Tracks progress through workflow stages"""
    
    def __init__(self, total_stages: int = 10):
        self.total_stages = total_stages
        self.current_stage = 0
        self.completed_stages = []
        self.stage_details = {}
        self.streaming_manager = StreamingManager()
    
    def start_stage(self, stage_name: str, description: str = ""):
        """Start a new stage"""
        self.current_stage += 1
        self.stage_details[stage_name] = {
            "start_time": datetime.now(),
            "description": description,
            "status": "in_progress"
        }
        
        self.streaming_manager.emit(
            "system",
            f"Starting stage {self.current_stage}/{self.total_stages}: {stage_name}",
            MessageType.PROGRESS,
            {"stage": stage_name, "stage_number": self.current_stage}
        )
    
    def complete_stage(self, stage_name: str, result: str = ""):
        """Complete a stage"""
        if stage_name in self.stage_details:
            self.stage_details[stage_name]["end_time"] = datetime.now()
            self.stage_details[stage_name]["status"] = "completed"
            self.stage_details[stage_name]["result"] = result
        
        self.completed_stages.append(stage_name)
        
        progress_percent = (len(self.completed_stages) / self.total_stages) * 100
        
        self.streaming_manager.emit(
            "system",
            f"Completed: {stage_name} ({progress_percent:.1f}%)",
            MessageType.PROGRESS,
            {"stage": stage_name, "progress": progress_percent}
        )
    
    def fail_stage(self, stage_name: str, error: str):
        """Mark a stage as failed"""
        if stage_name in self.stage_details:
            self.stage_details[stage_name]["end_time"] = datetime.now()
            self.stage_details[stage_name]["status"] = "failed"
            self.stage_details[stage_name]["error"] = error
        
        self.streaming_manager.emit(
            "system",
            f"Failed: {stage_name} - {error}",
            MessageType.ERROR,
            {"stage": stage_name, "error": error}
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "total_stages": self.total_stages,
            "progress_percent": (len(self.completed_stages) / self.total_stages) * 100,
            "stage_details": self.stage_details
        }

class AgentChatLogger:
    """Logs agent-to-agent communications for transparency"""
    
    def __init__(self):
        self.conversations: List[Dict[str, Any]] = []
        self.streaming_manager = StreamingManager()
    
    def log_agent_message(self, agent_name: str, message: str, 
                         reasoning: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """Log a message from an agent"""
        conversation_entry = {
            "timestamp": datetime.now(),
            "agent": agent_name,
            "message": message,
            "reasoning": reasoning,
            "metadata": metadata or {}
        }
        
        self.conversations.append(conversation_entry)
        
        # Stream the message
        self.streaming_manager.emit(
            agent_name,
            message,
            MessageType.AGENT_REASONING,
            {
                "reasoning": reasoning,
                "agent": agent_name,
                **(metadata or {})
            }
        )
    
    def log_agent_reasoning(self, agent_name: str, reasoning: str,
                           context: Optional[Dict[str, Any]] = None):
        """Log agent reasoning process"""
        self.log_agent_message(
            agent_name,
            f"Reasoning: {reasoning}",
            reasoning=reasoning,
            metadata={"context": context}
        )
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conversations = self.conversations
        if limit:
            conversations = conversations[-limit:]
        return conversations
    
    def get_agent_conversations(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get conversations for a specific agent"""
        return [conv for conv in self.conversations if conv["agent"] == agent_name]

class MilestoneStreamer:
    """Streams milestone updates to UI"""
    
    def __init__(self):
        self.streaming_manager = StreamingManager()
        self.milestones = []
    
    def emit_milestone(self, milestone: str, details: Optional[str] = None):
        """Emit a milestone update"""
        self.milestones.append({
            "milestone": milestone,
            "details": details,
            "timestamp": datetime.now()
        })
        
        self.streaming_manager.emit(
            "system",
            f"Milestone: {milestone}",
            MessageType.MILESTONE,
            {"details": details, "milestone": milestone}
        )
    
    def emit_completion(self, task_name: str, result: str):
        """Emit completion message"""
        self.streaming_manager.emit(
            "system",
            f"Completed: {task_name}",
            MessageType.COMPLETION,
            {"task": task_name, "result": result}
        )
    
    def emit_error(self, error: str, context: Optional[str] = None):
        """Emit error message"""
        self.streaming_manager.emit(
            "system",
            f"Error: {error}",
            MessageType.ERROR,
            {"context": context, "error": error}
        )

# Global instances
_streaming_manager = StreamingManager()
_progress_tracker = ProgressTracker()
_agent_chat_logger = AgentChatLogger()
_milestone_streamer = MilestoneStreamer()

def get_streaming_manager() -> StreamingManager:
    """Get global streaming manager instance"""
    return _streaming_manager

def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance"""
    return _progress_tracker

def get_agent_chat_logger() -> AgentChatLogger:
    """Get global agent chat logger instance"""
    return _agent_chat_logger

def get_milestone_streamer() -> MilestoneStreamer:
    """Get global milestone streamer instance"""
    return _milestone_streamer

def emit(role: str, text: str, message_type: MessageType = MessageType.MILESTONE,
         metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to emit messages"""
    _streaming_manager.emit(role, text, message_type, metadata)

def messages() -> Generator[Dict[str, Any], None, None]:
    """Generator for UI to consume messages"""
    for message in _streaming_manager.messages:
        yield message.to_dict()

def clear_messages():
    """Clear all messages"""
    _streaming_manager.clear_messages()

if __name__ == "__main__":
    # Test the streaming system
    streaming_manager = get_streaming_manager()
    progress_tracker = get_progress_tracker()
    agent_logger = get_agent_chat_logger()
    milestone_streamer = get_milestone_streamer()
    
    # Test milestone streaming
    milestone_streamer.emit_milestone("Starting content generation", "Initializing workflow")
    milestone_streamer.emit_milestone("Brand Bible parsed", "Voice and tone extracted")
    milestone_streamer.emit_completion("Content generation", "All platforms completed")
    
    # Test agent logging
    agent_logger.log_agent_message("ContentCraftsman", "Generated initial content structure")
    agent_logger.log_agent_reasoning("StyleEditor", "Checking for AI fingerprints and style violations")
    
    # Test progress tracking
    progress_tracker.start_stage("Content Generation", "Creating content for all platforms")
    progress_tracker.complete_stage("Content Generation", "Successfully generated content")
    
    # Print all messages
    print("All messages:")
    for message in streaming_manager.get_messages():
        print(f"- {message.role}: {message.content}")
    
    print(f"\nProgress: {progress_tracker.get_progress()}")
    print(f"Agent conversations: {len(agent_logger.get_conversation_history())}")
