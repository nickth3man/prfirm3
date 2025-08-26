"""
Streaming utilities for real-time progress updates.

This module provides streaming capabilities for the Virtual PR Firm to deliver
real-time progress updates, milestone notifications, and agent reasoning to users.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MilestoneType(Enum):
    """Types of milestones that can be streamed."""
    START = "start"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class Milestone:
    """Represents a milestone event."""
    type: MilestoneType
    message: str
    node_name: Optional[str] = None
    progress: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "message": self.message,
            "node_name": self.node_name,
            "progress": self.progress,
            "data": self.data,
            "timestamp": self.timestamp
        }

class ProgressTracker:
    """Tracks progress across multiple nodes."""
    
    def __init__(self, total_nodes: int = 10):
        self.total_nodes = total_nodes
        self.completed_nodes = 0
        self.current_node = None
        self.start_time = time.time()
    
    def start_node(self, node_name: str):
        """Mark the start of a node execution."""
        self.current_node = node_name
        logger.info(f"Starting node: {node_name}")
    
    def complete_node(self, node_name: str):
        """Mark the completion of a node."""
        self.completed_nodes += 1
        self.current_node = None
        logger.info(f"Completed node: {node_name} ({self.completed_nodes}/{self.total_nodes})")
    
    def get_progress(self) -> float:
        """Get current progress as a percentage."""
        return (self.completed_nodes / self.total_nodes) * 100
    
    def get_eta(self) -> Optional[float]:
        """Estimate time to completion."""
        if self.completed_nodes == 0:
            return None
        
        elapsed = time.time() - self.start_time
        rate = self.completed_nodes / elapsed
        remaining = self.total_nodes - self.completed_nodes
        
        return remaining / rate if rate > 0 else None

class StreamingManager:
    """Manages real-time streaming of progress and milestones."""
    
    def __init__(self, stream_callback: Optional[Callable[[Milestone], None]] = None):
        self.stream_callback = stream_callback
        self.progress_tracker = ProgressTracker()
        self.milestones: List[Milestone] = []
        self.is_streaming = False
    
    def start_streaming(self):
        """Start the streaming session."""
        self.is_streaming = True
        self.milestones = []
        self.progress_tracker = ProgressTracker()
        self.emit_milestone(MilestoneType.START, "Starting Virtual PR Firm pipeline")
    
    def stop_streaming(self):
        """Stop the streaming session."""
        self.is_streaming = False
        self.emit_milestone(MilestoneType.COMPLETE, "Pipeline completed")
    
    def emit_milestone(self, 
                      milestone_type: MilestoneType, 
                      message: str, 
                      node_name: Optional[str] = None,
                      data: Optional[Dict[str, Any]] = None):
        """Emit a milestone event."""
        if not self.is_streaming:
            return
        
        progress = self.progress_tracker.get_progress()
        milestone = Milestone(
            type=milestone_type,
            message=message,
            node_name=node_name,
            progress=progress,
            data=data
        )
        
        self._emit_milestone(milestone)
    
    def _emit_milestone(self, milestone: Milestone):
        """Internal method to emit a milestone."""
        self.milestones.append(milestone)
        
        if self.stream_callback:
            try:
                self.stream_callback(milestone)
            except Exception as e:
                logger.error(f"Error in stream callback: {e}")
        
        # Log milestone
        log_level = {
            MilestoneType.START: logging.INFO,
            MilestoneType.PROGRESS: logging.INFO,
            MilestoneType.COMPLETE: logging.INFO,
            MilestoneType.ERROR: logging.ERROR,
            MilestoneType.WARNING: logging.WARNING,
            MilestoneType.INFO: logging.INFO
        }.get(milestone.type, logging.INFO)
        
        logger.log(log_level, f"[{milestone.type.value.upper()}] {milestone.message}")
    
    def start_node(self, node_name: str):
        """Mark the start of a node execution."""
        self.progress_tracker.start_node(node_name)
        self.emit_milestone(
            MilestoneType.PROGRESS,
            f"Executing {node_name}",
            node_name=node_name
        )
    
    def complete_node(self, node_name: str, result: Optional[Dict[str, Any]] = None):
        """Mark the completion of a node."""
        self.progress_tracker.complete_node(node_name)
        self.emit_milestone(
            MilestoneType.PROGRESS,
            f"Completed {node_name}",
            node_name=node_name,
            data=result
        )
    
    def emit_error(self, message: str, node_name: Optional[str] = None, error_data: Optional[Dict[str, Any]] = None):
        """Emit an error milestone."""
        self.emit_milestone(
            MilestoneType.ERROR,
            message,
            node_name=node_name,
            data=error_data
        )
    
    def emit_warning(self, message: str, node_name: Optional[str] = None, warning_data: Optional[Dict[str, Any]] = None):
        """Emit a warning milestone."""
        self.emit_milestone(
            MilestoneType.WARNING,
            message,
            node_name=node_name,
            data=warning_data
        )
    
    def emit_info(self, message: str, node_name: Optional[str] = None, info_data: Optional[Dict[str, Any]] = None):
        """Emit an info milestone."""
        self.emit_milestone(
            MilestoneType.INFO,
            message,
            node_name=node_name,
            data=info_data
        )
    
    def get_milestones(self) -> List[Dict[str, Any]]:
        """Get all milestones as dictionaries."""
        return [milestone.to_dict() for milestone in self.milestones]
    
    def get_progress(self) -> float:
        """Get current progress percentage."""
        return self.progress_tracker.get_progress()
    
    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion."""
        return self.progress_tracker.get_eta()

class GradioStreamingManager(StreamingManager):
    """Streaming manager specifically for Gradio interfaces."""
    
    def __init__(self, progress_bar=None, status_text=None):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
    
    def _emit_milestone(self, milestone: Milestone):
        """Override to update Gradio components."""
        super()._emit_milestone(milestone)
        
        # Update progress bar
        if self.progress_bar and milestone.progress is not None:
            try:
                self.progress_bar.update(milestone.progress / 100)
            except Exception as e:
                logger.error(f"Error updating progress bar: {e}")
        
        # Update status text
        if self.status_text:
            try:
                status = f"{milestone.message}"
                if milestone.node_name:
                    status += f" ({milestone.node_name})"
                self.status_text.update(status)
            except Exception as e:
                logger.error(f"Error updating status text: {e}")

def create_streaming_manager(stream_callback: Optional[Callable[[Milestone], None]] = None) -> StreamingManager:
    """Factory function to create a streaming manager."""
    return StreamingManager(stream_callback)

def create_gradio_streaming_manager(progress_bar=None, status_text=None) -> GradioStreamingManager:
    """Factory function to create a Gradio streaming manager."""
    return GradioStreamingManager(progress_bar, status_text)

# Test function for development
if __name__ == "__main__":
    # Test streaming manager
    def test_callback(milestone: Milestone):
        print(f"Stream: {milestone.to_dict()}")
    
    manager = create_streaming_manager(test_callback)
    manager.start_streaming()
    
    manager.start_node("EngagementManagerNode")
    time.sleep(0.1)
    manager.complete_node("EngagementManagerNode")
    
    manager.start_node("BrandBibleIngestNode")
    time.sleep(0.1)
    manager.complete_node("BrandBibleIngestNode")
    
    manager.stop_streaming()
    
    print(f"Final progress: {manager.get_progress()}%")
    print(f"Total milestones: {len(manager.get_milestones())}")
