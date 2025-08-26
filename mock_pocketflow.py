"""Mock PocketFlow implementation for testing purposes.

This provides minimal implementations of Flow, BatchFlow, and Node
to enable testing without external dependencies.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class Node(ABC):
    """Base node class for PocketFlow nodes."""
    
    def __init__(self):
        self.next_nodes: List['Node'] = []
    
    def __rshift__(self, other: 'Node') -> 'Node':
        """Connect this node to another using >> operator."""
        self.next_nodes.append(other)
        return other
    
    @abstractmethod
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for execution."""
        pass
    
    @abstractmethod
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node logic."""
        pass
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> str:
        """Post-process the execution results."""
        return "completed"
    
    def run(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full node pipeline: prep -> exec -> post."""
        prep_res = self.prep(shared)
        exec_res = self.exec(prep_res)
        post_res = self.post(shared, prep_res, exec_res)
        
        # Update shared state with results from exec
        if isinstance(exec_res, dict):
            shared.update(exec_res)
        return shared


class Flow:
    """Simple flow implementation that runs nodes sequentially."""
    
    def __init__(self, start: Node):
        self.start = start
    
    def run(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Run the flow starting from the start node."""
        current_node = self.start
        
        while current_node:
            current_node.run(shared)
            # Move to next node (assuming linear flow for simplicity)
            current_node = current_node.next_nodes[0] if current_node.next_nodes else None
        
        return shared


class BatchFlow(Node):
    """Batch flow for parallel processing (simplified for testing)."""
    
    def __init__(self, start: Optional[Node] = None):
        super().__init__()
        self.start_node = start
    
    def prep(self, shared: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare parameter sets - to be overridden by subclasses."""
        return [{}]
    
    def exec(self, prep_res: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the batch flow - simplified for testing."""
        # For testing, just return empty result
        return {}
    
    def run(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Run the batch flow."""
        prep_res = self.prep(shared)
        exec_res = self.exec(prep_res)
        shared.update(exec_res)
        return shared