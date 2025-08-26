"""
Logging mixin for PocketFlow nodes.

Provides structured logging capabilities that can be mixed into existing node classes.
"""
from typing import Any, Optional, Dict
import time
from logging_config import get_node_logger, NodeLogger


class LoggingMixin:
    """
    Mixin class that adds structured logging to PocketFlow nodes.
    
    This mixin can be added to existing node classes to provide structured logging
    without modifying the core node implementation.
    """
    
    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Initialize logger - node name will be set when first accessed
        self._node_logger: Optional[NodeLogger] = None
        self._correlation_id: Optional[str] = None
    
    @property
    def node_logger(self) -> NodeLogger:
        """Get or create the node logger."""
        if self._node_logger is None:
            node_name = self.__class__.__name__
            self._node_logger = get_node_logger(node_name, self._correlation_id)
        return self._node_logger
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for this node's logging context."""
        self._correlation_id = correlation_id
        # Reset logger so it picks up new correlation ID
        self._node_logger = None
    
    def prep(self, shared: Dict[str, Any]) -> Any:
        """Wrapped prep method with logging."""
        self.node_logger.prep_start()
        try:
            result = super().prep(shared) if hasattr(super(), 'prep') else None
            self.node_logger.prep_end()
            return result
        except Exception as e:
            self.node_logger.error(f"Error in prep: {e}", action="prep")
            raise
    
    def exec(self, prep_res: Any) -> Any:
        """Wrapped exec method with logging."""
        # Get current retry count if available
        retry_count = getattr(self, 'cur_retry', 0)
        
        self.node_logger.exec_start(retry_count=retry_count)
        try:
            result = super().exec(prep_res) if hasattr(super(), 'exec') else None
            self.node_logger.exec_end(retry_count=retry_count)
            return result
        except Exception as e:
            self.node_logger.exec_retry(retry_count, str(e))
            raise
    
    def _exec(self, prep_res: Any) -> Any:
        """Override _exec to properly log retries."""
        # If this has a max_retries attribute, handle retry logging
        if hasattr(self, 'max_retries'):
            for self.cur_retry in range(self.max_retries):
                self.node_logger.exec_start(retry_count=self.cur_retry)
                try:
                    result = self.exec(prep_res)
                    self.node_logger.exec_end(retry_count=self.cur_retry)
                    return result
                except Exception as e:
                    if self.cur_retry == self.max_retries - 1:
                        self.node_logger.exec_fallback(self.cur_retry, str(e))
                        return self.exec_fallback(prep_res, e)
                    else:
                        self.node_logger.exec_retry(self.cur_retry, str(e))
                        if hasattr(self, 'wait') and self.wait > 0:
                            time.sleep(self.wait)
        else:
            # Fall back to single execution for nodes without retry
            return self.exec(prep_res)
    
    def exec_fallback(self, prep_res: Any, exc: Exception) -> Any:
        """Wrapped exec_fallback method with logging."""
        retry_count = getattr(self, 'cur_retry', 0)
        
        try:
            result = super().exec_fallback(prep_res, exc) if hasattr(super(), 'exec_fallback') else None
            return result
        except Exception as e:
            self.node_logger.error(f"Error in exec_fallback: {e}", action="exec_fallback")
            raise
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        """Wrapped post method with logging."""
        self.node_logger.post_start()
        try:
            result = super().post(shared, prep_res, exec_res) if hasattr(super(), 'post') else None
            self.node_logger.post_end(action_result=result)
            return result
        except Exception as e:
            self.node_logger.error(f"Error in post: {e}", action="post")
            raise


class LoggedNode(LoggingMixin):
    """
    Base class for nodes with built-in logging.
    
    This can be used as a base class for new nodes that want logging by default.
    Note: This requires importing Node from pocketflow, so it might be better
    to use the mixin approach for existing code.
    """
    pass