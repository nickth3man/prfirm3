"""
Central logging configuration for structured logging across nodes.

Provides JSON-structured logging with configurable log levels and consistent context fields.
"""
import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as structured JSON.
    
    Includes consistent fields: timestamp, level, node, action, retry_count, correlation_id, message
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Base structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add context fields if present
        if hasattr(record, 'node_name'):
            log_entry["node"] = record.node_name
        if hasattr(record, 'action'):
            log_entry["action"] = record.action
        if hasattr(record, 'retry_count'):
            log_entry["retry_count"] = record.retry_count
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, 'context'):
            log_entry.update(record.context)
            
        return json.dumps(log_entry)


class NodeLogger:
    """
    Logger wrapper for node operations with structured context.
    
    Provides methods for logging node lifecycle events with consistent structured fields.
    """
    
    def __init__(self, node_name: str, correlation_id: Optional[str] = None):
        self.node_name = node_name
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger(f"pocketflow.node.{node_name}")
    
    def _log(self, level: int, message: str, action: Optional[str] = None, 
             retry_count: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        """Internal method to add structured context to log records."""
        extra = {
            'node_name': self.node_name,
            'correlation_id': self.correlation_id
        }
        
        if action is not None:
            extra['action'] = action
        if retry_count is not None:
            extra['retry_count'] = retry_count
        if context:
            extra['context'] = context
            
        self.logger.log(level, message, extra=extra)
    
    def prep_start(self, context: Optional[Dict[str, Any]] = None):
        """Log the start of prep phase."""
        self._log(logging.INFO, "Node prep started", action="prep", context=context)
    
    def prep_end(self, context: Optional[Dict[str, Any]] = None):
        """Log the end of prep phase."""
        self._log(logging.INFO, "Node prep completed", action="prep", context=context)
    
    def exec_start(self, retry_count: int = 0, context: Optional[Dict[str, Any]] = None):
        """Log the start of exec phase."""
        self._log(logging.INFO, "Node exec started", action="exec", 
                 retry_count=retry_count, context=context)
    
    def exec_end(self, retry_count: int = 0, context: Optional[Dict[str, Any]] = None):
        """Log the end of exec phase."""
        self._log(logging.INFO, "Node exec completed", action="exec", 
                 retry_count=retry_count, context=context)
    
    def exec_retry(self, retry_count: int, error: str, context: Optional[Dict[str, Any]] = None):
        """Log exec retry attempt."""
        ctx = {"error": str(error)}
        if context:
            ctx.update(context)
        self._log(logging.WARNING, f"Node exec retry attempt {retry_count}", 
                 action="exec_retry", retry_count=retry_count, context=ctx)
    
    def exec_fallback(self, retry_count: int, error: str, context: Optional[Dict[str, Any]] = None):
        """Log exec fallback execution."""
        ctx = {"error": str(error)}
        if context:
            ctx.update(context)
        self._log(logging.ERROR, "Node exec fallback triggered", 
                 action="exec_fallback", retry_count=retry_count, context=ctx)
    
    def post_start(self, action_result: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log the start of post phase."""
        ctx = {}
        if action_result is not None:
            ctx["action_result"] = action_result
        if context:
            ctx.update(context)
        self._log(logging.INFO, "Node post started", action="post", context=ctx)
    
    def post_end(self, action_result: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log the end of post phase."""
        ctx = {}
        if action_result is not None:
            ctx["action_result"] = action_result
        if context:
            ctx.update(context)
        self._log(logging.INFO, "Node post completed", action="post", context=ctx)
    
    def error(self, message: str, action: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log an error."""
        self._log(logging.ERROR, message, action=action, context=context)
    
    def warning(self, message: str, action: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log a warning."""
        self._log(logging.WARNING, message, action=action, context=context)
    
    def info(self, message: str, action: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log(logging.INFO, message, action=action, context=context)
    
    def debug(self, message: str, action: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log(logging.DEBUG, message, action=action, context=context)


def setup_logging() -> None:
    """
    Setup structured logging configuration.
    
    Configures logging based on environment variables:
    - LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
    - LOG_FORMAT: Set format (json, text). Default: json
    """
    # Get configuration from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    
    # Validate log level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO, 
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    if log_level not in level_map:
        log_level = "INFO"
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_map[log_level])
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level_map[log_level])
    
    # Setup formatter based on format preference
    if log_format == "json":
        formatter = StructuredFormatter()
    else:
        # Default text formatter with some structure
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Setup specific logger for pocketflow nodes
    pocketflow_logger = logging.getLogger("pocketflow")
    pocketflow_logger.setLevel(level_map[log_level])


def get_node_logger(node_name: str, correlation_id: Optional[str] = None) -> NodeLogger:
    """
    Get a structured logger for a specific node.
    
    Args:
        node_name: Name of the node for logging context
        correlation_id: Optional correlation ID for tracking across nodes
    
    Returns:
        NodeLogger instance configured for the node
    """
    # Ensure logging is setup on first use
    setup_logging()
    return NodeLogger(node_name, correlation_id)


# Don't initialize logging automatically when module is imported
# Let the user call setup_logging() explicitly or on first use