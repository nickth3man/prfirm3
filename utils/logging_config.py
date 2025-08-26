"""
Comprehensive logging configuration for the Virtual PR Firm system.

This module provides centralized logging configuration with support for:
- Multiple log levels and formats
- File and console handlers
- Structured logging with context
- Log rotation and archival
- Performance monitoring
"""

import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class LogFormat(Enum):
    """Available log formats."""
    SIMPLE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED = "%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LoggerConfig:
    """Configuration for individual loggers."""
    name: str
    level: str = "INFO"
    handlers: List[str] = None
    propagate: bool = True
    
    def __post_init__(self):
        if self.handlers is None:
            self.handlers = ["console", "file"]


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        # Base structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                          "funcName", "levelname", "levelno", "lineno", 
                          "module", "msecs", "message", "pathname", "process",
                          "processName", "relativeCreated", "thread", "threadName",
                          "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def __init__(self, context: Dict[str, Any]):
        super().__init__()
        self.context = context
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to log record."""
        record.elapsed_time = time.time() - self.start_time
        record.memory_usage = self._get_memory_usage()
        return True
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0


class LoggingManager:
    """Centralized logging management."""
    
    def __init__(self, 
                 log_dir: Path = Path("./logs"),
                 default_level: str = "INFO",
                 default_format: LogFormat = LogFormat.DETAILED):
        """
        Initialize logging manager.
        
        Args:
            log_dir: Directory for log files
            default_level: Default logging level
            default_format: Default log format
        """
        self.log_dir = log_dir
        self.default_level = default_level
        self.default_format = default_format
        self.handlers: Dict[str, logging.Handler] = {}
        self.loggers: Dict[str, logging.Logger] = {}
        self.context: Dict[str, Any] = {}
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_default_logging(self):
        """Set up default logging configuration."""
        # Create handlers
        self._create_console_handler()
        self._create_file_handler("app.log")
        self._create_error_file_handler("errors.log")
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # Add handlers
        for handler in self.handlers.values():
            root_logger.addHandler(handler)
        
        # Add context filter
        if self.context:
            context_filter = ContextFilter(self.context)
            for handler in self.handlers.values():
                handler.addFilter(context_filter)
        
        # Log startup message
        root_logger.info("Logging system initialized")
    
    def _create_console_handler(self):
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.default_level)
        
        # Set formatter based on format type
        if self.default_format == LogFormat.JSON:
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(logging.Formatter(self.default_format.value))
        
        self.handlers["console"] = handler
    
    def _create_file_handler(self, filename: str, max_bytes: int = 10485760, backup_count: int = 5):
        """Create rotating file handler."""
        filepath = self.log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(self.default_level)
        
        # Set formatter
        if self.default_format == LogFormat.JSON:
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(logging.Formatter(self.default_format.value))
        
        self.handlers[f"file_{filename}"] = handler
    
    def _create_error_file_handler(self, filename: str):
        """Create error-only file handler."""
        filepath = self.log_dir / filename
        handler = logging.FileHandler(filepath)
        handler.setLevel(logging.ERROR)
        handler.setFormatter(logging.Formatter(LogFormat.DETAILED.value))
        
        self.handlers[f"error_{filename}"] = handler
    
    def create_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """Create a configured logger."""
        logger = logging.getLogger(name)
        
        # Set level
        if level:
            logger.setLevel(level)
        else:
            logger.setLevel(self.default_level)
        
        # Store reference
        self.loggers[name] = logger
        
        return logger
    
    def add_context(self, **kwargs):
        """Add global context to all log messages."""
        self.context.update(kwargs)
        
        # Update existing handlers
        context_filter = ContextFilter(self.context)
        for handler in self.handlers.values():
            # Remove old context filters
            handler.filters = [f for f in handler.filters if not isinstance(f, ContextFilter)]
            # Add new context filter
            handler.addFilter(context_filter)
    
    def enable_performance_logging(self):
        """Enable performance metrics in logs."""
        perf_filter = PerformanceFilter()
        for handler in self.handlers.values():
            handler.addFilter(perf_filter)
    
    def create_node_logger(self, node_name: str) -> logging.Logger:
        """Create a logger specifically for a node."""
        logger = self.create_logger(f"nodes.{node_name}")
        
        # Add node-specific handler if needed
        node_file = f"nodes/{node_name.lower()}.log"
        self._create_file_handler(node_file)
        logger.addHandler(self.handlers[f"file_{node_file}"])
        
        return logger
    
    def get_logger_stats(self) -> Dict[str, Any]:
        """Get statistics about logging activity."""
        stats = {
            "active_loggers": len(self.loggers),
            "handlers": len(self.handlers),
            "context_keys": list(self.context.keys()),
            "log_files": []
        }
        
        # Get log file sizes
        for log_file in self.log_dir.glob("*.log"):
            stats["log_files"].append({
                "name": log_file.name,
                "size_mb": log_file.stat().st_size / 1024 / 1024
            })
        
        return stats


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def initialize_logging(
    log_dir: Path = Path("./logs"),
    level: str = "INFO",
    format_type: LogFormat = LogFormat.DETAILED,
    enable_performance: bool = False
) -> LoggingManager:
    """Initialize global logging configuration."""
    global _logging_manager
    
    _logging_manager = LoggingManager(
        log_dir=log_dir,
        default_level=level,
        default_format=format_type
    )
    
    _logging_manager.setup_default_logging()
    
    if enable_performance:
        _logging_manager.enable_performance_logging()
    
    return _logging_manager


def get_logging_manager() -> LoggingManager:
    """Get global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = initialize_logging()
    return _logging_manager


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    manager = get_logging_manager()
    return manager.create_logger(name)


def add_logging_context(**kwargs):
    """Add context to all log messages."""
    manager = get_logging_manager()
    manager.add_context(**kwargs)


# Convenience function for structured logging
def log_structured(logger: logging.Logger, level: str, message: str, **kwargs):
    """Log a structured message with additional fields."""
    # Add extra fields to log record
    extra = {}
    for key, value in kwargs.items():
        extra[key] = value
    
    # Log with appropriate level
    log_func = getattr(logger, level.lower())
    log_func(message, extra=extra)