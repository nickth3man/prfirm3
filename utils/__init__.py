"""
Utility functions for the Virtual PR Firm.

This package provides essential utilities for LLM calls, validation, error handling,
and monitoring that support the Virtual PR Firm's content generation pipeline.

Key Utilities:
- call_llm: Robust LLM calling with retries and fallbacks
- validate_shared_store: Input validation for shared state
- streaming_manager: Real-time progress updates
- error_handler: Centralized error handling and logging
- performance_monitor: Execution timing and metrics
"""

from .llm_utils import call_llm, call_llm_with_fallback
from .validation import (
    validate_shared_store, validate_platforms, validate_brand_bible,
    sanitize_input, normalize_platform_name
)
from .streaming import (
    StreamingManager, ProgressTracker, 
    create_streaming_manager, create_gradio_streaming_manager
)
from .error_handler import (
    ErrorHandler, RetryConfig, ErrorContext, ErrorSeverity,
    get_global_error_handler
)
from .performance import (
    PerformanceMonitor, MetricsCollector,
    get_global_monitor, monitor_execution, record_metric
)
from .config import ConfigManager, get_config, get_config_manager

__all__ = [
    'call_llm',
    'call_llm_with_fallback', 
    'validate_shared_store',
    'validate_platforms',
    'validate_brand_bible',
    'sanitize_input',
    'normalize_platform_name',
    'StreamingManager',
    'ProgressTracker',
    'create_streaming_manager',
    'create_gradio_streaming_manager',
    'ErrorHandler',
    'RetryConfig',
    'ErrorContext',
    'ErrorSeverity',
    'get_global_error_handler',
    'PerformanceMonitor',
    'MetricsCollector',
    'get_global_monitor',
    'monitor_execution',
    'record_metric',
    'ConfigManager',
    'get_config',
    'get_config_manager'
]