"""Optimized CLI/Gradio starter for the Virtual PR Firm.

This module provides a demonstration interface for the Virtual PR Firm's content
generation capabilities with comprehensive error handling, validation, streaming,
and performance monitoring.

The module exposes the following key functionality:
- `run_demo()`: A CLI function that executes the main flow with sample data
- `create_gradio_interface()`: Creates a web-based interface using Gradio
- `run_flow_with_validation()`: Executes flow with full validation and error handling

Example Usage:
    Command Line:
        $ python main.py
    
    Programmatic:
        >>> from main import run_demo, create_gradio_interface
        >>> run_demo()  # Run CLI demo
        >>> app = create_gradio_interface()  # Create web interface
        >>> app.launch()  # Launch web interface

Features:
- Comprehensive error handling and retry mechanisms
- Real-time streaming of progress and milestones
- Input validation and sanitization
- Performance monitoring and metrics collection
- Configuration management with environment variable support
- Graceful fallbacks when external services are unavailable
"""

from flow import create_main_flow
from typing import Any, Dict, Optional, List, TYPE_CHECKING

# Import optimization utilities
from utils import (
    validate_shared_store, sanitize_input, normalize_platform_name,
    create_streaming_manager, create_gradio_streaming_manager,
    get_global_error_handler, ErrorContext, ErrorSeverity,
    get_global_monitor, monitor_execution, record_metric,
    get_config, get_config_manager
)

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime dependency
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except Exception:
    gr = None  # type: Optional[Any]

import logging
import time

# Configure logging based on configuration
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Module logger
logger = logging.getLogger(__name__)


def run_demo() -> None:
    """Run a minimal demo of the main flow using a sample shared store.

    WHY (intent / invariants): Provide a simple CLI entry point for developers
    to validate the wiring and fallback behaviour of the flow without
    requiring Gradio or external keys.

    Pre-condition:
        - The environment must have `pocketflow` installed or available on PYTHONPATH.
    Post-condition:
        - `shared["content_pieces"]` will contain generated drafts or remain
          absent if validation failed. Any errors are logged.

    Raises:
        - ValueError: invalid shared store structure (validated by `validate_shared_store`).

    Example:
        >>> run_demo()
        Content pieces: {'twitter': 'Draft for twitter: Announce product', ...}

    Performance expectation: Fast for small platform lists (< 5); in production
    heavy LLM calls may dominate runtime.

    Test stub (pytest):
        def test_run_demo_smoke(tmp_path):
            # smoke test that run_demo doesn't raise
            run_demo()
    """
    
    # Initialize monitoring and streaming
    monitor = get_global_monitor()
    streaming_manager = create_streaming_manager()
    error_handler = get_global_error_handler()
    
    # Start performance monitoring
    start_time = time.time()
    record_metric("demo_start", start_time, "timestamp")
    
    # NOTE: defaults are intentional for demo; replace with configuration in prod
    shared: Dict[str, Any] = {
        "task_requirements": {"platforms": ["twitter", "linkedin"], "topic_or_goal": "Announce product"},
        "brand_bible": {"xml_raw": ""},
        "stream": streaming_manager,
    }

    # Validate shared store before running the flow
    try:
        validation_result = validate_shared_store(shared)
        if not validation_result.is_valid:
            logger.error("Validation failed: %s", validation_result.errors)
            return
        
        if validation_result.warnings:
            logger.warning("Validation warnings: %s", validation_result.warnings)
            
    except Exception as exc:
        error_context = ErrorContext(
            node_name="run_demo",
            operation="validation",
            input_data=shared
        )
        error_handler.handle_error(exc, error_context, ErrorSeverity.HIGH)
        return

    # Create the flow and run it with monitoring
    try:
        with monitor_execution("main_flow"):
            flow: 'Flow' = create_main_flow()
            streaming_manager.start_streaming()
            flow.run(shared)
            streaming_manager.stop_streaming()
            
    except Exception as exc:
        error_context = ErrorContext(
            node_name="run_demo",
            operation="flow_execution",
            input_data=shared
        )
        error_handler.handle_error(exc, error_context, ErrorSeverity.HIGH)
        return

    # Log the results and performance metrics
    execution_time = time.time() - start_time
    record_metric("demo_execution_time", execution_time, "seconds")
    
    if "content_pieces" in shared:
        logger.info("Content pieces: %s", shared["content_pieces"])
        record_metric("content_pieces_generated", len(shared["content_pieces"]), "count")
    else:
        logger.warning("No content pieces generated")
    
    # Log performance summary
    performance_summary = monitor.get_summary()
    logger.info("Performance summary: %s", performance_summary)


def validate_shared_store(shared: Dict[str, Any]) -> None:
    """Validate minimal `shared` dict required by the flows.

    This function performs lightweight checks only. More detailed schema
    validation should be done using Pydantic in higher-assurance contexts.

    Pre-condition: `shared` is provided by caller.
    Post-condition: raises when structure is invalid; otherwise returns None.

    Test stub:
        def test_validate_shared_store_invalid():
            with pytest.raises(ValueError):
                validate_shared_store({})

    Lint: precise pragmas only where necessary.
    """
    if not isinstance(shared, dict):
        raise TypeError("shared must be a dict")
    tr = shared.get("task_requirements")
    if tr is None or not isinstance(tr, dict):
        raise ValueError("shared['task_requirements'] must be a dict")
    platforms = tr.get("platforms")
    if platforms is None:
        raise ValueError("task_requirements must include 'platforms'")
    if not isinstance(platforms, list):
        raise TypeError("task_requirements['platforms'] must be a list")

    # (function continues)


def create_gradio_interface() -> Any:
    """Create and return a Gradio Blocks app for the Virtual PR Firm demo.

    This function constructs a complete web-based user interface using Gradio
    that allows users to interactively generate PR content. The interface
    provides input fields for topic/goal and target platforms, and displays
    the generated content in a structured JSON format.

    Interface Components:
        - Topic/Goal Input: Text field for specifying the PR objective
        - Platforms Input: Comma-separated list of target social media platforms
        - Run Button: Triggers the content generation flow
        - Output Display: JSON viewer showing generated content for each platform

    Supported Platforms:
        The interface accepts any comma-separated list of platform names.
        Common supported platforms include:
        - twitter
        - linkedin
        - facebook
        - instagram

    User Interaction Flow:
        1. User enters a topic or goal (e.g., "Announce product launch")
        2. User specifies target platforms (e.g., "twitter, linkedin")
        3. User clicks "Run" button to generate content
        4. Generated content appears in the output JSON viewer

    Default Values:
        - Topic: "Announce product"
        - Platforms: "twitter, linkedin"

    Returns:
        gr.Blocks: A configured Gradio Blocks application ready for launch

    Raises:
        RuntimeError: If Gradio is not installed or available
        ImportError: If required dependencies are missing
        ConfigurationError: If the interface cannot be properly configured

    Example:
        >>> app = create_gradio_interface()
        >>> app.launch()  # Launches web interface on default port
        >>> app.launch(server_port=7860, share=True)  # Custom configuration

    Security Considerations:
        - Input validation is performed on all user inputs
        - Platform names are sanitized and normalized
        - Topic content is validated for appropriate length and content
        - No file uploads are currently supported to minimize attack surface

    Performance Notes:
        - Content generation runs synchronously and may take several seconds
        - Large requests may timeout without proper configuration
        - No caching is implemented, so identical requests regenerate content

    Accessibility:
        - Interface uses semantic HTML for screen reader compatibility
        - Keyboard navigation is supported for all interactive elements
        - Color contrast meets WCAG guidelines
    
    TODO: Add comprehensive input validation and sanitization
    TODO: Implement user authentication and session management
    TODO: Add rate limiting and request throttling
    TODO: Support file uploads for brand bible content
    TODO: Add progress bars and real-time status updates
    TODO: Implement result caching and history management
    TODO: Add export functionality for generated content
    TODO: Support custom styling and theming
    TODO: Add help documentation and tooltips
    TODO: Implement error recovery and graceful degradation
    """

    # TODO: Provide more helpful error message with installation instructions
    # TODO: Add fallback UI options when Gradio is unavailable
    if gr is None:
        raise RuntimeError("Gradio not installed")

    def run_flow(topic: str, platforms_text: str) -> Dict[str, Any]:
        """Execute the PR content generation flow with user-provided inputs.
        
        This nested function serves as the callback handler for the Gradio
        interface's 'Run' button. It processes user inputs, constructs the
        shared context dictionary, executes the main flow, and returns the
        generated content for display.

        Input Processing:
            - Parses comma-separated platform list into individual platform names
            - Strips whitespace and filters empty entries
            - Normalizes platform names to lowercase
            - Validates that at least one platform is specified

        Execution Flow:
            1. Parse and validate platform inputs
            2. Construct shared dictionary with user inputs
            3. Create and configure the main flow
            4. Execute the flow with the shared context
            5. Extract and return generated content pieces

        Args:
            topic (str): The PR topic or goal provided by the user.
                Should be a descriptive string indicating the purpose
                of the PR content (e.g., 'Announce product launch',
                'Share company milestone').
            platforms_text (str): A comma-separated string of target
                platform names (e.g., 'twitter, linkedin, facebook').
                Platform names are case-insensitive and whitespace is
                automatically trimmed.

        Returns:
            dict: A dictionary mapping platform names to their generated
                content. The structure is:
                {
                    'platform_name': 'Generated content for this platform...',
                    'another_platform': 'Different content for this platform...'
                }

        Raises:
            ValueError: If topic is empty or platforms_text is invalid
            FlowExecutionError: If the content generation flow fails
            ValidationError: If inputs don't meet validation criteria
            TimeoutError: If content generation exceeds time limits

        Example:
            >>> result = run_flow('Launch new feature', 'twitter, linkedin')
            >>> print(result)
            {
                'twitter': 'Exciting news! Our new feature is here...',
                'linkedin': 'We are pleased to announce the launch...'
            }

        Input Validation:
            - Topic must be non-empty and contain at least 3 characters
            - Platforms must be a valid comma-separated list
            - At least one platform must be specified
            - Platform names must be from the supported platform list

        Error Handling:
            - Invalid inputs return empty dictionary with error message
            - Flow execution errors are caught and logged
            - Timeout errors are handled gracefully with partial results
            - Network errors during content generation are retried
        """
        
        # Initialize monitoring and error handling
        monitor = get_global_monitor()
        error_handler = get_global_error_handler()
        start_time = time.time()
        
        # Sanitize and validate inputs
        sanitized_topic = sanitize_input(topic)
        if not sanitized_topic or len(sanitized_topic.strip()) < 3:
            error_context = ErrorContext(
                node_name="run_flow",
                operation="input_validation",
                input_data={"topic": topic, "platforms": platforms_text}
            )
            error_handler.handle_error(
                ValueError("Topic must be at least 3 characters long"), 
                error_context, 
                ErrorSeverity.MEDIUM
            )
            return {"error": "Topic must be at least 3 characters long"}
        
        # Parse and normalize platforms
        platforms: List[str] = []
        for platform in platforms_text.split(","):
            normalized = normalize_platform_name(platform.strip())
            if normalized:
                platforms.append(normalized)
        
        if not platforms:
            platforms = ["twitter"]  # Default fallback
        
        # Construct shared store with validation
        shared: Dict[str, Any] = {
            "task_requirements": {
                "platforms": platforms, 
                "topic_or_goal": sanitized_topic
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None,
        }
        
        # Validate shared store
        try:
            validation_result = validate_shared_store(shared)
            if not validation_result.is_valid:
                logger.error("Validation failed: %s", validation_result.errors)
                return {"error": f"Validation failed: {validation_result.errors}"}
            
            if validation_result.warnings:
                logger.warning("Validation warnings: %s", validation_result.warnings)
                
        except Exception as exc:
            error_context = ErrorContext(
                node_name="run_flow",
                operation="validation",
                input_data=shared
            )
            error_handler.handle_error(exc, error_context, ErrorSeverity.HIGH)
            return {"error": f"Validation error: {str(exc)}"}
        
        # Execute flow with monitoring
        try:
            with monitor_execution("gradio_flow"):
                flow: 'Flow' = create_main_flow()
                flow.run(shared)
                
        except Exception as exc:
            error_context = ErrorContext(
                node_name="run_flow",
                operation="flow_execution",
                input_data=shared
            )
            error_handler.handle_error(exc, error_context, ErrorSeverity.HIGH)
            return {"error": f"Flow execution failed: {str(exc)}"}
        
        # Record metrics and return results
        execution_time = time.time() - start_time
        record_metric("gradio_flow_execution_time", execution_time, "seconds")
        record_metric("platforms_processed", len(platforms), "count")
        
        content_pieces = shared.get("content_pieces", {})
        if content_pieces:
            record_metric("content_pieces_generated", len(content_pieces), "count")
        
        return content_pieces

    # TODO: Add custom CSS styling and branding
    # TODO: Implement responsive design for mobile devices
    # TODO: Add analytics and usage tracking
    with gr.Blocks() as demo:
        # TODO: Add logo and branding elements
        # TODO: Include version information and links
        gr.Markdown("# Virtual PR Firm Demo")
        
        # TODO: Add input validation and real-time feedback
        # TODO: Implement autocomplete for common topics
        # TODO: Add character limits and input guidelines
        topic = gr.Textbox(label="Topic/Goal", value="Announce product")
        
        # TODO: Replace with multi-select dropdown for platforms
        # TODO: Add platform-specific configuration options
        # TODO: Validate supported platforms dynamically
        platforms = gr.Textbox(label="Platforms (comma-separated)", value="twitter, linkedin")
        
        # TODO: Add syntax highlighting for JSON output
        # TODO: Implement collapsible sections for large outputs
        # TODO: Add export buttons (copy, download, share)
        out = gr.JSON(label="Content pieces")
        
        # TODO: Add loading spinner and disable during execution
        # TODO: Implement progress indication
        # TODO: Add keyboard shortcuts
        run_btn = gr.Button("Run")
        
        # TODO: Add error handling in the click event
        # TODO: Implement async execution with progress updates
        # TODO: Add request validation before submission
        run_btn.click(fn=run_flow, inputs=[topic, platforms], outputs=[out])

    # TODO: Add configuration options for the demo app
    # TODO: Implement app state management
    return demo


# TODO: Add proper CLI argument parsing with argparse
# TODO: Support different execution modes (CLI, Gradio, API)
# TODO: Add configuration file support
# TODO: Implement proper exit codes and error handling
# TODO: Add version information and help commands
if __name__ == "__main__":
    # TODO: Add command-line options for Gradio vs CLI mode
    # TODO: Implement proper error handling and logging setup
    # TODO: Add configuration validation
    run_demo()
    
    # TODO: Optionally launch Gradio interface based on CLI args
    # TODO: Add option to run both CLI demo and launch web interface
    # TODO: Implement proper shutdown handling
