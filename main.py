"""Production-ready CLI/Gradio starter for the Virtual PR Firm.

This module provides a comprehensive interface for the Virtual PR Firm's content
generation capabilities with enterprise-grade features including configuration
management, structured logging, input validation, error handling, and security.

The module exposes the following key functionality:
- `run_demo()`: A CLI function that executes the main flow with sample data
- `create_gradio_interface()`: Creates a web-based interface using Gradio
- `main()`: Production CLI with argument parsing and configuration

Example Usage:
    Command Line:
        $ python main.py --demo  # Run CLI demo
        $ python main.py --serve --port 8080  # Launch web interface
        $ python main.py --config config.yaml  # Use custom config
    
    Programmatic:
        >>> from main import run_demo, create_gradio_interface
        >>> run_demo()  # Run CLI demo
        >>> app = create_gradio_interface()  # Create web interface
        >>> app.launch()  # Launch web interface

Features:
- Comprehensive error handling and structured logging
- Configuration management with environment variables and YAML files
- Input validation and sanitization with security checks
- Rate limiting and abuse protection
- Request correlation and distributed tracing
- Graceful error recovery and user feedback
"""

from flow import create_main_flow
from typing import Any, Dict, Optional, List, TYPE_CHECKING
import argparse
import sys
import os

# Import our new modules
from config import get_config
from logging_config import setup_logging, get_logger, set_request_id, log_error_with_context
from validation import validate_and_sanitize_inputs, ValidationError, check_rate_limit

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime dependency
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except Exception:
    gr = None  # type: Optional[Any]

# Initialize logging and configuration
config = get_config()
setup_logging(
    level=config.logging.level,
    format_type=config.logging.format,
    log_file=config.logging.file
)

# Module logger
logger = get_logger(__name__)


def run_demo() -> None:
    """
    Run a minimal CLI demo of the main generation flow using a sample shared store.
    
    Constructs a small shared payload (platforms and topic taken from config), validates it with validate_shared_store, creates the main flow via create_main_flow(), runs the flow, and prints the resulting `content_pieces` to stdout. Intended as a lightweight developer smoke-test that exercises wiring without requiring Gradio.
    
    Raises:
        ValidationError: If the constructed shared store fails validation.
        Exception: Propagates unexpected errors raised during flow creation or execution.
    """
    # Set request ID for correlation
    request_id = set_request_id()
    logger.info("Starting demo run", extra={"request_id": request_id})
    
    try:
        # Use configuration for demo data
        shared: Dict[str, Any] = {
            "task_requirements": {
                "platforms": config.gradio.get("default_platforms", ["twitter", "linkedin"]),
                "topic_or_goal": config.gradio.get("default_topic", "Announce product")
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None,
        }

        # Validate shared store before running the flow
        try:
            from validation import validate_shared_store
            validated_shared = validate_shared_store(shared)
            logger.info("Shared store validated successfully", extra={"request_id": request_id})
        except ValidationError as exc:
            logger.error("Invalid shared store: %s", exc, extra={"request_id": request_id})
            raise
        except Exception as exc:
            log_error_with_context(exc, {"request_id": request_id, "context": "shared_store_validation"})
            raise

        # Create and run the main flow
        logger.info("Creating main flow", extra={"request_id": request_id})
        flow: 'Flow' = create_main_flow()
        
        logger.info("Running flow", extra={"request_id": request_id})
        flow.run(validated_shared)

        # Output results
        content_pieces = validated_shared.get("content_pieces", {})
        logger.info("Demo completed successfully", extra={
            "request_id": request_id,
            "platforms_processed": len(content_pieces),
            "content_generated": bool(content_pieces)
        })
        
        print("Content pieces:", content_pieces)
        
    except Exception as exc:
        log_error_with_context(exc, {"request_id": request_id, "context": "demo_execution"})
        print(f"Demo failed: {exc}")
        raise


def validate_shared_store(shared: Dict[str, Any]) -> None:
    """
    Validate the minimal structure required for the flow's shared store.
    
    Checks that `shared` is a dict containing a `task_requirements` dict which in turn contains a `platforms` list.
    Raises a TypeError if `shared` or `platforms` are the wrong type, and a ValueError if required keys are missing.
    
    Parameters:
        shared (dict): The shared store passed into flows; must include
            `task_requirements` (dict) with a `platforms` (list) entry.
    
    Raises:
        TypeError: If `shared` is not a dict or `platforms` is not a list.
        ValueError: If `task_requirements` or `platforms` are missing.
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
    """
    Create and return a Gradio Blocks application for the interactive Virtual PR Firm demo.
    
    The returned app provides a simple UI with inputs for a topic/goal and a
    comma-separated list of target platforms, a Run button that executes the
    server-side content-generation flow, and a JSON output area that displays the
    generated content pieces keyed by platform.
    
    Returns:
        gr.Blocks: A configured Gradio Blocks app ready to be launched.
    
    Raises:
        RuntimeError: If Gradio is not installed or available at runtime.
    """

    # TODO: Provide more helpful error message with installation instructions
    # TODO: Add fallback UI options when Gradio is unavailable
    if gr is None:
        raise RuntimeError("Gradio not installed")

    def run_flow(topic: str, platforms_text: str) -> Dict[str, Any]:
        """
        Run the PR content-generation flow for Gradio input and return generated content.
        
        Validates and sanitizes the user-provided topic and comma-separated platform list, enforces rate limits, constructs the shared flow input, executes the main flow, and returns the resulting content mapping. On recoverable failures (validation, rate limiting, or flow errors) returns a dictionary with an "error" key describing the problem rather than raising.
        
        Parameters:
            topic (str): The user-provided topic or goal for the PR content.
            platforms_text (str): Comma-separated platform names (e.g., "twitter, linkedin").
        
        Returns:
            Dict[str, Any]: On success, a mapping from platform names to generated content pieces.
                On error, a dict of the form {"error": "<human-readable message>"}.
        """
        # Set request ID for correlation
        request_id = set_request_id()
        logger.info("Starting flow execution", extra={
            "request_id": request_id,
            "topic_length": len(topic),
            "platforms_text": platforms_text
        })
        
        try:
            # Rate limiting check
            if not check_rate_limit(
                identifier=f"gradio_{request_id}",
                max_requests=config.security.rate_limit_requests,
                window_seconds=config.security.rate_limit_window
            ):
                error_msg = "Rate limit exceeded. Please try again later."
                logger.warning("Rate limit exceeded", extra={"request_id": request_id})
                return {"error": error_msg}
            
            # Validate and sanitize inputs
            try:
                validated_shared = validate_and_sanitize_inputs(topic, platforms_text)
                logger.info("Inputs validated successfully", extra={"request_id": request_id})
            except ValidationError as exc:
                logger.warning("Input validation failed", extra={
                    "request_id": request_id,
                    "error": str(exc),
                    "field": exc.field
                })
                return {"error": f"Validation error: {exc.message}"}
            except Exception as exc:
                log_error_with_context(exc, {"request_id": request_id, "context": "input_validation"})
                return {"error": "Input validation failed. Please check your inputs."}
            
            # Create and run the flow
            try:
                logger.info("Creating main flow", extra={"request_id": request_id})
                flow: 'Flow' = create_main_flow()
                
                logger.info("Running flow", extra={"request_id": request_id})
                flow.run(validated_shared)
                
                # Extract results
                content_pieces = validated_shared.get("content_pieces", {})
                logger.info("Flow completed successfully", extra={
                    "request_id": request_id,
                    "platforms_processed": len(content_pieces),
                    "content_generated": bool(content_pieces)
                })
                
                return content_pieces
                
            except Exception as exc:
                log_error_with_context(exc, {"request_id": request_id, "context": "flow_execution"})
                return {"error": f"Content generation failed: {str(exc)}"}
                
        except Exception as exc:
            log_error_with_context(exc, {"request_id": request_id, "context": "run_flow"})
            return {"error": "An unexpected error occurred. Please try again."}

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


def main():
    """
    Main CLI entry point that parses arguments and runs the requested mode (demo, serve, or info/health).
    
    Runs one of three primary flows based on command-line flags:
    - --demo: Executes a lightweight CLI demo (calls run_demo()).
    - --serve: Builds and launches the Gradio web interface created by create_gradio_interface().
    - --version / --info / --health: Print version, system information, or perform basic dependency health checks and exit.
    
    Behavior:
    - Loads configuration via get_config() (optionally from --config) and applies CLI overrides for host, port, log level, and log file.
    - Initializes structured logging via setup_logging() using the final configuration.
    - When serving, verifies Gradio availability and launches the app with config-driven host/port/auth/share options.
    - Returns process-style exit codes: 0 on normal success, 130 on KeyboardInterrupt, 1 on unexpected errors.
    
    Side effects:
    - May start a web server (Gradio) and/or initialize global logging.
    - Prints informational and error messages to stdout/stderr.
    
    Notes:
    - This function handles all argument parsing and top-level error handling; it is intended to be the module's main entrypoint for both local demos and production serving.
    """
    parser = argparse.ArgumentParser(
        description="Virtual PR Firm - AI-powered content generation for social media platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run CLI demo
  python main.py --serve --port 8080       # Launch web interface on port 8080
  python main.py --serve --config config.yaml  # Use custom configuration
  python main.py --version                 # Show version information
  python main.py --health                  # Run health checks
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Run CLI demo with sample data"
    )
    mode_group.add_argument(
        "--serve",
        action="store_true",
        help="Launch Gradio web interface"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for web interface (overrides config)"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host for web interface (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (overrides config)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (overrides config)"
    )
    
    # Information options
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health checks and exit"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Handle information requests first
        if args.version:
            print("Virtual PR Firm v1.0.0")
            print("Built with PocketFlow framework")
            return 0
        
        if args.info:
            import sys
            import platform
            print("System Information:")
            print(f"  Python: {sys.version}")
            print(f"  Platform: {platform.platform()}")
            print(f"  Architecture: {platform.architecture()}")
            return 0
        
        if args.health:
            print("Running health checks...")
            # Basic health checks
            try:
                import gradio as gr
                print("✓ Gradio is available")
            except ImportError:
                print("✗ Gradio not available")
            
            try:
                from pocketflow import Flow
                print("✓ PocketFlow is available")
            except ImportError:
                print("✗ PocketFlow not available")
            
            print("Health checks completed")
            return 0
        
        # Load configuration
        if args.config:
            config = get_config(args.config)
        else:
            config = get_config()
        
        # Override config with CLI args
        if args.port:
            config.gradio.port = args.port
        if args.host:
            config.gradio.host = args.host
        if args.log_level:
            config.logging.level = args.log_level
        if args.log_file:
            config.logging.file = args.log_file
        
        # Setup logging with updated config
        setup_logging(
            level=config.logging.level,
            format_type=config.logging.format,
            log_file=config.logging.file
        )
        
        logger.info("Application starting", extra={
            "mode": "demo" if args.demo else "serve",
            "config_file": args.config,
            "port": config.gradio.port if args.serve else None
        })
        
        # Execute requested mode
        if args.demo:
            run_demo()
            return 0
        
        elif args.serve:
            if gr is None:
                logger.error("Gradio not available. Install with: pip install gradio")
                return 1
            
            app = create_gradio_interface()
            logger.info(f"Launching Gradio interface on {config.gradio.host}:{config.gradio.port}")
            app.launch(
                server_name=config.gradio.host,
                server_port=config.gradio.port,
                share=config.gradio.share,
                auth=config.gradio.auth,
                show_error=config.gradio.show_error
            )
            return 0
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 130
    except Exception as exc:
        log_error_with_context(exc, {"context": "main_cli"})
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
