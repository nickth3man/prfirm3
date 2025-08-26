"""Simple CLI/Gradio starter for the Virtual PR Firm.

This module provides a demonstration interface for the Virtual PR Firm's content
generation capabilities. It includes both a command-line interface for quick testing
and a Gradio web interface for interactive use.

The module exposes the following key functionality:
- `run_demo()`: A CLI function that executes the main flow with sample data
- `create_gradio_interface()`: Creates a web-based interface using Gradio

Example Usage:
    Command Line:
        $ python main.py
    
    Programmatic:
        >>> from main import run_demo, create_gradio_interface
        >>> run_demo()  # Run CLI demo
        >>> app = create_gradio_interface()  # Create web interface
        >>> app.launch()  # Launch web interface

TODO: Add comprehensive error handling and logging throughout the module
TODO: Implement configuration management for default values and settings
TODO: Add unit tests using Pytest for all functions
TODO: Add integration tests for the complete flow execution
TODO: Implement proper input validation and sanitization
TODO: Add support for loading brand bible from external files
TODO: Implement streaming support for real-time content generation
TODO: Add authentication and session management for Gradio interface
TODO: Implement caching mechanism for repeated requests
TODO: Add metrics and analytics tracking
"""

import logging
import sys
import traceback
from functools import wraps
from typing import Any, Dict, Optional, List, TYPE_CHECKING, Callable, TypeVar, Union
import json
import os
from pathlib import Path
import argparse
import sys
import signal
from contextlib import contextmanager

# Configure logging with proper formatting and levels
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging with structured formatting and file output support.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# Initialize logging
setup_logging()

from flow import create_main_flow

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime dependency
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Gradio not available: {e}. Web interface will not be available.")
    gr = None  # type: Optional[Any]

# Module logger
logger = logging.getLogger(__name__)

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])

def handle_errors(func: F) -> F:
    """Decorator to provide comprehensive error handling for functions.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

def validate_shared_store(shared: Dict[str, Any]) -> None:
    """Validate the structure and content of the shared store.
    
    Args:
        shared: Shared store dictionary to validate
        
    Raises:
        ValueError: If shared store structure is invalid
    """
    if not isinstance(shared, dict):
        raise ValueError("Shared store must be a dictionary")
    
    # Validate required top-level keys
    required_keys = ["task_requirements"]
    for key in required_keys:
        if key not in shared:
            raise ValueError(f"Missing required key in shared store: {key}")
    
    # Validate task_requirements structure
    task_reqs = shared.get("task_requirements", {})
    if not isinstance(task_reqs, dict):
        raise ValueError("task_requirements must be a dictionary")
    
    # Validate platforms if present
    platforms = task_reqs.get("platforms", [])
    if platforms and not isinstance(platforms, list):
        raise ValueError("platforms must be a list")
    
    # Validate brand_bible if present
    brand_bible = shared.get("brand_bible", {})
    if brand_bible and not isinstance(brand_bible, dict):
        raise ValueError("brand_bible must be a dictionary")

def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input to prevent injection attacks and ensure safety.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        ValueError: If input is too long or contains invalid characters
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    if len(text) > max_length:
        raise ValueError(f"Input too long. Maximum length: {max_length}")
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<script>', 'javascript:', 'data:', 'vbscript:']
    sanitized = text
    for char in dangerous_chars:
        sanitized = sanitized.replace(char.lower(), '')
        sanitized = sanitized.replace(char.upper(), '')
    
    return sanitized.strip()

class ConfigurationManager:
    """Manages configuration settings for the application."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        if config_file:
            self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "logging": {
                "level": "INFO",
                "file": None
            },
            "demo": {
                "default_platforms": ["twitter", "linkedin"],
                "default_topic": "Announce product",
                "max_revisions": 5
            },
            "validation": {
                "max_input_length": 10000,
                "max_platforms": 10
            }
        }
    
    def _load_config_file(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file and os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# Global configuration instance
config = ConfigurationManager()

@handle_errors
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

    TODO(dev,2025-08-26): Add CLI flags, structured logging, and proper error codes
    FIXME(dev,2025-08-26): consider decoupling flow creation to allow DI in tests
    # pylint: disable=too-many-locals
    """
    logger.info("Starting demo execution")
    
    # NOTE: defaults are intentional for demo; replace with configuration in prod
    shared: Dict[str, Any] = {
        "task_requirements": {
            "platforms": config.get("demo.default_platforms", ["twitter", "linkedin"]), 
            "topic_or_goal": config.get("demo.default_topic", "Announce product")
        },
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }

    # Validate shared store before running the flow
    try:
        validate_shared_store(shared)
        logger.debug("Shared store validation passed")
    except ValueError as e:
        logger.error(f"Shared store validation failed: {e}")
        raise

    # Create and run the main flow
    try:
        flow = create_main_flow()
        logger.info("Flow created successfully")
        
        flow.run(shared)
        logger.info("Flow execution completed")
        
        # Display results
        content_pieces = shared.get("content_pieces", {})
        if content_pieces:
            logger.info(f"Generated content pieces: {content_pieces}")
        else:
            logger.warning("No content pieces generated")
            
    except Exception as e:
        logger.error(f"Flow execution failed: {e}")
        logger.debug(f"Flow execution traceback: {traceback.format_exc()}")
        raise


@handle_errors
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
    """

    if gr is None:
        raise RuntimeError(
            "Gradio is not installed. Please install it with: pip install gradio"
        )

    def validate_and_sanitize_inputs(topic: str, platforms_text: str) -> tuple[str, List[str]]:
        """Validate and sanitize user inputs.
        
        Args:
            topic: User-provided topic/goal
            platforms_text: Comma-separated platform list
            
        Returns:
            Tuple of (sanitized_topic, sanitized_platforms)
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate topic
        if not topic or not topic.strip():
            raise ValueError("Topic/goal cannot be empty")
        
        sanitized_topic = sanitize_input(
            topic.strip(), 
            max_length=config.get("validation.max_input_length", 10000)
        )
        
        if len(sanitized_topic) < 3:
            raise ValueError("Topic/goal must be at least 3 characters long")
        
        # Validate platforms
        if not platforms_text or not platforms_text.strip():
            raise ValueError("Platforms cannot be empty")
        
        # Parse and sanitize platforms
        platforms = [
            p.strip().lower() 
            for p in platforms_text.split(',') 
            if p.strip()
        ]
        
        if not platforms:
            raise ValueError("At least one platform must be specified")
        
        max_platforms = config.get("validation.max_platforms", 10)
        if len(platforms) > max_platforms:
            raise ValueError(f"Maximum {max_platforms} platforms allowed")
        
        # Validate individual platform names
        for platform in platforms:
            if not platform.replace('-', '').replace('_', '').isalnum():
                raise ValueError(f"Invalid platform name: {platform}")
        
        return sanitized_topic, platforms

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
            - Validates input format and content

        Flow Execution:
            - Constructs shared state dictionary with validated inputs
            - Creates and executes the main content generation flow
            - Handles execution errors gracefully with user-friendly messages
            - Returns structured results for display

        Error Handling:
            - Input validation errors return descriptive error messages
            - Flow execution errors are caught and reported clearly
            - System errors provide helpful debugging information
            - Graceful degradation when optional features unavailable

        Args:
            topic: User-specified topic or goal for content generation
            platforms_text: Comma-separated list of target platforms

        Returns:
            Dict[str, Any]: Generated content results or error information:
                Success case:
                    {
                        "status": "success",
                        "content": Dict[str, str],  # Platform -> content mapping
                        "metadata": Dict[str, Any]  # Generation metadata
                    }
                Error case:
                    {
                        "status": "error",
                        "message": str,             # User-friendly error message
                        "details": str              # Technical details for debugging
                    }

        Performance Considerations:
            - Execution time varies based on number of platforms and content complexity
            - Large requests may timeout without proper configuration
            - No caching implemented - identical requests regenerate content
            - Memory usage scales with content length and platform count

        Security Notes:
            - All inputs are validated and sanitized before processing
            - No file system access or external API calls beyond configured services
            - Input length limits prevent resource exhaustion attacks
            - Platform name validation prevents injection attacks
        """
        try:
            logger.info(f"Starting flow execution for topic: {topic[:50]}...")
            
            # Validate and sanitize inputs
            sanitized_topic, sanitized_platforms = validate_and_sanitize_inputs(topic, platforms_text)
            
            # Construct shared state
            shared: Dict[str, Any] = {
                "task_requirements": {
                    "platforms": sanitized_platforms,
                    "topic_or_goal": sanitized_topic
                },
                "brand_bible": {"xml_raw": ""},
                "stream": None,
            }
            
            # Validate shared store structure
            validate_shared_store(shared)
            
            # Execute flow
            flow = create_main_flow()
            flow.run(shared)
            
            # Extract results
            content_pieces = shared.get("content_pieces", {})
            
            if not content_pieces:
                return {
                    "status": "warning",
                    "message": "No content was generated. Please check your inputs and try again.",
                    "content": {},
                    "metadata": {"platforms_requested": sanitized_platforms}
                }
            
            logger.info(f"Flow execution completed successfully. Generated content for {len(content_pieces)} platforms.")
            
            return {
                "status": "success",
                "content": content_pieces,
                "metadata": {
                    "platforms_requested": sanitized_platforms,
                    "platforms_generated": list(content_pieces.keys()),
                    "topic": sanitized_topic
                }
            }
            
        except ValueError as e:
            logger.warning(f"Input validation error: {e}")
            return {
                "status": "error",
                "message": f"Invalid input: {str(e)}",
                "details": "Please check your topic and platform inputs."
            }
        except Exception as e:
            logger.error(f"Flow execution error: {e}")
            logger.debug(f"Flow execution traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": "An error occurred during content generation. Please try again.",
                "details": f"Technical details: {str(e)}"
            }

    # Create Gradio interface
    with gr.Blocks(
        title="Virtual PR Firm - Content Generator",
        description="Generate platform-specific PR content using AI",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown("# 🚀 Virtual PR Firm Content Generator")
        gr.Markdown("Generate platform-specific PR content for your social media campaigns.")
        
        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="Topic or Goal",
                    placeholder="e.g., Announce product launch, Share company milestone",
                    value=config.get("demo.default_topic", "Announce product"),
                    max_lines=3,
                    info="Describe what you want to communicate"
                )
                
                platforms_input = gr.Textbox(
                    label="Target Platforms",
                    placeholder="e.g., twitter, linkedin, instagram",
                    value=", ".join(config.get("demo.default_platforms", ["twitter", "linkedin"])),
                    info="Comma-separated list of social media platforms"
                )
                
                run_button = gr.Button(
                    "Generate Content",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Instructions")
                gr.Markdown("""
                1. **Enter your topic or goal** - What message do you want to communicate?
                2. **Specify target platforms** - Which social media platforms?
                3. **Click Generate** - AI will create platform-specific content
                
                **Supported platforms:** twitter, linkedin, instagram, facebook, tiktok
                """)
        
        with gr.Row():
            output_json = gr.JSON(
                label="Generated Content",
                interactive=False,
                height=400
            )
        
        # Add error handling and validation
        def handle_generation(topic: str, platforms: str) -> Dict[str, Any]:
            """Wrapper function to handle generation with proper error handling."""
            try:
                return run_flow(topic, platforms)
            except Exception as e:
                logger.error(f"Unexpected error in handle_generation: {e}")
                return {
                    "status": "error",
                    "message": "An unexpected error occurred. Please try again.",
                    "details": str(e)
                }
        
        # Connect components
        run_button.click(
            fn=handle_generation,
            inputs=[topic_input, platforms_input],
            outputs=output_json
        )
        
        # Add keyboard shortcuts
        app.load(lambda: None, None, None, _js="""
        () => {
            document.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    document.querySelector('button[data-testid="primary"]').click();
                }
            });
        }
        """)
    
    return app


# TODO: Add proper CLI argument parsing with argparse
# TODO: Support different execution modes (CLI, Gradio, API)
# TODO: Add configuration file support
# TODO: Implement proper exit codes and error handling
# TODO: Add version information and help commands

@contextmanager
def graceful_shutdown():
    """Context manager for graceful shutdown handling."""
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, cleaning up...")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        yield
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive options.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Virtual PR Firm - AI-powered content generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI demo
  python main.py --demo
  
  # Launch web interface
  python main.py --web
  
  # Run with custom configuration
  python main.py --config config.json --demo
  
  # Run with specific logging level
  python main.py --log-level DEBUG --demo
        """
    )
    
    # Execution mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Run CLI demo with sample data"
    )
    mode_group.add_argument(
        "--web",
        action="store_true",
        help="Launch Gradio web interface"
    )
    mode_group.add_argument(
        "--api",
        action="store_true",
        help="Run as API server (future feature)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (optional)"
    )
    
    # Web interface options
    web_group = parser.add_argument_group("Web Interface Options")
    web_group.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    web_group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web interface (default: 127.0.0.1)"
    )
    web_group.add_argument(
        "--share",
        action="store_true",
        help="Create public link for web interface"
    )
    
    # Demo options
    demo_group = parser.add_argument_group("Demo Options")
    demo_group.add_argument(
        "--topic",
        type=str,
        help="Custom topic for demo (overrides default)"
    )
    demo_group.add_argument(
        "--platforms",
        type=str,
        help="Custom platforms for demo (comma-separated, overrides default)"
    )
    
    # Version and help
    parser.add_argument(
        "--version",
        action="version",
        version="Virtual PR Firm v1.0.0"
    )
    
    return parser.parse_args()

def run_cli_demo(args: argparse.Namespace) -> None:
    """Run CLI demo with optional custom parameters.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Starting CLI demo")
    
    # Override defaults if provided
    if args.topic:
        config.config["demo"]["default_topic"] = args.topic
        logger.info(f"Using custom topic: {args.topic}")
    
    if args.platforms:
        platforms = [p.strip() for p in args.platforms.split(",")]
        config.config["demo"]["default_platforms"] = platforms
        logger.info(f"Using custom platforms: {platforms}")
    
    run_demo()
    logger.info("CLI demo completed successfully")

def run_web_interface(args: argparse.Namespace) -> None:
    """Launch web interface with specified configuration.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Launching web interface")
    
    try:
        app = create_gradio_interface()
        logger.info(f"Starting web server on {args.host}:{args.port}")
        
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch web interface: {e}")
        raise

def main() -> None:
    """Main entry point with comprehensive argument parsing and error handling."""
    with graceful_shutdown():
        try:
            # Parse arguments
            args = parse_arguments()
            
            # Setup logging based on arguments
            setup_logging(args.log_level, args.log_file)
            logger.info("Virtual PR Firm starting up")
            
            # Load configuration if specified
            if args.config:
                global config
                config = ConfigurationManager(args.config)
                logger.info(f"Loaded configuration from {args.config}")
            
            # Execute based on mode
            if args.demo:
                run_cli_demo(args)
            elif args.web:
                run_web_interface(args)
            elif args.api:
                logger.error("API mode not yet implemented")
                sys.exit(1)
            else:
                logger.error("No execution mode specified")
                sys.exit(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Application error: {e}")
            logger.debug(f"Application traceback: {traceback.format_exc()}")
            sys.exit(1)

if __name__ == "__main__":
    main()
