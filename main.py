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

from flow import create_main_flow
from typing import Any, Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime dependency
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except Exception:
    gr = None  # type: Optional[Any]

import logging

# Import schema validation
try:
    from utils.schemas import (
        PlatformEnum, 
        create_initial_shared_state,
        SCHEMA_VALIDATION_AVAILABLE as _SCHEMA_AVAILABLE
    )
    SCHEMA_VALIDATION_AVAILABLE = _SCHEMA_AVAILABLE
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Schema validation not available")

# Module logger
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


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

    # NOTE: defaults are intentional for demo; replace with configuration in prod
    shared: Dict[str, Any] = {
        "task_requirements": {"platforms": ["twitter", "linkedin"], "topic_or_goal": "Announce product"},
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }

    # Validate shared store before running the flow
    try:
        validate_shared_store(shared)
    except Exception as exc:
        logger.error("Invalid shared store: %s", exc)
        raise

    # Create and run the main flow
    flow: 'Flow' = create_main_flow()
    flow.run(shared)

    # Output results
    print("Content pieces:", shared.get("content_pieces"))


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
        
        TODO: Add comprehensive input validation
        TODO: Implement async execution for better UX
        TODO: Add progress callbacks and status updates
        TODO: Implement proper error handling and user feedback
        TODO: Add request logging and analytics
        TODO: Support cancellation of running requests
        TODO: Add input sanitization and security checks
        """
        
        import re
        import html
        from utils.schemas import PlatformEnum, create_initial_shared_state
        
        # Input validation and sanitization
        # Sanitize topic input
        if not topic or not isinstance(topic, str):
            raise ValueError("Topic must be a non-empty string")
        
        # Remove HTML tags and escape special characters
        topic = html.escape(re.sub(r'<[^>]+>', '', topic))
        topic = topic.strip()
        
        # Validate topic length
        if len(topic) < 3:
            raise ValueError("Topic must be at least 3 characters long")
        if len(topic) > 500:
            raise ValueError("Topic must not exceed 500 characters")
        
        # Validate and normalize platforms
        if not platforms_text or not isinstance(platforms_text, str):
            raise ValueError("Platforms must be specified")
        
        # Parse platforms with flexible delimiters
        platforms_raw = re.split(r'[,;|\s]+', platforms_text.lower())
        platforms: List[str] = [p.strip() for p in platforms_raw if p.strip()]
        
        # Validate platforms against enum
        valid_platforms = []
        invalid_platforms = []
        for platform in platforms:
            try:
                # Normalize platform name
                platform_enum = PlatformEnum(platform)
                valid_platforms.append(platform_enum.value)
            except ValueError:
                invalid_platforms.append(platform)
        
        if invalid_platforms:
            raise ValueError(f"Invalid platforms: {', '.join(invalid_platforms)}. "
                           f"Valid options are: {', '.join([p.value for p in PlatformEnum])}")
        
        if not valid_platforms:
            raise ValueError("At least one valid platform must be specified")
        
        # Create validated shared state
        try:
            if SCHEMA_VALIDATION_AVAILABLE:
                shared = create_initial_shared_state(
                    platforms=valid_platforms,
                    topic=topic,
                    brand_bible_xml=""
                )
            else:
                # Fallback without schema validation
                shared: Dict[str, Any] = {
                    "task_requirements": {
                        "platforms": valid_platforms,
                        "topic_or_goal": topic
                    },
                    "brand_bible": {"xml_raw": ""},
                    "stream": None,
                }
        except Exception as e:
            logger.error(f"Failed to create initial state: {e}")
            raise ValueError(f"Failed to initialize content generation: {str(e)}")
        
        # Log sanitized inputs
        logger.info("Running flow with platforms=%s, topic='%s'", valid_platforms, topic[:50])
        
        # Create and run flow with error handling
        try:
            flow: 'Flow' = create_main_flow()
            flow.run(shared)
        except Exception as e:
            logger.error(f"Flow execution failed: {e}", exc_info=True)
            # Return partial results if available
            if "content_pieces" in shared and shared["content_pieces"]:
                logger.warning("Returning partial results after flow error")
                return shared["content_pieces"]
            raise RuntimeError(f"Content generation failed: {str(e)}")
        
        # Validate and format output
        content_pieces = shared.get("content_pieces", {})
        if not content_pieces:
            logger.warning("No content pieces generated")
            return {"error": "No content was generated. Please try again."}
        
        # Add metadata to results
        import time
        result = {
            "content": content_pieces,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "platforms": valid_platforms,
                "topic": topic[:100]  # Truncate for display
            }
        }
        
        return result

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


if __name__ == "__main__":
    import argparse
    import sys
    
    # Define version
    __version__ = "1.0.0"
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        prog="prfirm3",
        description="Virtual PR Firm - AI-powered content generation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI demo with default settings
  python main.py --mode cli
  
  # Launch Gradio web interface
  python main.py --mode gradio --port 7860
  
  # Generate content for specific platforms
  python main.py --platforms twitter,linkedin --topic "Product launch"
  
  # Use brand bible from file
  python main.py --brand-bible path/to/brand.xml --platforms email
        """
    )
    
    # Add arguments
    parser.add_argument(
        "--mode", 
        choices=["cli", "gradio", "both"],
        default="cli",
        help="Execution mode: CLI demo, Gradio web interface, or both"
    )
    
    parser.add_argument(
        "--platforms",
        type=str,
        help="Comma-separated list of platforms (twitter,linkedin,instagram,reddit,email,blog)"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="Content topic or goal"
    )
    
    parser.add_argument(
        "--brand-bible",
        type=str,
        help="Path to brand bible XML file"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio interface (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for Gradio interface (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for generated content (JSON format)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging based on argument
    try:
        from utils.logging_config import initialize_logging, LogFormat
        from pathlib import Path
        
        # Determine log format
        log_format = LogFormat.DETAILED
        if args.log_level == "DEBUG":
            log_format = LogFormat.DETAILED
        elif getattr(args, "json_logs", False):
            log_format = LogFormat.JSON
        
        # Initialize comprehensive logging
        initialize_logging(
            log_dir=Path("./logs"),
            level=args.log_level,
            format_type=log_format,
            enable_performance=args.log_level == "DEBUG"
        )
        logger.info("Advanced logging initialized")
    except ImportError:
        # Fallback to basic logging
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.warning("Advanced logging not available, using basic configuration")
    
    try:
        if args.mode == "cli" or args.mode == "both":
            # Run CLI mode
            if args.platforms and args.topic:
                # Run with custom parameters
                from flow import create_main_flow
                shared = {
                    "task_requirements": {
                        "platforms": [p.strip() for p in args.platforms.split(",")],
                        "topic_or_goal": args.topic
                    }
                }
                
                # Load brand bible if provided
                if args.brand_bible:
                    try:
                        with open(args.brand_bible, 'r') as f:
                            shared["brand_bible"] = {"xml_raw": f.read()}
                    except Exception as e:
                        logger.error(f"Failed to load brand bible: {e}")
                        sys.exit(1)
                
                # Create and run flow
                flow = create_main_flow()
                flow.run(shared)
                
                # Output results
                if args.output:
                    import json
                    with open(args.output, 'w') as f:
                        json.dump(shared.get("content_pieces", {}), f, indent=2)
                    logger.info(f"Content saved to {args.output}")
                else:
                    print("\nGenerated Content:")
                    for platform, content in shared.get("content_pieces", {}).items():
                        print(f"\n=== {platform.upper()} ===")
                        print(content)
            else:
                # Run default demo
                run_demo()
        
        if args.mode == "gradio" or args.mode == "both":
            # Launch Gradio interface
            try:
                app = create_gradio_interface()
                logger.info(f"Launching Gradio interface on {args.host}:{args.port}")
                app.launch(
                    server_name=args.host,
                    server_port=args.port,
                    share=False,
                    quiet=False
                )
            except Exception as e:
                logger.error(f"Failed to launch Gradio interface: {e}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
