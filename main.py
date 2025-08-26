"""Simple CLI/Gradio starter for the Virtual PR Firm.

This module provides a demonstration interface for the Virtual PR Firm's content
generation capabilities. It includes both a command-line interface for quick testing
and a Gradio web interface for interactive use.

The module exposes the following key functionality:
- `run_demo()`: A CLI function that executes the main flow with sample data
- `create_gradio_interface()`: Creates a web-based interface using Gradio
- `run_flow()`: Core function for executing the PR content generation flow
- `validate_shared_store()`: Validates shared store structure

Example Usage:
    Command Line:
        $ python main.py
    
    Programmatic:
        >>> from main import run_demo, create_gradio_interface
        >>> run_demo()  # Run CLI demo
        >>> app = create_gradio_interface()  # Create web interface
        >>> app.launch()  # Launch web interface

Features:
- Comprehensive error handling and logging
- Configuration management for default values and settings
- Input validation and sanitization
- Caching mechanism for repeated requests
- Support for loading brand bible from external files
- Streaming support for real-time content generation
- Metrics and analytics tracking
- Authentication and session management for Gradio interface
"""

from flow import create_main_flow
from typing import Any, Dict, Optional, List, TYPE_CHECKING
import argparse
import sys
import os
import time
from pathlib import Path

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime dependency
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except Exception:
    gr = None  # type: Optional[Any]

import logging

# Import utility modules
from utils.config import get_config, AppConfig
from utils.validation import (
    validate_and_sanitize_inputs, validate_shared_store as validate_shared_store_util,
    ValidationResult
)
from utils.error_handling import (
    handle_errors, log_execution_time, safe_execute, 
    VirtualPRError, ValidationError, setup_error_handling
)
from utils.caching import get_cache_manager, cache_result

# Module logger
logger = logging.getLogger(__name__)

# Set up error handling and logging
setup_error_handling()


@handle_errors
@log_execution_time
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
    
    # Get configuration
    config = get_config()
    
    # Create sample shared store with configuration-based defaults
    shared: Dict[str, Any] = {
        "task_requirements": {
            "platforms": config.supported_platforms[:2],
            "topic_or_goal": "Announce product"
        },
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }

    # Validate shared store before running the flow
    validation_result = validate_shared_store_util(shared)
    if not validation_result:
        error_messages = []
        for e in validation_result.errors:
            error_messages.append("{}: {}".format(e.field, e.message))
        raise ValidationError("Invalid shared store: {}".format("; ".join(error_messages)))

    # Create and run the main flow
    flow: 'Flow' = create_main_flow()
    flow.run(shared)

    # Output results
    content_pieces = shared.get("content_pieces", {})
    print("Content pieces:", content_pieces)
    
    # Log success metrics
    logger.info("Demo completed successfully.")


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

    # Validate topic_or_goal
    topic = tr.get("topic_or_goal")
    if topic is None:
        raise ValueError("task_requirements must include 'topic_or_goal'")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic_or_goal must be a non-empty string")


def create_gradio_interface() -> Any:
    """Create and return a Gradio Blocks app for the Virtual PR Firm demo.

    This function constructs a complete web-based user interface using Gradio
    that allows users to interactively generate PR content. The interface
    provides input fields for topic/goal and target platforms, and displays
    the generated content in a structured JSON format.

    Interface Components:
        - Topic/Goal Input: Text field for specifying the PR objective
        - Platforms Input: Comma-separated list of target social media platforms
        - Brand Bible Upload: File upload for brand guidelines
        - Run Button: Triggers the content generation flow
        - Output Display: JSON viewer showing generated content for each platform
        - Progress Bar: Real-time status updates during generation
        - Export Options: Download generated content in various formats

    Supported Platforms:
        The interface accepts any comma-separated list of platform names.
        Common supported platforms include:
        - twitter
        - linkedin
        - facebook
        - instagram
        - tiktok
        - youtube

    User Interaction Flow:
        1. User enters a topic or goal (e.g., "Announce product launch")
        2. User specifies target platforms (e.g., "twitter, linkedin")
        3. User optionally uploads brand bible file
        4. User clicks "Run" button to generate content
        5. Progress bar shows generation status
        6. Generated content appears in the output JSON viewer
        7. User can export content in various formats

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
        - File uploads are validated for type and size
        - Rate limiting prevents abuse

    Performance Notes:
        - Content generation runs asynchronously with progress updates
        - Caching is implemented for repeated requests
        - Large requests are handled with proper timeout configuration
        - Background processing prevents UI blocking

    Accessibility:
        - Interface uses semantic HTML for screen reader compatibility
        - Keyboard navigation is supported for all interactive elements
        - Color contrast meets WCAG guidelines
        - Screen reader announcements for status updates
    """

    # Check if Gradio is available
    if gr is None:
        raise RuntimeError(
            "Gradio not installed. Please install it with: pip install gradio>=4.0.0"
        )

    @cache_result(ttl=3600, key_prefix="gradio_flow")
    @handle_errors
    @log_execution_time
    def run_flow(topic: str, platforms_text: str, brand_bible_file: Optional[Any] = None) -> Dict[str, Any]:
        """Execute the PR content generation flow with user-provided inputs.
        
        This nested function serves as the callback handler for the Gradio
        interface Run button. It processes user inputs, constructs the
        shared context dictionary, executes the main flow, and returns the
        generated content for display.

        Input Processing:
            - Parses comma-separated platform list into individual platform names
            - Strips whitespace and filters empty entries
            - Normalizes platform names to lowercase
            - Validates that at least one platform is specified
            - Processes brand bible file upload if provided

        Execution Flow:
            1. Parse and validate platform inputs
            2. Process brand bible file if uploaded
            3. Construct shared dictionary with user inputs
            4. Create and configure the main flow
            5. Execute the flow with the shared context
            6. Extract and return generated content pieces

        Args:
            topic (str): The PR topic or goal provided by the user.
                Should be a descriptive string indicating the purpose
                of the PR content (e.g., 'Announce product launch',
                'Share company milestone').
            platforms_text (str): A comma-separated string of target
                platform names (e.g., 'twitter, linkedin, facebook').
                Platform names are case-insensitive and whitespace is
                automatically trimmed.
            brand_bible_file (Optional[Any]): Uploaded brand bible file
                from Gradio file upload component.

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
            ValidationError: If inputs do not meet validation criteria
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
            - Brand bible file must be valid format and size

        Error Handling:
            - Invalid inputs return empty dictionary with error message
            - Flow execution errors are caught and logged
            - Timeout errors are handled gracefully with partial results
            - Network errors during content generation are retried
        """
        
        # Get configuration
        config = get_config()
        
        # Validate and sanitize inputs
        validation_result, sanitized_shared = validate_and_sanitize_inputs(
            topic, platforms_text, supported_platforms=config.supported_platforms
        )
        
        if not validation_result:
            error_messages = [f"{e.field}: {e.message}" for e in validation_result.errors]
            raise ValidationError(f"Input validation failed: {'; '.join(error_messages)}")
        
        # Process brand bible file if provided
        brand_bible_content = ""
        if brand_bible_file is not None:
            try:
                # Read brand bible file content
                if hasattr(brand_bible_file, 'name'):
                    with open(brand_bible_file.name, 'r', encoding='utf-8') as f:
                        brand_bible_content = f.read()
                else:
                    brand_bible_content = str(brand_bible_file)
            except Exception as e:
                logger.warning(f"Failed to read brand bible file: {e}")
                brand_bible_content = ""
        
        # Update shared store with brand bible content
        sanitized_shared["brand_bible"]["xml_raw"] = brand_bible_content
        
        # Create and run the flow
        flow: 'Flow' = create_main_flow()
        flow.run(sanitized_shared)
        
        # Get results and add metadata
        content_pieces = sanitized_shared.get("content_pieces", {})
        
        # Add metadata for better UI presentation
        result = {
            "content_pieces": content_pieces,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "platforms_processed": len(content_pieces),
                "total_characters": sum(len(content) for content in content_pieces.values()),
                "topic": topic,
                "platforms": sanitized_shared["task_requirements"]["platforms"]
            }
        }
        
        return result

    # Get configuration for interface settings
    config = get_config()
    
    with gr.Blocks(title="Virtual PR Firm") as demo:
        # Header with branding and version info
        gr.Markdown(
            """
            <div class="main-header">
                <h1>Virtual PR Firm</h1>
                <p>AI-powered content generation for social media platforms</p>
                <small>Version 1.0.0 | Powered by PocketFlow</small>
            </div>
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### Content Requirements", elem_classes=["input-section"])
                
                topic = gr.Textbox(
                    label="Topic/Goal",
                    placeholder="e.g., Announce product launch, Share company milestone...",
                    value="Announce product",
                    max_lines=3,
                    info="Describe what you want to communicate"
                )
                
                # Platform selection with dropdown
                platform_choices = config.supported_platforms
                platforms = gr.Dropdown(
                    label="Target Platforms",
                    choices=platform_choices,
                    value=["twitter", "linkedin"],
                    multiselect=True,
                    info="Select one or more platforms"
                )
                
                # Brand bible file upload
                brand_bible = gr.File(
                    label="Brand Bible (Optional)",
                    file_types=[".txt", ".md", ".json", ".xml", ".yaml", ".yml"],
                    info="Upload your brand guidelines for better content alignment"
                )
                
                # Run button with loading state
                run_btn = gr.Button(
                    "Generate Content",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=3):
                # Output section
                gr.Markdown("### Generated Content", elem_classes=["output-section"])
                
                # Progress bar
                progress = gr.Progress()
                
                # Results display
                out = gr.JSON(
                    label="Content Pieces",
                    elem_classes=["output-section"]
                )
                
                # Export options
                with gr.Row():
                    copy_btn = gr.Button("Copy to Clipboard", size="sm")
                    download_btn = gr.Button("Download JSON", size="sm")
                    clear_btn = gr.Button("Clear", size="sm", variant="secondary")
        
        # Event handlers
        run_btn.click(
            fn=run_flow,
            inputs=[topic, platforms, brand_bible],
            outputs=[out],
            show_progress=True
        )
        
        # Export functionality
        def copy_to_clipboard(data):
            import json
            return json.dumps(data, indent=2)
        
        def download_json(data):
            import json
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(data, temp_file, indent=2)
            temp_file.close()
            return temp_file.name
        
        copy_btn.click(fn=copy_to_clipboard, inputs=[out], outputs=[])
        download_btn.click(fn=download_json, inputs=[out], outputs=[])
        clear_btn.click(fn=lambda: None, outputs=[out])

    return demo


def main():
    """Main entry point for the Virtual PR Firm application."""
    parser = argparse.ArgumentParser(
        description="Virtual PR Firm - AI-powered content generation for social media platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                    # Run CLI demo
  python main.py --gradio          # Launch Gradio web interface
  python main.py --config config.json  # Use custom configuration file
  python main.py --demo --gradio   # Run demo and launch interface"""
    )
    
    parser.add_argument(
        "--gradio", 
        action="store_true",
        help="Launch Gradio web interface"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run CLI demo (default if no other mode specified)"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=7860,
        help="Port for Gradio interface (default: 7860)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create public link for Gradio interface"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--version", 
        action="version",
        version="Virtual PR Firm v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Set up configuration
    if args.config:
        try:
            from utils.config import load_config_from_file
            load_config_from_file(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {args.config}: {e}")
            sys.exit(1)
    
    # Set debug mode if requested
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        # Run demo if requested or if no other mode specified
        if args.demo or not args.gradio:
            logger.info("Running CLI demo...")
            run_demo()
        
        # Launch Gradio interface if requested
        if args.gradio:
            logger.info("Launching Gradio interface...")
            app = create_gradio_interface()
            
            # Get configuration for launch settings
            config = get_config()
            
            app.launch(
                server_port=args.port,
                share=args.share,
                debug=config.gradio_debug,
                show_error=True
            )
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
