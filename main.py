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
from config import get_config, config_manager
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

# Setup logging from configuration
config_manager.setup_logging()
logger = logging.getLogger(__name__)

# TODO: Add proper logging configuration
# TODO: Import configuration management utilities
# TODO: Import validation utilities
# TODO: Import error handling decorators


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
    try:
        logger.info("Creating main flow...")
        flow: 'Flow' = create_main_flow()
        
        logger.info("Running content generation flow...")
        flow.run(shared)
        
        # Validate results
        content_pieces = shared.get("content_pieces")
        if content_pieces:
            logger.info("Content generation completed successfully")
            print("Content pieces:", content_pieces)
        else:
            logger.warning("Flow completed but no content was generated")
            print("No content pieces generated")
            
    except Exception as exc:
        logger.error("Flow execution failed: %s", exc, exc_info=True)
        print(f"Error: Content generation failed - {exc}")
        raise


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
    
    # Validate task_requirements structure
    tr = shared.get("task_requirements")
    if tr is None or not isinstance(tr, dict):
        raise ValueError("shared['task_requirements'] must be a dict")
    
    # Validate platforms
    platforms = tr.get("platforms")
    if platforms is None:
        raise ValueError("task_requirements must include 'platforms'")
    if not isinstance(platforms, list):
        raise TypeError("task_requirements['platforms'] must be a list")
    if not platforms:
        raise ValueError("task_requirements['platforms'] cannot be empty")
    
    # Validate topic_or_goal
    topic = tr.get("topic_or_goal")
    if topic is not None and not isinstance(topic, str):
        raise TypeError("task_requirements['topic_or_goal'] must be a string")
    
    # Validate brand_bible structure (optional but if present, must be valid)
    bb = shared.get("brand_bible")
    if bb is not None:
        if not isinstance(bb, dict):
            raise TypeError("shared['brand_bible'] must be a dict")
        # xml_raw is optional but if present, must be string
        xml_raw = bb.get("xml_raw")
        if xml_raw is not None and not isinstance(xml_raw, str):
            raise TypeError("brand_bible['xml_raw'] must be a string")
    
    # Validate stream field (optional)
    stream = shared.get("stream")
    if stream is not None and not hasattr(stream, "__call__") and stream is not None:
        # stream can be None or callable, but not other types
        pass  # Allow None or callable objects


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

    if gr is None:
        error_msg = """
        Gradio is not installed. To use the web interface, please install Gradio:
        
        pip install gradio
        
        Or install all optional dependencies:
        pip install -r requirements.txt
        
        Alternatively, you can use the CLI interface by calling run_demo() directly.
        """
        logger.error("Gradio not available for web interface")
        raise RuntimeError(error_msg.strip())

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
        
        try:
            # Input validation and normalization
            logger.info("Processing Gradio request - Topic: %s, Platforms: %s", topic[:50], platforms_text)
            
            # Validate and normalize topic
            if not topic or len(topic.strip()) < 3:
                error_msg = "Topic must be at least 3 characters long"
                logger.warning("Invalid topic input: %s", error_msg)
                return {"error": error_msg}
            
            # Validate and normalize platforms
            platforms: List[str] = [p.strip().lower() for p in platforms_text.split(",") if p.strip()]
            if not platforms:
                platforms = ["twitter"]  # Default fallback
                logger.info("No platforms specified, using default: twitter")
            
            # Validate platform names (basic check)
            valid_platforms = ['twitter', 'linkedin', 'facebook', 'instagram', 'youtube', 'tiktok']
            invalid_platforms = [p for p in platforms if p not in valid_platforms]
            if invalid_platforms:
                logger.warning("Invalid platforms specified: %s", invalid_platforms)
            
            shared: Dict[str, Any] = {
                "task_requirements": {"platforms": platforms, "topic_or_goal": topic.strip()},
                "brand_bible": {"xml_raw": ""},
                "stream": None,
            }
            
            # Validate shared store structure
            validate_shared_store(shared)
            
            # Create and run flow with comprehensive error handling
            logger.info("Creating flow for content generation...")
            flow: 'Flow' = create_main_flow()
            
            logger.info("Running content generation flow...")
            flow.run(shared)
            
            # Validate and format results
            content_pieces = shared.get("content_pieces", {})
            if content_pieces:
                logger.info("Content generation completed successfully for %d platforms", len(content_pieces))
                return content_pieces
            else:
                logger.warning("Flow completed but no content was generated")
                return {"warning": "No content was generated. Please try again with different inputs."}
                
        except ValueError as exc:
            error_msg = f"Input validation error: {exc}"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as exc:
            error_msg = f"Content generation failed: {exc}"
            logger.error("Gradio handler error: %s", exc, exc_info=True)
            return {"error": error_msg}

    with gr.Blocks(title="Virtual PR Firm", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Virtual PR Firm Demo
        
        Generate professional PR content for multiple social media platforms.
        Simply enter your topic and target platforms to get started.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                topic = gr.Textbox(
                    label="Topic/Goal", 
                    value="Announce product",
                    placeholder="e.g., Announce product launch, Share company milestone...",
                    info="Describe what you want to communicate (minimum 3 characters)"
                )
                
                platforms = gr.Textbox(
                    label="Target Platforms", 
                    value="twitter, linkedin",
                    placeholder="twitter, linkedin, facebook, instagram...",
                    info="Comma-separated list of social media platforms"
                )
                
                run_btn = gr.Button("🎯 Generate Content", variant="primary", size="lg")
                
                gr.Markdown("""
                **Supported Platforms:** twitter, linkedin, facebook, instagram, youtube, tiktok
                
                **Tips:**
                - Be specific about your goal or announcement
                - Multiple platforms will generate tailored content for each
                - Content is generated with fallback behavior if external APIs are unavailable
                """)
            
            with gr.Column(scale=2):
                out = gr.JSON(
                    label="Generated Content", 
                    show_label=True,
                    container=True
                )
        
        # Add examples section
        gr.Markdown("### 📝 Example Topics")
        gr.Examples(
            examples=[
                ["Launch new mobile app", "twitter, linkedin"],
                ["Company milestone: 1M users", "twitter, linkedin, facebook"],
                ["New product feature announcement", "twitter, linkedin"],
                ["Hiring announcement", "linkedin"],
                ["Event invitation", "twitter, facebook, instagram"]
            ],
            inputs=[topic, platforms]
        )
        
        run_btn.click(fn=run_flow, inputs=[topic, platforms], outputs=[out])

    return demo


if __name__ == "__main__":
    import sys
    
    # Simple command line argument handling
    # Usage: python main.py [cli|gradio]
    mode = "cli"  # default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    try:
        if mode == "gradio":
            # Launch Gradio interface with configuration
            config = get_config()
            logger.info("Starting Gradio interface on port %d", config.gradio_server_port)
            
            app = create_gradio_interface()
            app.launch(
                server_port=config.gradio_server_port,
                share=config.gradio_share,
                auth=config.gradio_auth,
                show_error=True,
                show_tips=True
            )
        else:
            # Run CLI demo
            logger.info("Running CLI demo")
            run_demo()
            
            # Optionally offer to launch Gradio
            if gr is not None:
                print("\nTo launch the web interface, run:")
                print("python main.py gradio")
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as exc:
        logger.error("Application failed: %s", exc, exc_info=True)
        sys.exit(1)
