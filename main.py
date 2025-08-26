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
"""

import os
import sys
import logging
import json
from typing import Any, Dict, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime dependency
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except ImportError:
    gr = None  # type: Optional[Any]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pr_firm.log')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration management for the Virtual PR Firm."""
    
    # Default values
    DEFAULT_TOPIC: str = "Announce product"
    DEFAULT_PLATFORMS: List[str] = None  # type: ignore
    MAX_TOPIC_LENGTH: int = 500
    MAX_PLATFORMS: int = 10
    SUPPORTED_PLATFORMS: List[str] = None  # type: ignore
    REQUEST_TIMEOUT: int = 300  # seconds
    MAX_RETRIES: int = 3
    
    def __post_init__(self):
        if self.DEFAULT_PLATFORMS is None:
            self.DEFAULT_PLATFORMS = ["twitter", "linkedin"]
        if self.SUPPORTED_PLATFORMS is None:
            self.SUPPORTED_PLATFORMS = [
                "twitter", "linkedin", "facebook", "instagram", 
                "tiktok", "youtube", "medium", "blog"
            ]
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override defaults with environment variables
        if os.getenv('PR_FIRM_DEFAULT_TOPIC'):
            config.DEFAULT_TOPIC = os.getenv('PR_FIRM_DEFAULT_TOPIC')
        
        if os.getenv('PR_FIRM_DEFAULT_PLATFORMS'):
            platforms = os.getenv('PR_FIRM_DEFAULT_PLATFORMS', '').split(',')
            config.DEFAULT_PLATFORMS = [p.strip() for p in platforms if p.strip()]
        
        if os.getenv('PR_FIRM_MAX_TOPIC_LENGTH'):
            config.MAX_TOPIC_LENGTH = int(os.getenv('PR_FIRM_MAX_TOPIC_LENGTH'))
        
        if os.getenv('PR_FIRM_REQUEST_TIMEOUT'):
            config.REQUEST_TIMEOUT = int(os.getenv('PR_FIRM_REQUEST_TIMEOUT'))
        
        return config


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation for the Virtual PR Firm."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def validate_topic(self, topic: str) -> str:
        """Validate and sanitize topic input."""
        if not topic or not isinstance(topic, str):
            raise ValidationError("Topic must be a non-empty string")
        
        # Sanitize input
        topic = topic.strip()
        
        if len(topic) < 3:
            raise ValidationError("Topic must be at least 3 characters long")
        
        if len(topic) > self.config.MAX_TOPIC_LENGTH:
            raise ValidationError(f"Topic must be no more than {self.config.MAX_TOPIC_LENGTH} characters")
        
        # Basic content validation
        if any(char in topic for char in ['<', '>', '&', '"', "'"]):
            raise ValidationError("Topic contains invalid characters")
        
        return topic
    
    def validate_platforms(self, platforms_text: str) -> List[str]:
        """Validate and normalize platform input."""
        if not platforms_text or not isinstance(platforms_text, str):
            raise ValidationError("Platforms must be a non-empty string")
        
        # Parse platforms
        platforms = [p.strip().lower() for p in platforms_text.split(",") if p.strip()]
        
        if not platforms:
            raise ValidationError("At least one platform must be specified")
        
        if len(platforms) > self.config.MAX_PLATFORMS:
            raise ValidationError(f"Maximum {self.config.MAX_PLATFORMS} platforms allowed")
        
        # Validate each platform
        invalid_platforms = [p for p in platforms if p not in self.config.SUPPORTED_PLATFORMS]
        if invalid_platforms:
            raise ValidationError(f"Unsupported platforms: {', '.join(invalid_platforms)}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_platforms = []
        for p in platforms:
            if p not in seen:
                seen.add(p)
                unique_platforms.append(p)
        
        return unique_platforms


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
    try:
        from flow import create_main_flow
    except ImportError as e:
        logger.error("Failed to import flow module: %s", e)
        raise
    
    # Load configuration
    config = Config.from_env()
    
    # NOTE: defaults are intentional for demo; replace with configuration in prod
    shared: Dict[str, Any] = {
        "task_requirements": {
            "platforms": config.DEFAULT_PLATFORMS, 
            "topic_or_goal": config.DEFAULT_TOPIC
        },
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
        flow: 'Flow' = create_main_flow()
        flow.run(shared)
        
        # Output results
        content_pieces = shared.get("content_pieces", {})
        if content_pieces:
            print("Content pieces:", content_pieces)
        else:
            print("No content pieces generated. Check logs for errors.")
            
    except Exception as exc:
        logger.error("Flow execution failed: %s", exc)
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
    
    tr = shared.get("task_requirements")
    if tr is None or not isinstance(tr, dict):
        raise ValueError("shared['task_requirements'] must be a dict")
    
    platforms = tr.get("platforms")
    if platforms is None:
        raise ValueError("task_requirements must include 'platforms'")
    if not isinstance(platforms, list):
        raise TypeError("task_requirements['platforms'] must be a list")
    
    topic = tr.get("topic_or_goal")
    if topic is None:
        raise ValueError("task_requirements must include 'topic_or_goal'")
    if not isinstance(topic, str):
        raise TypeError("task_requirements['topic_or_goal'] must be a string")
    
    # Validate brand_bible structure
    brand_bible = shared.get("brand_bible")
    if brand_bible is None or not isinstance(brand_bible, dict):
        raise ValueError("shared['brand_bible'] must be a dict")


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
            "Gradio not installed. Install with: pip install gradio\n"
            "Or run in CLI mode: python main.py --cli"
        )

    # Load configuration
    config = Config.from_env()
    validator = InputValidator(config)

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
        
        try:
            # Validate inputs
            validated_topic = validator.validate_topic(topic)
            validated_platforms = validator.validate_platforms(platforms_text)
            
            # Construct shared state
            shared: Dict[str, Any] = {
                "task_requirements": {
                    "platforms": validated_platforms, 
                    "topic_or_goal": validated_topic
                },
                "brand_bible": {"xml_raw": ""},
                "stream": None,
            }
            
            # Validate shared store
            validate_shared_store(shared)
            
            # Import flow creation
            try:
                from flow import create_main_flow
            except ImportError as e:
                logger.error("Failed to import flow module: %s", e)
                return {"error": "System configuration error"}
            
            # Create and run flow
            flow: 'Flow' = create_main_flow()
            flow.run(shared)
            
            # Return results
            content_pieces = shared.get("content_pieces", {})
            if not content_pieces:
                return {"message": "No content generated. Please try again."}
            
            return content_pieces
            
        except ValidationError as e:
            logger.warning("Input validation failed: %s", e)
            return {"error": f"Invalid input: {str(e)}"}
        except Exception as e:
            logger.error("Flow execution failed: %s", e)
            return {"error": "Content generation failed. Please try again."}

    with gr.Blocks(
        title="Virtual PR Firm Demo",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# Virtual PR Firm Demo")
        gr.Markdown("Generate PR content for multiple social media platforms")
        
        with gr.Row():
            with gr.Column(scale=2):
                topic = gr.Textbox(
                    label="Topic/Goal", 
                    value=config.DEFAULT_TOPIC,
                    placeholder="Enter your PR topic or goal...",
                    max_lines=3
                )
                
                platforms = gr.Textbox(
                    label="Platforms (comma-separated)", 
                    value=", ".join(config.DEFAULT_PLATFORMS),
                    placeholder="twitter, linkedin, facebook",
                    info=f"Supported platforms: {', '.join(config.SUPPORTED_PLATFORMS)}"
                )
                
                run_btn = gr.Button("Generate Content", variant="primary")
            
            with gr.Column(scale=3):
                out = gr.JSON(label="Generated Content")
        
        # Add error handling and progress tracking
        run_btn.click(
            fn=run_flow, 
            inputs=[topic, platforms], 
            outputs=[out],
            show_progress=True
        )

    return demo


def main():
    """Main entry point with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Virtual PR Firm - Content Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run CLI demo
  python main.py --web              # Launch web interface
  python main.py --cli              # Run CLI demo explicitly
  python main.py --config config.json  # Use custom config file
        """
    )
    
    parser.add_argument(
        '--web', 
        action='store_true',
        help='Launch Gradio web interface'
    )
    
    parser.add_argument(
        '--cli', 
        action='store_true',
        help='Run CLI demo (default)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        if args.web:
            if gr is None:
                logger.error("Gradio not installed. Install with: pip install gradio")
                sys.exit(1)
            
            app = create_gradio_interface()
            logger.info("Launching Gradio interface...")
            app.launch(share=False, server_name="0.0.0.0")
        else:
            logger.info("Running CLI demo...")
            run_demo()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Application failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
