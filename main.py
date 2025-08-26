"""Simple CLI/Gradio starter for the Virtual PR Firm.

This module provides a demonstration interface for the Virtual PR Firm's content
generation capabilities. It includes both a command-line interface for quick testing
and a Gradio web interface for interactive use.
"""

from flow import create_main_flow
from typing import Any, Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pocketflow import Flow
    import gradio as gr

try:
    import gradio as gr
except Exception:
    gr = None

import logging

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def validate_shared_store(shared: Dict[str, Any]) -> None:
    """Validate minimal shared dict required by the flows."""
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
    
    if not platforms:
        raise ValueError("task_requirements['platforms'] must not be empty")
    
    for platform in platforms:
        if not isinstance(platform, str):
            raise TypeError(f"All platforms must be strings, got {type(platform)}")
        if not platform.strip():
            raise ValueError("Platform names cannot be empty strings")


def run_demo() -> None:
    """Run a minimal demo of the main flow using a sample shared store."""
    shared: Dict[str, Any] = {
        "task_requirements": {"platforms": ["twitter", "linkedin"], "topic_or_goal": "Announce product"},
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }

    try:
        validate_shared_store(shared)
    except Exception as exc:
        logger.error("Invalid shared store: %s", exc)
        return

    try:
        flow = create_main_flow()
        flow.run(shared)
        print("Content pieces:", shared.get("content_pieces"))
    except Exception as exc:
        logger.error("Flow execution failed: %s", exc)


def create_gradio_interface() -> Any:
    """Create and return a Gradio Blocks app for the Virtual PR Firm demo."""
    if gr is None:
        raise RuntimeError("Gradio not installed")

    def run_flow(topic: str, platforms_text: str) -> Dict[str, Any]:
        """Execute the PR content generation flow with user-provided inputs."""
        platforms: List[str] = [p.strip() for p in platforms_text.split(",") if p.strip()]
        
        shared: Dict[str, Any] = {
            "task_requirements": {"platforms": platforms or ["twitter"], "topic_or_goal": topic},
            "brand_bible": {"xml_raw": ""},
            "stream": None,
        }
        
        try:
            validate_shared_store(shared)
        except Exception as exc:
            logger.error("Invalid input from UI: %s", exc)
            raise
        
        flow = create_main_flow()
        flow.run(shared)
        return shared.get("content_pieces", {})

    with gr.Blocks() as demo:
        gr.Markdown("# Virtual PR Firm Demo")
        
        topic = gr.Textbox(label="Topic/Goal", value="Announce product")
        platforms = gr.Textbox(label="Platforms (comma-separated)", value="twitter, linkedin")
        out = gr.JSON(label="Content pieces")
        run_btn = gr.Button("Run")
        
        run_btn.click(fn=run_flow, inputs=[topic, platforms], outputs=[out])

    return demo


if __name__ == "__main__":
    run_demo()