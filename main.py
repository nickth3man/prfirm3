"""Simple CLI/Gradio starter for the Virtual PR Firm."""

from typing import Any, Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pocketflow import Flow  # type: ignore
    import gradio as gr  # type: ignore

try:
    import gradio as gr
except Exception:
    gr = None  # type: Optional[Any]

import logging

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def run_demo() -> None:
    """Run a minimal demo of the main flow using a sample shared store."""
    from flow import create_main_flow

    shared: Dict[str, Any] = {
        "task_requirements": {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Announce product",
        },
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }
    validate_shared_store(shared)
    flow: "Flow" = create_main_flow()
    flow.run(shared)
    print("Content pieces:", shared.get("content_pieces"))


def validate_shared_store(shared: Dict[str, Any]) -> None:
    """Validate minimal ``shared`` dict required by the flows."""
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
    for p in platforms:
        if not isinstance(p, str) or not p.strip():
            raise TypeError("each platform must be a non-empty string")
    topic = tr.get("topic_or_goal")
    if topic is None or not isinstance(topic, str):
        raise ValueError("task_requirements must include 'topic_or_goal' as string")


def create_gradio_interface() -> Any:
    """Create and return a minimal Gradio Blocks app for the demo."""
    if gr is None:
        raise RuntimeError("Gradio not installed")

    def run_flow(topic: str, platforms_text: str) -> Dict[str, Any]:
        platforms: List[str] = [
            p.strip() for p in platforms_text.split(",") if p.strip()
        ]
        shared: Dict[str, Any] = {
            "task_requirements": {
                "platforms": platforms or ["twitter"],
                "topic_or_goal": topic,
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None,
        }
        from flow import create_main_flow

        validate_shared_store(shared)
        flow: "Flow" = create_main_flow()
        flow.run(shared)
        return shared.get("content_pieces", {})

    with gr.Blocks() as demo:
        topic = gr.Textbox(label="Topic/Goal", value="Announce product")
        platforms = gr.Textbox(
            label="Platforms (comma-separated)", value="twitter, linkedin"
        )
        out = gr.JSON(label="Content pieces")
        run_btn = gr.Button("Run")
        run_btn.click(fn=run_flow, inputs=[topic, platforms], outputs=[out])

    return demo


if __name__ == "__main__":
    run_demo()
