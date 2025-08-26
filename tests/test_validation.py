import importlib.util
import pathlib
import sys
import pytest

repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from main import validate_shared_store

HAS_POCKETFLOW = importlib.util.find_spec("pocketflow") is not None
if HAS_POCKETFLOW:
    from flow import create_platform_formatting_flow
    from nodes import EngagementManagerNode


def test_validate_shared_store_platforms():
    with pytest.raises(ValueError):
        validate_shared_store(
            {"task_requirements": {"platforms": [], "topic_or_goal": ""}}
        )
    with pytest.raises(TypeError):
        validate_shared_store(
            {"task_requirements": {"platforms": ["twitter", 1], "topic_or_goal": ""}}
        )


@pytest.mark.skipif(not HAS_POCKETFLOW, reason="pocketflow not installed")
def test_platform_formatting_flow_invalid_platform():
    batch_flow = create_platform_formatting_flow()
    with pytest.raises(ValueError):
        batch_flow.prep(
            {"task_requirements": {"platforms": ["myspace"], "topic_or_goal": ""}}
        )


@pytest.mark.skipif(not HAS_POCKETFLOW, reason="pocketflow not installed")
def test_engagement_manager_prep_sanitizes():
    node = EngagementManagerNode()
    shared = {
        "task_requirements": {
            "platforms": ["twitter", "", 123],
            "intents_by_platform": "bad",
            "topic_or_goal": 42,
        }
    }
    res = node.prep(shared)
    assert res["platforms"] == ["twitter"]
    assert res["intents_by_platform"] == {}
    assert res["topic_or_goal"] == ""
    assert shared["validation_warnings"]
