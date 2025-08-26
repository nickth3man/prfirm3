def test_imports_and_flow():
    """Smoke test: import utils and run the demo flow to ensure no immediate errors."""
    # Import inside test so failures are captured by pytest
    from flow import create_main_flow

    flow = create_main_flow()
    shared = {
        "task_requirements": {"platforms": ["twitter"], "topic_or_goal": "Test"},
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }

    # Run the flow; nodes are defensive and should not raise
    flow.run(shared)

    assert "content_pieces" in shared


