#!/usr/bin/env python3
"""Temporary test script to understand flow structure with mock pocketflow."""

import sys
sys.path.insert(0, '.')

# Replace pocketflow with our mock
import mock_pocketflow
sys.modules['pocketflow'] = mock_pocketflow

# Now try to import the flow
try:
    from flow import create_main_flow
    print("✓ Successfully imported create_main_flow")
    
    # Test creating a flow
    flow = create_main_flow()
    print("✓ Successfully created main flow")
    
    # Test with minimal shared state
    shared = {
        "task_requirements": {
            "platforms": ["twitter"], 
            "topic_or_goal": "Test"
        },
        "brand_bible": {"xml_raw": ""},
        "stream": None,
    }
    
    print("✓ Testing flow.run with minimal shared state...")
    result = flow.run(shared)
    print(f"✓ Flow completed. Keys in result: {list(result.keys())}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()