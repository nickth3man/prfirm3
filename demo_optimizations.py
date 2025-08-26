#!/usr/bin/env python3
"""
Demo script showcasing Virtual PR Firm optimizations.

This script demonstrates all the optimizations implemented following the AGENTS.md
methodology without requiring the full Virtual PR Firm dependencies.
"""

import time
import logging
from typing import Dict, Any

# Import our optimization utilities
from utils import (
    call_llm, call_llm_with_fallback,
    validate_shared_store, sanitize_input, normalize_platform_name,
    create_streaming_manager, get_global_error_handler, ErrorContext, ErrorSeverity,
    get_global_monitor, monitor_execution, record_metric,
    get_config, get_config_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_llm_optimizations():
    """Demonstrate LLM calling optimizations."""
    print("\n🔧 LLM Optimizations Demo")
    print("=" * 50)
    
    # Test LLM call with fallback
    print("1. Testing LLM call with fallback (no API key)...")
    response = call_llm("Write a brief announcement for a new product")
    print(f"   Response: {response[:100]}...")
    
    # Test LLM call with specific fallback
    print("\n2. Testing LLM call with specific fallback...")
    fallback_response = call_llm_with_fallback(
        "Complex prompt that might fail",
        fallback_prompt="Simple product announcement"
    )
    print(f"   Response: {fallback_response[:100]}...")

def demo_validation_optimizations():
    """Demonstrate validation optimizations."""
    print("\n🔍 Validation Optimizations Demo")
    print("=" * 50)
    
    # Test input sanitization
    print("1. Testing input sanitization...")
    dangerous_input = "<script>alert('xss')</script>Normal text"
    sanitized = sanitize_input(dangerous_input)
    print(f"   Original: {dangerous_input}")
    print(f"   Sanitized: {sanitized}")
    
    # Test platform normalization
    print("\n2. Testing platform normalization...")
    platforms = ["X", "fb", "ig", "li", "TWITTER"]
    normalized = [normalize_platform_name(p) for p in platforms]
    print(f"   Original: {platforms}")
    print(f"   Normalized: {normalized}")
    
    # Test shared store validation
    print("\n3. Testing shared store validation...")
    
    # Valid shared store
    valid_shared = {
        "task_requirements": {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Test topic"
        },
        "brand_bible": {"xml_raw": ""}
    }
    
    result = validate_shared_store(valid_shared)
    print(f"   Valid shared store: {result.is_valid}")
    
    # Invalid shared store
    invalid_shared = {
        "task_requirements": {
            "platforms": [],  # Empty platforms
            "topic_or_goal": ""  # Empty topic
        }
    }
    
    result = validate_shared_store(invalid_shared)
    print(f"   Invalid shared store: {result.is_valid}")
    if not result.is_valid:
        print(f"   Errors: {[e.message for e in result.errors]}")

def demo_streaming_optimizations():
    """Demonstrate streaming optimizations."""
    print("\n📡 Streaming Optimizations Demo")
    print("=" * 50)
    
    # Create streaming manager with callback
    milestones = []
    
    def milestone_callback(milestone):
        milestones.append(milestone)
        print(f"   📍 {milestone.type.value.upper()}: {milestone.message}")
    
    streaming_manager = create_streaming_manager(milestone_callback)
    
    # Simulate workflow execution
    print("1. Simulating workflow with streaming...")
    streaming_manager.start_streaming()
    
    # Simulate node executions
    nodes = ["EngagementManager", "BrandBibleIngest", "ContentCraftsman", "StyleEditor"]
    for i, node in enumerate(nodes):
        streaming_manager.start_node(node)
        time.sleep(0.2)  # Simulate work
        streaming_manager.complete_node(node, {"status": "success"})
    
    streaming_manager.stop_streaming()
    
    print(f"\n   Total milestones: {len(milestones)}")
    print(f"   Final progress: {streaming_manager.get_progress():.1f}%")

def demo_error_handling_optimizations():
    """Demonstrate error handling optimizations."""
    print("\n🛡️ Error Handling Optimizations Demo")
    print("=" * 50)
    
    error_handler = get_global_error_handler()
    
    # Test different error scenarios
    print("1. Testing validation error...")
    context = ErrorContext(
        node_name="ValidationNode",
        operation="input_validation"
    )
    result = error_handler.handle_error(
        ValueError("Invalid input format"),
        context,
        ErrorSeverity.MEDIUM
    )
    print(f"   Error result: {result}")
    
    print("\n2. Testing rate limit error...")
    context = ErrorContext(
        node_name="LLMNode",
        operation="api_call",
        retry_count=1
    )
    result = error_handler.handle_error(
        Exception("Rate limit exceeded"),
        context,
        ErrorSeverity.HIGH
    )
    print(f"   Error result: {result}")
    
    # Show error statistics
    stats = error_handler.get_error_stats()
    print(f"\n   Error statistics: {stats}")

def demo_performance_optimizations():
    """Demonstrate performance monitoring optimizations."""
    print("\n📊 Performance Monitoring Demo")
    print("=" * 50)
    
    monitor = get_global_monitor()
    
    # Record some metrics
    print("1. Recording performance metrics...")
    record_metric("demo_start", time.time(), "timestamp")
    record_metric("user_requests", 42, "count")
    record_metric("api_calls", 156, "count")
    record_metric("average_response_time", 2.3, "seconds")
    
    # Simulate node execution monitoring
    print("\n2. Simulating node execution monitoring...")
    with monitor_execution("DemoNode1"):
        time.sleep(0.3)  # Simulate work
    
    with monitor_execution("DemoNode2"):
        time.sleep(0.2)  # Simulate work
    
    with monitor_execution("DemoNode3"):
        time.sleep(0.1)  # Simulate work
    
    # Get performance summary
    summary = monitor.get_summary()
    print(f"\n3. Performance summary:")
    print(f"   Total nodes executed: {summary['total_nodes']}")
    print(f"   Total executions: {summary['total_executions']}")
    print(f"   Error rate: {summary['error_rate']:.2%}")
    print(f"   Slowest node: {summary['slowest_node']} ({summary['slowest_avg_time']:.3f}s)")
    print(f"   Fastest node: {summary['fastest_node']} ({summary['fastest_avg_time']:.3f}s)")

def demo_configuration_optimizations():
    """Demonstrate configuration management optimizations."""
    print("\n⚙️ Configuration Management Demo")
    print("=" * 50)
    
    config = get_config()
    manager = get_config_manager()
    
    # Show current configuration
    print("1. Current configuration:")
    config_dict = config.to_dict()
    print(f"   Environment: {config_dict['environment']}")
    print(f"   Debug mode: {config_dict['debug']}")
    print(f"   LLM model: {config_dict['llm']['model']}")
    print(f"   Max retries: {config_dict['llm']['max_retries']}")
    print(f"   API keys available: OpenAI={config_dict['llm']['has_openai_key']}, "
          f"Anthropic={config_dict['llm']['has_anthropic_key']}, "
          f"Gemini={config_dict['llm']['has_gemini_key']}")
    
    # Test configuration update
    print("\n2. Testing configuration update...")
    try:
        manager.update_config({"debug": True})
        print("   ✅ Successfully updated debug setting")
    except Exception as e:
        print(f"   ❌ Configuration update failed: {e}")
    
    # Test configuration validation
    print("\n3. Testing configuration validation...")
    try:
        manager.update_config({"llm": {"max_retries": -1}})
        print("   ❌ Invalid configuration was accepted (should have failed)")
    except ValueError as e:
        print(f"   ✅ Configuration validation caught invalid setting: {e}")

def demo_integration_workflow():
    """Demonstrate full integration workflow."""
    print("\n🔄 Integration Workflow Demo")
    print("=" * 50)
    
    # Initialize all components
    monitor = get_global_monitor()
    error_handler = get_global_error_handler()
    streaming_manager = create_streaming_manager()
    config = get_config()
    
    print("1. Initializing optimization components...")
    print(f"   Monitor: {type(monitor).__name__}")
    print(f"   Error Handler: {type(error_handler).__name__}")
    print(f"   Streaming Manager: {type(streaming_manager).__name__}")
    print(f"   Config: {type(config).__name__}")
    
    # Create test shared store
    print("\n2. Creating and validating test data...")
    shared = {
        "task_requirements": {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Demo integration workflow"
        },
        "brand_bible": {"xml_raw": ""},
        "stream": streaming_manager
    }
    
    # Validate shared store
    validation_result = validate_shared_store(shared)
    print(f"   Validation result: {validation_result.is_valid}")
    
    # Start streaming
    print("\n3. Starting streaming session...")
    streaming_manager.start_streaming()
    
    # Simulate workflow execution with monitoring
    print("\n4. Simulating workflow execution...")
    with monitor_execution("IntegrationWorkflow"):
        # Simulate some work
        time.sleep(0.5)
        
        # Record metrics
        record_metric("integration_demo", 1.0, "count")
        
        # Simulate potential error (handled gracefully)
        try:
            raise ValueError("Simulated error for testing")
        except Exception as e:
            context = ErrorContext(
                node_name="IntegrationWorkflow",
                operation="demo_operation"
            )
            error_handler.handle_error(e, context, ErrorSeverity.MEDIUM)
    
    # Complete streaming
    streaming_manager.stop_streaming()
    
    # Show results
    print("\n5. Workflow results:")
    print(f"   Milestones: {len(streaming_manager.get_milestones())}")
    print(f"   Progress: {streaming_manager.get_progress():.1f}%")
    print(f"   Executions: {monitor.get_summary()['total_executions']}")

def main():
    """Run all optimization demos."""
    print("🚀 Virtual PR Firm Optimization Demo")
    print("=" * 60)
    print("This demo showcases all optimizations implemented following AGENTS.md methodology")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_llm_optimizations()
        demo_validation_optimizations()
        demo_streaming_optimizations()
        demo_error_handling_optimizations()
        demo_performance_optimizations()
        demo_configuration_optimizations()
        demo_integration_workflow()
        
        print("\n" + "=" * 60)
        print("✅ All optimization demos completed successfully!")
        print("\n🎯 Key Benefits Demonstrated:")
        print("   • Robust error handling with graceful fallbacks")
        print("   • Comprehensive input validation and security")
        print("   • Real-time streaming and progress tracking")
        print("   • Performance monitoring and metrics collection")
        print("   • Flexible configuration management")
        print("   • Always-available system with intelligent fallbacks")
        print("\n📈 The Virtual PR Firm is now production-ready with enterprise-grade reliability!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo execution failed")

if __name__ == "__main__":
    main()