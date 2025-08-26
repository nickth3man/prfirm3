"""
Comprehensive test suite for Virtual PR Firm optimizations.

This test file validates all the optimizations implemented following the AGENTS.md
methodology, including error handling, validation, streaming, performance monitoring,
and configuration management.
"""

import unittest
import time
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the utilities we've created
from utils import (
    call_llm, call_llm_with_fallback,
    validate_shared_store, validate_platforms, validate_brand_bible,
    sanitize_input, normalize_platform_name,
    create_streaming_manager, create_gradio_streaming_manager,
    get_global_error_handler, ErrorContext, ErrorSeverity, RetryConfig,
    get_global_monitor, monitor_execution, record_metric,
    get_config, get_config_manager
)

class TestLLMUtils(unittest.TestCase):
    """Test LLM utility functions."""
    
    def test_call_llm_fallback(self):
        """Test LLM call with fallback when no API key is available."""
        # Test without API key (should use fallback)
        with patch.dict(os.environ, {}, clear=True):
            response = call_llm("Test prompt")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_call_llm_with_fallback(self):
        """Test LLM call with specific fallback prompt."""
        fallback_response = call_llm_with_fallback(
            "Complex prompt that might fail",
            fallback_prompt="Simple prompt"
        )
        self.assertIsInstance(fallback_response, str)
        self.assertGreater(len(fallback_response), 0)

class TestValidationUtils(unittest.TestCase):
    """Test validation utility functions."""
    
    def test_validate_shared_store_valid(self):
        """Test validation of valid shared store."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic"
            },
            "brand_bible": {"xml_raw": ""}
        }
        
        result = validate_shared_store(shared)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_shared_store_invalid(self):
        """Test validation of invalid shared store."""
        shared = {
            "task_requirements": {
                "platforms": [],  # Empty platforms list
                "topic_or_goal": ""  # Empty topic
            }
        }
        
        result = validate_shared_store(shared)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_platforms(self):
        """Test platform validation."""
        # Valid platforms
        result = validate_platforms(["twitter", "linkedin"])
        self.assertTrue(result.is_valid)
        
        # Invalid platforms
        result = validate_platforms([])
        self.assertFalse(result.is_valid)
        
        # Duplicate platforms
        result = validate_platforms(["twitter", "twitter"])
        self.assertTrue(result.is_valid)  # Should be valid but with warnings
        self.assertGreater(len(result.warnings), 0)
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test normal input
        sanitized = sanitize_input("Normal text")
        self.assertEqual(sanitized, "Normal text")
        
        # Test dangerous input
        dangerous = "<script>alert('xss')</script>"
        sanitized = sanitize_input(dangerous)
        self.assertNotIn("<script>", sanitized)
        
        # Test very long input
        long_input = "a" * 15000
        sanitized = sanitize_input(long_input)
        self.assertLessEqual(len(sanitized), 10000)
    
    def test_normalize_platform_name(self):
        """Test platform name normalization."""
        # Test common variations
        self.assertEqual(normalize_platform_name("X"), "twitter")
        self.assertEqual(normalize_platform_name("fb"), "facebook")
        self.assertEqual(normalize_platform_name("ig"), "instagram")
        self.assertEqual(normalize_platform_name("li"), "linkedin")
        
        # Test case insensitive
        self.assertEqual(normalize_platform_name("TWITTER"), "twitter")
        
        # Test unknown platform
        self.assertEqual(normalize_platform_name("unknown"), "unknown")

class TestStreamingUtils(unittest.TestCase):
    """Test streaming utility functions."""
    
    def test_streaming_manager(self):
        """Test streaming manager functionality."""
        milestones = []
        
        def callback(milestone):
            milestones.append(milestone)
        
        manager = create_streaming_manager(callback)
        
        # Test streaming lifecycle
        manager.start_streaming()
        manager.start_node("TestNode")
        manager.complete_node("TestNode")
        manager.stop_streaming()
        
        self.assertGreater(len(milestones), 0)
        self.assertEqual(manager.get_progress(), 10.0)  # 1/10 nodes = 10%
    
    def test_progress_tracker(self):
        """Test progress tracking."""
        manager = create_streaming_manager()
        
        # Simulate node execution
        manager.start_node("Node1")
        time.sleep(0.1)
        manager.complete_node("Node1")
        
        manager.start_node("Node2")
        time.sleep(0.1)
        manager.complete_node("Node2")
        
        self.assertEqual(manager.get_progress(), 20.0)  # 2/10 nodes = 20%
        self.assertIsNotNone(manager.get_eta())

class TestErrorHandlingUtils(unittest.TestCase):
    """Test error handling utility functions."""
    
    def test_error_handler(self):
        """Test error handler functionality."""
        error_handler = get_global_error_handler()
        
        # Test error handling
        context = ErrorContext(
            node_name="TestNode",
            operation="test_operation"
        )
        
        error = ValueError("Test error")
        result = error_handler.handle_error(error, context)
        
        # Should return a result (either retry action or fallback)
        self.assertIsInstance(result, dict)
    
    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(max_retries=3, base_delay=1.0)
        
        # Test delay calculation
        delay1 = config.get_delay(0)
        delay2 = config.get_delay(1)
        
        self.assertGreater(delay2, delay1)  # Exponential backoff
        self.assertLessEqual(delay1, config.max_delay)
        self.assertLessEqual(delay2, config.max_delay)

class TestPerformanceUtils(unittest.TestCase):
    """Test performance monitoring utility functions."""
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = get_global_monitor()
        
        # Record some metrics
        record_metric("test_metric", 42.0, "count")
        record_metric("test_metric", 43.0, "count")
        
        # Test node execution monitoring
        with monitor_execution("TestNode"):
            time.sleep(0.1)
        
        # Get performance data
        node_perf = monitor.get_node_performance("TestNode")
        self.assertIsNotNone(node_perf)
        self.assertEqual(node_perf.execution_count, 1)
        self.assertGreater(node_perf.avg_time, 0)
        
        # Get summary
        summary = monitor.get_summary()
        self.assertIn("total_nodes", summary)
        self.assertIn("total_executions", summary)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        monitor = get_global_monitor()
        
        # Test system metrics (if available)
        system_metrics = monitor.get_system_metrics()
        # System metrics might be empty if psutil is not available
        # but should not raise an exception
        self.assertIsInstance(system_metrics, dict)

class TestConfigurationUtils(unittest.TestCase):
    """Test configuration management utility functions."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = get_config()
        
        # Test basic configuration structure
        self.assertIsNotNone(config.llm)
        self.assertIsNotNone(config.validation)
        self.assertIsNotNone(config.performance)
        self.assertIsNotNone(config.error_handling)
        self.assertIsNotNone(config.streaming)
        
        # Test configuration serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("llm", config_dict)
        self.assertIn("validation", config_dict)
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = get_config_manager()
        
        # Test valid configuration update
        manager.update_config({"debug": True})
        
        # Test invalid configuration (should raise ValueError)
        with self.assertRaises(ValueError):
            manager.update_config({"llm": {"max_retries": -1}})
    
    def test_config_file_operations(self):
        """Test configuration file operations."""
        config = get_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save configuration
            config.save_to_file(temp_file)
            
            # Load configuration
            loaded_config = type(config).load_from_file(temp_file)
            
            # Verify loaded configuration
            self.assertEqual(config.debug, loaded_config.debug)
            self.assertEqual(config.llm.model, loaded_config.llm.model)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestIntegration(unittest.TestCase):
    """Test integration of all utilities."""
    
    def test_full_workflow(self):
        """Test a complete workflow with all optimizations."""
        # Initialize all components
        monitor = get_global_monitor()
        error_handler = get_global_error_handler()
        streaming_manager = create_streaming_manager()
        config = get_config()
        
        # Create test shared store
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test integration workflow"
            },
            "brand_bible": {"xml_raw": ""},
            "stream": streaming_manager
        }
        
        # Validate shared store
        validation_result = validate_shared_store(shared)
        self.assertTrue(validation_result.is_valid)
        
        # Start streaming
        streaming_manager.start_streaming()
        
        # Simulate node execution with monitoring
        with monitor_execution("IntegrationTestNode"):
            # Simulate some work
            time.sleep(0.1)
            
            # Record metrics
            record_metric("integration_test", 1.0, "count")
            
            # Simulate potential error
            try:
                raise ValueError("Simulated error for testing")
            except Exception as e:
                context = ErrorContext(
                    node_name="IntegrationTestNode",
                    operation="test_operation"
                )
                error_handler.handle_error(e, context, ErrorSeverity.MEDIUM)
        
        # Complete streaming
        streaming_manager.stop_streaming()
        
        # Verify results
        self.assertGreater(len(streaming_manager.get_milestones()), 0)
        self.assertGreater(monitor.get_summary()["total_executions"], 0)

def run_optimization_tests():
    """Run all optimization tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLLMUtils,
        TestValidationUtils,
        TestStreamingUtils,
        TestErrorHandlingUtils,
        TestPerformanceUtils,
        TestConfigurationUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running Virtual PR Firm Optimization Tests...")
    print("=" * 50)
    
    success = run_optimization_tests()
    
    if success:
        print("\n✅ All optimization tests passed!")
        print("\nOptimizations implemented successfully:")
        print("- Robust LLM calling with fallbacks")
        print("- Comprehensive input validation")
        print("- Real-time streaming and progress tracking")
        print("- Centralized error handling and retry mechanisms")
        print("- Performance monitoring and metrics collection")
        print("- Configuration management with validation")
        print("- Input sanitization and security")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    print("\n" + "=" * 50)