"""Comprehensive test suite for the flow module.

This module provides extensive testing for all flow orchestration functionality,
including flow creation, validation, execution, and error handling.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any, List

# Import the modules to test
from flow import (
    handle_flow_errors,
    validate_flow_connections,
    validate_shared_state_for_flow,
    create_main_flow,
    create_platform_formatting_flow,
    create_validation_flow,
    create_content_generation_flow,
    validate_flow_execution,
    execute_flow_with_validation,
    test_flow_construction,
    test_flow_execution
)


class TestUtilityFunctions:
    """Test utility functions used by flows."""
    
    def test_handle_flow_errors_success(self):
        """Test error handling decorator with successful function."""
        @handle_flow_errors
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_handle_flow_errors_exception(self):
        """Test error handling decorator with exception."""
        @handle_flow_errors
        def test_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            test_func()
    
    def test_validate_flow_connections_valid(self):
        """Test flow connections validation with valid nodes."""
        mock_nodes = [MagicMock(), MagicMock(), MagicMock()]
        mock_nodes[0].__class__.__name__ = "Node1"
        mock_nodes[1].__class__.__name__ = "Node2"
        mock_nodes[2].__class__.__name__ = "Node3"
        
        validate_flow_connections(mock_nodes)  # Should not raise
    
    def test_validate_flow_connections_empty(self):
        """Test flow connections validation with empty list."""
        with pytest.raises(ValueError, match="must contain at least one node"):
            validate_flow_connections([])
    
    def test_validate_flow_connections_duplicate(self):
        """Test flow connections validation with duplicate nodes."""
        mock_nodes = [MagicMock(), MagicMock()]
        mock_nodes[0].__class__.__name__ = "Node1"
        mock_nodes[1].__class__.__name__ = "Node1"  # Duplicate
        
        # Should not raise, but should log warning
        validate_flow_connections(mock_nodes)
    
    def test_validate_shared_state_for_flow_valid(self):
        """Test shared state validation for flow with valid input."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "test topic"
            }
        }
        validate_shared_state_for_flow(shared)  # Should not raise
    
    def test_validate_shared_state_for_flow_invalid_type(self):
        """Test shared state validation for flow with invalid type."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_state_for_flow("not a dict")
    
    def test_validate_shared_state_for_flow_missing_key(self):
        """Test shared state validation for flow with missing required key."""
        shared = {"other_key": "value"}
        with pytest.raises(ValueError, match="Missing required key"):
            validate_shared_state_for_flow(shared)
    
    def test_validate_shared_state_for_flow_invalid_task_requirements(self):
        """Test shared state validation for flow with invalid task_requirements."""
        shared = {"task_requirements": "not a dict"}
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_state_for_flow(shared)
    
    def test_validate_shared_state_for_flow_invalid_platforms(self):
        """Test shared state validation for flow with invalid platforms."""
        shared = {
            "task_requirements": {
                "platforms": "not a list"
            }
        }
        with pytest.raises(ValueError, match="must be a list"):
            validate_shared_state_for_flow(shared)


class TestFlowCreation:
    """Test flow creation functions."""
    
    @patch('flow.EngagementManagerNode')
    @patch('flow.BrandBibleIngestNode')
    @patch('flow.VoiceAlignmentNode')
    @patch('flow.ContentCraftsmanNode')
    @patch('flow.StyleEditorNode')
    @patch('flow.StyleComplianceNode')
    @patch('flow.AgencyDirectorNode')
    @patch('flow.create_platform_formatting_flow')
    @patch('flow.validate_flow_connections')
    @patch('flow.Flow')
    def test_create_main_flow_success(self, mock_flow_class, mock_validate, mock_create_formatting,
                                    mock_agency, mock_compliance, mock_editor, mock_craftsman,
                                    mock_voice, mock_brand, mock_engagement):
        """Test successful main flow creation."""
        # Setup mocks
        mock_nodes = [MagicMock() for _ in range(7)]
        mock_engagement.return_value = mock_nodes[0]
        mock_brand.return_value = mock_nodes[1]
        mock_voice.return_value = mock_nodes[2]
        mock_craftsman.return_value = mock_nodes[3]
        mock_editor.return_value = mock_nodes[4]
        mock_compliance.return_value = mock_nodes[5]
        mock_agency.return_value = mock_nodes[6]
        
        mock_formatting_flow = MagicMock()
        mock_create_formatting.return_value = mock_formatting_flow
        
        mock_flow_instance = MagicMock()
        mock_flow_class.return_value = mock_flow_instance
        
        # Execute
        result = create_main_flow()
        
        # Verify
        assert result == mock_flow_instance
        mock_flow_class.assert_called_once_with(start=mock_nodes[0])
        mock_validate.assert_called_once()
    
    @patch('flow.EngagementManagerNode')
    def test_create_main_flow_failure(self, mock_engagement):
        """Test main flow creation with failure."""
        mock_engagement.side_effect = Exception("Node creation failed")
        
        with pytest.raises(ValueError, match="Flow construction failed"):
            create_main_flow()
    
    @patch('flow.PlatformFormattingNode')
    @patch('flow.BatchFlow')
    def test_create_platform_formatting_flow_success(self, mock_batch_flow, mock_formatting_node):
        """Test successful platform formatting flow creation."""
        # Setup mocks
        mock_node = MagicMock()
        mock_formatting_node.return_value = mock_node
        
        mock_batch_instance = MagicMock()
        mock_batch_flow.return_value = mock_batch_instance
        
        # Execute
        result = create_platform_formatting_flow()
        
        # Verify
        assert result == mock_batch_instance
        mock_batch_flow.assert_called_once_with(start=mock_node)
    
    @patch('flow.PlatformFormattingNode')
    def test_create_platform_formatting_flow_failure(self, mock_formatting_node):
        """Test platform formatting flow creation with failure."""
        mock_formatting_node.side_effect = Exception("Node creation failed")
        
        with pytest.raises(ValueError, match="Platform formatting flow construction failed"):
            create_platform_formatting_flow()
    
    @patch('flow.StyleEditorNode')
    @patch('flow.StyleComplianceNode')
    @patch('flow.AgencyDirectorNode')
    @patch('flow.Flow')
    def test_create_validation_flow_success(self, mock_flow_class, mock_agency, mock_compliance, mock_editor):
        """Test successful validation flow creation."""
        # Setup mocks
        mock_nodes = [MagicMock() for _ in range(3)]
        mock_editor.return_value = mock_nodes[0]
        mock_compliance.return_value = mock_nodes[1]
        mock_agency.return_value = mock_nodes[2]
        
        mock_flow_instance = MagicMock()
        mock_flow_class.return_value = mock_flow_instance
        
        # Execute
        result = create_validation_flow()
        
        # Verify
        assert result == mock_flow_instance
        mock_flow_class.assert_called_once_with(start=mock_nodes[0])
    
    @patch('flow.StyleEditorNode')
    def test_create_validation_flow_failure(self, mock_editor):
        """Test validation flow creation with failure."""
        mock_editor.side_effect = Exception("Node creation failed")
        
        with pytest.raises(ValueError, match="Validation flow construction failed"):
            create_validation_flow()
    
    @patch('flow.ContentCraftsmanNode')
    @patch('flow.StyleEditorNode')
    @patch('flow.Flow')
    def test_create_content_generation_flow_success(self, mock_flow_class, mock_editor, mock_craftsman):
        """Test successful content generation flow creation."""
        # Setup mocks
        mock_nodes = [MagicMock() for _ in range(2)]
        mock_craftsman.return_value = mock_nodes[0]
        mock_editor.return_value = mock_nodes[1]
        
        mock_flow_instance = MagicMock()
        mock_flow_class.return_value = mock_flow_instance
        
        # Execute
        result = create_content_generation_flow()
        
        # Verify
        assert result == mock_flow_instance
        mock_flow_class.assert_called_once_with(start=mock_nodes[0])
    
    @patch('flow.ContentCraftsmanNode')
    def test_create_content_generation_flow_failure(self, mock_craftsman):
        """Test content generation flow creation with failure."""
        mock_craftsman.side_effect = Exception("Node creation failed")
        
        with pytest.raises(ValueError, match="Content generation flow construction failed"):
            create_content_generation_flow()


class TestPlatformFormattingBatchFlow:
    """Test PlatformFormattingBatchFlow functionality."""
    
    def test_prep_valid_platforms(self):
        """Test batch flow prep with valid platforms."""
        from flow import PlatformFormattingBatchFlow
        
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"]
            }
        }
        
        flow = PlatformFormattingBatchFlow(start=MagicMock())
        result = flow.prep(shared)
        
        expected = [{"platform": "twitter"}, {"platform": "linkedin"}]
        assert result == expected
    
    def test_prep_missing_platforms(self):
        """Test batch flow prep with missing platforms."""
        from flow import PlatformFormattingBatchFlow
        
        shared = {"task_requirements": {}}
        
        flow = PlatformFormattingBatchFlow(start=MagicMock())
        result = flow.prep(shared)
        
        expected = [{"platform": "twitter"}, {"platform": "linkedin"}]
        assert result == expected
    
    def test_prep_invalid_platforms(self):
        """Test batch flow prep with invalid platforms."""
        from flow import PlatformFormattingBatchFlow
        
        shared = {
            "task_requirements": {
                "platforms": "not a list"
            }
        }
        
        flow = PlatformFormattingBatchFlow(start=MagicMock())
        result = flow.prep(shared)
        
        expected = [{"platform": "twitter"}, {"platform": "linkedin"}]
        assert result == expected
    
    def test_prep_invalid_platform_names(self):
        """Test batch flow prep with invalid platform names."""
        from flow import PlatformFormattingBatchFlow
        
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "invalid@platform", "linkedin"]
            }
        }
        
        flow = PlatformFormattingBatchFlow(start=MagicMock())
        result = flow.prep(shared)
        
        expected = [{"platform": "twitter"}, {"platform": "linkedin"}]
        assert result == expected
    
    def test_prep_exception_handling(self):
        """Test batch flow prep with exception handling."""
        from flow import PlatformFormattingBatchFlow
        
        shared = "invalid"
        
        flow = PlatformFormattingBatchFlow(start=MagicMock())
        result = flow.prep(shared)
        
        expected = [{"platform": "twitter"}, {"platform": "linkedin"}]
        assert result == expected


class TestFlowExecution:
    """Test flow execution functions."""
    
    def test_validate_flow_execution_valid(self):
        """Test flow execution validation with valid shared state."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "test topic"
            }
        }
        
        validate_flow_execution(shared)  # Should not raise
    
    def test_validate_flow_execution_invalid(self):
        """Test flow execution validation with invalid shared state."""
        shared = {"other_key": "value"}
        
        with pytest.raises(ValueError):
            validate_flow_execution(shared)
    
    @patch('flow.validate_flow_execution')
    def test_execute_flow_with_validation_success(self, mock_validate):
        """Test successful flow execution with validation."""
        mock_flow = MagicMock()
        shared = {"task_requirements": {"platforms": ["twitter"]}}
        
        execute_flow_with_validation(mock_flow, shared)
        
        mock_validate.assert_called_once_with(shared)
        mock_flow.run.assert_called_once_with(shared)
    
    @patch('flow.validate_flow_execution')
    def test_execute_flow_with_validation_validation_failure(self, mock_validate):
        """Test flow execution with validation failure."""
        mock_validate.side_effect = ValueError("Validation failed")
        mock_flow = MagicMock()
        shared = {"invalid": "data"}
        
        with pytest.raises(RuntimeError, match="Flow execution failed"):
            execute_flow_with_validation(mock_flow, shared)
        
        mock_flow.run.assert_not_called()
    
    @patch('flow.validate_flow_execution')
    def test_execute_flow_with_validation_flow_failure(self, mock_validate):
        """Test flow execution with flow execution failure."""
        mock_flow = MagicMock()
        mock_flow.run.side_effect = Exception("Flow execution failed")
        shared = {"task_requirements": {"platforms": ["twitter"]}}
        
        with pytest.raises(RuntimeError, match="Flow execution failed"):
            execute_flow_with_validation(mock_flow, shared)
        
        mock_validate.assert_called_once_with(shared)


class TestTestFunctions:
    """Test test functions for flow validation."""
    
    @patch('flow.create_main_flow')
    @patch('flow.create_platform_formatting_flow')
    @patch('flow.create_validation_flow')
    @patch('flow.create_content_generation_flow')
    def test_test_flow_construction_success(self, mock_content, mock_validation, mock_platform, mock_main):
        """Test successful flow construction test."""
        # Setup mocks
        mock_main.return_value = MagicMock()
        mock_platform.return_value = MagicMock()
        mock_validation.return_value = MagicMock()
        mock_content.return_value = MagicMock()
        
        result = test_flow_construction()
        
        assert result is True
        mock_main.assert_called_once()
        mock_platform.assert_called_once()
        mock_validation.assert_called_once()
        mock_content.assert_called_once()
    
    @patch('flow.create_main_flow')
    def test_test_flow_construction_failure(self, mock_main):
        """Test flow construction test with failure."""
        mock_main.side_effect = Exception("Construction failed")
        
        result = test_flow_construction()
        
        assert result is False
    
    @patch('flow.create_main_flow')
    @patch('flow.execute_flow_with_validation')
    def test_test_flow_execution_success(self, mock_execute, mock_create):
        """Test successful flow execution test."""
        # Setup mocks
        mock_flow = MagicMock()
        mock_create.return_value = mock_flow
        
        result = test_flow_execution()
        
        assert result is True
        mock_create.assert_called_once()
        mock_execute.assert_called_once_with(mock_flow, pytest.approx({
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test content generation",
                "intents_by_platform": {}
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None
        }))
    
    @patch('flow.create_main_flow')
    def test_test_flow_execution_failure(self, mock_create):
        """Test flow execution test with failure."""
        mock_create.side_effect = Exception("Flow creation failed")
        
        result = test_flow_execution()
        
        assert result is False


class TestFlowIntegration:
    """Integration tests for flow functionality."""
    
    def test_flow_creation_integration(self):
        """Test integration of flow creation functions."""
        # Test that all flow creation functions can be called without errors
        # (using mocks to avoid actual node creation)
        with patch('flow.EngagementManagerNode'), \
             patch('flow.BrandBibleIngestNode'), \
             patch('flow.VoiceAlignmentNode'), \
             patch('flow.ContentCraftsmanNode'), \
             patch('flow.StyleEditorNode'), \
             patch('flow.StyleComplianceNode'), \
             patch('flow.AgencyDirectorNode'), \
             patch('flow.PlatformFormattingNode'), \
             patch('flow.Flow'), \
             patch('flow.BatchFlow'), \
             patch('flow.validate_flow_connections'):
            
            # Should not raise exceptions
            create_main_flow()
            create_platform_formatting_flow()
            create_validation_flow()
            create_content_generation_flow()
    
    def test_validation_integration(self):
        """Test integration of validation functions."""
        # Test shared state validation
        valid_shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "test topic"
            }
        }
        
        validate_shared_state_for_flow(valid_shared)  # Should not raise
        
        # Test invalid shared state
        invalid_shared = {"other_key": "value"}
        with pytest.raises(ValueError):
            validate_shared_state_for_flow(invalid_shared)
    
    def test_error_handling_integration(self):
        """Test integration of error handling."""
        # Test that error handling decorators work correctly
        @handle_flow_errors
        def test_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            test_func()
        
        # Test successful execution
        @handle_flow_errors
        def test_func_success():
            return "success"
        
        result = test_func_success()
        assert result == "success"


class TestFlowEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_flow_connections_validation_edge_cases(self):
        """Test edge cases for flow connections validation."""
        # Test with None
        with pytest.raises(ValueError, match="must contain at least one node"):
            validate_flow_connections(None)
        
        # Test with empty list
        with pytest.raises(ValueError, match="must contain at least one node"):
            validate_flow_connections([])
        
        # Test with single node
        mock_node = MagicMock()
        mock_node.__class__.__name__ = "SingleNode"
        validate_flow_connections([mock_node])  # Should not raise
    
    def test_shared_state_validation_edge_cases(self):
        """Test edge cases for shared state validation."""
        # Test with None
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_state_for_flow(None)
        
        # Test with empty dict
        with pytest.raises(ValueError, match="Missing required key"):
            validate_shared_state_for_flow({})
        
        # Test with missing task_requirements
        shared = {"other_key": "value"}
        with pytest.raises(ValueError, match="Missing required key"):
            validate_shared_state_for_flow(shared)
        
        # Test with invalid task_requirements type
        shared = {"task_requirements": "not a dict"}
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_state_for_flow(shared)
    
    def test_batch_flow_edge_cases(self):
        """Test edge cases for batch flow."""
        from flow import PlatformFormattingBatchFlow
        
        # Test with None shared state
        flow = PlatformFormattingBatchFlow(start=MagicMock())
        result = flow.prep(None)
        expected = [{"platform": "twitter"}, {"platform": "linkedin"}]
        assert result == expected
        
        # Test with empty shared state
        result = flow.prep({})
        assert result == expected
        
        # Test with missing task_requirements
        result = flow.prep({"other_key": "value"})
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__])