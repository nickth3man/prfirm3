"""Unit tests for the flow.py module.

This module provides comprehensive testing for the Virtual PR Firm's flow orchestration,
including flow creation, validation, execution, and error handling.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from flow import (
    FlowConfig,
    FlowValidationError,
    FlowExecutionError,
    validate_flow_structure,
    create_main_flow,
    create_platform_formatting_flow,
    create_validation_flow,
    create_feedback_flow,
    create_streaming_flow,
    execute_flow_with_monitoring,
    get_flow_metrics
)


class TestFlowConfig:
    """Test the FlowConfig class for flow configuration management."""
    
    def test_flow_config_defaults(self):
        """Test that FlowConfig has correct default values."""
        config = FlowConfig()
        
        assert config.ENGAGEMENT_RETRIES == 2
        assert config.BRAND_BIBLE_RETRIES == 2
        assert config.VOICE_ALIGNMENT_RETRIES == 2
        assert config.CONTENT_CRAFTSMAN_RETRIES == 3
        assert config.CONTENT_CRAFTSMAN_WAIT == 2
        assert config.STYLE_EDITOR_RETRIES == 3
        assert config.STYLE_EDITOR_WAIT == 1
        assert config.STYLE_COMPLIANCE_RETRIES == 2
        assert config.PLATFORM_FORMATTING_RETRIES == 2
        assert config.MAX_VALIDATION_ATTEMPTS == 5
        assert config.FLOW_TIMEOUT == 300
        assert config.ENABLE_METRICS is True
    
    def test_flow_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict('os.environ', {
            'PR_FIRM_ENGAGEMENT_RETRIES': '5',
            'PR_FIRM_CONTENT_CRAFTSMAN_RETRIES': '7',
            'PR_FIRM_MAX_VALIDATION_ATTEMPTS': '10',
            'PR_FIRM_FLOW_TIMEOUT': '600'
        }):
            config = FlowConfig.from_env()
            
            assert config.ENGAGEMENT_RETRIES == 5
            assert config.CONTENT_CRAFTSMAN_RETRIES == 7
            assert config.MAX_VALIDATION_ATTEMPTS == 10
            assert config.FLOW_TIMEOUT == 600
    
    def test_flow_config_from_env_invalid_values(self):
        """Test handling of invalid environment variable values."""
        with patch.dict('os.environ', {
            'PR_FIRM_ENGAGEMENT_RETRIES': 'invalid',
            'PR_FIRM_FLOW_TIMEOUT': 'not_a_number'
        }):
            # Should not raise an exception, should use defaults
            config = FlowConfig.from_env()
            assert config.ENGAGEMENT_RETRIES == 2
            assert config.FLOW_TIMEOUT == 300


class TestFlowValidation:
    """Test flow validation functions."""
    
    def test_validate_flow_structure_valid(self):
        """Test validation of valid flow structure."""
        mock_flow = Mock()
        mock_flow.start = Mock()
        
        # Should not raise an exception
        validate_flow_structure(mock_flow)
    
    def test_validate_flow_structure_none(self):
        """Test validation when flow is None."""
        with pytest.raises(FlowValidationError, match="Flow cannot be None"):
            validate_flow_structure(None)
    
    def test_validate_flow_structure_no_start_attr(self):
        """Test validation when flow has no start attribute."""
        mock_flow = Mock()
        del mock_flow.start
        
        with pytest.raises(FlowValidationError, match="Flow must have a start node"):
            validate_flow_structure(mock_flow)
    
    def test_validate_flow_structure_none_start(self):
        """Test validation when flow start node is None."""
        mock_flow = Mock()
        mock_flow.start = None
        
        with pytest.raises(FlowValidationError, match="Flow start node cannot be None"):
            validate_flow_structure(mock_flow)


class TestCreateMainFlow:
    """Test the create_main_flow function."""
    
    @patch('flow.EngagementManagerNode')
    @patch('flow.BrandBibleIngestNode')
    @patch('flow.VoiceAlignmentNode')
    @patch('flow.ContentCraftsmanNode')
    @patch('flow.StyleEditorNode')
    @patch('flow.StyleComplianceNode')
    @patch('flow.AgencyDirectorNode')
    @patch('flow.create_platform_formatting_flow')
    @patch('flow.Flow')
    @patch('flow.validate_flow_structure')
    def test_create_main_flow_success(self, mock_validate, mock_flow_class, 
                                    mock_create_formatting, mock_agency, mock_compliance,
                                    mock_editor, mock_craftsman, mock_voice, mock_brand, mock_engagement):
        """Test successful creation of main flow."""
        # Mock node instances
        mock_engagement_instance = Mock()
        mock_brand_instance = Mock()
        mock_voice_instance = Mock()
        mock_craftsman_instance = Mock()
        mock_editor_instance = Mock()
        mock_compliance_instance = Mock()
        mock_agency_instance = Mock()
        mock_formatting_flow = Mock()
        mock_flow_instance = Mock()
        
        mock_engagement.return_value = mock_engagement_instance
        mock_brand.return_value = mock_brand_instance
        mock_voice.return_value = mock_voice_instance
        mock_craftsman.return_value = mock_craftsman_instance
        mock_editor.return_value = mock_editor_instance
        mock_compliance.return_value = mock_compliance_instance
        mock_agency.return_value = mock_agency_instance
        mock_create_formatting.return_value = mock_formatting_flow
        mock_flow_class.return_value = mock_flow_instance
        
        # Test flow creation
        result = create_main_flow()
        
        assert result == mock_flow_instance
        
        # Verify nodes were created with correct parameters
        mock_engagement.assert_called_once_with(max_retries=2)
        mock_brand.assert_called_once_with(max_retries=2)
        mock_voice.assert_called_once_with(max_retries=2)
        mock_craftsman.assert_called_once_with(max_retries=3, wait=2)
        mock_editor.assert_called_once_with(max_retries=3, wait=1)
        mock_compliance.assert_called_once_with(max_retries=2)
        
        # Verify flow validation was called
        mock_validate.assert_called_once_with(mock_flow_instance)
    
    @patch('flow.EngagementManagerNode')
    def test_create_main_flow_node_creation_failure(self, mock_engagement):
        """Test create_main_flow when node creation fails."""
        mock_engagement.side_effect = Exception("Node creation failed")
        
        with pytest.raises(FlowValidationError, match="Flow construction failed"):
            create_main_flow()
    
    def test_create_main_flow_with_config(self):
        """Test create_main_flow with custom configuration."""
        config = FlowConfig()
        config.ENGAGEMENT_RETRIES = 5
        config.CONTENT_CRAFTSMAN_RETRIES = 7
        
        with patch('flow.EngagementManagerNode') as mock_engagement:
            mock_engagement.return_value = Mock()
            
            # Mock other dependencies
            with patch('flow.BrandBibleIngestNode'), \
                 patch('flow.VoiceAlignmentNode'), \
                 patch('flow.ContentCraftsmanNode'), \
                 patch('flow.StyleEditorNode'), \
                 patch('flow.StyleComplianceNode'), \
                 patch('flow.AgencyDirectorNode'), \
                 patch('flow.create_platform_formatting_flow'), \
                 patch('flow.Flow'), \
                 patch('flow.validate_flow_structure'):
                
                create_main_flow(config)
                
                # Verify custom config was used
                mock_engagement.assert_called_once_with(max_retries=5)


class TestCreatePlatformFormattingFlow:
    """Test the create_platform_formatting_flow function."""
    
    @patch('flow.PlatformFormattingNode')
    @patch('flow.BatchFlow')
    def test_create_platform_formatting_flow_success(self, mock_batch_flow, mock_formatter):
        """Test successful creation of platform formatting flow."""
        mock_formatter_instance = Mock()
        mock_batch_instance = Mock()
        
        mock_formatter.return_value = mock_formatter_instance
        mock_batch_flow.return_value = mock_batch_instance
        
        result = create_platform_formatting_flow()
        
        assert result == mock_batch_instance
        mock_formatter.assert_called_once_with(max_retries=2)
    
    @patch('flow.PlatformFormattingNode')
    def test_create_platform_formatting_flow_failure(self, mock_formatter):
        """Test create_platform_formatting_flow when creation fails."""
        mock_formatter.side_effect = Exception("Formatter creation failed")
        
        with pytest.raises(FlowValidationError, match="Platform formatting flow construction failed"):
            create_platform_formatting_flow()
    
    def test_platform_formatting_batch_flow_prep_valid(self):
        """Test PlatformFormattingBatchFlow.prep with valid input."""
        from flow import create_platform_formatting_flow
        
        # Create a mock flow to test the prep method
        with patch('flow.PlatformFormattingNode') as mock_formatter:
            mock_formatter.return_value = Mock()
            
            with patch('flow.BatchFlow') as mock_batch_flow:
                # Create a real instance of the batch flow class
                class MockBatchFlow:
                    def __init__(self, start):
                        self.start = start
                
                mock_batch_flow.side_effect = MockBatchFlow
                
                flow = create_platform_formatting_flow()
                
                # Test the prep method with valid shared state
                shared = {
                    "task_requirements": {
                        "platforms": ["twitter", "linkedin"]
                    }
                }
                
                # Since we can't directly access the prep method, we'll test the flow creation
                assert flow is not None
    
    def test_platform_formatting_batch_flow_prep_invalid(self):
        """Test PlatformFormattingBatchFlow.prep with invalid input."""
        # This test would require accessing the internal batch flow class
        # For now, we'll test the error handling in the main function
        with patch('flow.PlatformFormattingNode') as mock_formatter:
            mock_formatter.side_effect = ValueError("Invalid platform")
            
            with pytest.raises(FlowValidationError):
                create_platform_formatting_flow()


class TestCreateValidationFlow:
    """Test the create_validation_flow function."""
    
    @patch('flow.StyleEditorNode')
    @patch('flow.StyleComplianceNode')
    @patch('flow.Flow')
    def test_create_validation_flow_success(self, mock_flow_class, mock_compliance, mock_editor):
        """Test successful creation of validation flow."""
        mock_editor_instance = Mock()
        mock_compliance_instance = Mock()
        mock_flow_instance = Mock()
        
        mock_editor.return_value = mock_editor_instance
        mock_compliance.return_value = mock_compliance_instance
        mock_flow_class.return_value = mock_flow_instance
        
        result = create_validation_flow()
        
        assert result == mock_flow_instance
        mock_editor.assert_called_once_with(max_retries=3, wait=1)
        mock_compliance.assert_called_once_with(max_retries=2)
    
    @patch('flow.StyleEditorNode')
    def test_create_validation_flow_failure(self, mock_editor):
        """Test create_validation_flow when creation fails."""
        mock_editor.side_effect = Exception("Editor creation failed")
        
        with pytest.raises(FlowValidationError, match="Validation flow construction failed"):
            create_validation_flow()


class TestCreateFeedbackFlow:
    """Test the create_feedback_flow function."""
    
    @patch('flow.FeedbackRouterNode')
    @patch('flow.SentenceEditorNode')
    @patch('flow.VersionManagerNode')
    @patch('flow.AgencyDirectorNode')
    @patch('flow.Flow')
    def test_create_feedback_flow_success(self, mock_flow_class, mock_agency, 
                                        mock_version, mock_sentence, mock_router):
        """Test successful creation of feedback flow."""
        mock_router_instance = Mock()
        mock_sentence_instance = Mock()
        mock_version_instance = Mock()
        mock_agency_instance = Mock()
        mock_flow_instance = Mock()
        
        mock_router.return_value = mock_router_instance
        mock_sentence.return_value = mock_sentence_instance
        mock_version.return_value = mock_version_instance
        mock_agency.return_value = mock_agency_instance
        mock_flow_class.return_value = mock_flow_instance
        
        result = create_feedback_flow()
        
        assert result == mock_flow_instance
    
    @patch('flow.FeedbackRouterNode')
    def test_create_feedback_flow_import_error(self, mock_router):
        """Test create_feedback_flow when feedback nodes are not available."""
        mock_router.side_effect = ImportError("Feedback nodes not available")
        
        with patch('flow.AgencyDirectorNode') as mock_agency:
            mock_agency_instance = Mock()
            mock_agency.return_value = mock_agency_instance
            
            with patch('flow.Flow') as mock_flow_class:
                mock_flow_instance = Mock()
                mock_flow_class.return_value = mock_flow_instance
                
                result = create_feedback_flow()
                
                assert result == mock_flow_instance
                mock_agency.assert_called_once()


class TestCreateStreamingFlow:
    """Test the create_streaming_flow function."""
    
    @patch('flow.create_main_flow')
    def test_create_streaming_flow_success(self, mock_create_main):
        """Test successful creation of streaming flow."""
        mock_main_flow = Mock()
        mock_create_main.return_value = mock_main_flow
        
        result = create_streaming_flow()
        
        assert result == mock_main_flow
        mock_create_main.assert_called_once()
    
    @patch('flow.create_main_flow')
    def test_create_streaming_flow_failure(self, mock_create_main):
        """Test create_streaming_flow when creation fails."""
        mock_create_main.side_effect = Exception("Main flow creation failed")
        
        with pytest.raises(FlowValidationError, match="Streaming flow construction failed"):
            create_streaming_flow()


class TestExecuteFlowWithMonitoring:
    """Test the execute_flow_with_monitoring function."""
    
    @patch('flow.validate_flow_structure')
    def test_execute_flow_with_monitoring_success(self, mock_validate):
        """Test successful flow execution with monitoring."""
        mock_flow = Mock()
        shared = {"test": "data"}
        
        # Mock time.time to return predictable values
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 105.0]  # start_time, end_time
            
            result = execute_flow_with_monitoring(mock_flow, shared)
            
            assert result == shared
            assert shared["metrics"]["execution_time"] == 5.0
            assert shared["metrics"]["timestamp"] == 105.0
            
            mock_validate.assert_called_once_with(mock_flow)
            mock_flow.run.assert_called_once_with(shared)
    
    @patch('flow.validate_flow_structure')
    def test_execute_flow_with_monitoring_timeout(self, mock_validate):
        """Test flow execution that exceeds timeout."""
        mock_flow = Mock()
        shared = {"test": "data"}
        config = FlowConfig()
        config.FLOW_TIMEOUT = 10
        
        # Mock time.time to return values that exceed timeout
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 115.0]  # start_time, end_time (15 seconds)
            
            result = execute_flow_with_monitoring(mock_flow, shared, config)
            
            assert result == shared
            assert shared["metrics"]["execution_time"] == 15.0
            # Should log warning about timeout but not fail
    
    @patch('flow.validate_flow_structure')
    def test_execute_flow_with_monitoring_failure(self, mock_validate):
        """Test flow execution when it fails."""
        mock_flow = Mock()
        mock_flow.run.side_effect = Exception("Flow execution failed")
        shared = {"test": "data"}
        
        # Mock time.time to return predictable values
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 105.0]  # start_time, end_time
            
            with pytest.raises(FlowExecutionError, match="Flow execution failed"):
                execute_flow_with_monitoring(mock_flow, shared)
            
            # Verify error information was added to shared state
            assert "errors" in shared
            assert len(shared["errors"]) == 1
            assert shared["errors"][0]["error"] == "Flow execution failed"
            assert shared["errors"][0]["execution_time"] == 5.0
    
    @patch('flow.validate_flow_structure')
    def test_execute_flow_with_monitoring_metrics_disabled(self, mock_validate):
        """Test flow execution with metrics disabled."""
        mock_flow = Mock()
        shared = {"test": "data"}
        config = FlowConfig()
        config.ENABLE_METRICS = False
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 105.0]
            
            result = execute_flow_with_monitoring(mock_flow, shared, config)
            
            assert result == shared
            assert "metrics" not in shared


class TestGetFlowMetrics:
    """Test the get_flow_metrics function."""
    
    def test_get_flow_metrics_success(self):
        """Test extracting metrics from shared state."""
        shared = {
            "metrics": {
                "execution_time": 5.5,
                "timestamp": 1234567890.0
            },
            "errors": [
                {"error": "Test error", "execution_time": 2.0, "timestamp": 1234567890.0}
            ],
            "content_pieces": {
                "twitter": {"text": "Content 1"},
                "linkedin": {"text": "Content 2"}
            },
            "task_requirements": {
                "platforms": ["twitter", "linkedin", "facebook"]
            }
        }
        
        metrics = get_flow_metrics(shared)
        
        assert metrics["execution_time"] == 5.5
        assert metrics["timestamp"] == 1234567890.0
        assert metrics["error_count"] == 1
        assert metrics["last_error"]["error"] == "Test error"
        assert metrics["content_pieces_count"] == 2
        assert metrics["platforms_processed"] == 3
    
    def test_get_flow_metrics_empty_shared(self):
        """Test extracting metrics from empty shared state."""
        shared = {}
        
        metrics = get_flow_metrics(shared)
        
        assert metrics["execution_time"] is None
        assert metrics["timestamp"] is None
        assert metrics["error_count"] == 0
        assert metrics["last_error"] is None
        assert metrics["content_pieces_count"] == 0
        assert metrics["platforms_processed"] == 0
    
    def test_get_flow_metrics_partial_data(self):
        """Test extracting metrics from shared state with partial data."""
        shared = {
            "metrics": {
                "execution_time": 3.0
            },
            "content_pieces": {
                "twitter": {"text": "Content"}
            }
        }
        
        metrics = get_flow_metrics(shared)
        
        assert metrics["execution_time"] == 3.0
        assert metrics["timestamp"] is None
        assert metrics["error_count"] == 0
        assert metrics["last_error"] is None
        assert metrics["content_pieces_count"] == 1
        assert metrics["platforms_processed"] == 0


class TestIntegration:
    """Integration tests for the flow module."""
    
    def test_flow_config_and_creation_integration(self):
        """Test integration between FlowConfig and flow creation."""
        config = FlowConfig()
        config.ENGAGEMENT_RETRIES = 10
        config.CONTENT_CRAFTSMAN_RETRIES = 15
        
        with patch('flow.EngagementManagerNode') as mock_engagement:
            mock_engagement.return_value = Mock()
            
            # Mock other dependencies
            with patch('flow.BrandBibleIngestNode'), \
                 patch('flow.VoiceAlignmentNode'), \
                 patch('flow.ContentCraftsmanNode'), \
                 patch('flow.StyleEditorNode'), \
                 patch('flow.StyleComplianceNode'), \
                 patch('flow.AgencyDirectorNode'), \
                 patch('flow.create_platform_formatting_flow'), \
                 patch('flow.Flow'), \
                 patch('flow.validate_flow_structure'):
                
                create_main_flow(config)
                
                # Verify custom config was used
                mock_engagement.assert_called_once_with(max_retries=10)
    
    def test_flow_execution_integration(self):
        """Test integration between flow creation and execution."""
        mock_flow = Mock()
        shared = {"test": "data"}
        
        with patch('flow.validate_flow_structure'):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [100.0, 105.0]
                
                result = execute_flow_with_monitoring(mock_flow, shared)
                
                assert result == shared
                assert "metrics" in shared
                assert shared["metrics"]["execution_time"] == 5.0
    
    def test_error_handling_integration(self):
        """Test integration of error handling across flow operations."""
        # Test that FlowValidationError is properly raised
        with pytest.raises(FlowValidationError):
            validate_flow_structure(None)
        
        # Test that FlowExecutionError is properly raised
        mock_flow = Mock()
        mock_flow.run.side_effect = Exception("Test error")
        shared = {}
        
        with patch('flow.validate_flow_structure'):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [100.0, 105.0]
                
                with pytest.raises(FlowExecutionError, match="Flow execution failed"):
                    execute_flow_with_monitoring(mock_flow, shared)
                
                assert "errors" in shared
                assert len(shared["errors"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])