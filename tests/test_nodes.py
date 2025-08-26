"""Comprehensive test suite for the nodes module.

This module provides extensive testing for all nodes in the Virtual PR Firm pipeline,
including validation, error handling, fallback behavior, and integration scenarios.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any, List

# Import the modules to test
from nodes import (
    handle_node_errors,
    validate_shared_state,
    sanitize_text,
    validate_platforms,
    emit_stream_milestone,
    EngagementManagerNode,
    BrandBibleIngestNode,
    VoiceAlignmentNode,
    PlatformFormattingNode,
    ContentCraftsmanNode,
    StyleEditorNode,
    StyleComplianceNode
)


class TestUtilityFunctions:
    """Test utility functions used by nodes."""
    
    def test_handle_node_errors_success(self):
        """Test error handling decorator with successful function."""
        @handle_node_errors
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_handle_node_errors_exception(self):
        """Test error handling decorator with exception."""
        @handle_node_errors
        def test_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            test_func()
    
    def test_validate_shared_state_valid(self):
        """Test shared state validation with valid input."""
        shared = {"key": "value"}
        validate_shared_state(shared)  # Should not raise
    
    def test_validate_shared_state_invalid_type(self):
        """Test shared state validation with invalid type."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_shared_state("not a dict")
    
    def test_validate_shared_state_missing_required_keys(self):
        """Test shared state validation with missing required keys."""
        shared = {"key": "value"}
        with pytest.raises(ValueError, match="Missing required key"):
            validate_shared_state(shared, required_keys=["missing_key"])
    
    def test_sanitize_text_valid(self):
        """Test text sanitization with valid input."""
        result = sanitize_text("test input")
        assert result == "test input"
    
    def test_sanitize_text_too_long(self):
        """Test text sanitization with input too long."""
        long_input = "x" * 10001
        with pytest.raises(ValueError, match="too long"):
            sanitize_text(long_input, max_length=10000)
    
    def test_sanitize_text_dangerous_chars(self):
        """Test text sanitization with dangerous characters."""
        dangerous_input = "test<script>alert('xss')</script>"
        result = sanitize_text(dangerous_input)
        assert "<script>" not in result
    
    def test_validate_platforms_valid(self):
        """Test platform validation with valid platforms."""
        platforms = ["twitter", "linkedin", "instagram"]
        result = validate_platforms(platforms)
        assert result == ["twitter", "linkedin", "instagram"]
    
    def test_validate_platforms_empty(self):
        """Test platform validation with empty list."""
        with pytest.raises(ValueError, match="At least one platform"):
            validate_platforms([])
    
    def test_validate_platforms_invalid_format(self):
        """Test platform validation with invalid format."""
        platforms = ["twitter", "invalid@platform", "linkedin"]
        result = validate_platforms(platforms)
        assert result == ["twitter", "linkedin"]  # Invalid one should be filtered out
    
    def test_emit_stream_milestone_with_stream(self):
        """Test stream milestone emission with available stream."""
        mock_stream = MagicMock()
        shared = {"stream": mock_stream}
        
        emit_stream_milestone(shared, "test message", "info")
        
        mock_stream.emit.assert_called_once_with("info", "test message")
    
    def test_emit_stream_milestone_no_stream(self):
        """Test stream milestone emission without stream."""
        shared = {"other_key": "value"}
        
        # Should not raise error
        emit_stream_milestone(shared, "test message", "info")
    
    def test_emit_stream_milestone_stream_error(self):
        """Test stream milestone emission with stream error."""
        mock_stream = MagicMock()
        mock_stream.emit.side_effect = Exception("Stream error")
        shared = {"stream": mock_stream}
        
        # Should not raise error, should handle gracefully
        emit_stream_milestone(shared, "test message", "info")


class TestEngagementManagerNode:
    """Test EngagementManagerNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = EngagementManagerNode()
        self.valid_shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test topic",
                "intents_by_platform": {
                    "twitter": {"value": "engagement"},
                    "linkedin": {"value": "thought_leadership"}
                }
            }
        }
    
    def test_prep_valid_input(self):
        """Test prep method with valid input."""
        result = self.node.prep(self.valid_shared)
        
        assert "platforms" in result
        assert "topic_or_goal" in result
        assert "intents_by_platform" in result
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Test topic"
    
    def test_prep_missing_task_requirements(self):
        """Test prep method with missing task_requirements."""
        shared = {"other_key": "value"}
        result = self.node.prep(shared)
        
        # Should use defaults
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Announce product"
    
    def test_prep_invalid_platforms(self):
        """Test prep method with invalid platforms."""
        shared = {
            "task_requirements": {
                "platforms": "not a list",
                "topic_or_goal": "Test topic"
            }
        }
        result = self.node.prep(shared)
        
        # Should use defaults for platforms
        assert result["platforms"] == ["twitter", "linkedin"]
    
    def test_prep_empty_topic(self):
        """Test prep method with empty topic."""
        shared = {
            "task_requirements": {
                "platforms": ["twitter"],
                "topic_or_goal": ""
            }
        }
        result = self.node.prep(shared)
        
        # Should use default topic
        assert result["topic_or_goal"] == "Announce product"
    
    def test_exec_valid_input(self):
        """Test exec method with valid input."""
        inputs = {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Test topic",
            "intents_by_platform": {}
        }
        
        result = self.node.exec(inputs)
        
        assert result["platforms"] == ["twitter", "linkedin"]
        assert result["topic_or_goal"] == "Test topic"
    
    def test_post_valid_result(self):
        """Test post method with valid result."""
        shared = {}
        prep_res = {"platforms": ["twitter"], "topic_or_goal": "Test"}
        exec_res = {"platforms": ["twitter"], "topic_or_goal": "Test"}
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert "task_requirements" in shared
        assert shared["task_requirements"]["platforms"] == ["twitter"]


class TestBrandBibleIngestNode:
    """Test BrandBibleIngestNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = BrandBibleIngestNode()
        self.valid_shared = {
            "brand_bible": {
                "xml_raw": """<brand_bible>
                    <name>Test Brand</name>
                    <voice>
                        <tone>professional</tone>
                        <forbidden_terms>bad,terrible</forbidden_terms>
                    </voice>
                </brand_bible>"""
            }
        }
    
    def test_prep_valid_xml(self):
        """Test prep method with valid XML."""
        result = self.node.prep(self.valid_shared)
        
        assert result == self.valid_shared["brand_bible"]["xml_raw"]
    
    def test_prep_missing_xml(self):
        """Test prep method with missing XML."""
        shared = {"brand_bible": {}}
        result = self.node.prep(shared)
        
        assert result == ""
    
    def test_exec_empty_xml(self):
        """Test exec method with empty XML."""
        result = self.node.exec("")
        
        assert "parsed" in result
        assert "warnings" in result
        assert result["warnings"] == ["no xml provided"]
    
    def test_exec_valid_xml(self):
        """Test exec method with valid XML."""
        xml_raw = """<brand_bible>
            <name>Test Brand</name>
            <voice>
                <tone>professional</tone>
            </voice>
        </brand_bible>"""
        
        result = self.node.exec(xml_raw)
        
        assert "parsed" in result
        assert "warnings" in result
    
    def test_post_valid_result(self):
        """Test post method with valid result."""
        shared = {}
        prep_res = "xml content"
        exec_res = {
            "parsed": {"name": "Test Brand"},
            "warnings": []
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert "brand_bible" in shared
        assert shared["brand_bible"]["parsed"]["name"] == "Test Brand"


class TestVoiceAlignmentNode:
    """Test VoiceAlignmentNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = VoiceAlignmentNode()
        self.valid_shared = {
            "brand_bible": {
                "parsed": {
                    "name": "Test Brand",
                    "voice": {
                        "tone": "professional",
                        "forbidden_terms": ["bad", "terrible"]
                    }
                }
            }
        }
    
    def test_prep_valid_parsed_data(self):
        """Test prep method with valid parsed data."""
        result = self.node.prep(self.valid_shared)
        
        assert result["name"] == "Test Brand"
        assert "voice" in result
    
    def test_prep_missing_parsed_data(self):
        """Test prep method with missing parsed data."""
        shared = {"brand_bible": {}}
        result = self.node.prep(shared)
        
        assert result == {}
    
    def test_exec_valid_parsed_data(self):
        """Test exec method with valid parsed data."""
        parsed = {
            "name": "Test Brand",
            "voice": {
                "tone": "professional",
                "forbidden_terms": ["bad", "terrible"]
            }
        }
        
        result = self.node.exec(parsed)
        
        assert "forbidden_terms" in result
        assert "required_phrases" in result
    
    def test_exec_empty_parsed_data(self):
        """Test exec method with empty parsed data."""
        result = self.node.exec({})
        
        assert "forbidden_terms" in result
        assert "required_phrases" in result
    
    def test_post_valid_result(self):
        """Test post method with valid result."""
        shared = {}
        prep_res = {"name": "Test Brand"}
        exec_res = {
            "forbidden_terms": ["bad"],
            "required_phrases": ["good"]
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert "brand_bible" in shared
        assert "persona_voice" in shared["brand_bible"]


class TestPlatformFormattingNode:
    """Test PlatformFormattingNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = PlatformFormattingNode()
        self.node.set_params({"platform": "twitter"})
        self.valid_shared = {
            "brand_bible": {
                "persona_voice": {
                    "forbidden_terms": ["bad"],
                    "required_phrases": ["good"]
                }
            },
            "task_requirements": {
                "intents_by_platform": {
                    "twitter": {"value": "engagement"}
                }
            }
        }
    
    def test_prep_valid_input(self):
        """Test prep method with valid input."""
        result = self.node.prep(self.valid_shared)
        
        assert len(result) == 3
        platform, persona, intent = result
        assert platform == "twitter"
        assert "forbidden_terms" in persona
        assert intent == "engagement"
    
    def test_prep_missing_params(self):
        """Test prep method with missing params."""
        node = PlatformFormattingNode()  # No params set
        result = node.prep(self.valid_shared)
        
        platform, persona, intent = result
        assert platform == "general"
    
    def test_exec_valid_input(self):
        """Test exec method with valid input."""
        inputs = ("twitter", {"forbidden_terms": ["bad"]}, "engagement")
        
        result = self.node.exec(inputs)
        
        assert "platform" in result
        assert result["platform"] == "twitter"
    
    def test_post_valid_result(self):
        """Test post method with valid result."""
        shared = {}
        prep_res = ("twitter", {}, "engagement")
        exec_res = {
            "platform": "twitter",
            "max_length": 280
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert "platform_guidelines" in shared
        assert "twitter" in shared["platform_guidelines"]


class TestContentCraftsmanNode:
    """Test ContentCraftsmanNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = ContentCraftsmanNode()
        self.valid_shared = {
            "platform_guidelines": {
                "twitter": {"max_length": 280},
                "linkedin": {"max_length": 2000}
            },
            "brand_bible": {
                "persona_voice": {
                    "forbidden_terms": ["bad"],
                    "required_phrases": ["good"]
                }
            },
            "task_requirements": {
                "topic_or_goal": "Test content generation"
            }
        }
    
    def test_prep_valid_input(self):
        """Test prep method with valid input."""
        result = self.node.prep(self.valid_shared)
        
        assert "platform_guidelines" in result
        assert "persona" in result
        assert "topic" in result
        assert result["topic"] == "Test content generation"
    
    def test_prep_missing_data(self):
        """Test prep method with missing data."""
        shared = {}
        result = self.node.prep(shared)
        
        assert result["platform_guidelines"] == {}
        assert result["persona"] == {}
        assert result["topic"] == ""
    
    def test_exec_valid_input(self):
        """Test exec method with valid input."""
        inputs = {
            "platform_guidelines": {"twitter": {"max_length": 280}},
            "persona": {"forbidden_terms": ["bad"]},
            "topic": "Test topic"
        }
        
        result = self.node.exec(inputs)
        
        assert "twitter" in result
        assert "Test topic" in result["twitter"]
    
    def test_post_valid_result(self):
        """Test post method with valid result."""
        shared = {}
        prep_res = {"platform_guidelines": {"twitter": {}}}
        exec_res = {
            "twitter": "Draft for twitter: Test topic"
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert "content_pieces" in shared
        assert "twitter" in shared["content_pieces"]


class TestStyleEditorNode:
    """Test StyleEditorNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = StyleEditorNode()
        self.valid_shared = {
            "content_pieces": {
                "twitter": {"text": "This is a test content with bad words"},
                "linkedin": {"text": "Professional content here"}
            },
            "brand_bible": {
                "persona_voice": {
                    "forbidden_terms": ["bad"],
                    "required_phrases": ["good"]
                }
            }
        }
    
    def test_prep_valid_input(self):
        """Test prep method with valid input."""
        result = self.node.prep(self.valid_shared)
        
        assert "content_pieces" in result
        assert "persona" in result
        assert "twitter" in result["content_pieces"]
    
    def test_prep_missing_data(self):
        """Test prep method with missing data."""
        shared = {}
        result = self.node.prep(shared)
        
        assert result["content_pieces"] == {}
        assert result["persona"] == {}
    
    def test_exec_valid_input(self):
        """Test exec method with valid input."""
        inputs = {
            "content_pieces": {
                "twitter": {"text": "This is a test content with bad words"}
            },
            "persona": {"forbidden_terms": ["bad"]}
        }
        
        result = self.node.exec(inputs)
        
        assert "twitter" in result
        # Should remove forbidden terms
        assert "bad" not in result["twitter"]
    
    def test_exec_with_em_dash(self):
        """Test exec method with em-dash replacement."""
        inputs = {
            "content_pieces": {
                "twitter": {"text": "This is a test—with em-dash"}
            },
            "persona": {"forbidden_terms": []}
        }
        
        result = self.node.exec(inputs)
        
        assert "twitter" in result
        # Should replace em-dash with regular dash
        assert "—" not in result["twitter"]
        assert "-" in result["twitter"]
    
    def test_post_valid_result(self):
        """Test post method with valid result."""
        shared = {}
        prep_res = {"content_pieces": {"twitter": {"text": "original"}}}
        exec_res = {"twitter": "rewritten content"}
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "default"
        assert "content_pieces" in shared
        assert shared["content_pieces"]["twitter"]["text"] == "rewritten content"


class TestStyleComplianceNode:
    """Test StyleComplianceNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = StyleComplianceNode()
        self.valid_shared = {
            "content_pieces": {
                "twitter": {"text": "This is compliant content"},
                "linkedin": {"text": "Professional content here"}
            }
        }
    
    def test_prep_valid_input(self):
        """Test prep method with valid input."""
        result = self.node.prep(self.valid_shared)
        
        assert "twitter" in result
        assert "linkedin" in result
    
    def test_prep_missing_data(self):
        """Test prep method with missing data."""
        shared = {}
        result = self.node.prep(shared)
        
        assert result == {}
    
    def test_exec_compliant_content(self):
        """Test exec method with compliant content."""
        content_pieces = {
            "twitter": {"text": "This is compliant content"},
            "linkedin": {"text": "Professional content here"}
        }
        
        result = self.node.exec(content_pieces)
        
        assert "twitter" in result
        assert "linkedin" in result
        assert "violations" in result["twitter"]
        assert "score" in result["twitter"]
    
    def test_exec_content_with_em_dash(self):
        """Test exec method with content containing em-dash."""
        content_pieces = {
            "twitter": {"text": "This has an em-dash—here"}
        }
        
        result = self.node.exec(content_pieces)
        
        assert "twitter" in result
        assert "violations" in result["twitter"]
        # Should detect em-dash violation
        violations = result["twitter"]["violations"]
        assert any(v["term"] == "mdash" for v in violations)
    
    def test_post_compliant_content(self):
        """Test post method with compliant content."""
        shared = {}
        prep_res = {"twitter": {"text": "compliant"}}
        exec_res = {
            "twitter": {"violations": [], "score": 100}
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "pass"
        assert "style_compliance" in shared
    
    def test_post_content_with_violations(self):
        """Test post method with content containing violations."""
        shared = {}
        prep_res = {"twitter": {"text": "with violations"}}
        exec_res = {
            "twitter": {"violations": [{"type": "forbidden"}], "score": 90}
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "revise"
        assert "style_compliance" in shared
        assert "workflow_state" in shared
        assert shared["workflow_state"]["revision_count"] == 1
    
    def test_post_max_revisions_reached(self):
        """Test post method when max revisions reached."""
        shared = {
            "workflow_state": {"revision_count": 5}
        }
        prep_res = {"twitter": {"text": "with violations"}}
        exec_res = {
            "twitter": {"violations": [{"type": "forbidden"}], "score": 90}
        }
        
        result = self.node.post(shared, prep_res, exec_res)
        
        assert result == "max_revisions"


class TestNodeIntegration:
    """Integration tests for node interactions."""
    
    def test_full_pipeline_integration(self):
        """Test full pipeline integration with sample data."""
        # Create sample shared state
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test integration",
                "intents_by_platform": {}
            },
            "brand_bible": {
                "xml_raw": """<brand_bible>
                    <name>Test Brand</name>
                    <voice>
                        <tone>professional</tone>
                        <forbidden_terms>bad,terrible</forbidden_terms>
                    </voice>
                </brand_bible>"""
            },
            "stream": None
        }
        
        # Test EngagementManagerNode
        engagement_node = EngagementManagerNode()
        engagement_result = engagement_node.prep(shared)
        engagement_node.post(shared, {}, engagement_result)
        
        assert "task_requirements" in shared
        assert shared["task_requirements"]["platforms"] == ["twitter", "linkedin"]
        
        # Test BrandBibleIngestNode
        brand_node = BrandBibleIngestNode()
        brand_result = brand_node.prep(shared)
        brand_exec_result = brand_node.exec(brand_result)
        brand_node.post(shared, brand_result, brand_exec_result)
        
        assert "brand_bible" in shared
        assert "parsed" in shared["brand_bible"]
        
        # Test VoiceAlignmentNode
        voice_node = VoiceAlignmentNode()
        voice_result = voice_node.prep(shared)
        voice_exec_result = voice_node.exec(voice_result)
        voice_node.post(shared, voice_result, voice_exec_result)
        
        assert "persona_voice" in shared["brand_bible"]
    
    def test_error_recovery_integration(self):
        """Test error recovery in integration scenarios."""
        # Test with invalid shared state
        shared = "invalid"
        
        with pytest.raises(ValueError):
            validate_shared_state(shared)
        
        # Test with missing required data
        shared = {"other_key": "value"}
        
        engagement_node = EngagementManagerNode()
        result = engagement_node.prep(shared)
        
        # Should use defaults and not crash
        assert "platforms" in result
        assert "topic_or_goal" in result


if __name__ == "__main__":
    pytest.main([__file__])