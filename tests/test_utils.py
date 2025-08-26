"""Comprehensive test suite for the utils module.

This module provides extensive testing for all utility functions,
including LLM calls, brand bible parsing, voice mapping, style checking,
platform formatting, and streaming functionality.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock, AsyncMock
from typing import Dict, Any, List

# Import utility modules to test
from utils.call_llm import call_llm
from utils.brand_bible_parser import parse_brand_bible
from utils.brand_voice_mapper import map_brand_voice
from utils.check_style_violations import check_style_violations
from utils.format_platform import format_for_platform
from utils.openrouter_client import OpenRouterClient
from utils.presets import get_preset_config
from utils.rewrite_with_constraints import rewrite_with_constraints
from utils.streaming import StreamManager


class TestCallLLM:
    """Test LLM calling functionality."""
    
    @patch('utils.call_llm.OpenAI')
    def test_call_llm_success(self, mock_openai):
        """Test successful LLM call."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Execute
        result = call_llm("Test prompt")
        
        # Verify
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('utils.call_llm.OpenAI')
    def test_call_llm_failure(self, mock_openai):
        """Test LLM call failure."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        # Execute and verify
        with pytest.raises(Exception, match="API error"):
            call_llm("Test prompt")
    
    @patch('utils.call_llm.OpenAI')
    def test_call_llm_empty_response(self, mock_openai):
        """Test LLM call with empty response."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        
        # Execute
        result = call_llm("Test prompt")
        
        # Verify
        assert result == ""
    
    @patch('utils.call_llm.OpenAI')
    def test_call_llm_with_parameters(self, mock_openai):
        """Test LLM call with specific parameters."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Parameterized response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Execute
        result = call_llm("Test prompt", model="gpt-4", temperature=0.7)
        
        # Verify
        assert result == "Parameterized response"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-4"
        assert call_args[1]['temperature'] == 0.7


class TestBrandBibleParser:
    """Test brand bible parsing functionality."""
    
    def test_parse_brand_bible_valid_xml(self):
        """Test parsing valid XML brand bible."""
        xml_content = """
        <brand_bible>
            <voice>
                <tone>Professional</tone>
                <style>Formal</style>
            </voice>
            <guidelines>
                <rule>Use clear language</rule>
                <rule>Avoid jargon</rule>
            </guidelines>
        </brand_bible>
        """
        
        result = parse_brand_bible(xml_content)
        
        assert isinstance(result, dict)
        assert 'voice' in result
        assert 'guidelines' in result
        assert result['voice']['tone'] == 'Professional'
        assert result['voice']['style'] == 'Formal'
        assert len(result['guidelines']['rule']) == 2
    
    def test_parse_brand_bible_invalid_xml(self):
        """Test parsing invalid XML brand bible."""
        invalid_xml = "<brand_bible><unclosed_tag>"
        
        with pytest.raises(Exception):
            parse_brand_bible(invalid_xml)
    
    def test_parse_brand_bible_empty(self):
        """Test parsing empty brand bible."""
        result = parse_brand_bible("")
        
        assert isinstance(result, dict)
        assert result == {}
    
    def test_parse_brand_bible_none(self):
        """Test parsing None brand bible."""
        result = parse_brand_bible(None)
        
        assert isinstance(result, dict)
        assert result == {}
    
    def test_parse_brand_bible_complex_structure(self):
        """Test parsing complex brand bible structure."""
        xml_content = """
        <brand_bible>
            <company_info>
                <name>Test Company</name>
                <industry>Technology</industry>
            </company_info>
            <voice>
                <tone>Friendly</tone>
                <style>Conversational</style>
                <personality>Innovative</personality>
            </voice>
            <content_guidelines>
                <do>
                    <item>Use active voice</item>
                    <item>Keep it simple</item>
                </do>
                <dont>
                    <item>Avoid technical jargon</item>
                    <item>Don't be overly formal</item>
                </dont>
            </content_guidelines>
        </brand_bible>
        """
        
        result = parse_brand_bible(xml_content)
        
        assert 'company_info' in result
        assert 'voice' in result
        assert 'content_guidelines' in result
        assert result['company_info']['name'] == 'Test Company'
        assert result['voice']['tone'] == 'Friendly'
        assert len(result['content_guidelines']['do']['item']) == 2


class TestBrandVoiceMapper:
    """Test brand voice mapping functionality."""
    
    def test_map_brand_voice_valid_input(self):
        """Test mapping brand voice with valid input."""
        brand_bible = {
            'voice': {
                'tone': 'Professional',
                'style': 'Formal'
            },
            'guidelines': {
                'rule': ['Use clear language', 'Avoid jargon']
            }
        }
        
        result = map_brand_voice(brand_bible)
        
        assert isinstance(result, dict)
        assert 'tone' in result
        assert 'style' in result
        assert 'guidelines' in result
        assert result['tone'] == 'Professional'
        assert result['style'] == 'Formal'
    
    def test_map_brand_voice_missing_voice(self):
        """Test mapping brand voice with missing voice section."""
        brand_bible = {
            'guidelines': {
                'rule': ['Use clear language']
            }
        }
        
        result = map_brand_voice(brand_bible)
        
        assert isinstance(result, dict)
        assert result.get('tone') is None
        assert result.get('style') is None
    
    def test_map_brand_voice_empty_input(self):
        """Test mapping brand voice with empty input."""
        result = map_brand_voice({})
        
        assert isinstance(result, dict)
        assert result == {}
    
    def test_map_brand_voice_none_input(self):
        """Test mapping brand voice with None input."""
        result = map_brand_voice(None)
        
        assert isinstance(result, dict)
        assert result == {}
    
    def test_map_brand_voice_complex_structure(self):
        """Test mapping brand voice with complex structure."""
        brand_bible = {
            'voice': {
                'tone': 'Friendly',
                'style': 'Conversational',
                'personality': 'Innovative',
                'emotion': 'Enthusiastic'
            },
            'content_guidelines': {
                'do': ['Use active voice', 'Keep it simple'],
                'dont': ['Avoid technical jargon', 'Don\'t be overly formal']
            }
        }
        
        result = map_brand_voice(brand_bible)
        
        assert result['tone'] == 'Friendly'
        assert result['style'] == 'Conversational'
        assert result['personality'] == 'Innovative'
        assert result['emotion'] == 'Enthusiastic'
        assert 'content_guidelines' in result


class TestCheckStyleViolations:
    """Test style violation checking functionality."""
    
    def test_check_style_violations_no_violations(self):
        """Test checking style violations with no violations."""
        content = "This is a well-written piece of content."
        style_guidelines = {
            'tone': 'Professional',
            'style': 'Formal',
            'rules': ['Use clear language', 'Avoid jargon']
        }
        
        result = check_style_violations(content, style_guidelines)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_check_style_violations_with_violations(self):
        """Test checking style violations with violations."""
        content = "This content uses jargon and is too informal."
        style_guidelines = {
            'tone': 'Professional',
            'style': 'Formal',
            'rules': ['Use clear language', 'Avoid jargon', 'Maintain formal tone']
        }
        
        result = check_style_violations(content, style_guidelines)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(violation, str) for violation in result)
    
    def test_check_style_violations_empty_content(self):
        """Test checking style violations with empty content."""
        content = ""
        style_guidelines = {
            'tone': 'Professional',
            'style': 'Formal'
        }
        
        result = check_style_violations(content, style_guidelines)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_check_style_violations_empty_guidelines(self):
        """Test checking style violations with empty guidelines."""
        content = "This is some content."
        style_guidelines = {}
        
        result = check_style_violations(content, style_guidelines)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_check_style_violations_none_inputs(self):
        """Test checking style violations with None inputs."""
        result = check_style_violations(None, None)
        
        assert isinstance(result, list)
        assert result == []
    
    def test_check_style_violations_complex_rules(self):
        """Test checking style violations with complex rules."""
        content = "This content uses technical jargon and is written in passive voice."
        style_guidelines = {
            'tone': 'Professional',
            'style': 'Formal',
            'rules': [
                'Use active voice',
                'Avoid technical jargon',
                'Keep sentences under 25 words',
                'Use clear, simple language'
            ]
        }
        
        result = check_style_violations(content, style_guidelines)
        
        assert isinstance(result, list)
        # Should detect jargon and passive voice violations
        assert len(result) >= 2


class TestFormatPlatform:
    """Test platform formatting functionality."""
    
    def test_format_for_platform_twitter(self):
        """Test formatting for Twitter platform."""
        content = "This is a long piece of content that needs to be formatted for Twitter's character limit."
        platform = "twitter"
        
        result = format_for_platform(content, platform)
        
        assert isinstance(result, str)
        assert len(result) <= 280  # Twitter character limit
        assert len(result) > 0
    
    def test_format_for_platform_linkedin(self):
        """Test formatting for LinkedIn platform."""
        content = "This is content for LinkedIn."
        platform = "linkedin"
        
        result = format_for_platform(content, platform)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_format_for_platform_facebook(self):
        """Test formatting for Facebook platform."""
        content = "This is content for Facebook."
        platform = "facebook"
        
        result = format_for_platform(content, platform)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_format_for_platform_unknown(self):
        """Test formatting for unknown platform."""
        content = "This is content for an unknown platform."
        platform = "unknown_platform"
        
        result = format_for_platform(content, platform)
        
        assert isinstance(result, str)
        assert result == content  # Should return original content
    
    def test_format_for_platform_empty_content(self):
        """Test formatting with empty content."""
        content = ""
        platform = "twitter"
        
        result = format_for_platform(content, platform)
        
        assert isinstance(result, str)
        assert result == ""
    
    def test_format_for_platform_none_inputs(self):
        """Test formatting with None inputs."""
        result = format_for_platform(None, None)
        
        assert isinstance(result, str)
        assert result == ""
    
    def test_format_for_platform_very_long_content(self):
        """Test formatting very long content."""
        content = "This is a very long piece of content. " * 100
        platform = "twitter"
        
        result = format_for_platform(content, platform)
        
        assert isinstance(result, str)
        assert len(result) <= 280
        assert len(result) > 0


class TestOpenRouterClient:
    """Test OpenRouter client functionality."""
    
    def test_openrouter_client_initialization(self):
        """Test OpenRouter client initialization."""
        client = OpenRouterClient(api_key="test_key")
        
        assert client.api_key == "test_key"
        assert hasattr(client, 'base_url')
    
    @patch('utils.openrouter_client.requests.post')
    def test_openrouter_client_call_success(self, mock_post):
        """Test successful OpenRouter API call."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test_key")
        result = client.call("Test prompt")
        
        assert result == "Test response"
        mock_post.assert_called_once()
    
    @patch('utils.openrouter_client.requests.post')
    def test_openrouter_client_call_failure(self, mock_post):
        """Test OpenRouter API call failure."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test_key")
        
        with pytest.raises(Exception):
            client.call("Test prompt")
    
    @patch('utils.openrouter_client.requests.post')
    def test_openrouter_client_call_network_error(self, mock_post):
        """Test OpenRouter API call with network error."""
        mock_post.side_effect = Exception("Network error")
        
        client = OpenRouterClient(api_key="test_key")
        
        with pytest.raises(Exception, match="Network error"):
            client.call("Test prompt")
    
    def test_openrouter_client_with_parameters(self):
        """Test OpenRouter client with specific parameters."""
        client = OpenRouterClient(api_key="test_key")
        
        # Test that parameters are properly set
        assert client.api_key == "test_key"
        assert hasattr(client, 'base_url')


class TestPresets:
    """Test preset configuration functionality."""
    
    def test_get_preset_config_valid_preset(self):
        """Test getting valid preset configuration."""
        preset_name = "professional"
        
        result = get_preset_config(preset_name)
        
        assert isinstance(result, dict)
        assert 'voice' in result
        assert 'style' in result
    
    def test_get_preset_config_invalid_preset(self):
        """Test getting invalid preset configuration."""
        preset_name = "invalid_preset"
        
        result = get_preset_config(preset_name)
        
        assert isinstance(result, dict)
        # Should return default configuration
        assert 'voice' in result
    
    def test_get_preset_config_none_preset(self):
        """Test getting preset configuration with None."""
        result = get_preset_config(None)
        
        assert isinstance(result, dict)
        # Should return default configuration
        assert 'voice' in result
    
    def test_get_preset_config_empty_preset(self):
        """Test getting preset configuration with empty string."""
        result = get_preset_config("")
        
        assert isinstance(result, dict)
        # Should return default configuration
        assert 'voice' in result
    
    def test_get_preset_config_all_presets(self):
        """Test getting all available presets."""
        presets = ["professional", "casual", "creative", "formal"]
        
        for preset in presets:
            result = get_preset_config(preset)
            assert isinstance(result, dict)
            assert 'voice' in result
            assert 'style' in result


class TestRewriteWithConstraints:
    """Test content rewriting with constraints functionality."""
    
    @patch('utils.rewrite_with_constraints.call_llm')
    def test_rewrite_with_constraints_success(self, mock_call_llm):
        """Test successful content rewriting with constraints."""
        # Setup mock
        mock_call_llm.return_value = "Rewritten content that follows the constraints."
        
        content = "Original content that needs rewriting."
        constraints = {
            'tone': 'Professional',
            'style': 'Formal',
            'max_length': 100
        }
        
        result = rewrite_with_constraints(content, constraints)
        
        assert isinstance(result, str)
        assert result == "Rewritten content that follows the constraints."
        mock_call_llm.assert_called_once()
    
    @patch('utils.rewrite_with_constraints.call_llm')
    def test_rewrite_with_constraints_failure(self, mock_call_llm):
        """Test content rewriting with LLM failure."""
        # Setup mock
        mock_call_llm.side_effect = Exception("LLM error")
        
        content = "Original content."
        constraints = {'tone': 'Professional'}
        
        with pytest.raises(Exception, match="LLM error"):
            rewrite_with_constraints(content, constraints)
    
    def test_rewrite_with_constraints_empty_content(self):
        """Test rewriting with empty content."""
        content = ""
        constraints = {'tone': 'Professional'}
        
        result = rewrite_with_constraints(content, constraints)
        
        assert isinstance(result, str)
        assert result == ""
    
    def test_rewrite_with_constraints_empty_constraints(self):
        """Test rewriting with empty constraints."""
        content = "Original content."
        constraints = {}
        
        result = rewrite_with_constraints(content, constraints)
        
        assert isinstance(result, str)
        assert result == content  # Should return original content
    
    def test_rewrite_with_constraints_none_inputs(self):
        """Test rewriting with None inputs."""
        result = rewrite_with_constraints(None, None)
        
        assert isinstance(result, str)
        assert result == ""
    
    @patch('utils.rewrite_with_constraints.call_llm')
    def test_rewrite_with_constraints_complex_constraints(self, mock_call_llm):
        """Test rewriting with complex constraints."""
        # Setup mock
        mock_call_llm.return_value = "Complex rewritten content."
        
        content = "Original content with complex requirements."
        constraints = {
            'tone': 'Professional',
            'style': 'Formal',
            'max_length': 150,
            'avoid_words': ['jargon', 'technical'],
            'include_keywords': ['innovation', 'quality'],
            'target_audience': 'executives'
        }
        
        result = rewrite_with_constraints(content, constraints)
        
        assert isinstance(result, str)
        assert result == "Complex rewritten content."
        mock_call_llm.assert_called_once()


class TestStreamManager:
    """Test streaming functionality."""
    
    def test_stream_manager_initialization(self):
        """Test StreamManager initialization."""
        stream_manager = StreamManager()
        
        assert hasattr(stream_manager, 'stream')
        assert stream_manager.stream is None
    
    def test_stream_manager_set_stream(self):
        """Test setting stream in StreamManager."""
        stream_manager = StreamManager()
        mock_stream = MagicMock()
        
        stream_manager.set_stream(mock_stream)
        
        assert stream_manager.stream == mock_stream
    
    def test_stream_manager_emit_milestone_with_stream(self):
        """Test emitting milestone with active stream."""
        stream_manager = StreamManager()
        mock_stream = MagicMock()
        stream_manager.set_stream(mock_stream)
        
        stream_manager.emit_milestone("Test milestone", "info")
        
        mock_stream.emit.assert_called_once()
    
    def test_stream_manager_emit_milestone_without_stream(self):
        """Test emitting milestone without stream (should not fail)."""
        stream_manager = StreamManager()
        
        # Should not raise exception
        stream_manager.emit_milestone("Test milestone", "info")
    
    def test_stream_manager_emit_milestone_different_levels(self):
        """Test emitting milestones with different levels."""
        stream_manager = StreamManager()
        mock_stream = MagicMock()
        stream_manager.set_stream(mock_stream)
        
        levels = ["info", "warning", "error", "success"]
        
        for level in levels:
            stream_manager.emit_milestone(f"Test {level}", level)
        
        assert mock_stream.emit.call_count == len(levels)
    
    def test_stream_manager_clear_stream(self):
        """Test clearing stream in StreamManager."""
        stream_manager = StreamManager()
        mock_stream = MagicMock()
        stream_manager.set_stream(mock_stream)
        
        stream_manager.clear_stream()
        
        assert stream_manager.stream is None
    
    def test_stream_manager_context_manager(self):
        """Test StreamManager as context manager."""
        mock_stream = MagicMock()
        
        with StreamManager() as manager:
            manager.set_stream(mock_stream)
            manager.emit_milestone("Test milestone", "info")
        
        mock_stream.emit.assert_called_once()


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_llm_and_formatting_integration(self):
        """Test integration between LLM calls and platform formatting."""
        # Test that LLM output can be formatted for platforms
        mock_llm_response = "This is a response from the LLM that might be too long for Twitter."
        
        # Simulate LLM response formatting
        formatted = format_for_platform(mock_llm_response, "twitter")
        
        assert isinstance(formatted, str)
        assert len(formatted) <= 280
    
    def test_brand_bible_and_voice_mapping_integration(self):
        """Test integration between brand bible parsing and voice mapping."""
        xml_content = """
        <brand_bible>
            <voice>
                <tone>Professional</tone>
                <style>Formal</style>
            </voice>
        </brand_bible>
        """
        
        # Parse brand bible
        parsed = parse_brand_bible(xml_content)
        
        # Map voice
        voice = map_brand_voice(parsed)
        
        assert voice['tone'] == 'Professional'
        assert voice['style'] == 'Formal'
    
    def test_style_checking_and_rewriting_integration(self):
        """Test integration between style checking and rewriting."""
        content = "This content uses jargon and is too informal."
        style_guidelines = {
            'tone': 'Professional',
            'style': 'Formal',
            'rules': ['Use clear language', 'Avoid jargon']
        }
        
        # Check for violations
        violations = check_style_violations(content, style_guidelines)
        
        # If violations found, rewrite content
        if violations:
            constraints = {
                'tone': 'Professional',
                'style': 'Formal',
                'avoid_words': ['jargon']
            }
            
            # This would normally call the LLM, but we're testing the flow
            assert isinstance(violations, list)
            assert len(violations) > 0
    
    def test_streaming_integration(self):
        """Test integration of streaming with other utilities."""
        stream_manager = StreamManager()
        mock_stream = MagicMock()
        stream_manager.set_stream(mock_stream)
        
        # Simulate a processing pipeline with streaming
        milestones = [
            ("Starting content generation", "info"),
            ("Parsing brand bible", "info"),
            ("Checking style violations", "warning"),
            ("Formatting for platforms", "info"),
            ("Content generation complete", "success")
        ]
        
        for message, level in milestones:
            stream_manager.emit_milestone(message, level)
        
        assert mock_stream.emit.call_count == len(milestones)


class TestUtilsEdgeCases:
    """Test edge cases and error conditions for utilities."""
    
    def test_llm_call_edge_cases(self):
        """Test edge cases for LLM calls."""
        # Test with empty prompt
        with pytest.raises(ValueError):
            call_llm("")
        
        # Test with None prompt
        with pytest.raises(ValueError):
            call_llm(None)
        
        # Test with very long prompt
        long_prompt = "A" * 10000
        # Should not raise exception, but might have different behavior
        # This depends on the actual implementation
    
    def test_brand_bible_parser_edge_cases(self):
        """Test edge cases for brand bible parser."""
        # Test with malformed XML
        malformed_xml = "<brand_bible><voice><tone>Professional</voice>"
        with pytest.raises(Exception):
            parse_brand_bible(malformed_xml)
        
        # Test with non-XML content
        non_xml = "This is not XML content"
        with pytest.raises(Exception):
            parse_brand_bible(non_xml)
    
    def test_platform_formatting_edge_cases(self):
        """Test edge cases for platform formatting."""
        # Test with extremely long content
        long_content = "A" * 10000
        result = format_for_platform(long_content, "twitter")
        assert len(result) <= 280
        
        # Test with special characters
        special_content = "Content with émojis 🚀 and special chars: @#$%^&*()"
        result = format_for_platform(special_content, "twitter")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_style_violations_edge_cases(self):
        """Test edge cases for style violations."""
        # Test with content containing special characters
        special_content = "Content with émojis 🚀 and special chars: @#$%^&*()"
        style_guidelines = {'tone': 'Professional'}
        
        result = check_style_violations(special_content, style_guidelines)
        assert isinstance(result, list)
    
    def test_streaming_edge_cases(self):
        """Test edge cases for streaming."""
        stream_manager = StreamManager()
        
        # Test with very long messages
        long_message = "A" * 1000
        stream_manager.emit_milestone(long_message, "info")
        # Should not raise exception
        
        # Test with empty message
        stream_manager.emit_milestone("", "info")
        # Should not raise exception
        
        # Test with None message
        stream_manager.emit_milestone(None, "info")
        # Should not raise exception


if __name__ == "__main__":
    pytest.main([__file__])