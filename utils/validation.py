"""
Validation utilities for the Virtual PR Firm.

This module provides comprehensive validation for shared store data, user inputs,
and platform configurations to ensure data integrity throughout the pipeline.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Supported platforms and their constraints
SUPPORTED_PLATFORMS = {
    "twitter": {"max_length": 280, "hashtags": True, "mentions": True},
    "linkedin": {"max_length": 3000, "hashtags": True, "mentions": True},
    "facebook": {"max_length": 63206, "hashtags": True, "mentions": True},
    "instagram": {"max_length": 2200, "hashtags": True, "mentions": True},
    "reddit": {"max_length": 40000, "hashtags": False, "mentions": True},
    "email": {"max_length": 100000, "hashtags": False, "mentions": False},
    "blog": {"max_length": 100000, "hashtags": False, "mentions": False}
}

# Forbidden style elements
FORBIDDEN_STYLE_ELEMENTS = {
    "em_dash": "—",
    "rhetorical_contrasts": ["not just", "not only", "not X, but Y", "not X; it's Y"],
    "tagline_framing": ["not just X; it's Y", "not X, but Y"]
}

@dataclass
class ValidationError:
    """Represents a validation error with context."""
    field: str
    message: str
    value: Any = None
    severity: str = "error"  # error, warning, info

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.is_valid: bool = True
    
    def add_error(self, field: str, message: str, value: Any = None):
        """Add a validation error."""
        self.errors.append(ValidationError(field, message, value, "error"))
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, value: Any = None):
        """Add a validation warning."""
        self.warnings.append(ValidationError(field, message, value, "warning"))
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": [{"field": e.field, "message": e.message, "value": e.value} 
                      for e in self.errors],
            "warnings": [{"field": w.field, "message": w.message, "value": w.value} 
                        for w in self.warnings]
        }

def validate_shared_store(shared: Dict[str, Any]) -> ValidationResult:
    """
    Validate the complete shared store structure.
    
    Args:
        shared: The shared store dictionary to validate
        
    Returns:
        ValidationResult with any errors or warnings
    """
    result = ValidationResult()
    
    # Check if shared is a dictionary
    if not isinstance(shared, dict):
        result.add_error("shared", "Shared store must be a dictionary", shared)
        return result
    
    # Validate task_requirements
    if "task_requirements" in shared:
        task_result = validate_task_requirements(shared["task_requirements"])
        result.merge(task_result)
    else:
        result.add_warning("task_requirements", "Missing task_requirements in shared store")
    
    # Validate brand_bible
    if "brand_bible" in shared:
        bible_result = validate_brand_bible(shared["brand_bible"])
        result.merge(bible_result)
    
    # Validate content_pieces if present
    if "content_pieces" in shared:
        content_result = validate_content_pieces(shared["content_pieces"])
        result.merge(content_result)
    
    # Validate workflow_state if present
    if "workflow_state" in shared:
        state_result = validate_workflow_state(shared["workflow_state"])
        result.merge(state_result)
    
    return result

def validate_task_requirements(task_req: Dict[str, Any]) -> ValidationResult:
    """Validate task_requirements structure."""
    result = ValidationResult()
    
    if not isinstance(task_req, dict):
        result.add_error("task_requirements", "Must be a dictionary", task_req)
        return result
    
    # Validate platforms
    if "platforms" in task_req:
        platform_result = validate_platforms(task_req["platforms"])
        result.merge(platform_result)
    else:
        result.add_warning("platforms", "Missing platforms in task_requirements")
    
    # Validate topic_or_goal
    if "topic_or_goal" in task_req:
        topic = task_req["topic_or_goal"]
        if not isinstance(topic, str):
            result.add_error("topic_or_goal", "Must be a string", topic)
        elif len(topic.strip()) == 0:
            result.add_error("topic_or_goal", "Cannot be empty", topic)
        elif len(topic) > 1000:
            result.add_warning("topic_or_goal", "Topic is very long, may affect performance", topic)
    else:
        result.add_warning("topic_or_goal", "Missing topic_or_goal in task_requirements")
    
    # Validate intents_by_platform if present
    if "intents_by_platform" in task_req:
        intents = task_req["intents_by_platform"]
        if not isinstance(intents, dict):
            result.add_error("intents_by_platform", "Must be a dictionary", intents)
        else:
            for platform, intent in intents.items():
                if platform not in SUPPORTED_PLATFORMS:
                    result.add_warning(f"intents_by_platform.{platform}", 
                                     f"Platform '{platform}' not in supported list")
                if not isinstance(intent, dict):
                    result.add_error(f"intents_by_platform.{platform}", 
                                   "Intent must be a dictionary", intent)
    
    return result

def validate_platforms(platforms: Any) -> ValidationResult:
    """Validate platform list."""
    result = ValidationResult()
    
    if not isinstance(platforms, list):
        result.add_error("platforms", "Must be a list", platforms)
        return result
    
    if len(platforms) == 0:
        result.add_error("platforms", "Cannot be empty", platforms)
        return result
    
    # Check for duplicates
    seen_platforms = set()
    for i, platform in enumerate(platforms):
        if not isinstance(platform, str):
            result.add_error(f"platforms[{i}]", "Must be a string", platform)
            continue
        
        platform_lower = platform.lower().strip()
        if platform_lower in seen_platforms:
            result.add_warning(f"platforms[{i}]", f"Duplicate platform '{platform}'")
        
        seen_platforms.add(platform_lower)
        
        # Check if platform is supported
        if platform_lower not in SUPPORTED_PLATFORMS:
            result.add_warning(f"platforms[{i}]", 
                             f"Platform '{platform}' not in supported list: {list(SUPPORTED_PLATFORMS.keys())}")
    
    return result

def validate_brand_bible(brand_bible: Dict[str, Any]) -> ValidationResult:
    """Validate brand_bible structure."""
    result = ValidationResult()
    
    if not isinstance(brand_bible, dict):
        result.add_error("brand_bible", "Must be a dictionary", brand_bible)
        return result
    
    # Validate xml_raw if present
    if "xml_raw" in brand_bible:
        xml_raw = brand_bible["xml_raw"]
        if not isinstance(xml_raw, str):
            result.add_error("brand_bible.xml_raw", "Must be a string", xml_raw)
        elif len(xml_raw) > 100000:
            result.add_warning("brand_bible.xml_raw", "XML content is very large", len(xml_raw))
    
    # Validate parsed_data if present
    if "parsed_data" in brand_bible:
        parsed = brand_bible["parsed_data"]
        if not isinstance(parsed, dict):
            result.add_error("brand_bible.parsed_data", "Must be a dictionary", parsed)
    
    return result

def validate_content_pieces(content_pieces: Dict[str, Any]) -> ValidationResult:
    """Validate content_pieces structure."""
    result = ValidationResult()
    
    if not isinstance(content_pieces, dict):
        result.add_error("content_pieces", "Must be a dictionary", content_pieces)
        return result
    
    for platform, content in content_pieces.items():
        if not isinstance(content, str):
            result.add_error(f"content_pieces.{platform}", "Content must be a string", content)
            continue
        
        # Check content length against platform limits
        if platform in SUPPORTED_PLATFORMS:
            max_length = SUPPORTED_PLATFORMS[platform]["max_length"]
            if len(content) > max_length:
                result.add_error(f"content_pieces.{platform}", 
                               f"Content exceeds {platform} limit of {max_length} characters", 
                               len(content))
        
        # Check for forbidden style elements
        style_result = validate_style_compliance(content, platform)
        for error in style_result.errors:
            result.add_error(f"content_pieces.{platform}.style", error.message, error.value)
    
    return result

def validate_workflow_state(workflow_state: Dict[str, Any]) -> ValidationResult:
    """Validate workflow_state structure."""
    result = ValidationResult()
    
    if not isinstance(workflow_state, dict):
        result.add_error("workflow_state", "Must be a dictionary", workflow_state)
        return result
    
    # Validate revision_count if present
    if "revision_count" in workflow_state:
        revision_count = workflow_state["revision_count"]
        if not isinstance(revision_count, int):
            result.add_error("workflow_state.revision_count", "Must be an integer", revision_count)
        elif revision_count < 0:
            result.add_error("workflow_state.revision_count", "Cannot be negative", revision_count)
        elif revision_count > 10:
            result.add_warning("workflow_state.revision_count", "High revision count may indicate issues", revision_count)
    
    # Validate current_node if present
    if "current_node" in workflow_state:
        current_node = workflow_state["current_node"]
        if not isinstance(current_node, str):
            result.add_error("workflow_state.current_node", "Must be a string", current_node)
    
    return result

def validate_style_compliance(content: str, platform: str) -> ValidationResult:
    """Validate style compliance for content."""
    result = ValidationResult()
    
    # Check for em-dash
    if FORBIDDEN_STYLE_ELEMENTS["em_dash"] in content:
        result.add_error("em_dash", "Em-dash (—) is forbidden", content)
    
    # Check for rhetorical contrasts
    for pattern in FORBIDDEN_STYLE_ELEMENTS["rhetorical_contrasts"]:
        if pattern.lower() in content.lower():
            result.add_error("rhetorical_contrast", f"Rhetorical contrast pattern '{pattern}' is forbidden", pattern)
    
    # Check for tagline framing
    for pattern in FORBIDDEN_STYLE_ELEMENTS["tagline_framing"]:
        if pattern.lower() in content.lower():
            result.add_error("tagline_framing", f"Tagline framing pattern '{pattern}' is forbidden", pattern)
    
    return result

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not isinstance(text, str):
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<script>', '</script>', 'javascript:', 'data:', 'vbscript:']
    sanitized = text
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Limit length
    if len(sanitized) > 10000:
        sanitized = sanitized[:10000]
    
    return sanitized.strip()

def normalize_platform_name(platform: str) -> str:
    """Normalize platform name to standard format."""
    if not isinstance(platform, str):
        return ""
    
    normalized = platform.lower().strip()
    
    # Handle common variations
    platform_mapping = {
        "x": "twitter",
        "twitter/x": "twitter",
        "fb": "facebook",
        "ig": "instagram",
        "insta": "instagram",
        "li": "linkedin",
        "mail": "email",
        "e-mail": "email"
    }
    
    return platform_mapping.get(normalized, normalized)

# Test function for development
if __name__ == "__main__":
    # Test validation functions
    test_shared = {
        "task_requirements": {
            "platforms": ["twitter", "linkedin"],
            "topic_or_goal": "Announce product launch"
        },
        "brand_bible": {
            "xml_raw": "<brand>...</brand>"
        }
    }
    
    result = validate_shared_store(test_shared)
    print("Validation result:", result.to_dict())
    
    # Test style compliance
    test_content = "This is not just a product—it's a revolution!"
    style_result = validate_style_compliance(test_content, "twitter")
    print("Style validation:", style_result.to_dict())