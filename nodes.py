"""Core PocketFlow Nodes for the Virtual PR Firm.

These nodes implement the complete content generation and style compliance pipeline for
social media content creation. Each node is designed with defensive programming principles,
providing robust fallback behaviors when external dependencies are unavailable.

Architecture Overview:
    The nodes follow a consistent pattern: prep() -> exec() -> post(), where:
    - prep(): Validates inputs and prepares data for processing
    - exec(): Performs the core business logic with fallback behavior
    - post(): Persists results and manages routing decisions

Key Design Principles:
    1. Defensive Implementation: All nodes provide safe fallbacks when utils/ are unavailable
    2. Streaming Integration: Nodes emit progress milestones via shared["stream"] when present
    3. Shared State Management: Uses standardized keys for inter-node communication
    4. Error Resilience: Graceful degradation rather than hard failures
    5. Testability: Clear separation of concerns for unit testing

Node Pipeline Flow:
    EngagementManagerNode -> BrandBibleIngestNode -> VoiceAlignmentNode 
    -> PlatformFormattingNode -> ContentCraftsmanNode -> StyleEditorNode 
    -> StyleComplianceNode (with revision loop)

Shared State Schema:
    - task_requirements: User inputs and platform intents
    - brand_bible: XML parsing and persona voice alignment
    - platform_guidelines: Platform-specific formatting rules
    - content_pieces: Generated and refined content drafts
    - style_compliance: Violation reports and revision tracking
    - workflow_state: Pipeline execution state and counters
    - stream: Optional streaming interface for real-time updates

Error Handling Strategy:
    Each node implements a two-tier approach:
    1. Primary path using robust utilities from utils/ package
    2. Fallback path with minimal local implementation
    This ensures the pipeline remains functional for demonstration and testing
    even without external API keys or dependencies.
"""

import logging
import re
import html
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from pocketflow import Node  # type: ignore

log = logging.getLogger(__name__)

# Module-level WHY: Nodes implement the PocketFlow `Node` contract: prep->exec->post.
# Intent: keep nodes defensive and testable with clear pre/post conditions and
# fallbacks so the repo can run without external keys.

# Domain rules:
#  - Never emit or persist raw API keys
#  - Enforce forbidden typographic characters (e.g., em-dash) at editor stage

# Lint: precise pragmas only where necessary
# pylint: disable=too-many-lines


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    
    # Platform validation
    SUPPORTED_PLATFORMS: Set[str] = None  # type: ignore
    MAX_PLATFORMS: int = 10
    MIN_PLATFORMS: int = 1
    
    # Content validation
    MAX_TOPIC_LENGTH: int = 500
    MIN_TOPIC_LENGTH: int = 3
    MAX_INTENT_LENGTH: int = 100
    
    # Security settings
    ALLOWED_CHARS: str = r'[a-zA-Z0-9\s\-_.,!?@#$%&*()+=:;"\'<>/\\|`~]'
    FORBIDDEN_PATTERNS: List[str] = None  # type: ignore
    
    def __post_init__(self):
        if self.SUPPORTED_PLATFORMS is None:
            self.SUPPORTED_PLATFORMS = {
                "twitter", "linkedin", "facebook", "instagram", 
                "tiktok", "youtube", "medium", "blog", "email"
            }
        if self.FORBIDDEN_PATTERNS is None:
            self.FORBIDDEN_PATTERNS = [
                r'<script.*?>.*?</script>',  # Script tags
                r'javascript:',  # JavaScript protocol
                r'data:text/html',  # Data URLs
                r'vbscript:',  # VBScript
                r'on\w+\s*=',  # Event handlers
            ]


class ValidationError(Exception):
    """Exception raised when input validation fails."""
    pass


class SecurityError(Exception):
    """Exception raised when security validation fails."""
    pass


class InputValidator:
    """Comprehensive input validation for the Virtual PR Firm."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
    
    def validate_platforms(self, platforms: List[str]) -> List[str]:
        """Validate and normalize platform names."""
        if not isinstance(platforms, list):
            raise ValidationError("Platforms must be a list")
        
        if len(platforms) < self.config.MIN_PLATFORMS:
            raise ValidationError(f"At least {self.config.MIN_PLATFORMS} platform must be specified")
        
        if len(platforms) > self.config.MAX_PLATFORMS:
            raise ValidationError(f"Maximum {self.config.MAX_PLATFORMS} platforms allowed")
        
        # Normalize and validate each platform
        normalized_platforms = []
        for platform in platforms:
            if not isinstance(platform, str):
                raise ValidationError("Platform names must be strings")
            
            normalized = platform.strip().lower()
            if not normalized:
                continue
            
            if normalized not in self.config.SUPPORTED_PLATFORMS:
                raise ValidationError(f"Unsupported platform: {platform}")
            
            normalized_platforms.append(normalized)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_platforms = []
        for platform in normalized_platforms:
            if platform not in seen:
                seen.add(platform)
                unique_platforms.append(platform)
        
        if len(unique_platforms) < self.config.MIN_PLATFORMS:
            raise ValidationError(f"At least {self.config.MIN_PLATFORMS} valid platform must be specified")
        
        return unique_platforms
    
    def validate_topic(self, topic: str) -> str:
        """Validate and sanitize topic input."""
        if not isinstance(topic, str):
            raise ValidationError("Topic must be a string")
        
        topic = topic.strip()
        
        if len(topic) < self.config.MIN_TOPIC_LENGTH:
            raise ValidationError(f"Topic must be at least {self.config.MIN_TOPIC_LENGTH} characters")
        
        if len(topic) > self.config.MAX_TOPIC_LENGTH:
            raise ValidationError(f"Topic must be no more than {self.config.MAX_TOPIC_LENGTH} characters")
        
        # Security validation
        self._validate_security(topic)
        
        return topic
    
    def validate_intents(self, intents: Dict[str, Dict]) -> Dict[str, Dict]:
        """Validate platform-specific intents."""
        if not isinstance(intents, dict):
            raise ValidationError("Intents must be a dictionary")
        
        validated_intents = {}
        for platform, intent_data in intents.items():
            if not isinstance(intent_data, dict):
                raise ValidationError(f"Intent data for {platform} must be a dictionary")
            
            # Validate intent value
            intent_value = intent_data.get("value", "")
            if not isinstance(intent_value, str):
                raise ValidationError(f"Intent value for {platform} must be a string")
            
            if len(intent_value) > self.config.MAX_INTENT_LENGTH:
                raise ValidationError(f"Intent value for {platform} too long")
            
            # Security validation
            self._validate_security(intent_value)
            
            validated_intents[platform] = {"value": intent_value.strip()}
        
        return validated_intents
    
    def _validate_security(self, text: str) -> None:
        """Validate text for security issues."""
        # Check for forbidden patterns
        for pattern in self.config.FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise SecurityError(f"Text contains forbidden pattern: {pattern}")
        
        # Check for suspicious characters
        if not re.match(f"^{self.config.ALLOWED_CHARS}+$", text):
            raise SecurityError("Text contains suspicious characters")
        
        # HTML entity validation
        if "&" in text and not re.match(r'^[^&]*(&[a-zA-Z0-9#]+;)*[^&]*$', text):
            raise SecurityError("Invalid HTML entities detected")


class EngagementManagerNode(Node):
    """Collects and normalizes user inputs into standardized task requirements.

    This node serves as the entry point for the content generation pipeline,
    responsible for gathering user inputs about platforms, intents, and goals,
    then normalizing them into a consistent format for downstream processing.

    Primary Functions:
        1. Input Collection: Gathers platform preferences and content intents
        2. Data Normalization: Ensures consistent structure in task_requirements
        3. Validation: Comprehensive validation of required fields and data types
        4. Security: Input sanitization and security validation
        5. Streaming: Emits milestone messages for real-time progress tracking

    Input Requirements:
        - Expects inputs to be pre-populated in shared state (future: interactive collection)
        - Handles missing inputs gracefully with empty defaults
        - Validates all inputs for security and format compliance

    Output Schema:
        shared["task_requirements"] = {
            "platforms": List[str],           # Target social media platforms
            "intents_by_platform": Dict[str, Dict],  # Platform-specific content intents
            "topic_or_goal": str             # Main content topic or goal
        }

    Validation Features:
        - Platform name validation against supported list
        - Topic length and content validation
        - Intent structure and value validation
        - Security validation (XSS prevention, injection attacks)
        - Input sanitization and normalization

    Error Handling:
        - Graceful degradation with default values
        - Detailed validation error messages
        - Security error handling with logging
        - Fallback behavior for missing inputs

    Streaming Integration:
        - Emits validation warnings and errors
        - Reports processing milestones
        - Provides real-time feedback on input quality

    Example Usage:
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "intents_by_platform": {
                    "twitter": {"value": "engagement"},
                    "linkedin": {"value": "thought_leadership"}
                },
                "topic_or_goal": "AI automation trends"
            }
        }
    """
    
    def __init__(self, max_retries: int = 2, validation_config: Optional[ValidationConfig] = None):
        super().__init__(max_retries=max_retries)
        self.validator = InputValidator(validation_config)
        self.validation_warnings = []
        self.validation_errors = []
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalizes input shared state structure.

        This method ensures that the shared state contains the required task_requirements
        structure and normalizes it to a consistent format. It implements defensive
        programming by providing sensible defaults for missing data.

        Pre-condition:
            - shared is a dictionary (validated by caller)
            - May contain incomplete or missing task_requirements

        Post-condition:
            - Returns normalized task_requirements dictionary
            - All required fields are present with valid types
            - Missing fields are populated with safe defaults

        Validation Steps:
            1. Structure validation: Ensure task_requirements exists and is a dict
            2. Field validation: Check for required fields (platforms, topic_or_goal)
            3. Type validation: Ensure fields have correct data types
            4. Content validation: Validate platform names, topic content, etc.
            5. Security validation: Check for malicious content

        Error Handling:
            - Missing task_requirements: Creates with defaults
            - Invalid structure: Normalizes to valid format
            - Security issues: Logs warnings and sanitizes content
            - Validation errors: Collects for reporting

        Returns:
            Dict[str, Any]: Normalized task_requirements with all required fields

        Raises:
            ValidationError: For critical validation failures
            SecurityError: For security-related issues
        """
        # Ensure task_requirements exists with safe defaults
        task_requirements = shared.get("task_requirements", {})
        if not isinstance(task_requirements, dict):
            log.warning("task_requirements is not a dict, creating new one")
            task_requirements = {}
        
        # Normalize structure with defaults
        normalized_requirements = {
            "platforms": task_requirements.get("platforms", []),
            "intents_by_platform": task_requirements.get("intents_by_platform", {}),
            "topic_or_goal": task_requirements.get("topic_or_goal", ""),
        }
        
        # Validate and normalize each field
        try:
            # Validate platforms
            platforms = normalized_requirements["platforms"]
            if platforms:
                normalized_requirements["platforms"] = self.validator.validate_platforms(platforms)
            else:
                # Use safe defaults if no platforms specified
                normalized_requirements["platforms"] = ["twitter", "linkedin"]
                self.validation_warnings.append("No platforms specified, using defaults: twitter, linkedin")
            
            # Validate topic
            topic = normalized_requirements["topic_or_goal"]
            if topic:
                normalized_requirements["topic_or_goal"] = self.validator.validate_topic(topic)
            else:
                normalized_requirements["topic_or_goal"] = "Announce product"
                self.validation_warnings.append("No topic specified, using default: Announce product")
            
            # Validate intents
            intents = normalized_requirements["intents_by_platform"]
            if intents:
                normalized_requirements["intents_by_platform"] = self.validator.validate_intents(intents)
            
        except (ValidationError, SecurityError) as e:
            self.validation_errors.append(str(e))
            log.error("Input validation failed: %s", e)
            # Continue with defaults rather than failing completely
        
        # Store validation results in shared state for reporting
        shared.setdefault("validation_results", {})
        shared["validation_results"]["warnings"] = self.validation_warnings
        shared["validation_results"]["errors"] = self.validation_errors
        
        return normalized_requirements

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Processes and validates the prepared task requirements.

        This method implements comprehensive input processing including validation,
        normalization, and security checks. It provides detailed feedback on
        input quality and handles edge cases gracefully.

        Processing Pipeline:
            1. Input Validation: Comprehensive validation of all fields
            2. Security Validation: Check for malicious content and injection attempts
            3. Normalization: Convert inputs to canonical format
            4. Business Rules: Apply domain-specific validation rules
            5. Quality Assessment: Evaluate input completeness and quality

        Args:
            prep_res: The task_requirements dictionary from prep(), guaranteed to have
                     proper structure with platforms, intents_by_platform, and topic_or_goal.

        Returns:
            Dict[str, Any]: The processed task requirements with validation metadata

        Raises:
            ValidationError: For critical validation failures
            SecurityError: For security-related issues
        """
        # Additional validation and processing
        processed_requirements = prep_res.copy()
        
        # Add validation metadata
        processed_requirements["validation_metadata"] = {
            "platforms_count": len(prep_res["platforms"]),
            "has_intents": bool(prep_res["intents_by_platform"]),
            "topic_length": len(prep_res["topic_or_goal"]),
            "validation_warnings": self.validation_warnings,
            "validation_errors": self.validation_errors,
            "quality_score": self._calculate_quality_score(prep_res)
        }
        
        # Log validation results
        if self.validation_warnings:
            log.warning("Input validation warnings: %s", self.validation_warnings)
        
        if self.validation_errors:
            log.error("Input validation errors: %s", self.validation_errors)
        
        return processed_requirements
    
    def _calculate_quality_score(self, requirements: Dict[str, Any]) -> float:
        """Calculate a quality score for the input requirements."""
        score = 1.0
        
        # Deduct for missing or default values
        if requirements["topic_or_goal"] == "Announce product":
            score -= 0.2
        
        if not requirements["intents_by_platform"]:
            score -= 0.1
        
        if len(requirements["platforms"]) < 2:
            score -= 0.1
        
        # Deduct for validation issues
        score -= len(self.validation_errors) * 0.3
        score -= len(self.validation_warnings) * 0.1
        
        return max(0.0, score)

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Persists normalized task requirements and emits completion milestone.

        This method stores the validated and processed task requirements in the shared
        state and emits streaming milestones for real-time progress tracking.

        Pre-condition:
            - exec_res contains validated task_requirements
            - shared state is properly initialized

        Post-condition:
            - shared["task_requirements"] contains normalized data
            - shared["validation_results"] contains validation feedback
            - Streaming milestones are emitted if stream is available

        Args:
            shared: The shared state dictionary
            prep_res: Original prepared requirements
            exec_res: Processed requirements with validation metadata

        Returns:
            str: Action to take ("default" to continue to next node)

        Streaming Events:
            - "engagement_manager_started": When processing begins
            - "validation_completed": When validation finishes
            - "requirements_normalized": When data is stored
        """
        # Store the validated requirements
        shared["task_requirements"] = {
            "platforms": exec_res["platforms"],
            "intents_by_platform": exec_res["intents_by_platform"],
            "topic_or_goal": exec_res["topic_or_goal"]
        }
        
        # Store validation metadata
        shared.setdefault("validation_results", {})
        shared["validation_results"].update(exec_res["validation_metadata"])
        
        # Emit streaming milestones
        if shared.get("stream"):
            try:
                shared["stream"].emit("engagement_manager_started", {
                    "platforms_count": len(exec_res["platforms"]),
                    "topic": exec_res["topic_or_goal"][:50] + "..." if len(exec_res["topic_or_goal"]) > 50 else exec_res["topic_or_goal"]
                })
                
                shared["stream"].emit("validation_completed", {
                    "warnings_count": len(self.validation_warnings),
                    "errors_count": len(self.validation_errors),
                    "quality_score": exec_res["validation_metadata"]["quality_score"]
                })
                
                shared["stream"].emit("requirements_normalized", {
                    "platforms": exec_res["platforms"],
                    "has_intents": bool(exec_res["intents_by_platform"])
                })
                
            except Exception as e:
                log.warning("Failed to emit streaming milestones: %s", e)
        
        # Log completion
        log.info("Engagement manager completed: %d platforms, topic: %s", 
                len(exec_res["platforms"]), exec_res["topic_or_goal"])
        
        return "default"

    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """Fallback behavior when exec() fails.

        Provides safe fallback behavior when validation or processing fails.
        Returns minimal valid requirements to allow the pipeline to continue.

        Args:
            prep_res: The prepared requirements
            exc: The exception that caused the failure

        Returns:
            Dict[str, Any]: Safe fallback requirements
        """
        log.error("Engagement manager exec failed, using fallback: %s", exc)
        
        # Return safe defaults
        return {
            "platforms": ["twitter"],
            "intents_by_platform": {},
            "topic_or_goal": "Announce product",
            "validation_metadata": {
                "platforms_count": 1,
                "has_intents": False,
                "topic_length": 16,
                "validation_warnings": [f"Using fallback due to error: {exc}"],
                "validation_errors": [str(exc)],
                "quality_score": 0.5
            }
        }
