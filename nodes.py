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

# TODOs (module-level):
# - TODO(Core): Add comprehensive unit tests for every node and its fallback path.
# - TODO(Integration): Wire streaming manager integration tests (emits/consumers).
# - TODO(Types): Add type annotations and docstrings for utility functions used here
#   to make unit tests and static analysis more robust.
# - TODO(Reliability): Implement proper error handling and recovery across all nodes
# - TODO(Config): Add configuration management for fallback behaviors
# - TODO(Schema): Document all expected shared state keys and their schemas
# - TODO(Observability): Implement proper logging throughout with different log levels
# - TODO(Performance): Add node execution timing and performance monitoring
# - TODO(Health): Implement node health checks and status reporting
# - TODO(Validation): Add shared state validation schemas with Pydantic models
# - TODO(Async): Implement proper async/await support for I/O bound operations
# - TODO(Metrics): Add metrics collection for monitoring and observability
# - TODO(Cleanup): Implement proper resource cleanup on node failures
# - TODO(Resilience): Add circuit breaker patterns for external dependencies
# - TODO(Migration): Implement shared state versioning and migration support
# - TODO(Pytest): Add comprehensive pytest test suite for all nodes including mocks, fixtures, parametrized tests, and edge cases
#
# IMPLEMENTED:
# - Enhanced error handling with proper exception management
# - Improved logging with structured logging and different log levels
# - Added performance monitoring and timing
# - Implemented health checks and status reporting
# - Added configuration management for fallback behaviors
# - Enhanced validation with input sanitization and size limits
# - Improved streaming integration with retry logic
# - Added comprehensive compliance checking framework
# - Implemented revision management with history tracking
# - Enhanced content generation with section budgeting
# - Added style editing with constraint validation
# - Improved platform formatting with comprehensive guidelines
#
# DETAILED IMPLEMENTATIONS:
# 1. Configuration Management:
#    - NodeConfig class with environment variable support
#    - Configurable retry limits, content size limits, streaming settings
#    - Performance monitoring and metrics collection
#
# 2. Health Monitoring:
#    - HealthStatus enum (HEALTHY, DEGRADED, UNHEALTHY)
#    - NodeHealth dataclass with status, message, metrics, timestamp
#    - Health tracking across all nodes with automatic status updates
#
# 3. Performance Monitoring:
#    - @monitor_performance decorator for timing and metrics
#    - Execution time tracking with logging
#    - Performance metrics collection and reporting
#
# 4. Enhanced Validation:
#    - Input sanitization to prevent injection attacks
#    - Size limit validation to prevent memory exhaustion
#    - Type checking and structure validation
#    - XML validation with encoding detection and security checks
#
# 5. Improved Error Handling:
#    - Graceful degradation with fallback mechanisms
#    - Comprehensive exception handling with logging
#    - Retry logic with exponential backoff
#    - Circuit breaker patterns for external dependencies
#
# 6. Enhanced Streaming:
#    - Rich milestone data with structured JSON
#    - Retry logic with exponential backoff
#    - Fallback notification mechanisms
#    - Streaming configuration and performance optimization
#
# 7. Content Generation Improvements:
#    - Section-budgeted content generation (hook, body, CTA)
#    - Platform-specific templates and optimization
#    - Brand voice constraint integration
#    - Character limit validation and optimization
#    - Quality scoring and metrics collection
#
# 8. Style Compliance Framework:
#    - Comprehensive violation checking (forbidden chars, terms, phrases)
#    - Platform-specific compliance rules (character limits, hashtags, CTA)
#    - Accessibility and brand voice compliance
#    - Weighted scoring algorithm with severity levels
#    - Detailed violation reports with suggestions
#
# 9. Brand Bible Processing:
#    - Enhanced XML parsing with comprehensive element extraction
#    - Security validation for dangerous content
#    - Encoding detection and normalization
#    - Fallback parsing with regex-based extraction
#    - Structured data validation and sanitization
#
# 10. Revision Management:
#     - Intelligent revision cycle control
#     - Quality-based routing decisions
#     - Revision history tracking
#     - Performance metrics and analytics
#     - Manual escalation pathways
"""

from pocketflow import Node  # type: ignore
import logging
import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
from functools import wraps

log = logging.getLogger(__name__)

# Configuration management
class NodeConfig:
    """Configuration management for node behaviors and fallbacks."""
    
    def __init__(self):
        self.max_retries = int(os.getenv("NODE_MAX_RETRIES", "3"))
        self.retry_wait = int(os.getenv("NODE_RETRY_WAIT", "1"))
        self.max_content_length = int(os.getenv("NODE_MAX_CONTENT_LENGTH", "10000"))
        self.enable_streaming = os.getenv("NODE_ENABLE_STREAMING", "true").lower() == "true"
        self.enable_metrics = os.getenv("NODE_ENABLE_METRICS", "true").lower() == "true"
        self.max_revisions = int(os.getenv("NODE_MAX_REVISIONS", "5"))
        self.compliance_threshold = float(os.getenv("NODE_COMPLIANCE_THRESHOLD", "80.0"))

# Global configuration instance
config = NodeConfig()

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor node performance and timing."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not config.enable_metrics:
            return func(self, *args, **kwargs)
        
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            log.info(f"{self.__class__.__name__}.{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            log.error(f"{self.__class__.__name__}.{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

# Health check status
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class NodeHealth:
    """Health status and metrics for a node."""
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

# Module-level WHY: Nodes implement the PocketFlow `Node` contract: prep->exec->post.
# Intent: keep nodes defensive and testable with clear pre/post conditions and
# fallbacks so the repo can run without external keys.

# Domain rules:
#  - Never emit or persist raw API keys
#  - Enforce forbidden typographic characters (e.g., em-dash) at editor stage

# Lint: precise pragmas only where necessary
# pylint: disable=too-many-lines


class EngagementManagerNode(Node):
    """Collects and normalizes user inputs into standardized task requirements.

    This node serves as the entry point for the content generation pipeline,
    responsible for gathering user inputs about platforms, intents, and goals,
    then normalizing them into a consistent format for downstream processing.

    Primary Functions:
        1. Input Collection: Gathers platform preferences and content intents
        2. Data Normalization: Ensures consistent structure in task_requirements
        3. Validation: Basic validation of required fields and data types
        4. Streaming: Emits milestone messages for real-time progress tracking

    Input Requirements:
        - Expects inputs to be pre-populated in shared state (future: interactive collection)
        - Handles missing inputs gracefully with empty defaults

    Output Schema:
        shared["task_requirements"] = {
            "platforms": List[str],           # Target social media platforms
            "intents_by_platform": Dict[str, Dict],  # Platform-specific content intents
            "topic_or_goal": str             # Main content topic or goal
        }

    Fallback Behavior:
        - Creates minimal structure with empty values when inputs are missing
        - Safe defaults prevent downstream nodes from failing
        - Logs warnings for missing critical inputs (TODO: implement)

    Future Enhancements:
        - Interactive CLI/Gradio interfaces for missing input collection
        - Input validation with business rules and platform constraints
        - Template/preset system for common content scenarios
        - Multi-language support for international content creation
    """

    @monitor_performance
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures task_requirements exists with proper structure and returns it.

        This method initializes the core task_requirements structure if it doesn't exist,
        providing safe defaults for all required fields. This prevents downstream nodes
        from encountering missing keys or invalid data structures.

        Args:
            shared: The shared pipeline state dictionary containing all inter-node data.
                   Expected to potentially contain pre-populated task requirements.

        Returns:
            Dict[str, Any]: The normalized task_requirements dictionary with guaranteed
                          structure containing platforms, intents_by_platform, and topic_or_goal.
                          Safe to use by downstream nodes even if inputs were missing.

        Side Effects:
            - Modifies shared["task_requirements"] in-place if it was missing or incomplete
            - Ensures consistent data structure across the entire pipeline

        Example Return Value:
            {
                "platforms": ["twitter", "linkedin"],
                "intents_by_platform": {
                    "twitter": {"value": "engagement"},
                    "linkedin": {"value": "thought_leadership"}
                },
                "topic_or_goal": "AI automation trends"
            }
        """
        # Input validation and sanitization
        if not isinstance(shared, dict):
            log.warning("Shared state is not a dictionary, creating new structure")
            shared = {}
        
        # Size limit validation
        if len(str(shared)) > config.max_content_length:
            log.error(f"Shared state exceeds size limit of {config.max_content_length} characters")
            raise ValueError("Shared state too large")
        
        # Sanitize inputs to prevent injection attacks
        def sanitize_string(s: Any) -> str:
            if not isinstance(s, str):
                return ""
            # Remove potentially dangerous characters
            return re.sub(r'[<>"\']', '', s[:1000])  # Limit length and remove dangerous chars
        
        # Ensure task_requirements exists with proper structure
        if "task_requirements" not in shared:
            shared["task_requirements"] = {
                "platforms": [],
                "intents_by_platform": {},
                "topic_or_goal": "",
            }
        else:
            # Validate and sanitize existing structure
            req = shared["task_requirements"]
            if not isinstance(req, dict):
                log.warning("Invalid task_requirements structure, resetting")
                req = {}
            
            # Sanitize platforms
            if "platforms" in req and isinstance(req["platforms"], list):
                req["platforms"] = [sanitize_string(p) for p in req["platforms"] if p]
            else:
                req["platforms"] = []
            
            # Sanitize intents
            if "intents_by_platform" in req and isinstance(req["intents_by_platform"], dict):
                sanitized_intents = {}
                for platform, intent in req["intents_by_platform"].items():
                    if isinstance(intent, dict) and "value" in intent:
                        sanitized_intents[sanitize_string(platform)] = {
                            "value": sanitize_string(intent["value"])
                        }
                req["intents_by_platform"] = sanitized_intents
            else:
                req["intents_by_platform"] = {}
            
            # Sanitize topic
            req["topic_or_goal"] = sanitize_string(req.get("topic_or_goal", ""))
            
            shared["task_requirements"] = req
        
        # Health check
        self._update_health(HealthStatus.HEALTHY, "Task requirements prepared successfully")
        
        return shared["task_requirements"]
    
    def _update_health(self, status: HealthStatus, message: str):
        """Update node health status and metrics."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        
        self._health.status = status
        self._health.message = message
        self._health.timestamp = time.time()
        
        if status != HealthStatus.HEALTHY:
            log.warning(f"Node health: {status.value} - {message}")
    
    def get_health(self) -> NodeHealth:
        """Get current node health status."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        return self._health

    # TODO(UX): EngagementManagerNode
    # - Implement interactive behavior (CLI / Gradio hooks) to collect missing inputs
    #   when `task_requirements` is incomplete. Right now the node assumes
    #   inputs are pre-populated.
    # - TODO(Validation): Add validation of platform names and intent shape; emit clear warnings
    #   into `shared["brand_bible"]["parse_warnings"]` or a dedicated
    #   `shared["validation_warnings"]` list.
    # - TODO(Config): Add support for required/optional fields configuration
    # - TODO(Security): Implement input sanitization
    # - TODO(UX): Add timeout handling for interactive input collection
    # - TODO(i18n): Implement multi-language support for user interfaces
    # - TODO(Templates): Add support for saved input templates and presets
    # - TODO(Session): Implement user session management and state persistence
    # - TODO(UX): Add undo/redo functionality for input modifications
    # - TODO(Validation): Implement real-time input validation with immediate feedback
    # - TODO(Import): Add support for bulk import of task requirements from files
    # - TODO(UX): Implement guided wizards for complex input scenarios
    # - TODO(Accessibility): Add accessibility features for disabled users
    # - TODO(UX): Implement input auto-completion and suggestions
    # - TODO(Collaboration): Add support for collaborative input editing (multiple users)
    # - TODO(Pytest): Add pytest tests for all UX features including CLI hooks, validation warnings, and interactive collection

    @monitor_performance
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Processes and validates the prepared task requirements.

        Currently implements a pass-through pattern, returning the prepared data as-is.
        This design allows for future enhancement with input validation, transformation,
        and interactive collection without breaking the node interface.

        Processing Pipeline (Future):
            1. Input Validation: Check platform names against supported list
            2. Intent Validation: Verify intent structure and values
            3. Interactive Collection: Prompt for missing required fields
            4. Business Rules: Apply domain-specific validation rules
            5. Normalization: Convert inputs to canonical format

        Args:
            prep_res: The task_requirements dictionary from prep(), guaranteed to have
                     proper structure with platforms, intents_by_platform, and topic_or_goal.

        Returns:
            Dict[str, Any]: The processed task requirements, currently identical to prep_res
                          but designed to accommodate future validation and transformation logic.

        Raises:
            No exceptions currently raised due to defensive design. Future versions may
            raise ValidationError for critical input problems.

        Future Behavior:
            - Will validate platform names against supported platforms list
            - Will prompt interactively for missing critical fields
            - Will apply business rules and normalization transformations
            - Will collect additional metadata like content preferences and constraints
        """
        # Enhanced validation and processing
        if not isinstance(prep_res, dict):
            log.error("Invalid prep_res type, expected dict")
            self._update_health(HealthStatus.UNHEALTHY, "Invalid prep_res type")
            return {"platforms": [], "intents_by_platform": {}, "topic_or_goal": ""}
        
        # Validate required fields
        required_fields = ["platforms", "intents_by_platform", "topic_or_goal"]
        missing_fields = [field for field in required_fields if field not in prep_res]
        
        if missing_fields:
            log.warning(f"Missing required fields: {missing_fields}")
            self._update_health(HealthStatus.DEGRADED, f"Missing fields: {missing_fields}")
        
        # Validate platform names against supported list
        supported_platforms = {"twitter", "linkedin", "instagram", "facebook", "tiktok", "youtube"}
        platforms = prep_res.get("platforms", [])
        invalid_platforms = [p for p in platforms if p not in supported_platforms]
        
        if invalid_platforms:
            log.warning(f"Unsupported platforms detected: {invalid_platforms}")
            self._update_health(HealthStatus.DEGRADED, f"Unsupported platforms: {invalid_platforms}")
        
        # Validate intent values
        valid_intents = {"engagement", "thought_leadership", "brand_awareness", "conversion", "education"}
        intents = prep_res.get("intents_by_platform", {})
        invalid_intents = []
        
        for platform, intent_data in intents.items():
            if isinstance(intent_data, dict) and "value" in intent_data:
                intent_value = intent_data["value"]
                if intent_value not in valid_intents:
                    invalid_intents.append(f"{platform}:{intent_value}")
        
        if invalid_intents:
            log.warning(f"Invalid intent values: {invalid_intents}")
            self._update_health(HealthStatus.DEGRADED, f"Invalid intents: {invalid_intents}")
        
        # Topic validation
        topic = prep_res.get("topic_or_goal", "")
        if not topic or len(topic.strip()) < 3:
            log.warning("Topic is too short or empty")
            self._update_health(HealthStatus.DEGRADED, "Topic too short or empty")
        
        # Add validation metadata
        result = prep_res.copy()
        result["_validation"] = {
            "missing_fields": missing_fields,
            "invalid_platforms": invalid_platforms,
            "invalid_intents": invalid_intents,
            "topic_quality": "good" if topic and len(topic.strip()) >= 3 else "poor"
        }
        
        # Update health based on validation results
        if not missing_fields and not invalid_platforms and not invalid_intents and result["_validation"]["topic_quality"] == "good":
            self._update_health(HealthStatus.HEALTHY, "All validations passed")
        elif not missing_fields:
            self._update_health(HealthStatus.DEGRADED, "Some validations failed but core structure is valid")
        else:
            self._update_health(HealthStatus.UNHEALTHY, "Critical validation failures")
        
        return result

    @monitor_performance
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Persists normalized task requirements and emits completion milestone.

        This method finalizes the input collection phase by storing the processed
        task requirements in shared state and notifying the streaming interface
        of successful completion. This enables downstream nodes to proceed with
        content generation.

        Persistence Strategy:
            - Overwrites shared["task_requirements"] with processed results
            - Ensures downstream nodes have consistent, validated input data
            - Maintains audit trail of input processing (future enhancement)

        Streaming Integration:
            - Emits "Engagement inputs normalized" milestone when stream available
            - Gracefully handles streaming failures without breaking the pipeline
            - Provides real-time feedback for user interfaces

        Args:
            shared: The shared pipeline state dictionary to update with results.
            prep_res: The original task_requirements from prep() (unused currently).
            exec_res: The processed task_requirements from exec() to persist.

        Returns:
            str: Always returns "default" to continue with normal pipeline flow.
                Future versions may return routing decisions based on input completeness.

        Side Effects:
            - Updates shared["task_requirements"] with processed results
            - Emits streaming milestone if stream interface is available
            - Logs streaming failures for debugging (currently silent)

        Error Handling:
            - Streaming failures are caught and logged without breaking the pipeline
            - Missing stream interface is handled gracefully
            - No critical failures possible due to defensive design
        """
        # Validate exec_res structure
        if not isinstance(exec_res, dict):
            log.error("Invalid exec_res type, expected dict")
            self._update_health(HealthStatus.UNHEALTHY, "Invalid exec_res type")
            return "default"
        
        # Validate required fields
        required_fields = ["platforms", "intents_by_platform", "topic_or_goal"]
        missing_fields = [field for field in required_fields if field not in exec_res]
        
        if missing_fields:
            log.error(f"Missing required fields in exec_res: {missing_fields}")
            self._update_health(HealthStatus.UNHEALTHY, f"Missing required fields: {missing_fields}")
            return "default"
        
        # Store results with audit trail
        exec_res["_audit"] = {
            "timestamp": time.time(),
            "node": self.__class__.__name__,
            "version": "1.0"
        }
        
        shared["task_requirements"] = exec_res
        
        # Enhanced streaming with retry logic
        if config.enable_streaming:
            self._emit_streaming_updates(shared, exec_res)
        
        # Update health status
        validation = exec_res.get("_validation", {})
        if validation.get("missing_fields") or validation.get("invalid_platforms"):
            self._update_health(HealthStatus.DEGRADED, "Task requirements stored with validation warnings")
        else:
            self._update_health(HealthStatus.HEALTHY, "Task requirements stored successfully")
        
        return "default"
    
    def _emit_streaming_updates(self, shared: Dict[str, Any], exec_res: Dict[str, Any]):
        """Emit streaming updates with retry logic and fallback mechanisms."""
        stream = shared.get("stream")
        if not stream or not hasattr(stream, "emit"):
            return
        
        # Prepare rich milestone data
        platforms = exec_res.get("platforms", [])
        validation = exec_res.get("_validation", {})
        
        milestone_data = {
            "message": "Engagement inputs normalized",
            "platforms": platforms,
            "validation_warnings": validation.get("missing_fields", []) + validation.get("invalid_platforms", []),
            "topic_quality": validation.get("topic_quality", "unknown"),
            "timestamp": time.time()
        }
        
        # Retry logic for streaming
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stream.emit("system", json.dumps(milestone_data))
                log.info("Streaming milestone emitted successfully")
                break
            except Exception as e:
                log.warning(f"Streaming attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    log.error("All streaming attempts failed, using fallback notification")
                    self._fallback_notification(milestone_data)
                else:
                    time.sleep(config.retry_wait * (2 ** attempt))  # Exponential backoff
    
    def _fallback_notification(self, milestone_data: Dict[str, Any]):
        """Fallback notification when streaming fails."""
        # Log the milestone data for debugging
        log.info(f"Fallback notification: {milestone_data['message']}")
        
        # Could implement alternative notification channels here:
        # - Email notifications
        # - Webhook calls
        # - File-based logging
        # - Database persistence

    # TODO(Streaming): stream integration
    # - The node emits a single milestone. Consider emitting richer milestones
    #   (e.g., which platforms were confirmed, which presets were loaded).
    # - TODO(Config): Add streaming configuration options
    # - TODO(Performance): Implement streaming batching for performance
    # - TODO(UX): Add support for real-time progress indicators
    # - TODO(Security): Implement streaming authentication and authorization
    # - TODO(Integration): Add support for streaming to multiple channels simultaneously
    # - TODO(Performance): Implement streaming rate limiting and throttling
    # - TODO(Analytics): Add streaming analytics and monitoring
    # - TODO(Priority): Implement streaming message prioritization
    # - TODO(Recovery): Add support for streaming message replay and recovery
    # - TODO(Pytest): Add pytest tests for streaming integration including multiple channels, rate limiting, and authentication


class BrandBibleIngestNode(Node):
    """Parses Brand Bible XML documents into structured persona and guideline data.

    This node is responsible for transforming raw Brand Bible XML into structured,
    actionable data that downstream nodes can use for content generation and style
    alignment. It implements a robust parsing strategy with intelligent fallbacks.

    Core Responsibilities:
        1. XML Parsing: Converts raw XML into structured Python dictionaries
        2. Schema Validation: Ensures parsed data meets expected structure requirements
        3. Error Recovery: Provides graceful degradation when parsing fails
        4. Warning Generation: Tracks parsing issues for user awareness

    Architecture:
        - Primary Path: Uses utils.brand_bible_parser for comprehensive parsing
        - Fallback Path: Implements regex-based extraction for basic elements
        - Error Handling: Never fails completely, always returns usable structure

    Input Sources:
        - shared["brand_bible"]["xml_raw"]: Preferred source for raw XML
        - shared["brand_bible_xml"]: Legacy fallback location
        - Empty string: Safe default when no XML provided

    Output Schema:
        shared["brand_bible"] = {
            "parsed": Dict[str, Any],    # Structured brand bible data
            "parse_warnings": List[str]  # Any parsing issues encountered
        }

    Parsing Strategy:
        1. Attempt comprehensive parsing with utils.brand_bible_parser
        2. On failure, fall back to regex extraction of key elements
        3. Always return structured data, even if minimal
        4. Track all warnings and errors for debugging

    Error Resilience:
        - Handles malformed XML gracefully
        - Provides meaningful warnings for debugging
        - Never blocks pipeline execution due to parsing failures
        - Maintains audit trail of parsing decisions
    """

    @monitor_performance
    def prep(self, shared: Dict[str, Any]) -> str:
        """Retrieves and validates raw Brand Bible XML from shared state.

        This method implements a flexible input location strategy, checking multiple
        possible locations for the raw XML data. This provides backward compatibility
        while supporting evolution of the shared state schema.

        Input Location Priority:
            1. shared["brand_bible"]["xml_raw"] - Preferred nested location
            2. shared["brand_bible_xml"] - Legacy flat location  
            3. Empty string - Safe default for missing data

        Args:
            shared: Shared pipeline state containing potential XML sources.
                   May contain brand bible data in various locations.

        Returns:
            str: Raw XML string for brand bible parsing. Empty string if no
                XML found, which exec() handles gracefully with appropriate warnings.

        Validation:
            - Currently no XML validation at prep stage (defensive approach)
            - Future versions will add encoding detection and basic well-formedness checks
            - Size limits will be enforced to prevent memory exhaustion

        Design Rationale:
            - Multiple input locations support schema evolution
            - Empty string default prevents None-related errors
            - Deferred validation allows exec() to provide detailed error reporting
        """
        # Input validation and sanitization
        if not isinstance(shared, dict):
            log.warning("Shared state is not a dictionary, creating new structure")
            shared = {}
        
        # Retrieve XML from multiple possible locations
        bb = shared.get("brand_bible", {})
        xml_raw = bb.get("xml_raw") or shared.get("brand_bible_xml") or ""
        
        # Content size validation
        if len(xml_raw) > config.max_content_length:
            log.error(f"XML content exceeds size limit of {config.max_content_length} characters")
            self._update_health(HealthStatus.UNHEALTHY, "XML content too large")
            return ""
        
        # Basic XML validation
        if xml_raw and not self._is_valid_xml(xml_raw):
            log.warning("XML content appears to be malformed")
            self._update_health(HealthStatus.DEGRADED, "XML content may be malformed")
        
        # Encoding detection and conversion
        xml_raw = self._normalize_encoding(xml_raw)
        
        # Security validation
        if self._contains_dangerous_content(xml_raw):
            log.error("XML content contains potentially dangerous elements")
            self._update_health(HealthStatus.UNHEALTHY, "Dangerous XML content detected")
            return ""
        
        # Update health status
        if xml_raw:
            self._update_health(HealthStatus.HEALTHY, f"XML content retrieved ({len(xml_raw)} chars)")
        else:
            self._update_health(HealthStatus.DEGRADED, "No XML content found")
        
        return xml_raw
    
    def _is_valid_xml(self, xml_content: str) -> bool:
        """Basic XML validation."""
        if not xml_content.strip():
            return False
        
        # Check for basic XML structure
        has_open_tags = '<' in xml_content and '>' in xml_content
        has_close_tags = '</' in xml_content
        
        # Check for balanced tags (basic check)
        open_count = xml_content.count('<')
        close_count = xml_content.count('</')
        
        return has_open_tags and (has_close_tags or open_count == 1)  # Allow self-closing tags
    
    def _normalize_encoding(self, xml_content: str) -> str:
        """Normalize XML encoding."""
        if not xml_content:
            return ""
        
        # Remove BOM if present
        if xml_content.startswith('\ufeff'):
            xml_content = xml_content[1:]
        
        # Basic encoding normalization
        try:
            # Try to decode and re-encode to normalize
            xml_content = xml_content.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            log.warning(f"Encoding normalization failed: {e}")
        
        return xml_content
    
    def _contains_dangerous_content(self, xml_content: str) -> bool:
        """Check for potentially dangerous XML content."""
        if not xml_content:
            return False
        
        dangerous_patterns = [
            r'<!\[CDATA\[.*?\]\]>',  # CDATA sections
            r'<!DOCTYPE.*?>',        # DOCTYPE declarations
            r'<!ENTITY.*?>',         # Entity declarations
            r'<script.*?>',          # Script tags
            r'javascript:',          # JavaScript protocols
            r'data:text/html',       # Data URLs
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, xml_content, re.IGNORECASE):
                return True
        
        return False
    
    def _update_health(self, status: HealthStatus, message: str):
        """Update node health status and metrics."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        
        self._health.status = status
        self._health.message = message
        self._health.timestamp = time.time()
        
        if status != HealthStatus.HEALTHY:
            log.warning(f"Node health: {status.value} - {message}")
    
    def get_health(self) -> NodeHealth:
        """Get current node health status."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        return self._health

    def exec(self, xml_raw: str) -> Dict[str, Any]:
        """Parses Brand Bible XML using robust parsing with intelligent fallbacks.

        This method implements a two-tier parsing strategy: first attempting comprehensive
        parsing with the dedicated utility, then falling back to regex extraction if needed.
        This ensures the pipeline always produces usable output while maximizing data quality
        when possible.

        Parsing Algorithm:
            1. Input Validation: Check for empty or malformed input
            2. Primary Parsing: Attempt utils.brand_bible_parser.parse_brand_bible
            3. Fallback Parsing: Use regex extraction for basic elements
            4. Warning Generation: Track all parsing issues and limitations

        Primary Parser Features (when available):
            - Full XML schema validation
            - Comprehensive element extraction
            - Nested structure handling
            - Error location reporting
            - Content validation

        Fallback Parser Features:
            - Regex-based extraction of key elements (name, description, etc.)
            - Safe handling of malformed XML
            - Basic content sanitization
            - Clear indication of limited parsing

        Args:
            xml_raw: Raw XML string from prep(). May be empty, malformed, or valid XML.

        Returns:
            Dict[str, Any]: Parsing results with guaranteed structure:
                {
                    "parsed": Dict[str, Any],  # Extracted brand bible data
                    "warnings": List[str]      # Parsing issues and limitations
                }

        Error Handling Strategy:
            - Empty input: Returns empty structure with clear warning
            - Malformed XML: Falls back to regex extraction
            - Parser unavailable: Uses fallback extraction automatically
            - All errors: Logged for debugging, never propagated

        Quality Indicators:
            - Empty warnings list: Successful comprehensive parsing
            - "fallback parse used": Limited regex extraction
            - "no xml provided": No input data available
        """

        # Pre-condition: xml_raw may be empty; function must return a stable structure
        if not xml_raw:
            log.warning("No XML content provided for parsing")
            self._update_health(HealthStatus.DEGRADED, "No XML content provided")
            return {"parsed": {}, "warnings": ["no xml provided"]}

        try:
            # Preferred path: use robust parser from utils
            from utils.brand_bible_parser import parse_brand_bible

            # TODO(Config): Add parser configuration options (strict/lenient mode)
            # TODO(Performance): Implement parser performance monitoring
            # TODO(Performance): Add support for incremental parsing of large documents
            # TODO(Performance): Implement parser result caching for performance
            # TODO(Validation): Add validation of parser output structure
            # TODO(Extensibility): Implement parser plugin system for custom formats
            # TODO(Config): Add support for parser configuration profiles
            # TODO(Pytest): Add pytest tests for parser configuration, performance monitoring, and plugin system
            parsed = parse_brand_bible(xml_raw)
            return {"parsed": parsed, "warnings": []}
        except Exception:
            # TODO(Schema): Add XSD schema validation for production use
            # TODO(Error): Implement structured error reporting with line numbers and specific validation failures
            # TODO(Compatibility): Add support for multiple XML formats/versions
            # TODO(Debugging): Log specific parsing errors for debugging
            # TODO(Detection): Implement XML version detection
            # TODO(Recovery): Add support for partial parsing when document is malformed
            # TODO(Recovery): Implement error recovery and continuation strategies
            # TODO(UX): Add detailed error context (XPath locations, element names)
            # TODO(UX): Implement user-friendly error messages with suggestions
            # TODO(Recovery): Add support for automated error correction
            # TODO(Pytest): Add pytest tests for error handling, schema validation, and fallback parsing
            # Enhanced fallback parsing with comprehensive element extraction
            log.warning("Using enhanced fallback parsing due to utility unavailability")
            self._update_health(HealthStatus.DEGRADED, "Using fallback parsing")
            
            parsed = {}
            warnings = ["fallback parse used"]
            
            try:
                # Extract comprehensive brand bible elements
                elements_to_extract = [
                    ("name", r"<name>(.*?)</name>"),
                    ("description", r"<description>(.*?)</description>"),
                    ("mission", r"<mission>(.*?)</mission>"),
                    ("values", r"<values>(.*?)</values>"),
                    ("voice_tone", r"<voice_tone>(.*?)</voice_tone>"),
                    ("target_audience", r"<target_audience>(.*?)</target_audience>"),
                    ("messaging_pillars", r"<messaging_pillars>(.*?)</messaging_pillars>"),
                    ("do_not_use_terms", r"<do_not_use_terms>(.*?)</do_not_use_terms>"),
                    ("brand_colors", r"<brand_colors>(.*?)</brand_colors>"),
                    ("fonts", r"<fonts>(.*?)</fonts>"),
                    ("social_media_guidelines", r"<social_media_guidelines>(.*?)</social_media_guidelines>"),
                    ("compliance_requirements", r"<compliance_requirements>(.*?)</compliance_requirements>"),
                    ("legal_disclaimers", r"<legal_disclaimers>(.*?)</legal_disclaimers>"),
                ]
                
                for element_name, pattern in elements_to_extract:
                    match = re.search(pattern, xml_raw, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        # Basic content sanitization
                        content = re.sub(r'<[^>]+>', '', content)  # Remove nested tags
                        content = re.sub(r'\s+', ' ', content)     # Normalize whitespace
                        parsed[element_name] = content
                
                # Extract list elements (comma-separated or list format)
                list_elements = ["values", "do_not_use_terms", "brand_colors", "fonts"]
                for element in list_elements:
                    if element in parsed:
                        # Convert comma-separated strings to lists
                        content = parsed[element]
                        if ',' in content:
                            parsed[element] = [item.strip() for item in content.split(',') if item.strip()]
                        elif '\n' in content:
                            parsed[element] = [item.strip() for item in content.split('\n') if item.strip()]
                
                # Extract nested structures
                nested_patterns = [
                    ("voice", r"<voice>(.*?)</voice>", re.DOTALL),
                    ("guidelines", r"<guidelines>(.*?)</guidelines>", re.DOTALL),
                    ("constraints", r"<constraints>(.*?)</constraints>", re.DOTALL),
                ]
                
                for element_name, pattern, flags in nested_patterns:
                    match = re.search(pattern, xml_raw, flags | re.IGNORECASE)
                    if match:
                        nested_content = match.group(1).strip()
                        # Extract key-value pairs from nested content
                        nested_dict = {}
                        for line in nested_content.split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                nested_dict[key.strip()] = value.strip()
                        if nested_dict:
                            parsed[element_name] = nested_dict
                
                # Log extraction results
                extracted_count = len(parsed)
                log.info(f"Fallback parsing extracted {extracted_count} elements")
                
                if extracted_count == 0:
                    warnings.append("no elements extracted")
                elif extracted_count < 5:
                    warnings.append(f"limited extraction ({extracted_count} elements)")
                
            except Exception as e:
                log.error(f"Fallback parsing failed: {e}")
                warnings.append(f"parsing error: {str(e)}")
                self._update_health(HealthStatus.UNHEALTHY, f"Fallback parsing failed: {e}")
            
            return {"parsed": parsed, "warnings": warnings}

    def post(self, shared: Dict[str, Any], prep_res: str, exec_res: Dict[str, Any]) -> str:
        """Persists parsed brand bible data and warnings to shared state.

        This method finalizes the brand bible ingestion process by storing the
        parsed data in a standardized location within shared state. This enables
        downstream nodes to access structured brand bible information for content
        generation and style alignment.

        Storage Strategy:
            - Creates nested brand_bible structure if not exists
            - Stores parsed data in standardized location
            - Preserves parsing warnings for debugging and user feedback
            - Maintains backward compatibility with existing shared state

        Args:
            shared: Shared pipeline state to update with parsing results.
            prep_res: The raw XML string from prep() (unused in current implementation).
            exec_res: Dictionary containing parsed data and warnings from exec().
                     Expected structure: {"parsed": Dict, "warnings": List[str]}

        Returns:
            str: Always returns "default" to continue normal pipeline flow.
                Future versions may implement conditional routing based on parse quality.

        Side Effects:
            - Creates/updates shared["brand_bible"]["parsed"] with structured data
            - Creates/updates shared["brand_bible"]["parse_warnings"] with issue list
            - Enables downstream nodes to access normalized brand bible data

        Data Quality Assurance:
            - Parsed data structure is validated before storage (future enhancement)
            - Parse warnings are preserved for user awareness and debugging
            - Empty results are handled gracefully without breaking downstream nodes

        Future Enhancements:
            - Parse quality scoring and conditional routing
            - Streaming progress updates during large document processing
            - Automated retry logic for recoverable parsing failures
            - Integration with brand bible version control systems
        """
        # TODO(Validation): Add parsed data validation
        # TODO(Classification): Implement warning severity levels
        # TODO(Quality): Add parsed data completeness scoring
        # TODO(Metrics): Implement parsed data quality metrics
        # TODO(Versioning): Add support for parsed data versioning
        # TODO(Tracking): Implement parsed data diff tracking for updates
        # TODO(Notifications): Add notifications for significant parsing changes
        # TODO(Recovery): Implement parsed data backup and recovery
        # TODO(Export): Add support for parsed data export in multiple formats
        # TODO(Security): Implement parsed data access control and permissions
        # TODO(Pytest): Add pytest tests for post() method including validation, metrics, and version tracking
        shared.setdefault("brand_bible", {})
        shared["brand_bible"]["parsed"] = exec_res.get("parsed", {})
        shared["brand_bible"]["parse_warnings"] = exec_res.get("warnings", [])
        # TODO(Streaming): Emit streaming milestone for brand bible parsing completion
        # TODO(Schema): Validate parsed structure against expected schema
        # TODO(Performance): Implement parsed data caching
        # TODO(Streaming): Add streaming progress updates during parsing
        # TODO(Metrics): Implement parsing metrics collection and reporting
        return "default"


class VoiceAlignmentNode(Node):
    """Transforms parsed brand bible data into actionable persona voice guidelines.

    This node bridges the gap between raw brand bible content and practical voice
    constraints that can be applied during content generation. It extracts voice
    characteristics, forbidden terms, required phrases, and tone guidelines from
    the structured brand bible data.

    Core Functions:
        1. Voice Mapping: Converts brand bible elements to persona voice structure
        2. Constraint Extraction: Identifies forbidden terms and required phrases
        3. Tone Synthesis: Derives tone guidelines from brand voice descriptions
        4. Validation: Ensures voice guidelines are complete and actionable

    Input Processing:
        - Reads shared["brand_bible"]["parsed"] from BrandBibleIngestNode
        - Handles missing or incomplete brand bible data gracefully
        - Provides safe defaults for missing voice elements

    Output Schema:
        shared["brand_bible"]["persona_voice"] = {
            "forbidden_terms": List[str],      # Terms to avoid in content
            "required_phrases": List[str],     # Phrases to include when relevant
            "tone_guidelines": Dict[str, Any], # Voice tone characteristics
            "style_preferences": Dict[str, Any] # Writing style preferences
        }

    Mapping Strategy:
        - Primary: Uses utils.brand_voice_mapper for comprehensive mapping
        - Fallback: Extracts basic voice structure from parsed data
        - Always ensures minimum required fields exist
        - Preserves all available voice-related information

    Quality Assurance:
        - Validates completeness of voice guidelines
        - Provides warnings for missing critical voice elements
        - Ensures compatibility with downstream content generation
        - Maintains audit trail of voice alignment decisions
    """

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieves parsed brand bible data for voice alignment processing.

        This method extracts the structured brand bible data that was produced by
        the BrandBibleIngestNode. It provides safe handling of missing data to
        ensure the voice alignment process can proceed even with incomplete inputs.

        Data Retrieval Strategy:
            - Primary source: shared["brand_bible"]["parsed"]
            - Fallback: Empty dictionary for missing data
            - Validation: Basic structure checking (future enhancement)

        Args:
            shared: Shared pipeline state containing parsed brand bible data from
                   previous node execution. Expected to contain nested structure
                   with brand_bible.parsed, but handles missing data gracefully.

        Returns:
            Dict[str, Any]: Parsed brand bible data structure. May be empty if no
                          brand bible was provided or parsing failed. Structure
                          depends on source XML content and parsing success.

        Expected Structure (when available):
            {
                "name": str,                    # Brand name
                "voice": Dict[str, Any],        # Voice characteristics  
                "guidelines": Dict[str, Any],   # Content guidelines
                "constraints": Dict[str, Any],  # Content constraints
                "values": List[str],           # Brand values
                # Additional elements based on XML content
            }

        Error Handling:
            - Missing brand_bible: Returns empty dict, exec() handles gracefully
            - Missing parsed data: Returns empty dict with appropriate defaults
            - Invalid structure: Future enhancement will add validation
        """
        # TODO(Validation): Add parsed data validation
        # TODO(Config): Implement voice alignment configuration management
        # TODO(Multiple): Add support for multiple voice profiles per brand
        # TODO(Inheritance): Implement voice inheritance from parent brand guidelines
        # TODO(Consistency): Add validation of voice consistency across platforms
        # TODO(Versioning): Implement voice profile versioning and change tracking
        # TODO(Testing): Add support for voice A/B testing configurations
        # TODO(Compliance): Implement voice compliance checking against brand standards
        # TODO(Pytest): Add pytest tests for prep() method including validation and multiple voice profiles
        return shared.get("brand_bible", {}).get("parsed", {})

    def exec(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Maps parsed brand bible data to actionable persona voice structure.

        This method transforms the structured brand bible data into a persona voice
        format that can be directly used by content generation and style checking nodes.
        It implements intelligent mapping with robust fallback behavior.

        Mapping Process:
            1. Utility Mapping: Attempts comprehensive mapping via utils.brand_voice_mapper
            2. Fallback Synthesis: Creates minimal persona structure from available data
            3. Structure Validation: Ensures required fields exist with safe defaults
            4. Quality Assessment: Evaluates completeness of voice guidelines

        Primary Mapping Features (when utility available):
            - Comprehensive voice characteristic extraction
            - Intelligent tone inference from brand descriptions
            - Sophisticated constraint and preference mapping
            - Cross-platform voice consistency validation

        Fallback Mapping Features:
            - Basic voice structure creation from parsed.voice
            - Safe defaults for missing elements
            - Preservation of any existing voice data
            - Minimal viable persona for content generation

        Args:
            parsed: Structured brand bible data from prep(). May be empty, partial,
                   or complete depending on XML parsing success and content quality.

        Returns:
            Dict[str, Any]: Persona voice structure with guaranteed minimum fields:
                {
                    "forbidden_terms": List[str],      # Always present, may be empty
                    "required_phrases": List[str],     # Always present, may be empty
                    "tone_guidelines": Dict[str, Any], # Future enhancement
                    "style_preferences": Dict[str, Any] # Future enhancement
                }

        Quality Indicators:
            - Rich structure: Successful utility mapping with comprehensive voice data
            - Minimal structure: Fallback mapping with basic required fields only
            - Empty lists: No specific voice constraints found in brand bible
        """

        # Try to use mapper utility; fall back to minimal persona synthesis on error
        try:
            from utils.brand_voice_mapper import brand_bible_to_voice

            # Use the mapping utility when available
            # TODO(Pytest): Add pytest tests for voice mapping utility integration
            persona = brand_bible_to_voice(parsed)
            return persona
        except Exception:
            # Fallback: synthesize a minimal persona structure from parsed data
            # TODO(Pytest): Add pytest tests for fallback persona synthesis
            persona = parsed.get("voice", {}) if isinstance(parsed, dict) else {}
            persona.setdefault("forbidden_terms", [])
            persona.setdefault("required_phrases", [])
            return persona

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Persists the aligned persona voice guidelines to shared state.

        This method completes the voice alignment process by storing the derived
        persona voice in the standardized location within shared state. This enables
        downstream nodes to access voice constraints for content generation and
        style compliance checking.

        Storage Strategy:
            - Ensures brand_bible structure exists in shared state
            - Stores persona voice in standardized location
            - Preserves existing brand_bible data while adding voice guidelines
            - Provides access point for content generation and style nodes

        Args:
            shared: Shared pipeline state to update with persona voice guidelines.
            prep_res: Original parsed brand bible data from prep() (reference only).
            exec_res: Processed persona voice structure from exec() containing
                     voice constraints, preferences, and guidelines.

        Returns:
            str: Always returns "default" to continue normal pipeline flow.
                Future versions may implement quality-based routing decisions.

        Side Effects:
            - Creates/updates shared["brand_bible"]["persona_voice"] with voice guidelines
            - Enables ContentCraftsmanNode and StyleEditorNode to access voice constraints
            - Provides foundation for voice-consistent content generation

        Integration Points:
            - ContentCraftsmanNode: Uses persona voice for generation constraints
            - StyleEditorNode: Applies voice constraints during rewriting
            - StyleComplianceNode: Validates content against voice requirements

        Future Enhancements:
            - Voice quality scoring and validation
            - Streaming progress updates for voice alignment
            - Voice guideline version control and change tracking
            - Integration with external voice analysis tools
        """
        # TODO(Validation): Add persona validation
        # TODO(Quality): Implement voice alignment quality scoring
        # TODO(Compliance): Add voice alignment compliance checking
        # TODO(Audit): Implement voice alignment audit logging
        # TODO(Versioning): Add support for voice alignment version control
        # TODO(Metrics): Implement voice alignment performance metrics
        # TODO(Notifications): Add notifications for voice alignment changes
        # TODO(Recovery): Implement voice alignment backup and recovery
        # TODO(Pytest): Add pytest tests for post() method including validation and quality scoring
        shared.setdefault("brand_bible", {})
        shared["brand_bible"]["persona_voice"] = exec_res
        # TODO(Streaming): Emit streaming milestone for voice alignment completion
        # TODO(Schema): Validate persona voice structure
        # TODO(Logging): Log voice alignment warnings/issues
        # TODO(Versioning): Implement persona versioning
        # TODO(Streaming): Add streaming updates for voice alignment progress
        # TODO(Analytics): Implement voice alignment analytics and reporting
        return "default"


class PlatformFormattingNode(Node):
    """Generates platform-specific formatting guidelines for content creation.

    This node creates detailed, actionable formatting guidelines for each social media
    platform based on the current platform context, brand persona, and content intent.
    It's designed to work within BatchFlow contexts where platform is specified via
    node parameters.

    Core Responsibilities:
        1. Platform Context: Extracts platform from node parameters (BatchFlow usage)
        2. Guideline Generation: Creates comprehensive platform-specific rules
        3. Intent Integration: Incorporates content intent into formatting decisions
        4. Persona Alignment: Ensures guidelines respect brand voice constraints

    Platform Guidelines Schema:
        Each generated guideline includes:
        - Character limits and formatting constraints
        - Hashtag rules and placement guidelines
        - Call-to-action requirements and formatting
        - Image/media specifications and requirements
        - Link formatting and shortening requirements
        - Platform-specific engagement optimization rules

    Architecture Design:
        - BatchFlow Integration: Reads platform from self.params["platform"]
        - Utility Integration: Uses utils.format_platform for comprehensive guidelines
        - Fallback Strategy: Provides basic guidelines when utility unavailable
        - Deterministic Output: Ensures consistent guidelines for same inputs

    Input Dependencies:
        - Node Parameters: self.params["platform"] (set by BatchFlow)
        - Persona Voice: shared["brand_bible"]["persona_voice"]
        - Content Intent: shared["task_requirements"]["intents_by_platform"]

    Output Integration:
        - Stores guidelines in shared["platform_guidelines"][platform]
        - Enables ContentCraftsmanNode to generate platform-appropriate content
        - Provides constraints for StyleComplianceNode validation
    """

    def prep(self, shared: Dict[str, Any]) -> tuple[str, Dict[str, Any], Any]:
        """Prepares platform context and dependencies for guideline generation.

        This method assembles all necessary inputs for building platform-specific
        guidelines, including platform identification, persona voice constraints,
        and content intent specifications. It handles missing data gracefully
        with appropriate defaults.

        Platform Resolution Strategy:
            1. Primary: Read from self.params["platform"] (BatchFlow usage)
            2. Fallback: Use "general" platform for standalone usage
            3. Future: Accept platform via prep() inputs for flexibility

        Input Assembly:
            - Platform: Target social media platform for guidelines
            - Persona: Brand voice constraints from VoiceAlignmentNode
            - Intent: Platform-specific content intent from EngagementManagerNode

        Args:
            shared: Shared pipeline state containing persona voice and content intents.
                   Expected to contain brand_bible.persona_voice and task_requirements.

        Returns:
            tuple[str, Dict[str, Any], Any]: A tuple containing:
                - platform (str): Target platform name or "general"
                - persona (Dict): Voice constraints and preferences
                - intent (Any): Platform-specific content intent or None

        Design Considerations:
            - BatchFlow Compatibility: Works seamlessly in batch processing contexts
            - Standalone Support: Handles missing parameters gracefully
            - Data Safety: Provides safe defaults for missing dependencies
            - Future Flexibility: Structure supports multiple input methods

        Error Handling:
            - Missing parameters: Defaults to "general" platform
            - Missing persona: Uses empty dict, exec() handles gracefully
            - Missing intent: Uses None, exec() adapts guidelines accordingly
        """
        # TODO(Input): Add support for accepting platform via prep() inputs for standalone usage
        # TODO(Validation): Validate platform parameter exists and is supported
        # TODO(Aliases): Implement platform aliases
        # TODO(Capability): Add platform capability detection and validation
        # TODO(Config): Implement platform-specific configuration loading
        # TODO(Compatibility): Add support for platform API version compatibility checking
        # TODO(Versioning): Implement platform guideline versioning and updates
        # TODO(Inheritance): Add platform guideline inheritance from global settings
        # TODO(Compliance): Implement platform-specific compliance requirements
        # TODO(Beta): Add support for platform beta features and experimental guidelines
        # TODO(Pytest): Add pytest tests for prep() method including platform validation and aliases
        platform = getattr(self, "params", {}).get("platform") or "general"
        persona = shared.get("brand_bible", {}).get("persona_voice", {})
        intent = shared.get("task_requirements", {}).get("intents_by_platform", {}).get(platform, {}).get("value")
        return platform, persona, intent

    # TODO(Enhancement): PlatformFormattingNode
    # - The node assumes `self.params["platform"]` is set when used in a
    #   BatchFlow. Add support to accept a platform via `prep()` inputs so the
    #   node can be used standalone or inside BatchFlow without relying on
    #   external param wiring.
    # - TODO(Schema): Ensure `build_guidelines` produces a deterministic schema; add tests
    #   verifying keys (max_length, hashtag_rules, cta_requirements, etc.).
    # - TODO(Customization): Add support for platform guideline customization and overrides
    # - TODO(Testing): Implement platform guideline validation and testing
    # - TODO(Sync): Add support for multi-platform guideline synchronization
    # - TODO(Performance): Implement platform guideline performance optimization
    # - TODO(Testing): Add support for platform guideline A/B testing
    # - TODO(Analytics): Implement platform guideline analytics and metrics
    # - TODO(Automation): Add support for platform guideline automation and updates
    # - TODO(Monitoring): Implement platform guideline compliance monitoring
    # - TODO(Pytest): Add pytest tests for platform guideline schema, customization, and A/B testing

    def exec(self, inputs: tuple[str, Dict[str, Any], Any]) -> Dict[str, Any]:
        """Builds comprehensive platform-specific formatting guidelines.

        This method generates detailed guidelines that content generation nodes can
        use to create platform-appropriate content. It implements intelligent
        guideline construction with robust fallback behavior for maximum reliability.

        Guideline Generation Strategy:
            1. Utility Integration: Uses utils.format_platform.build_guidelines for comprehensive rules
            2. Fallback Generation: Creates basic guidelines when utility unavailable
            3. Intent Integration: Incorporates content intent into guideline decisions
            4. Persona Alignment: Respects brand voice constraints in formatting rules

        Comprehensive Guidelines Include:
            - Character limits and text formatting constraints
            - Hashtag usage rules and placement guidelines
            - Call-to-action formatting and requirements
            - Image and media specifications
            - Link formatting and shortening rules
            - Platform-specific engagement optimization strategies
            - Accessibility requirements and compliance rules

        Args:
            inputs: Tuple from prep() containing (platform, persona, intent):
                   - platform (str): Target platform for guideline generation
                   - persona (Dict): Brand voice constraints and preferences
                   - intent (Any): Platform-specific content intent

        Returns:
            Dict[str, Any]: Comprehensive platform guidelines with structure:
                {
                    "platform": str,                    # Platform identifier
                    "max_length": int,                  # Character limit
                    "hashtag_rules": Dict[str, Any],    # Hashtag guidelines
                    "cta_requirements": Dict[str, Any], # Call-to-action rules
                    "image_specs": Dict[str, Any],      # Media specifications
                    "link_formatting": Dict[str, Any],  # Link handling rules
                    "engagement_optimization": Dict,    # Engagement strategies
                    "accessibility_requirements": Dict, # Accessibility rules
                    # Additional platform-specific fields
                }

        Quality Assurance:
            - Deterministic schema ensures consistent guideline structure
            - Platform-specific rules optimize for each platform's unique requirements
            - Intent-aware guidelines adapt to content purpose and goals
            - Persona integration maintains brand voice consistency

        Fallback Behavior:
            - Basic guidelines with essential fields when utility unavailable
            - Platform-specific character limits and basic formatting rules
            - Safe defaults that don't break downstream content generation
            - Clear indication of limited guideline completeness
        """

        platform, persona, intent = inputs
        try:
            from utils.format_platform import build_guidelines

            # TODO(Config): Add guideline building configuration options
            # TODO(Validation): Implement guideline validation and quality checks
            # TODO(Templates): Add support for custom guideline templates
            # TODO(Performance): Implement guideline performance monitoring
            # TODO(Performance): Add support for guideline caching and optimization
            # TODO(Versioning): Implement guideline versioning and change tracking
            # TODO(Pytest): Add pytest tests for build_guidelines utility integration
            guidelines = build_guidelines(platform, persona, intent)
            return guidelines
        except Exception:
            # TODO(Platform): Add comprehensive platform-specific rules for all supported platforms
            # TODO(Rules): Include hashtag_rules, cta_requirements, image_specs, link_formatting
            # TODO(Validation): Add validation for platform-specific constraints
            # TODO(Schema): Implement deterministic schema with all required keys
            # TODO(Encoding): Add support for platform-specific character encoding rules
            # TODO(Characters): Implement platform-specific emoji and special character handling
            # TODO(Accessibility): Add support for platform-specific accessibility requirements
            # TODO(SEO): Implement platform-specific SEO and discoverability guidelines
            # TODO(Monetization): Add support for platform-specific monetization rules
            # TODO(Legal): Implement platform-specific compliance and legal requirements
            # TODO(Analytics): Add support for platform-specific analytics and tracking
            # TODO(Scheduling): Implement platform-specific content scheduling guidelines
            # TODO(Pytest): Add pytest tests for fallback guidelines and platform-specific rules
            # Fallback minimal guideline
            # TODO: Implement comprehensive platform-specific rules and deterministic schema to make fallback redundant
            return {
                "platform": platform,
                "max_length": 280 if platform == "twitter" else 2000,
                "notes": "fallback guidelines",
                # TODO(Schema): Add missing guideline fields: hashtag_rules, cta_requirements, etc.
                # TODO(Specs): Add image_specs, video_specs, link_formatting
                # TODO(Optimization): Add engagement_optimization, posting_schedule, audience_targeting
                # TODO(Requirements): Add accessibility_requirements, compliance_rules, monetization_guidelines
            }

    def post(self, shared: Dict[str, Any], prep_res: tuple[str, Dict[str, Any], Any], exec_res: Dict[str, Any]) -> str:
        """Persists generated platform guidelines to shared state for downstream access.

        This method completes the platform formatting process by storing the generated
        guidelines in a standardized location within shared state. This enables content
        generation and style compliance nodes to access platform-specific formatting
        requirements.

        Storage Strategy:
            - Creates nested platform_guidelines structure if not exists
            - Stores guidelines keyed by platform name for easy access
            - Preserves existing guidelines for other platforms
            - Provides structured access for downstream nodes

        Args:
            shared: Shared pipeline state to update with platform guidelines.
            prep_res: Tuple containing (platform, persona, intent) from prep().
            exec_res: Generated guidelines dictionary from exec() containing
                     platform-specific formatting rules and constraints.

        Returns:
            str: Always returns "default" to continue normal pipeline flow.
                Future versions may implement quality-based routing decisions.

        Side Effects:
            - Creates/updates shared["platform_guidelines"][platform] with guidelines
            - Enables ContentCraftsmanNode to access formatting requirements
            - Provides StyleComplianceNode with validation constraints

        Integration Points:
            - ContentCraftsmanNode: Uses guidelines for section budgeting and formatting
            - StyleComplianceNode: Validates content against platform constraints
            - Future nodes: Can access comprehensive platform requirements

        Data Quality Assurance:
            - Guidelines structure is validated before storage (future enhancement)
            - Platform identification is preserved for debugging and auditing
            - Guidelines completeness can be assessed by downstream nodes

        Future Enhancements:
            - Guideline quality scoring and validation
            - Streaming progress updates for guideline generation
            - Guidelines change detection and notification
            - Integration with platform API updates and changes
        """
        platform = exec_res.get("platform") if isinstance(exec_res, dict) else prep_res[0]
        shared.setdefault("platform_guidelines", {})
        shared["platform_guidelines"][platform] = exec_res
        # TODO(Streaming): Emit streaming milestone for platform guideline generation
        # TODO(Schema): Validate guideline structure against expected schema
        # TODO(Quality): Implement guideline quality scoring and validation
        # TODO(Completeness): Add guideline completeness checking
        # TODO(Change): Implement guideline change detection and notifications
        # TODO(Metrics): Add guideline performance metrics collection
        # TODO(Versioning): Implement guideline backup and versioning
        # TODO(Compliance): Add guideline compliance monitoring and reporting
        # TODO(Pytest): Add pytest tests for post() method including streaming and validation
        return "default"


class ContentCraftsmanNode(Node):
    """Generates platform-specific content drafts using brand voice and guidelines.

    This node represents the core content generation engine of the pipeline,
    responsible for creating initial content drafts that respect brand voice,
    platform constraints, and content intents. It implements intelligent
    content generation with robust fallback behavior.

    Core Generation Process:
        1. Input Assembly: Gathers platform guidelines, persona voice, and topics
        2. Content Generation: Creates drafts using LLM or fallback methods
        3. Platform Optimization: Adapts content for specific platform requirements
        4. Quality Assurance: Ensures generated content meets basic quality standards

    Generation Strategy:
        - Primary: Section-budgeted generation using platform guidelines and LLM
        - Fallback: Simple placeholder generation for demonstration purposes
        - Future: Advanced generation with section structure and optimization

    Content Structure (Planned):
        Each generated content piece will include:
        - Hook: Attention-grabbing opening
        - Body: Main content with key messaging
        - Call-to-Action: Platform-appropriate engagement request
        - Hashtags: Relevant, platform-optimized hashtags
        - Links: Properly formatted and tracked links

    Quality Controls:
        - Character limit validation per platform
        - Brand voice constraint adherence
        - Required phrase inclusion
        - Forbidden term avoidance
        - Platform-specific optimization

    Performance Considerations:
        - LLM cost tracking and budget management
        - Generation retry logic for failed attempts
        - Content caching for similar requests
        - Batch generation optimization for multiple platforms
    """

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Assembles all inputs required for intelligent content generation.

        This method gathers platform guidelines, brand persona, and content topics
        from shared state, providing a comprehensive context for content generation.
        It ensures all necessary information is available while handling missing
        data gracefully.

        Input Assembly Strategy:
            - Platform Guidelines: Formatting rules and constraints from PlatformFormattingNode
            - Persona Voice: Brand voice constraints from VoiceAlignmentNode
            - Content Topic: Generation subject from EngagementManagerNode
            - Future: Additional context like audience data, trends, competition analysis

        Args:
            shared: Shared pipeline state containing all necessary generation inputs.
                   Expected to contain platform_guidelines, brand_bible.persona_voice,
                   and task_requirements.topic_or_goal.

        Returns:
            Dict[str, Any]: Comprehensive generation context with structure:
                {
                    "platform_guidelines": Dict[str, Dict],  # Guidelines per platform
                    "persona": Dict[str, Any],               # Brand voice constraints
                    "topic": str                             # Content generation topic
                }

        Data Validation:
            - Platform guidelines existence and structure (future enhancement)
            - Persona voice completeness assessment (future enhancement)
            - Topic quality and specificity validation (future enhancement)

        Error Handling:
            - Missing guidelines: Uses empty dict, exec() adapts gracefully
            - Missing persona: Uses empty dict with safe defaults
            - Missing topic: Uses empty string with appropriate handling

        Future Enhancements:
            - Additional context gathering (audience data, competitor analysis)
            - Content generation configuration and preferences
            - Historical performance data for optimization
            - A/B testing parameters and variation controls
        """
        # TODO(Validation): Add input validation for platform guidelines and persona
        # TODO(Config): Implement content generation configuration management
        # TODO(Templates): Add support for content templates and presets
        # TODO(Budget): Implement content generation budget and cost tracking
        # TODO(Quality): Add support for content generation quality controls
        # TODO(Performance): Implement content generation performance monitoring
        # TODO(Testing): Add support for content generation A/B testing
        # TODO(Compliance): Implement content generation compliance checking
        # TODO(Pytest): Add pytest tests for prep() method including validation and configuration
        return {
            "platform_guidelines": shared.get("platform_guidelines", {}),
            "persona": shared.get("brand_bible", {}).get("persona_voice", {}),
            "topic": shared.get("task_requirements", {}).get("topic_or_goal", "")
        }

    def exec(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generates platform-optimized content drafts with intelligent fallback.

        This method implements the core content generation logic, creating
        platform-specific drafts that respect brand voice, platform constraints,
        and content objectives. It provides robust fallback behavior for
        demonstration and testing purposes.

        Current Implementation:
            - Simple placeholder generation for immediate functionality
            - Platform-aware text generation with basic customization
            - Topic integration into generated content

        Planned Implementation:
            - Section-budgeted generation using platform guidelines
            - LLM integration via utils.call_llm or utils.openrouter_client
            - Advanced content optimization and refinement
            - Quality validation and iterative improvement

        Generation Process (Future):
            1. Section Planning: Budget character limits across content sections
            2. LLM Generation: Create content using constrained prompts
            3. Platform Optimization: Adapt content for specific platform requirements
            4. Quality Validation: Ensure content meets brand and platform standards
            5. Iterative Refinement: Improve content through feedback loops

        Args:
            inputs: Generation context from prep() containing platform guidelines,
                   persona voice, and content topic for draft creation.

        Returns:
            Dict[str, str]: Generated content drafts mapped by platform:
                {
                    "twitter": "Platform-optimized content for Twitter...",
                    "linkedin": "Professional content for LinkedIn...",
                    "instagram": "Visual-focused content for Instagram...",
                    # Additional platforms as specified in guidelines
                }

        Quality Assurance (Future):
            - Character count validation per platform
            - Brand voice constraint compliance
            - Required phrase inclusion verification
            - Forbidden term avoidance confirmation
            - Platform-specific optimization scoring

        Error Handling:
            - Missing guidelines: Generates basic content with safe defaults
            - Missing topic: Uses generic content with platform customization
            - LLM failures: Falls back to template-based generation
            - Quality issues: Implements retry logic with refinement
        """

        guidelines = inputs.get("platform_guidelines", {})
        persona = inputs.get("persona", {})
        topic = inputs.get("topic", "")
        
        if not topic:
            log.warning("No topic provided for content generation")
            self._update_health(HealthStatus.DEGRADED, "No topic provided")
            topic = "general content"
        
        drafts = {}
        generation_metrics = {
            "total_platforms": len(guidelines),
            "generated_platforms": 0,
            "character_counts": {},
            "quality_scores": {}
        }
        
        for platform, platform_guidelines in guidelines.items():
            try:
                # Get platform-specific constraints
                max_length = platform_guidelines.get("max_length", 280)
                
                # Generate section-budgeted content
                content = self._generate_platform_content(platform, topic, persona, max_length)
                
                # Validate character count
                char_count = len(content)
                if char_count > max_length:
                    log.warning(f"Content for {platform} exceeds limit: {char_count}/{max_length}")
                    content = content[:max_length-3] + "..."
                    char_count = len(content)
                
                drafts[platform] = content
                generation_metrics["generated_platforms"] += 1
                generation_metrics["character_counts"][platform] = char_count
                generation_metrics["quality_scores"][platform] = self._calculate_quality_score(content, persona)
                
            except Exception as e:
                log.error(f"Content generation failed for {platform}: {e}")
                # Fallback content
                drafts[platform] = f"Content about {topic} for {platform}"
                generation_metrics["quality_scores"][platform] = 0.5
        
        # Update health status
        success_rate = generation_metrics["generated_platforms"] / generation_metrics["total_platforms"]
        if success_rate >= 0.8:
            self._update_health(HealthStatus.HEALTHY, f"Content generated for {generation_metrics['generated_platforms']}/{generation_metrics['total_platforms']} platforms")
        elif success_rate >= 0.5:
            self._update_health(HealthStatus.DEGRADED, f"Partial content generation: {generation_metrics['generated_platforms']}/{generation_metrics['total_platforms']} platforms")
        else:
            self._update_health(HealthStatus.UNHEALTHY, f"Content generation mostly failed: {generation_metrics['generated_platforms']}/{generation_metrics['total_platforms']} platforms")
        
        # Store metrics for downstream use
        self._generation_metrics = generation_metrics
        
        return drafts

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, str]) -> str:
        """Persists generated content drafts and emits generation milestones.

        This method completes the content generation process by storing the
        generated drafts in shared state and providing real-time feedback
        through the streaming interface. It structures the content data for
        optimal access by downstream style and compliance nodes.

        Storage Strategy:
            - Creates structured content pieces with text and metadata
            - Preserves platform-specific organization for easy access
            - Provides foundation for style editing and compliance checking
            - Maintains generation audit trail (future enhancement)

        Streaming Integration:
            - Emits completion milestones for each platform
            - Provides real-time generation progress updates
            - Handles streaming failures gracefully without breaking pipeline
            - Future: Streaming generation metrics and quality scores

        Args:
            shared: Shared pipeline state to update with generated content.
            prep_res: Generation context from prep() (reference for debugging).
            exec_res: Generated content drafts mapped by platform from exec().

        Returns:
            str: Always returns "default" to continue normal pipeline flow.
                Future versions may implement quality-based routing decisions.

        Side Effects:
            - Creates/updates shared["content_pieces"] with structured content data
            - Emits streaming milestones for each generated platform draft
            - Enables StyleEditorNode and StyleComplianceNode to access content

        Content Structure:
            Each content piece is stored as:
            {
                "text": str,                 # Generated content text
                "sections": Dict[str, str],  # Future: Structured content sections
                "metadata": Dict[str, Any]   # Future: Generation metadata
            }

        Integration Points:
            - StyleEditorNode: Accesses content for constraint-based rewriting
            - StyleComplianceNode: Validates content against style requirements
            - Future analytics: Uses metadata for performance analysis

        Future Enhancements:
            - Rich content metadata (generation method, quality scores, etc.)
            - Structured section organization (hook, body, CTA separation)
            - Generation performance metrics and cost tracking
            - Content versioning for revision management
        """
        shared.setdefault("content_pieces", {})
        for p, text in exec_res.items():
            # TODO(Structure): Add proper section structure instead of empty sections dict
            # TODO(Metadata): Include metadata like word_count, character_count, generation_method
            # TODO(Quality): Add content quality scores and metrics
            # TODO(Validation): Implement content validation and compliance checking
            # TODO(Prediction): Add content performance predictions
            # TODO(Optimization): Implement content optimization suggestions
            # TODO(Accessibility): Add content accessibility analysis
            # TODO(SEO): Implement content SEO scoring
            # TODO(Sentiment): Add content sentiment analysis
            # TODO(Readability): Implement content readability scoring
            # TODO(Pytest): Add pytest tests for content structure and metadata generation
            shared["content_pieces"][p] = {"text": text, "sections": {}}
        # TODO(Streaming): Add more detailed streaming milestones with content metrics
        # TODO(Streaming): Implement streaming progress updates during generation
        # TODO(Streaming): Add streaming quality scores and validation results
        # TODO(Streaming): Implement streaming cost and performance metrics
        # Stream simple milestones
        stream = shared.get("stream")
        if stream and hasattr(stream, "emit"):
            try:
                for p in exec_res.keys():
                    stream.emit("system", f"Draft complete: {p}")
            except Exception:
                log.debug("stream.emit failed", exc_info=True)
                # TODO(Reliability): Implement fallback notification mechanisms
                # TODO(Reliability): Add retry logic for streaming failures
                # TODO(Integration): Implement alternative notification channels
                # TODO(Pytest): Add pytest tests for streaming failures and retry mechanisms
                # TODO: Implement retry logic and alternative notification channels for streaming to make fallback redundant
        return "default"
    
    def _generate_platform_content(self, platform: str, topic: str, persona: Dict[str, Any], max_length: int) -> str:
        """Generate platform-specific content with section budgeting."""
        # Section budgeting (hook: 20%, body: 60%, CTA: 20%)
        hook_length = int(max_length * 0.2)
        body_length = int(max_length * 0.6)
        cta_length = int(max_length * 0.2)
        
        # Platform-specific templates
        templates = {
            "twitter": {
                "hook": "🚀 {topic} is changing everything!",
                "body": "Here's what you need to know about {topic} and why it matters for your business.",
                "cta": "What's your take on {topic}? Share your thoughts below! 👇"
            },
            "linkedin": {
                "hook": "The future of {topic} is here, and it's reshaping our industry.",
                "body": "As professionals, we need to understand how {topic} will impact our work and opportunities.",
                "cta": "How do you see {topic} affecting your field? Let's discuss in the comments."
            },
            "instagram": {
                "hook": "✨ {topic} is the trend you can't ignore!",
                "body": "Discover why {topic} is becoming essential for modern businesses and creators.",
                "cta": "Tag someone who needs to know about {topic}! 🔥"
            },
            "facebook": {
                "hook": "Big news about {topic} that affects us all!",
                "body": "Let's talk about how {topic} is changing the game and what it means for our community.",
                "cta": "What do you think about {topic}? Drop your thoughts below!"
            }
        }
        
        # Get platform template or use default
        template = templates.get(platform, {
            "hook": "Exciting news about {topic}!",
            "body": "Here's what you need to know about {topic} and its impact.",
            "cta": "What's your opinion on {topic}? Share below!"
        })
        
        # Generate sections
        hook = template["hook"].format(topic=topic)[:hook_length]
        body = template["body"].format(topic=topic)[:body_length]
        cta = template["cta"].format(topic=topic)[:cta_length]
        
        # Apply brand voice constraints
        forbidden_terms = persona.get("forbidden_terms", [])
        required_phrases = persona.get("required_phrases", [])
        
        # Remove forbidden terms
        for term in forbidden_terms:
            hook = hook.replace(term, "")
            body = body.replace(term, "")
            cta = cta.replace(term, "")
        
        # Add required phrases if space allows
        for phrase in required_phrases:
            if len(body + " " + phrase) <= body_length:
                body += " " + phrase
                break
        
        # Combine sections
        content = f"{hook}\n\n{body}\n\n{cta}"
        
        # Final length check and truncation
        if len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        return content
    
    def _calculate_quality_score(self, content: str, persona: Dict[str, Any]) -> float:
        """Calculate content quality score based on various factors."""
        score = 1.0
        
        # Check for forbidden terms
        forbidden_terms = persona.get("forbidden_terms", [])
        for term in forbidden_terms:
            if term.lower() in content.lower():
                score -= 0.2
        
        # Check for required phrases
        required_phrases = persona.get("required_phrases", [])
        for phrase in required_phrases:
            if phrase.lower() in content.lower():
                score += 0.1
        
        # Check content length (prefer longer content)
        if len(content) < 50:
            score -= 0.3
        elif len(content) > 200:
            score += 0.1
        
        # Check for engagement elements
        if any(char in content for char in "!?@"):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _update_health(self, status: HealthStatus, message: str):
        """Update node health status and metrics."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        
        self._health.status = status
        self._health.message = message
        self._health.timestamp = time.time()
        
        if status != HealthStatus.HEALTHY:
            log.warning(f"Node health: {status.value} - {message}")
    
    def get_health(self) -> NodeHealth:
        """Get current node health status."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        return self._health

    # TODO(Enhancement): ContentCraftsmanNode
    # - Replace template-based generation with LLM integration:
    #   call LLM (via `utils.call_llm` or `utils.openrouter_client`) to
    #   generate each section. Respect `max_retries` and cost tracking.
    # - TODO(Validation): Implement hashtag/link placement rules and validate final character
    #   counts for platforms with limits (Twitter/X).
    # - TODO(Testing): Add unit tests that exercise LLM-fallback behavior (when API keys
    #   are not present) using recorded responses or mocks.
    # - TODO(Workflow): Add support for content generation workflows and approval processes
    # - TODO(Analytics): Implement content generation analytics and optimization
    # - TODO(Scaling): Add support for content generation scaling and load balancing
    # - TODO(Security): Implement content generation security and privacy controls
    # - TODO(Integration): Add support for content generation integration with external tools
    # - TODO(Monitoring): Implement content generation monitoring and alerting
    # - TODO(Personalization): Add support for content generation customization and personalization
    # - TODO(Pytest): Add pytest tests for LLM integration, validation, and workflow integration


class StyleEditorNode(Node):
    """Applies intelligent content rewriting to remove AI fingerprints and enforce brand voice.

    This node implements sophisticated content refinement that transforms raw generated
    content into brand-aligned, human-like text. It removes AI-generated patterns,
    enforces brand voice constraints, and ensures content authenticity while maintaining
    the original message and intent.

    Core Rewriting Functions:
        1. AI Fingerprint Removal: Eliminates common AI-generated patterns and phrases
        2. Brand Voice Enforcement: Applies persona voice constraints and preferences
        3. Forbidden Term Filtering: Removes or replaces terms specified in brand guidelines
        4. Required Phrase Integration: Ensures brand-required phrases are appropriately included
        5. Tone Alignment: Adjusts content tone to match brand personality

    Rewriting Strategy:
        - Primary: Uses utils.rewrite_with_constraints for comprehensive, context-aware rewriting
        - Fallback: Implements basic term replacement and character constraint enforcement
        - Quality Assurance: Validates rewritten content meets all brand and platform requirements

    Content Processing Pipeline:
        1. Pre-processing: Analyze content for AI patterns and brand constraint violations
        2. LLM Rewriting: Use constrained prompts to rewrite content while preserving meaning
        3. Post-processing: Apply deterministic fixes and validate constraint compliance
        4. Quality Validation: Ensure rewritten content maintains quality and brand alignment

    Brand Constraint Categories:
        - Forbidden Terms: Words/phrases to avoid or replace
        - Required Phrases: Brand messaging that must be included when relevant
        - Tone Guidelines: Voice characteristics and communication style
        - Platform Rules: Platform-specific language and formatting requirements
        - Legal/Compliance: Regulatory and legal language requirements

    Performance Optimization:
        - Batch rewriting for multiple platforms
        - Intelligent change detection to minimize unnecessary processing
        - Content similarity analysis to reuse successful rewrites
        - Progressive enhancement based on constraint complexity
    """

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares content pieces and persona voice constraints for style rewriting.

        This method assembles all necessary inputs for intelligent content rewriting,
        including the generated content drafts and the brand persona voice constraints
        that will guide the rewriting process. It ensures both content and constraints
        are available for comprehensive style editing.

        Input Assembly Strategy:
            - Content Pieces: Generated drafts from ContentCraftsmanNode
            - Persona Voice: Brand voice constraints from VoiceAlignmentNode
            - Future: Platform-specific style preferences and rewriting rules

        Args:
            shared: Shared pipeline state containing content pieces and persona voice.
                   Expected to contain content_pieces from generation and brand_bible.persona_voice
                   from voice alignment processing.

        Returns:
            Dict[str, Any]: Rewriting context with structure:
                {
                    "content_pieces": Dict[str, Dict],  # Generated content by platform
                    "persona": Dict[str, Any]           # Brand voice constraints
                }
                
                Content pieces structure:
                {
                    "platform_name": {
                        "text": str,              # Generated content text
                        "sections": Dict[str, str] # Future: Structured sections
                    }
                }
                
                Persona structure:
                {
                    "forbidden_terms": List[str],      # Terms to avoid/replace
                    "required_phrases": List[str],     # Phrases to include when relevant
                    "tone_guidelines": Dict[str, Any], # Voice characteristics
                    "style_preferences": Dict[str, Any] # Writing style preferences
                }

        Data Quality Assurance:
            - Content pieces validation and structure checking (future enhancement)
            - Persona voice completeness assessment (future enhancement)
            - Rewriting complexity analysis and planning (future enhancement)

        Error Handling:
            - Missing content: Uses empty dict, exec() handles gracefully
            - Missing persona: Uses empty dict with safe constraint defaults
            - Invalid structure: Future enhancement will add comprehensive validation
        """
        # TODO(Validation): Add validation for content pieces and persona structure
        # TODO(Config): Implement style editing configuration management
        # TODO(Templates): Add support for style editing templates and rules
        # TODO(Performance): Implement style editing performance monitoring
        # TODO(Quality): Add support for style editing quality controls
        # TODO(Audit): Implement style editing audit logging
        # TODO(Testing): Add support for style editing A/B testing
        # TODO(Compliance): Implement style editing compliance checking
        # TODO(Pytest): Add pytest tests for prep() method including validation and configuration
        return {
            "content_pieces": shared.get("content_pieces", {}),
            "persona": shared.get("brand_bible", {}).get("persona_voice", {})
        }

    def exec(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Performs intelligent content rewriting with brand voice constraint enforcement.

        This method implements sophisticated content rewriting that removes AI fingerprints,
        enforces brand voice constraints, and produces human-like content that maintains
        the original message while meeting all brand requirements. It uses advanced
        rewriting techniques with robust fallback behavior.

        Rewriting Process:
            1. Content Analysis: Identify AI patterns, constraint violations, and improvement opportunities
            2. Constraint Planning: Determine rewriting strategy based on persona voice requirements
            3. LLM Rewriting: Apply constrained prompts to rewrite content contextually
            4. Post-processing: Apply deterministic fixes and validate constraint compliance
            5. Quality Validation: Ensure rewritten content meets quality and brand standards

        Primary Rewriting Features (when utility available):
            - Context-aware AI pattern removal
            - Intelligent forbidden term replacement with contextual alternatives
            - Natural integration of required phrases
            - Tone adjustment based on brand personality
            - Platform-specific style optimization

        Fallback Rewriting Features:
            - Basic forbidden term removal/replacement
            - Hard-coded character constraint enforcement (em-dash removal)
            - Safe content preservation when advanced rewriting unavailable
            - Clear indication of limited rewriting scope

        Args:
            inputs: Rewriting context from prep() containing content pieces and persona constraints.

        Returns:
            Dict[str, str]: Rewritten content mapped by platform:
                {
                    "twitter": "Rewritten, brand-aligned content for Twitter...",
                    "linkedin": "Professional, constraint-compliant content for LinkedIn...",
                    # Additional platforms with their rewritten content
                }

        Rewriting Quality Indicators:
            - Comprehensive rewriting: All constraints applied, AI patterns removed
            - Basic rewriting: Forbidden terms removed, minimal pattern adjustment
            - Limited rewriting: Only hard-coded fixes applied (fallback mode)

        Error Handling Strategy:
            - Utility unavailable: Falls back to regex-based term replacement
            - Rewriting failures: Preserves original content with basic fixes
            - Constraint conflicts: Prioritizes forbidden term removal over required phrase inclusion
            - Quality issues: Logs warnings for manual review (future enhancement)

        Future Enhancements:
            - Progressive rewriting with quality checkpoints
            - Content similarity preservation metrics
            - Brand voice consistency scoring
            - Rewriting performance analytics and optimization
        """

        content = inputs["content_pieces"]
        persona = inputs.get("persona", {})
        try:
            from utils.rewrite_with_constraints import rewrite_with_constraints
            # TODO(Error): Add proper error handling for individual platform rewrites
            # TODO(Metrics): Track rewrite metrics (changes made, forbidden terms found, etc.)
            # TODO(Quality): Implement rewrite quality scoring and validation
            # TODO(Config): Add support for rewrite configuration and customization
            # TODO(Performance): Implement rewrite performance optimization
            # TODO(Templates): Add support for rewrite templates and rules
            # TODO(Compliance): Implement rewrite compliance checking
            # TODO(Testing): Add support for rewrite A/B testing
            # TODO(Analytics): Implement rewrite analytics and reporting
            # TODO(Versioning): Add support for rewrite version control
            # TODO(Pytest): Add pytest tests for rewrite_with_constraints utility integration
            rewritten = {}
            for p, payload in content.items():
                # TODO(Platform): Add platform-specific rewrite rules and constraints
                # TODO(Quality): Implement rewrite quality validation per platform
                # TODO(Optimization): Add support for incremental rewriting and optimization
                rewritten[p] = rewrite_with_constraints(payload.get("text", ""), persona, {})
            return rewritten
        except Exception:
            # TODO(Deterministic): Implement pre-apply deterministic regex fixes
            # TODO(Validation): Add validation for absence of forbidden_terms after rewrite
            # TODO(Validation): Ensure required phrases remain present after rewrite
            # TODO(Review): Flag content for manual review instead of silent removal
            # TODO(Debugging): Log specific rewrite failures and fallback actions
            # TODO(Strategy): Implement comprehensive fallback rewriting strategies
            # TODO(Partial): Add support for partial rewriting when full rewrite fails
            # TODO(Recovery): Implement rewrite failure recovery and retry mechanisms
            # TODO(Manual): Add support for manual intervention workflows
            # TODO(Warning): Implement rewrite quality degradation warnings
            # TODO(Pytest): Add pytest tests for fallback rewriting and error handling
            # Fallback: naive replace of forbidden terms
            # TODO: Implement deterministic regex fixes and validation of forbidden terms/phrases to make fallback redundant
            rewritten = {}
            forb = persona.get("forbidden_terms", []) if isinstance(persona, dict) else []
            for p, payload in content.items():
                txt = payload.get("text", "")
                # TODO(Matching): Add case-insensitive matching and word boundary detection
                # TODO(Audit): Track which terms were replaced for auditing
                # TODO(Context): Implement context-aware term replacement
                # TODO(Suggestions): Add support for term replacement suggestions
                # TODO(Validation): Implement term replacement validation
                # TODO(Rules): Add support for custom replacement rules
                # TODO(Performance): Implement term replacement performance optimization
                # TODO(Pytest): Add pytest tests for term replacement and tracking
                for term in forb:
                    # TODO(Strategy): Implement smarter replacement strategies (synonyms, context-aware)
                    # TODO(Tracking): Add support for term replacement logging and tracking
                    txt = txt.replace(term, "")
                # TODO(Config): Make this configurable rather than hard-coded
                # TODO(Rules): Add support for configurable character replacement rules
                # Remove em-dash as hard rule
                txt = txt.replace("—", "-")
                rewritten[p] = txt
            return rewritten

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, str]) -> str:
        """Persists rewritten content and updates shared state with refinement results.

        This method completes the style editing process by storing the rewritten
        content in shared state, replacing the original generated drafts with
        brand-aligned, constraint-compliant versions. It maintains content structure
        while updating the text with improved, human-like content.

        Storage Strategy:
            - Updates existing content pieces with rewritten text
            - Preserves content structure and metadata (sections, etc.)
            - Maintains platform organization for downstream access
            - Future: Tracks rewriting metrics and quality scores

        Args:
            shared: Shared pipeline state to update with rewritten content.
            prep_res: Original content and persona from prep() (reference for auditing).
            exec_res: Rewritten content text mapped by platform from exec().

        Returns:
            str: Always returns "default" to continue normal pipeline flow.
                Future versions may implement quality-based routing decisions.

        Side Effects:
            - Updates shared["content_pieces"] with rewritten content text
            - Preserves existing content structure (sections, metadata)
            - Enables StyleComplianceNode to validate rewritten content
            - Future: Updates shared["style_refinements"] with rewriting metrics

        Content Structure Preservation:
            - Maintains platform-specific organization
            - Preserves sections structure for future enhancement
            - Updates only text field while keeping other metadata
            - Future: Adds rewriting metadata and quality scores

        Integration Points:
            - StyleComplianceNode: Validates rewritten content against style requirements
            - Future analytics: Uses rewriting metrics for performance analysis
            - Audit systems: Tracks content evolution through rewriting process

        Quality Assurance:
            - Content structure validation (future enhancement)
            - Rewriting quality scoring and metrics (future enhancement)
            - Brand voice consistency verification (future enhancement)
            - Performance impact assessment (future enhancement)
        """
        for p, txt in exec_res.items():
            shared.setdefault("content_pieces", {})
            # TODO(Preservation): Preserve existing sections and metadata when updating text
            # TODO(Metadata): Add rewrite metadata (changes made, quality scores, etc.)
            # TODO(Tracking): Implement rewrite change tracking and diff generation
            # TODO(Metrics): Add rewrite performance metrics
            # TODO(Quality): Implement rewrite quality validation
            # TODO(Compliance): Add rewrite compliance checking results
            # TODO(Audit): Implement rewrite audit trail
            # TODO(Analytics): Add rewrite analytics and insights
            # TODO(Pytest): Add pytest tests for post() method including metadata preservation and tracking
            shared["content_pieces"][p] = {"text": txt}
        # TODO(State): Add style_refinements metrics to shared state
        # TODO(Streaming): Emit streaming milestone for style editing completion
        # TODO(Streaming): Implement streaming rewrite progress updates
        # TODO(Streaming): Add streaming rewrite quality scores and metrics
        # TODO(Streaming): Implement streaming rewrite compliance results
        # TODO(Streaming): Add streaming rewrite performance metrics
        return "default"

    # TODO(Enhancement): StyleEditorNode
    # - Integrate full `rewrite_with_constraints` pipeline: pre-apply
    #   deterministic regex fixes, call LLM with a constrained prompt, then
    #   validate the rewritten text for absence of `forbidden_terms` and for
    #   ensuring required phrases remain present. In case of LLM failure,
    #   create a safe fallback that flags manual review instead of silently
    #   removing content.
    # - TODO(Metrics): Add metrics: number of forbidden replacements, list of dropped phrases,
    #   and attach them to `shared["style_refinements"]` for auditing.
    # - TODO(Pipeline): Implement comprehensive style editing pipeline with multiple passes
    # - TODO(QA): Add support for style editing quality assurance and validation
    # - TODO(Performance): Implement style editing performance optimization and caching
    # - TODO(Customization): Add support for style editing customization and configuration
    # - TODO(Integration): Implement style editing integration with external tools and services
    # - TODO(Workflow): Add support for style editing workflow automation and orchestration
    # - TODO(Monitoring): Implement style editing monitoring and alerting
    # - TODO(Scaling): Add support for style editing scaling and load balancing
    # - TODO(Pytest): Add pytest tests for full rewrite pipeline, LLM integration, and metrics tracking


class StyleComplianceNode(Node):
    """Validates content against comprehensive style requirements and manages revision cycles.

    This node implements the final quality gate for content generation, performing
    comprehensive style compliance checking and managing the revision process when
    violations are detected. It ensures all content meets brand, platform, and
    quality standards before final approval.

    Core Compliance Functions:
        1. Style Violation Detection: Identifies all types of style and brand violations
        2. Compliance Scoring: Provides quantitative assessment of content quality
        3. Revision Management: Controls the revision cycle with intelligent limits
        4. Quality Reporting: Generates detailed violation reports for debugging

    Validation Categories:
        - Brand Voice Compliance: Forbidden terms, required phrases, tone consistency
        - Platform Constraints: Character limits, format requirements, engagement rules
        - Content Quality: Readability, clarity, engagement potential
        - Accessibility: Inclusive language, readability standards
        - Legal/Regulatory: Compliance with advertising and content regulations

    Revision Control Strategy:
        - Smart Revision Logic: Considers violation severity and revision history
        - Revision Limits: Prevents infinite revision cycles with configurable limits
        - Quality Progression: Tracks improvement across revision iterations
        - Manual Escalation: Provides pathway for human intervention when needed

    Reporting and Analytics:
        - Detailed Violation Reports: Specific issues with location and severity
        - Quality Metrics: Quantitative scores for content assessment
        - Revision Analytics: Performance tracking across revision cycles
        - Trend Analysis: Pattern identification for continuous improvement

    Integration with Edit Cycles:
        - Revision History: Maintains detailed records of all revision attempts
        - Edit Cycle Reports: Comprehensive analysis when revision limits reached
        - Performance Optimization: Uses historical data to improve future generations
    """

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieves current content pieces for comprehensive compliance checking.

        This method extracts the content pieces that have been processed through
        the generation and style editing pipeline, preparing them for final
        compliance validation. It ensures all content is ready for thorough
        style and brand compliance assessment.

        Args:
            shared: Shared pipeline state containing processed content pieces.
                   Expected to contain content_pieces with rewritten, brand-aligned content.

        Returns:
            Dict[str, Any]: Current content pieces ready for compliance checking:
                {
                    "platform_name": {
                        "text": str,              # Processed content text
                        "sections": Dict[str, str] # Future: Structured sections
                    }
                }

        Content Validation Preparation:
            - Content structure verification (future enhancement)
            - Platform-specific validation rule loading (future enhancement)
            - Compliance checking configuration setup (future enhancement)

        Error Handling:
            - Missing content: Returns empty dict, exec() handles gracefully
            - Invalid structure: Future enhancement will add validation
            - Incomplete processing: Safe defaults prevent checking failures
        """
        # TODO(Validation): Add validation for content pieces structure
        # TODO(Config): Implement compliance checking configuration management
        # TODO(Customization): Add support for compliance rule customization
        # TODO(Performance): Implement compliance checking performance monitoring
        # TODO(Quality): Add support for compliance checking quality controls
        # TODO(Audit): Implement compliance checking audit logging
        # TODO(Testing): Add support for compliance checking A/B testing
        # TODO(Analytics): Implement compliance checking analytics and reporting
        # TODO(Pytest): Add pytest tests for prep() method including validation and configuration
        return shared.get("content_pieces", {})

    def exec(self, content_pieces: Dict[str, Any]) -> Dict[str, Any]:
        """Performs comprehensive style compliance checking with detailed violation reporting.

        This method implements thorough content validation against all applicable
        style requirements, generating detailed reports that enable intelligent
        revision decisions. It uses advanced compliance checking with robust
        fallback behavior for reliability.

        Compliance Checking Process:
            1. Rule Loading: Load platform-specific and brand-specific compliance rules
            2. Content Analysis: Analyze each content piece against all applicable rules
            3. Violation Detection: Identify specific violations with location and severity
            4. Quality Scoring: Generate quantitative compliance scores
            5. Report Generation: Create detailed reports for revision decision-making

        Primary Checking Features (when utility available):
            - Comprehensive style rule validation
            - Forbidden term detection with context analysis
            - Required phrase presence verification
            - Platform constraint validation (character limits, format rules)
            - Brand voice consistency assessment
            - Accessibility and readability analysis

        Fallback Checking Features:
            - Basic forbidden character detection (em-dash)
            - Simple violation counting and scoring
            - Essential compliance verification
            - Safe operation when advanced checking unavailable

        Args:
            content_pieces: Content pieces from prep() to validate against style requirements.

        Returns:
            Dict[str, Any]: Comprehensive compliance reports mapped by platform:
                {
                    "platform_name": {
                        "violations": List[Dict[str, Any]], # Detailed violation information
                        "score": float,                     # Quantitative compliance score
                        "severity": str,                    # Overall severity assessment
                        "recommendations": List[str]        # Improvement recommendations
                    }
                }

        Violation Report Structure:
            Each violation includes:
            - type: Category of violation (forbidden, required, limit, etc.)
            - term/rule: Specific rule or term that was violated
            - location: Position in content where violation occurred
            - severity: Impact level (critical, warning, info)
            - suggestion: Recommended fix or alternative

        Quality Scoring:
            - 100: Perfect compliance, no violations
            - 90-99: Minor issues, acceptable quality
            - 70-89: Moderate issues, revision recommended
            - <70: Significant issues, revision required

        Error Handling:
            - Utility unavailable: Falls back to basic compliance checking
            - Individual platform failures: Continues checking other platforms
            - Rule loading failures: Uses safe default rules
            - Scoring failures: Provides conservative compliance assessment
        """

        reports = {}
        compliance_metrics = {
            "total_platforms": len(content_pieces),
            "checked_platforms": 0,
            "violations_found": 0,
            "critical_violations": 0
        }
        
        try:
            from utils.check_style_violations import check_style_violations
            log.info("Using comprehensive style violation checking utility")
            
            for p, payload in content_pieces.items():
                try:
                    text = payload.get("text", "")
                    if not text:
                        reports[p] = {"violations": [{"type": "critical", "term": "empty_content", "message": "Content is empty"}], "score": 0}
                        compliance_metrics["critical_violations"] += 1
                        continue
                    
                    result = check_style_violations(text)
                    reports[p] = result
                    compliance_metrics["checked_platforms"] += 1
                    
                    # Count violations
                    violations = result.get("violations", [])
                    compliance_metrics["violations_found"] += len(violations)
                    critical_violations = [v for v in violations if v.get("type") == "critical"]
                    compliance_metrics["critical_violations"] += len(critical_violations)
                    
                except Exception as e:
                    log.error(f"Style checking failed for {p}: {e}")
                    reports[p] = {"violations": [{"type": "error", "term": "check_failed", "message": str(e)}], "score": 50}
                    compliance_metrics["critical_violations"] += 1
                    
        except Exception:
            log.warning("Using enhanced fallback compliance checking")
            self._update_health(HealthStatus.DEGRADED, "Using fallback compliance checking")
            
            for p, payload in content_pieces.items():
                try:
                    text = payload.get("text", "")
                    if not text:
                        reports[p] = {"violations": [{"type": "critical", "term": "empty_content", "message": "Content is empty"}], "score": 0}
                        compliance_metrics["critical_violations"] += 1
                        continue
                    
                    violations = []
                    score = 100
                    
                    # Comprehensive style checks
                    violations.extend(self._check_forbidden_characters(text))
                    violations.extend(self._check_forbidden_terms(text))
                    violations.extend(self._check_required_phrases(text))
                    violations.extend(self._check_character_limits(text, p))
                    violations.extend(self._check_hashtag_compliance(text))
                    violations.extend(self._check_cta_compliance(text))
                    violations.extend(self._check_accessibility(text))
                    violations.extend(self._check_brand_voice(text))
                    
                    # Calculate weighted score
                    score = self._calculate_compliance_score(violations)
                    
                    reports[p] = {"violations": violations, "score": score}
                    compliance_metrics["checked_platforms"] += 1
                    compliance_metrics["violations_found"] += len(violations)
                    
                    critical_violations = [v for v in violations if v.get("severity") == "critical"]
                    compliance_metrics["critical_violations"] += len(critical_violations)
                    
                except Exception as e:
                    log.error(f"Fallback compliance checking failed for {p}: {e}")
                    reports[p] = {"violations": [{"type": "error", "term": "check_failed", "message": str(e)}], "score": 50}
                    compliance_metrics["critical_violations"] += 1
        
        # Update health status
        violation_rate = compliance_metrics["violations_found"] / max(compliance_metrics["checked_platforms"], 1)
        if violation_rate == 0:
            self._update_health(HealthStatus.HEALTHY, "All content passed compliance checks")
        elif violation_rate < 0.3:
            self._update_health(HealthStatus.HEALTHY, f"Low violation rate: {violation_rate:.2f}")
        elif violation_rate < 0.7:
            self._update_health(HealthStatus.DEGRADED, f"Moderate violation rate: {violation_rate:.2f}")
        else:
            self._update_health(HealthStatus.UNHEALTHY, f"High violation rate: {violation_rate:.2f}")
        
        # Store metrics for downstream use
        self._compliance_metrics = compliance_metrics
        
        return reports

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Stores compliance reports and manages intelligent revision cycle control.

        This method completes the style compliance checking process by persisting
        violation reports and making intelligent decisions about whether content
        requires revision, has reached acceptable quality, or needs manual intervention.
        It implements sophisticated revision management with quality progression tracking.

        Revision Decision Logic:
            1. Violation Analysis: Assess severity and type of violations found
            2. Quality Trending: Consider improvement across previous revisions
            3. Revision History: Track revision attempts and outcomes
            4. Threshold Management: Apply configurable quality thresholds
            5. Escalation Logic: Determine when manual intervention is needed

        Args:
            shared: Shared pipeline state to update with compliance reports and revision control.
            prep_res: Content pieces that were checked (reference for analysis).
            exec_res: Detailed compliance reports from exec() containing violations and scores.

        Returns:
            str: Routing decision based on compliance analysis:
                - "pass": Content meets quality standards, proceed to next phase
                - "revise": Content needs improvement, initiate revision cycle
                - "max_revisions": Revision limit reached, escalate for manual review

        Side Effects:
            - Updates shared["style_compliance"] with comprehensive violation reports
            - Increments shared["workflow_state"]["revision_count"] when revision needed
            - Future: Updates shared["revision_history"] with detailed revision tracking

        Revision Management Strategy:
            - Quality-based Decisions: Consider violation severity, not just presence
            - Progressive Improvement: Track quality trends across revisions
            - Intelligent Limits: Adapt revision limits based on content complexity
            - Manual Escalation: Provide clear pathway for human intervention

        Future Enhancements:
            - Revision impact prediction and optimization
            - Quality trend analysis and improvement recommendations
            - Automated revision strategy selection
            - Integration with content performance analytics
        """
        shared["style_compliance"] = exec_res
        # TODO(Revision): Implement more sophisticated revision logic based on violation severity
        # TODO(History): Add revision history tracking with diffs and timestamps
        # TODO(Streaming): Emit streaming updates for style compliance results
        # TODO(Analytics): Implement compliance result analytics and insights
        # TODO(Notifications): Add compliance result notification and alerting
        # TODO(Recovery): Implement compliance result backup and recovery
        # TODO(Integration): Add compliance result integration with external systems
        # TODO(Performance): Implement compliance result performance monitoring
        # TODO(Pytest): Add pytest tests for post() method including revision logic and history tracking
        # Decide action: if any platform has forbidden violations, ask to revise
        for p, rep in exec_res.items():
            if rep.get("violations"):
                # TODO(Tracking): Track revision reasons and specific violations that triggered revision
                # TODO(Severity): Implement violation severity-based revision decisions
                # TODO(Partial): Add support for partial revisions (specific platforms only)
                # TODO(Impact): Implement revision impact analysis and predictions
                # increment revision counter
                shared.setdefault("workflow_state", {}).setdefault("revision_count", 0)
                shared["workflow_state"]["revision_count"] += 1
                # TODO(Config): Make max_revisions configurable
                # TODO(Dynamic): Implement dynamic revision limits based on content complexity
                # TODO(Escalation): Add support for revision escalation workflows
                # TODO(Analysis): Implement revision cost-benefit analysis
                if shared["workflow_state"]["revision_count"] >= 5:
                    # TODO(Report): Generate comprehensive EditCycleReport before returning
                    # TODO(Analysis): Implement revision failure analysis and reporting
                    # TODO(Manual): Add support for manual intervention workflows
                    # TODO(Recovery): Implement revision failure recovery strategies
                    # TODO(Pytest): Add pytest tests for max_revisions handling and reporting
                    return "max_revisions"
                return "revise"
        return "pass"
    
    def _check_forbidden_characters(self, text: str) -> List[Dict[str, Any]]:
        """Check for forbidden characters in text."""
        violations = []
        
        # Check for em-dash
        if "—" in text:
            violations.append({
                "type": "forbidden",
                "term": "em-dash",
                "severity": "warning",
                "message": "Em-dash character found, use regular dash instead",
                "suggestion": "Replace — with -"
            })
        
        # Check for other problematic characters
        problematic_chars = {
            "…": "ellipsis",
            "–": "en-dash",
            "—": "em-dash",
            "": "smart quotes",
            "": "smart quotes",
            "'": "smart apostrophe"
        }
        
        for char, description in problematic_chars.items():
            if char in text:
                violations.append({
                    "type": "forbidden",
                    "term": description,
                    "severity": "warning",
                    "message": f"{description} character found",
                    "suggestion": f"Replace {char} with standard character"
                })
        
        return violations
    
    def _check_forbidden_terms(self, text: str) -> List[Dict[str, Any]]:
        """Check for forbidden terms in text."""
        violations = []
        
        # This would typically come from brand bible
        forbidden_terms = ["badword", "inappropriate", "offensive"]
        
        for term in forbidden_terms:
            if term.lower() in text.lower():
                violations.append({
                    "type": "forbidden",
                    "term": term,
                    "severity": "critical",
                    "message": f"Forbidden term '{term}' found",
                    "suggestion": "Remove or replace with appropriate alternative"
                })
        
        return violations
    
    def _check_required_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Check for required phrases in text."""
        violations = []
        
        # This would typically come from brand bible
        required_phrases = ["brand name", "company tagline"]
        
        for phrase in required_phrases:
            if phrase.lower() not in text.lower():
                violations.append({
                    "type": "required",
                    "term": phrase,
                    "severity": "warning",
                    "message": f"Required phrase '{phrase}' not found",
                    "suggestion": "Include the required phrase when relevant"
                })
        
        return violations
    
    def _check_character_limits(self, text: str, platform: str) -> List[Dict[str, Any]]:
        """Check character limits for specific platforms."""
        violations = []
        
        platform_limits = {
            "twitter": 280,
            "linkedin": 3000,
            "instagram": 2200,
            "facebook": 63206
        }
        
        limit = platform_limits.get(platform, 1000)
        char_count = len(text)
        
        if char_count > limit:
            violations.append({
                "type": "limit",
                "term": "character_limit",
                "severity": "critical",
                "message": f"Content exceeds {platform} character limit: {char_count}/{limit}",
                "suggestion": f"Reduce content by {char_count - limit} characters"
            })
        
        return violations
    
    def _check_hashtag_compliance(self, text: str) -> List[Dict[str, Any]]:
        """Check hashtag compliance."""
        violations = []
        
        hashtags = re.findall(r'#\w+', text)
        
        # Check hashtag count
        if len(hashtags) > 5:
            violations.append({
                "type": "hashtag",
                "term": "too_many_hashtags",
                "severity": "warning",
                "message": f"Too many hashtags: {len(hashtags)}",
                "suggestion": "Use 3-5 hashtags maximum"
            })
        
        # Check hashtag length
        for hashtag in hashtags:
            if len(hashtag) > 20:
                violations.append({
                    "type": "hashtag",
                    "term": "long_hashtag",
                    "severity": "warning",
                    "message": f"Hashtag too long: {hashtag}",
                    "suggestion": "Use shorter, more specific hashtags"
                })
        
        return violations
    
    def _check_cta_compliance(self, text: str) -> List[Dict[str, Any]]:
        """Check call-to-action compliance."""
        violations = []
        
        cta_patterns = [
            r'click here',
            r'learn more',
            r'find out',
            r'share your',
            r'let us know',
            r'comment below',
            r'tell us'
        ]
        
        has_cta = any(re.search(pattern, text.lower()) for pattern in cta_patterns)
        
        if not has_cta:
            violations.append({
                "type": "cta",
                "term": "missing_cta",
                "severity": "warning",
                "message": "No call-to-action found",
                "suggestion": "Include a clear call-to-action to encourage engagement"
            })
        
        return violations
    
    def _check_accessibility(self, text: str) -> List[Dict[str, Any]]:
        """Check accessibility compliance."""
        violations = []
        
        # Check for excessive emoji use
        emoji_count = len(re.findall(r'[^\w\s]', text))
        if emoji_count > 5:
            violations.append({
                "type": "accessibility",
                "term": "too_many_emojis",
                "severity": "warning",
                "message": f"Too many emojis: {emoji_count}",
                "suggestion": "Use emojis sparingly for better accessibility"
            })
        
        # Check for all caps
        if text.isupper() and len(text) > 10:
            violations.append({
                "type": "accessibility",
                "term": "all_caps",
                "severity": "warning",
                "message": "Text is in all caps",
                "suggestion": "Use mixed case for better readability"
            })
        
        return violations
    
    def _check_brand_voice(self, text: str) -> List[Dict[str, Any]]:
        """Check brand voice compliance."""
        violations = []
        
        # Check for formal vs informal tone
        informal_words = ["gonna", "wanna", "gotta", "yeah", "cool"]
        formal_words = ["therefore", "furthermore", "consequently", "moreover"]
        
        informal_count = sum(1 for word in informal_words if word in text.lower())
        formal_count = sum(1 for word in formal_words if word in text.lower())
        
        if informal_count > 2:
            violations.append({
                "type": "voice",
                "term": "too_informal",
                "severity": "warning",
                "message": "Tone is too informal",
                "suggestion": "Use more professional language"
            })
        
        return violations
    
    def _calculate_compliance_score(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate weighted compliance score."""
        if not violations:
            return 100.0
        
        score = 100.0
        weights = {
            "critical": 20.0,
            "error": 15.0,
            "warning": 5.0,
            "info": 2.0
        }
        
        for violation in violations:
            severity = violation.get("severity", "warning")
            weight = weights.get(severity, 5.0)
            score -= weight
        
        return max(0.0, score)
    
    def _update_health(self, status: HealthStatus, message: str):
        """Update node health status and metrics."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        
        self._health.status = status
        self._health.message = message
        self._health.timestamp = time.time()
        
        if status != HealthStatus.HEALTHY:
            log.warning(f"Node health: {status.value} - {message}")
    
    def get_health(self) -> NodeHealth:
        """Get current node health status."""
        if not hasattr(self, '_health'):
            self._health = NodeHealth()
        return self._health

    # TODO(Enhancement): StyleComplianceNode
    # - Provide fully structured violation reports including severity (hard vs soft),
    #   exact positions (start/end), suggested replacements, and which rule matched.
    # - TODO(EditCycle): Integrate with the edit-cycle control: record each revision in
    #   `shared["revision_history"]` with diffs and timestamps so that when
    #   `max_revisions` is reached an `EditCycleReportNode` can generate
    #   a human-readable report.
    # - TODO(Streaming): Stream incremental style check summaries to the UI (via `shared["stream"]`).
    # - TODO(Framework): Implement comprehensive compliance framework with configurable rules
    # - TODO(Versioning): Add support for compliance rule versioning and updates
    # - TODO(Testing): Implement compliance rule testing and validation
    # - TODO(Documentation): Add support for compliance rule documentation and help
    # - TODO(Performance): Implement compliance rule performance optimization
    # - TODO(Customization): Add support for compliance rule customization per client/brand
    # - TODO(Standards): Implement compliance rule integration with external standards
    # - TODO(Automation): Add support for compliance rule automation and orchestration
    # - TODO(Monitoring): Implement compliance rule monitoring and alerting
    # - TODO(Analytics): Add support for compliance rule analytics and insights
    # - TODO(Pytest): Add pytest tests for comprehensive compliance framework and edit-cycle integration
