# flow.py
"""Flow orchestration for the Virtual PR Firm.

This module implements the complete flow orchestration for content generation,
style enforcement, and validation using the PocketFlow architecture.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pocketflow import Flow, BatchFlow  # type: ignore
from nodes import (
    EngagementManagerNode,
    BrandBibleIngestNode,
    VoiceAlignmentNode,
    PlatformFormattingNode,
    ContentCraftsmanNode,
    StyleEditorNode,
    StyleComplianceNode,
    AgencyDirectorNode,
)

log = logging.getLogger(__name__)


@dataclass
class FlowConfig:
    """Configuration for flow execution parameters."""
    
    # Node retry configurations
    ENGAGEMENT_RETRIES: int = 2
    BRAND_BIBLE_RETRIES: int = 2
    VOICE_ALIGNMENT_RETRIES: int = 2
    CONTENT_CRAFTSMAN_RETRIES: int = 3
    CONTENT_CRAFTSMAN_WAIT: int = 2
    STYLE_EDITOR_RETRIES: int = 3
    STYLE_EDITOR_WAIT: int = 1
    STYLE_COMPLIANCE_RETRIES: int = 2
    PLATFORM_FORMATTING_RETRIES: int = 2
    
    # Flow execution settings
    MAX_VALIDATION_ATTEMPTS: int = 5
    FLOW_TIMEOUT: int = 300  # seconds
    ENABLE_METRICS: bool = True
    
    @classmethod
    def from_env(cls) -> 'FlowConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        import os
        
        if os.getenv('PR_FIRM_ENGAGEMENT_RETRIES'):
            config.ENGAGEMENT_RETRIES = int(os.getenv('PR_FIRM_ENGAGEMENT_RETRIES'))
        
        if os.getenv('PR_FIRM_CONTENT_CRAFTSMAN_RETRIES'):
            config.CONTENT_CRAFTSMAN_RETRIES = int(os.getenv('PR_FIRM_CONTENT_CRAFTSMAN_RETRIES'))
        
        if os.getenv('PR_FIRM_MAX_VALIDATION_ATTEMPTS'):
            config.MAX_VALIDATION_ATTEMPTS = int(os.getenv('PR_FIRM_MAX_VALIDATION_ATTEMPTS'))
        
        if os.getenv('PR_FIRM_FLOW_TIMEOUT'):
            config.FLOW_TIMEOUT = int(os.getenv('PR_FIRM_FLOW_TIMEOUT'))
        
        return config


class FlowValidationError(Exception):
    """Exception raised when flow validation fails."""
    pass


class FlowExecutionError(Exception):
    """Exception raised when flow execution fails."""
    pass


def validate_flow_structure(flow: Flow) -> None:
    """Validate that the flow structure is correct.
    
    Args:
        flow: The flow to validate
        
    Raises:
        FlowValidationError: If the flow structure is invalid
    """
    if not flow:
        raise FlowValidationError("Flow cannot be None")
    
    if not hasattr(flow, 'start'):
        raise FlowValidationError("Flow must have a start node")
    
    if not flow.start:
        raise FlowValidationError("Flow start node cannot be None")


def create_main_flow(config: Optional[FlowConfig] = None) -> Flow:
    """Create the main content generation flow.
    
    This function wires together all nodes in the Virtual PR Firm pipeline,
    implementing the complete workflow from input collection to final output.
    
    Args:
        config: Optional configuration for flow parameters
        
    Returns:
        Flow: The complete orchestrated flow ready for execution
        
    Raises:
        FlowValidationError: If flow construction fails
    """
    if config is None:
        config = FlowConfig.from_env()
    
    try:
        # Initialize all nodes with configured retry parameters
        engagement_manager = EngagementManagerNode(max_retries=config.ENGAGEMENT_RETRIES)
        brand_bible_ingest = BrandBibleIngestNode(max_retries=config.BRAND_BIBLE_RETRIES)
        voice_alignment = VoiceAlignmentNode(max_retries=config.VOICE_ALIGNMENT_RETRIES)
        content_craftsman = ContentCraftsmanNode(
            max_retries=config.CONTENT_CRAFTSMAN_RETRIES, 
            wait=config.CONTENT_CRAFTSMAN_WAIT
        )
        style_editor = StyleEditorNode(
            max_retries=config.STYLE_EDITOR_RETRIES, 
            wait=config.STYLE_EDITOR_WAIT
        )
        style_compliance = StyleComplianceNode(max_retries=config.STYLE_COMPLIANCE_RETRIES)
        agency_director = AgencyDirectorNode()
        
        # Wire the main pipeline
        engagement_manager >> brand_bible_ingest
        brand_bible_ingest >> voice_alignment
        voice_alignment >> create_platform_formatting_flow(config)
        
        # Connect formatting flow to content generation
        formatting_flow = create_platform_formatting_flow(config)
        formatting_flow >> content_craftsman
        content_craftsman >> style_editor
        
        # Create validation loop with maximum attempts
        style_editor >> style_compliance
        style_compliance - "pass" >> agency_director
        style_compliance - "revise" >> style_editor  # Loop back for revisions
        style_compliance - "max_revisions" >> agency_director  # Exit after max attempts
        
        # Create the main flow starting from engagement manager
        main_flow = Flow(start=engagement_manager)
        
        # Validate the constructed flow
        validate_flow_structure(main_flow)
        
        log.info("Main flow created successfully with %d validation attempts limit", 
                config.MAX_VALIDATION_ATTEMPTS)
        
        return main_flow
        
    except Exception as e:
        log.error("Failed to create main flow: %s", e)
        raise FlowValidationError(f"Flow construction failed: {e}")


def create_platform_formatting_flow(config: Optional[FlowConfig] = None) -> BatchFlow:
    """Create a batch flow for platform-specific formatting guidelines.
    
    This function creates a BatchFlow that processes each platform independently,
    generating platform-specific formatting guidelines for content creation.
    
    Args:
        config: Optional configuration for flow parameters
        
    Returns:
        BatchFlow: A batch flow that processes multiple platforms
        
    Raises:
        FlowValidationError: If flow construction fails
    """
    if config is None:
        config = FlowConfig.from_env()
    
    class PlatformFormattingBatchFlow(BatchFlow):
        """Batch flow for processing multiple platforms."""
        
        def prep(self, shared: Dict[str, Any]) -> List[Dict[str, str]]:
            """Prepare platform list for batch processing.
            
            Args:
                shared: The shared state dictionary
                
            Returns:
                List of parameter dictionaries, one for each platform
                
            Raises:
                ValueError: If platform configuration is invalid
            """
            # Validate the structure of shared["task_requirements"]
            task_requirements = shared.get("task_requirements")
            if not task_requirements or not isinstance(task_requirements, dict):
                raise ValueError("shared['task_requirements'] must be a dict")
            
            platforms = task_requirements.get("platforms", [])
            if not platforms:
                # Default to common platforms if none specified
                platforms = ["twitter", "linkedin"]
                log.warning("No platforms specified, using defaults: %s", platforms)
            
            # Validate platform list
            if not isinstance(platforms, list):
                raise ValueError("task_requirements['platforms'] must be a list")
            
            if not platforms:
                raise ValueError("At least one platform must be specified")
            
            # Validate each platform
            supported_platforms = [
                "twitter", "linkedin", "facebook", "instagram", 
                "tiktok", "youtube", "medium", "blog"
            ]
            
            invalid_platforms = [p for p in platforms if p not in supported_platforms]
            if invalid_platforms:
                log.warning("Unsupported platforms detected: %s", invalid_platforms)
                # Filter out unsupported platforms instead of failing
                platforms = [p for p in platforms if p in supported_platforms]
            
            if not platforms:
                raise ValueError("No valid platforms specified")
            
            # Create parameter dict for each platform
            platform_params = [{"platform": platform} for platform in platforms]
            
            log.info("Prepared %d platforms for formatting: %s", len(platforms), platforms)
            return platform_params
    
    try:
        # Create the formatting node that will process each platform
        platform_formatter = PlatformFormattingNode(max_retries=config.PLATFORM_FORMATTING_RETRIES)
        
        # Create the batch flow
        batch_flow = PlatformFormattingBatchFlow(start=platform_formatter)
        
        log.info("Platform formatting batch flow created successfully")
        return batch_flow
        
    except Exception as e:
        log.error("Failed to create platform formatting flow: %s", e)
        raise FlowValidationError(f"Platform formatting flow construction failed: {e}")


def create_validation_flow(config: Optional[FlowConfig] = None) -> Flow:
    """Create the validation and compliance checking flow.
    
    This function creates a flow that validates content against style guidelines,
    checks for violations, and manages the revision loop.
    
    Args:
        config: Optional configuration for flow parameters
        
    Returns:
        Flow: The validation flow with revision loop
        
    Raises:
        FlowValidationError: If flow construction fails
    """
    if config is None:
        config = FlowConfig.from_env()
    
    try:
        # Create validation nodes
        style_editor = StyleEditorNode(
            max_retries=config.STYLE_EDITOR_RETRIES, 
            wait=config.STYLE_EDITOR_WAIT
        )
        style_compliance = StyleComplianceNode(max_retries=config.STYLE_COMPLIANCE_RETRIES)
        
        # Wire the validation loop
        style_editor >> style_compliance
        style_compliance - "revise" >> style_editor  # Loop back for revisions
        
        # Create the validation flow
        validation_flow = Flow(start=style_editor)
        
        log.info("Validation flow created with %d max validation attempts", 
                config.MAX_VALIDATION_ATTEMPTS)
        
        return validation_flow
        
    except Exception as e:
        log.error("Failed to create validation flow: %s", e)
        raise FlowValidationError(f"Validation flow construction failed: {e}")


def create_feedback_flow() -> Flow:
    """Create an interactive feedback flow for user refinements.
    
    This function creates a flow that handles user feedback, allowing for
    iterative content refinement based on user input.
    
    Returns:
        Flow: The feedback handling flow
        
    Raises:
        FlowValidationError: If flow construction fails
    """
    try:
        # Import feedback nodes if they exist
        try:
            from nodes import FeedbackRouterNode, SentenceEditorNode, VersionManagerNode
            
            feedback_router = FeedbackRouterNode()
            sentence_editor = SentenceEditorNode()
            version_manager = VersionManagerNode()
            
            # Wire feedback routing
            feedback_router - "sentence_edit" >> sentence_editor
            feedback_router - "rollback" >> version_manager
            feedback_router - "finalize" >> AgencyDirectorNode()
            
            # Loop back for continued editing
            sentence_editor >> feedback_router
            version_manager >> feedback_router
            
            feedback_flow = Flow(start=feedback_router)
            
            log.info("Feedback flow created successfully")
            return feedback_flow
            
        except ImportError:
            log.warning("Feedback nodes not available, using simple flow")
            
            # Fallback to simple flow without feedback
            feedback_flow = Flow(start=AgencyDirectorNode())
            
            log.info("Fallback feedback flow created")
            return feedback_flow
            
    except Exception as e:
        log.error("Failed to create feedback flow: %s", e)
        raise FlowValidationError(f"Feedback flow construction failed: {e}")


def create_streaming_flow() -> Flow:
    """Create a flow with streaming capabilities for real-time updates.
    
    This function creates a flow that supports real-time streaming of
    progress updates and intermediate results.
    
    Returns:
        Flow: The streaming-enabled flow
        
    Raises:
        FlowValidationError: If flow construction fails
    """
    try:
        # For now, return the main flow with streaming support
        # In the future, this could be enhanced with dedicated streaming nodes
        config = FlowConfig.from_env()
        main_flow = create_main_flow(config)
        
        log.info("Streaming flow created successfully")
        return main_flow
        
    except Exception as e:
        log.error("Failed to create streaming flow: %s", e)
        raise FlowValidationError(f"Streaming flow construction failed: {e}")


def execute_flow_with_monitoring(flow: Flow, shared: Dict[str, Any], 
                                config: Optional[FlowConfig] = None) -> Dict[str, Any]:
    """Execute a flow with monitoring and error handling.
    
    Args:
        flow: The flow to execute
        shared: The shared state dictionary
        config: Optional configuration for execution parameters
        
    Returns:
        Dict[str, Any]: The updated shared state
        
    Raises:
        FlowExecutionError: If flow execution fails
    """
    if config is None:
        config = FlowConfig.from_env()
    
    start_time = time.time()
    
    try:
        # Validate flow before execution
        validate_flow_structure(flow)
        
        # Execute the flow
        log.info("Starting flow execution")
        flow.run(shared)
        
        execution_time = time.time() - start_time
        log.info("Flow execution completed in %.2f seconds", execution_time)
        
        # Check for timeout
        if execution_time > config.FLOW_TIMEOUT:
            log.warning("Flow execution exceeded timeout of %d seconds", config.FLOW_TIMEOUT)
        
        # Collect metrics if enabled
        if config.ENABLE_METRICS:
            shared.setdefault("metrics", {})
            shared["metrics"]["execution_time"] = execution_time
            shared["metrics"]["timestamp"] = time.time()
        
        return shared
        
    except Exception as e:
        execution_time = time.time() - start_time
        log.error("Flow execution failed after %.2f seconds: %s", execution_time, e)
        
        # Add error information to shared state
        shared.setdefault("errors", [])
        shared["errors"].append({
            "error": str(e),
            "execution_time": execution_time,
            "timestamp": time.time()
        })
        
        raise FlowExecutionError(f"Flow execution failed: {e}")


def get_flow_metrics(shared: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from the shared state.
    
    Args:
        shared: The shared state dictionary
        
    Returns:
        Dict[str, Any]: Flow execution metrics
    """
    metrics = shared.get("metrics", {})
    errors = shared.get("errors", [])
    
    return {
        "execution_time": metrics.get("execution_time"),
        "timestamp": metrics.get("timestamp"),
        "error_count": len(errors),
        "last_error": errors[-1] if errors else None,
        "content_pieces_count": len(shared.get("content_pieces", {})),
        "platforms_processed": len(shared.get("task_requirements", {}).get("platforms", []))
    }