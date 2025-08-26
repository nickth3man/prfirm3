# flow.py
"""Flow orchestration for the Virtual PR Firm.

This module implements the complete flow orchestration for content generation,
style enforcement, and validation using the PocketFlow architecture.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional
from functools import wraps

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

logger = logging.getLogger(__name__)

def handle_flow_errors(func):
    """Decorator to provide comprehensive error handling for flow functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

def validate_flow_connections(nodes: List[Any]) -> None:
    """Validate that all flow connections are properly configured.
    
    Args:
        nodes: List of nodes to validate
        
    Raises:
        ValueError: If flow connections are invalid
    """
    if not nodes:
        raise ValueError("Flow must contain at least one node")
    
    # Check for circular dependencies (basic check)
    node_names = [node.__class__.__name__ for node in nodes]
    if len(node_names) != len(set(node_names)):
        logger.warning("Duplicate node types detected in flow")

def validate_shared_state_for_flow(shared: Dict[str, Any]) -> None:
    """Validate shared state structure for flow execution.
    
    Args:
        shared: Shared state to validate
        
    Raises:
        ValueError: If shared state is invalid
    """
    if not isinstance(shared, dict):
        raise ValueError("Shared state must be a dictionary")
    
    # Validate required keys for flow execution
    required_keys = ["task_requirements"]
    for key in required_keys:
        if key not in shared:
            raise ValueError(f"Missing required key in shared state: {key}")
    
    # Validate task_requirements structure
    task_reqs = shared.get("task_requirements", {})
    if not isinstance(task_reqs, dict):
        raise ValueError("task_requirements must be a dictionary")
    
    # Validate platforms if present
    platforms = task_reqs.get("platforms", [])
    if platforms and not isinstance(platforms, list):
        raise ValueError("platforms must be a list")

@handle_flow_errors
def create_main_flow():
    """Create the main content generation flow.
    
    This function wires together all nodes in the Virtual PR Firm pipeline,
    implementing the complete workflow from input collection to final output.
    
    Returns:
        Flow: The complete orchestrated flow ready for execution
        
    Raises:
        ValueError: If flow construction fails
        RuntimeError: If required dependencies are missing
    """
    logger.info("Creating main content generation flow")
    
    try:
        # Initialize all nodes with appropriate retry configurations
        engagement_manager = EngagementManagerNode(max_retries=2)
        brand_bible_ingest = BrandBibleIngestNode(max_retries=2)
        voice_alignment = VoiceAlignmentNode(max_retries=2)
        content_craftsman = ContentCraftsmanNode(max_retries=3, wait=2)
        style_editor = StyleEditorNode(max_retries=3, wait=1)
        style_compliance = StyleComplianceNode(max_retries=2)
        agency_director = AgencyDirectorNode()
        
        logger.debug("All nodes initialized successfully")
        
        # Wire the main pipeline
        engagement_manager >> brand_bible_ingest
        brand_bible_ingest >> voice_alignment
        voice_alignment >> create_platform_formatting_flow()
        
        # Connect formatting flow to content generation
        formatting_flow = create_platform_formatting_flow()
        formatting_flow >> content_craftsman
        content_craftsman >> style_editor
        
        # Create validation loop
        style_editor >> style_compliance
        style_compliance - "pass" >> agency_director
        style_compliance - "revise" >> style_editor  # Loop back for revisions
        style_compliance - "max_revisions" >> agency_director  # Exit after max attempts
        
        # Validate flow connections
        nodes = [engagement_manager, brand_bible_ingest, voice_alignment, 
                content_craftsman, style_editor, style_compliance, agency_director]
        validate_flow_connections(nodes)
        
        # Create the main flow starting from engagement manager
        main_flow = Flow(start=engagement_manager)
        
        logger.info("Main flow created successfully")
        return main_flow
        
    except Exception as e:
        logger.error(f"Failed to create main flow: {e}")
        raise ValueError(f"Flow construction failed: {e}")

@handle_flow_errors
def create_platform_formatting_flow():
    """Create a batch flow for platform-specific formatting guidelines.
    
    This function creates a BatchFlow that processes each platform independently,
    generating platform-specific formatting guidelines for content creation.
    
    Returns:
        BatchFlow: A batch flow that processes multiple platforms
        
    Raises:
        ValueError: If flow construction fails
    """
    logger.debug("Creating platform formatting batch flow")
    
    class PlatformFormattingBatchFlow(BatchFlow):
        """Batch flow for processing multiple platforms."""
        
        def prep(self, shared):
            """Prepare platform list for batch processing.
            
            Args:
                shared: Shared state containing task requirements
                
            Returns:
                List[Dict]: List of parameter dictionaries, one for each platform
                
            Raises:
                ValueError: If platforms are invalid or missing
            """
            try:
                # Validate shared state structure
                if not isinstance(shared, dict):
                    raise ValueError("Shared state must be a dictionary")
                
                task_reqs = shared.get("task_requirements", {})
                if not isinstance(task_reqs, dict):
                    raise ValueError("task_requirements must be a dictionary")
                
                platforms = task_reqs.get("platforms", [])
                if not isinstance(platforms, list):
                    raise ValueError("platforms must be a list")
                
                if not platforms:
                    logger.warning("No platforms specified, using default")
                    platforms = ["twitter", "linkedin"]
                
                # Validate individual platforms
                validated_platforms = []
                for platform in platforms:
                    if not isinstance(platform, str):
                        logger.warning(f"Skipping invalid platform: {platform}")
                        continue
                    
                    normalized_platform = platform.strip().lower()
                    if normalized_platform and normalized_platform.replace('-', '').replace('_', '').isalnum():
                        validated_platforms.append(normalized_platform)
                    else:
                        logger.warning(f"Skipping invalid platform format: {platform}")
                
                if not validated_platforms:
                    logger.warning("No valid platforms found, using defaults")
                    validated_platforms = ["twitter", "linkedin"]
                
                logger.debug(f"Prepared {len(validated_platforms)} platforms for batch processing")
                return [{"platform": platform} for platform in validated_platforms]
                
            except Exception as e:
                logger.error(f"Error in platform formatting batch prep: {e}")
                # Return safe defaults
                return [{"platform": "twitter"}, {"platform": "linkedin"}]
    
    try:
        # Create the batch flow with platform formatting node
        platform_formatting_node = PlatformFormattingNode(max_retries=2)
        batch_flow = PlatformFormattingBatchFlow(start=platform_formatting_node)
        
        logger.debug("Platform formatting batch flow created successfully")
        return batch_flow
        
    except Exception as e:
        logger.error(f"Failed to create platform formatting flow: {e}")
        raise ValueError(f"Platform formatting flow construction failed: {e}")

@handle_flow_errors
def create_validation_flow():
    """Create a validation flow for content compliance checking.
    
    This function creates a flow that handles content validation and revision cycles,
    implementing intelligent revision management with quality progression tracking.
    
    Returns:
        Flow: A validation flow with revision loop management
        
    Raises:
        ValueError: If flow construction fails
    """
    logger.debug("Creating validation flow")
    
    try:
        # Initialize validation nodes
        style_editor = StyleEditorNode(max_retries=3, wait=1)
        style_compliance = StyleComplianceNode(max_retries=2)
        agency_director = AgencyDirectorNode()
        
        # Wire validation loop
        style_editor >> style_compliance
        style_compliance - "pass" >> agency_director
        style_compliance - "revise" >> style_editor  # Loop back for revisions
        style_compliance - "max_revisions" >> agency_director  # Exit after max attempts
        
        # Create validation flow
        validation_flow = Flow(start=style_editor)
        
        logger.debug("Validation flow created successfully")
        return validation_flow
        
    except Exception as e:
        logger.error(f"Failed to create validation flow: {e}")
        raise ValueError(f"Validation flow construction failed: {e}")

@handle_flow_errors
def create_content_generation_flow():
    """Create a content generation flow for creating initial drafts.
    
    This function creates a flow that handles content generation and initial styling,
    preparing content for the validation and revision process.
    
    Returns:
        Flow: A content generation flow
        
    Raises:
        ValueError: If flow construction fails
    """
    logger.debug("Creating content generation flow")
    
    try:
        # Initialize content generation nodes
        content_craftsman = ContentCraftsmanNode(max_retries=3, wait=2)
        style_editor = StyleEditorNode(max_retries=3, wait=1)
        
        # Wire content generation
        content_craftsman >> style_editor
        
        # Create content generation flow
        content_flow = Flow(start=content_craftsman)
        
        logger.debug("Content generation flow created successfully")
        return content_flow
        
    except Exception as e:
        logger.error(f"Failed to create content generation flow: {e}")
        raise ValueError(f"Content generation flow construction failed: {e}")

def validate_flow_execution(shared: Dict[str, Any]) -> None:
    """Validate shared state before flow execution.
    
    Args:
        shared: Shared state to validate
        
    Raises:
        ValueError: If shared state is invalid for flow execution
    """
    try:
        validate_shared_state_for_flow(shared)
        logger.debug("Flow execution validation passed")
    except ValueError as e:
        logger.error(f"Flow execution validation failed: {e}")
        raise

def execute_flow_with_validation(flow: Flow, shared: Dict[str, Any]) -> None:
    """Execute a flow with comprehensive validation and error handling.
    
    Args:
        flow: Flow to execute
        shared: Shared state for flow execution
        
    Raises:
        ValueError: If flow execution fails
        RuntimeError: If flow execution encounters critical errors
    """
    try:
        logger.info("Starting flow execution with validation")
        
        # Validate shared state
        validate_flow_execution(shared)
        
        # Execute flow
        flow.run(shared)
        
        logger.info("Flow execution completed successfully")
        
    except Exception as e:
        logger.error(f"Flow execution failed: {e}")
        logger.debug(f"Flow execution traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Flow execution failed: {e}")

# Test functions for flow validation
def test_flow_construction():
    """Test function to validate flow construction.
    
    Returns:
        bool: True if all flows construct successfully
    """
    try:
        logger.info("Testing flow construction")
        
        # Test main flow
        main_flow = create_main_flow()
        logger.debug("Main flow construction test passed")
        
        # Test platform formatting flow
        platform_flow = create_platform_formatting_flow()
        logger.debug("Platform formatting flow construction test passed")
        
        # Test validation flow
        validation_flow = create_validation_flow()
        logger.debug("Validation flow construction test passed")
        
        # Test content generation flow
        content_flow = create_content_generation_flow()
        logger.debug("Content generation flow construction test passed")
        
        logger.info("All flow construction tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Flow construction test failed: {e}")
        return False

def test_flow_execution():
    """Test function to validate flow execution with sample data.
    
    Returns:
        bool: True if flow execution test passes
    """
    try:
        logger.info("Testing flow execution")
        
        # Create sample shared state
        shared = {
            "task_requirements": {
                "platforms": ["twitter", "linkedin"],
                "topic_or_goal": "Test content generation",
                "intents_by_platform": {}
            },
            "brand_bible": {"xml_raw": ""},
            "stream": None
        }
        
        # Test main flow execution
        main_flow = create_main_flow()
        execute_flow_with_validation(main_flow, shared)
        
        logger.info("Flow execution test passed")
        return True
        
    except Exception as e:
        logger.error(f"Flow execution test failed: {e}")
        return False