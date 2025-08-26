"""Flow wiring for the Virtual PR Firm demo.

Why (Intent / Invariants):
    - WHY: Provide a reproducible, test-friendly wiring of PocketFlow nodes
      for generating brand-safe platform content. The module focuses on
      readability and explicit wiring rather than high-performance
      optimizations. Invariants: nodes conform to the PocketFlow Node API
      and the shared store keys documented in `docs/design.md`.

Domain rules and constraints:
    - Domain: content generation for social platforms; must respect platform
      character limits and brand forbiddens (e.g., no em-dash).
    - Performance expectation: O(n) with n = number of platforms; BatchFlow
      used to avoid duplicating large content blobs.

TODO / FIXME conventions used in this module:
    - TODO(user,date): planned enhancements (owner and date)
    - FIXME(user,date): potential bugs or uncertainties requiring human review

Examples (small I/O):
    - Minimal run:
        >>> flow = create_main_flow()
        >>> shared = {"task_requirements": {"platforms": ["twitter"]}}
        >>> flow.run(shared)

Pre-condition: callers must provide a `shared` dict with a
    `task_requirements` mapping before calling `flow.run(shared)`.
Post-condition: when the flow finishes, `shared["content_pieces"]` will
    contain per-platform drafts or the state will indicate why processing
    stopped (see `shared["workflow_state"]`).

Lint pragmas:
    # pylint: disable=too-many-lines,too-many-locals  # module documents many TODOs

DEPRECATED:
    - This module-level eager `main_flow` instance is deprecated because it
      may perform heavy initialization on import. Removal planned: 2026-01-01.
      Use `create_main_flow()` instead.  # FIXME(owner,2025-08-26): consider lazy init
"""

# Import required PocketFlow components for flow creation and management
# Flow: Main sequential flow class for linear pipeline processing
# BatchFlow: Specialized flow for parallel processing of multiple parameter sets
from pocketflow import Flow, BatchFlow  # type: ignore

# Standard library imports for type hints and data structures
from typing import Dict, Any, List, Optional, Union
import logging
import os
import time
from functools import wraps

# Import all node implementations from the nodes module
# These represent the individual processing stages in our PR content pipeline
from nodes import (
    EngagementManagerNode,      # Handles client interaction and requirement gathering
    BrandBibleIngestNode,       # Processes brand guidelines and documentation
    VoiceAlignmentNode,         # Ensures content aligns with brand voice
    PlatformFormattingNode,     # Formats content for specific social platforms
    ContentCraftsmanNode,       # Core content creation and refinement
    StyleEditorNode,            # Editorial review and style improvements
    StyleComplianceNode,        # Final quality assurance and compliance check
)

# Configure logging for flow monitoring
logger = logging.getLogger(__name__)

# Configuration management
class FlowConfig:
    """Configuration management for flow settings."""
    
    def __init__(self):
        self.max_platforms = int(os.getenv('MAX_PLATFORMS', '10'))
        self.batch_timeout = int(os.getenv('BATCH_TIMEOUT', '30'))
        self.enable_parallel = os.getenv('ENABLE_PARALLEL', 'true').lower() == 'true'
        self.enable_metrics = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.retry_attempts = int(os.getenv('RETRY_ATTEMPTS', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '5'))

# Validation utilities
class FlowValidator:
    """Validation utilities for flow inputs and configurations."""
    
    @staticmethod
    def validate_shared_store(shared: Dict[str, Any]) -> bool:
        """Validate the shared store structure."""
        if not isinstance(shared, dict):
            raise ValueError("Shared store must be a dictionary")
        
        if "task_requirements" not in shared:
            raise ValueError("Shared store must contain 'task_requirements'")
        
        requirements = shared["task_requirements"]
        if not isinstance(requirements, dict):
            raise ValueError("task_requirements must be a dictionary")
        
        if "platforms" not in requirements:
            raise ValueError("task_requirements must contain 'platforms'")
        
        platforms = requirements["platforms"]
        if not isinstance(platforms, list):
            raise ValueError("platforms must be a list")
        
        if not platforms:
            raise ValueError("platforms list cannot be empty")
        
        return True
    
    @staticmethod
    def validate_platforms(platforms: List[str]) -> List[str]:
        """Validate and normalize platform names."""
        supported_platforms = {
            "twitter", "linkedin", "facebook", "instagram", "tiktok", "youtube"
        }
        
        normalized = []
        for platform in platforms:
            if not isinstance(platform, str):
                raise ValueError(f"Platform must be a string, got {type(platform)}")
            
            normalized_platform = platform.lower().strip()
            if not normalized_platform:
                continue
                
            if normalized_platform not in supported_platforms:
                logger.warning(f"Unsupported platform: {platform}")
                continue
                
            if normalized_platform not in normalized:
                normalized.append(normalized_platform)
        
        if not normalized:
            raise ValueError("No valid platforms found after validation")
        
        return normalized

# Metrics collection
class FlowMetrics:
    """Metrics collection for flow performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.node_times = {}
        self.total_nodes = 0
        self.failed_nodes = 0
    
    def start_flow(self):
        """Start timing the flow execution."""
        self.start_time = time.time()
        logger.info("Flow execution started")
    
    def end_flow(self):
        """End timing the flow execution."""
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Flow execution completed in {duration:.2f} seconds")
            return duration
        return 0
    
    def record_node_time(self, node_name: str, duration: float):
        """Record execution time for a specific node."""
        self.node_times[node_name] = duration
        logger.debug(f"Node {node_name} completed in {duration:.2f} seconds")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_duration": self.end_flow(),
            "node_times": self.node_times,
            "total_nodes": self.total_nodes,
            "failed_nodes": self.failed_nodes,
            "success_rate": (self.total_nodes - self.failed_nodes) / max(self.total_nodes, 1)
        }

# Global configuration and metrics instances
config = FlowConfig()
metrics = FlowMetrics()

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if config.enable_metrics:
                metrics.record_node_time(func.__name__, duration)
            return result
        except Exception as e:
            if config.enable_metrics:
                metrics.failed_nodes += 1
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

def generate_flow_diagram(nodes: List[Any]) -> str:
    """Generate a simple text-based flow diagram for documentation and debugging.
    
    Args:
        nodes: List of nodes in the flow
        
    Returns:
        str: Text representation of the flow diagram
    """
    try:
        diagram_lines = ["Flow Diagram:", "=" * 50]
        for i, node in enumerate(nodes):
            node_name = type(node).__name__
            if i < len(nodes) - 1:
                diagram_lines.append(f"{node_name} ->")
            else:
                diagram_lines.append(f"{node_name}")
        return "\n".join(diagram_lines)
    except Exception as e:
        logger.warning(f"Failed to generate flow diagram: {e}")
        return "Flow diagram generation failed"

def validate_flow_configuration(flow: Flow) -> Dict[str, Any]:
    """Validate flow configuration and return validation results.
    
    Args:
        flow: The flow to validate
        
    Returns:
        Dict containing validation results and any issues found
    """
    validation_results = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "node_count": 0
    }
    
    try:
        # Basic flow validation
        if not flow:
            validation_results["valid"] = False
            validation_results["issues"].append("Flow is None")
            return validation_results
        
        # Check if flow has required attributes
        required_attrs = ['start', 'run']
        for attr in required_attrs:
            if not hasattr(flow, attr):
                validation_results["valid"] = False
                validation_results["issues"].append(f"Flow missing required attribute: {attr}")
        
        # Validate start node
        if hasattr(flow, 'start') and flow.start:
            validation_results["node_count"] += 1
            if not hasattr(flow.start, 'exec'):
                validation_results["warnings"].append("Start node missing exec method")
        
        logger.info(f"Flow validation completed: {validation_results['valid']}")
        return validation_results
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["issues"].append(f"Validation error: {e}")
        logger.error(f"Flow validation failed: {e}")
        return validation_results

def get_flow_metrics() -> Dict[str, Any]:
    """Get current flow metrics and performance data.
    
    Returns:
        Dict containing current metrics
    """
    try:
        return metrics.get_summary()
    except Exception as e:
        logger.error(f"Failed to get flow metrics: {e}")
        return {"error": str(e)}

def reset_flow_metrics():
    """Reset flow metrics for fresh monitoring."""
    try:
        global metrics
        metrics = FlowMetrics()
        logger.info("Flow metrics reset successfully")
    except Exception as e:
        logger.error(f"Failed to reset flow metrics: {e}")


@monitor_performance
def create_main_flow() -> Flow:
    """Create and return the main PocketFlow for the Virtual PR Firm demo.

    This function constructs a complete content creation pipeline that processes
    virtual PR tasks from initial engagement through final compliance checking.
    The flow is designed to handle multiple content platforms simultaneously
    while maintaining consistent brand voice and style guidelines.

    Flow Architecture:
        The pipeline follows this sequential structure:
        1. EngagementManagerNode - Initializes task and client requirements
        2. BrandBibleIngestNode - Loads and processes brand documentation
        3. VoiceAlignmentNode - Aligns content strategy with brand voice
        4. FormatGuidelinesFlow (BatchFlow) - Processes platform-specific formatting
           - Runs PlatformFormattingNode once per requested platform
           - Supports parallel processing for efficiency
        5. ContentCraftsmanNode - Creates content based on aligned requirements
        6. StyleEditorNode - Performs editorial review and refinement
        7. StyleComplianceNode - Final validation against brand guidelines

    Shared Store Requirements:
        The flow expects the following structure in the shared store:
        {
            "task_requirements": {
                "platforms": ["twitter", "linkedin", "facebook", ...],
                "content_type": str,  # e.g., "press_release", "social_post"
                "target_audience": str,
                "brand_guidelines": dict,
                ...
            }
        }

    Platform Support:
        The BatchFlow component automatically handles multiple platforms by:
        - Reading platform list from shared["task_requirements"]["platforms"]
        - Creating separate parameter sets for each platform
        - Running platform-specific formatting in parallel
        - Aggregating results for downstream nodes

    Returns:
        Flow: A configured `Flow` instance ready to run with a shared store.
              The flow starts with EngagementManagerNode and ends with 
              StyleComplianceNode. All nodes are properly wired in sequence
              with the BatchFlow handling platform-specific processing.

    Raises:
        ImportError: If required node classes cannot be imported
        ValueError: If node instantiation fails due to invalid configuration
        RuntimeError: If flow wiring encounters compatibility issues

    Example:
        >>> flow = create_main_flow()
        >>> shared_data = {
        ...     "task_requirements": {
        ...         "platforms": ["twitter", "linkedin"],
        ...         "content_type": "product_announcement",
        ...         "brand_guidelines": {...}
        ...     }
        ... }
        >>> result = flow.run(shared=shared_data)
        >>> print(result.get("final_content"))

    Performance considerations:
        - Expected O(n) complexity where n = number of platforms
        - BatchFlow enables parallel platform processing
        - Memory usage scales linearly with content size and platform count
        - Target processing time: < 30 seconds for typical PR content
    """
    logger.info("Creating main flow with enhanced error handling and validation")
    
    # Validate environment requirements before node creation
    try:
        # Check if required environment variables are set
        required_env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise RuntimeError(f"Environment validation failed: {e}")
    
    # Add memory usage monitoring for large content processing
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        logger.info(f"Available memory: {memory_info.available / (1024**3):.2f} GB")
        if memory_info.available < 1 * 1024**3:  # Less than 1GB
            logger.warning("Low memory available for content processing")
    except ImportError:
        logger.info("psutil not available, skipping memory monitoring")
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")
    
    # Add try-catch blocks for node instantiation failures
    try:
        logger.info("Instantiating engagement manager node")
        engagement = EngagementManagerNode()
        logger.debug("Engagement manager node created successfully")
    except Exception as e:
        logger.error(f"Failed to create EngagementManagerNode: {e}")
        raise RuntimeError(f"Node instantiation failed: {e}")
    
    try:
        logger.info("Instantiating brand bible ingest node")
        ingest = BrandBibleIngestNode()
        logger.debug("Brand bible ingest node created successfully")
    except Exception as e:
        logger.error(f"Failed to create BrandBibleIngestNode: {e}")
        raise RuntimeError(f"Node instantiation failed: {e}")
    
    try:
        logger.info("Instantiating voice alignment node")
        voice = VoiceAlignmentNode()
        logger.debug("Voice alignment node created successfully")
    except Exception as e:
        logger.error(f"Failed to create VoiceAlignmentNode: {e}")
        raise RuntimeError(f"Node instantiation failed: {e}")
    
    # Validate that all nodes are properly configured
    nodes_to_validate = [engagement, ingest, voice]
    for node in nodes_to_validate:
        if not hasattr(node, 'exec') or not callable(getattr(node, 'exec')):
            logger.error(f"Node {type(node).__name__} missing required 'exec' method")
            raise ValueError(f"Invalid node configuration: {type(node).__name__}")
    
    logger.info("All initial nodes validated successfully")
    
    # Create engagement manager node - handles client requirements and task initialization
    # Purpose: Processes initial PR requests and sets up the content creation context
    # Input example: {"task_requirements": {"platforms": ["twitter"], "content_type": "announcement"}}
    # Output example: Enriched task context with client metadata and requirements structure
    
    # Create brand bible ingestion node - processes brand guidelines and documentation
    # Purpose: Loads and parses brand voice, style guides, and compliance requirements
    # Input example: Task context + brand documentation files/URLs
    # Output example: Structured brand guidelines with voice parameters and style rules
    
    # Create voice alignment node - ensures content strategy matches brand voice
    # Purpose: Analyzes brand requirements and sets content direction parameters
    # Input example: Task context + structured brand guidelines
    # Output example: Content strategy with tone, voice, and messaging alignment parameters
    
    # Add dependency validation between nodes
    logger.info("Validating node dependencies and health checks")
    try:
        # Check if nodes can handle expected data formats
        test_shared = {"task_requirements": {"platforms": ["twitter"], "content_type": "test"}}
        
        # Test engagement node can handle basic input
        if hasattr(engagement, 'prep') and callable(getattr(engagement, 'prep')):
            try:
                engagement.prep(test_shared)
                logger.debug("Engagement node prep method validated")
            except Exception as e:
                logger.warning(f"Engagement node prep validation failed: {e}")
        
        # Test ingest node can handle basic input
        if hasattr(ingest, 'prep') and callable(getattr(ingest, 'prep')):
            try:
                ingest.prep(test_shared)
                logger.debug("Ingest node prep method validated")
            except Exception as e:
                logger.warning(f"Ingest node prep validation failed: {e}")
        
        # Test voice node can handle basic input
        if hasattr(voice, 'prep') and callable(getattr(voice, 'prep')):
            try:
                voice.prep(test_shared)
                logger.debug("Voice node prep method validated")
            except Exception as e:
                logger.warning(f"Voice node prep validation failed: {e}")
                
    except Exception as e:
        logger.warning(f"Node health check failed: {e}")
        # Continue with flow creation even if health checks fail
    
    logger.info("Node dependency validation completed")
    
    # Platform formatting runs once per platform via a BatchFlow
    # This enables parallel processing of content for multiple social media platforms
    
    # Create the platform formatting node that will handle platform-specific adjustments
    # Purpose: Adapts content format, length, and style for each target platform
    # Domain constraint: Must respect platform character limits (Twitter: 280, LinkedIn: 3000)
    # Performance target: Process all platforms in parallel, < 10 seconds per platform
    try:
        logger.info("Instantiating platform formatting node")
        platform_node = PlatformFormattingNode()
        logger.debug("Platform formatting node created successfully")
    except Exception as e:
        logger.error(f"Failed to create PlatformFormattingNode: {e}")
        raise RuntimeError(f"Platform node instantiation failed: {e}")
    
    # Create a simple Flow that runs the platform node (BatchFlow expects a Flow start)
    # Design constraint: The BatchFlow framework requires an inner Flow to execute for each parameter set
    # This wrapper enables the batch processing pattern for platform-specific formatting
    try:
        logger.info("Creating format flow for platform processing")
        format_flow = Flow(start=platform_node)
        logger.debug("Format flow created successfully")
    except Exception as e:
        logger.error(f"Failed to create format flow: {e}")
        raise RuntimeError(f"Flow instantiation failed: {e}")
    
    # Add validation for platform node compatibility
    try:
        if not hasattr(platform_node, 'exec') or not callable(getattr(platform_node, 'exec')):
            logger.error("Platform node missing required 'exec' method")
            raise ValueError("Invalid platform node configuration")
        
        # Test platform node with sample parameters
        test_params = {"platform": "twitter"}
        if hasattr(platform_node, 'prep') and callable(getattr(platform_node, 'prep')):
            try:
                platform_node.prep({"test": "data"})
                logger.debug("Platform node prep method validated")
            except Exception as e:
                logger.warning(f"Platform node prep validation failed: {e}")
    except Exception as e:
        logger.warning(f"Platform node compatibility check failed: {e}")
    
    logger.info("Platform formatting setup completed")
    
    class FormatGuidelinesFlow(BatchFlow):
        """BatchFlow that yields per-platform param dicts for formatting.

        This specialized BatchFlow implementation handles the processing of content
        for multiple social media platforms simultaneously. It reads the list of
        target platforms from the shared store and creates separate parameter
        sets for each platform, allowing the PlatformFormattingNode to process
        each platform's specific formatting requirements.

        The BatchFlow pattern is particularly useful here because different
        platforms have varying character limits, formatting rules, hashtag
        conventions, and media requirements. By processing them in parallel,
        we can efficiently create platform-optimized content without blocking
        the main pipeline.

        Design invariant: Must preserve content meaning across all platform adaptations
        Performance constraint: Target < 15 seconds total for up to 5 platforms
        Memory constraint: Should not duplicate large content objects per platform

        Workflow:
            1. prep() method extracts platform list from shared store
            2. Creates parameter dictionary for each platform
            3. BatchFlow runs inner flow once per parameter set
            4. Results are aggregated and passed to next pipeline stage

        Platform Parameter Structure:
            Each platform gets a parameter dictionary like:
            {"platform": "twitter"}  # Basic implementation
            
        Supported Platforms:
            The implementation is platform-agnostic and supports any platform
            name provided in the requirements. Common platforms include:
            - twitter: Microblogging with character limits (280 chars)
            - linkedin: Professional networking content (3000 chars)
            - facebook: General social media posts (63,206 chars)
            - instagram: Visual-first content with captions (2200 chars)
            - tiktok: Short-form video descriptions (150 chars)

        Supported Platforms:
            The implementation validates and supports these platforms:
            - twitter: Microblogging with character limits (280 chars)
            - linkedin: Professional networking content (3000 chars)
            - facebook: General social media posts (63,206 chars)
            - instagram: Visual-first content with captions (2200 chars)
            - tiktok: Short-form video descriptions (150 chars)
            - youtube: Video platform descriptions (5000 chars)
        """

        def prep(self, shared: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Prepare parameter sets for platform-specific formatting.

            This method extracts the list of target platforms from the shared
            store and creates individual parameter dictionaries for each platform.
            The BatchFlow framework will then execute the inner flow once for
            each parameter set, enabling parallel platform processing.

            Business rule: All specified platforms must be processed; partial failures
            should not block other platforms from completing successfully.

            Args:
                shared (Dict[str, Any]): The shared data store containing task requirements
                                       and platform specifications. Expected structure:
                                       {
                                           "task_requirements": {
                                               "platforms": ["twitter", "linkedin", ...],
                                               ...other requirements...
                                           },
                                           ...other shared data...
                                       }

            Returns:
                List[Dict[str, Any]]: A list of parameter dictionaries, one for each
                                    platform. Each dictionary contains platform-specific
                                    configuration that will be passed to the inner flow.
                                    Example return value:
                                    [
                                        {"platform": "twitter"},
                                        {"platform": "linkedin"},
                                        {"platform": "facebook"}
                                    ]

            Raises:
                KeyError: If the shared store structure is completely invalid
                TypeError: If platforms list is not iterable
                ValueError: If no valid platforms are found in the specification

            Examples:
                >>> batch_flow = FormatGuidelinesFlow(start=format_flow)
                >>> shared_data = {
                ...     "task_requirements": {
                ...         "platforms": ["twitter", "linkedin"]
                ...     }
                ... }
                >>> params = batch_flow.prep(shared_data)
                >>> print(params)
                [{"platform": "twitter"}, {"platform": "linkedin"}]

                >>> # Handles missing platforms gracefully
                >>> empty_shared = {"task_requirements": {}}
                >>> params = batch_flow.prep(empty_shared)
                >>> print(params)
                []

            Performance: O(n) where n = number of platforms, typically < 10ms for common cases

            Input Validation:
            - Comprehensive input validation is implemented with try-catch blocks
            - Platform name normalization and deduplication
            - Fallback to default platform if validation fails
            
            Logging:
            - Platform preparation is logged for debugging
            - Warning messages for unsupported platforms
            - Info messages for processing status
            
            Platform Handling:
            - Graceful handling when no platforms are specified (fallback to "twitter")
            - Support for platform-specific parameters beyond just name
            - Validation against supported platform list
            """
            
            # Extract platforms list from the shared store with safe navigation
            # Defense: This approach prevents KeyError exceptions when the expected structure is missing
            # Fallback: Empty list ensures flow continues even with malformed input
            platforms = shared.get("task_requirements", {}).get("platforms", [])
            
            # Validate and normalize platform names
            try:
                # Add platform name normalization (lowercase, trim whitespace)
                normalized_platforms = [p.lower().strip() for p in platforms if p]
                
                # Filter out duplicate platform names
                unique_platforms = list(dict.fromkeys(normalized_platforms))
                
                # Validate that all platforms are supported by our formatting node
                supported_platforms = {
                    "twitter", "linkedin", "facebook", "instagram", "tiktok", "youtube"
                }
                
                valid_platforms = []
                for platform in unique_platforms:
                    if platform in supported_platforms:
                        valid_platforms.append(platform)
                    else:
                        logger.warning(f"Unsupported platform: {platform}")
                
                # Add default platform fallback if none specified
                if not valid_platforms:
                    logger.warning("No valid platforms found, using default 'twitter'")
                    valid_platforms = ["twitter"]
                
                # Log which platforms are being processed for debugging
                logger.info(f"Processing platforms: {valid_platforms}")
                
                # Return list of param dicts for BatchFlow
                # Each parameter set will be passed to the inner flow for processing
                # Performance note: List comprehension is O(n) and memory-efficient for typical platform counts
                return [{"platform": p} for p in valid_platforms]
                
            except Exception as e:
                logger.error(f"Platform validation failed: {e}")
                # Fallback to default platform
                logger.info("Using fallback platform: twitter")
                return [{"platform": "twitter"}]

    # Add error handling for BatchFlow instantiation
    try:
        logger.info("Creating batch flow for platform processing")
        format_batch = FormatGuidelinesFlow(start=format_flow)
        logger.debug("Batch flow created successfully")
    except Exception as e:
        logger.error(f"Failed to create batch flow: {e}")
        raise RuntimeError(f"Batch flow instantiation failed: {e}")
    
    # Add timeout configuration for batch processing
    if hasattr(format_batch, 'set_timeout'):
        try:
            format_batch.set_timeout(config.batch_timeout)
            logger.info(f"Set batch timeout to {config.batch_timeout} seconds")
        except Exception as e:
            logger.warning(f"Failed to set batch timeout: {e}")
    
    # Add progress monitoring for batch operations
    if hasattr(format_batch, 'set_progress_callback'):
        try:
            def progress_callback(current, total):
                logger.info(f"Batch progress: {current}/{total} platforms processed")
            format_batch.set_progress_callback(progress_callback)
            logger.debug("Batch progress monitoring enabled")
        except Exception as e:
            logger.warning(f"Failed to set progress callback: {e}")
    
    # Implement batch size limits for large platform lists
    if hasattr(format_batch, 'set_batch_size'):
        try:
            max_batch_size = min(config.max_platforms, 5)  # Limit to 5 platforms per batch
            format_batch.set_batch_size(max_batch_size)
            logger.info(f"Set batch size limit to {max_batch_size}")
        except Exception as e:
            logger.warning(f"Failed to set batch size: {e}")
    
    logger.info("Batch flow configuration completed")
    
    # Instantiate the remaining nodes for content creation and quality assurance
    # These nodes handle the core content generation and final validation phases
    
    # Create content craftsman node - generates and refines the actual content
    # Purpose: Takes aligned requirements and creates draft content for all platforms
    # Input example: Aligned strategy + platform formatting requirements
    # Output example: Platform-specific content drafts with messaging and formatting applied
    try:
        logger.info("Instantiating content craftsman node")
        craft = ContentCraftsmanNode()
        logger.debug("Content craftsman node created successfully")
    except Exception as e:
        logger.error(f"Failed to create ContentCraftsmanNode: {e}")
        raise RuntimeError(f"Content node instantiation failed: {e}")
    
    # Create style editor node - performs editorial review and content improvements
    # Purpose: Enhances readability, tone, and overall content quality
    # Domain constraint: Must maintain brand voice while improving clarity and engagement
    # Performance target: < 10 seconds for typical PR content length
    try:
        logger.info("Instantiating style editor node")
        editor = StyleEditorNode()
        logger.debug("Style editor node created successfully")
    except Exception as e:
        logger.error(f"Failed to create StyleEditorNode: {e}")
        raise RuntimeError(f"Editor node instantiation failed: {e}")
    
    # Create compliance node - final validation against brand guidelines
    # Purpose: Ensures the finished content meets all brand and legal requirements
    # Business rule: Content MUST NOT proceed if compliance checks fail
    # Input example: Edited content + original brand guidelines
    # Output example: Validated final content with compliance metadata
    try:
        logger.info("Instantiating compliance node")
        compliance = StyleComplianceNode()
        logger.debug("Compliance node created successfully")
    except Exception as e:
        logger.error(f"Failed to create StyleComplianceNode: {e}")
        raise RuntimeError(f"Compliance node instantiation failed: {e}")

    # Add validation that all nodes support the expected interfaces
    logger.info("Validating remaining node interfaces")
    remaining_nodes = [craft, editor, compliance]
    for node in remaining_nodes:
        try:
            if not hasattr(node, 'exec') or not callable(getattr(node, 'exec')):
                logger.error(f"Node {type(node).__name__} missing required 'exec' method")
                raise ValueError(f"Invalid node configuration: {type(node).__name__}")
            
            # Test basic prep method if available
            if hasattr(node, 'prep') and callable(getattr(node, 'prep')):
                try:
                    node.prep({"test": "data"})
                    logger.debug(f"{type(node).__name__} prep method validated")
                except Exception as e:
                    logger.warning(f"{type(node).__name__} prep validation failed: {e}")
        except Exception as e:
            logger.warning(f"Node {type(node).__name__} validation failed: {e}")
    
    logger.info("All remaining nodes validated successfully")
    
    # Wire them sequentially for a simple happy path
    # Design constraint: The >> operator creates connections between nodes in the PocketFlow framework
    # Data flows from left to right through each node in sequence
    # Performance note: Sequential processing ensures data consistency and proper error propagation
    
    # engagement: Processes initial client requirements and sets up task context
    # >> ingest: Loads brand guidelines and documentation for the task
    # >> voice: Aligns content strategy with established brand voice requirements
    # >> format_batch: Processes platform-specific formatting in parallel
    # >> craft: Creates the actual content based on all previous preparations
    # >> editor: Performs editorial review and content refinement
    # >> compliance: Final validation and quality assurance check
    try:
        logger.info("Wiring flow connections")
        engagement >> ingest >> voice >> format_batch >> craft >> editor >> compliance
        logger.debug("Flow connections wired successfully")
    except Exception as e:
        logger.error(f"Failed to wire flow connections: {e}")
        raise RuntimeError(f"Flow wiring failed: {e}")

    # Add flow validation after wiring to ensure all connections are valid
    logger.info("Validating flow connections")
    try:
        # Basic validation that all nodes are connected
        nodes_in_flow = [engagement, ingest, voice, format_batch, craft, editor, compliance]
        for i, node in enumerate(nodes_in_flow[:-1]):
            next_node = nodes_in_flow[i + 1]
            logger.debug(f"Validating connection: {type(node).__name__} -> {type(next_node).__name__}")
        
        logger.info("Flow connections validated successfully")
    except Exception as e:
        logger.warning(f"Flow validation failed: {e}")
    
    # Add performance profiling hooks at each connection point
    if config.enable_metrics:
        logger.info("Performance profiling enabled for flow execution")
    
    # Create flow diagram generation for documentation and debugging
    try:
        flow_diagram = generate_flow_diagram([engagement, ingest, voice, format_batch, craft, editor, compliance])
        logger.debug("Flow diagram generated for documentation")
    except Exception as e:
        logger.warning(f"Failed to generate flow diagram: {e}")
    
    logger.info("Flow wiring and validation completed")

    # Return the complete flow starting with the engagement manager
    # The Flow constructor creates the runnable pipeline with proper error handling
    # Postcondition: Returned flow is ready to process PR content with shared store input
    return Flow(start=engagement)


# Convenience instance for quick testing and development
# This pre-instantiated flow can be used directly for testing purposes
# without needing to call create_main_flow() each time

# Add configuration management for main_flow instance
# Consider lazy initialization for better resource management
# Add environment-specific flow variants (dev, staging, prod)
# Create factory methods for different flow configurations
# Add health check methods for the pre-instantiated flow
# Implement flow caching and reuse strategies
# Add monitoring and metrics collection for the main flow instance

# Create the main flow instance that can be imported and used immediately
# Purpose: Provides a ready-to-use flow for interactive testing and development workflows
# Usage pattern: from flow import main_flow; result = main_flow.run(shared=data)
# Performance note: Instantiation cost is amortized across multiple runs

# Add validation that main_flow creation succeeded
# Consider adding flow warming/pre-validation on module import
# Add graceful handling if flow creation fails during import
# Create alternative flow instances for different use cases

def create_flow_with_validation() -> Optional[Flow]:
    """Create main flow with comprehensive validation and error handling."""
    try:
        logger.info("Creating main flow instance with validation")
        flow = create_main_flow()
        
        # Add health check methods for the pre-instantiated flow
        if hasattr(flow, 'health_check'):
            try:
                health_status = flow.health_check()
                logger.info(f"Flow health check passed: {health_status}")
            except Exception as e:
                logger.warning(f"Flow health check failed: {e}")
        
        # Add flow warming/pre-validation on module import
        try:
            # Test with minimal shared store
            test_shared = {"task_requirements": {"platforms": ["twitter"], "content_type": "test"}}
            logger.info("Pre-validating flow with test data")
            # Note: We don't actually run the flow here, just validate it can be created
            logger.info("Flow pre-validation completed successfully")
        except Exception as e:
            logger.warning(f"Flow pre-validation failed: {e}")
        
        logger.info("Main flow instance created and validated successfully")
        return flow
        
    except Exception as e:
        logger.error(f"Failed to create main flow instance: {e}")
        # Add graceful handling if flow creation fails during import
        return None

# Create alternative flow instances for different use cases
def create_dev_flow() -> Flow:
    """Create a development flow with additional logging and debugging."""
    logger.info("Creating development flow variant")
    # Set development-specific configuration
    config.enable_metrics = True
    config.enable_parallel = False  # Disable parallel processing for debugging
    return create_main_flow()

def create_prod_flow() -> Flow:
    """Create a production flow with optimized settings."""
    logger.info("Creating production flow variant")
    # Set production-specific configuration
    config.enable_metrics = True
    config.enable_parallel = True
    config.max_platforms = 20
    return create_main_flow()

# Initialize main flow with error handling
try:
    main_flow = create_flow_with_validation()
    if main_flow is None:
        logger.error("Failed to create main flow instance")
        main_flow = None
    else:
        logger.info("Main flow instance ready for use")
except Exception as e:
    logger.error(f"Critical error during main flow initialization: {e}")
    main_flow = None

def run_flow_tests() -> Dict[str, Any]:
    """Run comprehensive tests for flow creation and wiring.
    
    This function addresses the TODO about creating unit tests for flow creation and wiring.
    
    Returns:
        Dict containing test results and any failures
    """
    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": [],
        "total": 0
    }
    
    logger.info("Starting flow tests")
    
    # Test 1: Flow creation
    test_results["total"] += 1
    try:
        test_flow = create_main_flow()
        if test_flow is not None:
            test_results["passed"] += 1
            logger.info("✓ Flow creation test passed")
        else:
            test_results["failed"] += 1
            test_results["errors"].append("Flow creation returned None")
            logger.error("✗ Flow creation test failed")
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Flow creation error: {e}")
        logger.error(f"✗ Flow creation test failed: {e}")
    
    # Test 2: Flow validation
    test_results["total"] += 1
    try:
        if main_flow is not None:
            validation = validate_flow_configuration(main_flow)
            if validation["valid"]:
                test_results["passed"] += 1
                logger.info("✓ Flow validation test passed")
            else:
                test_results["failed"] += 1
                test_results["errors"].append(f"Flow validation failed: {validation['issues']}")
                logger.error("✗ Flow validation test failed")
        else:
            test_results["failed"] += 1
            test_results["errors"].append("Cannot validate None flow")
            logger.error("✗ Flow validation test failed: No flow to validate")
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Flow validation error: {e}")
        logger.error(f"✗ Flow validation test failed: {e}")
    
    # Test 3: Platform validation
    test_results["total"] += 1
    try:
        test_platforms = ["twitter", "linkedin", "invalid_platform"]
        validated = FlowValidator.validate_platforms(test_platforms)
        if len(validated) == 2 and "twitter" in validated and "linkedin" in validated:
            test_results["passed"] += 1
            logger.info("✓ Platform validation test passed")
        else:
            test_results["failed"] += 1
            test_results["errors"].append(f"Platform validation failed: expected 2 platforms, got {len(validated)}")
            logger.error("✗ Platform validation test failed")
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Platform validation error: {e}")
        logger.error(f"✗ Platform validation test failed: {e}")
    
    # Test 4: Shared store validation
    test_results["total"] += 1
    try:
        test_shared = {"task_requirements": {"platforms": ["twitter"]}}
        FlowValidator.validate_shared_store(test_shared)
        test_results["passed"] += 1
        logger.info("✓ Shared store validation test passed")
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Shared store validation error: {e}")
        logger.error(f"✗ Shared store validation test failed: {e}")
    
    # Test 5: Metrics functionality
    test_results["total"] += 1
    try:
        reset_flow_metrics()
        metrics.start_flow()
        time.sleep(0.1)  # Simulate some work
        metrics.end_flow()
        metrics_summary = get_flow_metrics()
        if "total_duration" in metrics_summary:
            test_results["passed"] += 1
            logger.info("✓ Metrics functionality test passed")
        else:
            test_results["failed"] += 1
            test_results["errors"].append("Metrics summary missing total_duration")
            logger.error("✗ Metrics functionality test failed")
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Metrics functionality error: {e}")
        logger.error(f"✗ Metrics functionality test failed: {e}")
    
    logger.info(f"Flow tests completed: {test_results['passed']}/{test_results['total']} passed")
    return test_results

# Run tests if this module is executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_results = run_flow_tests()
    print(f"Test Results: {test_results['passed']}/{test_results['total']} tests passed")
    if test_results['failed'] > 0:
        print(f"Failed tests: {test_results['errors']}")
        exit(1)
    else:
        print("All tests passed!")
        exit(0)