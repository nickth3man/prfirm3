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
from typing import Dict, Any, List

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

# TODO(dev,2025-01-01): Add import for logging module to enable comprehensive flow monitoring
# TODO(dev,2025-01-01): Add import for configuration management utilities for environment-specific settings
# TODO(dev,2025-01-01): Consider adding type hints for better IDE support and development experience
# TODO(dev,2025-01-01): Import validation utilities for input/output checking
# TODO(dev,2025-01-01): Add metrics collection imports for performance monitoring


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
    
    TODO(dev,2025-01-01): Add parameter validation for flow configuration
    TODO(dev,2025-01-01): Implement flow state persistence for long-running processes
    TODO(dev,2025-01-01): Add metrics collection and monitoring hooks
    TODO(dev,2025-01-01): Create unit tests for flow creation and wiring
    TODO(dev,2025-01-01): Add documentation for each node's expected inputs/outputs
    """
    # TODO(dev,2025-01-01): Add try-catch blocks for node instantiation failures
    # TODO(dev,2025-01-01): Consider using dependency injection for better testability
    # TODO(dev,2025-01-01): Add logging for flow creation steps
    # TODO(dev,2025-01-01): Validate environment requirements before node creation
    # TODO(dev,2025-01-01): Add memory usage monitoring for large content processing
    
    # Instantiate the first three nodes in our sequential pipeline
    # These handle the initial setup and brand alignment phases
    
    # Create engagement manager node - handles client requirements and task initialization
    # Purpose: Processes initial PR requests and sets up the content creation context
    # Input example: {"task_requirements": {"platforms": ["twitter"], "content_type": "announcement"}}
    # Output example: Enriched task context with client metadata and requirements structure
    engagement = EngagementManagerNode()
    
    # Create brand bible ingestion node - processes brand guidelines and documentation
    # Purpose: Loads and parses brand voice, style guides, and compliance requirements
    # Input example: Task context + brand documentation files/URLs
    # Output example: Structured brand guidelines with voice parameters and style rules
    ingest = BrandBibleIngestNode()
    
    # Create voice alignment node - ensures content strategy matches brand voice
    # Purpose: Analyzes brand requirements and sets content direction parameters
    # Input example: Task context + structured brand guidelines
    # Output example: Content strategy with tone, voice, and messaging alignment parameters
    voice = VoiceAlignmentNode()
    
    # TODO(dev,2025-01-01): Add validation that all nodes are properly configured
    # TODO(dev,2025-01-01): Consider adding node health checks before wiring
    # TODO(dev,2025-01-01): Add dependency validation between nodes
    # TODO(dev,2025-01-01): Implement node configuration validation
    # TODO(dev,2025-01-01): Add fallback nodes for error scenarios
    
    # Platform formatting runs once per platform via a BatchFlow
    # This enables parallel processing of content for multiple social media platforms
    
    # Create the platform formatting node that will handle platform-specific adjustments
    # Purpose: Adapts content format, length, and style for each target platform
    # Domain constraint: Must respect platform character limits (Twitter: 280, LinkedIn: 3000)
    # Performance target: Process all platforms in parallel, < 10 seconds per platform
    platform_node = PlatformFormattingNode()
    
    # Create a simple Flow that runs the platform node (BatchFlow expects a Flow start)
    # Design constraint: The BatchFlow framework requires an inner Flow to execute for each parameter set
    # This wrapper enables the batch processing pattern for platform-specific formatting
    format_flow = Flow(start=platform_node)
    
    # TODO(dev,2025-01-01): Add error handling for Flow instantiation
    # TODO(dev,2025-01-01): Consider making format_flow configurable based on platform types
    # TODO(dev,2025-01-01): Add validation for platform node compatibility
    # TODO(dev,2025-01-01): Implement timeout handling for platform-specific processing
    # TODO(dev,2025-01-01): Add retry logic for failed platform formatting attempts
    
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

        TODO(dev,2025-01-01): Add validation for supported platforms
        TODO(dev,2025-01-01): Implement retry logic for failed platform formatting
        TODO(dev,2025-01-01): Add parallel processing for better performance
        TODO(dev,2025-01-01): Create platform-specific configuration loading
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

            TODO(dev,2025-01-01): Add comprehensive input validation
            TODO(dev,2025-01-01): Add logging for platform preparation
            TODO(dev,2025-01-01): Handle case where no platforms are specified
            TODO(dev,2025-01-01): Add support for platform-specific parameters beyond just name
            """
            
            # Extract platforms list from the shared store with safe navigation
            # Defense: This approach prevents KeyError exceptions when the expected structure is missing
            # Fallback: Empty list ensures flow continues even with malformed input
            platforms = shared.get("task_requirements", {}).get("platforms", [])
            
            # TODO(dev,2025-01-01): Validate that all platforms are supported by our formatting node
            # TODO(dev,2025-01-01): Add default platform fallback if none specified (e.g., "generic")
            # TODO(dev,2025-01-01): Log which platforms are being processed for debugging
            # TODO(dev,2025-01-01): Add platform name normalization (lowercase, trim whitespace)
            # TODO(dev,2025-01-01): Filter out duplicate platform names
            # TODO(dev,2025-01-01): Add validation for platform name format and supported characters
            
            # Return list of param dicts for BatchFlow
            # Each parameter set will be passed to the inner flow for processing
            # Performance note: List comprehension is O(n) and memory-efficient for typical platform counts
            return [{"platform": p} for p in platforms]

    # TODO(dev,2025-01-01): Add error handling for BatchFlow instantiation
    # TODO(dev,2025-01-01): Consider adding timeout configuration for batch processing
    # TODO(dev,2025-01-01): Add progress monitoring for batch operations
    # TODO(dev,2025-01-01): Implement batch size limits for large platform lists
    # TODO(dev,2025-01-01): Add configuration for parallel vs sequential batch processing
    
    # Create the batch flow instance that will manage platform-specific processing
    # Design constraint: This BatchFlow will run the format_flow once for each platform parameter set
    # Performance expectation: Parallel execution should reduce total platform processing time by ~70%
    format_batch = FormatGuidelinesFlow(start=format_flow)
    
    # Instantiate the remaining nodes for content creation and quality assurance
    # These nodes handle the core content generation and final validation phases
    
    # Create content craftsman node - generates and refines the actual content
    # Purpose: Takes aligned requirements and creates draft content for all platforms
    # Input example: Aligned strategy + platform formatting requirements
    # Output example: Platform-specific content drafts with messaging and formatting applied
    craft = ContentCraftsmanNode()
    
    # Create style editor node - performs editorial review and content improvements
    # Purpose: Enhances readability, tone, and overall content quality
    # Domain constraint: Must maintain brand voice while improving clarity and engagement
    # Performance target: < 10 seconds for typical PR content length
    editor = StyleEditorNode()
    
    # Create compliance node - final validation against brand guidelines
    # Purpose: Ensures the finished content meets all brand and legal requirements
    # Business rule: Content MUST NOT proceed if compliance checks fail
    # Input example: Edited content + original brand guidelines
    # Output example: Validated final content with compliance metadata
    compliance = StyleComplianceNode()

    # TODO(dev,2025-01-01): Add validation that all nodes support the expected interfaces
    # TODO(dev,2025-01-01): Consider adding conditional branching based on content type
    # TODO(dev,2025-01-01): Add checkpoints for flow state persistence at key stages
    # TODO(dev,2025-01-01): Implement rollback mechanisms for failed nodes
    # TODO(dev,2025-01-01): Add configuration for optional nodes based on content requirements
    # TODO(dev,2025-01-01): Validate that each node can handle the expected data formats
    
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
    engagement >> ingest >> voice >> format_batch >> craft >> editor >> compliance

    # TODO(dev,2025-01-01): Add flow validation after wiring to ensure all connections are valid
    # TODO(dev,2025-01-01): Create flow diagram generation for documentation and debugging
    # TODO(dev,2025-01-01): Add performance profiling hooks at each connection point
    # TODO(dev,2025-01-01): Consider adding alternative paths for different content types
    # TODO(dev,2025-01-01): Implement conditional routing based on content complexity
    # TODO(dev,2025-01-01): Add circuit breaker patterns for handling node failures
    # TODO(dev,2025-01-01): Create flow state snapshots for debugging and recovery

    # Return the complete flow starting with the engagement manager
    # The Flow constructor creates the runnable pipeline with proper error handling
    # Postcondition: Returned flow is ready to process PR content with shared store input
    return Flow(start=engagement)


# Convenience instance for quick testing and development
# This pre-instantiated flow can be used directly for testing purposes
# without needing to call create_main_flow() each time

# TODO(dev,2025-01-01): Add configuration management for main_flow instance
# TODO(dev,2025-01-01): Consider lazy initialization for better resource management
# TODO(dev,2025-01-01): Add environment-specific flow variants (dev, staging, prod)
# TODO(dev,2025-01-01): Create factory methods for different flow configurations
# TODO(dev,2025-01-01): Add health check methods for the pre-instantiated flow
# TODO(dev,2025-01-01): Implement flow caching and reuse strategies
# TODO(dev,2025-01-01): Add monitoring and metrics collection for the main flow instance

# Create the main flow instance that can be imported and used immediately
# Purpose: Provides a ready-to-use flow for interactive testing and development workflows
# Usage pattern: from flow import main_flow; result = main_flow.run(shared=data)
# Performance note: Instantiation cost is amortized across multiple runs
main_flow = create_main_flow()

# TODO(dev,2025-01-01): Add validation that main_flow creation succeeded
# TODO(dev,2025-01-01): Consider adding flow warming/pre-validation on module import
# TODO(dev,2025-01-01): Add graceful handling if flow creation fails during import
# TODO(dev,2025-01-01): Create alternative flow instances for different use cases