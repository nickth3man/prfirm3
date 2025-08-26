"""
Pydantic schemas for shared state validation in the Virtual PR Firm system.

This module defines comprehensive data models for validating the structure
and content of the shared state dictionary that flows through all nodes.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime
from enum import Enum


class PlatformEnum(str, Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"
    EMAIL = "email"
    BLOG = "blog"


class ContentStatus(str, Enum):
    """Status of content generation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    GENERATED = "generated"
    VALIDATING = "validating"
    APPROVED = "approved"
    REVISION_REQUIRED = "revision_required"
    FAILED = "failed"


class TaskRequirements(BaseModel):
    """User input requirements for content generation."""
    platforms: List[PlatformEnum] = Field(
        default_factory=list,
        description="List of platforms to generate content for"
    )
    topic_or_goal: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Main topic or goal for content generation"
    )
    intents: Optional[Dict[PlatformEnum, str]] = Field(
        default=None,
        description="Platform-specific intents/objectives"
    )
    reddit_details: Optional[Dict[str, str]] = Field(
        default=None,
        description="Reddit-specific details (subreddit, rules, description)"
    )
    
    @validator('platforms')
    def validate_platforms(cls, v):
        if not v:
            raise ValueError("At least one platform must be specified")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate platforms not allowed")
        return v
    
    @validator('reddit_details')
    def validate_reddit_details(cls, v, values):
        if v and 'platforms' in values and PlatformEnum.REDDIT in values['platforms']:
            required_fields = ['subreddit_name', 'rules', 'description']
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"Reddit details must include {field}")
        return v


class BrandBible(BaseModel):
    """Brand guidelines and voice configuration."""
    xml_raw: str = Field(
        default="",
        description="Raw XML content of brand bible"
    )
    parsed_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed brand bible data"
    )
    persona_voice: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted persona and voice characteristics"
    )
    forbidden_phrases: List[str] = Field(
        default_factory=lambda: ["em dash", "rhetorical contrasts"],
        description="List of forbidden phrases and patterns"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ContentPiece(BaseModel):
    """Individual content piece for a platform."""
    platform: PlatformEnum
    content: str = Field(..., min_length=1)
    status: ContentStatus = ContentStatus.PENDING
    revision_count: int = Field(default=0, ge=0, le=5)
    validation_errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific metadata (hashtags, links, etc.)"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    version: int = Field(default=1, ge=1)
    
    @validator('revision_count')
    def validate_revision_count(cls, v):
        if v > 5:
            raise ValueError("Maximum revision count (5) exceeded")
        return v


class StreamingState(BaseModel):
    """Streaming manager state and configuration."""
    enabled: bool = Field(default=False)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    current_node: Optional[str] = None
    websocket_url: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationResult(BaseModel):
    """Results from content validation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.now)


class SharedState(BaseModel):
    """Complete shared state schema for the Virtual PR Firm system."""
    # Core requirements
    task_requirements: TaskRequirements
    brand_bible: BrandBible = Field(default_factory=BrandBible)
    
    # Generated content
    content_pieces: Dict[PlatformEnum, ContentPiece] = Field(
        default_factory=dict,
        description="Platform-mapped content pieces"
    )
    
    # Platform formatting guidelines
    platform_guidelines: Dict[PlatformEnum, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Platform-specific formatting guidelines"
    )
    
    # Validation and compliance
    validation_results: Dict[PlatformEnum, ValidationResult] = Field(
        default_factory=dict,
        description="Validation results per platform"
    )
    
    # Streaming and UI state
    stream: Optional[StreamingState] = None
    
    # User feedback and versioning
    feedback_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of user feedback and edits"
    )
    
    # System metadata
    request_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    # Cost tracking
    llm_costs: Dict[str, float] = Field(
        default_factory=dict,
        description="Accumulated LLM API costs"
    )
    
    # Error tracking
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered during processing"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
    
    def add_error(self, node: str, error: str, details: Optional[Dict] = None):
        """Add an error to the error tracking list."""
        self.errors.append({
            "node": node,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now()
        })
    
    def get_platform_content(self, platform: PlatformEnum) -> Optional[str]:
        """Get content for a specific platform."""
        if platform in self.content_pieces:
            return self.content_pieces[platform].content
        return None
    
    def is_platform_complete(self, platform: PlatformEnum) -> bool:
        """Check if content generation is complete for a platform."""
        if platform not in self.content_pieces:
            return False
        return self.content_pieces[platform].status == ContentStatus.APPROVED


def validate_shared_state(shared: Dict[str, Any]) -> SharedState:
    """
    Validate and parse a shared state dictionary.
    
    Args:
        shared: Raw shared state dictionary
        
    Returns:
        Validated SharedState model
        
    Raises:
        ValidationError: If the shared state is invalid
    """
    return SharedState(**shared)


def create_initial_shared_state(
    platforms: List[str],
    topic: str,
    brand_bible_xml: str = "",
    reddit_details: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create an initial validated shared state.
    
    Args:
        platforms: List of platform names
        topic: Main topic or goal
        brand_bible_xml: Optional brand bible XML content
        reddit_details: Optional Reddit-specific details
        
    Returns:
        Dictionary representation of validated SharedState
    """
    # Convert platform strings to enums
    platform_enums = []
    for p in platforms:
        try:
            platform_enums.append(PlatformEnum(p.lower()))
        except ValueError:
            raise ValueError(f"Unsupported platform: {p}")
    
    # Build task requirements
    task_req = {
        "platforms": platform_enums,
        "topic_or_goal": topic
    }
    
    if reddit_details and PlatformEnum.REDDIT in platform_enums:
        task_req["reddit_details"] = reddit_details
    
    # Create and validate state
    state = SharedState(
        task_requirements=TaskRequirements(**task_req),
        brand_bible=BrandBible(xml_raw=brand_bible_xml)
    )
    
    return state.model_dump()