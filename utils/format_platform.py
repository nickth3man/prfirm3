from typing import Dict, List, Any, Optional
import re

class PlatformFormatter:
    """Generates platform-specific guidelines with unified schema"""
    
    def __init__(self):
        # Platform-specific default guidelines
        self.platform_defaults = {
            "linkedin": {
                "limits": {
                    "max_length": 3000,
                    "min_length": 1000,
                    "paragraph_max": 150
                },
                "structure": {
                    "intro_required": True,
                    "body_sections": 2,
                    "conclusion_required": True,
                    "cta_required": True
                },
                "style": {
                    "tone": "professional",
                    "formality": "high",
                    "hashtag_placement": "end",
                    "hashtag_count": [0, 5],
                    "mention_style": "professional"
                },
                "content_rules": {
                    "industry_focus": True,
                    "thought_leadership": True,
                    "benefit_focused": True,
                    "data_driven": True
                }
            },
            "twitter": {
                "limits": {
                    "max_length": 280,
                    "min_length": 50,
                    "thread_ok_above": 240
                },
                "structure": {
                    "intro_required": False,
                    "body_sections": 1,
                    "conclusion_required": False,
                    "cta_required": False
                },
                "style": {
                    "tone": "conversational",
                    "formality": "medium",
                    "hashtag_placement": "inline",
                    "hashtag_count": [0, 3],
                    "mention_style": "casual"
                },
                "content_rules": {
                    "trending_topics": True,
                    "engagement_focused": True,
                    "conversational": True,
                    "timely": True
                }
            },
            "instagram": {
                "limits": {
                    "max_length": 2200,
                    "min_length": 100,
                    "line_breaks": "liberal"
                },
                "structure": {
                    "intro_required": False,
                    "body_sections": 1,
                    "conclusion_required": False,
                    "cta_required": True
                },
                "style": {
                    "tone": "authentic",
                    "formality": "low",
                    "hashtag_placement": "end",
                    "hashtag_count": [8, 20],
                    "mention_style": "friendly"
                },
                "content_rules": {
                    "visual_storytelling": True,
                    "behind_scenes": True,
                    "user_generated": True,
                    "emotional": True
                }
            },
            "reddit": {
                "limits": {
                    "max_length": 10000,
                    "min_length": 200,
                    "markdown_support": True
                },
                "structure": {
                    "intro_required": True,
                    "body_sections": 3,
                    "conclusion_required": False,
                    "cta_required": False
                },
                "style": {
                    "tone": "community",
                    "formality": "medium",
                    "hashtag_placement": "none",
                    "hashtag_count": [0, 0],
                    "mention_style": "none"
                },
                "content_rules": {
                    "community_focused": True,
                    "helpful": True,
                    "authentic": True,
                    "rule_compliant": True
                }
            },
            "email": {
                "limits": {
                    "max_length": 2000,
                    "min_length": 100,
                    "subject_max": 50
                },
                "structure": {
                    "intro_required": True,
                    "body_sections": 2,
                    "conclusion_required": True,
                    "cta_required": True
                },
                "style": {
                    "tone": "professional",
                    "formality": "high",
                    "hashtag_placement": "none",
                    "hashtag_count": [0, 0],
                    "mention_style": "formal"
                },
                "content_rules": {
                    "clear_purpose": True,
                    "action_oriented": True,
                    "personalized": True,
                    "scannable": True
                }
            },
            "blog": {
                "limits": {
                    "max_length": 3000,
                    "min_length": 800,
                    "word_count": [800, 3000]
                },
                "structure": {
                    "intro_required": True,
                    "body_sections": 3,
                    "conclusion_required": True,
                    "cta_required": True
                },
                "style": {
                    "tone": "educational",
                    "formality": "medium",
                    "hashtag_placement": "none",
                    "hashtag_count": [0, 0],
                    "mention_style": "professional"
                },
                "content_rules": {
                    "seo_optimized": True,
                    "detailed": True,
                    "educational": True,
                    "link_rich": True
                }
            }
        }
    
    def build_guidelines(self, platform: str, persona_voice: Dict[str, Any], 
                        intent: str, platform_nuance: Optional[Dict[str, Any]] = None,
                        reddit_rules: Optional[str] = None, 
                        reddit_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Build platform-specific guidelines
        
        Args:
            platform: Platform name (linkedin, twitter, instagram, reddit, email, blog)
            persona_voice: Voice constraints from brand bible
            intent: Content intent/purpose
            platform_nuance: Platform-specific adjustments
            reddit_rules: Reddit subreddit rules (for reddit platform)
            reddit_description: Reddit subreddit description (for reddit platform)
            
        Returns:
            Unified guidelines schema
        """
        if platform not in self.platform_defaults:
            raise ValueError(f"Unsupported platform: {platform}")
        
        # Start with platform defaults
        guidelines = self.platform_defaults[platform].copy()
        
        # Apply persona voice adjustments
        guidelines = self._apply_persona_voice(guidelines, persona_voice)
        
        # Apply intent adjustments
        guidelines = self._apply_intent(guidelines, intent)
        
        # Apply platform nuance adjustments
        if platform_nuance:
            guidelines = self._apply_platform_nuance(guidelines, platform_nuance)
        
        # Apply platform-specific customizations
        if platform == "reddit":
            guidelines = self._apply_reddit_customizations(guidelines, reddit_rules, reddit_description)
        
        # Add metadata
        guidelines["metadata"] = {
            "platform": platform,
            "intent": intent,
            "persona_voice_applied": True,
            "customizations_applied": platform_nuance is not None
        }
        
        return guidelines
    
    def _apply_persona_voice(self, guidelines: Dict[str, Any], persona_voice: Dict[str, Any]) -> Dict[str, Any]:
        """Apply persona voice constraints to guidelines"""
        if "tone_axes" in persona_voice:
            axes = persona_voice["tone_axes"]
            
            # Adjust tone based on formality
            if axes.get("formality", 0.5) > 0.7:
                guidelines["style"]["tone"] = "professional"
                guidelines["style"]["formality"] = "high"
            elif axes.get("formality", 0.5) < 0.3:
                guidelines["style"]["tone"] = "casual"
                guidelines["style"]["formality"] = "low"
            
            # Adjust based on technicality
            if axes.get("technicality", 0.5) > 0.7:
                guidelines["content_rules"]["detailed"] = True
                guidelines["limits"]["min_length"] = max(guidelines["limits"]["min_length"], 500)
            
            # Adjust based on authority
            if axes.get("authority", 0.5) > 0.7:
                guidelines["content_rules"]["thought_leadership"] = True
                guidelines["content_rules"]["expert_focused"] = True
        
        # Apply style preferences
        if "style_preferences" in persona_voice:
            prefs = persona_voice["style_preferences"]
            
            if "sentence_length" in prefs:
                if prefs["sentence_length"] == "short":
                    guidelines["limits"]["paragraph_max"] = min(guidelines["limits"]["paragraph_max"], 100)
                elif prefs["sentence_length"] == "long":
                    guidelines["limits"]["paragraph_max"] = max(guidelines["limits"]["paragraph_max"], 200)
            
            if "vocabulary_level" in prefs:
                if prefs["vocabulary_level"] == "simple":
                    guidelines["content_rules"]["accessible_language"] = True
                elif prefs["vocabulary_level"] == "technical":
                    guidelines["content_rules"]["technical_terms"] = True
        
        # Apply forbidden terms
        if "strict_forbiddens" in persona_voice:
            guidelines["content_rules"]["forbidden_terms"] = persona_voice["strict_forbiddens"]
        
        # Apply required phrases
        if "required_phrases" in persona_voice:
            guidelines["content_rules"]["required_phrases"] = persona_voice["required_phrases"]
        
        return guidelines
    
    def _apply_intent(self, guidelines: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """Apply content intent to guidelines"""
        intent_lower = intent.lower()
        
        # Common intent patterns
        if "thought leadership" in intent_lower or "expert" in intent_lower:
            guidelines["content_rules"]["thought_leadership"] = True
            guidelines["content_rules"]["industry_focus"] = True
            guidelines["limits"]["min_length"] = max(guidelines["limits"]["min_length"], 800)
        
        elif "engagement" in intent_lower or "conversation" in intent_lower:
            guidelines["content_rules"]["engagement_focused"] = True
            guidelines["content_rules"]["conversational"] = True
            guidelines["style"]["tone"] = "conversational"
        
        elif "promotion" in intent_lower or "sales" in intent_lower:
            guidelines["content_rules"]["promotional"] = True
            guidelines["content_rules"]["cta_focused"] = True
            guidelines["structure"]["cta_required"] = True
        
        elif "education" in intent_lower or "informative" in intent_lower:
            guidelines["content_rules"]["educational"] = True
            guidelines["content_rules"]["detailed"] = True
            guidelines["limits"]["min_length"] = max(guidelines["limits"]["min_length"], 500)
        
        elif "storytelling" in intent_lower or "narrative" in intent_lower:
            guidelines["content_rules"]["storytelling"] = True
            guidelines["content_rules"]["emotional"] = True
            guidelines["structure"]["intro_required"] = True
        
        return guidelines
    
    def _apply_platform_nuance(self, guidelines: Dict[str, Any], platform_nuance: Dict[str, Any]) -> Dict[str, Any]:
        """Apply platform-specific nuance adjustments"""
        for category, adjustments in platform_nuance.items():
            if category == "limits":
                for limit_type, value in adjustments.items():
                    if limit_type in guidelines["limits"]:
                        guidelines["limits"][limit_type] = value
            
            elif category == "style":
                for style_type, value in adjustments.items():
                    if style_type in guidelines["style"]:
                        guidelines["style"][style_type] = value
            
            elif category == "content_rules":
                for rule_type, value in adjustments.items():
                    guidelines["content_rules"][rule_type] = value
            
            elif category == "structure":
                for struct_type, value in adjustments.items():
                    if struct_type in guidelines["structure"]:
                        guidelines["structure"][struct_type] = value
        
        return guidelines
    
    def _apply_reddit_customizations(self, guidelines: Dict[str, Any], 
                                   reddit_rules: Optional[str], 
                                   reddit_description: Optional[str]) -> Dict[str, Any]:
        """Apply Reddit-specific customizations"""
        if reddit_rules:
            guidelines["content_rules"]["subreddit_rules"] = reddit_rules
            guidelines["content_rules"]["rule_compliant"] = True
        
        if reddit_description:
            guidelines["content_rules"]["subreddit_context"] = reddit_description
            guidelines["content_rules"]["community_aligned"] = True
        
        return guidelines
    
    def get_platform_summary(self, guidelines: Dict[str, Any]) -> str:
        """Generate a human-readable summary of platform guidelines"""
        platform = guidelines.get("metadata", {}).get("platform", "unknown")
        intent = guidelines.get("metadata", {}).get("intent", "general")
        
        limits = guidelines.get("limits", {})
        style = guidelines.get("style", {})
        content_rules = guidelines.get("content_rules", {})
        
        summary_parts = [
            f"Platform: {platform.title()}",
            f"Intent: {intent}",
            f"Length: {limits.get('min_length', 0)}-{limits.get('max_length', 0)} chars",
            f"Tone: {style.get('tone', 'general')}",
            f"Formality: {style.get('formality', 'medium')}"
        ]
        
        # Add hashtag info if applicable
        if style.get("hashtag_count", [0, 0]) != [0, 0]:
            hashtag_range = style["hashtag_count"]
            summary_parts.append(f"Hashtags: {hashtag_range[0]}-{hashtag_range[1]}")
        
        # Add key content rules
        key_rules = []
        for rule, value in content_rules.items():
            if isinstance(value, bool) and value:
                key_rules.append(rule.replace("_", " "))
        
        if key_rules:
            summary_parts.append(f"Focus: {', '.join(key_rules[:3])}")
        
        return " | ".join(summary_parts)
    
    def validate_guidelines(self, guidelines: Dict[str, Any]) -> List[str]:
        """Validate guidelines structure and return any issues"""
        issues = []
        
        # Check required sections
        required_sections = ["limits", "structure", "style", "content_rules"]
        for section in required_sections:
            if section not in guidelines:
                issues.append(f"Missing required section: {section}")
        
        # Validate limits
        if "limits" in guidelines:
            limits = guidelines["limits"]
            if "max_length" not in limits:
                issues.append("Missing max_length in limits")
            elif not isinstance(limits["max_length"], int) or limits["max_length"] <= 0:
                issues.append("Invalid max_length value")
        
        # Validate style
        if "style" in guidelines:
            style = guidelines["style"]
            if "tone" not in style:
                issues.append("Missing tone in style")
            if "formality" not in style:
                issues.append("Missing formality in style")
        
        return issues

# Global formatter instance
_formatter = None

def get_platform_formatter() -> PlatformFormatter:
    """Get or create global platform formatter instance"""
    global _formatter
    if _formatter is None:
        _formatter = PlatformFormatter()
    return _formatter

def build_platform_guidelines(platform: str, persona_voice: Dict[str, Any], 
                             intent: str, platform_nuance: Optional[Dict[str, Any]] = None,
                             reddit_rules: Optional[str] = None, 
                             reddit_description: Optional[str] = None) -> Dict[str, Any]:
    """Build platform-specific guidelines"""
    formatter = get_platform_formatter()
    return formatter.build_guidelines(platform, persona_voice, intent, platform_nuance, 
                                    reddit_rules, reddit_description)

def get_platform_summary(guidelines: Dict[str, Any]) -> str:
    """Generate a human-readable summary of platform guidelines"""
    formatter = get_platform_formatter()
    return formatter.get_platform_summary(guidelines)

def validate_guidelines(guidelines: Dict[str, Any]) -> List[str]:
    """Validate guidelines structure and return any issues"""
    formatter = get_platform_formatter()
    return formatter.validate_guidelines(guidelines)

if __name__ == "__main__":
    # Test the platform formatter
    formatter = PlatformFormatter()
    
    # Test persona voice
    persona_voice = {
        "tone_axes": {
            "formality": 0.8,
            "enthusiasm": 0.6,
            "technicality": 0.4,
            "authority": 0.7
        },
        "style_preferences": {
            "sentence_length": "medium",
            "vocabulary_level": "accessible"
        },
        "strict_forbiddens": ["revolutionary", "cutting-edge"],
        "required_phrases": ["empowering businesses"]
    }
    
    # Test LinkedIn guidelines
    linkedin_guidelines = formatter.build_guidelines(
        "linkedin", 
        persona_voice, 
        "thought leadership"
    )
    print("LinkedIn Guidelines:")
    print(linkedin_guidelines)
    
    # Test Twitter guidelines
    twitter_guidelines = formatter.build_guidelines(
        "twitter", 
        persona_voice, 
        "engagement"
    )
    print("\nTwitter Guidelines:")
    print(twitter_guidelines)
    
    # Test Reddit guidelines with custom rules
    reddit_guidelines = formatter.build_guidelines(
        "reddit",
        persona_voice,
        "education",
        reddit_rules="No self-promotion, Be helpful",
        reddit_description="A community for learning about technology"
    )
    print("\nReddit Guidelines:")
    print(reddit_guidelines)
    
    # Test summaries
    print(f"\nLinkedIn Summary: {formatter.get_platform_summary(linkedin_guidelines)}")
    print(f"Twitter Summary: {formatter.get_platform_summary(twitter_guidelines)}")
    print(f"Reddit Summary: {formatter.get_platform_summary(reddit_guidelines)}")
    
    # Test validation
    issues = formatter.validate_guidelines(linkedin_guidelines)
    if issues:
        print(f"Validation issues: {issues}")
    else:
        print("Guidelines are valid")
