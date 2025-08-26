from typing import Dict, List, Any, Optional
import re
import logging

class BrandVoiceMapper:
    """Maps Brand Bible XML to persona voice constraints with controlled vocabulary"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Controlled vocabulary for tone mapping
        self.tone_vocabulary = {
            "professional": ["formal", "business", "corporate", "executive"],
            "approachable": ["friendly", "welcoming", "accessible", "personable"],
            "innovative": ["creative", "forward-thinking", "progressive", "cutting-edge"],
            "authoritative": ["expert", "knowledgeable", "confident", "trusted"],
            "conversational": ["casual", "relatable", "natural", "chatty"],
            "educational": ["informative", "helpful", "explanatory", "teaching"],
            "inspirational": ["motivational", "uplifting", "encouraging", "aspirational"],
            "technical": ["detailed", "precise", "analytical", "methodical"]
        }
        
        # Default voice constraints
        self.default_constraints = {
            "tone_axes": {
                "formality": 0.7,  # 0 = casual, 1 = formal
                "enthusiasm": 0.5,  # 0 = reserved, 1 = enthusiastic
                "technicality": 0.3,  # 0 = simple, 1 = technical
                "authority": 0.6   # 0 = humble, 1 = authoritative
            },
            "style_preferences": {
                "sentence_length": "medium",  # short, medium, long
                "paragraph_length": "medium",
                "vocabulary_level": "accessible",
                "active_voice": True,
                "contractions": False
            },
            "strict_forbiddens": [
                "—",  # em-dash
                "revolutionary",
                "cutting-edge", 
                "game-changing",
                "disruptive",
                "paradigm shift"
            ],
            "rhetorical_patterns": {
                "avoid_contrasts": True,  # "not just X, but Y"
                "avoid_hyperbole": True,
                "avoid_cliches": True
            }
        }
    
    def brand_bible_to_voice(self, parsed_brand_bible: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parsed Brand Bible to persona voice constraints
        
        Args:
            parsed_brand_bible: Parsed Brand Bible data
            
        Returns:
            Persona voice constraints dictionary
        """
        voice_constraints = self.default_constraints.copy()
        
        # Extract voice data from parsed brand bible
        voice_data = self._extract_voice_data(parsed_brand_bible)
        
        # Map tone to controlled vocabulary
        if "tone" in voice_data:
            voice_constraints["tone_axes"] = self._map_tone_to_axes(voice_data["tone"])
        
        # Add forbidden terms
        if "forbidden_terms" in voice_data:
            voice_constraints["strict_forbiddens"].extend(voice_data["forbidden_terms"])
        
        # Add required phrases
        if "required_phrases" in voice_data:
            voice_constraints["required_phrases"] = voice_data["required_phrases"]
        
        # Add style guidelines
        if "style_guidelines" in voice_data:
            voice_constraints["style_preferences"].update(
                self._map_style_guidelines(voice_data["style_guidelines"])
            )
        
        # Ensure uniqueness in forbidden terms
        voice_constraints["strict_forbiddens"] = list(set(voice_constraints["strict_forbiddens"]))
        
        return voice_constraints
    
    def _extract_voice_data(self, parsed_brand_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Extract voice-related data from parsed brand bible"""
        voice_data = {}
        
        # Extract from identity/voice path
        identity = parsed_brand_bible.get("identity", {})
        voice = identity.get("voice", {})
        
        # Extract tone
        if "tone" in voice:
            tone_value = voice["tone"]
            if isinstance(tone_value, dict) and "text" in tone_value:
                voice_data["tone"] = tone_value["text"]
            elif isinstance(tone_value, str):
                voice_data["tone"] = tone_value
        
        # Extract forbidden terms
        if "forbidden_terms" in voice:
            forbidden_value = voice["forbidden_terms"]
            voice_data["forbidden_terms"] = self._extract_terms(forbidden_value)
        
        # Extract required phrases
        if "required_phrases" in voice:
            required_value = voice["required_phrases"]
            voice_data["required_phrases"] = self._extract_terms(required_value)
        
        # Extract style guidelines
        if "style_guidelines" in voice:
            style_value = voice["style_guidelines"]
            voice_data["style_guidelines"] = self._extract_guidelines(style_value)
        
        return voice_data
    
    def _extract_terms(self, value: Any) -> List[str]:
        """Extract list of terms from XML value"""
        terms = []
        
        if isinstance(value, dict):
            # Handle term/phrase elements
            for key in ["term", "phrase"]:
                if key in value:
                    term_value = value[key]
                    if isinstance(term_value, list):
                        for term in term_value:
                            if isinstance(term, dict) and "text" in term:
                                terms.append(term["text"])
                            elif isinstance(term, str):
                                terms.append(term)
                    elif isinstance(term_value, dict) and "text" in term_value:
                        terms.append(term_value["text"])
                    elif isinstance(term_value, str):
                        terms.append(term_value)
        elif isinstance(value, str):
            # Handle comma-separated string
            terms = [term.strip() for term in value.split(',') if term.strip()]
        
        return terms
    
    def _extract_guidelines(self, value: Any) -> Dict[str, Any]:
        """Extract style guidelines from XML value"""
        guidelines = {}
        
        if isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, dict) and "text" in val:
                    guidelines[key] = val["text"]
                elif isinstance(val, str):
                    guidelines[key] = val
        
        return guidelines
    
    def _map_tone_to_axes(self, tone_description: str) -> Dict[str, float]:
        """Map tone description to numerical axes"""
        tone_lower = tone_description.lower()
        axes = self.default_constraints["tone_axes"].copy()
        
        # Analyze tone description for keywords
        for tone_category, keywords in self.tone_vocabulary.items():
            for keyword in keywords:
                if keyword in tone_lower:
                    self._apply_tone_category(tone_category, axes)
        
        # Apply specific tone mappings
        if "professional" in tone_lower or "formal" in tone_lower:
            axes["formality"] = 0.8
            axes["authority"] = 0.7
        
        if "approachable" in tone_lower or "friendly" in tone_lower:
            axes["formality"] = 0.4
            axes["enthusiasm"] = 0.6
        
        if "innovative" in tone_lower or "creative" in tone_lower:
            axes["technicality"] = 0.6
            axes["enthusiasm"] = 0.7
        
        if "conversational" in tone_lower or "casual" in tone_lower:
            axes["formality"] = 0.3
            axes["enthusiasm"] = 0.6
        
        if "authoritative" in tone_lower or "expert" in tone_lower:
            axes["authority"] = 0.8
            axes["formality"] = 0.7
        
        if "technical" in tone_lower or "detailed" in tone_lower:
            axes["technicality"] = 0.7
            axes["formality"] = 0.6
        
        return axes
    
    def _apply_tone_category(self, category: str, axes: Dict[str, float]):
        """Apply tone category adjustments to axes"""
        adjustments = {
            "professional": {"formality": 0.1, "authority": 0.1},
            "approachable": {"formality": -0.1, "enthusiasm": 0.1},
            "innovative": {"technicality": 0.1, "enthusiasm": 0.1},
            "authoritative": {"authority": 0.1, "formality": 0.1},
            "conversational": {"formality": -0.1, "enthusiasm": 0.1},
            "educational": {"technicality": 0.05, "authority": 0.05},
            "inspirational": {"enthusiasm": 0.1, "authority": 0.05},
            "technical": {"technicality": 0.1, "formality": 0.05}
        }
        
        if category in adjustments:
            for axis, adjustment in adjustments[category].items():
                if axis in axes:
                    axes[axis] = max(0.0, min(1.0, axes[axis] + adjustment))
    
    def _map_style_guidelines(self, guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Map style guidelines to preferences"""
        preferences = {}
        
        for key, value in guidelines.items():
            key_lower = key.lower()
            value_lower = value.lower() if isinstance(value, str) else str(value).lower()
            
            if "active" in key_lower and "voice" in key_lower:
                preferences["active_voice"] = "active" in value_lower
            
            elif "sentence" in key_lower and "length" in key_lower:
                if "short" in value_lower:
                    preferences["sentence_length"] = "short"
                elif "long" in value_lower:
                    preferences["sentence_length"] = "long"
                else:
                    preferences["sentence_length"] = "medium"
            
            elif "paragraph" in key_lower and "length" in key_lower:
                if "short" in value_lower:
                    preferences["paragraph_length"] = "short"
                elif "long" in value_lower:
                    preferences["paragraph_length"] = "long"
                else:
                    preferences["paragraph_length"] = "medium"
            
            elif "vocabulary" in key_lower or "language" in key_lower:
                if "simple" in value_lower or "basic" in value_lower:
                    preferences["vocabulary_level"] = "simple"
                elif "technical" in value_lower or "advanced" in value_lower:
                    preferences["vocabulary_level"] = "technical"
                else:
                    preferences["vocabulary_level"] = "accessible"
            
            elif "contraction" in key_lower:
                preferences["contractions"] = "allow" in value_lower or "use" in value_lower
        
        return preferences
    
    def validate_voice_constraints(self, voice_constraints: Dict[str, Any]) -> List[str]:
        """Validate voice constraints and return any issues"""
        issues = []
        
        # Check required fields
        required_fields = ["tone_axes", "style_preferences", "strict_forbiddens"]
        for field in required_fields:
            if field not in voice_constraints:
                issues.append(f"Missing required field: {field}")
        
        # Validate tone axes
        if "tone_axes" in voice_constraints:
            axes = voice_constraints["tone_axes"]
            expected_axes = ["formality", "enthusiasm", "technicality", "authority"]
            
            for axis in expected_axes:
                if axis not in axes:
                    issues.append(f"Missing tone axis: {axis}")
                elif not isinstance(axes[axis], (int, float)) or axes[axis] < 0 or axes[axis] > 1:
                    issues.append(f"Invalid tone axis value for {axis}: {axes[axis]}")
        
        # Validate style preferences
        if "style_preferences" in voice_constraints:
            prefs = voice_constraints["style_preferences"]
            
            if "sentence_length" in prefs and prefs["sentence_length"] not in ["short", "medium", "long"]:
                issues.append(f"Invalid sentence_length: {prefs['sentence_length']}")
            
            if "vocabulary_level" in prefs and prefs["vocabulary_level"] not in ["simple", "accessible", "technical"]:
                issues.append(f"Invalid vocabulary_level: {prefs['vocabulary_level']}")
        
        # Validate forbidden terms
        if "strict_forbiddens" in voice_constraints:
            forbidden = voice_constraints["strict_forbiddens"]
            if not isinstance(forbidden, list):
                issues.append("strict_forbiddens must be a list")
            else:
                for term in forbidden:
                    if not isinstance(term, str):
                        issues.append(f"Invalid forbidden term: {term}")
        
        return issues
    
    def get_voice_summary(self, voice_constraints: Dict[str, Any]) -> str:
        """Generate a human-readable summary of voice constraints"""
        summary_parts = []
        
        # Tone summary
        if "tone_axes" in voice_constraints:
            axes = voice_constraints["tone_axes"]
            tone_desc = []
            
            if axes.get("formality", 0.5) > 0.7:
                tone_desc.append("formal")
            elif axes.get("formality", 0.5) < 0.3:
                tone_desc.append("casual")
            else:
                tone_desc.append("moderate")
            
            if axes.get("enthusiasm", 0.5) > 0.7:
                tone_desc.append("enthusiastic")
            elif axes.get("enthusiasm", 0.5) < 0.3:
                tone_desc.append("reserved")
            
            if axes.get("technicality", 0.5) > 0.7:
                tone_desc.append("technical")
            elif axes.get("technicality", 0.5) < 0.3:
                tone_desc.append("simple")
            
            if axes.get("authority", 0.5) > 0.7:
                tone_desc.append("authoritative")
            elif axes.get("authority", 0.5) < 0.3:
                tone_desc.append("humble")
            
            summary_parts.append(f"Tone: {', '.join(tone_desc)}")
        
        # Style summary
        if "style_preferences" in voice_constraints:
            prefs = voice_constraints["style_preferences"]
            style_desc = []
            
            if prefs.get("active_voice", True):
                style_desc.append("active voice")
            
            if prefs.get("contractions", False):
                style_desc.append("contractions allowed")
            else:
                style_desc.append("no contractions")
            
            if "sentence_length" in prefs:
                style_desc.append(f"{prefs['sentence_length']} sentences")
            
            if "vocabulary_level" in prefs:
                style_desc.append(f"{prefs['vocabulary_level']} vocabulary")
            
            summary_parts.append(f"Style: {', '.join(style_desc)}")
        
        # Forbidden terms summary
        if "strict_forbiddens" in voice_constraints:
            forbidden = voice_constraints["strict_forbiddens"]
            if forbidden:
                summary_parts.append(f"Forbidden terms: {len(forbidden)} terms")
        
        return "; ".join(summary_parts)

# Global mapper instance
_mapper = None

def get_brand_voice_mapper() -> BrandVoiceMapper:
    """Get or create global brand voice mapper instance"""
    global _mapper
    if _mapper is None:
        _mapper = BrandVoiceMapper()
    return _mapper

def brand_bible_to_voice(parsed_brand_bible: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parsed Brand Bible to persona voice constraints"""
    mapper = get_brand_voice_mapper()
    return mapper.brand_bible_to_voice(parsed_brand_bible)

def validate_voice_constraints(voice_constraints: Dict[str, Any]) -> List[str]:
    """Validate voice constraints and return any issues"""
    mapper = get_brand_voice_mapper()
    return mapper.validate_voice_constraints(voice_constraints)

def get_voice_summary(voice_constraints: Dict[str, Any]) -> str:
    """Generate a human-readable summary of voice constraints"""
    mapper = get_brand_voice_mapper()
    return mapper.get_voice_summary(voice_constraints)

if __name__ == "__main__":
    # Test the brand voice mapper
    mapper = BrandVoiceMapper()
    
    # Test with sample parsed brand bible
    sample_parsed = {
        "identity": {
            "voice": {
                "tone": {"text": "professional, approachable, innovative"},
                "forbidden_terms": {
                    "term": [
                        {"text": "revolutionary"},
                        {"text": "cutting-edge"},
                        {"text": "game-changing"}
                    ]
                },
                "required_phrases": {
                    "phrase": [
                        {"text": "empowering businesses"},
                        {"text": "streamlined solutions"}
                    ]
                },
                "style_guidelines": {
                    "guideline": [
                        {"text": "Use active voice"},
                        {"text": "Avoid jargon"},
                        {"text": "Keep sentences medium length"}
                    ]
                }
            }
        }
    }
    
    # Convert to voice constraints
    voice_constraints = mapper.brand_bible_to_voice(sample_parsed)
    print("Voice constraints:")
    print(voice_constraints)
    
    # Validate constraints
    issues = mapper.validate_voice_constraints(voice_constraints)
    if issues:
        print(f"Validation issues: {issues}")
    else:
        print("Voice constraints are valid")
    
    # Get summary
    summary = mapper.get_voice_summary(voice_constraints)
    print(f"Voice summary: {summary}")
