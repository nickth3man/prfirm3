from typing import Dict, List, Any, Optional
import re
from .check_style_violations import get_style_violation_checker
from .openrouter_client import get_openrouter_client

class ConstrainedRewriter:
    """Removes AI fingerprints without introducing banned terms"""
    
    def __init__(self):
        self.style_checker = get_style_violation_checker()
        self.llm_client = get_openrouter_client()
        
        # Common AI fingerprint patterns to fix
        self.ai_patterns = {
            "stiff_transitions": [
                (r'\bFurthermore\b', "Additionally"),
                (r'\bMoreover\b', "Also"),
                (r'\bIn addition\b', "Plus"),
                (r'\bAdditionally\b', "Also"),
                (r'\bNotably\b', "Importantly"),
                (r'\bSignificantly\b', "Importantly")
            ],
            "generic_openings": [
                (r'\bIn today\'s fast-paced world\b', "Today"),
                (r'\bAs we all know\b', "We know"),
                (r'\bIt\'s no secret that\b', "Clearly"),
                (r'\bThe truth is\b', "The fact is"),
                (r'\bIn the modern era\b', "Nowadays"),
                (r'\bIn this digital age\b', "Today")
            ],
            "predictable_structures": [
                (r'\bFirst\b', "To start"),
                (r'\bSecond\b', "Next"),
                (r'\bThird\b', "Finally"),
                (r'\bLet\'s explore\b', "Let's look at"),
                (r'\bHere\'s how\b', "Here's what"),
                (r'\bIn summary\b', "To sum up")
            ],
            "excessive_qualifiers": [
                (r'\bvery\s+(\w+)\b', r'\1'),
                (r'\breally\s+(\w+)\b', r'\1'),
                (r'\bquite\s+(\w+)\b', r'\1'),
                (r'\bextremely\s+(\w+)\b', r'\1'),
                (r'\babsolutely\s+(\w+)\b', r'\1')
            ]
        }
    
    def rewrite_with_constraints(self, text: str, voice: Dict[str, Any], 
                                guidelines: Dict[str, Any]) -> str:
        """
        Rewrite text to remove AI fingerprints while respecting constraints
        
        Args:
            text: Original text to rewrite
            voice: Voice constraints from brand bible
            guidelines: Platform guidelines
            
        Returns:
            Rewritten text
        """
        # First, check current violations
        initial_check = self.style_checker.check_style_violations(text)
        
        if initial_check["passed"]:
            # No strict violations, just minor improvements
            return self._apply_minor_improvements(text, voice, guidelines)
        
        # Apply automatic fixes first
        rewritten_text = self._apply_automatic_fixes(text, voice)
        
        # Use LLM for more complex rewrites if needed
        if self._needs_llm_rewrite(rewritten_text, voice, guidelines):
            rewritten_text = self._llm_rewrite(rewritten_text, voice, guidelines)
        
        # Final validation
        final_check = self.style_checker.check_style_violations(rewritten_text)
        
        # Ensure we didn't introduce new violations
        if not final_check["passed"]:
            # Fallback to safer rewrite
            rewritten_text = self._safe_rewrite(text, voice, guidelines)
        
        return rewritten_text
    
    def _apply_automatic_fixes(self, text: str, voice: Dict[str, Any]) -> str:
        """Apply automatic pattern-based fixes"""
        fixed_text = text
        
        # Apply AI pattern fixes
        for pattern_category, patterns in self.ai_patterns.items():
            for pattern, replacement in patterns:
                fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
        
        # Fix em-dashes
        fixed_text = re.sub(r'—', '–', fixed_text)  # Replace with en-dash
        
        # Fix rhetorical contrasts
        fixed_text = self._fix_rhetorical_contrasts(fixed_text)
        
        # Fix forbidden terms
        if "strict_forbiddens" in voice:
            fixed_text = self._replace_forbidden_terms(fixed_text, voice["strict_forbiddens"])
        
        return fixed_text
    
    def _fix_rhetorical_contrasts(self, text: str) -> str:
        """Fix rhetorical contrast patterns"""
        # Pattern: "not just X; it's Y" -> "X is Y"
        text = re.sub(
            r'\bnot\s+just\s+(\w+[^;]*);\s*it\'s\s+(\w+[^.]*\.)',
            r'\1 is \2',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern: "not X, but Y" -> "X is Y"
        text = re.sub(
            r'\bnot\s+(\w+[^,]*),\s+but\s+(\w+[^.]*\.)',
            r'\1 is \2',
            text,
            flags=re.IGNORECASE
        )
        
        # Pattern: "not X. Instead, Y" -> "X is Y"
        text = re.sub(
            r'\bnot\s+(\w+[^.]*)\.\s*Instead[^,]*,\s*(\w+[^.]*\.)',
            r'\1 is \2',
            text,
            flags=re.IGNORECASE
        )
        
        return text
    
    def _replace_forbidden_terms(self, text: str, forbidden_terms: List[str]) -> str:
        """Replace forbidden terms with alternatives"""
        replacements = {
            "revolutionary": "innovative",
            "cutting-edge": "advanced",
            "game-changing": "transformative",
            "disruptive": "innovative",
            "paradigm shift": "major change",
            "next-generation": "advanced",
            "breakthrough": "significant advancement",
            "innovative solution": "effective solution"
        }
        
        for term in forbidden_terms:
            if term.lower() in text.lower():
                replacement = replacements.get(term.lower(), "effective")
                text = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    replacement,
                    text,
                    flags=re.IGNORECASE
                )
        
        return text
    
    def _needs_llm_rewrite(self, text: str, voice: Dict[str, Any], 
                          guidelines: Dict[str, Any]) -> bool:
        """Determine if text needs LLM-based rewriting"""
        check = self.style_checker.check_style_violations(text)
        
        # Need LLM if there are still strict violations
        if check["strict_count"] > 0:
            return True
        
        # Need LLM if score is low
        if check["score"] < 70:
            return True
        
        # Need LLM if text is too long for platform
        platform = guidelines.get("metadata", {}).get("platform", "")
        if platform == "twitter" and len(text) > 280:
            return True
        
        return False
    
    def _llm_rewrite(self, text: str, voice: Dict[str, Any], 
                     guidelines: Dict[str, Any]) -> str:
        """Use LLM to rewrite text with constraints"""
        platform = guidelines.get("metadata", {}).get("platform", "general")
        intent = guidelines.get("metadata", {}).get("intent", "general")
        
        # Build prompt for LLM
        prompt = self._build_rewrite_prompt(text, voice, guidelines, platform, intent)
        
        try:
            response = self.llm_client.call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract rewritten text from response
            rewritten_text = self._extract_rewritten_text(response)
            
            # Validate the rewrite
            if self._validate_rewrite(rewritten_text, voice, guidelines):
                return rewritten_text
            else:
                # Fallback to original if rewrite is invalid
                return text
                
        except Exception as e:
            print(f"LLM rewrite failed: {e}")
            return text
    
    def _build_rewrite_prompt(self, text: str, voice: Dict[str, Any], 
                             guidelines: Dict[str, Any], platform: str, intent: str) -> str:
        """Build prompt for LLM rewrite"""
        prompt_parts = [
            f"Rewrite the following text for {platform} with intent '{intent}':",
            "",
            f"Original text:",
            text,
            "",
            "Requirements:",
            "- Remove AI fingerprints and generic language",
            "- Maintain the original meaning and key points",
            "- Follow platform guidelines",
            "- Use natural, conversational tone"
        ]
        
        # Add voice constraints
        if "tone_axes" in voice:
            axes = voice["tone_axes"]
            prompt_parts.append(f"- Tone: {self._describe_tone_axes(axes)}")
        
        if "strict_forbiddens" in voice:
            forbidden = voice["strict_forbiddens"]
            prompt_parts.append(f"- Never use these terms: {', '.join(forbidden)}")
        
        # Add platform-specific requirements
        if platform == "twitter":
            prompt_parts.append("- Keep under 280 characters")
        elif platform == "linkedin":
            prompt_parts.append("- Professional, thought-leadership tone")
        elif platform == "instagram":
            prompt_parts.append("- Authentic, storytelling tone")
        
        prompt_parts.append("")
        prompt_parts.append("Rewritten text:")
        
        return "\n".join(prompt_parts)
    
    def _describe_tone_axes(self, axes: Dict[str, float]) -> str:
        """Describe tone axes in human-readable format"""
        descriptions = []
        
        if axes.get("formality", 0.5) > 0.7:
            descriptions.append("formal")
        elif axes.get("formality", 0.5) < 0.3:
            descriptions.append("casual")
        
        if axes.get("enthusiasm", 0.5) > 0.7:
            descriptions.append("enthusiastic")
        elif axes.get("enthusiasm", 0.5) < 0.3:
            descriptions.append("reserved")
        
        if axes.get("technicality", 0.5) > 0.7:
            descriptions.append("technical")
        elif axes.get("technicality", 0.5) < 0.3:
            descriptions.append("simple")
        
        if axes.get("authority", 0.5) > 0.7:
            descriptions.append("authoritative")
        elif axes.get("authority", 0.5) < 0.3:
            descriptions.append("humble")
        
        return ", ".join(descriptions) if descriptions else "balanced"
    
    def _extract_rewritten_text(self, response: str) -> str:
        """Extract rewritten text from LLM response"""
        # Look for the rewritten text after "Rewritten text:"
        if "Rewritten text:" in response:
            parts = response.split("Rewritten text:")
            if len(parts) > 1:
                return parts[1].strip()
        
        # If no clear marker, return the response as-is
        return response.strip()
    
    def _validate_rewrite(self, rewritten_text: str, voice: Dict[str, Any], 
                         guidelines: Dict[str, Any]) -> bool:
        """Validate that rewrite meets requirements"""
        # Check for style violations
        check = self.style_checker.check_style_violations(rewritten_text)
        
        # Must pass style check
        if not check["passed"]:
            return False
        
        # Check platform requirements
        platform = guidelines.get("metadata", {}).get("platform", "")
        if platform == "twitter" and len(rewritten_text) > 280:
            return False
        
        # Check for forbidden terms
        if "strict_forbiddens" in voice:
            forbidden_terms = voice["strict_forbiddens"]
            for term in forbidden_terms:
                if term.lower() in rewritten_text.lower():
                    return False
        
        return True
    
    def _safe_rewrite(self, text: str, voice: Dict[str, Any], 
                     guidelines: Dict[str, Any]) -> str:
        """Perform a safe rewrite that prioritizes avoiding violations"""
        # Apply only the safest automatic fixes
        safe_text = text
        
        # Fix em-dashes (always safe)
        safe_text = re.sub(r'—', '–', safe_text)
        
        # Remove excessive qualifiers (usually safe)
        safe_text = re.sub(r'\bvery\s+(\w+)\b', r'\1', safe_text, flags=re.IGNORECASE)
        safe_text = re.sub(r'\breally\s+(\w+)\b', r'\1', safe_text, flags=re.IGNORECASE)
        
        # Replace forbidden terms with safe alternatives
        if "strict_forbiddens" in voice:
            safe_text = self._replace_forbidden_terms(safe_text, voice["strict_forbiddens"])
        
        return safe_text
    
    def _apply_minor_improvements(self, text: str, voice: Dict[str, Any], 
                                 guidelines: Dict[str, Any]) -> str:
        """Apply minor improvements to already good text"""
        improved_text = text
        
        # Apply gentle improvements
        improved_text = self._apply_automatic_fixes(improved_text, voice)
        
        # Ensure platform compliance
        platform = guidelines.get("metadata", {}).get("platform", "")
        if platform == "twitter" and len(improved_text) > 280:
            # Truncate safely
            improved_text = improved_text[:277] + "..."
        
        return improved_text
    
    def get_rewrite_summary(self, original_text: str, rewritten_text: str, 
                           voice: Dict[str, Any], guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of rewrite changes"""
        original_check = self.style_checker.check_style_violations(original_text)
        rewritten_check = self.style_checker.check_style_violations(rewritten_text)
        
        return {
            "original_score": original_check["score"],
            "rewritten_score": rewritten_check["score"],
            "score_improvement": rewritten_check["score"] - original_check["score"],
            "original_violations": original_check["total_violations"],
            "rewritten_violations": rewritten_check["total_violations"],
            "violations_fixed": original_check["total_violations"] - rewritten_check["total_violations"],
            "original_length": len(original_text),
            "rewritten_length": len(rewritten_text),
            "length_change": len(rewritten_text) - len(original_text),
            "platform": guidelines.get("metadata", {}).get("platform", "unknown"),
            "intent": guidelines.get("metadata", {}).get("intent", "general")
        }

# Global rewriter instance
_rewriter = None

def get_constrained_rewriter() -> ConstrainedRewriter:
    """Get or create global constrained rewriter instance"""
    global _rewriter
    if _rewriter is None:
        _rewriter = ConstrainedRewriter()
    return _rewriter

def rewrite_with_constraints(text: str, voice: Dict[str, Any], 
                            guidelines: Dict[str, Any]) -> str:
    """Rewrite text to remove AI fingerprints while respecting constraints"""
    rewriter = get_constrained_rewriter()
    return rewriter.rewrite_with_constraints(text, voice, guidelines)

if __name__ == "__main__":
    # Test the constrained rewriter
    rewriter = ConstrainedRewriter()
    
    # Test text with AI fingerprints
    test_text = """
    In today's fast-paced world, our revolutionary solution—cutting-edge technology that will game-change your business!
    
    Furthermore, it's not just another tool; it's the future of productivity. Additionally, it's really very extremely powerful.
    
    Here's what you need to know:
    First, we provide innovative solutions.
    Second, we deliver exceptional results.
    Third, we exceed expectations.
    
    In conclusion, this is the best solution available.
    """
    
    # Test voice constraints
    voice_constraints = {
        "tone_axes": {
            "formality": 0.7,
            "enthusiasm": 0.6,
            "technicality": 0.4,
            "authority": 0.7
        },
        "strict_forbiddens": ["revolutionary", "cutting-edge", "game-changing"],
        "required_phrases": ["empowering businesses"]
    }
    
    # Test guidelines
    guidelines = {
        "metadata": {
            "platform": "linkedin",
            "intent": "thought leadership"
        },
        "limits": {
            "max_length": 3000
        }
    }
    
    # Perform rewrite
    rewritten_text = rewriter.rewrite_with_constraints(test_text, voice_constraints, guidelines)
    
    print("Original text:")
    print(test_text)
    print("\nRewritten text:")
    print(rewritten_text)
    
    # Get rewrite summary
    summary = rewriter.get_rewrite_summary(test_text, rewritten_text, voice_constraints, guidelines)
    print(f"\nRewrite Summary:")
    print(f"Score improvement: {summary['score_improvement']}")
    print(f"Violations fixed: {summary['violations_fixed']}")
    print(f"Length change: {summary['length_change']} characters")
