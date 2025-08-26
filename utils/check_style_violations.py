import re
from typing import Dict, List, Any, Tuple
import logging

class StyleViolationChecker:
    """Detects style violations and AI fingerprints with strict enforcement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # The 7 Deadly Sins - strict violations that must be fixed
        self.strict_violations = {
            "em_dash": {
                "pattern": r'—',
                "description": "Em-dash (—) is forbidden",
                "severity": "strict",
                "suggestion": "Replace with en-dash (–) or hyphen (-)"
            },
            "rhetorical_contrasts": {
                "patterns": [
                    r'\bnot\s+just\s+\w+[^.]*;\s*it\'s\s+\w+',
                    r'\bnot\s+\w+[^.]*,\s+but\s+\w+',
                    r'\bnot\s+\w+[^.]*\.\s*Instead[^.]*\.',
                    r'\bnot\s+\w+[^.]*\.\s*Rather[^.]*\.'
                ],
                "description": "Rhetorical contrasts are forbidden",
                "severity": "strict",
                "suggestion": "Use direct statements instead of contrasts"
            },
            "overused_terms": {
                "terms": [
                    "revolutionary",
                    "cutting-edge",
                    "game-changing",
                    "disruptive",
                    "paradigm shift",
                    "next-generation",
                    "breakthrough",
                    "innovative solution"
                ],
                "description": "Overused marketing terms are forbidden",
                "severity": "strict",
                "suggestion": "Use more specific, descriptive language"
            },
            "excessive_hashtags": {
                "pattern": r'#\w+',
                "max_count": 5,
                "description": "Too many hashtags",
                "severity": "strict",
                "suggestion": "Limit hashtags to 3-5 relevant ones"
            },
            "generic_closings": {
                "patterns": [
                    r'\bIn conclusion[^.]*\.',
                    r'\bTo summarize[^.]*\.',
                    r'\bIn summary[^.]*\.',
                    r'\bTo wrap up[^.]*\.'
                ],
                "description": "Generic closing phrases are forbidden",
                "severity": "strict",
                "suggestion": "Use specific, action-oriented closings"
            }
        }
        
        # Soft violations - AI fingerprints that should be avoided
        self.soft_violations = {
            "predictable_structure": {
                "patterns": [
                    r'^\d+\.\s+\w+[^.]*\.\s*\n\d+\.\s+\w+[^.]*\.\s*\n\d+\.\s+\w+[^.]*\.',
                    r'First[^.]*\.\s*Second[^.]*\.\s*Third[^.]*\.',
                    r'Let\'s\s+explore[^.]*\.\s*Here\'s\s+how[^.]*\.\s*In\s+summary[^.]*\.'
                ],
                "description": "Predictable list structure",
                "severity": "soft",
                "suggestion": "Vary sentence structure and flow"
            },
            "stiff_transitions": {
                "patterns": [
                    r'\bFurthermore[^.]*\.',
                    r'\bMoreover[^.]*\.',
                    r'\bAdditionally[^.]*\.',
                    r'\bIn addition[^.]*\.'
                ],
                "description": "Stiff academic transitions",
                "severity": "soft",
                "suggestion": "Use natural, conversational transitions"
            },
            "monotone_rhythm": {
                "pattern": r'^[A-Z][^.]*\.\s*[A-Z][^.]*\.\s*[A-Z][^.]*\.',
                "description": "Monotone sentence rhythm",
                "severity": "soft",
                "suggestion": "Vary sentence length and structure"
            },
            "platitudes": {
                "patterns": [
                    r'\bIn today\'s\s+fast-paced\s+world[^.]*\.',
                    r'\bAs\s+we\s+all\s+know[^.]*\.',
                    r'\bIt\'s\s+no\s+secret\s+that[^.]*\.',
                    r'\bThe\s+truth\s+is[^.]*\.'
                ],
                "description": "Generic platitudes",
                "severity": "soft",
                "suggestion": "Use specific, concrete statements"
            },
            "excessive_qualifiers": {
                "patterns": [
                    r'\bvery\s+\w+',
                    r'\breally\s+\w+',
                    r'\bquite\s+\w+',
                    r'\bextremely\s+\w+'
                ],
                "description": "Excessive qualifiers weaken impact",
                "severity": "soft",
                "suggestion": "Use stronger, more specific words"
            }
        }
    
    def check_style_violations(self, text: str) -> Dict[str, Any]:
        """
        Check text for style violations
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with violations and score
        """
        violations = []
        strict_count = 0
        soft_count = 0
        
        # Check strict violations
        for violation_type, config in self.strict_violations.items():
            found_violations = self._check_violation_type(text, config, violation_type)
            violations.extend(found_violations)
            strict_count += len(found_violations)
        
        # Check soft violations
        for violation_type, config in self.soft_violations.items():
            found_violations = self._check_violation_type(text, config, violation_type)
            violations.extend(found_violations)
            soft_count += len(found_violations)
        
        # Calculate score (100 = perfect, 0 = many violations)
        total_violations = strict_count + soft_count
        base_score = 100
        
        # Strict violations heavily penalize score
        score_penalty = (strict_count * 15) + (soft_count * 5)
        final_score = max(0, base_score - score_penalty)
        
        return {
            "violations": violations,
            "score": final_score,
            "strict_count": strict_count,
            "soft_count": soft_count,
            "total_violations": total_violations,
            "passed": strict_count == 0,  # Must have no strict violations to pass
            "summary": self._generate_summary(violations, final_score)
        }
    
    def _check_violation_type(self, text: str, config: Dict[str, Any], violation_type: str) -> List[Dict[str, Any]]:
        """Check for a specific type of violation"""
        found_violations = []
        
        if "pattern" in config:
            # Single pattern
            matches = list(re.finditer(config["pattern"], text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                found_violations.append({
                    "type": violation_type,
                    "description": config["description"],
                    "severity": config["severity"],
                    "suggestion": config["suggestion"],
                    "position": match.start(),
                    "matched_text": match.group(),
                    "context": self._get_context(text, match.start(), match.end())
                })
        
        elif "patterns" in config:
            # Multiple patterns
            for pattern in config["patterns"]:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
                for match in matches:
                    found_violations.append({
                        "type": violation_type,
                        "description": config["description"],
                        "severity": config["severity"],
                        "suggestion": config["suggestion"],
                        "position": match.start(),
                        "matched_text": match.group(),
                        "context": self._get_context(text, match.start(), match.end())
                    })
        
        elif "terms" in config:
            # Term-based violations
            for term in config["terms"]:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    found_violations.append({
                        "type": violation_type,
                        "description": config["description"],
                        "severity": config["severity"],
                        "suggestion": config["suggestion"],
                        "position": match.start(),
                        "matched_text": match.group(),
                        "context": self._get_context(text, match.start(), match.end())
                    })
        
        # Handle special cases
        if violation_type == "excessive_hashtags":
            hashtag_matches = list(re.finditer(config["pattern"], text))
            if len(hashtag_matches) > config["max_count"]:
                found_violations.append({
                    "type": violation_type,
                    "description": f"{config['description']} ({len(hashtag_matches)} found, max {config['max_count']})",
                    "severity": config["severity"],
                    "suggestion": config["suggestion"],
                    "position": hashtag_matches[0].start(),
                    "matched_text": f"{len(hashtag_matches)} hashtags",
                    "context": "Multiple hashtags found throughout text"
                })
        
        return found_violations
    
    def _get_context(self, text: str, start: int, end: int, context_length: int = 50) -> str:
        """Get context around a violation"""
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)
        
        context = text[context_start:context_end]
        
        # Add ellipsis if we're not at the beginning/end
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context
    
    def _generate_summary(self, violations: List[Dict[str, Any]], score: float) -> str:
        """Generate a summary of violations"""
        if not violations:
            return f"Style check passed! Score: {score}/100"
        
        strict_violations = [v for v in violations if v["severity"] == "strict"]
        soft_violations = [v for v in violations if v["severity"] == "soft"]
        
        summary_parts = []
        
        if strict_violations:
            summary_parts.append(f"{len(strict_violations)} strict violation(s) found")
        
        if soft_violations:
            summary_parts.append(f"{len(soft_violations)} soft violation(s) found")
        
        summary_parts.append(f"Score: {score}/100")
        
        if strict_violations:
            summary_parts.append("Content needs revision")
        elif soft_violations:
            summary_parts.append("Content could be improved")
        else:
            summary_parts.append("Content meets style guidelines")
        
        return " | ".join(summary_parts)
    
    def get_violation_details(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed breakdown of violations"""
        violation_types = {}
        severity_breakdown = {"strict": 0, "soft": 0}
        
        for violation in violations:
            # Count by type
            violation_type = violation["type"]
            if violation_type not in violation_types:
                violation_types[violation_type] = 0
            violation_types[violation_type] += 1
            
            # Count by severity
            severity = violation["severity"]
            severity_breakdown[severity] += 1
        
        return {
            "violation_types": violation_types,
            "severity_breakdown": severity_breakdown,
            "total_violations": len(violations)
        }
    
    def suggest_improvements(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions based on violations"""
        suggestions = []
        seen_suggestions = set()
        
        for violation in violations:
            suggestion = violation["suggestion"]
            if suggestion not in seen_suggestions:
                suggestions.append(suggestion)
                seen_suggestions.add(suggestion)
        
        return suggestions
    
    def check_platform_specific(self, text: str, platform: str) -> Dict[str, Any]:
        """Check for platform-specific violations"""
        platform_violations = []
        
        if platform == "twitter":
            # Check Twitter-specific rules
            if len(text) > 280:
                platform_violations.append({
                    "type": "twitter_length",
                    "description": "Exceeds Twitter character limit",
                    "severity": "strict",
                    "suggestion": "Shorten content to 280 characters or less",
                    "position": 0,
                    "matched_text": f"{len(text)} characters",
                    "context": "Entire tweet"
                })
        
        elif platform == "linkedin":
            # Check LinkedIn-specific rules
            if len(text) < 1000:
                platform_violations.append({
                    "type": "linkedin_length",
                    "description": "LinkedIn posts should be 1000+ characters for better engagement",
                    "severity": "soft",
                    "suggestion": "Expand content to 1000+ characters",
                    "position": 0,
                    "matched_text": f"{len(text)} characters",
                    "context": "Entire post"
                })
        
        elif platform == "instagram":
            # Check Instagram-specific rules
            hashtag_count = len(re.findall(r'#\w+', text))
            if hashtag_count < 8:
                platform_violations.append({
                    "type": "instagram_hashtags",
                    "description": "Instagram posts benefit from 8-20 hashtags",
                    "severity": "soft",
                    "suggestion": "Add more relevant hashtags (8-20 recommended)",
                    "position": 0,
                    "matched_text": f"{hashtag_count} hashtags",
                    "context": "Entire caption"
                })
        
        return {
            "platform_violations": platform_violations,
            "platform": platform
        }

# Global checker instance
_checker = None

def get_style_violation_checker() -> StyleViolationChecker:
    """Get or create global style violation checker instance"""
    global _checker
    if _checker is None:
        _checker = StyleViolationChecker()
    return _checker

def check_style_violations(text: str) -> Dict[str, Any]:
    """Check text for style violations"""
    checker = get_style_violation_checker()
    return checker.check_style_violations(text)

def check_platform_specific(text: str, platform: str) -> Dict[str, Any]:
    """Check for platform-specific violations"""
    checker = get_style_violation_checker()
    return checker.check_platform_specific(text, platform)

if __name__ == "__main__":
    # Test the style violation checker
    checker = StyleViolationChecker()
    
    # Test text with various violations
    test_text = """
    In today's fast-paced world, our revolutionary solution—cutting-edge technology that will game-change your business!
    
    Not just another tool; it's the future of productivity. Furthermore, it's really very extremely powerful.
    
    Here's what you need to know:
    1. First, we provide innovative solutions.
    2. Second, we deliver exceptional results.
    3. Third, we exceed expectations.
    
    In conclusion, this is the best solution available. #innovation #technology #gamechanging #revolutionary #cuttingedge #breakthrough #nextgen #disruptive #paradigm #shift
    """
    
    # Check for violations
    result = checker.check_style_violations(test_text)
    print("Style Check Results:")
    print(f"Score: {result['score']}/100")
    print(f"Passed: {result['passed']}")
    print(f"Summary: {result['summary']}")
    
    print(f"\nViolations found: {result['total_violations']}")
    print(f"Strict violations: {result['strict_count']}")
    print(f"Soft violations: {result['soft_count']}")
    
    # Show violation details
    print("\nViolation Details:")
    for violation in result['violations']:
        print(f"- {violation['type']}: {violation['description']}")
        print(f"  Suggestion: {violation['suggestion']}")
        print(f"  Context: {violation['context']}")
        print()
    
    # Get improvement suggestions
    suggestions = checker.suggest_improvements(result['violations'])
    print("Improvement Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    # Test platform-specific checks
    print("\nPlatform-Specific Checks:")
    platforms = ["twitter", "linkedin", "instagram"]
    for platform in platforms:
        platform_result = checker.check_platform_specific(test_text, platform)
        print(f"{platform.title()}: {len(platform_result['platform_violations'])} violations")
