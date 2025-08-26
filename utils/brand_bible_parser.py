import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
import logging
from lxml import etree
import re

class BrandBibleParser:
    """Parser for Brand Bible XML with validation and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Required fields for validation
        self.required_fields = [
            "identity/name",
            "identity/voice/tone",
            "messaging/platforms"
        ]
        
        # Optional fields that enhance functionality
        self.optional_fields = [
            "identity/voice/forbidden_terms",
            "identity/voice/required_phrases",
            "identity/voice/style_guidelines",
            "messaging/platforms/linkedin",
            "messaging/platforms/twitter",
            "messaging/platforms/instagram",
            "messaging/platforms/reddit",
            "messaging/platforms/email",
            "messaging/platforms/blog"
        ]
    
    def parse(self, xml_str: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parse Brand Bible XML and return parsed data with missing fields
        
        Args:
            xml_str: XML string content
            
        Returns:
            Tuple of (parsed_data, missing_fields)
        """
        try:
            # First try with lxml for better error handling
            return self._parse_with_lxml(xml_str)
        except Exception as e:
            self.logger.warning(f"lxml parsing failed, falling back to ElementTree: {e}")
            return self._parse_with_elementtree(xml_str)
    
    def _parse_with_lxml(self, xml_str: str) -> Tuple[Dict[str, Any], List[str]]:
        """Parse using lxml for better error handling"""
        try:
            # Parse XML with lxml
            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.fromstring(xml_str.encode('utf-8'), parser)
            
            # Convert to dictionary
            parsed_data = self._element_to_dict(root)
            
            # Validate and find missing fields
            missing_fields = self._validate_fields(parsed_data)
            
            return parsed_data, missing_fields
            
        except etree.XMLSyntaxError as e:
            self.logger.error(f"XML syntax error: {e}")
            raise ValueError(f"Invalid XML format: {e}")
        except Exception as e:
            self.logger.error(f"lxml parsing error: {e}")
            raise
    
    def _parse_with_elementtree(self, xml_str: str) -> Tuple[Dict[str, Any], List[str]]:
        """Parse using ElementTree as fallback"""
        try:
            # Parse XML with ElementTree
            root = ET.fromstring(xml_str)
            
            # Convert to dictionary
            parsed_data = self._element_to_dict(root)
            
            # Validate and find missing fields
            missing_fields = self._validate_fields(parsed_data)
            
            return parsed_data, missing_fields
            
        except ET.ParseError as e:
            self.logger.error(f"XML parse error: {e}")
            raise ValueError(f"Invalid XML format: {e}")
        except Exception as e:
            self.logger.error(f"ElementTree parsing error: {e}")
            raise
    
    def _element_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Handle attributes
        if element.attrib:
            result["@attributes"] = dict(element.attrib)
        
        # Handle text content
        if element.text and element.text.strip():
            result["text"] = element.text.strip()
        
        # Handle child elements
        for child in element:
            tag = child.tag
            child_dict = self._element_to_dict(child)
            
            # Handle lists (multiple elements with same tag)
            if tag in result:
                if not isinstance(result[tag], list):
                    result[tag] = [result[tag]]
                result[tag].append(child_dict)
            else:
                result[tag] = child_dict
        
        return result
    
    def _validate_fields(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Validate required fields and return missing ones"""
        missing_fields = []
        
        for field_path in self.required_fields:
            if not self._field_exists(parsed_data, field_path):
                missing_fields.append(field_path)
        
        return missing_fields
    
    def _field_exists(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if a field exists in the parsed data"""
        path_parts = field_path.split('/')
        current = data
        
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        return True
    
    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> Optional[Any]:
        """Get field value from parsed data"""
        path_parts = field_path.split('/')
        current = data
        
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def extract_voice_guidelines(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract voice and tone guidelines from parsed data"""
        voice_data = {}
        
        # Extract tone
        tone_path = "identity/voice/tone"
        if self._field_exists(parsed_data, tone_path):
            tone_value = self._get_field_value(parsed_data, tone_path)
            if isinstance(tone_value, dict) and "text" in tone_value:
                voice_data["tone"] = tone_value["text"]
            elif isinstance(tone_value, str):
                voice_data["tone"] = tone_value
        
        # Extract forbidden terms
        forbidden_path = "identity/voice/forbidden_terms"
        if self._field_exists(parsed_data, forbidden_path):
            forbidden_value = self._get_field_value(parsed_data, forbidden_path)
            voice_data["forbidden_terms"] = self._extract_term_list(forbidden_value)
        
        # Extract required phrases
        required_path = "identity/voice/required_phrases"
        if self._field_exists(parsed_data, required_path):
            required_value = self._get_field_value(parsed_data, required_path)
            voice_data["required_phrases"] = self._extract_term_list(required_value)
        
        # Extract style guidelines
        style_path = "identity/voice/style_guidelines"
        if self._field_exists(parsed_data, style_path):
            style_value = self._get_field_value(parsed_data, style_path)
            voice_data["style_guidelines"] = self._extract_guidelines(style_value)
        
        return voice_data
    
    def extract_platform_guidelines(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract platform-specific guidelines from parsed data"""
        platform_data = {}
        
        platforms_path = "messaging/platforms"
        if not self._field_exists(parsed_data, platforms_path):
            return platform_data
        
        platforms = self._get_field_value(parsed_data, platforms_path)
        if not isinstance(platforms, dict):
            return platform_data
        
        for platform_name, platform_config in platforms.items():
            if isinstance(platform_config, dict):
                platform_data[platform_name] = {
                    "style": self._extract_style(platform_config),
                    "hashtag_usage": self._extract_hashtag_usage(platform_config),
                    "content_rules": self._extract_content_rules(platform_config)
                }
        
        return platform_data
    
    def _extract_term_list(self, value: Any) -> List[str]:
        """Extract list of terms from XML value"""
        terms = []
        
        if isinstance(value, dict):
            # Handle term elements
            if "term" in value:
                if isinstance(value["term"], list):
                    for term in value["term"]:
                        if isinstance(term, dict) and "text" in term:
                            terms.append(term["text"])
                        elif isinstance(term, str):
                            terms.append(term)
                elif isinstance(value["term"], dict) and "text" in value["term"]:
                    terms.append(value["term"]["text"])
                elif isinstance(value["term"], str):
                    terms.append(value["term"])
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
    
    def _extract_style(self, platform_config: Dict[str, Any]) -> str:
        """Extract style from platform configuration"""
        if "style" in platform_config:
            style_value = platform_config["style"]
            if isinstance(style_value, dict) and "text" in style_value:
                return style_value["text"]
            elif isinstance(style_value, str):
                return style_value
        return ""
    
    def _extract_hashtag_usage(self, platform_config: Dict[str, Any]) -> str:
        """Extract hashtag usage from platform configuration"""
        if "hashtag_usage" in platform_config:
            hashtag_value = platform_config["hashtag_usage"]
            if isinstance(hashtag_value, dict) and "text" in hashtag_value:
                return hashtag_value["text"]
            elif isinstance(hashtag_value, str):
                return hashtag_value
        return ""
    
    def _extract_content_rules(self, platform_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content rules from platform configuration"""
        rules = {}
        
        for key, value in platform_config.items():
            if key not in ["style", "hashtag_usage"]:
                if isinstance(value, dict) and "text" in value:
                    rules[key] = value["text"]
                elif isinstance(value, str):
                    rules[key] = value
        
        return rules
    
    def validate_xml_structure(self, xml_str: str) -> Tuple[bool, List[str]]:
        """Validate XML structure and return validation results"""
        errors = []
        
        try:
            # Try to parse with lxml for detailed error reporting
            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.fromstring(xml_str.encode('utf-8'), parser)
            
            # Check for required root element
            if root.tag != "brand_bible":
                errors.append("Root element must be 'brand_bible'")
            
            # Check for required child elements
            required_children = ["identity", "messaging"]
            for child_name in required_children:
                if root.find(child_name) is None:
                    errors.append(f"Missing required child element: {child_name}")
            
            # Check identity structure
            identity = root.find("identity")
            if identity is not None:
                if identity.find("name") is None:
                    errors.append("Missing 'name' element in identity")
                if identity.find("voice") is None:
                    errors.append("Missing 'voice' element in identity")
                else:
                    voice = identity.find("voice")
                    if voice.find("tone") is None:
                        errors.append("Missing 'tone' element in voice")
            
            # Check messaging structure
            messaging = root.find("messaging")
            if messaging is not None:
                if messaging.find("platforms") is None:
                    errors.append("Missing 'platforms' element in messaging")
            
            return len(errors) == 0, errors
            
        except etree.XMLSyntaxError as e:
            errors.append(f"XML syntax error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def create_sample_xml(self) -> str:
        """Create a sample Brand Bible XML structure"""
        sample_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<brand_bible version="1.0">
    <identity>
        <name>TechFlow Solutions</name>
        <voice>
            <tone>professional, approachable, innovative</tone>
            <forbidden_terms>
                <term>cutting-edge</term>
                <term>revolutionary</term>
                <term>game-changing</term>
            </forbidden_terms>
            <required_phrases>
                <phrase>empowering businesses</phrase>
                <phrase>streamlined solutions</phrase>
            </required_phrases>
            <style_guidelines>
                <guideline>Use active voice</guideline>
                <guideline>Avoid jargon</guideline>
                <guideline>Focus on benefits</guideline>
            </style_guidelines>
        </voice>
    </identity>
    <messaging>
        <platforms>
            <linkedin>
                <style>formal, thought-leadership, benefit-focused</style>
                <hashtag_usage>minimal, relevant</hashtag_usage>
                <content_length>1000-3000 characters</content_length>
            </linkedin>
            <twitter>
                <style>conversational, engaging, concise</style>
                <hashtag_usage>strategic, trending</hashtag_usage>
                <content_length>280 characters max</content_length>
            </twitter>
            <instagram>
                <style>visual, storytelling, authentic</style>
                <hashtag_usage>relevant, branded</hashtag_usage>
                <content_length>2200 characters max</content_length>
            </instagram>
            <reddit>
                <style>community-focused, helpful, authentic</style>
                <hashtag_usage>none</hashtag_usage>
                <content_length>flexible</content_length>
            </reddit>
            <email>
                <style>professional, clear, action-oriented</style>
                <hashtag_usage>none</hashtag_usage>
                <content_length>brief and focused</content_length>
            </email>
            <blog>
                <style>detailed, educational, SEO-optimized</style>
                <hashtag_usage>none</hashtag_usage>
                <content_length>1000-3000 words</content_length>
            </blog>
        </platforms>
    </messaging>
</brand_bible>'''
        return sample_xml

# Global parser instance
_parser = None

def get_brand_bible_parser() -> BrandBibleParser:
    """Get or create global brand bible parser instance"""
    global _parser
    if _parser is None:
        _parser = BrandBibleParser()
    return _parser

def parse_brand_bible(xml_str: str) -> Tuple[Dict[str, Any], List[str]]:
    """Parse Brand Bible XML and return parsed data with missing fields"""
    parser = get_brand_bible_parser()
    return parser.parse(xml_str)

def validate_brand_bible(xml_str: str) -> Tuple[bool, List[str]]:
    """Validate Brand Bible XML structure"""
    parser = get_brand_bible_parser()
    return parser.validate_xml_structure(xml_str)

if __name__ == "__main__":
    # Test the brand bible parser
    parser = BrandBibleParser()
    
    # Test with sample XML
    sample_xml = parser.create_sample_xml()
    print("Sample XML created successfully")
    
    # Test parsing
    parsed_data, missing_fields = parser.parse(sample_xml)
    print(f"Parsed successfully. Missing fields: {missing_fields}")
    
    # Test validation
    is_valid, errors = parser.validate_xml_structure(sample_xml)
    print(f"Validation result: {is_valid}")
    if errors:
        print(f"Validation errors: {errors}")
    
    # Test voice extraction
    voice_guidelines = parser.extract_voice_guidelines(parsed_data)
    print(f"Voice guidelines: {voice_guidelines}")
    
    # Test platform extraction
    platform_guidelines = parser.extract_platform_guidelines(parsed_data)
    print(f"Platform guidelines: {platform_guidelines}")
    
    # Test with invalid XML
    try:
        invalid_xml = "<invalid>xml</invalid>"
        parsed_data, missing_fields = parser.parse(invalid_xml)
        print("Should have failed with invalid XML")
    except ValueError as e:
        print(f"Correctly caught invalid XML: {e}")
