import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

class PresetsStore:
    """Manages local JSON persistence of presets"""
    
    def __init__(self, storage_dir: str = "presets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Preset file paths
        self.brand_bibles_file = self.storage_dir / "brand_bibles.json"
        self.email_signatures_file = self.storage_dir / "email_signatures.json"
        self.blog_styles_file = self.storage_dir / "blog_styles.json"
        
        # Initialize files if they don't exist
        self._init_files()
    
    def _init_files(self):
        """Initialize preset files if they don't exist"""
        if not self.brand_bibles_file.exists():
            self._save_json(self.brand_bibles_file, {})
        
        if not self.email_signatures_file.exists():
            self._save_json(self.email_signatures_file, {})
        
        if not self.blog_styles_file.exists():
            self._save_json(self.blog_styles_file, {})
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save JSON to file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Brand Bible methods
    def save_brand_bible(self, name: str, xml_content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a Brand Bible preset"""
        try:
            brand_bibles = self._load_json(self.brand_bibles_file)
            
            brand_bibles[name] = {
                "xml_content": xml_content,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self._save_json(self.brand_bibles_file, brand_bibles)
            return True
        except Exception as e:
            print(f"Error saving brand bible: {e}")
            return False
    
    def get_brand_bible(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a Brand Bible preset"""
        brand_bibles = self._load_json(self.brand_bibles_file)
        return brand_bibles.get(name)
    
    def list_brand_bibles(self) -> List[str]:
        """List all Brand Bible preset names"""
        brand_bibles = self._load_json(self.brand_bibles_file)
        return list(brand_bibles.keys())
    
    def delete_brand_bible(self, name: str) -> bool:
        """Delete a Brand Bible preset"""
        try:
            brand_bibles = self._load_json(self.brand_bibles_file)
            if name in brand_bibles:
                del brand_bibles[name]
                self._save_json(self.brand_bibles_file, brand_bibles)
                return True
            return False
        except Exception as e:
            print(f"Error deleting brand bible: {e}")
            return False
    
    # Email signature methods
    def save_email_signature(self, name: str, signature: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save an Email signature preset"""
        try:
            signatures = self._load_json(self.email_signatures_file)
            
            signatures[name] = {
                "signature": signature,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self._save_json(self.email_signatures_file, signatures)
            return True
        except Exception as e:
            print(f"Error saving email signature: {e}")
            return False
    
    def get_email_signature(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an Email signature preset"""
        signatures = self._load_json(self.email_signatures_file)
        return signatures.get(name)
    
    def list_email_signatures(self) -> List[str]:
        """List all Email signature preset names"""
        signatures = self._load_json(self.email_signatures_file)
        return list(signatures.keys())
    
    def delete_email_signature(self, name: str) -> bool:
        """Delete an Email signature preset"""
        try:
            signatures = self._load_json(self.email_signatures_file)
            if name in signatures:
                del signatures[name]
                self._save_json(self.email_signatures_file, signatures)
                return True
            return False
        except Exception as e:
            print(f"Error deleting email signature: {e}")
            return False
    
    # Blog style methods
    def save_blog_style(self, name: str, style: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a Blog style preset"""
        try:
            styles = self._load_json(self.blog_styles_file)
            
            styles[name] = {
                "style": style,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self._save_json(self.blog_styles_file, styles)
            return True
        except Exception as e:
            print(f"Error saving blog style: {e}")
            return False
    
    def get_blog_style(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a Blog style preset"""
        styles = self._load_json(self.blog_styles_file)
        return styles.get(name)
    
    def list_blog_styles(self) -> List[str]:
        """List all Blog style preset names"""
        styles = self._load_json(self.blog_styles_file)
        return list(styles.keys())
    
    def delete_blog_style(self, name: str) -> bool:
        """Delete a Blog style preset"""
        try:
            styles = self._load_json(self.blog_styles_file)
            if name in styles:
                del styles[name]
                self._save_json(self.blog_styles_file, styles)
                return True
            return False
        except Exception as e:
            print(f"Error deleting blog style: {e}")
            return False
    
    # General methods
    def list_all_presets(self) -> Dict[str, List[str]]:
        """List all presets by category"""
        return {
            "brand_bibles": self.list_brand_bibles(),
            "email_signatures": self.list_email_signatures(),
            "blog_styles": self.list_blog_styles()
        }
    
    def search_presets(self, query: str) -> Dict[str, List[str]]:
        """Search presets by name"""
        query_lower = query.lower()
        results = {
            "brand_bibles": [],
            "email_signatures": [],
            "blog_styles": []
        }
        
        # Search brand bibles
        for name in self.list_brand_bibles():
            if query_lower in name.lower():
                results["brand_bibles"].append(name)
        
        # Search email signatures
        for name in self.list_email_signatures():
            if query_lower in name.lower():
                results["email_signatures"].append(name)
        
        # Search blog styles
        for name in self.list_blog_styles():
            if query_lower in name.lower():
                results["blog_styles"].append(name)
        
        return results
    
    def export_presets(self, export_path: str) -> bool:
        """Export all presets to a single JSON file"""
        try:
            export_data = {
                "brand_bibles": self._load_json(self.brand_bibles_file),
                "email_signatures": self._load_json(self.email_signatures_file),
                "blog_styles": self._load_json(self.blog_styles_file),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error exporting presets: {e}")
            return False
    
    def import_presets(self, import_path: str) -> bool:
        """Import presets from a JSON file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import brand bibles
            if "brand_bibles" in import_data:
                brand_bibles = self._load_json(self.brand_bibles_file)
                brand_bibles.update(import_data["brand_bibles"])
                self._save_json(self.brand_bibles_file, brand_bibles)
            
            # Import email signatures
            if "email_signatures" in import_data:
                signatures = self._load_json(self.email_signatures_file)
                signatures.update(import_data["email_signatures"])
                self._save_json(self.email_signatures_file, signatures)
            
            # Import blog styles
            if "blog_styles" in import_data:
                styles = self._load_json(self.blog_styles_file)
                styles.update(import_data["blog_styles"])
                self._save_json(self.blog_styles_file, styles)
            
            return True
        except Exception as e:
            print(f"Error importing presets: {e}")
            return False

# Global presets store instance
_presets_store = None

def get_presets_store() -> PresetsStore:
    """Get or create global presets store instance"""
    global _presets_store
    if _presets_store is None:
        _presets_store = PresetsStore()
    return _presets_store

# Convenience functions
def save_brand_bible(name: str, xml_content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Save a Brand Bible preset"""
    store = get_presets_store()
    return store.save_brand_bible(name, xml_content, metadata)

def get_brand_bible(name: str) -> Optional[Dict[str, Any]]:
    """Get a Brand Bible preset"""
    store = get_presets_store()
    return store.get_brand_bible(name)

def list_brand_bibles() -> List[str]:
    """List all Brand Bible preset names"""
    store = get_presets_store()
    return store.list_brand_bibles()

def save_email_signature(name: str, signature: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Save an Email signature preset"""
    store = get_presets_store()
    return store.save_email_signature(name, signature, metadata)

def get_email_signature(name: str) -> Optional[Dict[str, Any]]:
    """Get an Email signature preset"""
    store = get_presets_store()
    return store.get_email_signature(name)

def list_email_signatures() -> List[str]:
    """List all Email signature preset names"""
    store = get_presets_store()
    return store.list_email_signatures()

def save_blog_style(name: str, style: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Save a Blog style preset"""
    store = get_presets_store()
    return store.save_blog_style(name, style, metadata)

def get_blog_style(name: str) -> Optional[Dict[str, Any]]:
    """Get a Blog style preset"""
    store = get_presets_store()
    return store.get_blog_style(name)

def list_blog_styles() -> List[str]:
    """List all Blog style preset names"""
    store = get_presets_store()
    return store.list_blog_styles()

if __name__ == "__main__":
    # Test the presets store
    store = PresetsStore()
    
    # Test Brand Bible presets
    print("Testing Brand Bible presets...")
    success = store.save_brand_bible("Test Brand", "<brand>Test content</brand>")
    print(f"Save success: {success}")
    
    bible = store.get_brand_bible("Test Brand")
    print(f"Retrieved: {bible}")
    
    bibles = store.list_brand_bibles()
    print(f"All brand bibles: {bibles}")
    
    # Test Email signature presets
    print("\nTesting Email signature presets...")
    success = store.save_email_signature("Professional", "Best regards,\nJohn Doe")
    print(f"Save success: {success}")
    
    signature = store.get_email_signature("Professional")
    print(f"Retrieved: {signature}")
    
    signatures = store.list_email_signatures()
    print(f"All signatures: {signatures}")
    
    # Test Blog style presets
    print("\nTesting Blog style presets...")
    success = store.save_blog_style("Technical", "Formal, detailed, code examples")
    print(f"Save success: {success}")
    
    style = store.get_blog_style("Technical")
    print(f"Retrieved: {style}")
    
    styles = store.list_blog_styles()
    print(f"All styles: {styles}")
    
    # Test search
    print("\nTesting search...")
    results = store.search_presets("Test")
    print(f"Search results: {results}")
    
    # Test export/import
    print("\nTesting export/import...")
    export_success = store.export_presets("test_export.json")
    print(f"Export success: {export_success}")
    
    # Clean up test data
    store.delete_brand_bible("Test Brand")
    store.delete_email_signature("Professional")
    store.delete_blog_style("Technical")
