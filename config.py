"""Configuration management for the Virtual PR Firm.

This module provides centralized configuration management with support for:
- Environment variable overrides
- Default configurations
- Validation of configuration values
- Easy configuration loading and management
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    """Configuration for a single node."""
    max_retries: int = 1
    wait: int = 0

@dataclass
class FlowConfig:
    """Configuration for the entire flow system."""
    # Node configurations
    engagement_manager: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=2, wait=0))
    brand_bible_ingest: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=2, wait=0))
    voice_alignment: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=2, wait=0))
    content_craftsman: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=3, wait=2))
    style_editor: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=3, wait=1))
    style_compliance: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=2, wait=0))
    agency_director: NodeConfig = field(default_factory=lambda: NodeConfig(max_retries=1, wait=0))
    
    # Global settings
    log_level: str = "INFO"
    enable_streaming: bool = False
    default_platforms: list = field(default_factory=lambda: ["twitter", "linkedin"])
    
    # Gradio interface settings
    gradio_server_port: int = 7860
    gradio_share: bool = False
    gradio_auth: Optional[tuple] = None

class ConfigManager:
    """Manages configuration loading and environment variable overrides."""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> FlowConfig:
        """Load configuration with environment variable overrides."""
        config = FlowConfig()
        
        # Load node configurations
        for node_name in [
            "engagement_manager", "brand_bible_ingest", "voice_alignment",
            "content_craftsman", "style_editor", "style_compliance", "agency_director"
        ]:
            node_config = self._load_node_config(node_name)
            setattr(config, node_name, node_config)
        
        # Load global settings
        config.log_level = os.getenv("PRFIRM3_LOG_LEVEL", config.log_level)
        config.enable_streaming = os.getenv("PRFIRM3_ENABLE_STREAMING", "false").lower() == "true"
        
        # Load Gradio settings
        try:
            config.gradio_server_port = int(os.getenv("PRFIRM3_GRADIO_PORT", str(config.gradio_server_port)))
        except ValueError:
            log.warning("Invalid PRFIRM3_GRADIO_PORT value, using default: %d", config.gradio_server_port)
        
        config.gradio_share = os.getenv("PRFIRM3_GRADIO_SHARE", "false").lower() == "true"
        
        # Handle Gradio auth
        gradio_auth = os.getenv("PRFIRM3_GRADIO_AUTH")
        if gradio_auth and ":" in gradio_auth:
            username, password = gradio_auth.split(":", 1)
            config.gradio_auth = (username, password)
            log.info("Gradio authentication enabled")
        
        log.info("Configuration loaded successfully")
        return config
    
    def _load_node_config(self, node_name: str) -> NodeConfig:
        """Load configuration for a specific node."""
        env_prefix = f"PRFIRM3_{node_name.upper()}"
        
        # Get default config
        default_configs = {
            "engagement_manager": NodeConfig(max_retries=2, wait=0),
            "brand_bible_ingest": NodeConfig(max_retries=2, wait=0),
            "voice_alignment": NodeConfig(max_retries=2, wait=0),
            "content_craftsman": NodeConfig(max_retries=3, wait=2),
            "style_editor": NodeConfig(max_retries=3, wait=1),
            "style_compliance": NodeConfig(max_retries=2, wait=0),
            "agency_director": NodeConfig(max_retries=1, wait=0),
        }
        
        node_config = default_configs.get(node_name, NodeConfig())
        
        # Check for environment overrides
        max_retries_env = os.getenv(f"{env_prefix}_MAX_RETRIES")
        if max_retries_env:
            try:
                node_config.max_retries = int(max_retries_env)
                log.info("Environment override for %s max_retries: %d", node_name, node_config.max_retries)
            except ValueError:
                log.warning("Invalid %s_MAX_RETRIES value: %s", env_prefix, max_retries_env)
        
        wait_env = os.getenv(f"{env_prefix}_WAIT")
        if wait_env:
            try:
                node_config.wait = int(wait_env)
                log.info("Environment override for %s wait: %d", node_name, node_config.wait)
            except ValueError:
                log.warning("Invalid %s_WAIT value: %s", env_prefix, wait_env)
        
        return node_config
    
    def get_node_config(self, node_name: str) -> Dict[str, Any]:
        """Get configuration dictionary for a node."""
        node_config = getattr(self.config, node_name, NodeConfig())
        return {
            "max_retries": node_config.max_retries,
            "wait": node_config.wait
        }
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        log.info("Logging configured with level: %s", self.config.log_level)

# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> FlowConfig:
    """Get the global configuration instance."""
    return config_manager.config

def get_node_config(node_name: str) -> Dict[str, Any]:
    """Get configuration for a specific node."""
    return config_manager.get_node_config(node_name)