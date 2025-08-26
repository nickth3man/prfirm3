# Configuration Management

The Virtual PR Firm supports comprehensive configuration management through environment variables and the `config.py` module.

## Node Configuration

Each node in the pipeline can be configured with the following parameters:

### Available Nodes

- `engagement_manager` - Collects and normalizes user inputs
- `brand_bible_ingest` - Processes brand bible content
- `voice_alignment` - Aligns content with brand voice
- `content_craftsman` - Generates initial content
- `style_editor` - Refines and edits content
- `style_compliance` - Validates style compliance
- `agency_director` - Final approval and routing

### Node Parameters

Each node supports these configuration options:

- `max_retries` - Maximum number of retry attempts on failure (default varies by node)
- `wait` - Wait time in seconds between retries (default varies by node)

## Environment Variables

### Node Configuration

Override node parameters using environment variables:

```bash
# Format: PRFIRM3_{NODE_NAME}_{PARAMETER}
export PRFIRM3_CONTENT_CRAFTSMAN_MAX_RETRIES=5
export PRFIRM3_CONTENT_CRAFTSMAN_WAIT=3
export PRFIRM3_STYLE_EDITOR_MAX_RETRIES=2
```

### Global Settings

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR)
export PRFIRM3_LOG_LEVEL=DEBUG

# Enable streaming updates
export PRFIRM3_ENABLE_STREAMING=true
```

### Gradio Interface Settings

```bash
# Server port for Gradio interface
export PRFIRM3_GRADIO_PORT=8080

# Enable public sharing
export PRFIRM3_GRADIO_SHARE=true

# Basic authentication (username:password)
export PRFIRM3_GRADIO_AUTH=admin:secret123
```

## Default Configuration

### Node Defaults

```python
{
    "engagement_manager": {"max_retries": 2, "wait": 0},
    "brand_bible_ingest": {"max_retries": 2, "wait": 0},
    "voice_alignment": {"max_retries": 2, "wait": 0},
    "content_craftsman": {"max_retries": 3, "wait": 2},
    "style_editor": {"max_retries": 3, "wait": 1},
    "style_compliance": {"max_retries": 2, "wait": 0},
    "agency_director": {"max_retries": 1, "wait": 0}
}
```

### Global Defaults

- Log Level: `INFO`
- Streaming: `false`
- Gradio Port: `7860`
- Gradio Share: `false`
- Gradio Auth: `None`

## Usage Examples

### Running with Custom Configuration

```bash
# High retry configuration for production
export PRFIRM3_CONTENT_CRAFTSMAN_MAX_RETRIES=5
export PRFIRM3_CONTENT_CRAFTSMAN_WAIT=5
export PRFIRM3_STYLE_EDITOR_MAX_RETRIES=4
export PRFIRM3_STYLE_EDITOR_WAIT=3

python main.py
```

### Debug Mode

```bash
# Enable debug logging
export PRFIRM3_LOG_LEVEL=DEBUG

python main.py
```

### Secure Gradio Deployment

```bash
# Run on custom port with authentication
export PRFIRM3_GRADIO_PORT=8080
export PRFIRM3_GRADIO_AUTH=admin:secure_password_123

python main.py gradio
```

## Programmatic Configuration

You can also access configuration programmatically:

```python
from config import get_config, get_node_config

# Get full configuration
config = get_config()
print(f"Gradio port: {config.gradio_server_port}")

# Get specific node configuration
craftsman_config = get_node_config("content_craftsman")
print(f"Craftsman retries: {craftsman_config['max_retries']}")
```

## Configuration Validation

The configuration system automatically validates:

- Integer values for retries and wait times
- Valid log levels
- Port number ranges
- Authentication format

Invalid values will fall back to defaults with warning messages in the logs.