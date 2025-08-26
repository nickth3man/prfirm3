# Structured Logging for PocketFlow

This project implements structured, configurable logging across PocketFlow nodes for improved observability and debugging.

## Features

✅ **Central logging configuration** with environment variables  
✅ **JSON structured logging** with consistent fields  
✅ **Node lifecycle logging** (prep, exec, post, retries, fallbacks)  
✅ **Correlation ID tracking** across node execution  
✅ **Configurable log levels and formats**  
✅ **Zero-impact integration** via LoggingMixin  

## Quick Start

### 1. Environment Configuration

Configure logging via environment variables:

```bash
# Log level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=INFO

# Log format (json, text)  
export LOG_FORMAT=json
```

### 2. Add Logging to Nodes

Use the `LoggingMixin` to add logging to existing nodes:

```python
from pocketflow import Node
from logging_mixin import LoggingMixin

class MyNode(LoggingMixin, Node):
    def exec(self, prep_res):
        # Your node logic here
        return "result"
    
    def post(self, shared, prep_res, exec_res):
        # Important: call super() to ensure logging happens
        return super().post(shared, prep_res, exec_res) or "default"
```

### 3. Set Correlation ID

Track requests across nodes with correlation IDs:

```python
import uuid

# Generate correlation ID
correlation_id = str(uuid.uuid4())

# Set on individual nodes
node.set_correlation_id(correlation_id)

# Or set on entire flow (recommended)
from flow_enhanced import create_qa_flow
flow = create_qa_flow(correlation_id)
```

## Log Output Examples

### JSON Format (default)

```json
{
  "timestamp": "2025-08-26T10:27:39.920175Z",
  "level": "INFO", 
  "message": "Node prep started",
  "node": "GetQuestionNode",
  "action": "prep",
  "correlation_id": "d316cde1-b04b-46b2-920d-133d7c63e963"
}
```

### Text Format

```
2025-08-26 10:24:02,153 - pocketflow.node.MyNode - INFO - Node prep started
```

## Log Fields

### Standard Fields (always present)
- `timestamp`: ISO 8601 UTC timestamp
- `level`: Log level (DEBUG, INFO, WARNING, ERROR) 
- `message`: Human-readable log message

### Contextual Fields (when available)
- `node`: Node class name
- `action`: Lifecycle phase (prep, exec, post, exec_retry, exec_fallback)
- `retry_count`: Current retry attempt (0-based)
- `correlation_id`: Request tracking ID
- `context`: Additional structured data
- `error`: Error message (for retry/fallback logs)

## Node Lifecycle Events

The logging system automatically captures these events:

| Event | Level | Description |
|-------|-------|-------------|
| `prep` start/end | INFO | Data preparation phase |
| `exec` start/end | INFO | Core execution logic |
| `exec_retry` | WARNING | Retry attempt after failure |
| `exec_fallback` | ERROR | Fallback execution after all retries |
| `post` start/end | INFO | Post-processing and result storage |

## Retry and Fallback Logging

Nodes with retry logic automatically log retry attempts and fallbacks:

```json
{
  "timestamp": "2025-08-26T10:25:59.203742Z",
  "level": "WARNING",
  "message": "Node exec retry attempt 0", 
  "node": "RetryNode",
  "action": "exec_retry",
  "retry_count": 0,
  "correlation_id": "test-retry-456",
  "error": "Simulated failure on attempt 1"
}
```

## Configuration Options

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR, CRITICAL | INFO | Minimum log level |
| `LOG_FORMAT` | json, text | json | Output format |

### Programmatic Configuration

```python
from logging_config import setup_logging
import os

# Configure via environment
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LOG_FORMAT"] = "json"

# Reinitialize logging
setup_logging()
```

## Testing

Run the comprehensive test suite:

```bash
# Run all logging tests
python tests/test_logging.py

# Test specific functionality
python test_basic_logging.py
python test_retry_logging.py
python test_fallback_logging.py
```

## Examples

### Basic Usage

```python
from nodes import GetQuestionNode, AnswerNode
import uuid

# Create nodes with logging
question_node = GetQuestionNode()
answer_node = AnswerNode()

# Set shared correlation ID
correlation_id = str(uuid.uuid4())
question_node.set_correlation_id(correlation_id)
answer_node.set_correlation_id(correlation_id)

# Run nodes - logs will be automatically generated
shared = {"question": "What is AI?"}
question_node.run(shared)
answer_node.run(shared)
```

### Production Flow

```python
from flow_enhanced import create_qa_flow
from logging_config import get_node_logger
import uuid

# Setup request tracking
correlation_id = str(uuid.uuid4())
logger = get_node_logger("RequestHandler", correlation_id)

logger.info("Processing user request")

# Create flow with shared correlation ID
flow = create_qa_flow(correlation_id)

# Execute with automatic logging
shared = {"question": "User question here"}
result = flow.run(shared)

logger.info("Request completed", context={"result": result})
```

## Advanced Usage

### Custom Context Data

Add structured context to logs:

```python
class MyNode(LoggingMixin, Node):
    def exec(self, prep_res):
        self.node_logger.info("Processing data", 
                            context={"input_size": len(prep_res)})
        return process_data(prep_res)
```

### Direct Logger Usage

Use the logger directly for custom events:

```python
from logging_config import get_node_logger

logger = get_node_logger("DataProcessor", correlation_id)
logger.debug("Starting data validation") 
logger.warning("Data quality issue detected", 
               context={"invalid_records": 5})
logger.error("Processing failed", context={"error_code": "E001"})
```

## Best Practices

1. **Always use correlation IDs** for request tracking
2. **Call super().post()** in post methods to ensure logging
3. **Add context data** to make logs more useful
4. **Use appropriate log levels** (DEBUG for development, INFO+ for production)
5. **Include error details** in context when logging failures
6. **Set LOG_FORMAT=text** for local development if preferred

## Troubleshooting

### No logs appearing
- Check `LOG_LEVEL` environment variable
- Ensure `setup_logging()` is called (automatic on first use)
- Verify nodes inherit from `LoggingMixin`

### Missing post logs
- Ensure `super().post()` is called in custom post methods
- Check method resolution order with `print(NodeClass.__mro__)`

### Inconsistent correlation IDs  
- Set correlation ID before running nodes
- Use enhanced flow creation: `create_qa_flow(correlation_id)`
- Verify correlation ID is passed to all nodes in the flow