# Virtual PR Firm

A comprehensive AI-powered content generation system for social media platforms, built with PocketFlow.

## 🚀 Features

### Core Functionality
- **Multi-platform content generation** for Twitter, LinkedIn, Facebook, Instagram, and more
- **Brand voice alignment** with customizable brand guidelines
- **Intelligent content optimization** for each platform's unique requirements
- **Real-time content generation** with streaming support

### Infrastructure Improvements
- **Comprehensive configuration management** with environment variable support
- **Robust input validation and sanitization** with security protections
- **Advanced logging system** with correlation ID tracking and structured output
- **Intelligent caching mechanism** with multi-tier support (memory + Redis)
- **Professional CLI interface** with multiple operation modes

### Security & Reliability
- **Rate limiting and abuse protection** for web interface
- **Input sanitization** to prevent injection attacks
- **Comprehensive error handling** with graceful degradation
- **Request tracking and observability** for debugging and monitoring

## 📦 Installation

### Prerequisites
- Python 3.8+
- Redis (optional, for distributed caching)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
# Copy and customize the configuration template
python -m cli config generate --output config.yaml

# Set your API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

## 🛠️ Usage

### Command Line Interface

The application provides a comprehensive CLI with multiple operation modes:

#### Start Web Interface
```bash
# Start the Gradio web interface
python -m cli serve --port 8080

# With authentication
python -m cli serve --port 8080 --auth admin:password

# With SSL
python -m cli serve --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

#### Run Content Generation
```bash
# Generate content for specific platforms
python -m cli run \
  --topic "Product launch announcement" \
  --platforms twitter,linkedin,facebook \
  --output-format json

# With brand bible file
python -m cli run \
  --topic "Company milestone" \
  --platforms twitter,linkedin \
  --brand-bible-file brand-guidelines.xml \
  --output-file results.json

# Dry run to validate inputs
python -m cli run \
  --topic "Test topic" \
  --platforms twitter \
  --dry-run
```

#### Configuration Management
```bash
# Generate configuration template
python -m cli config generate --output my-config.yaml

# Validate configuration file
python -m cli config validate --file config.yaml

# Show current configuration
python -m cli config show --format yaml
```

#### Data Validation
```bash
# Validate input data file
python -m cli validate --input data.json

# Validate with specific format
python -m cli validate --input data.yaml --format yaml
```

#### System Health
```bash
# Check system health
python -m cli health

# Get system information
python -m cli info
```

### Programmatic Usage

```python
from flow import create_main_flow
from validation import validate_shared_store
from config import get_config

# Load configuration
config = get_config()

# Prepare input data
shared = {
    "task_requirements": {
        "platforms": ["twitter", "linkedin"],
        "topic_or_goal": "Announce new product launch"
    },
    "brand_bible": {"xml_raw": ""}
}

# Validate inputs
validate_shared_store(shared)

# Create and run flow
flow = create_main_flow()
flow.run(shared)

# Get results
content_pieces = shared.get("content_pieces", {})
print(content_pieces)
```

## ⚙️ Configuration

The application uses a hierarchical configuration system:

1. **Default values** (built-in)
2. **Configuration file** (YAML/JSON)
3. **Environment variables** (override file settings)
4. **Command line arguments** (override all others)

### Configuration File Example

```yaml
# config.yaml
debug: false
environment: production

logging:
  level: INFO
  format: json
  file: logs/app.log
  max_size: 10485760
  backup_count: 5
  correlation_id: true

gradio:
  port: 7860
  host: 0.0.0.0
  share: false
  auth: null
  ssl_verify: true

llm:
  provider: openai
  model: gpt-4o
  temperature: 0.7
  max_tokens: 4000
  timeout: 60
  retries: 3
  retry_delay: 1
  api_key: ${OPENAI_API_KEY}

flow:
  timeout: 300
  max_retries: 3
  retry_delay: 5
  enable_streaming: true
  enable_caching: true
  cache_ttl: 3600
  cache_size: 1000

security:
  enable_auth: false
  enable_rate_limiting: true
  rate_limit_requests: 60
  rate_limit_window: 60
  max_request_size: 10485760
  allowed_origins: ["*"]
  session_timeout: 3600
```

### Environment Variables

```bash
# Application settings
export DEBUG=true
export ENVIRONMENT=production

# Logging
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
export LOG_FILE=logs/app.log

# Gradio interface
export GRADIO_PORT=8080
export GRADIO_HOST=0.0.0.0
export GRADIO_SHARE=true
export DEMO_PASSWORD=your-secure-password

# LLM providers
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export OPENAI_API_KEY=your-api-key
export ANTHROPIC_API_KEY=your-api-key

# Caching
export REDIS_URL=redis://localhost:6379

# Security
export ENABLE_AUTH=true
export RATE_LIMIT_REQUESTS=100
```

## 🔧 Development

### Project Structure

```
virtual-pr-firm/
├── main.py              # Main application entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration management
├── validation.py        # Input validation and sanitization
├── logging_config.py    # Logging system
├── caching.py           # Caching mechanism
├── flow.py              # Flow definitions
├── nodes.py             # Node implementations
├── utils/               # Utility functions
│   ├── call_llm.py
│   ├── brand_bible_parser.py
│   └── ...
├── tests/               # Test suite
│   ├── test_config.py
│   ├── test_validation.py
│   └── ...
├── requirements.txt     # Dependencies
└── README.md           # This file
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=. tests/

# Run with verbose output
pytest -v tests/
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Set up pre-commit hooks (optional)
pre-commit install

# Run linting
flake8 .
black .
isort .
```

## 🔒 Security Features

### Input Validation
- **Content filtering** to prevent spam and inappropriate content
- **Platform validation** with alias support and normalization
- **File upload security** with size limits and extension validation
- **XML sanitization** to prevent injection attacks

### Rate Limiting
- **Token bucket algorithm** for request throttling
- **Per-client rate limits** with configurable windows
- **Automatic cleanup** of expired rate limit entries
- **Graceful degradation** when limits are exceeded

### Authentication
- **Password-based authentication** for web interface
- **Session management** with configurable timeouts
- **Secure session storage** with encryption support
- **OAuth integration** support (Google, GitHub)

## 📊 Monitoring & Observability

### Logging
- **Structured logging** with JSON format support
- **Correlation ID tracking** across all operations
- **Sensitive data filtering** to prevent credential leaks
- **Log rotation** with configurable size limits

### Metrics
- **Cache hit/miss ratios** for performance monitoring
- **Request latency tracking** with percentile reporting
- **Error rate monitoring** with categorization
- **User behavior analytics** for optimization

### Health Checks
- **System health monitoring** with component status
- **Dependency checking** for external services
- **Configuration validation** with detailed error reporting
- **Performance benchmarking** for capacity planning

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "-m", "cli", "serve", "--host", "0.0.0.0"]
```

### Production Considerations

1. **Environment Configuration**
   - Use production configuration files
   - Set secure environment variables
   - Enable authentication and rate limiting

2. **Monitoring Setup**
   - Configure log aggregation (ELK stack, etc.)
   - Set up metrics collection (Prometheus, etc.)
   - Implement alerting for critical issues

3. **Security Hardening**
   - Use HTTPS with valid certificates
   - Implement proper authentication
   - Configure firewall rules
   - Regular security updates

4. **Performance Optimization**
   - Enable Redis for distributed caching
   - Configure appropriate rate limits
   - Monitor resource usage
   - Scale horizontally as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Security**: Report security issues privately to maintainers

## 🗺️ Roadmap

### Planned Features
- [ ] **Multi-language support** for international content
- [ ] **Advanced analytics dashboard** for content performance
- [ ] **A/B testing framework** for content optimization
- [ ] **Integration APIs** for third-party platforms
- [ ] **Advanced brand voice customization** with AI training
- [ ] **Content scheduling** and automation features

### Technical Improvements
- [ ] **GraphQL API** for flexible data querying
- [ ] **WebSocket support** for real-time updates
- [ ] **Microservices architecture** for better scalability
- [ ] **Advanced caching strategies** with predictive loading
- [ ] **Machine learning models** for content optimization
