# Virtual PR Firm - AI Content Generation System

**ALWAYS follow these instructions first. Only search for additional information or run other commands if the information here is incomplete or found to be in error.**

## Working Effectively

### Prerequisites and Environment Setup
- Ensure Python 3.10+ is installed (tested with 3.12.3)
- **CRITICAL DEPENDENCY ISSUE**: Network connectivity to PyPI is unreliable in this environment
- Several required packages cannot be installed due to network timeouts: `pocketflow`, `gradio`, `openai`, `anthropic`

### Bootstrap Commands (REQUIRED FIRST STEPS)
```bash
# Fix the broken requirements.txt file (contains built-in modules)
python -c "
with open('requirements.txt', 'w') as f:
    f.write('''pocketflow>=0.0.1
gradio>=4.0.0
openai>=1.0.0
anthropic>=0.7.0
PyYAML>=6.0
lxml>=4.9.0
requests>=2.31.0
python-dotenv>=1.0.0
spacy>=3.7.0
nltk>=3.8
websockets>=11.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.15.0''')
"

# Install pytest for testing
pip install pytest
```

### Dependency Installation (EXPECT FAILURES)
```bash
# NEVER CANCEL: The following command will likely timeout after 3-5 minutes due to network issues
# Set timeout to 10+ minutes and expect ReadTimeoutError from PyPI
pip install -r requirements.txt --timeout 600
```
**Expected result**: Installation will fail with `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.`

**Alternative approach when network allows**:
```bash
# Try installing packages individually with longer timeouts
pip install pocketflow --timeout 600
pip install gradio --timeout 600  
pip install openai --timeout 600
# Each command may take 5-10 minutes or fail with timeout
```

### What Works Without Dependencies
```bash
# Test basic Python functionality - ALWAYS WORKS
python -c "import json, os, sys, logging; print('Built-in modules work')"

# Check available pre-installed packages
python -c "import requests, yaml; print('Some packages are pre-installed')"

# Run a minimal demo without dependencies
python -c "
print('Virtual PR Firm - Content Generation System')
print('Main components:')
print('- main.py: CLI and Gradio web interface')
print('- flow.py: PocketFlow workflow orchestration')
print('- nodes.py: Content generation pipeline nodes')
print('- utils/: Brand bible parsing, LLM calls, platform formatting')
"
```

## Testing and Validation

### Run Tests (FAST - completes in under 1 second)
```bash
# NEVER CANCEL: Test execution is very fast, typically 0.22 seconds
python -m pytest tests/ -v
```
**Expected result**: Tests will FAIL with `ModuleNotFoundError: No module named 'pocketflow'` but execute quickly.

### Validate Code Syntax
```bash
# Check for Python syntax errors - CRITICAL STEP
python -m py_compile main.py
python -m py_compile flow.py
python -m py_compile nodes.py
```
**Expected result**: main.py will FAIL with syntax error due to unescaped quotes in docstrings.

## Code Structure and Navigation

### Key Files and Their Purpose
- **main.py**: Entry point with CLI and Gradio web interface (HAS SYNTAX ERRORS)
- **flow.py**: PocketFlow workflow orchestration using nodes
- **nodes.py**: Individual content generation pipeline components
- **utils/**: Utility modules for platform formatting, LLM calls, brand bible parsing
- **tests/test_smoke.py**: Basic smoke test (fails due to missing dependencies)
- **docs/design.md**: Comprehensive design documentation
- **requirements.txt**: Python dependencies (FIXED - originally had built-in modules)

### Important Directories
```bash
# View repository structure
ls -la
# Key directories:
# .github/workflows/ - CI configuration
# utils/ - Core utility modules
# docs/ - Design documentation  
# tests/ - Test suite
```

### Working with the Codebase
```bash
# View main entry points
head -50 main.py    # CLI and Gradio interface
head -50 flow.py    # Workflow orchestration
head -30 tests/test_smoke.py  # Basic tests

# Check utility modules
ls utils/
# Key utilities:
# - brand_bible_parser.py: XML parsing for brand guidelines
# - call_llm.py: LLM API integration
# - format_platform.py: Platform-specific formatting
# - streaming.py: Real-time progress updates
```

## Known Issues and Limitations

### CANNOT BUILD OR RUN (Due to Dependencies)
- **Cannot install PocketFlow**: Network timeouts prevent pip installation
- **Cannot run main.py**: Syntax errors in docstrings + missing gradio
- **Cannot run content generation**: Missing openai, anthropic packages
- **Cannot start web interface**: Missing gradio package
- **Tests fail**: Missing pocketflow dependency

### Syntax Issues That MUST Be Fixed
```bash
# main.py has unescaped quotes in docstrings around line 225
# Fix by replacing problematic docstring with simple version:
# Change: """Long docstring with "unescaped quotes" """
# To: """Simple docstring without problematic quotes."""
```

### What DOES Work
- Basic Python imports and built-in modules
- File structure analysis and code reading
- pytest execution (fails but runs quickly)
- YAML parsing with pre-installed PyYAML
- HTTP requests with pre-installed requests library

## Application Architecture

### Content Generation Flow (When Dependencies Available)
1. **EngagementManagerNode**: Collect user inputs (platforms, topics, brand bible)
2. **BrandBibleIngestNode**: Parse XML brand guidelines  
3. **VoiceAlignmentNode**: Apply brand voice constraints
4. **PlatformFormattingNode**: Generate platform-specific formatting rules
5. **ContentCraftsmanNode**: Create content drafts
6. **StyleEditorNode**: Editorial review and refinement
7. **StyleComplianceNode**: Final compliance checking

### Supported Platforms
- Twitter/X (character limits, hashtag rules)
- LinkedIn (professional tone, CTA formatting)
- Instagram (caption format, emoji usage)
- Reddit (subreddit-specific rules, markdown)
- Email (subject lines, single CTA)
- Blog (headers, link density)

### Key Features (When Functional)
- **Real-time streaming**: Progress updates via Gradio interface
- **Brand Bible integration**: XML parsing for voice/tone consistency
- **Style enforcement**: No em-dash, no rhetorical contrasts
- **Version control**: Content iterations with rollback
- **Cost tracking**: LLM usage monitoring
- **Multi-platform optimization**: Platform-specific formatting

## Manual Validation Scenarios (CANNOT CURRENTLY EXECUTE)

When dependencies are available, validate changes by:

1. **Basic Flow Test**:
```bash
python main.py  # Should start CLI demo
# Expected: Content generation for sample platforms
# Current status: FAILS due to syntax error + missing dependencies
```

2. **Web Interface Test**:
```bash
python -c "from main import create_gradio_interface; app = create_gradio_interface(); app.launch()"
# Expected: Web interface at http://localhost:7860
# Current status: FAILS due to missing gradio package
```

3. **Platform Content Generation**:
- Input: "Announce new product launch"
- Platforms: "twitter, linkedin"
- Expected: Platform-optimized content with brand compliance
- Current status: CANNOT TEST due to missing pocketflow

4. **Utility Module Testing**:
```bash
# Test individual utility modules (when dependencies available)
python -c "from utils.brand_bible_parser import parse_xml; print('Parser works')"
python -c "from utils.format_platform import build_guidelines; print('Formatter works')"
# Current status: FAILS due to missing dependencies in utility modules
```

## CI/CD Pipeline
```bash
# CI workflow (.github/workflows/ci.yml):
# 1. Python 3.10 setup
# 2. pip install -r requirements.txt  
# 3. python -m pytest -q
# Expected build time: < 2 minutes when dependencies install successfully
```

## Development Guidelines

### Before Making Changes
1. **ALWAYS** fix the syntax error in main.py first
2. **ALWAYS** check if dependencies can be installed before testing
3. **ALWAYS** validate Python syntax with `python -m py_compile filename.py`
4. **ALWAYS** run the fast test suite even if it fails

### Code Quality Standards
- Follow PEP 8 style guidelines  
- Use type hints throughout the codebase
- Document all public methods with docstrings
- Add TODO comments with ownership and date
- Maintain compatibility with PocketFlow Node API

### Testing Strategy
- Smoke tests in tests/test_smoke.py (runs in ~0.2 seconds)
- Unit tests for utility modules when dependencies available
- Integration tests for complete content generation flow
- Manual validation through web interface

---

## Troubleshooting Common Issues

### Dependency Installation Failures
```bash
# If pip install fails with timeout:
# 1. Check network connectivity
curl -s https://pypi.org/ > /dev/null && echo "PyPI accessible" || echo "PyPI blocked"

# 2. Try with different timeout values
pip install pocketflow --timeout 900  # 15 minutes

# 3. Use alternative index if available
pip install pocketflow -i https://test.pypi.org/simple/ --timeout 600
```

### Syntax Errors in main.py
```bash
# Check for specific error location
python -c "
try:
    import ast
    with open('main.py', 'r') as f:
        ast.parse(f.read())
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax error at line {e.lineno}: {e.msg}')
"
```

### Import Errors
```bash
# Check what's actually importable
python -c "
try:
    from flow import create_main_flow
    print('Flow import: OK')
except Exception as e:
    print(f'Flow import failed: {e}')

try:
    import gradio
    print('Gradio: OK')
except Exception as e:
    print(f'Gradio failed: {e}')
"
```

### Performance Issues
```bash
# Time critical operations
time python -m pytest tests/ -v           # Should be < 1 second
time python -m py_compile main.py         # Should be < 1 second  
time python -c "import yaml, requests"    # Should be < 1 second
```

**REMEMBER**: This codebase currently CANNOT be fully built or run due to network dependency issues. Focus on code analysis, syntax fixes, and structural improvements that can be validated without external dependencies.