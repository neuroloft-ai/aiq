# Contributing to AIQ

Thank you for your interest in contributing to AIQ!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/aiq.git`
3. Install in development mode: `pip install -e ".[dev,all]"`
4. Run tests: `pytest tests/`

## How to Contribute

### Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, minimal reproduction steps
- For security issues (PII detection bugs), please email directly instead of public issues

### Adding a Detector

AIQ's governance module (A31) uses a detector pattern. To add a new detector:

1. Add the detection function in `aiq/a31/classifier.py`:
   ```python
   def _detect_my_issue(content: str, chunk_id: str) -> list[ClassificationFinding]:
       # Your regex patterns here
       ...
   ```

2. Add a config toggle in `A31Config`:
   ```python
   detect_my_issue: bool = True
   ```

3. Wire it in `Classifier.run()` alongside the other detectors

4. Add a test in `tests/test_a31.py`

### Adding a File Format

To support a new file format in the loader:

1. Add `_load_FORMAT()` in `aiq/loader.py`
2. Add the extension to the `load_file()` dispatch
3. Add any new dependency as an optional extra in `pyproject.toml`

### Adding an LLM Provider

1. Add `_PROVIDER_client()` in `aiq/llm.py`
2. Add it to `create_llm_client()` dispatch
3. Add any SDK dependency as an optional extra in `pyproject.toml`

## Code Standards

- All modules must work without LLM (rule-based fallback required)
- No hardcoded credentials, API keys, or personal data
- Every module follows the `ModuleOutput` contract: `(detected, resolved, remaining)`
- Tag content, never delete — original content is always preserved
- Add docstrings following the project format (see any module for the template)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_a31.py

# With coverage
pytest tests/ --cov=aiq --cov-report=term-missing
```

## Pull Request Process

1. Create a branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Add tests for new functionality
4. Run `pytest` and ensure all tests pass
5. Submit a PR with a clear description of what and why
