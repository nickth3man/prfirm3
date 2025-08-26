---
applyTo: "**/tests/*.py"
---

# Tests Instructions

## Tests guidelines

- Keep tests small and deterministic. Each test should assert one behavior.
- Use `pytest` and fixtures when appropriate.
- Use clear test names (e.g., `test_function_does_x_when_condition_y`).
- Avoid network calls or file system writes in unit tests; mock external interactions.
- For slow or integration-level tests, mark them with `@pytest.mark.integration`.

## Running tests

- Run a single test file: `pytest -q tests/test_smoke.py`
- Run full test suite: `pytest`
