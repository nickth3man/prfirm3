# Copilot Instructions

This repository uses Python 3.11 and contains a small PocketFlow example project.

## How to build and test

- Install dependencies: `python -m pip install -r requirements.txt` (the setup workflow caches pip)
- Run a quick smoke test: `pytest -q tests/test_smoke.py`
- Run full test suite: `pytest`

## Coding conventions

- Follow existing project style. Keep functions and variables descriptively named.
- Add unit tests for new code and aim for clear, deterministic behavior.
- Avoid adding heavy external dependencies unless necessary.

## When creating pull requests for this repo

- Ensure all tests pass locally and in CI
- Keep changes scoped to the issue description
- Update `README.md` or `docs/` when adding user-visible features

## Notes for Copilot

- The project expects `requirements.txt` to list Python dependencies.
- Use the `tests/` folder for unit tests; a smoke test exists at `tests/test_smoke.py`.
- Prefer standard libraries and lightweight third-party packages.

## Agentic Coding Guidance

- Reference: `@agentic-coding.mdc` (see `.cursor/rules/agentic-coding.mdc` in the repo).
- Follow the **Agentic Coding** principle: humans provide requirements and high-level design; agents implement small, focused changes.
- Prefer small iterative edits, add tests, and ask for human clarification on ambiguous or high-risk decisions.
