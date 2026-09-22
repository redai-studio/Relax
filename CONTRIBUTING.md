# Contributing to Relax

Thank you for your interest in contributing to Relax! This document provides guidelines and information for contributors.

> For a more detailed guide covering development environment setup, testing, and documentation, see our full [Contributing Guide](docs/en/guide/how-to-contribute.md).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [License](#license)

## Code of Conduct

This project follows a standard code of conduct. Please be respectful, inclusive, and constructive in all interactions. We are committed to providing a welcoming and harassment-free experience for everyone.

## How Can I Contribute?

### Code Contributions

- **Bug Fixes** — Find and fix issues in the codebase
- **New Features** — Implement new algorithms, models, or utilities
- **Performance** — Optimize training throughput, memory usage, or startup time
- **Refactoring** — Improve code quality without changing behavior

### Non-Code Contributions

- **Documentation** — Improve guides, add examples, fix typos (bilingual: English + Chinese)
- **Bug Reports** — Report issues with clear reproduction steps
- **Feature Requests** — Propose enhancements via GitHub Issues
- **Examples** — Add new training examples or tutorials
- **Testing** — Improve test coverage and add integration tests

## Development Setup

```bash
# Clone the repository
git clone https://github.com/redai-studio/Relax.git
cd Relax

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Development Workflow

### 1. Create a Branch

```bash
# Feature
git checkout -b feature/your-feature-name

# Bug fix
git checkout -b fix/your-bug-fix

# Documentation
git checkout -b docs/your-doc-change
```

### 2. Make Changes

- Follow existing code patterns and conventions
- Add or update tests for your changes
- Update documentation if applicable (both English and Chinese)

### 3. Validate

```bash
# Run pre-commit checks (lint + format)
pre-commit run --all-files

# Run tests
pytest tests/
```

### 4. Submit a Pull Request

Push your branch and open a PR against `main`. Fill out the PR template completely.

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/), line width 119 (configured in `pyproject.toml`)
- **Imports**: Managed by `isort` via Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Required for public functions and classes
- **Logging**: Use `relax.utils.logging_utils.get_logger(__name__)` — never `print()` or `logging.getLogger()`
- **Copyright Header**: All `.py` files under `relax/` must include:
  ```python
  # Copyright (c) 2026 Relax Authors. All Rights Reserved.
  ```
- **No Wildcard Imports**: `from x import *` is not allowed

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix      | Usage                                    |
| :---------- | :--------------------------------------- |
| `feat:`     | New feature                              |
| `fix:`      | Bug fix                                  |
| `docs:`     | Documentation only                       |
| `style:`    | Code style (formatting, no logic change) |
| `refactor:` | Code refactoring                         |
| `test:`     | Adding or updating tests                 |
| `chore:`    | Maintenance, CI, build changes           |
| `perf:`     | Performance improvements                 |

Example:

```
feat(rollout): add streaming data consumption for async mode

- Implement StreamingDataLoader for TransferQueue
- Add configurable staleness via --max-staleness
- Update documentation with async training guide
```

## Pull Request Process

### Before Submitting

- [ ] Code compiles and runs without errors
- [ ] `pre-commit run --all-files` passes
- [ ] Tests pass (`pytest tests/`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow Conventional Commits

### PR Review

1. **Automated CI** — Linting, formatting, and test checks run automatically
2. **Code Review** — At least one maintainer reviews the PR
3. **Feedback** — Address review comments promptly
4. **Approval & Merge** — Maintainer merges after approval

### Tips for a Good PR

- Keep PRs focused and reasonably sized
- Provide a clear description of **what**, **why**, and **how**
- Link related issues (e.g., `Fixes #123`)
- Add screenshots or logs for UI or behavior changes

## Reporting Bugs

Use the [Bug Report template](https://github.com/redai-studio/Relax/issues/new?template=bug_report.md) and include:

- **Environment** — OS, Python version, CUDA version, GPU type
- **Steps to Reproduce** — Minimal commands to trigger the bug
- **Expected vs Actual Behavior** — What you expected and what happened
- **Logs / Error Messages** — Full stack trace or relevant log output
- **Configuration** — Training script, args, or config files used

## Requesting Features

Use the [Feature Request template](https://github.com/redai-studio/Relax/issues/new?template=feature_request.md) and include:

- **Problem Statement** — What problem does this solve?
- **Proposed Solution** — How should it work?
- **Alternatives Considered** — Other approaches you thought about
- **Use Case** — Concrete examples of how you'd use it

## Community

- **GitHub Issues** — Bug reports and feature requests
- **GitHub Discussions** — Questions and general discussion
- **WeChat Group** — Join our community (see [README](README.md) for QR code)

## License

By contributing to Relax, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
