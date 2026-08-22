# Contributing to Market Modeling

Thank you for your interest in contributing! This document outlines the guidelines for contributing to this project.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md) (to be added). Please be respectful and constructive.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Git

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/market-modeling.git
cd market-modeling

# Install dependencies with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
uv run pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

Follow the code style guidelines below.

### 3. Run Quality Checks

```bash
# Format code
uv run ruff check --fix .
uv run ruff format .
uv run black .

# Type check
uv run mypy app.py --ignore-missing-imports

# Run tests
uv run pytest -v

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
# or
git commit -m "fix: resolve issue description"
```

Use conventional commit messages:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `refactor:` — Code restructuring
- `test:` — Adding tests
- `chore:` — Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Open a Pull Request against `main` branch.

## Code Style

### Python

- **Formatter:** Ruff (primary) + Black (compatibility)
- **Linter:** Ruff with strict rules
- **Type Checker:** MyPy (strict mode for new code)
- **Line Length:** 100 characters
- **Quotes:** Double quotes
- **Imports:** Sorted by Ruff (isort compatible)

### Configuration Files

All tool configurations are in `pyproject.toml`:
- `[tool.ruff]` — Linting and formatting
- `[tool.black]` — Black formatting (fallback)
- `[tool.mypy]` — Type checking
- `[tool.pytest.ini_options]` — Test configuration

### Streamlit App Specifics

- `print()` statements are allowed (used for debugging in Streamlit)
- Global `st.session_state` mutations are expected
- Type ignores for external libraries (PyMC, ArviZ, Streamlit, Plotly) are acceptable

## Testing

### Test Structure

```
tests/
├── test_utils.py         # Utility functions
├── test_schema.py        # Schema inference
├── test_validation.py    # Data validation
└── test_canonicalize.py  # Canonicalization & capabilities
```

### Writing Tests

- Use `pytest` with descriptive test names
- Follow `test_<function>_<scenario>` naming
- Use fixtures for common test data
- Aim for >80% coverage on core utilities
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

### Running Tests

```bash
# All tests
uv run pytest -v

# With coverage
uv run pytest --cov=. --cov-report=term-missing

# Specific test file
uv run pytest tests/test_utils.py -v

# Skip slow tests
uv run pytest -m "not slow"
```

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted (ruff, black)
- [ ] Type checking passes (mypy)
- [ ] Pre-commit hooks pass
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated (if applicable)

### PR Description

Include:
- **What** — Summary of changes
- **Why** — Motivation/context
- **How** — Implementation approach
- **Testing** — How you verified the changes
- **Screenshots** — For UI changes

### Review Process

1. Automated checks must pass (CI)
2. At least one maintainer review
3. Address review comments
4. Squash and merge (maintainer)

## Issue Reporting

### Bug Reports

Include:
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior
- Error traceback (if applicable)
- Sample data (if relevant)

### Feature Requests

Include:
- Use case / problem statement
- Proposed solution
- Alternatives considered
- Implementation complexity estimate

## Architecture Notes

### Single-File App

The main application is intentionally a single file (`app.py`) for:
- Easy deployment to Streamlit Cloud
- Simplified dependency management
- Self-contained distribution

When adding features, consider:
- Keeping related functions grouped with clear section headers
- Adding type hints for all new functions
- Writing tests for pure functions (utilities, validation, canonicalization)

### Key Modules (Logical Sections)

| Section | Lines | Purpose |
|---------|-------|---------|
| Configuration | 20-82 | Constants, aliases, dataclasses |
| Utilities | 99-137 | Normalization, formatting, fingerprinting |
| Synthetic Data | 143-245 | Ground-truth sample generator |
| Schema Inference | 257-330 | Role scoring, inference, mapping |
| Validation | 356-426 | Blocking/warning findings |
| Canonicalization | 429-477 | Type coercion, derived fields |
| Capabilities | 480-519 | Analysis readiness matrix |
| Summaries | 527-642 | Entity summary, growth, portfolio |
| Model Construction | 649-847 | Complexity selection, PyMC fitting |
| Posterior/Diagnostics | 854-1001 | Elasticity extraction, PPC, prediction |
| Visual Helpers | 1009-1050 | Status boxes, uncertainty bands |
| Pages | 1058-1554 | All UI pages |
| State/UI | 1562-1764 | Session state, navigation, main |

## Dependency Management

### Adding Dependencies

1. Add to `pyproject.toml` under `[project.dependencies]`
2. Add pinned version to `requirements.txt`
3. Update `uv.lock` (run `uv lock`)
4. Test locally

### Updating Dependencies

```bash
# Update all
uv lock --upgrade

# Update specific
uv add package@latest
```

Dependabot will create PRs for security updates automatically.

## Release Process

1. Update version in `pyproject.toml` and `app.py` (`APP_VERSION`)
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v3.1.0`
4. Push tag: `git push origin v3.1.0`
5. GitHub Actions builds and publishes (to be configured)

## Questions?

- Open a [Discussion](https://github.com/yourusername/market-modeling/discussions)
- Check existing [Issues](https://github.com/yourusername/market-modeling/issues)
- Review the [README](README.md) for usage details

---

**Thank you for contributing!** 🎉