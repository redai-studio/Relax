# How to Contribute

Thank you for your interest in contributing to Relax! This guide will help you get started.

## Getting Started

### 1. Set Up Development Environment

Create a virtual environment and install dependencies:

```bash
# Clone the repository
git clone https://github.com/redai-studio/Relax.git
cd Relax

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Start Ray and Deploy Services

```bash
# Start Ray cluster
ray start --head

# Deploy all services
python -m relax.core.controller deploy --config configs/env.yaml
```

### 3. Run Example Experiment

```bash
# Run basic example
python relax/entrypoints/train.py

# Run DeepEyes example
cd examples/deepeyes
bash run_deepeyes.sh
```

## Development Workflow

### 1. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/utils/test_metrics_service.py

# Run with coverage
pytest --cov=relax tests/
```

### 4. Format Code

```bash
# Format with black
black relax/

# Sort imports
isort relax/

# Run linter
flake8 relax/
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve bug in metrics service"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Write docstrings for public functions
- Keep functions focused and small

Example:

```python
def compute_reward(
    response: str,
    ground_truth: dict,
    reward_type: str = "f1"
) -> float:
    """
    Compute reward for a response.
    
    Args:
        response: Model's generated response
        ground_truth: Ground truth data
        reward_type: Type of reward to compute
        
    Returns:
        Reward score between 0 and 1
    """
    if reward_type == "f1":
        return compute_f1_score(response, ground_truth["answer"])
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep documentation up to date

## Testing Guidelines

### Writing Tests

```python
import pytest
from relax.utils.metrics.client import MetricsClient

def test_metrics_client_log_metric():
    """Test logging metrics."""
    client = MetricsClient(service_url="http://localhost:8000/metrics")
    
    # Log metric
    client.log_metric(step=1, metric_name="test/metric", metric_value=0.5)
    
    # Verify buffered metrics
    assert client.get_buffered_metrics_count(step=1) == 1
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Use mocks for external dependencies

## Documentation Guidelines

### Adding Documentation

1. Add markdown files to `docs/guide/` or `docs/zh/guide/`
2. Update `.vitepress/config.mts` to add to sidebar
3. Include code examples and diagrams
4. Provide both English and Chinese versions

### Building Documentation

```bash
# Start documentation dev server
make docs-dev

# Build documentation
make docs-build

# Preview built documentation
make docs-preview
```

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### PR Description

Include:

- **What**: What changes were made
- **Why**: Why these changes are needed
- **How**: How the changes work
- **Testing**: How the changes were tested

Example:

```markdown
## What
Add support for custom reward functions in DeepEyes example

## Why
Users need flexibility to define custom reward logic for their tasks

## How
- Added `custom_reward.py` module
- Updated configuration to support custom reward functions
- Added documentation and examples

## Testing
- Added unit tests for custom reward functions
- Tested with DeepEyes example
- Verified backward compatibility
```

## Review Process

1. **Automated Checks**: CI/CD runs tests and linters
2. **Code Review**: Maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: Get approval from maintainers
5. **Merge**: PR is merged to main branch

## Community Guidelines

### Be Respectful

- Be kind and respectful to others
- Welcome newcomers
- Provide constructive feedback
- Assume good intentions

### Ask for Help

- Use GitHub Discussions for questions
- Join our WeChat group
- Check existing issues and PRs

### Report Issues

When reporting bugs:

- Use a clear, descriptive title
- Describe steps to reproduce
- Include error messages and logs
- Specify your environment (OS, Python version, etc.)

## Areas to Contribute

### Code

- New features
- Bug fixes
- Performance improvements
- Code refactoring

### Documentation

- Improve existing docs
- Add new guides
- Translate to other languages
- Fix typos and errors

### Examples

- Add new examples
- Improve existing examples
- Add tutorials

### Testing

- Add new tests
- Improve test coverage
- Add integration tests

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and discuss ideas
- **WeChat Group**: Join our community
- **Email**: Contact maintainers

## License

By contributing to Relax, you agree that your contributions will be licensed under the Apache 2.0 License.

## Thank You!

Thank you for contributing to Relax! Your contributions help make this project better for everyone.
