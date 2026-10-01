# How to Contribute

Thank you for your interest in Relax! This guide walks you through the contribution process.

::: tip Where to Start

If you would like to contribute to Relax but have not decided what to work on, explore these starting points:

- [Hackathon 2nd Edition](https://github.com/redai-studio/Relax/issues/321): Browse this round's tasks, participation rules, and claim instructions. Choose an area that interests you and participate individually or as a team.
- [Good first issue](https://github.com/redai-studio/Relax/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22): Find introductory tasks suitable for first-time contributors to the project.

Before starting, read the task description and existing discussion to understand the requirements and current progress. Claim Hackathon tasks according to the rules on the event page.

:::

## Development Setup

### 1. Get the Code

**Fork** [redai-studio/Relax](https://github.com/redai-studio/Relax) on GitHub, then clone your fork locally. Replace `<your_user_name>` with your GitHub username:

```bash
git clone https://github.com/<your_user_name>/Relax.git
cd Relax
git remote add upstream https://github.com/redai-studio/Relax.git

# Sync with the main branch of the upstream repository
git checkout main
git pull upstream main
```

`origin` points to your fork, and `upstream` points to the Relax repository. Before starting a new contribution, switch to your local `main` and pull upstream updates, then create a working branch. Make your changes on working branches and keep your local `main` for syncing with upstream.

### 2. Set Up the Development Environment

See the [installation guide](./installation.md) for environment requirements.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit and register Git hooks
pip install pre-commit
pre-commit install

# Install Relax in development mode
pip install -e .
```

### 3. Run an Example Experiment

If you need to validate the training environment, follow the [quick start](./quick-start.md) to prepare the models and data, then run an example. For multi-turn vision-language training, see the [DeepEyes example](../examples/deepeyes.md).

## Development Workflow

### 1. Create a Working Branch

```bash
git checkout -b feat/your-change
```

Use a branch prefix that describes the type of change, for example:

- `feat/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation updates
- `chore/`: Maintenance tasks

For other prefixes, see the commit conventions below.

### 2. Development and Debugging

Implement your feature or fix on the working branch, following these guidelines:

- Write clear, readable code
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 3. Run Unit Tests

After changing the code, add tests for new or fixed behavior and choose the test scope appropriate for your changes:

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/utils/test_metrics_service.py

# Run with coverage
pytest --cov=relax tests/
```

### 4. Commit Changes

After completing the relevant validation, review your changes and stage the files you intend to commit. Replace `<changed-files>` with actual paths, separated by spaces:

```bash
git status
git diff
git add <changed-files>
git commit -m "feat: describe your change"
```

If a hook modifies files or reports errors, review and fix the changes, then run `git add` and `git commit` again until the checks pass and the commit succeeds.

Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 5. Open a PR

Push your working branch to your fork:

```bash
git push origin feat/your-change
```

After pushing, open a PR on GitHub. Select your working branch in your fork as the source and **`main` in `redai-studio/Relax`** as the target, then fill out the [PR template](https://github.com/redai-studio/Relax/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

Address any [CI failures](#ci) and review feedback on the same branch. Validate, commit, and push your changes to update the PR.

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Write docstrings for public functions
- Give each function a single responsibility and keep its implementation concise

Example:

```python
def compute_reward(
    response: str,
    ground_truth: dict,
    reward_type: str = "f1"
) -> float:
    """
    Compute the reward score for a generated response.
    
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

1. Add English and Chinese Markdown files to `docs/en/guide/` and `docs/zh/guide/`, respectively
2. Update `docs/.vitepress/config.mts` to add the pages to the sidebar
3. Include code examples and diagrams
4. Keep both language versions consistent

### Building Documentation

Install Node.js, then run the following commands from the repository root:

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

Explain the following in your PR description:

- **What**: What changes were made
- **Why**: Why these changes are needed
- **How**: How the changes are implemented
- **Testing**: What tests or other validation were performed

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

## CI

PRs run CPU checks, GPU unit tests, and GPU/NPU integration tests. To operate CI, put one command on the first line of a new PR comment. Rerun and cancel are available to the PR author and contributors with repository write access.

| Command | Usage |
| --- | --- |
| `/rerun` | Retry failed jobs in the latest failed or timed-out workflows for the current commit |
| `/rerun <target>` | Rerun one workflow or check; use `all` for all completed CI workflows |
| `/cancel <workflow>` | Cancel an entire workflow, including its matrix jobs; `all` cancels all active CI workflows |
| `/help` | Show commands and targets |
| `/review` | Request a code review |

Targets use `workflow` or `workflow/job` names. For example, `/rerun ci` reruns the whole pre-commit/CPU workflow, while `/rerun ci/pre-commit` selects its pre-commit check. Use `all` to select all configured workflows.

| Target | Workflow / check |
| --- | --- |
| `ci` | All pre-commit and CPU checks |
| `ci/pre-commit` | Pre-commit checks |
| `ci/cpu-310`, `ci/cpu-311`, `ci/cpu-312` | CPU tests on Python 3.10 / 3.11 / 3.12 |
| `gpu-unit` | GPU unit tests |
| `integration` | All GPU/NPU integration tests |
| `integration/gpu-async` | Qwen3-4B GPU async training |
| `integration/gpu-vl` | Qwen3-VL-4B GPU training |
| `integration/npu-async` | Qwen3-4B NPU async training |

`/cancel` supports workflow-level cancellation only: it cancels an entire workflow (`ci`, `gpu-unit`, `integration`, or `all`), not an individual job (such as `integration/gpu-vl`).

`/rerun` applies only to completed CI runs: it can only rerun jobs that already exist for the current commit. If a run is still in progress, wait for it to finish or cancel it before rerunning.

For agent-assisted CI operations, use the [relax-github-ci skill](../../../skills/relax-github-ci/SKILL.md).

## Communication and Feedback

Keep discussions focused on the topic and respect different opinions. When offering criticism or suggestions, explain the specific issue and your reasoning. Be patient with contributors who are new to the project.

### Questions and Discussions

If you run into a problem, search existing issues, PRs, and discussions to see whether someone has already found a solution. If you still need help, ask in GitHub Discussions or join the WeChat group.

### Reporting Bugs

Briefly describe the problem in the issue title and include the following information to help others investigate and reproduce it:

- Steps to reproduce
- Expected results and actual behavior
- Error messages and relevant logs
- Your environment, such as the operating system, Python version, and hardware configuration

## Thank You!

Every contribution helps improve Relax. Thank you for contributing!
