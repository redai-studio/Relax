# Content Verification Guide

This guide explains how to verify documentation content against the Relax source code to ensure accuracy.

## Why Verification Matters

Documentation that describes non-existent APIs, wrong parameter names, or outdated behavior is worse than no documentation. Users will waste time trying APIs that don't work, file bug reports for "broken" features, and lose trust in the docs.

## Verification Process

### 1. Identify Source Files

For any feature being documented, locate the primary source files:

| Feature Area | Primary Source Files |
|---|---|
| Metrics Service | `relax/utils/metrics/service.py`, `relax/utils/metrics/client.py`, `relax/utils/metrics/__init__.py` |
| Health Check | `relax/utils/health_system.py` |
| Controller | `relax/core/controller.py` |
| Services | `relax/core/service.py` |
| Actor (Training) | `relax/components/actor.py`, `relax/components/actor_fwd.py` |
| Critic | `relax/components/critic.py` |
| Rollout | `relax/components/rollout.py`, `relax/engine/rollout/sglang_rollout.py` |
| Checkpoint Engine | `relax/distributed/checkpoint_service/` (entire directory) |
| Notification | `relax/utils/metrics/adapters/apprise.py` |
| Arguments/Config | `relax/utils/arguments.py` |
| Data/Dataset | `relax/utils/stream_dataloader.py`, `relax/utils/training/data.py`, `relax/utils/training/streaming_dataset.py` |
| Ray Utilities | `relax/distributed/ray/` (entire directory) |
| Megatron Backend | `relax/backends/megatron/` |
| SGLang Backend | `relax/backends/sglang/` |
| Router | `relax/engine/router/router.py` |
| Timeline Trace | `relax/utils/metrics/timeline_trace.py` |
| Transfer Queue | `transfer_queue/` (entire directory) |

### 2. Verify Import Paths

Check that documented imports actually work:

```python
# Doc says:
from relax.utils.metrics import MetricsClient

# Verify by reading:
# 1. relax/utils/metrics/__init__.py — does it export MetricsClient?
# 2. relax/utils/metrics/client.py — does MetricsClient class exist?
```

Common pitfalls:
- Class exists but is not exported in `__init__.py`
- Import path changed during refactoring
- Class was renamed but docs still use old name

### 3. Verify Function Signatures

For every function/method shown in the doc, read the actual source and check:

```python
# Doc says:
metrics.log_scalar(key: str, value: float, step: int)

# Source actually has:
def log_metric(self, step: int, metric_name: str, metric_value: Union[float, int, str, dict], ...)

# This is WRONG — the actual method is log_metric, not log_scalar,
# and the parameter order and names are different!
```

What to check:
- Method name (exact spelling)
- Parameter names (exact spelling)
- Parameter order
- Parameter types
- Default values
- Return type

### 4. Verify CLI Arguments

For documented CLI flags, check the argument parser:

```python
# Read relax/utils/arguments.py or the relevant argument group
# Search for: add_argument("--flag-name", ...)
# Verify: name, type, default, help text, choices
```

Common pitfalls:
- Flag name uses underscores in code but hyphens in CLI
- Flag was removed in a recent commit
- Default value changed

### 5. Verify Config Keys

For documented YAML config keys:

```python
# Read configs/env.yaml and the code that loads it
# Verify that the key name and structure match
```

### 6. Verify Architecture Claims

When the doc says "X is deployed as a Ray Serve deployment" or "Y runs in a background thread":

```python
# Check the source:
# - @ray.remote decorator → Ray actor
# - @serve.deployment decorator → Ray Serve
# - threading.Thread → background thread
# - asyncio → async
```

## Red Flags

Watch for these signs that documentation may be inaccurate:

1. **Perfect-looking API that doesn't exist** — if a function signature looks too clean and convenient, it might be aspirational rather than real
2. **Inconsistent naming** — if the doc uses `log_scalar` but the source has `log_metric`, something is wrong
3. **Missing error handling** — if the doc shows a simple `client.do_thing()` but the source requires try/except or checking return values
4. **Outdated defaults** — defaults tend to change during development; always check current source
5. **Ghost features** — features mentioned in design docs or PRs that were never fully implemented

## What to Do When Source and Expectation Differ

1. **Trust the source code** — it is the ground truth
2. **Document what exists** — not what was planned
3. **Explain known limitations in context** — put them near the affected configuration; reserve warning containers for consequential risks
4. **Do not publish uncertain claims** — clarify ambiguous source behavior in internal notes before describing it to users

## Internal Verification Checklist Template

Use this checklist in internal working notes or the PR's validation evidence for each doc page. Do not copy it into the user page or publish it as a test/training acceptance report. Test counts, run logs, machine traces, and verification history belong with this evidence, not in guide content or navigation.

Use verified behavior to explain required settings, limitations, and troubleshooting. Do not turn one successful test or experiment into an unconditional feature, performance, quality, or reliability guarantee.

```markdown
## Verification for [feature-name] docs

- [ ] All import paths verified against __init__.py files
- [ ] All class names match source definitions
- [ ] All method signatures match source (names, types, defaults, return types)
- [ ] All CLI arguments verified against argument parser
- [ ] All config keys verified against config loader
- [ ] Architecture diagram matches actual component relationships
- [ ] No references to removed/renamed APIs
- [ ] Code examples are runnable (no syntax errors, correct API usage)
- [ ] All repository paths exist (every file/dir path like `relax/foo/bar.py` or `scripts/training/*/run-*.sh` mentioned in the doc must be confirmed to exist in the repo via `ls` or `read_file`)
- [ ] English and Chinese versions have identical technical content
- [ ] All internal links point to existing doc pages
- [ ] Relevant repository guides were read for terminology, narration, and level of detail; no report content was copied
- [ ] User content covers necessary purpose, setup, configuration, limitations, and remedies without forcing each into a separate heading or adding test/acceptance-report material
- [ ] Both languages read naturally on their own, with context around examples, clear actions, and consistent terms; “80% of the way to ASD-STE100” is a comprehension goal, not a sentence pattern or compliance claim
- [ ] Repetitive warnings, mechanical lists, internal-facing language, and redundant verification wording have been removed without hiding important risks
- [ ] Docs build and emitted link/render checks completed; missing hardware validation is reported in the PR, not the user page
```
