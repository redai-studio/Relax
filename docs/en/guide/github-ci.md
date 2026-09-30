# GitHub CI

## Quick Start

Relax runs CPU checks, GPU unit tests, and GPU/NPU integration tests on PRs, including drafts. Post one command per PR comment:

```text
/rerun
```

This retries failed jobs in the latest failed/timed-out workflows for the current PR head. To retry one check, use `/rerun cpu-310`. Commands do not create missing workflow runs or approve fork workflows.

## Commands

| Command | Behavior | Who can use it |
| --- | --- | --- |
| `/rerun` or `/rerun failed` | Retry failed jobs in current-head failed/timed-out workflows | PR author or repository write/maintain/admin |
| `/rerun <target>` | Rerun a workflow or one check and its dependent jobs | Same as `/rerun` |
| `/rerun all` | Rerun all configured workflows that have completed | Same as `/rerun` |
| `/cancel <workflow>` or `/cancel all` | Request cancellation of whole active workflows | Same as `/rerun` |
| `/help` | Show commands and targets | Anyone |
| `/bypass <workflow>` or `/bypass all` | Record a comment that ci-bypass evaluates when a workflow runs | Configured CI team |
| `/review` | Request Nyanpasu review | Controlled by Nyanpasu |

The command action handles `/rerun`, `/cancel`, and `/help`. ShigureLab/ci-bypass reads bypass labels/comments; Nyanpasu handles `/review`. These have independent entry points. `/bypass` does not post a bot reply or automatically rerun CI.

Use `/rerun`, not `/re-run`. There is no `/ci` or `/unbypass`. Only newly created comments execute rerun/cancel/help; editing a comment does not execute it again. The standard agent attribution footer is accepted after a command.

## Targets

| Target | Workflow / check |
| --- | --- |
| `ci` | `ci.yml`: pre-commit and CPU tests |
| `pre-commit` | Pre-commit Checks |
| `cpu-310`, `cpu-311`, `cpu-312` | Tests (Python 3.10 / 3.11 / 3.12) |
| `gpu-unit` | `unittest.yml`: GPU unit tests |
| `integration` | `integration-test.yml`: GPU/NPU integration tests |
| `gpu-async` | Qwen3-4B-4xgpu-async |
| `gpu-vl` | Qwen3-VL-4B-2xgpu |
| `npu-async` | Qwen3-4B-4xnpu-async |

`/cancel` and `/bypass` accept workflow targets: `ci`, `gpu-unit`, `integration`, or `all`. Cancelling a workflow also cancels its sibling matrix jobs. Deployment workflows are excluded, including from `all`.

## Bypass

The CI team is listed in `.github/ci/config.json`: `SigureMo`, `NINGBENZHE`, and `Yangruipis`. Members can add a `ci-bypass: <workflow>` / `ci-bypass: all` label, or post:

```text
/bypass gpu-unit
```

A reason is optional; for example, `/bypass gpu-unit --reason runner maintenance` is also accepted. Then rerun the entire workflow with `/rerun gpu-unit` to reevaluate the gate. A job-only rerun can reuse an earlier gate result. If the workflow is still active, wait or explicitly cancel it before rerunning.

Following Paddle's workflow structure, ShigureLab/ci-bypass directly checks the label/comment author against the configured team. A separate `synchronize` workflow removes `ci-bypass:` labels when new commits are pushed. This cleanup only removes labels: historical `/bypass` comments remain eligible under ci-bypass's comment rules. To withdraw a comment-based exemption, remove or edit that comment and rerun the workflow.

## Configuration and Maintenance

| File / setting | Purpose |
| --- | --- |
| `.github/ci/config.json` | Shared CI team and workflow/job mappings |
| `.github/actions/ci-commands` | Rerun, cancel, and help using `GITHUB_TOKEN` |
| `.github/actions/pr-welcome` | Welcome template and comment publishing |
| `.github/workflows/check-bypass.yml` | Reusable ShigureLab/ci-bypass gate on `ubuntu-slim` |
| `.github/workflows/remove-ci-bypass-labels.yml` | Independent label cleanup on new commits |
| Secret `WELCOME_BOT_TOKEN` | Token belonging to the welcome bot, initially `rai-studio-bot` |
| Variable `WELCOME_BOT_LOGIN` | Override the expected welcome bot login when renaming the account |

The JSON is loaded directly by JavaScript. Keep the team nonempty and use the exact GitHub login spelling: ci-bypass compares usernames case-sensitively. When adding a target, verify its workflow/job name; entries with a `job` support rerun only.

Only the welcome workflow needs `WELCOME_BOT_TOKEN`, with repository permission to write PR comments. It verifies the token owner before publishing. Rerun/cancel use `GITHUB_TOKEN` with Actions write permission; their status replies use its Issues write permission. Both comment-handling workflows check out trusted default-branch code without persisting credentials. Configure Nyanpasu's `/review` wake word separately.

Welcome text is maintained in `.github/actions/pr-welcome/comment.md`. Reopening a PR does not add a second welcome. Create any desired bypass labels in repository settings; the gate only reads labels and comments.

Use Node.js 24, matching `actions/github-script`, for module checks. The pre-commit hook installs a pinned PyPI distribution of oxfmt for CI JavaScript/JSON; no npm parser dependency is needed:

```bash
node --input-type=module -e "await import('./.github/actions/ci-commands/index.mjs'); await import('./.github/actions/pr-welcome/index.mjs')"
pre-commit run --all-files
```

After merging to the default branch and configuring the bot, use a disposable draft PR to verify welcome publishing, command permissions, rerun/cancel, bypass evaluation, and label cleanup. Local module checks do not verify live GitHub permissions or GPU/NPU execution.

## Stacked PRs

Register the stack with `gh stack`. For a native GitHub stack based on `main`, the existing `pull_request.branches: [main]` filters apply to its stack base. Manually chaining PR base branches does not provide this behavior. See [GitHub's stacked PR reference](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests).

## Troubleshooting

- **No matching run:** verify head SHA, workflow triggers, stack registration, and fork workflow approval. `/rerun` cannot create a run.
- **Run still active:** wait or cancel explicitly; cancellation is asynchronous.
- **Head changed:** inspect the new checks and submit a new command.
- **Processing reply / API timeout:** inspect the linked workflow and target run before retrying. Duplicate events and reruns of the command workflow do not execute the same comment twice.
- **Bypass gate error:** regular tests run; errors do not grant an exemption.
- **Welcome token missing or mismatched:** configure the bot token and expected login. This does not affect rerun/cancel or bypass.

## Next Steps

- [Contribution guide](./how-to-contribute.md)
- [Debugging](./debugging.md)
