# Bypass and CI maintenance

## Exemptions

Read `ciTeam` in [config.json](../../../.github/ci/config.json) and confirm the authenticated actor is a listed member. CI team membership does not itself authorize an exemption: the user's request must cover the affected workflow and code.

Prefer a workflow label for authorized maintenance operations:

```bash
gh pr edit <PR> --repo redai-studio/Relax --add-label 'ci-bypass: gpu-unit'
```

Workflow targets are entries without a `job` in `config.json`. Use `ci-bypass: <workflow>` or `ci-bypass: all`; create needed labels in repository settings before use. ShigureLab/ci-bypass checks the label actor against the configured team.

A comment alternative is `/bypass <workflow>` or `/bypass all`, with an optional reason after the target. The gate reads comments directly; this command creates no status reply, label or automatic rerun. The current pattern accepts the command at the start of the comment, a target followed by whitespace or end of input, and optional trailing text.

After adding an exemption, rerun the whole affected workflow using `gh run rerun <RUN_ID> --repo redai-studio/Relax` so the gate is reevaluated. A job-only rerun can reuse the old gate result. Wait for an active workflow or cancel it only within the authorized scope.

The independent synchronize workflow removes `ci-bypass:` labels after new commits. Historical bypass comments remain eligible under ci-bypass's comment rules. To withdraw an exemption, remove its label or edit/delete its comment, then rerun the workflow.

## Implementation and configuration

| Location / setting | Responsibility |
| --- | --- |
| `.github/ci/config.json` | CI team and workflow/job mappings |
| `.github/actions/ci-commands/` | Comment commands using workflow `GITHUB_TOKEN` |
| `.github/workflows/check-bypass.yml` | Reusable `ShigureLab/ci-bypass@v2` gate |
| `.github/workflows/remove-ci-bypass-labels.yml` | Independent label cleanup on synchronize |
| `.github/actions/pr-welcome/` | Welcome publisher and bilingual Markdown template |
| Secret `WELCOME_BOT_TOKEN` | Credential belonging to the welcome bot, with permission to write PR comments |
| Variable `WELCOME_BOT_LOGIN` | Required welcome account login; must match the token owner |

Keep `ciTeam` nonempty and use exact GitHub login spelling because ci-bypass compares usernames case-sensitively. Verify workflow filenames and job names when changing targets. Only welcome publishing needs the bot secret; the action verifies its account before posting.

Comment commands and welcome publishing load trusted default-branch code. Configure Nyanpasu's `/review` wake word separately. For a native GitHub stack, `pull_request.branches` matches the stack base; manually chaining PR base branches has different trigger semantics.

## Validation

Use Node.js 24 for ESM checks and run `pre-commit run --all-files`. Validate changed workflows and render the welcome template after editing it. The oxfmt hook uses `ShigureLab/oxfmt-pre-commit-mirror`; its version and file scope are configured in `.pre-commit-config.yaml`.

After deploying to the default branch, verify welcome publishing, command permissions, rerun/cancel, bypass and label cleanup on a disposable draft PR. Local checks do not establish live token permissions or GPU/NPU execution.
