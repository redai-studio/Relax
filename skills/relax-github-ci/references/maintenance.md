# Bypass and CI maintenance

## Exemptions

Read `ciTeam` in [config.json](../../../.github/ci/config.json) and confirm the authenticated actor is a listed member. CI team membership does not itself authorize an exemption: the user's request must cover the affected workflow and code.

Prefer a workflow label for authorized maintenance operations:

```bash
gh pr edit <PR> --repo redai-studio/Relax --add-label 'ci-bypass: gpu-unit'
```

Workflow targets are entries without a `job` in `config.json`: `ci`, `gpu-unit`, and `integration`. Job targets such as `integration/gpu-async` are not valid bypass targets. Use `ci-bypass: <workflow>` or `ci-bypass: all`; create needed labels in repository settings before use. ShigureLab/ci-bypass checks the label actor against the configured team.

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
| `.github/actions/pr-welcome/` | Welcome publisher, live merge requirements and bilingual Markdown template |
| `.github/workflows/review-signal.yml` | Unprivileged review event relay; consumers subscribe to `Review Signal` via `workflow_run` |
| Secret `WELCOME_BOT_TOKEN` | Credential belonging to the welcome bot, with permission to write PR comments and read team membership (`read:org`, or organization Members read for a fine-grained token) |
| Variable `WELCOME_BOT_LOGIN` | Required welcome account login; must match the token owner |

Keep `ciTeam` nonempty and use exact GitHub login spelling because ci-bypass compares usernames case-sensitively. Verify workflow filenames and job names when changing targets. Only welcome publishing needs the bot secret; the action verifies its account before posting.

Comment commands and welcome publishing load trusted default-branch code. Configure Nyanpasu's `/review` wake word separately. For a native GitHub stack, `pull_request.branches` matches the stack base; manually chaining PR base branches has different trigger semantics.

The welcome comment refreshes on PR changes, review signal runs and configured CI workflow runs. Add new CI workflow names to `pr-welcome.yml` when extending CI. Fork runs with empty PR associations are resolved using both head repository and branch; publishing is serialized per PR. The relay does not execute PR code or pass credentials to consumers.

Merge status reads active rulesets, classic branch protection, current reviews and required checks. The requirement table covers approval count, required Team approvals and aggregate CI status. Team candidates exclude the PR author; approvals only count when GitHub reports repository push access. The approval-count row puts the minimum count and write-access requirement in the status column, alongside approvals and requests for changes. Conditional team file patterns remain marked for GitHub to decide. Bypass eligibility is not inferred from a bot's limited view.

CI uses GraphQL status-check contexts and GitHub's `isRequired(pullRequestNumber)` flag, preferring the test merge commit when a required context is present there. Missing required contexts remain pending. API errors fail the refresh instead of being converted into unknown rows; an absent classic branch-protection rule and inaccessible Team membership are handled explicitly.

The comment shows the contribution guide first, CI commands second and merge requirements last. The greeting mentions the PR author; reviewer usernames in merge requirements are inline code without mentions. CI is summarized without listing individual checks. The merge block contains GitHub's overall state and the requirement table without footer links. Other rules, including latest-push approval, are represented only by the overall state; no push-history, commit-comparison or review-thread queries are made. Fetching all active rules does not provide a complete GitHub merge-eligibility verdict: the status renderer interprets the three supported categories, while GitHub decides the final status and bypass eligibility.

## Validation

Use Node.js 24 for ESM checks and run `pre-commit run --all-files`. Validate changed workflows and render the welcome template after editing it. The oxfmt hook uses `ShigureLab/oxfmt-pre-commit-mirror`; its version and file scope are configured in `.pre-commit-config.yaml`.

After deploying to the default branch, verify welcome publishing, command permissions, rerun/cancel, bypass and label cleanup on a disposable draft PR. Local checks do not establish live token permissions or GPU/NPU execution.
