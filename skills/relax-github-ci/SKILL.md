---
name: relax-github-ci
description: Inspect and operate Relax GitHub PR CI. Use when checking required checks, retrying failed jobs, cancelling workflows, applying authorized exemptions, or maintaining CI automation.
---

# Relax GitHub CI

## Inspect the current PR

Identify the repository and PR from the request or current branch. Capture the current head and required checks before choosing an operation:

```bash
gh pr view <PR> --repo redai-studio/Relax --json number,url,headRefOid,headRefName,headRepository,headRepositoryOwner,baseRefName,state
gh pr checks <PR> --repo redai-studio/Relax --required
```

Read `.github/ci/config.json` for workflow/job targets. Match runs to the PR, head SHA, workflow and head repository; an old run can have a recent update time after a rerun. Inspect logs for failed required or specifically requested checks.

For the selected workflow on the current head, prefer an active run, then a non-cancelled run; use a cancelled run only when none survive. Within each group, choose the highest run number. Concurrency can cancel a higher-numbered duplicate while a lower-numbered run continues. For fork runs with an empty `pull_requests` array, match both head branch and head repository. Deployment workflows are outside the configured CI targets.

## Choose the operation

Prefer `gh run` or the REST API when the authenticated credential has Actions write access. See [rerun operations](references/rerun.md) for retrying failed jobs, workflows or individual jobs. To cancel a workflow, use `gh run cancel <RUN_ID> --repo redai-studio/Relax` or `POST repos/redai-studio/Relax/actions/runs/<RUN_ID>/cancel`.

Use the narrowest scope that addresses the request. A job rerun includes dependent jobs; workflow cancellation includes sibling matrix jobs. Refresh the PR head immediately before a write and reassess if it changed.

For an explicitly requested exemption, read [bypass and CI maintenance](references/maintenance.md). CI team membership and an authorized exemption request are both required.

## PR comment interface

Use comment commands when direct Actions access is unavailable and the actor is eligible. Rerun/cancel comments are available to the PR author and users with repository write/maintain/admin permission. The command action silently ignores `/rerun`, `/cancel`, and `/help` comments whose GitHub `user.type` is `Bot`, without replying. Put a command on the first line of a new PR comment; later lines are ordinary comment text.

| Command | Scope |
| --- | --- |
| `/rerun` or `/rerun failed` | Failed jobs in the latest failed/timed-out workflows on the current head |
| `/rerun <target>` | One configured workflow or job and its dependents |
| `/rerun all` | All completed workflows from the configured targets |
| `/cancel <workflow>` or `/cancel all` | Whole active workflows, including their matrix jobs |
| `/help` | Contributor commands and targets |
| `/review` | Nyanpasu review request |

Targets with a `job` field support rerun only. Use a prepared body file with `gh pr comment <PR> --repo redai-studio/Relax --body-file <FILE>` and retain the returned URL. Editing a comment does not execute rerun/cancel/help again. The command action records source comment IDs to prevent duplicate execution.

## Verify the result

Inspect the affected run after writing. Distinguish a request being accepted from a new attempt starting or finishing. Verify cancellation reaches a terminal state. If an API call times out or returns EOF, read back the run or comment before retrying.

For comment commands, check the reply and its workflow link, then verify the actual run. Neither API nor comment reruns create an absent workflow run or approve a fork workflow.

For automation changes, read the relevant action/workflow and [maintenance reference](references/maintenance.md). Use `gh-stack` for native stacked PR operations.
