---
name: relax-github-ci
description: Inspect and operate Relax GitHub PR CI. Use when checking required checks, retrying failed jobs, cancelling workflows, applying authorized exemptions, or maintaining CI automation.
---

# Relax GitHub CI

## Inspect the current PR

Identify the repository and PR from the request or current branch. Capture the current head and required checks before choosing an operation:

```bash
gh pr view <PR> --repo redai-studio/Relax --json number,url,headRefOid,headRefName,baseRefName,state
gh pr checks <PR> --repo redai-studio/Relax --required
```

Read `.github/ci/config.json` for workflow/job targets. Match runs to the PR, head SHA, workflow and head repository; an old run can have a recent update time after a rerun. Inspect logs for failed required or specifically requested checks.

## Choose the operation

Prefer `gh run` or the REST API when the authenticated credential has Actions write access. Use PR comment commands when direct Actions access is unavailable and the actor is eligible to use the comment interface. See [rerun, cancellation and comment commands](references/commands.md) for exact operations.

Use the narrowest scope that addresses the request. A job rerun includes dependent jobs; workflow cancellation includes sibling matrix jobs. Refresh the PR head immediately before a write and reassess if it changed.

For an explicitly requested exemption, read [bypass and CI maintenance](references/maintenance.md). CI team membership and an authorized exemption request are both required.

## Verify the result

Inspect the affected run after writing. Distinguish a request being accepted from a new attempt starting or finishing. Verify cancellation reaches a terminal state. If an API call times out or returns EOF, read back the run or comment before retrying.

For automation changes, read the relevant action/workflow and [maintenance reference](references/maintenance.md). Use `gh-stack` for native stacked PR operations.
