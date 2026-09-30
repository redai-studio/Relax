# Rerun, cancellation and comment commands

## Direct operations

Use these commands when the authenticated credential can write Actions in the target repository. Start with the PR's required checks and use the [shared target configuration](../../../.github/ci/config.json) to select its workflow or job.

```bash
gh run view <RUN_ID> --repo redai-studio/Relax --json headSha,event,status,conclusion,jobs,url
gh run view <RUN_ID> --repo redai-studio/Relax --log-failed
```

For the selected workflow on the current head, prefer an active run, then a non-cancelled run; use a cancelled run only when none survive. Within each group, choose the highest run number. Concurrency can cancel a higher-numbered duplicate while a lower-numbered run continues, so the largest run number alone does not identify the surviving run.

Confirm the PR association; for fork runs with an empty `pull_requests` array, match both head branch and head repository. Deployment workflows are outside the configured CI targets.

| Operation | GitHub CLI |
| --- | --- |
| Retry failed jobs | `gh run rerun <RUN_ID> --failed --repo redai-studio/Relax` |
| Retry a workflow | `gh run rerun <RUN_ID> --repo redai-studio/Relax` |
| Retry one job and its dependents | `gh run rerun <RUN_ID> --job <JOB_ID> --repo redai-studio/Relax` |
| Cancel a workflow | `gh run cancel <RUN_ID> --repo redai-studio/Relax` |

Obtain job IDs from `gh run view <RUN_ID> --json jobs --jq '.jobs[] | {name, databaseId}'`, or the REST jobs endpoint with `filter=latest`; do not guess IDs from browser URLs. Rerun requires a completed run. Wait for an active run, or cancel it within the authorized scope before rerunning.

Equivalent REST operations use `gh api --method POST <endpoint>`:

| Operation | Endpoint |
| --- | --- |
| Retry failed jobs | `repos/redai-studio/Relax/actions/runs/<RUN_ID>/rerun-failed-jobs` |
| Retry a workflow | `repos/redai-studio/Relax/actions/runs/<RUN_ID>/rerun` |
| Retry a job | `repos/redai-studio/Relax/actions/jobs/<JOB_ID>/rerun` |
| Cancel a workflow | `repos/redai-studio/Relax/actions/runs/<RUN_ID>/cancel` |

See [GitHub's workflow run API](https://docs.github.com/en/rest/actions/workflow-runs) for permissions and request semantics.

## PR comment interface

Rerun/cancel comments are available to the PR author and users with repository write/maintain/admin permission. Put a command on the first line of a new PR comment; later lines are ordinary comment text.

| Command | Scope |
| --- | --- |
| `/rerun` or `/rerun failed` | Failed jobs in the latest failed/timed-out workflows on the current head |
| `/rerun <target>` | One configured workflow or job and its dependents |
| `/rerun all` | All completed workflows from the configured targets |
| `/cancel <workflow>` or `/cancel all` | Whole active workflows, including their matrix jobs |
| `/help` | Contributor commands and targets |
| `/review` | Nyanpasu review request |

Targets with a `job` field support rerun only. Use a prepared body file with `gh pr comment <PR> --repo redai-studio/Relax --body-file <FILE>` and retain the returned URL. Editing a comment does not execute rerun/cancel/help again. The command action records source comment IDs to prevent duplicate execution.

Check the command reply and its workflow link, then verify the actual run. Neither the API nor comment reruns create an absent workflow run or approve a fork workflow.
