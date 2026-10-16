# Rerun CI

## Direct operations

Use these commands when the authenticated credential can write Actions in the target repository. Start with the PR's required checks and use the [shared target configuration](../../../.github/ci/config.json) to select its workflow or job.

```bash
gh run view <RUN_ID> --repo redai-studio/Relax --json headSha,event,status,conclusion,jobs,url
gh run view <RUN_ID> --repo redai-studio/Relax --log-failed
```

| Operation | GitHub CLI |
| --- | --- |
| Retry failed jobs | `gh run rerun <RUN_ID> --failed --repo redai-studio/Relax` |
| Retry a workflow | `gh run rerun <RUN_ID> --repo redai-studio/Relax` |
| Retry one job and its dependents | `gh run rerun <RUN_ID> --job <JOB_ID> --repo redai-studio/Relax` |

Obtain job IDs from `gh run view <RUN_ID> --json jobs --jq '.jobs[] | {name, databaseId}'`, or the REST jobs endpoint with `filter=latest`; do not guess IDs from browser URLs. Rerun requires a completed run. Wait for an active run, or cancel it within the authorized scope before rerunning.

Equivalent REST operations use `gh api --method POST <endpoint>`:

| Operation | Endpoint |
| --- | --- |
| Retry failed jobs | `repos/redai-studio/Relax/actions/runs/<RUN_ID>/rerun-failed-jobs` |
| Retry a workflow | `repos/redai-studio/Relax/actions/runs/<RUN_ID>/rerun` |
| Retry a job | `repos/redai-studio/Relax/actions/jobs/<JOB_ID>/rerun` |

See [GitHub's workflow run API](https://docs.github.com/en/rest/actions/workflow-runs) for permissions and request semantics.
