# External agent capacity

Apply this check when agents submit work to a centralized remote platform with a hard concurrency limit. Mark it not applicable for direct local execution without an external slot gate.

## Resolve effective values

Read the final launch script and apply Relax defaults:

```text
agentic_concurrency
  -> over_sampling_batch_size
  -> rollout_batch_size
```

Read train settings and every Eval dataset's effective override:

```text
agentic_concurrency
n_samples_per_prompt
agentic_eval_concurrency
n_samples_per_eval_prompt per dataset
agentic_prelaunch
partial_rollout
fully_async
Eval enabled or disabled
```

When `agentic_eval_concurrency` is unset, Relax derives each dataset's logical Eval Group concurrency by rounding train
Session capacity up to that dataset's Eval Group size. Eval datasets run serially.

## Calculate session limits

```text
T = agentic_concurrency * n_samples_per_prompt
G_d = dataset_d.n_samples_per_eval_prompt
C_d = explicit agentic_eval_concurrency or ceil(T / G_d)
E_d = C_d * G_d
E_peak = max(E_d across Eval datasets)
```

Thus derived `E_d >= T`. Ordinary RM uses singleton Runtime Groups and Group RM keeps multi-Session Groups, while both
retain the same `E_d` Session ceiling.

For a shared train/Eval executor:

| Eval enabled | Train sessions remain resident during Eval | Required slots |
| --- | --- | --- |
| No | — | `external_slots >= T` |
| Yes | No | `external_slots >= max(T, E_peak)` |
| Yes | Yes | `external_slots >= T + E_peak` |

Select **Yes** for retained train sessions when `--agentic-prelaunch`, `--partial-rollout`, or `--fully-async` is enabled. Prelaunch shares train resident capacity, so it does not increase `T`; it makes `T` overlap with `E_peak`.

For dedicated executors:

```text
train_external_slots >= T
eval_external_slots >= E_peak
```

## Find the external limit

Inspect the external platform configuration, server executor size, sandbox pool, queue, or deployment quota. Do not infer the value from Relax GPU count or Ray concurrency. If the platform has separate train and Eval pools, record both.

## Verdict

| Result | Condition |
| --- | --- |
| `PASS` | Known slots satisfy the applicable inequality |
| `UNSAFE` | Known slots are below the requirement |
| `UNVERIFIED` | A hard limit exists but its effective slots are unknown |

For train and Group-RM Eval, fewer slots can split capacity across incomplete multi-Session Groups so none reaches the
all-Session first-request barrier. Ordinary-RM Eval uses singleton Runtime Groups, where fewer slots primarily reduce
throughput.

A deadlock-shaped signal is `warming_prepare_group_count > 0` with unchanged resident counts and recurring
`idle_heartbeat`, while no Group becomes ready. Prepare has no dedicated first-request-barrier timeout, so an external
slot-holding cycle needs agent/client/platform cancellation or run cleanup to break it.

Report:

```text
Execution shape: local | remote shared | remote dedicated
T:
E per dataset:
E_peak:
Train retained during Eval: yes/no
Required slots:
Configured slots:
Verdict: PASS | UNSAFE | UNVERIFIED
Evidence:
```

Source anchors:

- `relax/utils/arguments.py`: Agentic concurrency defaults
- `relax/agentic/pipeline/runtime.py::agentic_eval_concurrency_from_args`
- `relax/agentic/rollout.py::_agentic_eval_runtime_concurrency`
- `docs/en/guide/agentic-rollout.md#configure-runtime-behavior`
