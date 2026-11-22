# External agent capacity

Resolve actual Group concurrency, Group size, Eval concurrency, prelaunch/retention, and shared versus dedicated pools.

```text
C = agentic_concurrency -> over_sampling_batch_size -> rollout_batch_size
N = n_samples_per_prompt
L = C * N
G_d = dataset_d.n_samples_per_eval_prompt
C_d = explicit agentic_eval_concurrency or ceil(L / G_d)
E_d = C_d * G_d
E_peak = max(E_d across serial Eval datasets)
```

For a shared executor, estimate `max(L, E_peak)` available task slots when training resources exit before Eval;
use `L + E_peak` when prelaunch, partial rollout or fully async keeps train resources resident during Eval. Dedicated
pools have separate train and Eval budgets. Check per-host quotas and other jobs. These values describe logical demand;
background cleanup can temporarily retain additional physical resources. The executor owns its physical quota.

Train and Eval both lease fully registered Groups before every member's first IR is required. Ready members generate
independently, allowing progress when siblings queue at the executor. Complete Group export/filter/replenishment stays.
Check application dependencies on queued tasks in the same saturated executor; independent model generation cannot
resolve such cycles.

The launched `--agent-command` cleans up remote jobs, containers, and descendants on natural exit and when Relax sends
SIGTERM because of a timeout or another reason. Inspect its implementation
and actual executor state. Relax's process cleanup is a best-effort fallback and does not discover remote quotas.

Report `C`, `N`, `L`, each `E_d`, overlap and available quotas. Use `PASS` for known adequate quotas and observed command
cleanup; `UNSAFE` for known insufficient resources or application dependency cycles; `UNVERIFIED` when evidence is
missing. Logical completion and finalized sample counts do not prove physical resources have exited.

Source anchors: `relax/utils/arguments.py`, `relax/agentic/pipeline/runtime.py::agentic_eval_concurrency_from_args`,
`relax/agentic/rollout.py::_agentic_eval_runtime_concurrency`, and the agent adapter/executor implementation.
