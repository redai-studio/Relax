# Local Inference Recovery

Relax supports rebuilding a failed Rollout or standalone ActorFwd service after it raises an exception, then continuing training. During recovery, the framework pauses new data production and consumption, waits for old operations to finish, and synchronizes the original training Actor's current weights to the new instance. Successful local recovery preserves the training Actor, optimizer, and scheduler without a Controller global restart.

This handles explicit exceptions that have returned, such as an HTTP error after request retries are exhausted. It adds no hang detection and does not make every inference failure locally recoverable. Unsafe recovery, recovery failure, or an exhausted budget falls back to the existing global restart path.

## Configuration

These settings are environment variables, not CLI arguments. Set them in the **training Job that runs the Controller**, before starting the training entrypoint. Setting them only in an SGLang subprocess has no effect.

| Environment variable | Default | Meaning |
|---|---|---|
| `RELAX_ENABLE_INFERENCE_RECOVERY` | Depends on the training mode | `true` or `1` enables recovery; `false` or `0` disables it. Controls local recovery of both Rollout and standalone ActorFwd. |
| `RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS` | `3` | Non-negative integer. A cumulative attempt budget shared by Rollout and ActorFwd; `0` sends the first failure to global restart. A negative or non-integer value raises an error when recovery is enabled. |
| `RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT` | `180` | Positive number of seconds to wait for old operations to exit, also used during handoff to global restart. This is not a total recovery timeout. |

| Training mode | Behavior when the switch is unset | Locally recoverable roles |
|---|---|---|
| Full-async (`--fully-async`) | Enabled when configuration checks pass; otherwise keeps the existing behavior | Rollout and standalone ActorFwd when actually deployed |
| Hybrid-async (`--hybrid`) | Enabled when configuration checks pass; otherwise keeps the existing behavior | Rollout. Forward computation runs inside the training Actor; there is no standalone ActorFwd service. |
| Colocate without hybrid | Disabled; explicitly set `true` or `1` to enable | Rollout |

Values are case-insensitive and ignore surrounding whitespace, so `TRUE` is equivalent to `true`. With an explicit `true` or `1`, unsupported settings cause startup to raise `Unsupported inference recovery configuration` and list the conflicting fields. When the switch is unset, unsupported configurations keep the existing behavior. Setting `false` or `0` disables the feature and skips its configuration checks.

### Enabling and disabling recovery

In your training launcher, before running `python -m relax.entrypoints.train`, set:

```bash
export RELAX_ENABLE_INFERENCE_RECOVERY=true
export RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS=3
export RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT=180
```

To disable recovery, set `RELAX_ENABLE_INFERENCE_RECOVERY=false`. These settings are read at training startup; changing the environment outside a running process does not reconfigure the current job.

For Ray Jobs, pass these variables to the remote training entrypoint. If your launcher already uses `RUNTIME_ENV_JSON`, merge the settings into its existing `env_vars` before submission, preserving the other runtime settings:

```bash
RUNTIME_ENV_JSON="$(python - <<'PY'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env.setdefault("env_vars", {}).update({
    "RELAX_ENABLE_INFERENCE_RECOVERY": "true",
    "RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS": "3",
    "RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT": "180",
})
print(json.dumps(runtime_env))
PY
)"
export RUNTIME_ENV_JSON
```

Keep using `ray job submit --runtime-env-json="${RUNTIME_ENV_JSON}"` in the submission command. The `scripts/entrypoint/ray-job.sh` script generates the base `RUNTIME_ENV_JSON`, so merge after that generation and before actual Job submission. Exporting variables only in the submission shell does not guarantee that they reach the remote Job.

## Supported configurations and conflicts

Recovery supports framework-managed static resources, the Megatron training backend, standard SGLang Rollout, and GRPO. The training Actor, data queue, and recovery state service must remain available. Local recovery does not repair the training Actor itself, recover lost nodes or GPUs, or migrate or shrink resources after failure.

| Requirement | Required configuration |
|---|---|
| Training and algorithm | `train_backend=megatron`, `advantage_estimator=grpo`, and `kl_coef=0`; do not enable `--use-kl-loss`, Critic, or OPD. |
| Generation entrypoint | Keep the default `relax.engine.rollout.sglang_rollout.generate_rollout`; do not set `--custom-generate-function-path` or `--dynamic-sampling-filter-path`, or enable Agentic Rollout. |
| Model and reward | Leave `--multimodal-keys` unset; do not enable `--lora-adapter-mode` or configure GenRM instances. |
| Resources and routing | Do not use `--rollout-external`, `--autoscaler-config`, `--sglang-config`, `--sglang-router-ip`, or `--use-slime-router`. Keep framework-managed engines and routing. |
| Other execution options | Do not enable `--use-fault-tolerance`, `--use-dynamic-global-batch-size`, `--debug-train-only`, or `--debug-rollout-only`. `--use-fault-tolerance` selects a separate Rollout fault-tolerance path; it does not enable this feature. |
| Colocate | Require training offload, Rollout offload, and weight backup: `offload_train=True`, `offload_rollout=True`, and `enable_weights_backuper=True`. Argument validation for `--colocate` enables offload automatically; keep weight backup enabled. |
| Hybrid | Require `offload_train=False` and `offload_rollout=False`; do not combine with `--offload`, `--offload-train`, or `--offload-rollout`. |
| Routing replay | Both full-async and hybrid local recovery reject `--use-routing-replay`. Hybrid argument validation automatically sets `fully_async=True`, so the same restriction applies. |

These fields refer to the final parsed configuration. For example, if startup reports `enable_weights_backuper=False`, check the launcher for `--disable-weights-backuper`. See the [full-async](./fully-async-training.md) and [hybrid](./hybrid-training.md) guides for the complete mode configuration.

`--partial-rollout` itself is not a conflict, but recovery may still discard partial and pending samples. It does not guarantee continuation of a response that was only partly generated before the failure.

## Recovery flow

1. When Rollout or standalone ActorFwd raises a run exception, the Controller checks the remaining recovery budget and pauses new work.
2. It waits for in-flight training and old inference operations to finish, so an old instance cannot keep writing data or be removed during an unfinished weight transfer.
3. It uses the original training Actor's next training position to clear untrained data and discard pending generation samples. The input read cursor does not move backward.
4. It confirms that old inference children have exited, then rebuilds the failed inference service on the same resource group. The healthy training Actor and other services remain.
5. It transfers the original training Actor's current weights to the replacement and waits for weight synchronization and required memory restoration before resuming generation and training.

The budget counts attempts, and successful recovery does not reset it. With a budget of `3`, one Rollout recovery, one ActorFwd recovery, and another Rollout recovery use the entire budget; the next exception falls back to global restart. A failed recovery attempt also falls back immediately rather than repeatedly rebuilding until the remaining budget is consumed. Global restart creates a fresh recovery state and resets the local budget.

::: warning Data and policy behavior
Recovery may discard partially generated samples and batches that have finished generation but have not been trained. It does not promise zero data loss. Data order, batches, and policy versions can change after recovery; strict on-policy behavior and step-by-step equivalence to a fault-free run are not guaranteed. This is why recovery is disabled by default for pure colocate. Jobs sensitive to these changes should set `RELAX_ENABLE_INFERENCE_RECOVERY=false`.
:::

## Health checks and global restart

`--use-health-check` controls the existing health checks; it is not required to enable local recovery. Keeping it enabled provides a fallback. This feature adds no detection for silent hangs.

Both paths can observe the same failure. A health callback defers duplicate handling only while a local recovery thread is actively handling that role; otherwise it retains the existing restart path. If a health check takes ownership first, it can still trigger global restart. Enabling recovery therefore does not guarantee local handling of every inference failure.

Budget exhaustion, failed rebuilding or weight restoration, concurrent role failures, and faults during weight synchronization fall back to the existing global restart path. If old work cannot exit safely, the job terminates to avoid overlapping old and new resource operations. A completely unreachable service process is also not guaranteed to complete local stop and rebuild.

`--max-global-restart` separately limits global restarts and defaults to `3`; exceeding it ends the job. This is a different counter from the local budget. Global restarts reached through local escalation or health checks both use this limit. Global restart rebuilds training state; restoring from an existing checkpoint depends on the existing save/load configuration. Local recovery does not save an extra checkpoint or guarantee that global restart retains the latest training progress.

For diagnosis, find `Local recovery completed` in the logs. It records the role, cumulative attempt count, replacement generation, next training position, and elapsed time. An exhausted budget reports `Local recovery budget (...) exhausted`. Continue checking training logs to establish whether global restart and training actually resumed; service status alone is insufficient. See [Health Check Manager](./health-check-manager.md) for the health-check path.
