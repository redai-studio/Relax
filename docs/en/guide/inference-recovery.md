# Local Inference Recovery

**Local recovery discards samples that have been read but not yet trained, as well as partially generated Rollout samples. Batches and policy versions may change after recovery, so this feature is intended for off-policy training and does not guarantee strict on-policy behavior. If your job is sensitive to on-policy behavior, do not add `--use-fault-tolerance` to the training command.**

Relax supports rebuilding a failed Rollout or standalone ActorFwd service after it raises an exception, then continuing training. During recovery, the framework pauses new data production and consumption, waits for old operations to finish, and synchronizes the original training Actor's current weights to the new instance. Successful local recovery preserves the training Actor, optimizer, and scheduler without a Controller global restart.

This handles explicit exceptions that have returned, such as an HTTP error after request retries are exhausted. It adds no hang detection and does not make every inference failure locally recoverable. Unsafe recovery, recovery failure, or an exhausted budget falls back to the existing global restart path.

## Configuration

Use the existing `--use-fault-tolerance` flag to enable fault tolerance; it is disabled by default. When the configuration meets the requirements below, the Controller handles local inference-service recovery and disables the legacy engine-level health monitors and automatic rebuilding. Unsupported configurations retain the existing Rollout engine-level fault tolerance. The two implementations do not handle recovery concurrently.

The following environment variables configure only the local recovery budget and wait time. Set them in the **training Job that runs the Controller**, before starting the training entrypoint. Setting them only in an SGLang subprocess has no effect.

| Environment variable | Default | Meaning |
|---|---|---|
| `RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS` | `3` | Non-negative integer. A cumulative attempt budget shared by Rollout and ActorFwd; `0` sends the first failure to global restart. Negative values raise when local recovery is enabled; non-integer values are rejected by environment validation at startup. |
| `RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT` | `180` | Positive number of seconds to wait for old operations to exit, also used during handoff to global restart. This is not a total recovery timeout. |

| Training mode | Locally recoverable roles when the CLI flag is enabled and requirements are met |
|---|---|
| Full-async (`--fully-async`) | Rollout and standalone ActorFwd when actually deployed |
| Hybrid-async (`--hybrid`) | Rollout. Forward computation runs inside the training Actor; there is no standalone ActorFwd service. |
| Colocate without hybrid | Rollout |

Synchronous training with separate Actor and Rollout GPUs uses engine-level fault tolerance.

Both variables follow the shared environment configuration rules: empty values use the default, as if unset. Values that cannot be converted to the declared numeric type are rejected at startup, even when local recovery is disabled.

### Enabling and disabling recovery

Add `--use-fault-tolerance` to your existing training command. To adjust the local recovery budget and wait time, set these variables before starting training:

```bash
export RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS=3
export RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT=180
```

To disable automatic fault tolerance, remove `--use-fault-tolerance` from the training command. These settings are read at training startup; changing the environment outside a running process does not reconfigure the current job.

For Ray Jobs, pass these variables to the remote training entrypoint. If your launcher already uses `RUNTIME_ENV_JSON`, merge the settings into its existing `env_vars` before submission, preserving the other runtime settings:

```bash
RUNTIME_ENV_JSON="$(python - <<'PY'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env.setdefault("env_vars", {}).update({
    "RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS": "3",
    "RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT": "180",
})
print(json.dumps(runtime_env))
PY
)"
export RUNTIME_ENV_JSON
```

Keep using `ray job submit --runtime-env-json="${RUNTIME_ENV_JSON}"` in the submission command. The `scripts/entrypoint/ray-job.sh` script generates the base `RUNTIME_ENV_JSON`, so merge after that generation and before actual Job submission. Exporting variables only in the submission shell does not guarantee that they reach the remote Job.

## Supported configurations and fallback

Recovery supports framework-managed static resources, the Megatron training backend, standard SGLang Rollout, and GRPO. The training Actor, data queue, and recovery state service must remain available. Local recovery does not repair the training Actor itself, recover lost nodes or GPUs, or migrate or shrink resources after failure.

| Requirement | Required configuration |
|---|---|
| Training and algorithm | `train_backend=megatron`, `advantage_estimator=grpo`, and `kl_coef=0`; do not enable `--use-kl-loss`, Critic, or OPD. |
| Generation entrypoint | Keep the default `relax.engine.rollout.sglang_rollout.generate_rollout`; do not set `--custom-generate-function-path` or `--dynamic-sampling-filter-path`, or enable Agentic Rollout. |
| Model and reward | Leave `--multimodal-keys` unset; do not enable `--lora-adapter-mode` or configure GenRM instances. |
| Resources and routing | Do not use `--rollout-external`, `--autoscaler-config`, `--sglang-config`, `--sglang-router-ip`, or `--use-slime-router`. Keep framework-managed engines and routing. |
| TransferQueue | Component recovery requires SimpleStorage. A job using MooncakeStore keeps engine-level fault tolerance. With `--tq-rdma-mode=auto`, the backend selected after initialization and attach checks determines the recovery mode. |
| Other execution options | Do not enable `--use-dynamic-global-batch-size`, `--debug-train-only`, or `--debug-rollout-only`. |
| Colocate | Require training offload, Rollout offload, and weight backup: `offload_train=True`, `offload_rollout=True`, and `enable_weights_backuper=True`. Argument validation for `--colocate` enables offload automatically; keep weight backup enabled. |
| Hybrid | Require `offload_train=False` and `offload_rollout=False`; do not combine with `--offload`, `--offload-train`, or `--offload-rollout`. |
| Old-policy snapshots | With `--keep-old-actor`, colocate and hybrid use engine-level fault tolerance to preserve the old policy used for training. |
| Routing replay | Both full-async and hybrid local recovery reject `--use-routing-replay`. Hybrid argument validation automatically sets `fully_async=True`, so the same restriction applies. All modes use engine-level fault tolerance when rollout replay is delivered by reference. |

These fields refer to the final parsed configuration. When requirements are not met, `--use-fault-tolerance` retains engine-level fault tolerance. For example, `enable_weights_backuper=False` prevents service-level local recovery; check the launcher for `--disable-weights-backuper`. See the [full-async](./fully-async-training.md) and [hybrid](./hybrid-training.md) guides for the complete mode configuration.

`--partial-rollout` itself is not a conflict, but recovery may still discard partial and pending samples. It does not guarantee continuation of a response that was only partly generated before the failure.

## Recovery flow

1. When Rollout or standalone ActorFwd raises a run exception, the Controller checks the remaining recovery budget and pauses new work.
2. It waits for in-flight training and old inference operations to finish, so an old instance cannot keep writing data or be removed during an unfinished weight transfer.
3. It uses the original training Actor's next training position to clear untrained data and discard pending generation samples. The input read cursor does not move backward.
4. It confirms that old inference children have exited, then rebuilds the failed inference service on the same resource group. The healthy training Actor and other services remain.
5. It transfers the original training Actor's current weights to the replacement and waits for weight synchronization and required memory restoration before resuming generation and training.

Before replacement, the old Rollout stops accepting evaluations and waits for every accepted request to settle. Recovery can discard an evaluation whose generation failed. For evaluations that produced results, it waits for scoring and reporting. If the Controller observes a scoring or reporting error before global restart takes over, the job fails. Evaluation and active operations share the `RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT` deadline. When it expires, global restart takes over before partition cleanup. When a run with component recovery enabled enters its normal shutdown path, it checks for known scoring or reporting errors before propagating a generation error. If only generation has failed, the job reports that error without waiting for an unfinished judge.

In fully-async and hybrid training, a Rollout failure triggers global restart if the Actor has already committed its final step or if a consumer needed to rebuild an untrained partition has finished its run. A dataset fingerprint mismatch when loading a saved cursor fails the job immediately.

The budget counts attempts, and successful recovery does not reset it. With a budget of `3`, one Rollout recovery, one ActorFwd recovery, and another Rollout recovery use the entire budget; the next exception falls back to global restart. A failed recovery attempt also falls back immediately rather than repeatedly rebuilding until the remaining budget is consumed. Global restart creates a fresh recovery state and resets the local budget.

::: warning Data and policy behavior
Recovery may discard partially generated samples and batches that have finished generation but have not been trained. It does not promise zero data loss. Data order, batches, and policy versions can change after recovery; strict on-policy behavior and step-by-step equivalence to a fault-free run are not guaranteed. Jobs sensitive to these changes should not enable `--use-fault-tolerance`.
:::

With component recovery enabled, Rollout uses the engine pool allocated at startup. `POST /rollout/scale_out` returns HTTP 409 for both Ray-native capacity requests (`num_replicas`) and external engine registration (`engine_urls`). The Controller coordinates engine recovery. Calling `GET /rollout/recover_rollout_engines` manually also returns HTTP 409.

## Health checks and global restart

`--use-health-check` controls the existing health checks; it is not required to enable local recovery. Keeping it enabled provides a fallback. This feature adds no detection for silent hangs.

Both paths can observe the same failure. A health callback defers duplicate handling only while a local recovery thread is actively handling that role; otherwise it retains the existing restart path. If a health check takes ownership first, it can still trigger global restart. Enabling recovery therefore does not guarantee local handling of every inference failure.

Budget exhaustion, failed rebuilding or weight restoration, and concurrent role failures fall back to the existing global restart path. If old work cannot exit safely, the job terminates to avoid overlapping old and new resource operations. A completely unreachable service process is also not guaranteed to complete local stop and rebuild.

In full-async and hybrid modes with component recovery enabled, the Actor must confirm that the receivers are ready and that Rollout has resumed. If either handshake fails, the Actor reports a training error and leaves the step uncommitted. With health checks enabled, the existing restart policy handles the error. With health checks disabled, the job exits with an error.

`--max-global-restart` separately limits global restarts and defaults to `3`; exceeding it ends the job. This is a different counter from the local budget. Global restarts reached through local escalation or health checks both use this limit. Global restart rebuilds training state; restoring from an existing checkpoint depends on the existing save/load configuration. Local recovery does not save an extra checkpoint or guarantee that global restart retains the latest training progress.

For diagnosis, find `Local recovery completed` in the logs. It records the role, cumulative attempt count, replacement generation, next training position, and elapsed time. An exhausted budget reports `Local recovery budget (...) exhausted`. Continue checking training logs to establish whether global restart and training actually resumed; service status alone is insufficient. See [Health Check Manager](./health-check-manager.md) for the health-check path.
