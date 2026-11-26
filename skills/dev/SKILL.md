---
name: relax-dev-debug
description: Develop and debug the Relax reinforcement learning project. Use
  this skill whenever modifying code in the relax/ directory, or running remote
  training jobs on a Ray cluster for validation. Also use it when the user
  mentions training, debugging training runs, submitting Ray jobs, or fixing
  training errors.
---

# Relax Development & Debugging

This skill covers two workflows: making code changes to the Relax project, and validating those changes by running a training job on a remote Ray cluster.

For project structure and coupling details, see `AGENTS.md`.

______________________________________________________________________

## Part 1: Development

### Minimum-change principle

Unless the user explicitly asks for a refactoring, apply the smallest diff that achieves the goal. This means:

- Only touch files and lines directly required by the change. Don't "improve" nearby code, reformat untouched lines, or rename things that aren't part of the task.
- Preserve the existing code style in each file — naming conventions, import ordering, indentation, comment style.
- Don't introduce new dependencies unless the change itself demands it.
- Don't alter function signatures, class hierarchies, or public APIs unless that is the stated purpose of the change.

If the user says "refactor" or "restructure", the minimum-change constraint is lifted — but confirm the scope before proceeding.

### 测试准入与验证

新增、修改或审查测试前，阅读 [测试质量准则](../code-review/references/test-quality-checklist.md)，并遵守 `AGENTS.md` 的 Testing。

1. 先定位真实风险、已有覆盖和产品入口，再决定是否需要永久用例；一次性验证、脚本参数核对不默认进入仓库。
2. 对需要新增或重写的用例，明确生产调用、mock 边界、独立期望、可观察结果，以及哪个具体错误应使它失败。不得用源码/补丁/AST 检查或 fixture 自己的计算替代被测能力。
3. 优先扩展已有相关用例。参数行按独立故障选择，不能为每个脚本、常量和实现细节机械生成测试。
4. 实际验证与结论一一对应；记录版本、命令、结果和硬件限制。没有运行的旧版本对照、mutation、GPU 或多节点路径，明确标为未验证。
5. 删除或合并用例前指出失去的保护、保留的更强用例或失效的旧契约。review-only 请求只输出证据与建议，不自动修改测试或启动训练。

______________________________________________________________________

## Part 2: Debugging (Remote Training Validation)

**Do not enter this workflow on your own.** Only proceed when the user explicitly asks to debug/validate on a remote cluster and provides a `RAY_ADDRESS`. Code changes alone do not trigger a debug run — the user decides when to test on real hardware.

The goal is to submit a training run to a remote Ray cluster via `scripts/entrypoint/ray-job.sh`, monitor it via `ray job logs`, and either confirm it works or fix errors and retry.

______________________________________________________________________

### Standard launch flow (canonical)

Every training launch — the **first** one and every **resubmit after a fix** — follows the same three steps. Do not skip the pre-flight cleanup; stale Ray Serve apps from a failed previous job will silently break the new job at Router startup.

**Step A — Pre-flight cleanup (always run first):**

```bash
ray serve shutdown -y
```

Drops all stale Ray Serve applications/deployments left behind by a previous run. This is **NOT** a forbidden destructive op (see `skills/ssh-ray-cluster/SKILL.md`) — `ray serve shutdown` only tears down Serve state, not the Ray runtime, not training processes, and not other tenants' jobs. Always run it before resubmitting.

**Step B — Submit the training job:**

```bash
bash scripts/entrypoint/ray-job.sh scripts/training/text/<run-script>.sh
```

(or whichever subdirectory matches the run script). The entrypoint handles residual python/sglang cleanup, env setup, and `ray job submit` for you. Capture the JOB_ID and the log file path that the run script writes to (typically `log/<EXPERIMENT>-<TIMESTAMP>.log`).

**Step C — Monitor the new log file:**

Tail/grep the log file for progress (`step`, `iteration`) and errors (`Error`, `Traceback`, `Exception`, `OOM`). Apply the noise-filter pattern from "Step 2: Monitor the job" below.

**On error → fix → resubmit:** loop back to Step A (the `ray serve shutdown -y` is mandatory each time). Stop after 3 consecutive failed resubmits and report to the user.

**Strictly forbidden during this flow** (per `skills/ssh-ray-cluster/SKILL.md`): `ray stop`, `pkill -9 python`, `bash scripts/tools/kill_for_ray.sh`, `ray job stop` against unrelated jobs, `rm -rf /tmp/ray/`. `ray serve shutdown -y` and `ray job stop <our-job-id>` are the **only** state-changing operations allowed without explicit user approval.

______________________________________________________________________

> **⚠️ MANDATORY**: All `ray job submit` commands in this debugging workflow **MUST** include `RAY_NO_WAIT=1` so the submission is non-blocking. This allows you to immediately proceed to monitoring via `ray job logs` without the shell hanging on the submit call. Every command example below already includes it — do not omit it.

### Workflow overview

1. Pick a training script from `scripts/training/` (or create a new one).
2. **Edit the script directly** to adjust training parameters (model paths, hyperparameters, resource config, etc.).
3. Submit via `ray-job.sh`.

### Available training scripts

```
scripts/training/
├── text/           # Text-only models (Qwen3-4B, Qwen3-30B, Qwen3.5, etc.)
├── multimodal/     # Vision-language & omni models (Qwen3-VL, Qwen3-Omni, Qwen3.5)
├── genrm/          # GenRM (LLM-as-judge) configurations
└── hpc/            # HPC-specific configurations
```

Choose the script closest to your target configuration and modify it in place.

### Prerequisites

The user must provide the Ray cluster address and model directory (ask if missing). If the user gives a `RAY_ADDRESS` without a port (e.g. `x.x.x.x`), assume port `6379` and use `x.x.x.x:6379`.

| Variable    | Purpose                                     | Example                                        |
| ----------- | ------------------------------------------- | ---------------------------------------------- |
| `RAY_ADDRESS` | Ray cluster URL                           | `x.x.x.x:6379`                                |
| `MODEL_DIR`   | Base directory containing models & data on the cluster | `/path/to/model` |

#### Environment variables

The entrypoint `scripts/entrypoint/ray-job.sh` auto-detects or defaults most variables:

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `RAY_ADDRESS` | **Yes** | — | Ray cluster address. Set as env var before calling the script. |
| `MODEL_DIR` | **Yes** | — | Base path for models/data on the cluster. Training scripts resolve paths relative to this. |
| `WORKING_DIR` | **Remote debug only** | — | Set to `./` when debugging from a local checkout. Adds `--working-dir` to `ray job submit` so the cluster can access your local code. Required when iterating on code changes without syncing to the cluster. |
| `RAY_NO_WAIT` | No | — | Set to `1` to add `--no-wait` to `ray job submit`, making it non-blocking. **Recommended for debugging** — the script returns immediately after submission so you can monitor via `ray job logs` separately. |

### Step 0 (TorchJob only): Prepare the Ray cluster

When debugging on a TorchJob cluster, the Ray cluster is **not** pre-existing — you need to start it first. The entrypoint scripts handle Ray cluster formation and then block indefinitely, keeping the cluster alive for job submission.

#### Single-node (1 pod)

```bash
bash scripts/entrypoint/local.sh && sleep 10d
```

This sources the local entrypoint (starts a Ray head node with auto-detected GPUs) and then sleeps to keep the cluster alive.

#### Multi-node (N pods)

Run on **every pod** (head + workers). The `spmd-multinode.sh` script auto-detects head vs worker role via `MASTER_ADDR` vs `POD_NAME`:

```bash
bash scripts/entrypoint/spmd-multinode.sh <(echo sleep 10d)
```

This passes a dummy "training script" (`sleep 10d`) so the entrypoint completes cluster formation and then blocks. The head node waits for all workers to join before executing the sleep. Worker nodes join and sleep automatically.

> After the cluster is up, proceed to Step 1 to submit the actual training job via `ray-job.sh`.

### Step 1: Submit the training job

First check that the cluster is reachable:

```bash
ray status --address="$RAY_ADDRESS"
```

Then submit via `ray-job.sh`


```bash
RAY_NO_WAIT=1 WORKING_DIR="./" RAY_ADDRESS=x.x.x.x:6379 MODEL_DIR=/path/to/model \
    bash -x scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen3-4B-8xgpu.sh
```

To use a different model/config, just pick a different script:

```bash
RAY_NO_WAIT=1 WORKING_DIR="./" RAY_ADDRESS=x.x.x.x:6379 MODEL_DIR=/path/to/workspace \
    bash -x scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen3-4B-8xgpu-async.sh
```

When debugging from a local code checkout (not from the cluster's synced code), set `WORKING_DIR`

This tells `ray job submit` to upload your local working directory to the cluster, ensuring your latest code changes are used.

Capture the JOB_ID from the output.

### Step 2: Monitor the job

Check every 10 minutes using `ray job logs`.

#### Log noise filtering

`ray job logs` output is very verbose. To avoid wasting tokens, **always** pipe through filters that strip high-frequency noise lines. The mandatory exclusion patterns are:

| Pattern | Example | Reason |
|---------|---------|--------|
| SGLang decode throughput | `Decode batch, #running-req: 95, #token: 275989, token usage: 0.18, cuda graph: True, gen throughput (token/s): 357` | Printed every ~1 s per TP rank; purely operational |
| SGLang prefill throughput | `Prefill batch, #new-seq: ..., #new-token: ..., #cached-token: ...` | Same frequency as decode; no diagnostic value |
| First rollout sample dump | `First rollout sample: [...]` | Contains full prompt text; single line can exceed 2 k tokens |
| Rollout sample content | `Rollout samples:` or `Sample [0-9]` followed by long text | Same reason as above |

Apply the filter in **every** `ray job logs` invocation using `grep -v`:

```bash
RAY_LOG_FILTER='grep -vE "(Decode batch, #running-req|Prefill batch, #new-seq|First rollout sample:|Rollout samples:|gen throughput \(token/s\))"'
```

```bash
# Job status
ray job status --address="$RAY_ADDRESS" "$JOB_ID"

# Training progress (look for step/iteration >= 2)
ray job logs --address="$RAY_ADDRESS" "$JOB_ID" 2>&1 | eval "$RAY_LOG_FILTER" | grep -iE "iteration\s+[0-9]|step\s+[0-9]"

# Check for errors
ray job logs --address="$RAY_ADDRESS" "$JOB_ID" 2>&1 | eval "$RAY_LOG_FILTER" | grep -iE "error|traceback|exception|CUDA|OOM" | tail -30
```

### Step 3a: Normal path — kill after step 2

```bash
ray job stop --address="$RAY_ADDRESS" "$JOB_ID"
```

Report success and stop.

### Step 3b: Error path — analyze, fix, resubmit

1. **Extract context**: `ray job logs --address="$RAY_ADDRESS" "$JOB_ID" 2>&1 | eval "$RAY_LOG_FILTER" | grep -iEB 20 "error|traceback|exception" | tail -80`
2. **Identify root cause** — common issues: import errors, shape mismatches, CUDA OOM, Ray version mismatch.
3. **Fix the code** — apply a minimal fix. If the fix involves training parameters, edit the training script directly.
4. **Resubmit** — run the same `ray-job.sh` command again.
5. **Retry limit** — stop after 3 consecutive failures and report to user.

______________________________________________________________________

## Troubleshooting

### Residual Ray Serve applications blocking new jobs

Failed jobs can leave Ray Serve deployments in `DEPLOY_FAILED` or `RUNNING` state. New jobs will fail at Router startup (`assert process.is_alive()`) if stale applications remain. Clean them up before resubmitting:

```bash
# List current serve applications
curl -s http://${CLUSTER_IP}:8265/api/serve/applications/ | python3 -m json.tool

# Delete all serve applications
curl -s -X DELETE http://${CLUSTER_IP}:8265/api/serve/applications/
```

### RuntimeEnv key conflicts

If both `ray job submit --runtime-env-json '{"env_vars": {"KEY": "val"}}'` and `ray.init(runtime_env={"env_vars": {"KEY": "val2"}})` set the same env var key, Ray raises a `ValueError`. Workaround: use the `bash -c` wrapper (Pattern B above) to export env vars in the shell instead.

### Diagnostic env vars leak into ALL Ray actors via `RUNTIME_ENV_JSON`

Anything you put in `RUNTIME_ENV_JSON.env_vars` (in `scripts/entrypoint/ray-job.sh`) is propagated to **every** Ray worker and **every** Ray Serve infrastructure actor — `ProxyActor`, `ServeController`, `DCSCoordinator`, `MetricsService`, `SimpleStorageUnit`, `TransferQueueController`, `Lock`, `HealthStatus`, etc. — not just GPU train workers. This bites diagnostic flags hard:

- A faulthandler / py-spy / NCCL-trace flag intended for the train actor will fire in the no-GPU Serve infra actors too, flooding the controller log and often crashing the Serve controller itself (the symptom looks like the training run died, but the diagnostic killed the controller).
- **`CUDA_VISIBLE_DEVICES` is NOT a usable GPU-only gate** in `worker_process_setup_hook`. Ray sets `CUDA_VISIBLE_DEVICES=""` on non-GPU actors (you can confirm via Ray's own FutureWarning), so a check like `if cvd: enable_diag()` still passes on `ProxyActor`. Don't use it.

Correct pattern for opt-in diagnostics that must run only in train actors:

1. Keep the env var as the activation gate in `RUNTIME_ENV_JSON` (e.g. `RELAX_FAULTHANDLER_INTERVAL_SEC=15`).
2. Read the env var and call `faulthandler.enable()` / `dump_traceback_later()` from inside `MegatronTrainRayActor.init()` (or a sibling actor's `__init__`) — **never** from `relax/utils/logging_utils.install_asyncio_noise_filter`, which is the Ray `worker_process_setup_hook` and runs in every actor.
3. Reference: `relax/backends/megatron/actor.py::_install_faulthandler_periodic_dump` is the canonical example.

### MASTER_ADDR defaults to 127.0.0.1

`relax/utils/utils.py` reads `os.environ.get("MASTER_ADDR", "127.0.0.1")` to set `SLIME_HOST_IP` for rollout workers. If `MASTER_ADDR` is not set, the rollout engine gets `127.0.0.1` which is unreachable from other workers. This is auto-detected by `scripts/entrypoint/ray-job.sh` via `ray list nodes`.

______________________________________________________________________

## References

### GenRM Validation Command

Use the dedicated GenRM training script:

```bash
RAY_NO_WAIT=1 WORKING_DIR="./" RAY_ADDRESS=x.x.x.x:6379 MODEL_DIR=/path/to/workspace \
    bash -x scripts/entrypoint/ray-job.sh scripts/training/genrm/run-qwen3-4B-8xgpu-genrm.sh
```

To adjust GenRM parameters, edit `scripts/training/genrm/run-qwen3-4B-8xgpu-genrm.sh` directly.
