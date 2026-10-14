# OSWorld + Nex agentic training

[中文](README.md)

This recipe trains Qwen3-VL-8B-Thinking with GRPO, Relax resident agentic rollout, and Nex desktop sandboxes. It requires access to a Nex deployment and a working OSWorld desktop template; cloning the repository alone is insufficient.

## Files and configuration

- `env.example.sh`: public deployment template without real endpoints or credentials.
- `env.local.sh`: private copy of the template, ignored by Git; never commit it.
- `scripts/run-qwen3vl-8B-osworld-8xgpu.sh`: authoritative training configuration and submission script.
- `run_qwen3vl_8B_osworld_outcome.sh`: compatibility wrapper forwarding to that script.
- `run_agent_app.sh`: per-session entry called by Relax, not a standalone training launcher.
- `app/`: desktop environment, model adapter, rewards and metrics; `app/osworld_config.yaml` controls interaction settings.
- `scripts/prepare_data.py` and `train_tasks.txt`: dataset conversion and the fixed 12-task manifest.
- `scripts/analyze_run.py` and `plot_training.py`: offline analysis. `reward_osworld.py` is a compatibility scorer; the training script uses agent metadata and a custom advantage callback.

The training and agent entry scripts read `env.local.sh` directly. No additional configuration file is needed. Launch-time model/data/output roots take precedence.

## Prerequisites

1. Prepare the training image, Megatron, SGLang, Ray and model weights using the [Relax installation guide](../../docs/en/guide/installation.md). This configuration needs eight GPUs on one node with sufficient memory; requirements vary with trajectories. The script fixes `MEGATRON=/root/Megatron-LM`; edit that line for another image. All workers need identical code, dependencies and accessible paths.

2. Install the external OSWorld fork at the revision inspected for this recipe:

   ```bash
   git clone https://github.com/fxsc03/OSWorld-main-nex.git /data/OSWorld-main-nex
   git -C /data/OSWorld-main-nex checkout 50b4a85778cfc13ce00e3bf63a3cd08bd260fb96
   python3 -m venv /data/osworld-venv
   /data/osworld-venv/bin/python -m pip install -r /data/OSWorld-main-nex/requirements.txt qwen-agent
   ```

   Keep the agent environment separate to avoid replacing training dependencies. Host installation does not install guest dependencies: the Nex template needs desktop applications, LibreOffice/UNO, MCP tools and task assets. Follow the fork's Nex deployment documentation to validate networking, task reset and evaluators.

3. Create private configuration once, without overwriting an existing file:

   ```bash
   cd examples/osworld_agentic
   cp -n env.example.sh env.local.sh
   chmod 600 env.local.sh
   ```

   Fill in `OSWORLD_REPO`, `OSWORLD_AGENT_PYTHON`, `NEX_API_BASE`, `NEX_WORKSPACE_ID`, `NEX_API_KEY`, and `NEX_TEMPLATE` (or `NEX_TEMPLATE_ID`). `MCP_SRC_ROOT` is the repository root containing `mcp/`. The external Nex client contains deployment-specific defaults, so the agent entry requires explicit Nex settings.

   The private file must be readable at the same path on all agent workers. Ignored files are not uploaded with Ray code packages; provision them securely or configure worker environments beforehand. Do not put credentials in command lines or runtime-env JSON, which may appear in Jobs metadata or logs.

4. ClearML is enabled by default. Configure your own deployment on training nodes, for example with `clearml-init`. The template contains no account credentials or internal endpoints.

## Launch

After configuring `env.local.sh`, run from the Relax root:

```bash
MODEL_DIR=/data/models \
DATA_DIR=/data/rl_data \
SAVE_DIR=/data/outputs \
bash examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh
```

- Weights: `${MODEL_DIR}/Qwen3-VL-8B-Thinking`.
- Dataset: `${DATA_DIR}/osworld/train.parquet`. If absent, the script converts tasks under the external repository's `evaluation_examples/examples` using the fixed manifest. Unknown task IDs or an empty manifest fail explicitly. Existing parquet is reused; regenerate it deliberately after changing the manifest.
- Generated train/eval files use the same tasks, so they are not an independent validation split. Periodic evaluation is not configured by default.
- Outputs: `${SAVE_DIR}/qwen3vl-8b-osworld-8xgpu/`.

## Current parameters

| Setting                                  | Value                                                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| Initial model                            | Qwen3-VL-8B-Thinking                                                                         |
| Algorithm                                | GRPO, custom trajectory advantages, TIS                                                      |
| Sampling                                 | 12 groups × 8 trajectories = 96 per rollout                                                  |
| Updates                                  | 2 per rollout; 200 rollouts                                                                  |
| Learning rate                            | 5e-6, constant                                                                               |
| KL                                       | k3, coefficient 0.001; reference updated every rollout                                       |
| PPO clip                                 | Lower/upper both 0.2                                                                         |
| Training parallelism                     | TP4 / PP1 / CP1; DP2 on eight GPUs                                                           |
| Per-GPU training / log-prob token budget | Both 49152; not the model context limit                                                      |
| Rollout context / response limit         | 16384 / 8192 tokens                                                                          |
| SGLang                                   | 2 GPUs per engine, static memory fraction 0.7, consistent hashing                            |
| Memory                                   | Vision trunk frozen; selective `core_attn mlp` recomputation; optimizer CPU offload disabled |
| Sandboxes                                | Prelaunch enabled; active and prelaunch share 12 group permits, up to 96 sessions            |
| Dynamic filter                           | Disabled                                                                                     |
| Agent                                    | Up to 30 turns, 2-image history, session timeout 21600 seconds                               |
| Checkpoints                              | Disabled; each launch starts from initial weights without optimizer resume                   |

Actor and rollout are colocated on the same eight physical GPUs. Session concurrency does not guarantee an identical Nex instance count: retries and delayed server-side cleanup still require quota monitoring.

## Rewards and trajectories

Per-trajectory shaped reward is `(success ? 1 : -1) - 0.05 × T/30 - 0.2 × I(T>=30)`. Success means evaluator outcome ≥ 0.5; T is the trajectory turn count. Advantages are normalized over eight logical trajectories per group, then assigned to their exported training records, rather than grouping eight exported records together.

96 trajectories may produce many more than 96 training records. To match the handoff implementation, `rollout/raw_reward` is the **export-record-count-weighted shaped reward**, not trajectory success rate. Inspect success, turn counts and errors alongside it. All-success/all-failure groups have zero policy advantages, but the enabled KL term can still produce gradients.

The 2-image history limit applies to each model context, not the total screenshots in a trajectory. Optional screenshot dumps cannot be counted as a direct measure of interaction turns.

## Logs and privacy

- Submission: `examples/osworld_agentic/log/<experiment>-<timestamp>.log`.
- Agent: `examples/osworld_agentic/log/agent/<run-tag>/`.
- Training exports: `${SAVE_DIR}/<experiment>/rollout_result/<run-tag>/train/`.
- Trajectory rewards: `reward_dump.jsonl` in the same run directory.
- TensorBoard: `${SAVE_DIR}/<experiment>/tensorboard_log/<run-tag>/`. ClearML experiment names include the launch timestamp.

```bash
python examples/osworld_agentic/scripts/analyze_run.py \
  --dump /data/outputs/qwen3vl-8b-osworld-8xgpu/rollout_result/<run-tag>/reward_dump.jsonl \
  --groups-per-rollout 12 --samples-per-group 8 --md
```

The analyzer's grouping assumes the current fixed batches without dynamic filtering. Recheck its assumptions if the sampling flow changes.

Git ignores private configuration and common log, screenshot and cache directories; ignore rules are not redaction. Desktop screenshots, task content, agent logs and JSON exports can contain sensitive data and need separate review before publication. Never force-add private files. Configuration loading disables shell tracing, but do not print the private environment elsewhere.

This recipe review does not certify the external OSWorld fork, sandbox image or historical Git commits as privacy-audited. In particular, the external fork contains deployment-specific default endpoints and needs separate review before redistribution.

## CHANGELOG

Experiments are listed in ascending launch-time order (UTC+8). Heading times correspond to experiment-name timestamps. Updates mean optimizer updates, indexed from zero. Results come from retained submission arguments and logs; script edits are not validated experiments.

### 2026-10-11 18:33 — Initial integration experiment

- **Configuration:** Qwen3-VL-8B-Thinking, 12 tasks × 8 trajectories, 10 rollouts with 3 updates each; LR `1e-6` / cosine, KL coefficient `0.02`, 32K training budget and context, full/uniform recomputation, optimizer CPU offload enabled.
- **Result:** 9 updates logged before the SGLang engine refused connections during cache flushing; the job failed before completion. Ten rollouts allowed at most 30 updates, insufficient for the subsequently requested long run.
- **Next:** Increase to 200 rollouts and adjust offload/recomputation. Agents own logical sessions and export model-visible contexts; 96 trajectories can produce over 1400 records without creating that many sandboxes.

### 2026-10-11 22:58 — 200 rollouts, disable offload and reduce recomputation

- **Change:** Rollouts 10 → 200; disable optimizer CPU offload; full → selective recomputation. Keep 32K training budget and context.
- **Result:** CUDA OOM on the first training attempt; no completed optimizer update.
- **Next:** Reduce the token budget and align length constraints.

### 2026-10-12 01:15 — Reduce training budget to 16K

- **Change:** `max_tokens_per_gpu: 32768 → 16384`; context remains 32768.
- **Result:** Argument validation failed because training budget × CP was smaller than rollout context. Training did not start.

### 2026-10-12 01:16 — Reduce rollout context to 16K

- **Change:** `rollout_max_context_len: 32768 → 16384`.
- **Result:** Argument validation failed again: the 16384 prompt limit equaled context length, leaving no generation token. Training did not start.

### 2026-10-12 01:21 — Fix prompt/context constraints

- **Change:** Prompt limit reduced to 16383; keep 16K context and training budget.
- **Result:** One completed update, then another OOM. Not yet stable.
- **Next:** Add MLP recomputation to reduce peak training memory.

### 2026-10-12 02:01 — Add MLP recomputation

- **Change:** Explicit selective recomputation of `core_attn mlp`.
- **Result:** Actor initialization failed with an uncorrectable ECC error, before training. This was not a loss/reward failure.

### 2026-10-12 02:07 — Resubmit after the ECC error

- **Change:** Retain 16K budget/context, selective `core_attn mlp` recomputation and disabled optimizer CPU offload.
- **Result:** Completed 57 updates (19 rollouts), maximum grad norm 2.854678, then deliberately stopped to correct rewards; did not finish 200 rollouts.
- **Issue:** Reward semantics were not yet fully aligned with the handoff, so this curve is not a reproduction of that experiment.

### 2026-10-12 15:42 — Restore handoff reward semantics

- **Change:** Restore trajectory outcome shaping, eight-trajectory advantage normalization and export-count-weighted `rollout/raw_reward`. Retain LR `1e-6` / cosine, three updates per rollout and 16K training budget.
- **Result:** Completed 21 updates (7 rollouts), maximum grad norm 4.100425, then deliberately stopped to align learning rate and training settings.
- **Definition:** Shaped reward is `(success ? 1 : -1) - 0.05 × T/30 - 0.2 × I(T>=30)`. Session-deduplicated success rate is separate from raw reward.

### 2026-10-12 21:02 — LR 5e-6 / constant, 32K, freeze vision

- **Change:** LR `1e-6 → 5e-6`, cosine → constant, training budget 16K → 32K, freeze vision trunk. Retain three updates per rollout, KL `0.02`, reference updates every rollout and selective recomputation. Sandbox prelaunch remains disabled.
- **Result:** Completed 147 updates (49 rollouts), then deliberately stopped for the prelaunch experiment. Four `grad_norm > 10` events occurred at updates 11, 35, 71 and 110, all the third update within their rollout. Maximum grad norm: 491186.094; maximum unscaled `train/kl_loss`: 14710.166.
- **Finding:** Large spikes correlated with k3 KL outliers after repeated updates. Grad norm is logged before clipping; `clip_grad=1.0` does not require it to be below one.

### 2026-10-13 22:15 — Enable sandbox prelaunch

- **Change:** Enable `agentic-prelaunch`; active and prelaunch share 12 group permits, up to 96 sessions. Retain 32K, three updates per rollout and KL `0.02`.
- **Result:** Logs contain 14 updates before stopping. The third update again spiked: grad norm 189.610, unscaled KL loss 2.846, weighted KL approximately 96.5% of total loss. Prelaunch did not eliminate this training stability issue.
- **Next:** Reduce KL coefficient to `0.001` and updates per rollout to two. Without token-level or separate PG/KL gradients, no specific offending-token cause was established. No isolated prelaunch speedup was measured with other variables controlled.

### 2026-10-14 00:58 — KL 0.001, two updates, 24K, dynamic filter enabled

- **Change:** KL `0.02 → 0.001`; updates per rollout `3 → 2`, logical trajectories per update `32 → 48`; training budget `32K → 24K`; enable trajectory-outcome dynamic filtering. Retain LR `5e-6`, prelaunch, frozen vision and reference updates every rollout.
- **Result:** Two updates completed, maximum grad norm 1.159800, no spike above ten. Deliberately stopped to disable filtering as requested. Too short to establish long-term stability or filtering benefit.

### 2026-10-14 01:48 — Disable dynamic filter and continue training

- **Change:** Only disable dynamic filtering; retain KL `0.001`, two updates per rollout, 24K, LR `5e-6` / constant, prelaunch and frozen vision.
- **Snapshot at 2026-10-14 13:20:39:** 27 rollouts and 54 updates (0–53) completed. Grad norm minimum/mean/maximum: **0.561670 / 0.954595 / 1.337857**; **zero** events above ten. Maximum unscaled KL loss: **0.003474**. The previous large spikes did not recur in this window.
- **Curves:** Rollout 0 → 26 raw reward `-0.403194 → 0.144336`, session success `48.96% → 80.21%`; latest actor training MFU approximately `74.39%`. Two endpoints do not establish monotonic improvement or independent evaluation. Actor MFU is not full-cycle utilization.
- **Limits:** KL coefficient, update count and token budget changed together, so stability cannot be attributed solely to KL coefficient. The experiment was still running, not finished with 200 rollouts.

### 2026-10-14 — Code and documentation cleanup during the active run, no restart

- **Script-only change, not a live experiment:** Set training/log-prob budgets to 48K; the active run remains at 24K. Follow `create-training-script` for arrays, direct launch and automatic entrypoint detection; fix LR `5e-6`, 200 rollouts and two updates in the arguments, removing the indirect `OSWORLD_UPDATES_PER_ROLLOUT` variable. Retain blocking submission and disabled checkpoint writes.
- **Metrics/dumps:** Distinguish sessions and exports; add session/task/turn/export and parse-attempt identifiers; fix raw reward plot labels/range; close metrics adapters on normal completion. These changes are not a new controlled experiment.
- **Time audit at 12:15:20:** The first 24 step times totaled 10h 21m 18.736s, averaging **1553.281s per rollout**. Adding 3m 57.812s before timing began and 1m 27.839s pending the next report reconciled with Job runtime of 10h 26m 44.387s. Each point covers a rollout with two updates; cumulative timing for rollouts 1–23 differed from log intervals by only 0.458s.
- **Public recipe:** Provide `env.example.sh`, directly load ignored `env.local.sh`, remove redundant `env.sh` and its ignore exception. Update bilingual instructions and reject missing/empty task manifests. Keep internal endpoints, credentials and private paths out of documentation.
- **Validation:** All pre-commit hooks passed before commit; relevant tests: 61 passed, one opt-in live Ray Serve test skipped. Submission arguments were verified with isolated stubs; the latest 48K script has not completed real GPU validation. Shared-prefix training, broadcasts and GPU-to-CPU copies were analyzed without implementing new optimization algorithms.
