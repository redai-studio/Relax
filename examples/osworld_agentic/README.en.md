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
