# Keye-VL 2.0 RL

[中文](./README.md)

These recipes train `Kwai-Keye/Keye-VL-2.0-30B-A3B` with Megatron/Bridge and SGLang **0.5.21**, sharing eight GPUs between training and rollout. The MoE language backbone has approximately 30B total / 3B active parameters, 48 layers, 128 experts with top-8 routing, and a top-2048 sparse attention indexer. Image training adds the native Keye vision tower, projector and three-axis mRoPE. The adapter lives in [relax/models/keye_vl2](../../../relax/models/keye_vl2).

Use a compatible Relax training environment with Megatron/Bridge, Transformer Engine and `fast_hadamard_transform`; matching the SGLang version alone is insufficient.

## Recipes and limits

| Script under `scripts/`                  | Task                                                       | Training parallelism         | Prompts × samples |
| ---------------------------------------- | ---------------------------------------------------------- | ---------------------------- | ----------------- |
| `run-keye-vl2-30B-A3B-8xgpu-math.sh`     | DAPO Math, AIME-2024 evaluation, checkpoint every 20 steps | TP4 / SP / PP1 / CP1 / EP8   | 16 × 8            |
| `run-keye-vl2-30B-A3B-8xgpu-openr1mm.sh` | OpenR1-MM image RL, trainable vision and projector         | TP1 / PP1 / CP2 zigzag / EP8 | 4 × 2             |
| `keye-vl2-30B-A3B.sh`                    | Qwen3-MoE dimensions with Keye's RoPE base                 | —                            | —                 |

Text training uses TP4 with sequence parallelism; image training still requires TP1/PP1. Dynamic CP, contiguous CP, video and audio are unsupported. Images are embedded before per-document zigzag partitioning; K/V and indexer K are gathered across CP ranks. Text rollout defaults to four TP2 SGLang engines; `SGLANG_TP` controls GPUs per engine. The image recipe retains one TP8 engine. Settings live in the shell scripts.

## Prepare assets

Run commands from the repository root. `MODEL_DIR` and `DATA_DIR` are parent directories; `SAVE_DIR` is the output root.

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/data
export SAVE_DIR=/path/to/runs

hf download Kwai-Keye/Keye-VL-2.0-30B-A3B \
  --local-dir "${MODEL_DIR}/Keye-VL-2.0-30B-A3B"
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "${DATA_DIR}/dapo-math-17k"
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir "${DATA_DIR}/aime-2024"
python scripts/tools/process_aime.py \
  --input "${DATA_DIR}/aime-2024/aime-2024.jsonl"

hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir "${DATA_DIR}/multimodal-open-r1-8k-verified"
python scripts/tools/process_openr1.py \
  --input-dir "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet" \
  --output-dir "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
```

Keep all model weights, tokenizer/processor assets and custom Python files. The converted image dataset contains `prompt`, `image` and `label`. Skip preparation when these assets already exist.

## Quick start

Direct execution sources `scripts/entrypoint/local.sh`, which cleans existing local Ray/Python training processes. Use an exclusive eight-GPU node.

```bash
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=8 N_SAMPLES=2 GLOBAL_BATCH_SIZE=16 RESPONSE_LEN=512 \
  EXP_NAME=keye-dapo-smoke \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh

NUM_ROLLOUT=200 EXP_NAME=keye-dapo-200 \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh

NUM_ROLLOUT=2 RESPONSE_LEN=512 SAVE_INTERVAL=1 EXP_NAME=keye-openr1mm-smoke \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-openr1mm.sh
```

Each rollout samples `16 × 8 = 128` responses with a default global batch of 128 (overridable via `GLOBAL_BATCH_SIZE`), giving one optimizer update per rollout. Other training-effect settings match `scripts/training/text/run-qwen3-4B-8xgpu.sh`. Responses are capped at 8192 tokens; dynamic filtering and extra prompt/context length limits are disabled. AIME evaluation runs every 20 rollouts with 8 samples per question, up to 16384 response tokens and top-p 0.7. The recipe uses GRPO group advantages, asymmetric clipping, TIS, `low_var_kl` KL loss with coefficient 0 and constant learning rate `1e-6`. Keye retains its performance settings of 32768 training tokens per GPU and CPU optimizer offload. Text decode uses fused RoPE, a paged indexer preserving BF16 rounding, and SGLang’s native paged GQA kernel. It reads only the selected top-2048 KV slots without expanding KV heads. Decode uses CUDA graphs; prefill remains eager. Overlap scheduling and radix cache remain disabled. The image recipe currently retains eager scheduling.

Training disables TE RoPE fusion to preserve the same BF16 rounding as rollout. Both MoE routers compute FP64 dot products followed by FP32 logits and a fixed top-k order. RMSNorm uses a fixed row reduction, and residual addition rounds to BF16 first. MoE activation, routing weights and expert merging preserve the training rounding order. Training and rollout share the sparse GQA forward; backward recomputes standard masked SDPA gradients. Router replay is not required. For TP>1 rollout, attention output and expert down weights remain complete; inputs are gathered before the full-K contraction to avoid rounding partial outputs to BF16. QKV and gate/up weights remain TP-sharded.

Training attention truncates future KV after the last position in each query chunk, with 1024 queries per chunk. The FP32 teacher retains GQA groups to avoid repeated KV expansion and conversion. Indexer auxiliary KL, full-layer recomputation and CPU optimizer offload remain enabled. CP chunks split at zigzag half boundaries, and causality follows token order rather than compressed image coordinates.

The text recipe defaults to `SAVE_CHECKPOINT=1`: save a complete checkpoint every `SAVE_INTERVAL` rollouts (default 20). With one optimizer update per rollout, this is every 20 steps. Retain the latest two checkpoints and load/save at the stable `${SAVE_DIR}/${EXP_NAME}`. Keep both variables unchanged to resume; change `EXP_NAME` for a new experiment. Set `SAVE_CHECKPOINT=0` to disable saving; a restart then initializes from HF and loses prior optimizer steps.

The image recipe defaults to `SAVE_CHECKPOINT=1`, 4096 response tokens and 200 rollouts. It saves complete checkpoints every 20 rollouts, retains two, and resumes from the same stable `${SAVE_DIR}/${EXP_NAME}` path. Keep both variables unchanged to resume; change `EXP_NAME` for a new experiment. Encoded vision features are not cached across requests, so updated vision weights are used.

A 512-token smoke run checks generation, rewards, backward and weight transfer. Truncation can produce all-zero group rewards; this does not demonstrate RL quality. Use the longer response budget and inspect rewards, within-group variation and truncation during training.

## Existing Ray cluster and logs

Use the actual Ray Jobs HTTP endpoint:

```bash
export RAY_ADDRESS=http://127.0.0.1:8265
RELAX_SKIP_CLUSTER_CLEANUP=1 WORKING_DIR=./ \
  bash scripts/entrypoint/ray-job.sh \
  examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh
```

Submission blocks and follows logs by default. Set `RAY_NO_WAIT=1` explicitly for background submission, then run `ray job logs --address="${RAY_ADDRESS}" <job-id>`. Submission logs are under this example's `scripts/log/`; with no-wait they contain submission output only.

Both recipes enable ClearML and metrics service using existing runtime credentials. `PROJECT_NAME` selects the project and `EXP_NAME` supplies the task-name prefix. Each launch creates a new ClearML task and prints its link.
