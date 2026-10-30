# Qwen3.8 Flash Next SFT on 40 H20 GPUs

[中文](./README.h20.zh-CN.md)

This recipe runs multimodal SFT on five nodes with eight H20 GPUs per node.
It uses the full 48-layer model, one MTP layer, TP4/PP5/DP2/EP4, and the
pipeline layout `Et*8|t*11|t*11|t*12|t*6mL` (8/11/11/12/6 layers plus MTP).
By default, packing uses a 65,536-token budget and global batch size 1,024. The vision
model, vision projection, PLE, and attention indexer are frozen; PLE is
offloaded to CPU. The frozen vision components still participate in multimodal
forward computation. Training parameters stay on GPU, with full uniform
activation recomputation.

## Prerequisites

- Five dedicated nodes with eight H20 GPUs each, running the same Relax checkout
  and compatible Megatron-LM, Megatron Bridge, DeepEP, Transformer Engine, and
  Qwen3.8-capable Transformers dependencies. The standard image installs
  Megatron-LM at `/root/Megatron-LM`; override `MEGATRON` for another location.
  This recipe does not install packages or patch the training image.
- A complete Hugging Face model directory with weights, configuration,
  tokenizer, and multimodal processor assets.
- JSONL SFT data with `messages`, optional `tools`, and the supported multimodal
  message format. Media paths must resolve on every worker. Chat templating uses
  `preserve_thinking=true` and `add_generation_prompt=false`.
- ClearML credentials configured in the runtime environment.

Model, data, output, and source directories must be accessible at the same paths
on all nodes. Verify model configuration, tokenizer/processor, and training
entrypoint imports in the worker environment before launching.

## Configure

[`launch.sh`](launch.sh) is the shared configuration entrypoint for paths,
experiment names, and common training parameters. Edit its defaults or override
them through environment variables. The three required paths are:

```bash
export MODEL_DIR="/path/to/hf-model"
export DATA_DIR="/path/to/train.jsonl"
export SAVE_DIR="/path/to/run-output"
```

`DATA_DIR` accepts an exact `.jsonl` file or a directory containing `train.jsonl`.
`SAVE_DIR` stores submission logs. Use a distinct output directory for a separate
experiment. No separate `launch.env` file is needed.

| Variable             | Default                  | Purpose                                                                            |
| -------------------- | ------------------------ | ---------------------------------------------------------------------------------- |
| `RELAX_DIR`          | Located from `launch.sh` | Select the complete Relax checkout, including this recipe.                         |
| `EXP_NAME`           | `qwen38-qsa-40xh20`      | Stable experiment name; launch time is appended to the ClearML task name.          |
| `PROJECT_NAME`       | `Relax/sft/qwen38-qsa`   | ClearML project.                                                                   |
| `NUM_STEPS`          | `10`                     | Optimizer updates requested for the run.                                           |
| `GBS`                | `1024`                   | Global batch size; keep divisible by the fixed dense DP size of 2.                 |
| `SEQ_LENGTH`         | `65536`                  | Sequence length and maximum input context length.                                  |
| `MAX_TOKENS_PER_GPU` | `65536`                  | Dynamic batching/packing token budget, configured separately from sequence length. |
| `LR` / `MIN_LR`      | `2e-5` / `2e-6`          | Peak and minimum learning rates.                                                   |
| `LR_DECAY_ITERS`     | `2023`                   | Cosine learning-rate schedule length; not automatically changed with `NUM_STEPS`.  |
| `LR_WARMUP_FRACTION` | `0.05`                   | Fraction of the learning-rate schedule used for warmup.                            |
| `SEED`               | `1234`                   | Both training and data/rollout random seeds.                                       |

For example, after setting the three required paths:

```bash
GBS=512 NUM_STEPS=10 EXP_NAME=qwen38-h20-gbs512 \
bash examples/models/qwen3.8-flash-next/launch.sh
```

Both `launch.sh` and direct invocation of the training script read these defaults
from `launch.sh`. `RELAX_DIR` may point to another complete checkout at the same
path on all nodes; the selected checkout must contain this recipe. Its training
script, model configuration, and worker import path are used together. For an
existing Ray cluster, the standard Ray Jobs entrypoint also comes from that
checkout. `MEGATRON` remains configured in the training script.

Changing batch size, sequence length, or packing budget requires checking GPU
memory and training behavior again. The fixed 40-GPU parallel layout and the
freezing/offload settings remain in the training script.

## Launch

For a multi-node TorchJob, run the following command on **every node** from the
repository root. The standard entrypoint starts Ray and submits training on the
head node; the other nodes join as workers.

```bash
bash scripts/entrypoint/spmd-multinode.sh examples/models/qwen3.8-flash-next/launch.sh
```

The job environment must provide `MASTER_ADDR` (the head Pod's hostname,
matching its `POD_NAME`), `POD_NAME`, `HOST_IP`, and `WORLD_SIZE=5`. Each node uses
eight GPUs by default. Use this entrypoint on dedicated job nodes; it cleans
stale processes and starts a new Ray cluster.

Set these node variables in the job environment before starting the SPMD
entrypoint. It reads `WORLD_SIZE` before executing `launch.sh`, so setting
`WORLD_SIZE` inside `launch.sh` would be too late.

To use another checkout for a TorchJob, invoke both `spmd-multinode.sh` and
`launch.sh` from that checkout. `RELAX_DIR` in `launch.sh` cannot switch an SPMD
entrypoint that has already started.

For an **existing Ray cluster**, run only on the Ray head instead:

```bash
bash examples/models/qwen3.8-flash-next/launch.sh
```

This delegates to `scripts/entrypoint/ray-job.sh`, preserves existing jobs by
default, and rejects another active Relax training job before submission.
The Ray Jobs address defaults to `http://127.0.0.1:8265` on the head; override
`RAY_JOBS_ADDRESS` only when using a different endpoint.

Both launch paths use the same training script under `scripts/`. Submission
blocks and follows logs by default, writing
`${SAVE_DIR}/logs/<submission-id>.submit.log`. For asynchronous submission set
`RAY_NO_WAIT=1`; leave it unset or empty for blocking operation (`0` is also
nonempty and enables asynchronous submission).

## Steps, metrics, and checkpoints

The default is a **10-step short run**. Change `NUM_STEPS` for your own run
length. `EXP_NAME` and `PROJECT_NAME` optionally override
the script's ClearML experiment and project defaults. Task names include the
experiment name and launch time; each submission creates a separate task.

Each launch initializes from the HF model. This recipe does **not** save or
resume training checkpoints. Additional CLI arguments are forwarded to the
training entrypoint.

## Preview and caches

With the three path variables set, preview arguments and runtime settings
without submitting a job, importing GPU dependencies, or creating output files:

```bash
DRY_RUN=1 bash examples/models/qwen3.8-flash-next/launch.sh
```

Compilation caches use node-local defaults under `${TMPDIR:-/tmp}/relax-qwen38`.
The standard entrypoint's kernel-cache configuration still applies. Override
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, or `CUDA_CACHE_PATH` only if your
environment needs a different local cache location.
