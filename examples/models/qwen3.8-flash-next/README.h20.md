# Qwen3.8 Flash Next SFT on 40 H20 GPUs

[中文](./README.h20.zh-CN.md)

This recipe runs multimodal SFT on five nodes with eight H20 GPUs each:
full 48 layers + MTP1, TP4/PP5/DP2/EP4, layout `Et*8|t*11|t*11|t*12|t*6mL`,
64K packing and GBS1024. Vision/projection/PLE/indexer are frozen; frozen vision
still participates in forward computation. The PLE table is on CPU, with full
uniform activation recomputation and training parameters on GPU.

## Configure and launch

Use a compatible Relax image on all nodes, with Qwen3.8 model/processor support
and ClearML credentials. The script discovers Megatron-LM from the Python runtime;
set `MEGATRON` to select another checkout. Model, data, output and source paths
must be accessible at the same locations on every node.

```bash
export MODEL_DIR="/path/to/hf-model"
export DATA_DIR="/path/to/train.jsonl"
export SAVE_DIR="/path/to/run-output"
```

`DATA_DIR` accepts a JSONL file or a directory containing `train.jsonl`, with
`messages`, optional `tools`, and supported multimodal messages. Media paths must
resolve on every worker. The recipe does not install dependencies or patch images.

For a dedicated multi-node TorchJob, run on **every node** from the selected checkout:

```bash
bash scripts/entrypoint/spmd-multinode.sh examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

The job environment must provide `MASTER_ADDR` (head hostname matching `POD_NAME`),
`POD_NAME`, `HOST_IP` and `WORLD_SIZE=5`; each node defaults to eight GPUs.
This standard entrypoint cleans stale processes and starts Ray.
For an existing Ray cluster, run only on the head:

```bash
bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

The script sources the standard Ray Jobs setup once and rejects another active
Relax training job before setup. Submission blocks by default; set `RAY_NO_WAIT=1`
for asynchronous submission. Logs are saved under `${SAVE_DIR}/logs/`.

## Settings and preview

Settings live in the single training script. Optional environment overrides are
`NUM_STEPS` (10), `GBS` (1024), `SEQ_LENGTH` / `MAX_TOKENS_PER_GPU` (65536),
`LR` / `MIN_LR` (2e-5 / 2e-6), `LR_DECAY_ITERS` (2023), `LR_WARMUP_FRACTION` (0.05),
`SEED` (1234), `EXP_NAME` and `PROJECT_NAME`. Keep GBS divisible by dense DP2.
Additional CLI arguments are forwarded. Each launch starts from HF; this recipe
does **not** save or resume training checkpoints.

`RELAX_DIR` can select another complete checkout containing this recipe. For a
TorchJob, invoke both the SPMD entrypoint and recipe from that checkout.
`RAY_DASHBOARD_ADDRESS` selects the Ray Jobs HTTP endpoint (default head-local
`http://127.0.0.1:8265`); the old `RAY_JOBS_ADDRESS` name remains a fallback.

With the three required paths set, preview without Ray/GPU setup or output writes:

```bash
DRY_RUN=1 bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
```

Preview defaults `MEGATRON` to `/root/Megatron-LM` without importing it; override
it for another image layout. Standard entrypoint compiler-cache settings apply.
