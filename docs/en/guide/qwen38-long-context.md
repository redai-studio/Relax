# Qwen3.8 512K / 1M training

Two 32-GPU recipes cover original-length, doubled-sample, and full-length capacity scenarios. They require a CUDA 13 Blackwell image with HybridEP, the FlashQLA small-head fix, and the PLE local-computation patch. Scripts alone do not add these capabilities to an older checkout.

## Configuration and data

Shared settings: 4 nodes × 8 GPUs, TP1/PP2/EP8/ETP1, 48 layers split 24+24, MTP1, GBS32/MBS1, and seed9714. Vision, projection, PLE, and indexer parameters are frozen; full uniform recomputation uses two layers per chunk, with all optimizer state offloaded.

| Scenario | Data transformation | CP / DP | Sequence limit | Ordinary packing budget per GPU | PLE transport |
| --- | --- | --- | --- | --- | --- |
| `real512` | Original samples of length ≤512K | 8 / 2 | 512K | 64K | `full` |
| `filled512` | Repeat each source sample and loss mask to 512K | 8 / 2 | 512K | 64K | `full` |
| `real1m` | Double each source sample and loss mask | 16 / 1 | 1M | 32K | `adaptive` |
| `filled1m` | Repeat each source sample and loss mask to 1M | 16 / 1 | 1M | 32K | `adaptive` |

K=1024 and M=1024². Dataset classes transform tokens and unshifted supervision masks after native SFT rendering, exclude source samples longer than 512K, and leave label construction to the framework. Only text causal-LM SFT is supported; the first token of repeated samples must be unsupervised. The 1M samples are artificially extended, and filled inputs measure capacity/performance rather than natural long-context quality.

The 1M recipe's 32K budget is **not a hard per-GPU input limit**. Oversized samples occupy their own microbatch; a 1M singleton at CP16 still has 64K local tokens. `full` retains PLE local computation; see the [PLE usage guide](./qwen38-qsa.md#local-computation-for-frozen-ple).

## Run

Prepare an exclusive 32-GPU Ray cluster and model, data, and output paths accessible on every node. The entrypoint does not clean up existing jobs or guarantee GPU sharing with other training. Use the new image's Megatron and HybridEP; the image supplies `RELAX_HYBRID_EP_COMMIT`.

```bash
export MODEL_DIR=/shared/models/Qwen3.8-Flash-Next
export DATA_DIR=/shared/data/train.jsonl
export SAVE_DIR=/shared/output/qwen38
export MEGATRON=/root/Megatron-LM
# Optional: otherwise Ray resolves the existing cluster's Jobs address.
export RAY_JOBS_ADDRESS=http://127.0.0.1:8265

# Inspect actual argv and worker environment, without submission or output writes.
DRY_RUN=1 bash scripts/training/text/run-qwen38-512k-cp.sh

# Default ceiling: 200 steps. Run scenarios separately on the same GPUs.
bash scripts/training/text/run-qwen38-512k-cp.sh
DATA_MODE=filled bash scripts/training/text/run-qwen38-512k-cp.sh
bash scripts/training/text/run-qwen38-1m-cp.sh
DATA_MODE=filled bash scripts/training/text/run-qwen38-1m-cp.sh
```

Alternatively, select a scenario through `SCENARIO=real512 bash scripts/training/text/run-qwen38-long-context.sh`.

Submission is asynchronous by default. Scenarios have separate default `EXP_NAME` values; each `SUBMISSION_ID` is unique. Submission logs and effective arguments are saved under `$SAVE_DIR/$EXP_NAME/logs/`. Common overrides include `NUM_STEPS`, `NUM_EPOCH`, `EXP_NAME`, `SAVE_INTERVAL`, `MAX_TOKENS_PER_GPU`, and `QWEN38_PLE_INPUT_HALO_TRANSPORT`. Set `USE_CLEARML=1` to enable ClearML with your credentials. Checkpoint paths must use `SAVE_DIR`, `EXP_NAME`, and `SAVE_INTERVAL`; extra `--load` / `--save` arguments are rejected. Additional training arguments can be appended; changed configurations require revalidation.

Checkpoint saving is disabled by default. Provide at least 6400 accepted samples for 200 steps, or enough data epochs. Do not count incomplete final batches as GBS32 validation.

## Save and resume

```bash
# Save iter_0000099 / iter_0000199 after 100 / 200 updates.
EXP_NAME=real512-save SAVE_INTERVAL=100 \
  bash scripts/training/text/run-qwen38-512k-cp.sh

# Resume after update 100, retaining the total ceiling of 200.
EXP_NAME=real512-save RESUME=1 SAVE_INTERVAL=100 \
  bash scripts/training/text/run-qwen38-512k-cp.sh --ckpt-step 99
```

A fresh run rejects an existing checkpoint instead of resuming implicitly. Resume requires `RESUME=1`, the original `EXP_NAME`, the checkpoint tracker, and the generated `recipe-config.json`. The entrypoint compares scenario, data/model paths, parallel configuration, packing budget, GBS, seed, learning rates, and extra arguments to prevent cross-scenario loading. Do not edit the record to bypass validation.

Resume automatically adds `--use-checkpoint-opt_param-scheduler`, restoring the saved LR/WD schedule. `NUM_STEPS` controls the stopping point, not a new restored schedule. Model, optimizer, and data position use native checkpoint recovery. Without `SAVE_INTERVAL`, data position is not saved separately; this recipe has no separate data-state directory.

## Validation scope

These are historical 32-GPU runs with the same model/data design, measured over training time rather than total wall time. Both filled scenarios were deliberately stopped after 40 complete steps; their status is STOPPED, not a completed 200-step run.

| Scenario | Validated steps | Observed TPS/card | Peak reserved | Minimum sampled free memory |
| --- | --- | --- | --- | --- |
| real512 | 200 + 10 resumed steps | 2572.16 (excluding the copy window) | 221.05 GiB | 21.81 GiB |
| filled512 | 40-step complete prefix | 3316.92 | 231.19 GiB | 12.35 GiB |
| real1m | 100 | 2501.36 | 195.26 GiB | 46.21 GiB |
| filled1m | 40-step complete prefix | 2696.45 | 216.39 GiB | 25.85 GiB |

Loss, MTP loss, and grad norm were finite for all complete steps. The 10 resumed real512 steps matched continuous training exactly for loss, MTP loss, grad norm, LR, and data position. Save/resume was not independently validated for the other scenarios.

**These GPU runs and the image build were not repeated on this branch.** Historical runs included first-step-only cache release and separate data-position recording, neither delivered here. Published recipes retain main's default cache-release behavior, so historical throughput and reserved peaks are not guaranteed. CPU checks cover actual parser acceptance for all scenarios, tokens and supervision masks, scenario isolation, and resume arguments; GPU capacity still needs validation in the target environment.
