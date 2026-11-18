# Qwen3.8 512K / 1M training

Two 32-GPU recipes cover original-length, doubled-sample, and full-length capacity scenarios. They require a CUDA 13 Blackwell image with HybridEP, the FlashQLA small-head fix, and the PLE local-computation patch. Scripts alone do not add these capabilities to an older checkout.

## Configuration and data

Shared settings: 4 nodes × 8 GPUs, TP1/PP2/EP8/ETP1, 48 layers split 24+24, MTP1, GBS32/MBS1, and seed9714. Vision, projection, PLE, and indexer parameters are frozen; full uniform recomputation uses two layers per chunk, with all optimizer state offloaded.

| Scenario | Data transformation | CP / DP | Sequence limit | Ordinary packing budget per GPU | PLE transport |
| --- | --- | --- | --- | --- | --- |
| `mixed512` | Original samples of length ≤512K | 8 / 2 | 512K | 64K | `full` |
| `filled512` | Repeat each source sample and loss mask to 512K | 8 / 2 | 512K | 64K | `full` |
| `mixed1m` | Double each source sample and loss mask | 16 / 1 | 1M | 32K | `adaptive` |
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
export RAY_API_SERVER_ADDRESS=http://127.0.0.1:8265

# Inspect actual argv and worker environment, without submission or output writes.
DRY_RUN=1 bash scripts/training/text/run-qwen38-512k-cp.sh

# Default ceiling: 200 steps. Run scenarios separately on the same GPUs.
bash scripts/training/text/run-qwen38-512k-cp.sh
DATA_MODE=filled bash scripts/training/text/run-qwen38-512k-cp.sh
bash scripts/training/text/run-qwen38-1m-cp.sh
DATA_MODE=filled bash scripts/training/text/run-qwen38-1m-cp.sh
```

Alternatively, select a scenario through `SCENARIO=mixed512 bash scripts/training/text/run-qwen38-long-context.sh`.

Submission waits for the job to finish and follows its logs by default; set `RAY_NO_WAIT=1` for asynchronous submission. Scenarios have separate default `EXP_NAME` values; each `SUBMISSION_ID` is unique. Submission logs and effective arguments are saved under `$SAVE_DIR/$EXP_NAME/logs/`. Common overrides include `NUM_STEPS`, `EXP_NAME`, `SAVE_INTERVAL`, `MAX_TOKENS_PER_GPU`, and `QWEN38_PLE_INPUT_HALO_TRANSPORT`. Checkpoint paths must use `SAVE_DIR`, `EXP_NAME`, and `SAVE_INTERVAL`; extra `--load` / `--save` arguments are rejected. Additional training arguments can be appended; changed configurations require revalidation.

Recipes do not explicitly clear the CUDA cache (`--empty-unused-memory-level 0`), including the first step. Checkpoint saving is disabled by default. `NUM_STEPS` controls optimizer updates; the native iterator reads across data epochs as needed. Do not count incomplete final batches as GBS32 validation.

## Save and resume

```bash
# Save iter_0000099 / iter_0000199 after 100 / 200 updates.
EXP_NAME=mixed512-save SAVE_INTERVAL=100 \
  bash scripts/training/text/run-qwen38-512k-cp.sh

# Resume after update 100, retaining the total ceiling of 200.
EXP_NAME=mixed512-save RESUME=1 SAVE_INTERVAL=100 \
  bash scripts/training/text/run-qwen38-512k-cp.sh --ckpt-step 99
```

A fresh run rejects an existing checkpoint instead of resuming implicitly. Resume requires `RESUME=1`, the original `EXP_NAME`, the checkpoint tracker, and the generated `recipe-config.json`. The entrypoint compares scenario, data/model paths, parallel configuration, packing budget, GBS, seed, learning rates, and extra arguments to prevent cross-scenario loading. Do not edit the record to bypass validation.

Resume automatically adds `--use-checkpoint-opt_param-scheduler`, restoring the saved LR/WD schedule. `NUM_STEPS` controls the stopping point, not a new restored schedule. Model, optimizer, and data position use native checkpoint recovery. Without `SAVE_INTERVAL`, data position is not saved separately; this recipe has no separate data-state directory.

## Validation scope

This branch completed 20 steps per scenario on 4 nodes × 8 Blackwell GPUs (approximately 269 GiB visible memory per GPU), without explicit cache clearing at any step. TPS/card and MFU use aggregate training time over steps 5–19. Memory covers the complete 20-step prefix; free memory is sampled through NVML.

| Scenario | TPS/card | MFU | Peak reserved | Minimum sampled free memory |
| --- | ---: | ---: | ---: | ---: |
| mixed512 | 2624 | 5.52% | 219.33 GiB | 24.72 GiB |
| filled512 | 3314 | 7.07% | 229.51 GiB | 14.14 GiB |
| mixed1m | 2496 | 5.28% | 176.11 GiB | 66.93 GiB |
| filled1m | 2701 | 5.87% | 214.85 GiB | 25.29 GiB |

All scenarios had complete GBS 32 batches, finite loss/MTP loss/grad norm, and no reported skipped updates. Per-step comparisons cover loss, MTP loss, grad norm, LR, and token/packing metrics.

mixed512 and mixed1m matched history exactly. For the first 20 steps, filled512 had maximum absolute loss difference 3.59e-6 and relative grad-norm difference 0.41%; filled1m had 3.30e-6 and 0.65%, respectively.

Extending the first filled512 run to 25 steps produced a maximum relative grad-norm difference of 2.02%; a separate 20-step repeat with the same configuration matched history exactly. Repeated runs are not guaranteed to be bitwise identical; the cause remains unresolved.

A further 100-step mixed1m run used no explicit cache clearing: steps 5–99 reached 2580 TPS/card and 5.46% MFU, with peak reserved memory of 195.21 GiB and minimum sampled free memory of 47.26 GiB. LR, token and packing metrics matched history; maximum absolute loss/MTP CE differences were 2.87e-4/3.09e-4 and relative grad-norm difference was 31.21%. All values were finite, but the cause remains unresolved and long-run numerical equivalence is not established. This run did not save checkpoints.

A same-code 10-step mixed512 cache-control comparison matched loss/MTP loss/grad norm/LR exactly. Over steps 5–9, no explicit cache clearing achieved 2884 TPS/card, versus 1444 with per-step clearing (level 1).

Earlier long runs covered 200 mixed512 steps plus 10 resumed steps, 100 mixed1m steps, and 40 steps for each filled scenario. The resumed mixed512 loss/MTP loss/grad norm/LR/data position matched continuous training exactly; save/resume was not separately validated for the other scenarios. This 20-step revalidation did not repeat GPU checkpoint recovery or rebuild the image. All 70 relevant CPU regressions passed without skips, covering the real parser, tokens and supervision masks, scenario isolation, and resume arguments.
