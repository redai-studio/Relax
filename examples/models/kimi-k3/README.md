# Kimi K3 Training and Export

[中文](./README.zh-CN.md)

## Overview

Relax provides full-parameter and LoRA SFT for Kimi K3, with image inputs, packing, fixed CP, PP and EP. The model configuration is `scripts/models/kimi-k3.sh`; public recipes live in `examples/models/kimi-k3/scripts/`.

Use a training image built with the pinned Megatron/Bridge, FLA and NVRx dependencies in `docker/Dockerfile` or `docker/Dockerfile.cu13`. Model and dataset directories must be readable from every training node. These recipes are reference configurations; validate memory use and numerical behavior on the target cluster before a long run.

## Training

Run commands from the repository root on an existing Ray cluster. `MODEL_DIR` points directly to the original HF model directory; `DATA_DIR` points to the prepared dataset directory. Existing `HF_CHECKPOINT`, `HELLASWAG_DATA_DIR`, `OPENR1MM_DATA_DIR`, `POKEMON_DATA_DIR` and `LLAVA_DATA_DIR` overrides remain supported.

```bash
export MODEL_DIR=/shared/models/Kimi-K3
export DATA_DIR=/shared/data/openr1mm
export SAVE_DIR=/shared/checkpoints
export EXP_NAME=kimi-k3-openr1mm-full
DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh
bash scripts/entrypoint/ray-job.sh \
  examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh
```

| Recipe filename                             |     GPUs | Input                                              |
| ------------------------------------------- | -------: | -------------------------------------------------- |
| `run-kimi-k3-hellaswag-128xb300.sh`         | 128 B300 | `train.jsonl`, `validation.jsonl`                  |
| `run-kimi-k3-hellaswag-128xb300-rawtext.sh` | 128 B300 | Same files; raw context/target baseline comparison |
| `run-kimi-k3-openr1mm-128xb300.sh`          | 128 B300 | `train.parquet`                                    |
| `run-kimi-k3-llava-onevision-128xb300.sh`   | 128 B300 | `train/` JSONL shards and `READY.json`             |
| `run-kimi-k3-pokemon-64xb300.sh`            |  64 B300 | `pokemon_gpt4o_zh.parquet`                         |
| `run-kimi-k3-pokemon-128xb300.sh`           | 128 B300 | Same Parquet file                                  |
| `run-kimi-k3-pokemon-192xgpu-b300.sh`       | 192 B300 | Same Parquet file                                  |
| `run-kimi-k3-pokemon-lora-64xb300.sh`       |  64 B300 | Same Parquet file; language-backbone LoRA          |

`SAVE_DIR` enables saving to the stable `${SAVE_DIR}/${EXP_NAME}` directory and loading from the same directory. Without it, recipes do not save checkpoints. OpenR1-MM and OneVision save **model weights only** by default; optimizer and scheduler state are not preserved. `SAVE_OPTIMIZER=1` enables full-state saving, which needs substantially more host memory and storage. A model-only checkpoint is not an exact training resume.

OpenR1-MM and OneVision default to frozen vision; set `FREEZE_VISION_TOWER=0` for joint vision training and revalidate memory use. `FLA_TILELANG=0` is passed to training actors by default; an explicit environment override is supported. ClearML uses the existing runtime configuration. Ray submission waits for completion and streams logs by default; set `RAY_NO_WAIT=1` to submit in the background, where submission success does not establish training success.

## OneVision Data Preparation

The processing tool remains at `scripts/tools/prepare_llavaonevision.py`.

```bash
python -m scripts.tools.prepare_llavaonevision prepare \
  --output-dir /shared/data/onevision --workers 8
export DATA_DIR=/shared/data/onevision/sft
```

Preparation pins the dataset revision in a manifest, extracts embedded images into shared files and publishes `READY.json` only after all shards complete. Re-running resumes completed shards. `--subsets` produces `SUBSET_READY.json` for smoke preparation; it is not a complete dataset readiness marker.

The `sample` subcommand can instead select a fixed number of valid rows from locally cached Parquet shards. Its images are inline data URIs. For a complete dataset, use `prepare` and make the extracted image paths accessible from every training node.

## Export

Run conversion inside the compatible training image, connected to a Ray cluster. `CKPT_PATH` must point to one `iter_*` directory. The original HF directory supplies model configuration, tokenizer and native quantization layout.

```bash
export CKPT_PATH=/shared/checkpoints/experiment/iter_0000100
python examples/models/kimi-k3/tools/convert_kimi_k3_torch_dist_to_hf_parallel.py \
  --input-dir "${CKPT_PATH}" --origin-hf-dir "${MODEL_DIR}" \
  --output-dir "${CKPT_PATH}_hf" \
  --world-size 16 --tp 4 --pp 1 --ep 16 --expert-tp 1 \
  --cpus-per-worker 16
```

The parallel exporter requires PP=1, expert-TP=1 and EP=world-size. It validates the output before publishing. Existing output is rejected unless `--replace-output` is explicitly supplied.

For a language LoRA checkpoint, use the **exact original model used for training**:

```bash
python examples/models/kimi-k3/tools/merge_kimi_k3_lora_to_hf.py \
  --input-dir "${CKPT_PATH}" --origin-hf-dir "${MODEL_DIR}" \
  --output-dir "${CKPT_PATH}_hf" --workers 8 --cpus-per-worker 4
```

This produces full HF weights with LoRA merged before native MXFP4 quantization. Vision adapters are not supported by this merge path. `compare_kimi_k3_hf_exports.py` and `validate_kimi_k3_parallel_export.py` in the same tools directory provide shard comparison and quantization checks.

## Validation and Serving

The temporary SGLang validator requires a log directory:

```bash
python examples/models/kimi-k3/tools/validate_kimi_k3_sglang.py \
  --model-path "${CKPT_PATH}_hf" --log-dir "${CKPT_PATH}_serve_logs" \
  --tp-size 8 -- \
  --weight-loader-prefetch-checkpoints --context-length 16384 \
  --max-running-requests 16 --cuda-graph-max-bs 16 --disable-decode-cuda-graph
```

Its built-in checks are text-only. Use `scripts/tools/eval_openr1mm.py` for image evaluation; inspect `--help` for endpoint, dataset and output arguments.

## Reinforcement Learning

Kimi K3 supports colocated GRPO with native MXFP4 rollout weights. Training keeps BF16 expert parameters; after each update, Bridge converts routed experts into paired `weight_packed` / E8M0 `weight_scale` tensors. Dense, shared-expert and vision weights retain their configured unquantized format. Quantization configuration is read from either the top level or `text_config`.

The SGLang v0.5.17 patch shipped in this repository must be installed on **every rollout node**. Its reload hooks preserve captured runtime storage while refreshing MXFP4 repacks, fused decode buffers, MLA projections and AttnRes caches. The router supports worker removal and re-registration without applying stale requests or health checks to a replacement worker. PP weight conversion broadcasts cleaned configuration copies and avoids repeating TP/EP collectives after weights have already been gathered.

| Recipe                                                                  | Layout                                     | Purpose                                                              |
| ----------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| `scripts/training/text/run-kimi-k3-5l-8xgpu-grpo.sh`                    | Training TP2/EP4/ETP1, rollout TP8; 8 GPUs | Reduced 5-layer/128-expert checkpoint, text-only DAPO math smoke run |
| `examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh` | Training TP4/PP4/CP2/EP32/ETP1; 128 B300   | Full-parameter multimodal OpenR1-MM GRPO, including vision           |

The full recipe uses PP layer counts 21/24/24/24, eight 16-GPU rollout engines with attention TP8/DP2 and MoE EP16, CPU optimizer offload (default fraction 0.9), dummy rollout initialization followed by a full weight push, and no reference model. It clears inherited allocator settings, limits compiler threads and keeps Ray Serve probes responsive during weight synchronization. Defaults are 200 training rollouts and a model-only save every 200 rollouts, retaining one checkpoint. It initializes from HF unless `LOAD_DIR` is set. Model-only checkpoints do **not** restore optimizer state.

```bash
export MODEL_DIR=/shared/models/Kimi-K3
export DATA_DIR=/shared/data/openr1mm
export SAVE_DIR=/shared/checkpoints
export EXP_NAME=kimi-k3-openr1mm-grpo
export HF_CHECKPOINT="${MODEL_DIR}"
export PROMPT_SET="${DATA_DIR}/train.parquet"
DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh
bash scripts/entrypoint/ray-job.sh \
  examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh
```

The full recipe defaults to `${MODEL_DIR}/Kimi-K3` for model weights and `${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet` for data with `prompt`, `label` and `image` fields. The example above sets `HF_CHECKPOINT` and `PROMPT_SET` explicitly to reuse the SFT directory layout. The reduced recipe expects `dapo-math-17k.jsonl` under `DATA_DIR` and the reduced checkpoint directly under `MODEL_DIR`. Its entropy coefficient 0.01 keeps gradients nonzero when the unretrained smoke checkpoint receives zero rewards; it is not a quality baseline. Optional `SAVE_DIR` enables full-state saves and resume for this reduced recipe. Experiment names and checkpoint paths are stable; timestamps appear only in log/task names. ClearML uses existing runtime configuration.

The full recipe enables expert-routed weight updates by default; set `COLOCATE_EXPERT_WEIGHT_ROUTING=0` to use broadcast. Dynamic sampling filters are disabled by default; they can be enabled with explicit CLI arguments.

Both recipes enable `OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG=1` and `FLA_TILELANG=0` through `--train-env-vars`. Fake QAT uses a straight-through estimator: routed-expert forward weights use the same quantize/dequantize grid as online MXFP4 export, while gradients reach the BF16 master. The hook is disabled by default outside these recipes. Rollout sends exactly one raw media token per image; the training processor retains the original prompt and performs its own image expansion.

### Bounded Synchronous Checkpoint Saving

Both Dockerfiles apply `docker/patch/megatron/sync-save-bounded-staging.patch` after the main Megatron patch. The full GRPO recipe explicitly enables it; set `MEGATRON_SYNC_SAVE_BOUNDED_STAGING=0` to revert that recipe to the original synchronous writer. For other launches, the patch defaults to disabled. Async saving is unchanged.

- `MEGATRON_SYNC_SAVE_BOUNDED_STAGING=1`: stage a bounded window and write sequentially instead of preloading the complete checkpoint onto CPU.
- `MEGATRON_SYNC_SAVE_STAGE_BYTES`: per-rank live staging budget, default 1 GiB. A single larger tensor is allowed when the window is empty, so the bound is `max(budget, largest_tensor)`.
- Budget `0` stages/writes one tensor at a time; it does **not** select the original writer.
- This bounds live staging tensors, not total RSS, optimizer/offload memory, serialization scratch or filesystem page cache. Writing may trade throughput for memory; measure on the target storage.

The recipe forwards both environment variables to training actors. Old images without this patch continue using their original save implementation even if these variables are set; rebuild or patch all nodes consistently to obtain the optimization. Existing running jobs do not pick up the patch. Checkpoint format and loading remain unchanged. See `docker/patch/megatron/sync-save-bounded-staging.md` for failure handling and validation limits.

### Diagnosing Train/Rollout Differences

`examples/models/kimi-k3/tools/probe_mxfp4_mismatch_floor.py` compares prompt logprobs for native MXFP4 and BF16-dequantized copies, using identical token IDs and the same SGLang engine. Supply `--mxfp4-dir`, `--bf16-dir`, `--data` and optionally `--n-prompts`. It uses TP1, so each tested checkpoint must fit on one GPU. This is a diagnostic reference, not a proof that an online weight push is correct: also check reload storage identity, finite losses/gradients and repeated updates in the target image. CPU tests do not establish CUDA-graph, quantization-kernel or multi-node save/reload correctness.

## Limits

- MTP, dynamic CP, all-gather CP and VPP are not supported for Kimi K3. Fixed CP uses zigzag partitioning; TP greater than one requires sequence parallelism.
- Large context lengths, joint vision training and full-state checkpoint saves require separate capacity validation. These scripts do not establish 256k-context readiness.
- CPU unit tests and Gloo tests do not validate NCCL, multi-node Ray export, CUDA quantization or actual image serving. Run those checks in the target image and cluster before merging for production use.

## Next Steps

- [SFT Training](../../../docs/en/guide/sft-training.md)
- [Model Checkpoint Conversion](../../../docs/en/guide/model-conversion.md)
- [LoRA Training](../../../docs/en/guide/low-rank-adaptation-training.md)
