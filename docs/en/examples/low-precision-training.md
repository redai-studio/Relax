# Low-Precision Training (FP8 & INT4)

Relax supports low-precision RL post-training along two axes: **FP8 training** (Megatron-LM native, real FP8 forward) and **INT4 fake-QAT** (BF16 master weights with INT4 fake-quant on MoE expert layers). Both modes drive a **real low-precision rollout** in SGLang and synchronize weights via NCCL after every training step.

## Overview

Two end-to-end recipes are wired up in this repository:

| Mode               | Training side                                                                 | Rollout side                              | Reference launch script                                       |
| ------------------ | ----------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------- |
| **FP8**            | Megatron-LM native FP8 (`e4m3`, blockwise)                                    | SGLang FP8 inference (real FP8 weights)   | `scripts/training/text/run-qwen3-30B-A3B-fp8-8xgpu.sh`        |
| **INT4 fake-QAT**  | BF16 forward + STE INT4 fake-quant on `TEGroupedLinear` (MoE experts only, symmetric) | SGLang W4A16 inference (compressed-tensors, **symmetric**, group_size=128) | `scripts/training/text/run-qwen3-30B-A3B-int4-8xgpu.sh` |

Four offline tools support the workflow:

- `scripts/tools/convert_hf_to_fp8.py` — quantize a BF16/FP16 HF checkpoint to FP8.
- `scripts/tools/convert_fp8_to_bf16.py` — dequantize a block-quantized FP8 HF checkpoint back to BF16 (the inverse of `convert_hf_to_fp8.py`; used when you start from a pre-quantized FP8 release and need a pure BF16 HF for other tooling).
- `scripts/tools/convert_hf_to_int4.py` — quantize a BF16 HF checkpoint to W4A16 (compressed-tensors).
- `scripts/tools/convert_moe_int4_to_bf16.py` — dequantize a W4A16 HF checkpoint back to BF16 (used when you start from a pre-quantized W4A16 release and need a BF16 HF for non-bridge workflows or other tooling).

## Architecture

Both modes use the standard colocate (`--colocate`) layout: actor and rollout time-share the same GPUs. The low-precision plumbing only changes what flows between them.

```
                 ┌──────────────────────────────────────────────────────┐
                 │                  Training side (Actor)               │
                 │  Megatron-LM, transformer_engine, --bf16             │
                 │                                                      │
                 │  FP8 mode:   real FP8 forward (TE blockwise e4m3)    │
                 │  INT4 mode:  BF16 forward + fake-int4 STE on         │
                 │              TEGroupedLinear._get_weight_tensors()   │
                 └────────────────────────┬─────────────────────────────┘
                                          │
                            per-step weight sync via NCCL
                                          │
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                  Rollout side (SGLang)              │
                 │                                                     │
                 │  FP8 mode:   real FP8 weights                       │
                 │              quantizer_fp8.quantize_params_fp8      │
                 │  INT4 mode:  real W4A16 (AWQ pack)                  │
                 │              quantizer_compressed_tensors           │
                 │                  .quantize_params_compressed_tensors│
                 └─────────────────────────────────────────────────────┘
```

The weight-update pipeline (`relax/backends/megatron/weight_update/`) dispatches on `quantization_config.quant_method` read from `--hf-checkpoint/config.json`:

- `quant_method == "fp8"` → `quantize_params_fp8` (`weight_conversion/processors/quantizer_fp8.py`)
- `quant_method == "compressed-tensors"` → `quantize_params_compressed_tensors` (`weight_conversion/processors/quantizer_compressed_tensors.py`)

## Offline Quantization Tools

### `convert_hf_to_fp8.py`

Quantize a BF16/FP16 HF safetensors checkpoint to FP8.

```bash
python scripts/tools/convert_hf_to_fp8.py \
  --model-dir /path/to/Qwen3-30B-A3B \
  --save-dir  /path/to/Qwen3-30B-A3B-FP8 \
  --strategy  block \
  --block-size 128 128 \
  --max-workers 4
```

| Flag             | Default | Description                                                                                          |
| ---------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `--model-dir`    | —       | Source HF safetensors directory.                                                                     |
| `--save-dir`     | —       | Output directory.                                                                                    |
| `--strategy`     | `block` | One of `block` / `channel` / `tensor`. `block` writes the `fp8` layout; `channel` writes `compressed-tensors`. |
| `--block-size`   | —       | Two ints (e.g. `128 128`) when `--strategy=block`.                                                   |
| `--max-workers`  | `1`     | Thread pool size for shard-parallel processing.                                                      |
| `--scale-fmt`    | `None`  | Optional, set to `ue8m0` to emit UE8M0 scales.                                                       |

Skipped modules (kept as-is): `layernorm`, `embed`, `router`, `lm_head`, `mlp.gate.*`, `norm`, `eh_proj`, `weights_proj`, `conv1d`, `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`. The set is hardcoded in the script's key filter.

Output:

- Quantized `*.safetensors` shards (FP8 weights + `weight_scale_inv` / `weight_scale`).
- Updated `config.json` with a `quantization_config` block. For `block`/`tensor` the block is `{"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [...], "modules_to_not_convert": [...]}`. For `channel` it follows the compressed-tensors schema.
- Refreshed `model.safetensors.index.json`.

### `convert_fp8_to_bf16.py`

Dequantize a block-quantized FP8 HF checkpoint back to BF16. Use this when you start from a pre-quantized FP8 release and need a pure BF16 HF for downstream tooling.

```bash
python scripts/tools/convert_fp8_to_bf16.py \
  --model-dir /path/to/Qwen3-30B-A3B-FP8 \
  --save-dir  /path/to/Qwen3-30B-A3B-bf16 \
  --max-workers 4
```

| Flag             | Default | Description                                                                 |
| ---------------- | ------- | --------------------------------------------------------------------------- |
| `--model-dir`    | —       | Source FP8 HF safetensors directory.                                        |
| `--save-dir`     | —       | Output directory.                                                           |
| `--max-workers`  | `1`     | Thread pool size for shard-parallel processing.                             |

Each FP8 `weight` is paired with its `weight_scale_inv` and dequantized via a Triton kernel (`weight_dequant_kernel`). Shards are processed in parallel; scale tensors that live in a different shard are pulled on demand via `safetensors.safe_open`. Tensors with `element_size() > 1` (already non-FP8) are copied through unchanged; FP8 tensors whose paired `_scale_inv` cannot be located are kept as-is with a warning.

Output:

- BF16 `*.safetensors` shards (dequantized FP8 weights; `_scale_inv` tensors are dropped).
- `config.json` with the `quantization_config` block stripped so downstream loaders don't try to dequantize already-dequantized weights.
- Refreshed `model.safetensors.index.json` without the obsolete `_scale_inv` entries.

::: tip
For the FP8 training workflow you usually do **not** need this script — bridge mode (`--megatron-to-hf-mode bridge`) reads the FP8 HF directly. This tool is for offline conversion when you need a BF16 HF as input to a different pipeline (e.g. as a `--ref-load` source for another recipe, or to feed `convert_hf_to_int4.py`).
:::

### `convert_hf_to_int4.py` {#convert-hf-to-int4}

Quantize a BF16 HF checkpoint to W4A16 (compressed-tensors). Uses the `fake_int4_quant_cuda` kernel, which must be built first (see [Build the int4_qat kernel](#build-int4-qat-kernel)).

```bash
python scripts/tools/convert_hf_to_int4.py \
  --model-dir /path/to/Qwen3-30B-A3B \
  --save-dir  /path/to/Qwen3-30B-A3B-int4 \
  --group-size 128 \
  --is-symmetric \
  --max-workers 4
```

| Flag              | Default                                                                                                                                                            | Description                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `--model-dir`     | —                                                                                                                                                                  | Source HF safetensors directory.                                                                                     |
| `--save-dir`     | —                                                                                                                                                                  | Output directory.                                                                                                    |
| `--group-size`   | `32`                                                                                                                                                               | INT4 group size; the training script uses `128`.                                                                     |
| `--is-symmetric` | `false` (CLI default) — **set this flag for INT4 fake-QAT training**                                                                                               | Symmetric quantization. Required to match the training-side STE (which is hardcoded symmetric); without it train/rollout distributions diverge. |
| `--ignore-rules` | `re:.*lm_head.*`, `re:.*norm.*`, `re:.*embed.*`, `re:.*self_attn.*`, `re:.*shared_experts.*`, `re:.*mlp\.(gate|up|gate_up|down)_proj.*`, `re:.*mlp\.gate\.*`       | Patterns (regex with `re:` prefix or literal prefix match) for keys to keep in original dtype. Default ignores everything except MoE expert `linear_fc1`/`linear_fc2`. |
| `--max-workers`  | `1`                                                                                                                                                                | Thread pool size.                                                                                                    |

::: warning
The default `--ignore-rules` is tuned for an MoE topology where only **expert** weights get quantized. If you change the ignore list, make sure it stays in sync with the training-side fake-QAT scope (which only touches `TEGroupedLinear`, i.e. MoE expert layers) — otherwise rollout and training will see different quantization patterns.
:::

::: danger
**Always pass `--is-symmetric` when the resulting checkpoint will be used as the `--hf-checkpoint` for INT4 fake-QAT training.** The training-side STE in `docker/patch/megatron/20260506-85bced0ae.patch` is hardcoded symmetric (`q_max=7`, no zero-point). If the W4A16 checkpoint is asymmetric (the CLI default), `pack_layer(sym=False)` will produce zero-point-shifted weights at rollout that differ from what training "saw", breaking the central premise of QAT.
:::

Output:

- Quantized `*.safetensors` with `weight_packed` (int32-packed int4), `weight_scale`, `weight_shape`, and (if asymmetric) `weight_zero_point` triplets per matched weight.
- Updated `config.json` with a compressed-tensors `quantization_config` block.

### `convert_moe_int4_to_bf16.py`

Dequantize a W4A16 compressed-tensors HF checkpoint to BF16. Use this when you start from a pre-quantized W4A16 release and need a pure BF16 HF for tooling.

```bash
python scripts/tools/convert_moe_int4_to_bf16.py \
  --model-dir /path/to/Qwen3-30B-A3B-int4
  # default output: /path/to/Qwen3-30B-A3B-int4_bf16
```

| Flag                          | Default               | Description                                                                                  |
| ----------------------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| `--model-dir`                 | —                     | Source W4A16 HF checkpoint.                                                                  |
| `--output-dir`                | `<model-dir>_bf16`    | Output directory.                                                                            |
| `--files`                     | all `*.safetensors`   | Limit to a subset of shards (useful when re-running after a partial failure).                |
| `--config-path`               | `<model-dir>/config.json` | Override path to read `group_size` from.                                                  |
| `--overwrite`                 | `false`               | Re-process shards even if the output already exists.                                         |
| `--keep-quantization-config`  | `false`               | Keep the `quantization_config` in the output `config.json` instead of stripping it.          |

Output:

- BF16 `*.safetensors` shards (expert `weight_packed` triplets are merged back to `.weight`; non-expert tensors are copied verbatim).
- `config.json` with `quantization_config` stripped (unless `--keep-quantization-config`).
- Sidecar `quantization_config.json` containing the stripped block plus an augmented `ignore` list (top-level namespaces such as `vision_tower` / `mm_projector` that have plain `.weight` keys but no `weight_packed` triplets are added).

::: tip
For the INT4 fake-QAT training workflow you usually do **not** need this script — bridge mode (`--megatron-to-hf-mode bridge`) loads W4A16 directly via the patched `build_conversion_tasks` in `megatron/bridge/models/qwen/qwen3_moe_bridge.py`, which injects synthetic `.weight` keys for the packed triplets.
:::

## Qwen3-30B Training Workflows

Relax ships two reference recipes for Qwen3-30B-A3B (8-GPU colocate): **FP8 native training** and **INT4 fake-QAT**. Both share the same Megatron patch and colocate layout — only the weight path and launch script differ.

### Common Prerequisites

1. A BF16 HF checkpoint (e.g. `Qwen3-30B-A3B`).
2. The Megatron patch at `docker/patch/megatron/20260506-85bced0ae.patch` applied (baked into the project Dockerfile). It provides both the FP8 overrides and the INT4 `_FakeInt4QuantizationSTE` that overrides `TEGroupedLinear._get_weight_tensors()`.

The FP8 recipe additionally needs a TransformerEngine build with FP8 blockwise scaling support. The INT4 recipe additionally needs the `fake_int4_quant_cuda` CUDA extension built — see [Build the int4_qat kernel](#build-int4-qat-kernel) below.

### FP8 Recipe

#### Steps

1. **Quantize the HF checkpoint to FP8:**

   ```bash
   python scripts/tools/convert_hf_to_fp8.py \
     --model-dir ${MODEL_DIR}/Qwen3-30B-A3B \
     --save-dir  ${MODEL_DIR}/Qwen3-30B-A3B-FP8 \
     --strategy  block --block-size 128 128
   ```

2. **Configure the path slots in the launch script** (`scripts/training/text/run-qwen3-30B-A3B-fp8-8xgpu.sh`):

   | Slot                | Should point to                                                                                                                                            |
   | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `--hf-checkpoint`   | The **FP8 HF directory** produced in step 1 (e.g. `${MODEL_DIR}/Qwen3-30B-A3B-FP8`). Drives SGLang init and the push-side `quantize_params_fp8` config.    |
   | `--ref-load`        | Also the **FP8 HF directory** — in pure FP8 training the reference model and actor share one FP8 HF (forward runs in native FP8 on both sides).            |
   | `--load` / `--save` | A **BF16 Megatron checkpoint directory** for resume / save (identical to a plain BF16 run; can be left empty on cold start).                               |

3. **Launch training:**

   ```bash
   bash scripts/training/text/run-qwen3-30B-A3B-fp8-8xgpu.sh
   ```


### INT4 fake-QAT Recipe

#### Build the int4_qat kernel {#build-int4-qat-kernel}

```bash
cd relax/backends/megatron/kernels/int4_qat
pip install -e . --no-build-isolation
```

The build produces `fake_int4_quant_cuda.cpython-<py>-x86_64-linux-gnu.so` in the same directory. The rollout-side `quantizer_compressed_tensors.py` and `convert_hf_to_int4.py` both `import fake_int4_quant_cuda` from this kernel.

#### Steps

1. **(Optional) Quantize the HF checkpoint to W4A16 — must be symmetric:**

   ```bash
   python scripts/tools/convert_hf_to_int4.py \
     --model-dir ${MODEL_DIR}/Qwen3-30B-A3B \
     --save-dir  ${MODEL_DIR}/Qwen3-30B-A3B-int4 \
     --group-size 128 \
     --is-symmetric
   ```

   `--is-symmetric` is required to align with the training-side STE. If you already have a W4A16 release, open its `config.json` and check `config_groups.group_0.weights.symmetric == true`; if it's `false`, regenerate (or run `convert_moe_int4_to_bf16.py` to get BF16 and then re-quantize with `--is-symmetric`).

2. **Configure the path slots in the launch script** (`scripts/training/text/run-qwen3-30B-A3B-int4-8xgpu.sh`) — the two HF paths play **distinct** roles, do not point them at the same directory:

   | Slot                | Should point to                                                                                                                                                                                                                |
   | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | `--hf-checkpoint`   | The **W4A16 INT4 HF directory** (e.g. `${EXP_DIR}/Qwen3-30B-A3B-int4`). Its `config.json` carries `quantization_config.quant_method == "compressed-tensors"`, which is what routes the per-step push through `quantize_params_compressed_tensors`. |
   | `--ref-load`        | The **BF16 HF directory** (e.g. `${EXP_DIR}/Qwen3-30B-A3B`, the un-quantized original). The STE adds INT4 quant noise on top of real BF16 weights in the forward path, so it must load real BF16 — not W4A16.                  |
   | `--load` / `--save` | A **BF16 Megatron checkpoint directory** for resume / save (can be left empty on cold start; Megatron will initialize from `--ref-load`).                                                                                       |

3. **Launch training:**

   ```bash
   bash scripts/training/text/run-qwen3-30B-A3B-int4-8xgpu.sh
   ```


## Kimi-K2.6 256xGPU INT4 QAT (Text & Multimodal)

For very large MoE models like Kimi-K2.6 — where the only available HF release is already W4A16 — Relax ships a slightly different INT4 fake-QAT recipe: **two distinct checkpoints** (one INT4 for SGLang inference, one BF16 cast for Megatron training) instead of a single W4A16 HF driving both sides. The training side still runs the same BF16 forward + STE INT4 fake-quant on MoE experts, but the **inference side loads the original W4A16 release verbatim** (its param dict registers `weight_packed`/`weight_scale`/`weight_shape`), avoiding re-quantizing a trillion-param model at init.

Two launchers cover both modalities:

| Script                                                       | Data                              | Algorithm | Reward     |
| ------------------------------------------------------------ | --------------------------------- | --------- | ---------- |
| `scripts/training/text/run-kimi-k2.6-256xgpu-int4.sh`        | `dapo-math-17k`                   | GRPO      | `math`     |
| `scripts/training/multimodal/run-kimi-k2.6-256xgpu-int4.sh`  | `multimodal-open-r1-8k-verified`  | GRPO      | `openr1mm` |

### Prerequisite

Cast the original W4A16 release to a BF16 HF directory **once** — Megatron's bridge needs real BF16 weights to load, since the STE only adds quant noise in the forward path:

```bash
python -m relax.utils.quant_cast.convert_moe_int4_to_bf16 \
    --model-dir  ${MODEL_DIR}/Kimi-K2.6 \
    --output-dir ${MODEL_DIR}/Kimi-K2.6_bf16
```

### Dual-checkpoint layout

```bash
HF_INT4="${MODEL_DIR}/Kimi-K2.6/"        # original compressed-tensors W4A16 release
HF_BF16="${MODEL_DIR}/Kimi-K2.6_bf16/"   # produced by the prerequisite above

CKPT_ARGS=(
   --hf-checkpoint        ${HF_INT4}   # AutoConfig → quant_method/group_size → routes push to compressed-tensors
   --sglang-hf-checkpoint ${HF_INT4}   # SGLang loads W4A16 verbatim (param dict = weight_packed/scale/shape)
   --ref-load             ${HF_BF16}   # Megatron bridge loads BF16; STE rounds to INT4 each forward
   --megatron-to-hf-mode  bridge
)
```

Each flag plays a distinct role:

| Flag                      | Checkpoint   | Read by                              | Purpose                                                                                                                       |
| ------------------------- | ------------ | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `--hf-checkpoint`         | INT4 (W4A16) | `AutoConfig` (push-side dispatcher)  | Sets `hf_config.quantization_config.quant_method == "compressed-tensors"` so the per-step push auto-routes through `quantize_params_compressed_tensors`. |
| `--sglang-hf-checkpoint`  | INT4 (W4A16) | SGLang engine init                   | **Must be the INT4 dir**, not the BF16 cast — otherwise SGLang's param dict has plain `.weight` keys and pushes are silently dropped with `X.weight_packed not found in params_dict`. |
| `--ref-load`              | BF16         | Megatron bridge loader               | Real BF16 working/master weights; the STE adds INT4 quant noise on top each forward.                                          |

### Launch

```bash
# text-only
bash scripts/entrypoint/ray-job.sh scripts/training/text/run-kimi-k2.6-256xgpu-int4.sh
# multimodal
bash scripts/entrypoint/ray-job.sh scripts/training/multimodal/run-kimi-k2.6-256xgpu-int4.sh
```

Both scripts share the same parallelism and INT4 plumbing:

| Setting                                            | Value                                                                                  | Notes                                                                                                  |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Parallelism                                        | TP=8, PP=8, CP=4, EP=32, ETP=1                                                         | 256 GPUs total. INT4 QAT only changes the weight-update path, not parallelism.                         |
| `OPEN_TRAINING_INT4_FAKE_QAT_FLAG`                 | `1`                                                                                    | Trips the `_FakeInt4QuantizationSTE` inside `TEGroupedLinear._get_weight_tensors()`.                   |
| `OPEN_TRAINING_INT4_GROUP_SIZE`                    | `32`                                                                                   | Matches the W4A16 release's per-group scale layout (Kimi uses **32**, not 128 as in the Qwen3-30B recipe). |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`   | `256`                                                                                  | DeepEP low-latency dispatch buffer; default 128 collides with cuda_graph capture at bs=128.            |
| `--rollout-num-gpus-per-engine`                    | `16`                                                                                   | One SGLang engine per 16 GPUs → 16 engines across 256 GPUs.                                            |
| `--sglang-{dp-size,ep-size}`                       | `16` each                                                                              | DP-attention + EP within each 16-GPU engine.                                                           |
| `--sglang-mem-fraction-static`                     | `0.7`                                                                                  | Leaves headroom for the weight-update buffer at this scale.                                            |
| Optimizer                                          | Adam + `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` | Required at the 1T-param scale to fit fp32 master weights.                                  |
| Recompute                                          | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`     | Full activation checkpointing — required at this scale.                                                |

The two scripts differ in data, reward, and minor algorithm tuning:

- **Text** (`run-kimi-k2.6-256xgpu-int4.sh`): `dapo-math-17k` with `--rm-type math`, `--rollout-max-response-len 16384`, `--global-batch-size 256`, `--lr 1e-6`, plus an `EVAL_ARGS` block (AIME-2024, `--eval-interval 20`).
- **Multimodal** (`run-kimi-k2.6-256xgpu-int4.sh` under `scripts/training/multimodal/`): `multimodal-open-r1-8k-verified` with `--rm-type openr1mm`, `--multimodal-keys '{"image":"image"}'`, `--image-max-token-num 256`, `--rollout-max-prompt-len 2048` / `--rollout-max-response-len 4096`, `--global-batch-size 512`, `--lr 5e-6`. The multimodal launcher additionally sets `--vision-dp-when-tp` and `--decoder-first-pipeline-num-layers 1 --decoder-last-pipeline-num-layers 6` to fit the vision tower into the PP-8 layout.

::: warning
Do not swap `--sglang-hf-checkpoint` to the BF16 cast for "consistency". SGLang's parameter registration happens once at init; if the registered keys are `.weight` (BF16) but the push sends `.weight_packed` (INT4), every push is dropped silently and training proceeds with stale rollout weights.
:::

::: tip
This recipe assumes the W4A16 release was produced with **symmetric** quantization (matching the training-side STE). If you regenerate the W4A16 from BF16 via `convert_hf_to_int4.py`, always pass `--is-symmetric` — see the warning in the [Offline Quantization Tools](#convert-hf-to-int4) section above.
:::
