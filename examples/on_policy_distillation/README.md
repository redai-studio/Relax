# On-Policy Distillation Examples

## Overview

On-Policy Distillation (OPD) trains the student model on its own rollout data while matching the teacher's token-level log-probabilities, enabling knowledge transfer from a large teacher to a smaller student.

Relax's OPD supports:

- **Token-Selection Modes**: `student_sampled`, `student_topk`, `teacher_topk`, `union`, determining which tokens' log-probabilities are used for KL computation.
- **Application Methods**: `adv` (modifying advantage) and `loss` (modifying loss), determining how the distillation signal is injected into training.

## Token-Selection Modes

| Mode              | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `student_sampled` | Compute KL only on student-sampled tokens                       |
| `student_topk`    | Compute KL on the student's top-K token set                     |
| `teacher_topk`    | Compute KL on the teacher's top-K token set                     |
| `union`           | Compute KL on the union of student and teacher top-K token sets |

## Application Methods: adv and loss

- **adv (advantage)**: `--opd-kl-coef 1.0 --opd-loss-coef 0.0`
- **loss**: `--opd-kl-coef 0.0 --opd-loss-coef 1.0`

Only one can be enabled at a time; they cannot be used simultaneously.

## Top-K Normalization Modes

For top-K modes (`student_topk`, `teacher_topk`, `union`), `--opd-norm-mode` controls how the tail probability mass is handled:

| Mode             | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| `tail` (default) | Keep the tail mass;                                                              |
| `norm`           | Normalize teacher and student top-K probabilities each to 1, discarding the tail |
| `trunc`          | Truncate directly;                                                               |

## SGLang Patch

When using top-K modes (`student_topk`, `teacher_topk`, `union`), the `RELAX_OPD_PER_POS_TOKEN_IDS` feature requires a source-level patch to SGLang:

```bash
cd /sgl-workspace/sglang
patch -p1 < /path/to/your/Relax/docker/patch/latest/sglang_per_pos_topk.patch
```

This patch enables per-position token ID log-prob collection, base64 encoding, and `drop_token_ids` optimization in SGLang. It only needs to be applied once at install time.

## Environment Variables

| Variable                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RELAX_OPD_PREEXPANDED_PATCH` | Set to `1` to enable the multimodal pre-expanded patch. **Required for multimodal OPD**. Because the tokenizer is not idempotent, this passes through parameters so SGLang directly accepts expanded `input_ids + image_data（base64)`; the SGLang-side tokenizer becomes a no-op and will not re-detokenize/tokenize during inference. Default: `0`. Affects both teacher and student engines (in adv mode, student-at-teacher-topk prefill also sends pre-expanded data). |
| `RELAX_OPD_PER_POS_TOKEN_IDS` | Set to `1` to enable per-position token ID log-prob collection. Changes the SGLang transfer format from `list[[int, float, None]]` to base64 encoding, significantly reducing serialization overhead. **Required** for `student_topk`, `teacher_topk`, `union` modes; not needed for `student_sampled`. Default: `0`. Requires the SGLang patch above.                                                                                                                      |

## Performance Benchmark

SGLang per-position top-K log-prob transfer optimization (Qwen3-4B, simulated data batch=128, prompt_len=1024, resp_len=7168, top_k=64, concurrency=128):

| Config                                                 | E2E      |
| ------------------------------------------------------ | -------- |
| `tokenizer-worker-num=1` + `list()`                    | 260.169s |
| `tokenizer-worker-num=1` + `base64` + `drop_token_ids` | 24.659s  |

## Example Scripts

### Text OPD

| Script                                                                                                   | Description                                                                                                                         |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [`math_opd/run-opd-qwen35-35B-A3B-8xgpu-colocate.sh`](math_opd/run-opd-qwen35-35B-A3B-8xgpu-colocate.sh) | Math OPD, student: Qwen3.5-35B-A3B; teacher: Qwen3.5-35B-A3B-GRPO-dapomath17k-400step; `student_sampled` + adv mode, 8 GPU colocate |

### Multimodal OPD

| Script                                                                                                                                             | Description                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [`vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-colocate.sh`](vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-colocate.sh)                   | Qwen3.5-9B → Qwen3.5-35B-A3B, 8 GPU colocate          |
| [`vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-2teacher-colocate.sh`](vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-2teacher-colocate.sh) | Same as above, dual teacher replicas                  |
| [`vision_opd/run-opd-qwen3.5_35ba3b-122ba10b-128xgpu-colocate.sh`](vision_opd/run-opd-qwen3.5_35ba3b-122ba10b-128xgpu-colocate.sh)                 | Qwen3.5-35B-A3B → Qwen3.5-122B-A10B, 128 GPU colocate |

## Common Combinations

Based on [`math_opd/run-opd-qwen35-35B-A3B-8xgpu-colocate.sh`](math_opd/run-opd-qwen35-35B-A3B-8xgpu-colocate.sh), modify key variables to switch modes:

> **KL type note**: `student_sampled` supports only `reverse_kl` and `low_var_kl`; `student_topk`, `teacher_topk`, and `union` support `reverse_kl`, `forward_kl`, and `jsd` (set via `--opd-kl-type`; `jsd` can be tuned with `--opd-jsd-alpha`).

**1. student_sampled + adv + reverse_kl (default, lowest overhead)**

```bash
OPD_KL_COEF=1.0
OPD_LOSS_COEF=0.0
OPD_TOKEN_SELECTION=student_sampled
OPD_KL_TYPE=reverse_kl
# RELAX_OPD_PER_POS_TOKEN_IDS not needed
```

**2. student_topk + adv + jsd**

```bash
export RELAX_OPD_PER_POS_TOKEN_IDS=1
OPD_KL_COEF=1.0
OPD_LOSS_COEF=0.0
OPD_TOKEN_SELECTION=student_topk
OPD_KL_TYPE=jsd
# Add to OPD_ARGS: --opd-log-prob-top-k 64 --opd-jsd-alpha 0.5
```

**3. union + loss + forward_kl**

```bash
export RELAX_OPD_PER_POS_TOKEN_IDS=1
OPD_KL_COEF=0.0
OPD_LOSS_COEF=1.0
OPD_TOKEN_SELECTION=union
OPD_KL_TYPE=forward_kl
# --opd-log-prob-top-k 64
```

**4. teacher_topk + adv + reverse_kl**

```bash
export RELAX_OPD_PER_POS_TOKEN_IDS=1
OPD_KL_COEF=1.0
OPD_LOSS_COEF=0.0
OPD_TOKEN_SELECTION=teacher_topk
OPD_KL_TYPE=reverse_kl
# --opd-log-prob-top-k 64
```

**5. Multimodal union + loss + jsd**

```bash
export RELAX_OPD_PREEXPANDED_PATCH=1     # required for multimodal
export RELAX_OPD_PER_POS_TOKEN_IDS=1     # required for topk modes
OPD_KL_COEF=0.0
OPD_LOSS_COEF=1.0
OPD_TOKEN_SELECTION=union
OPD_KL_TYPE=jsd
# Add to OPD_ARGS: --opd-log-prob-top-k 64 --opd-jsd-alpha 0.5
```
