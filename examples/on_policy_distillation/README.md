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

The Relax v0.5.17 training images include the independent per-position top-K patch. Cross-model top-K queries (`student_topk`, `union`, and `teacher_topk` advantage mode) require this patch; stock SGLang queries the same token set at every position. The patch preserves returned query IDs and supports scheduler and logits chunking.

For an existing v0.5.17 environment that does not have the patch, apply it once after the main Relax SGLang patch:

```bash
cd /sgl-workspace/sglang
git apply --check /path/to/Relax/docker/patch/latest/sglang_per_pos_topk.patch
git apply /path/to/Relax/docker/patch/latest/sglang_per_pos_topk.patch
```

## Native Image OPD

Image OPD, OPSD, and MOPD use SGLang v0.5.17 native processing: each request sends collapsed prompt IDs, the original sampled response IDs, and standard base64 images. The client processor still determines the expanded prompt length for logprob alignment. Teacher and student must use compatible tokenizers and image processors, including image resizing settings; audio and video requests are not supported by this path.

When OPSD changes the teacher prompt but supplies no teacher media override, it reuses student images. An explicit empty teacher media mapping selects text-only teacher input. Student cross-top-K prefill always uses student images.

Both direct teacher endpoints and the student router must preserve `return_prompt_token_ids`. Relax checks the full expanded sequence, response logprob IDs, and cross-top-K query IDs before consuming results. Missing or mismatched IDs fail the image request. The text-based `RadixTreeMiddleware` is unsupported for this path. Configure stop tokens for media placeholders; image OPD cannot rewrite sampled response IDs or apply a different teacher EOS ID.

## Environment Variables

| Variable                      | Description                                                                                                                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SGLANG_MM_AVOID_RETOKENIZE`  | Set to `true` on native image teacher and student engines. This is the SGLang v0.5.17 default; Relax-managed teachers set it explicitly. External endpoints need the same processor behavior.                            |
| `RELAX_OPD_PER_POS_TOKEN_IDS` | Set to `1` for compact base64 logprob transport in top-K runs. Requires the independent patch above. Per-position query width is sent in each request, and returned IDs are retained for alignment checks. Default: `0`. |

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
export SGLANG_MM_AVOID_RETOKENIZE=true  # native image processing
export RELAX_OPD_PER_POS_TOKEN_IDS=1     # compact top-K transport
OPD_KL_COEF=0.0
OPD_LOSS_COEF=1.0
OPD_TOKEN_SELECTION=union
OPD_KL_TYPE=jsd
# Add to OPD_ARGS: --opd-log-prob-top-k 64 --opd-jsd-alpha 0.5
```
