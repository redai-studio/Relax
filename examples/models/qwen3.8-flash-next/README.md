# Qwen3.8-Flash-Next Training

[中文](./README.zh-CN.md)

## Overview

Qwen3.8-Flash-Next is a multimodal mixture-of-experts model with a vision encoder and 48 language-model layers. The language backbone has 125B parameters with 6B active per token, plus a 51B n-gram embedding table and a 4B multi-token prediction (MTP) module. It interleaves three Gated DeltaNet (GDN) layers with one Qwen Sparse Attention (QSA) layer. QSA selects up to 512 four-token blocks. Gated Residual mixes four residual streams, and per-layer embedding (PLE) adds bigram and trigram features. The released model has a native context of 262,144 tokens. [Model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [Architecture report](https://arxiv.org/abs/2608.30320)

Relax provides Megatron/Bridge full-model and LoRA supervised fine-tuning (SFT) and colocated Megatron/SGLang GRPO recipes. The shared model shape lives in [`qwen38-flash-next.sh`](../../../scripts/models/qwen38-flash-next.sh). The full-model SFT recipes below use BF16, a frozen QSA indexer, a frozen vision encoder and projector, and one trainable MTP layer. H800 freezes the n-gram table. The H20 and Blackwell SFT recipes freeze the full PLE module.

## Quick Start

Use the same Relax revision and matching image on every worker. [`docker/Dockerfile.cu13`](../../../docker/Dockerfile.cu13) builds the `relax` target with `GPU_ARCH=hopper` or `GPU_ARCH=blackwell` and includes the pinned Megatron/Bridge patches. The examples below run from the repository root on the head of a dedicated, already-running Ray cluster. Put the HF checkpoint, input data, images, and outputs on paths shared by all workers.

Set `MODEL_DIR` to the full HF checkpoint directory, including the MTP weights. Prepare an SFT JSONL with complete `messages` and optional `tools`. For multimodal samples, place one `<image>` marker in the user message per entry in `images`. [SFT input format](../../../docs/en/guide/sft-training.md)

```json
{"messages":[{"role":"user","content":"<image>What shape is shown?"},{"role":"assistant","content":"A triangle."}],"images":["/shared/data/triangle.png"]}
```

**Hopper (128×H800).** Set `PROFILE_STEP=0` for a run without stack capture. The script starts from HF for 15 updates by default. Set `SAVE_INTERVAL` to enable training checkpoint saves and resume, and retain `SAVE_DIR/EXP_NAME` for the next launch.

```bash
export MODEL_DIR=/shared/models/Qwen3.8-Flash-Next
export DATA_DIR=/shared/data/train.jsonl
export SAVE_DIR=/shared/checkpoints
export RELAX_SKIP_CLUSTER_CLEANUP=1

DRY_RUN=1 PROFILE_STEP=0 USE_CLEARML=0 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-128xh800.sh
# After checking the arguments, submit from the Ray head.
PROFILE_STEP=0 USE_CLEARML=0 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-128xh800.sh
```

**32×280G Blackwell.** In the Blackwell image, point `MEGATRON` at its patched checkout. `RELAX_HYBRID_EP_COMMIT` comes from the image. Set the Ray Jobs HTTP endpoint and use a stable `EXP_NAME` to resume at `SAVE_DIR/EXP_NAME`.

```bash
export MODEL_DIR=/shared/models/Qwen3.8-Flash-Next
export DATA_DIR=/shared/data/train.jsonl
export SAVE_DIR=/shared/checkpoints
export MEGATRON=/root/Megatron-LM
export RAY_JOBS_ADDRESS=http://ray-head:8265
export EXP_NAME=qwen38-sft-blackwell

DRY_RUN=1 bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh
bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh
```

| Recipe                                                                                                                                                                                          | GPUs              | Default SFT configuration                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [H800 multimodal](./scripts/run-qwen38-qsa-sft-128xh800.sh)                                                                                                                                     | 128×H800          | TP4/PP8/CP1/EP8/ETP1, GBS512, 32K sample limit and 64K packing budget                                                                     |
| [H20 multimodal](./scripts/run-qwen38-qsa-sft-40xh20.sh)                                                                                                                                        | 40×H20            | TP4/PP5/CP1/EP4/ETP1, GBS1024, 64K sample/packing limits, 100 updates from HF                                                             |
| [Blackwell multimodal](./scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh)                                                                                                                        | 32×280G Blackwell | TP2/PP4/CP1/EP8/ETP1, GBS512, 128K sample/packing limits, HybridEP and mixed QSA with Grouped long attention                              |
| [NV text](./scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh) / [NV multimodal](./scripts/run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh)                                                      | 32×280G Blackwell | Same parallel degrees and 128K budgets, with cuDNN QSA for long documents and a separate PP partition for text                            |
| [Blackwell long-context](./scripts/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh)                                                                                                                 | 32×280G Blackwell | 128K soft packing budget, up to 256K for a single sample, GBS4096, HybridEP by default. Long singleton samples can exhaust device memory. |
| [Blackwell OpenR1-MM SFT](./scripts/run-qwen38-qsa-sft-openr1mm-32xgpu-blackwell.sh)                                                                                                            | 32×280G Blackwell | 32K sample/packing limits, GBS128, MoK by default. Set `MOE_BACKEND=hybridep` for the reference backend.                                  |
| [Blackwell LoRA SFT](https://github.com/redai-studio/Relax/blob/c75b0a1cb3d03119b02933036287b8c6ad605cf0/examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-lora-sft-8xgpu-blackwell.sh) | 8×Blackwell       | TP2/PP1/CP1/EP8, 128K sample/packing limits, GBS512, rank-32 language adapters in merge mode                                              |

For OpenR1-MM SFT, convert the public dataset's `problem` and `solution` into user and assistant messages and save the corresponding image to the shared `images` path. See the [OpenR1-MM dataset](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified).

OpenR1-MM GRPO takes converted `prompt`/`label`/`image` Parquet. DAPO Math GRPO takes DAPO Math 17K and AIME 2024 JSONL for training and evaluation, respectively. [GRPO data utility](../../../scripts/tools/process_openr1.py) · [H800 OpenR1-MM](./scripts/run-qwen38-flash-next-openr1mm-128xgpu.sh) · [Blackwell OpenR1-MM](./scripts/run-qwen38-flash-next-openr1mm-grpo-blackwell.sh) · [Blackwell DAPO Math](./scripts/run-qwen38-flash-next-dapo-math-grpo-blackwell.sh)

## Validation and Benchmarks

The published comparisons in this table describe the recorded configurations and step windows. TPS/GPU uses total logical tokens divided by the summed time and GPU count. End-to-end TPS uses full step time, while Actor TPS uses training time. Reference MFU in the H20 comparison uses a fixed useful-work FLOPs estimate and a 148 TFLOP/s/GPU reference peak.

| Hardware and workload                                   | Public result                                                                                                                                                                                                                                                                                                                   | Evidence                                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 32×H20, full-model multimodal SFT smoke test on CUDA 12 | Two image-bearing optimizer updates completed.                                                                                                                                                                                                                                                                                  | [Image and training smoke test](https://github.com/redai-studio/Relax/pull/517#issuecomment-6745583406) |
| 32×H20, 128K packing, full-model multimodal SFT         | Selective logits alone measured 1,026.0 end-to-end TPS/GPU and 32.42% reference MFU over steps 5–9. Selective logits with fused linear CE measured 1,014.4 TPS/GPU and 32.06% reference MFU over the same window. The runs used different revisions and hosts.                                                                  | [Hopper CE validation](https://github.com/redai-studio/Relax/pull/643)                                  |
| 32×Blackwell, multimodal SFT                            | Grouped to cuDNN QSA increased end-to-end TPS/GPU from 3,235.2 to 3,560.4 (+10.05%) over steps 20–49 with the same PP layout.                                                                                                                                                                                                   | [NV QSA results](https://github.com/redai-studio/Relax/pull/509)                                        |
| 32×Blackwell, text SFT                                  | cuDNN QSA and a revised PP layout increased Actor TPS/GPU from 3,224.8 to 3,487.7 (+8.15%) over steps 20–45. The baseline was profiled during part of that window.                                                                                                                                                              | [NV QSA results](https://github.com/redai-studio/Relax/pull/509)                                        |
| 32×Blackwell, 128K packing, historical CuTe/MoK runtime | Grouped to CuTe at CP1 improved Actor throughput by 17.23% over steps 3–5. A separate CP1 run reached 200 cumulative updates, including a resume from checkpoint 49.                                                                                                                                                            | [CuTe and CP results](https://github.com/redai-studio/Relax/pull/691)                                   |
| 32×Blackwell, 128K packing, historical static CP2/CP4   | GDN route prebuild improved Actor throughput by 15.27%/18.82% versus the respective CP baselines. PLE convolution fusion and TP selector together added 5.79%/11.37% over route prebuild. The resulting configurations used 16.07%/26.00% less peak allocated memory and had 8.10%/26.16% lower Actor throughput than CuTe CP1. | [CuTe and CP results](https://github.com/redai-studio/Relax/pull/691)                                   |

### OpenSearch-VL SFT on Blackwell

The OpenSearch-VL run fine-tuned Qwen3.8-Flash-Next-F0 on 32 Blackwell GPUs (280G) with cuDNN QSA and froze the vision encoder and PLE. The evaluation compared the original F0 checkpoint and the final BF16 SFT checkpoint at step 66 (67 optimizer updates). Both used TP2 on Blackwell, a 32K context and up to 10 turns per question under the same JSON `crop` protocol. [V\*Bench evaluation protocol](../../opensearch_vl/evaluation.md)

| Checkpoint        | Correct / total | Pass@1 | Train-deduplicated Pass@1 |
| ----------------- | --------------- | ------ | ------------------------- |
| F0 base           | 39 / 191        | 20.42% | 20.42%                    |
| OpenSearch-VL SFT | 161 / 191       | 84.29% | 84.29%                    |

All 191 questions remained after training-set deduplication. The base model stopped on 146 questions due to invalid tool-call formatting. The SFT model answered 122 of those questions correctly. Both models scored 39/45 on the remaining questions. These scores measure task completion under the shared crop protocol. [OpenSearch-VL data preparation](../../opensearch_vl/data-preparation.md)

### H800 Multimodal SFT at 64K and 128K

The 128-GPU comparisons initialized from the same HF checkpoint within each pair and matched per-step token counts. The 64K run used TP4/PP8/CP1/EP8/ETP1 and GBS512. The 128K runs used TP8/PP8/CP1/EP8/ETP1 and GBS1024. TPS/GPU below is the mean of the logged end-to-end step rates.

| Packing and configuration change                                                          | Timed updates |       Step time |  End-to-end TPS/GPU |
| ----------------------------------------------------------------------------------------- | ------------: | --------------: | ------------------: |
| 64K, rebalance the PP layers and keep Adam state on GPU                                   |          2–10 | 40.33 → 38.85 s | 1,032.88 → 1,071.36 |
| 128K, Triton v2 → experimental Triton QSA with K/V prefetch and revised backward          |          4–27 | 97.08 → 93.11 s |     909.41 → 946.04 |
| 128K, add a Triton GDN input-projection forward kernel to the experimental QSA + NVLS run |           4–8 | 93.05 → 91.72 s |     940.83 → 956.84 |

The 64K candidate peaked at 65.93 GiB of sampled device memory. The 128K GDN forward candidate completed nine matched updates, with maximum relative differences of 0.111% in training loss and 0.056% in MTP loss. Adding a split-K projection weight-gradient kernel in a follow-up run yielded 898.55 TPS/GPU over steps 4–8. The timed window includes a slow step. The 128K kernel variants were experimental. [Reference H800 configuration](./scripts/run-qwen38-qsa-sft-128xh800.sh)

### H800 SFT with Trainable PLE

A 32K experiment on 128 H800 GPUs compared a frozen CPU n-gram table (`QWEN38_PLE_CPU_OFFLOAD=1`) with a trainable GPU table (`QWEN38_PLE_CPU_OFFLOAD=0`). Both runs used the same starting weights, batches, and parallel topology for 50 updates. The trainable variant omitted the n-gram table from the freeze rules, set `RELAX_QWEN38_PLE_STREAMED=0`, and used launcher changes to reduce memory. Losses average updates 6–50. The TPS/GPU average excludes the checkpoint-write step 26 from both runs.

| PLE n-gram table | Mean TPS/GPU | Mean training loss | Mean MTP loss |
| ---------------- | -----------: | -----------------: | ------------: |
| Frozen on CPU    |       528.72 |            0.90678 |       1.20293 |
| Trainable on GPU |       433.87 |            0.90643 |       1.19943 |

The trainable table had nonzero Adam moments in the step-50 checkpoint. A separate 64×H800 run with 32K packing, TP8/PP8/CP1/EP8, and trainable PLE saved all eight table shards after 500 updates. Full-size optimizer reload and a matched post-resume update remain to be verified. The public H800 launcher uses the frozen CPU table.

### Public Reproduction Gaps

- Reproduce the H800 throughput comparisons on a public dataset and measure the current 40×H20 launcher end to end. Record the dataset revision and split, image digest, effective configuration, training loss, MTP loss, gradient norm, held-out loss, TPS/GPU, and peak device memory. The launcher PRs document their configuration checks. [H800 recipe](https://github.com/redai-studio/Relax/pull/637) · [H20 defaults](https://github.com/redai-studio/Relax/pull/635)
- Publish the prepared OpenR1-MM SFT input or its deterministic conversion and split. Add base vs. SFT checkpoint results on the same held-out split with a published scoring rule and decoding settings. For GRPO, capture reward, train/rollout log-probability discrepancy, and a held-out score. [OpenR1-MM evaluator](../../../scripts/tools/eval_openr1mm.py)
- On 280G Blackwell, rerun CuTe and MoK with the released image and matching control configurations. Record the MoK vs. HybridEP results, 256K singleton memory and completion, and checkpoint save/resume. Keep the data, packing, recomputation, and measurement window fixed when comparing backends. [CuTe validation scope](https://github.com/redai-studio/Relax/pull/691) · [MoK validation scope](https://github.com/redai-studio/Relax/pull/704)

## Hopper Optimizations

| Area                           | Settings in the reference recipes                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QSA attention and selector     | Both recipes pair `MCORE_QSA_SPARSE_BACKEND=flex` with `RELAX_QWEN38_QSA_FUSED_ATTENTION=triton_v2` and `RELAX_QWEN38_QSA_FUSED_SELECTOR=1`. The fused path uses selected blocks on eligible inputs. H800 sets `RELAX_QWEN38_QSA_SELECTOR_TILE_MIB=384` to bound selector tiles.                                                                                                                                                                                                    |
| Logits and CE                  | Both recipes enable `--sft-chunked-logits`, `RELAX_SFT_SELECTIVE_LOGITS=1`, and `RELAX_SFT_SELECTIVE_MULTIMODAL_LOGITS=1`. Selection uses the effective loss mask on eligible CP1 packed SFT and MTP batches. Both default to `RELAX_SFT_FUSED_LINEAR_CE=0`. Hopper fused linear CE is available through the Quack path under its runtime guards. [Selective logits](https://github.com/redai-studio/Relax/pull/637) · [Hopper CE](https://github.com/redai-studio/Relax/pull/643)  |
| MoE and parallelism            | Both use `--sequence-parallel`, `--moe-token-dispatcher-type flex`, `--moe-flex-dispatcher-backend deepep`, and `--moe-grouped-gemm`. H800 adds `--moe-flex-dispatcher-num-sms 24` and `--moe-shared-expert-overlap`.                                                                                                                                                                                                                                                               |
| Activations, PLE and optimizer | Both enable `RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1` and place the frozen n-gram table on CPU with `QWEN38_PLE_CPU_OFFLOAD=1`. H800 adds HC mix/combine and PLE convolution fusion, `--chunked-optimizer-state-offload`, `--overlap-grad-reduce`, and `--overlap-param-gather`. Its `RELAX_QWEN38_RETAIN_QSA_ATTENTION=1` policy `last_stage_l48` retains layer 48 attention and full activations for layer 47 and MTP on the final stage. Set it to `0` to use regular recomputation. |

`RELAX_QWEN38_PLE_STREAMED=1` is an independent memory option for frozen PLE parameters with packed THD input and CP>1. [Streamed PLE validation](https://github.com/redai-studio/Relax/pull/666)

The public [H20 synthetic QSA replay](https://github.com/xiaoliang0601/Relax/blob/0f6fe4baea288d1aa5419e2e3814e7ff8ad24c38/examples/models/qwen3.8-flash-next/qsa_benchmark/RESULTS.md) reports operator time for different head geometries and document-length mixes. Use per-document lengths and local Q/KV head counts when comparing sparse kernels on packed samples.

## Blackwell Optimizations

| Area                   | Settings and prerequisites                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Mixed QSA              | The 32-GPU SFT recipe selects `RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed`. Flash handles short documents and `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped` is the long-document default. `cudnn_pr1765` is a separate CP1 option for local 12Q/1KV/D256 and needs the pinned cuDNN Frontend package. The NV recipes use Flash through 2,051 tokens per document. [NV image and routing](../../../docs/en/guide/qwen38-qsa.md#nv-runtime-requirements)         |
| CuTe QSA and static CP | Set `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=cute` in the 32-GPU recipe. For CP2 or CP4, pair `QWEN38_CP_SIZE=2` or `4` with `RELAX_QWEN38_QSA_CP=1`. Optional controls are `RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE=1`, `RELAX_QWEN38_PLE_CP_FUSED_CONV=1`, and `RELAX_QWEN38_QSA_CP_TP_SELECTOR=1` with the `torch` selector. This path uses BF16, static packed CP and frozen Stage-1 indexing. [CuTe setup and constraints](../../../docs/en/guide/qwen38-qsa.md) |
| Megatron MoE           | The 128K recipe uses `--moe-flex-dispatcher-backend hybridep`, `--moe-hybridep-pad-variable-tokens`, `--moe-shared-expert-overlap`, `--overlap-grad-reduce`, and `--overlap-param-gather`. MoK is a separate choice in the long-context and OpenR1-MM SFT recipes. It selects `--moe-megakernel-backend mok` and requires the SM103 derived image and node-local EP8 in this configuration. [MoK image and launch options](../../../docker/mok/README.md)    |
| Fused model operators  | The 128K and 256K recipes enable `RELAX_QWEN38_HC_FUSED_MIX=1`, `RELAX_QWEN38_HC_FUSED_COMBINE=1`, and `RELAX_QWEN38_PLE_FUSED_CONV=1`. The 128K recipe exposes `NVTE_USE_CUTLASS_GROUPED_GEMM` as a separate MoE GEMM option, defaulting to `0`.                                                                                                                                                                                                            |
| Long-context memory    | The 256K recipe uses chunked optimizer-state offload, CPU PLE placement and `RELAX_QWEN38_HC_RECOMPUTE=1`. Its sample-length limit and soft packing budget are set separately through `SEQ_LENGTH` and `MAX_TOKENS_PER_GPU`.                                                                                                                                                                                                                                 |

For longer contexts, the Blackwell HybridEP path now admits TP1/PP2/EP8/ETP1 with CP8 or CP16 on 32 GPUs when mixed CuTe QSA CP is enabled. With fully frozen PLE and static packed THD, set `QWEN38_PLE_INPUT_HALO_CP=1` and `QWEN38_PLE_INPUT_HALO_TRANSPORT=adaptive` to compute PLE for local output tokens and exchange the left convolution boundary when the document shards are long enough. The historical three-update comparisons kept local computation enabled and changed transport from `full` to `adaptive`. Peak PyTorch allocated memory fell from 204.05 to 200.02 GiB at 512K/CP8 and from 171.72 to 157.67 GiB at 1M/CP16. Training loss, MTP loss, and grad norm matched on the three updates in each comparison. [Long-context settings](https://github.com/redai-studio/Relax/blob/c75b0a1cb3d03119b02933036287b8c6ad605cf0/docs/en/guide/qwen38-qsa.md#runtime-prerequisites-for-long-context-cp) · [Memory results](https://github.com/redai-studio/Relax/pull/716)

Prepare comparisons with separate checkpoint directories. Verify the executed attention and MoE paths from the trace when reporting kernel-specific gains. [Performance tuning](../../../docs/en/guide/performance-tuning.md)
