# Qwen3.8 Flash Next on 32 Blackwell SM103 GPUs

The text and image-text SFT recipes use four eight-GPU nodes, 131,072-token context, global batch 512, and 50 optimizer updates by default. Each recipe saves a resumable checkpoint at update 50 and logs to online ClearML.

| Recipe in `scripts/` (replace `<data>` with `text` or `multimodal`) | Parallelism        | Optimizations                                                                 |
| ------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------- |
| `run-qwen38-qsa-<data>-sft-baseline-50step-32xgpu-blackwell.sh`     | TP2, PP4, EP8, CP1 | Distributed optimizer, HybridEP, native mixed QSA                             |
| `run-qwen38-qsa-<data>-sft-direct-only-50step-32xgpu-blackwell.sh`  | Same               | Direct QSA on long-document suffixes                                          |
| `run-qwen38-qsa-<data>-sft-paired-50step-32xgpu-blackwell.sh`       | Same               | Direct QSA with dynamically sized query pairs                                 |
| `run-qwen38-qsa-<data>-sft-paired-mfsdp-50step-32xgpu-blackwell.sh` | Same               | Same paired attention with Megatron FSDP, FP32 gradient communication         |
| `run-qwen38-qsa-<data>-sft-mfsdp-50step-32xgpu-blackwell.sh`        | TP2, PP2, EP8, CP1 | Full MFSDP recipe with 26/22 layers, DeepEPv2 and BF16 gradient communication |

For a backend comparison, use the two PP4 paired recipes with the same data and separate checkpoint directories. The PP2 recipe measures the combined effect of the parallel split, sharding, communication and operator changes.

## Run

Build the repository's Blackwell `docker/Dockerfile.cu13` profile. Its Megatron tree contains the pinned Bridge and MCore sources with the `common`, `cu13`, and `blackwell` sections of `docker/patch/latest/megatron.patch` applied in that order. Use the matching Relax checkout on all nodes. Set `MEGATRON` to this patched tree and provide the HF checkpoint, matching text or image-text dataset, and writable checkpoint root via `MODEL_DIR`, `DATA_DIR`, and `SAVE_DIR`. Set `RAY_JOBS_ADDRESS` and the usual online ClearML credentials in the cluster environment.

```bash
recipe=examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-text-sft-mfsdp-50step-32xgpu-blackwell.sh
DRY_RUN=1 bash "${recipe}"
RELAX_SKIP_CLUSTER_CLEANUP=1 bash scripts/entrypoint/ray-job.sh "${recipe}"
```

Set `NUM_STEPS` and `SAVE_INTERVAL` together for a shorter verification. Use a new `SAVE_DIR` or `EXP_NAME` for independent runs. Multimodal examples use the `images` field and a 128-token budget per image. The framework derives rank-side image preprocessing from train prefetch, per-rank fetch and staleness.

## Performance observations

Each text run processed 405,940,021 effective tokens over 50 updates. Each image-text run processed 288,971,498 effective tokens and averaged 4.6816 images per sample. Token counts, image counts and global batch matched the baseline at every update. Steady-state TPS is the sum of effective tokens divided by 32 GPUs and the sum of actor training seconds over zero-based updates 3–49. Full-run TPS includes all 50 updates. MFU uses the same actor-time weighting as steady-state TPS.

| Recipe                                        | Text full TPS | Text steady TPS | Text MFU | Image-text full TPS | Image-text steady TPS | Image-text MFU |
| --------------------------------------------- | ------------: | --------------: | -------: | ------------------: | --------------------: | -------------: |
| PP4 baseline                                  |      3,368.23 |        3,542.91 |    7.42% |            3,138.99 |              3,372.75 |          7.08% |
| PP4 direct QSA and dynamic pairing            |      3,602.98 |        3,818.76 |    8.00% |            3,287.39 |              3,562.05 |          7.48% |
| PP4 direct QSA and dynamic pairing with MFSDP |      3,508.19 |        3,699.30 |    7.75% |            3,115.73 |              3,365.88 |          7.06% |
| PP2 full MFSDP recipe                         |      4,776.38 |        5,004.44 |   10.48% |            4,548.28 |              4,888.87 |         10.26% |

Direct QSA and dynamic pairing increased steady-state throughput by 7.79% on text and 5.61% on image-text with the distributed optimizer. Switching these matched PP4 runs to MFSDP reduced throughput by 3.13% and 5.51%, respectively. The full PP2 recipe increased throughput by 41.25% and 44.95% over the measured baselines. Mean loss over the last ten updates was 0.53926 for both text runs, and 0.43161 and 0.43177 for the image-text baseline and full recipe.

- Paired QSA reads shared K/V blocks once and preserves the selected blocks and causal tail separately for each query. Specializing on the number of sparse queries produced hundreds of kernel variants and an observed 3,125 tokens/s/GPU. Runtime row counts reduced the observed variants to two attention kernels and one merge kernel per node.
- MFSDP uses independent all-gather and reduce-scatter communicators. The recipe retains selected BF16 parameter buckets within each optimizer update with 8 GiB and 22 GiB budgets on the two pipeline stages. Unbounded retention ran out of device memory in the third update.
- Microbatch ordering uses the existing CPU process group to send one schedule within each model-parallel group. Creating additional Gloo subgroups during actor initialization stalled on the four-node release check.
- The packed GDN and ragged expert kernels handle the measured model dimensions. Expanded DeepEPv2 routing avoids a separate expert permutation and omits verified padding. The main SFT head pools supervised rows across documents and uses 8,192-row loss chunks. Expert weight-gradient fusion is limited to eligible BF16 grouped linears with FP32 gradient buffers.
- The multimodal path rebuilds images on the stage that owns the vision encoder. Stage detection follows nested MFSDP wrappers to find the encoder at any depth.

Compare workload counters alongside speed. The PP4 backend comparison uses the matched effective token and image counts per update. The full PP2 recipe uses a different packing policy and pipeline layout.

The PP4 comparison keeps global weight-gradient fusion enabled in both recipes. The image patch makes grouped BF16 experts accumulate into persistent FP32 buffers even when pipeline microbatches reverse their forward order. The shared output projection accumulates gradients from every loss chunk and MTP use before reduce-scatter. The four-GPU regression compares individual gradients and two Adam updates against explicit matrix products with dense DP4 and expert DP1.
