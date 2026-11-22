# Qwen3.8 Flash Next on 32 Blackwell SM103 GPUs

The text and multimodal SFT modes use four eight-GPU nodes, 131,072-token context, global batch 512, and 50 optimizer updates by default. Each mode saves a resumable checkpoint at update 50 and logs to online ClearML.

Both modes use TP2, PP2, EP8 and CP1, with 26 decoder layers on the first pipeline stage and 22 on the second. Megatron FSDP uses DeepEPv2 dispatch and BF16 gradient communication.

Run the [SFT script](scripts/run-qwen38-qsa-sft-mfsdp-50step-32xgpu-blackwell.sh) with `--modality text` or `--modality multimodal` as its first argument pair. Additional arguments are passed to training. Each mode keeps its own default experiment name and checkpoint directory.

## Run

Build the repository's Blackwell `docker/Dockerfile.cu13` profile. Its Megatron tree contains the pinned Bridge and MCore sources with the `common`, `cu13`, and `blackwell` sections of `docker/patch/latest/megatron.patch` applied in that order. Use the matching Relax checkout on all nodes. Set `MEGATRON` to this patched tree and provide the HF checkpoint, matching text or image-text dataset, and writable checkpoint root via `MODEL_DIR`, `DATA_DIR`, and `SAVE_DIR`. Set `RAY_JOBS_ADDRESS` and the usual online ClearML credentials in the cluster environment.

```bash
recipe=examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-mfsdp-50step-32xgpu-blackwell.sh
modality=text  # Use multimodal for an image-text dataset.
DRY_RUN=1 bash "${recipe}" --modality "${modality}"
RELAX_SKIP_CLUSTER_CLEANUP=1 bash scripts/entrypoint/ray-job.sh "${recipe}" --modality "${modality}"
```

Set `NUM_STEPS` and `SAVE_INTERVAL` together for a shorter verification. Use a new `SAVE_DIR` or `EXP_NAME` for independent runs. Multimodal examples use the `images` field and a 128-token budget per image. The framework derives rank-side image preprocessing from train prefetch, per-rank fetch and staleness.

## Performance observations

Over 50 updates, the text recipe processed 405,940,021 effective tokens at 4,813 tokens/s/GPU. The multimodal recipe processed 288,971,498 effective tokens at 4,521 tokens/s/GPU and averaged 4.6816 images per sample. Across zero-based updates 3–49, throughput was 5,046 tokens/s/GPU for text and 4,876 for multimodal, measured using actor training time.

- Paired QSA reads shared K/V blocks once and preserves the selected blocks and causal tail separately for each query. Specializing on the number of sparse queries produced hundreds of kernel variants and an observed 3,125 tokens/s/GPU. Runtime row counts reduced the observed variants to two attention kernels and one merge kernel per node.
- MFSDP uses independent all-gather and reduce-scatter communicators. The recipe retains selected BF16 parameter buckets within each optimizer update with 8 GiB and 22 GiB budgets on the two pipeline stages. Unbounded retention ran out of device memory in the third update.
- The packed GDN and ragged expert kernels handle the measured model dimensions. Expanded DeepEPv2 routing avoids a separate expert permutation and omits verified padding. The main SFT head pools supervised rows across documents and uses 8,192-row loss chunks. Expert weight-gradient fusion is limited to eligible BF16 grouped linears with FP32 gradient buffers.
- The multimodal path rebuilds images on the stage owning the vision encoder, including nested MFSDP wrappers. An earlier depth-limited check missed this stage and failed on the first update. Checkpoint and optimizer-state experiments that failed recovery and numerical checks were excluded.
