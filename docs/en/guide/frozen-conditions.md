# Frozen conditions in image SFT

Image SFT can reuse deterministic frozen conditions on each training GPU. This
is opt-in: `condition_cache_bytes=0` preserves online encoding. It uses the same
bounded LRU for SenseNova and Qwen-Image-Edit, without a cache service, shared
memory, disk artifacts or SGLang/RL dependencies.

## Supported conditions

| Adapter | Reused fields | Still computed on every sample |
| --- | --- | --- |
| SenseNova SFT | Frozen vision/understanding prefix K/V and image positional IDs | Noise initialization, target image preparation, timestep/noise sampling, generation and loss |
| Qwen-Image-Edit SFT | Frozen multimodal prompt embeddings and deterministic reference VAE latents | Target VAE posterior sampling, timestep/noise sampling, DiT and flow-matching loss |

SenseNova validates the actual parameter dependencies **after** LoRA injection.
Custom adapters touching the understanding path cannot enable its cache. Such
adapters retain prefix gradients when caching is disabled. Qwen supports DiT
LoRA and exactly one reference image; it uses Diffusers' `QwenImageEditPipeline`
(the original Qwen-Image-Edit model, not Edit-Plus). Frozen encoder/VAE parameters
remain registered inside the FSDP forward root. Full-parameter Qwen training and
SenseNova's fixed pixel tiers are outside this adapter's scope.

The Qwen objective uses uniform training timesteps and unweighted rectified-flow
MSE against noise minus sampled clean latents. Target normalization follows the
[Diffusers Qwen-Image training recipe](https://github.com/huggingface/diffusers/blob/v0.37.0/examples/dreambooth/train_dreambooth_lora_qwen_image.py);
prompt/reference conditioning and token geometry follow its
[edit pipeline](https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py).
Training images use Relax's deterministic aspect-preserving resize/center crop
to `max_pixels`, aligned to 32 pixels, rather than inference's fixed area.

## Configuration and ownership

Add these fields to `--i2i-sft-config`:

```json
{"condition_cache_bytes": 2147483648, "condition_cache_entries": 128}
```

The limits are **per rank**, in bytes and entries. Both apply; an oversized
condition is computed online and not admitted. Admission counts unique backing
tensor storage, including views. The bound excludes the active forward/backward
working set and CUDA allocator reservations, so leave training headroom. Do not
interpret a bounded cache as a bound on total GPU memory. There is no disk cache.

Both adapters also support the existing bounded CPU image prefetcher. Set
`image_prefetch_depth=4` and `image_prefetch_workers=2` in the same config to
overlap image decoding and resizing with training. The prepared sample order
and conditioning keys stay unchanged. Prefetch defaults to off; measure its
benefit for your storage and resolution, using the same setting in cache-on
and cache-off comparisons.

Each cache belongs exclusively to one loaded, immutable encoder/processor and
is created after checkpoint restoration. This ownership is the weight and
processor namespace: entries never cross model loads, resumes, ranks or jobs.
Within it, SHA-256 keys include the adapter conditioning version, output dtype,
final prompt, ordered post-preprocessing reference image contents/mode/geometry,
and target geometry. Changing a file's pixels at the same path cannot hit an old
entry. Target pixels are not dependencies of these conditions. Future stochastic
preprocessing must finish before keying; stochastic encoder outputs cannot be
registered as reusable fields.

The training stream borrows tensors read-only without copies or host transfers.
Any rank miss causes all ranks to encode, using the existing decode-error
rendezvous. This preserves nested FSDP collective order. Rank rotation or small
budgets can reduce the **effective** hit rate below local hit rates; measure it.
Errors, final completion and explicit trainer close release entries. Worker
termination releases process-owned GPU memory. Checkpoints do not store cache
contents; cache capacity may change on resume without changing the training
contract. Reusing a model after replacing frozen weights requires a fresh cache.

Future RL adapters can use the same lookup/request/admission interface after
their own preprocessing, with a cache owned by each immutable frozen encoder
version. FSDP callers must retain all-rank hit agreement when skipping an encoder
would skip collectives. Only deterministic conditions may cross candidate
evaluations; noise, sampled target latents, trajectories and rewards remain
candidate-specific. No rollout service or cross-process feature transport is
part of this SFT implementation.

## Launch and export

The existing image SFT recipe accepts `CONDITION_CACHE_BYTES` and
`CONDITION_CACHE_ENTRIES`. Its default adapter remains SenseNova. To train Qwen:

```bash
MODEL_PATH=/shared/models/Qwen-Image-Edit DATA_ROOT=/shared/data \
METADATA_PATH=/shared/data/train.json DATASET_SIZE=64 \
GPUS=2 MODE=lora CONDITIONING_IMAGES=1 MAX_PIXELS=262144 \
CONDITION_CACHE_BYTES=2147483648 CONDITION_CACHE_ENTRIES=128 \
SAVE_DIR=/shared/runs/qwen-edit STEPS=64 RAY_NO_WAIT=1 \
bash scripts/entrypoint/ray-job.sh \
  scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh \
  --model-adapter-path relax.models.qwen_image.edit_sft.QwenImageEditSFTAdapter
```

Use a separate output directory per run. All workers need the complete local
model snapshot, metadata and images. No model download happens inside training.
Use the original [image SFT guide](./i2i-sft.md) for data/resume controls.
Qwen uses the same export command with an **output directory**; the resulting
Diffusers LoRA weights load with
`pipeline.load_lora_weights(directory, weight_name="pytorch_lora_weights.safetensors")`,
including in offline mode, and preserve rank/alpha scaling. Continue training
from DCP, not the exported LoRA.

## Runtime requirements and metrics

FSDP LoRA with frozen BF16 weights and FP32 masters requires PyTorch 2.8 or
newer, with a matching torchvision build. Model snapshots must be available
locally on every training worker.

Training metrics under `condition_cache/` report lookups, local and effective
hits, encodes, evictions, oversized conditions, and resident/peak bytes. Use
effective hits to assess reuse across FSDP ranks. A one-pass dataset has no
reuse benefit; leave caching disabled for that workload.

Per-rank `i2i SFT measurement` logs record step completion, global sample count,
elapsed time, GPU memory and cache counters. For end-to-end throughput, sum
consecutive completion intervals for each rank and use the largest rank total;
count the global batch once and retain checkpoint steps. The existing
`--use-pytorch-profiler` controls expose named conditioning-encoder ranges for
investigating cache misses and remaining compute.
