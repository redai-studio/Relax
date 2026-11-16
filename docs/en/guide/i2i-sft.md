# Image-to-Image SFT

Image-edit supervised training uses Relax's standard `train` entrypoint, Controller,
Actor, RayTrainGroup, and FSDP2. A CPU SFT producer writes sample indices to
TransferQueue, and training workers read the images. The head node does not need
the model or raw images mounted. The model directory, metadata, and images must be
accessible to every training worker; the checkpoint output directory must be
shared across workers.

The current model adapter supports SenseNova U1.5 MoT Edit with full-parameter or
LoRA training. Inputs can contain one source conditioning image, or two
conditioning images ordered as source then glyph. The target image provides
supervision and does not count toward the conditioning-image limit. The default
1024² pixel budget is not a fixed square resolution: images retain their aspect
ratio and are aligned to the model's patch size.

## Data

`--prompt-data` points to a JSON array. Image paths are relative to `data_root`,
or can be absolute:

```json
[
  {
    "image": "target/001.png",
    "edit_image": ["source/001.png", "glyph/001.png"],
    "prompt": "替换图片中的文字"
  }
]
```

Without a glyph image, keep only the first `edit_image` entry and set
`CONDITIONING_IMAGES=1`. `DATASET_SIZE` must match the number of metadata rows.
Workers verify that the dataset hash is consistent across all ranks.

## Launch

Submit through the project entrypoint on an existing Ray cluster. Replace the
example paths with paths accessible to your training workers:

```bash
MODEL_PATH=/shared/models/SenseNova \
DATA_ROOT=/shared/data \
METADATA_PATH=/shared/data/train.json \
DATASET_SIZE=64 \
SAVE_DIR=/shared/runs/i2i-smoke \
MODE=lora CONDITIONING_IMAGES=1 STEPS=5 RAY_NO_WAIT=1 \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh
```

Use `MODE=full CONDITIONING_IMAGES=2` for full-parameter training with a glyph
conditioning image. The recipe uses BF16 computation, FP32 gradient reduction and
optimizer master weights, and a 200-optimizer-step linear warmup followed by a
constant learning rate. Its default LoRA rank and alpha are both 32, with a
learning rate of 1e-4; full-parameter text replacement uses 1e-6.

## Frozen Condition Cache

In LoRA training, the model parts that encode the conditioning images and the
condition prefix stay frozen. A sample with the same prompt and conditioning
images therefore produces the same prefix features in every epoch. The condition
cache keeps these features, the prefix K/V of every layer together with
`indexes_image`, on each training GPU. When the sample appears again, that
micro-batch reuses them and skips conditioning-image embedding and prefix
encoding. The cache is off by default; the fixed-tiers recipe exposes it through
environment variables:

```bash
MODEL_PATH=/shared/models/SenseNova \
DATA_ROOT=/shared/data \
METADATA_PATH=/shared/data/train.json \
DATASET_SIZE=64 \
SAVE_DIR=/shared/runs/i2i-cache \
MODE=lora CONDITIONING_IMAGES=1 STEPS=24 \
CONDITION_CACHE_ENABLED=1 CONDITION_CACHE_MAX_BYTES=8589934592 \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh
```

With the default 8 GPUs and a global batch of 8, the 64 samples take 8 steps per
epoch, so 24 steps train three epochs and samples repeat from the second epoch
on. When `FIXED_PIXELS_TIERS` is unset, this recipe resizes images by
`MAX_PIXELS` like the recipe above. The recipe writes the settings into
`--i2i-sft-config`; other launchers can set the same keys directly:

| Environment variable | `i2i_sft_config` key | Default | Description |
| --- | --- | --- | --- |
| `CONDITION_CACHE_ENABLED` | `condition_cache_enabled` | `0` (`false`) | Set to `1` to enable the cache |
| `CONDITION_CACHE_MAX_ENTRIES` | `condition_cache_max_entries` | `256` | Maximum cached conditions on each rank |
| `CONDITION_CACHE_MAX_BYTES` | `condition_cache_max_bytes` | `1073741824` (1 GiB) | Maximum tensor storage bytes on each rank |
| `BASELINE_STATS_DIR` | `baseline_stats_dir` | unset | Directory for per-rank encoding statistics |
| `BASELINE_WARMUP_STEPS` | `baseline_warmup_steps` | `5` | Steps excluded from the timing summary |

Both limits apply to each rank separately, and cached tensors use the same GPU
memory as training. An entry is charged the bytes of the tensor storages it
holds. When a new condition would exceed either limit, the least recently used
entries are evicted; a condition that alone exceeds `CONDITION_CACHE_MAX_BYTES`
is not cached and its micro-batch encodes it as usual.

A micro-batch reuses cached features only when every rank finds its condition
in the cache. If any rank misses, all ranks encode online, which keeps the FSDP
communication order identical on every rank. As a result, the first epoch has no
reuse, and the reuse rate falls when each rank sees more distinct conditions
than its budget can hold or when more ranks must hit at the same time. The
cache key includes the encoder version, the decoded RGB pixels and their order,
the prompt, preprocessing and target size, the tokenizer, the dtype, and the
attention implementation. A cached value is therefore reused only for the same
inputs and encoder weights.

Full-parameter training (`MODE=full`) updates the encoders, so conditions are
always encoded online even when the cache is enabled. Loading a checkpoint
clears the cache. Cached features are not written to checkpoints, the cache
starts empty after resume, and the cache and statistics settings can change
between the original run and a resumed run.

At the end of training, each rank logs a line starting with
`i2i condition cache rank=<rank> stats=`. `effective_reuses` counts reused
micro-batches, `local_hits` counts lookups this rank found, and
`group_forced_online` counts local hits encoded online because another rank
missed. A large `evictions` count together with a `peak_charged_bytes` near the
limit suggests a larger `CONDITION_CACHE_MAX_BYTES`.

To estimate the budget, run a short job with the cache disabled and
`BASELINE_STATS_DIR` set to a directory shared by all workers. Each rank writes
`rank_<rank>_steps.jsonl` and `rank_<rank>_summary.json` there. Combine them
after the job:

```bash
python -m relax.engine.sft.condition_tracker /shared/runs/i2i-baseline
```

The command writes `baseline_summary.json`. In it,
`feature_tensor_bytes.past_key_values_bytes` and
`feature_tensor_bytes.indexes_image_bytes` give the size of one cached condition,
and the `share_pct` values of `stage_timings_ms.ref_vision_encode_ms` and
`stage_timings_ms.prefix_encode_ms` show how much of a step the cache can skip.
The statistics time stages with CUDA events and wait for them at the end of each
step, so leave `BASELINE_STATS_DIR` unset for regular training.

## Resume and Constraints

Set `LOAD_DIR` to the previous output directory. `STEPS` is the total step count to
reach, not the number of additional steps. Resume loads the latest committed
`i2i_sft/iter_*` checkpoint, including optimizer and scheduler state, per-rank RNG
state, and completed steps. The Controller resets the producer to the Actor's
restored step, so prefetch does not change the resumed sample order. Changes to
the dataset, GPU count, batch size, LoRA settings, or optimizer configuration are
rejected for exact resume.

Each optimizer step consumes one global batch. The microbatch size must be 1,
and the global batch size must be divisible by the GPU count. At the end of each
epoch, samples from the beginning are repeated to fill the final global batch.
This path supports synchronous FSDP data parallelism only. Text packing, text
PPL evaluation or prediction, RL rollout/reward, and TP/PP/CP are not part of it.

The SenseNova adapter includes the native model and preprocessing; it does not
depend on the DiffSynth runtime. Adapted source files retain the upstream copyright
and source attribution in their headers; the Apache-2.0 license is provided in the
repository root `LICENSE`. Model weights and tokenizers are supplied separately
and remain subject to their own licenses.

## Export

Export a committed checkpoint in a separate CPU process:

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/i2i-smoke/i2i_sft/iter_0000005 \
  --output /shared/exports/adapter.safetensors
```

LoRA exports use parameter names relative to the DiT and BF16 weights. Export
currently requires alpha to equal rank. For full-parameter checkpoints, set
`--output` to a new directory; export writes BF16 model shards and copies the
original model's tokenizer configuration. Exports do not include optimizer state.
Use the original DCP checkpoint to continue training.
