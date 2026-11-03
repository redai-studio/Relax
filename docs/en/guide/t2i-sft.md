# Text-to-Image SFT and Frozen Conditioning Reuse

`QwenImageSFTAdapter` runs Qwen-Image DiT LoRA through Relax's synchronous FSDP2
SFT loop. It shares the sampler, optimizer loop, checkpointing, bounded cache and
numerical verifier with SenseNova. The existing `QwenImageAdapter` implements RL
and is not an SFT entrypoint.

The text encoder and VAE remain frozen. The recipe uses BF16 computation and
FP32 gradient reduction and optimizer masters. Full-parameter and text-encoder
LoRA training are not supported by this entrypoint.

## Data and launch

Supply a JSON array of image/caption pairs. Paths are relative to `DATA_ROOT`.
Captions should describe their target images; image-edit instructions are not
equivalent captions. Do not include `edit_image`.

```json
[
  {"image": "images/001.jpg", "prompt": "A cat sitting beside a window."}
]
```

Submit on an existing Ray cluster, using worker-visible paths:

```bash
MODEL_PATH=/shared/models/Qwen-Image \
DATA_ROOT=/shared/data/captions METADATA_PATH=/shared/data/captions/train.json \
DATASET_SIZE=128 SAVE_DIR=/shared/runs/qwen-sft EXP_NAME=qwen-r01 SAVE_INTERVAL=32 \
GPUS=8 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SIZE=1 STEPS=32 \
HEIGHT=512 WIDTH=512 RAY_NO_WAIT=1 RELAX_SKIP_CLUSTER_CLEANUP=1 \
TEXT_ENCODER_PLACEMENT=sharded \
RELAX_CONDITION_CACHE=1 RELAX_CONDITION_CACHE_MAX_BYTES=1073741824 \
RELAX_CONDITION_CACHE_RUN_DIR=/shared/runs/qwen-sft/cached-r01 \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh
```

Each rank accumulates its samples sequentially. Microbatch size must be 1 and
the effective batch must be divisible by the GPU count. Defaults are LoRA
rank/alpha 32, learning rate `1e-4`, and 200 optimizer updates of warmup.
Height and width must be multiples of 16. Relax's preprocessing applies EXIF
orientation, RGB conversion, aspect-preserving bilinear cover resize and a
fixed-size center crop. `RANDOM_CROP=1` and `CAPTION_DROPOUT` enable stochastic
preprocessing per presentation; both default to off. Returned text length is
limited to 512, while encoding retains the original full context before removing
the template prefix and truncating hidden states. Image and condition prefetch
remain disabled in this recipe to preserve augmentation ordering.

## Text encoder placement

`TEXT_ENCODER_PLACEMENT` sets `text_encoder_placement` in `--t2i-sft-config`.
Both accept `sharded` (the default) or `replicated`, independently of whether the
condition cache is enabled.

| Value | Frozen text encoder behavior |
| --- | --- |
| `sharded` | One independent FSDP group shards the encoder across ranks. Online encoding gathers its weights; a unanimous cache hit skips that encoder call and its collectives. |
| `replicated` | Every GPU keeps the complete frozen encoder resident. Encoding uses local weights, without encoder FSDP all-gathers. |

Replicated placement uses native FSDP `ignored_params`. The encoder stays
registered on the model, retaining its ownership and mutation checks; FSDP does
not shard, move, cast or reduce gradients for those parameters. The adapter
loads them onto the model device in BF16, requires frozen parameters in eval
mode, and rejects unfreezing a replicated encoder during training. This option
does not change DiT LoRA training or cache capacity.

Replication keeps full encoder weights on each GPU until the model is released,
including cache-hit steps. Sharding reduces this residency between steps, but
online encoding still needs full weights and communication buffers. In
particular, `--no-fsdp-reshard-after-forward` retains gathered weights through
backward; it does not make the encoder persistently replicated across updates.
Measure startup and training peaks before selecting a placement: replication
does not necessarily increase the online peak, but it uses more memory than a
sharded encoder on cache-hit steps. Encoder weights are separate from the
bounded feature-cache budget.

## Cache contract and measurement

By default, only deterministic frozen `prompt_embeds` are reused. The common runtime hashes
checkpoint contents; the adapter declares effective token IDs, attention mask,
template, truncation, encoder configuration, attention settings and dtype.
Image filenames do not authorize reuse. Target VAE posterior sampling, diffusion
noise and timesteps execute every time. Encoder mutation or unfreezing disables
reuse of old entries.

Optional `CACHE_VAE_POSTERIOR=1` (`cache_vae_posterior: true` in the config)
stores text embeddings and deterministic VAE posterior parameters in one bounded
entry. The key also covers the actual augmented RGB pixels, dimensions,
normalization and VAE configuration. Posterior `sample()` still runs at its
original position for every microbatch, followed by diffusion noise and timestep
sampling; sampled latents are never cached. This mode requires VAE tiling and
slicing to be disabled. Posterior parameters must use contiguous or `channels_last_3d`
layout; the cache records and restores that layout before sampling so subsequent
`randn_like` draws keep their element mapping. Other layouts are rejected.
Reuse requires the same effective image/caption pair;
identical captions across different target images are no longer reused separately.
Crops and caption dropout keep their normal execution order and key their actual outputs.

This option keeps the frozen VAE resident on each GPU using native `ignored_params`.
Use the same `CACHE_VAE_POSTERIOR` for online and cached comparisons: a change in
weight placement must not be counted as feature reuse. It defaults off and adds
resident weights and cache payload; measure both low-reuse costs and repeated epochs.

Set `RELAX_CONDITION_CACHE=0` for online encoding. The cache owns immutable
tensor snapshots within one training attempt; consumers own independent tensors.
The example allows 1 GiB of payload per node and an independent 1 GiB of write
staging, plus metadata and in-flight tensors. LRU eviction respects active leases.
Missing, corrupt or oversized entries fall back to online computation. Ranks agree
on hits before skipping sharded encoder collectives. An adapter may declare that
all conditioning modules are complete frozen replicas with no branch collectives;
after backend validation, each rank can decide hits and fallback independently.
The runtime agrees on this execution policy at initialization and keeps it fixed,
including after cache misses or weight invalidation. Completion and caught errors
close run-owned resources; dead-owner recovery reclaims entries after forced
termination.

Use a fresh log directory for every run. Set `RELAX_CONDITION_VERIFY=1`,
`RELAX_CONDITION_VERIFY_RTOL=0`, `DETERMINISTIC_ATTENTION=1` and
`LR_WARMUP_ITERS=0` to compare loss, fully accumulated and reduced per-parameter
gradients, and actual FP32 parameter deltas for an effective batch. Verification
performs extra reference work and must be separate from performance runs. Fix
data order, batch, resolution, optimizer and initialization; measure cold build,
steady complete training loops, total job time and low-reuse overhead separately.
Keep text encoder placement, gradient checkpointing and FSDP reshard settings
identical in each online/cache comparison. Where memory permits replication,
compare both placements and include the faster feasible online configuration in
the baseline, then measure caching with that same configuration. Report the
placement and its memory cost rather than attributing a placement change to
feature reuse.

## Checkpoints and export

`SAVE_INTERVAL` controls checkpoint frequency. `LOAD_DIR` restores the latest
committed `t2i_sft/iter_*`, including optimizer, scheduler, per-rank RNG and sampler
cursor. `STEPS` is the total number of updates to reach. Exact resume checks the
model/data/batch/LoRA/training contract. A warm cache is not required to resume.
Text encoder placement and posterior reuse are part of that contract; use the same
`TEXT_ENCODER_PLACEMENT` and `CACHE_VAE_POSTERIOR` when restoring a checkpoint. Both placements save the
trainable DiT LoRA weights rather than adding frozen encoder weights to the
adapter checkpoint.

The existing exporter dispatches through the checkpoint's adapter:

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/qwen-sft/qwen-r01/t2i_sft/iter_0000032 \
  --output /shared/exports/qwen-image-lora
```

The output is a Diffusers transformer LoRA directory retaining the trained FP32
masters; export requires alpha equal to rank. Use DCP checkpoints to resume
training. RL cache integration is future work: an RL adapter can declare frozen
deterministic conditions and reuse the same storage/lifecycle, without caching
candidate-specific noise, trajectories or rewards.
