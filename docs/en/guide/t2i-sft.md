# Qwen-Image Text-to-Image SFT

`QwenImageSFTAdapter` connects Qwen-Image to Relax's synchronous FSDP2 SFT loop,
reusing image SFT data cursors, optimizer updates and checkpoint support. The entry
point trains DiT LoRA while keeping the text encoder and VAE frozen in eval mode.
By default, every sample runs text encoding, VAE encoding and posterior sampling
online, then samples diffusion noise and a timestep for the flow-matching objective.

The default recipe uses BF16 computation with FP32 gradient reduction and optimizer
master weights. Full-parameter training and text-encoder LoRA are not supported by
this entry point.

## Data and launch

Prepare a JSON array of target images and captions. Image paths are relative to
`DATA_ROOT`:

```json
[
  {"image": "images/001.jpg", "prompt": "A cat sitting beside a window."}
]
```

Captions should describe the target image. The T2I data contract rejects the
image-editing `edit_image` field. Submit to an existing Ray cluster, using model,
data and output paths accessible to every worker. Replace `<ray-head>` below
with your Ray head host; `RAY_DASHBOARD_ADDRESS` is the Ray Jobs HTTP endpoint:

```bash
MODEL_PATH=/shared/models/Qwen-Image \
DATA_ROOT=/shared/data/captions METADATA_PATH=/shared/data/captions/train.json \
DATASET_SIZE=128 SAVE_DIR=/shared/runs/qwen-sft EXP_NAME=qwen-r01 SAVE_INTERVAL=32 \
GPUS=8 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SIZE=1 STEPS=32 \
HEIGHT=512 WIDTH=512 RAY_NO_WAIT=1 RELAX_SKIP_CLUSTER_CLEANUP=1 \
TEXT_ENCODER_PLACEMENT=sharded RAY_DASHBOARD_ADDRESS="http://<ray-head>:8265" \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-qwen-image-sft.sh
```

`DATASET_SIZE` must match the metadata row count. Each rank accumulates its samples
sequentially: `MICRO_BATCH_SIZE` must be 1, and the effective batch must be divisible
by the total GPU count. Defaults are LoRA rank/alpha 32, learning rate `1e-4` and
200 optimizer warmup updates; set `LR_WARMUP_ITERS` explicitly for short runs.

`HEIGHT` and `WIDTH` must be multiples of 16. Preprocessing applies EXIF orientation,
RGB conversion, aspect-preserving bilinear resizing and a center crop to the
requested size. Set `RANDOM_CROP=1` or `CAPTION_DROPOUT` for per-occurrence random
cropping or caption dropout; both are disabled by default. This entry point requires
`IMAGE_PREFETCH_DEPTH=0` so random preprocessing runs in the training thread and
preserves random-number call order.

The maximum returned text length is 512. Encoding retains the model's full template
context before removing the template prefix and truncating the output.
`GRADIENT_CHECKPOINTING` defaults to enabled; `DETERMINISTIC_ATTENTION=1` enables
deterministic settings.

The launcher passes `--t2i-sft-config` as JSON and keeps paths, including spaces,
as separate command arguments. Run the training script directly with `DRY_RUN=1`
to inspect the command without contacting Ray or creating output directories.
Submission logs go to `RUN_DIR`, which defaults to `${SAVE_DIR}/${EXP_NAME}/logs`;
use a fresh log directory for each run.

## Frozen text-encoder placement

`TEXT_ENCODER_PLACEMENT` sets `t2i_sft_config.text_encoder_placement`:

| Value | Online training behavior |
| --- | --- |
| `sharded` (default) | Shard the text encoder as a separate FSDP group and gather its weights for encoding. |
| `replicated` | Keep a complete frozen text encoder on each GPU and encode without its FSDP all-gathers. |

Replication uses native FSDP `ignored_params`. The adapter places these weights and
sets their dtype before wrapping; parameters remain registered in the model.
Unfreezing them or changing placement during a run is unsupported. Replication
keeps additional weights resident on each GPU; choose placement using measured
startup and training memory peaks. It does not change the DiT LoRA objective.

## Save, resume and export

`SAVE_INTERVAL` controls checkpoint frequency. Set `LOAD_DIR` to resume the latest
committed `t2i_sft/iter_*` checkpoint. `STEPS` is the total target update count, not
the number of additional updates. Resume restores the optimizer, scheduler,
per-rank RNG state and data cursor, and checks data, model, batch, LoRA and training
configuration. Use the same text-encoder placement when resuming.
`SAVE_CHECKPOINTS=0` disables checkpoint writes.

The existing export tool dispatches through the adapter recorded in the checkpoint:

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/qwen-sft/qwen-r01/t2i_sft/iter_0000032 \
  --output /shared/exports/qwen-image-lora
```

The result is a Diffusers-loadable transformer LoRA directory with FP32 master
weights; alpha must equal rank. Continue training from the original DCP checkpoint,
and use the exported directory for model loading.

## Optional frozen conditioning cache

Add `RELAX_CONDITION_CACHE=1` to the normal launch command above to reuse frozen
text embeddings and VAE posterior parameters. `RELAX_CONDITION_CACHE_MAX_BYTES`
sets the per-node payload budget (default 8 GiB); `RELAX_CONDITION_CACHE=0` restores
online encoding. No separate launcher or validation tool is required. Posterior
sampling remains online with the same RNG order and tensor layout; crops and
caption dropout key their actual outputs. The frozen VAE is replicated in both modes.

Keep encoder placement, checkpointing and FSDP reshard settings identical across
the comparison and on checkpoint resume. All ranks agree before skipping an
encoder; a warm cache is not required to resume. See [Frozen conditioning cache](./frozen-conditioning-cache.md)
for configuration, ownership, limits and invalidation.
