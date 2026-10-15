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
