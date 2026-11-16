# SenseNova image-edit RL with ART

ART reuses Relax's SenseNova SFT model implementation and the FSDP2 actor, with
SGLang generating cached ODE trajectories. The supported profile is synchronous
colocation: one node, eight GPUs, one SGLang replica per GPU, generation-branch
full tuning, CFG=1 and no thinking. It is not full-model tuning of the
understanding branch. GRPO and image SFT retain their existing entry points.

## Environment and inputs

Use the SGLang **v0.5.21** environment and apply
`docker/patch/sglang/v0.5.21.patch` to its source before installation. Restart
existing engines after updating the patch. The default older SGLang image is
not sufficient. CUDA, FlashAttention, Ray, PyTorch and Transformers must match
the supported training image; do not upgrade only one package in place.

The optional TextPecker example additionally needs `openai`, `lpips` and offline
AlexNet weights at `$TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth`
(or the equivalent `torch.hub.get_dir()/checkpoints` location). Start an
OpenAI-compatible TextPecker service separately and supply its `/v1` endpoint
and served model name. OCR runs remotely; LPIPS runs on the training host CPU.

Supply a SenseNova-U1.5 checkpoint visible to every worker and a JSONL prompt
manifest. Each row has `prompt` plus `metadata.images` (ordered condition image
paths), `metadata.gt_path` (a separate ground-truth image) and
`metadata.target_texts` (explicit OCR targets). Use absolute worker-visible
paths. Ground truth must not be one of the condition images.

```json
{"prompt":"Replace the sign text with NEW","metadata":{"images":["/data/source.png"],"gt_path":"/data/target.png","target_texts":["NEW"]}}
```

## Launch

Set your model, prompt, service and output paths explicitly. Output directories
must be empty and outside the code, input and resume directories. Cache budgets
are limits, not preallocations; size them for the actual resolution and host RAM.
The following is a 1024-pixel-budget example with 64 candidates per acquisition:

```bash
export MODEL_PATH=/models/SenseNova-U1.5
export PROMPT_SET=/data/train.jsonl
export REWARD_ENDPOINT=http://reward-host:8848/v1
export REWARD_MODEL_NAME=TextPecker
export ARTIFACT_ROOT=/runs/art-example/artifacts
export SAVE_DIR=/runs/art-example/checkpoints
export N_PROMPTS=8 N_SAMPLES=8 RESOLUTION=1024 STEPS=20
export NUM_ROLLOUT=100 SAVE_INTERVAL=50
export ART_TRAIN_STEP_INDICES='[0,10,19]'
export ART_MAX_CACHE_BYTES=34359738368 ART_MAX_REPLAY_BYTES=68719476736
export ART_MIN_FREE_BYTES=751619276800
export ART_GROUP_OVERLAP=1 ART_OVERLAP_PENDING_GROUPS=2

# Read-only validation: does not initialize Ray or CUDA or submit a job.
bash scripts/training/diffusion/run-sensenova-i2i-art-full-8xgpu.sh --mode preflight

# Use an otherwise idle, isolated Ray cluster; configure its addresses/runtime.
export ART_EXCLUSIVE_RAY=1 RELAX_ISOLATED_JOB=1
export RAY_ADDRESS=ray-head:6379
export RAY_DASHBOARD_ADDRESS=http://ray-head:8265
RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh \
  scripts/training/diffusion/run-sensenova-i2i-art-full-8xgpu.sh --mode submit
```

`--mode config` prints the generated training arguments without loading models.
For resume, set `LOAD_DIR` to the previous checkpoint root, choose fresh output
directories and set `NUM_ROLLOUT` above the restored cursor. Only completed
two-window acquisitions are resumable. Per-rank RNG, optimizer/LR counters,
model and dataset cursor must belong to the same acquisition.

## Algorithm and scheduling boundaries

- Every complete K-candidate condition group is scored before computing terminal
  covariance. Raw weighted scalar rewards are not GRPO-normalized.
- K must be even. Two disjoint windows each contain K/2 candidates per prompt;
  the global candidate/time denominator includes masked rows.
- The actor makes two AdamW updates and publishes the live policy once. There is
  no EMA, re-noising, old-policy forward or log-probability objective.
- Original FP32 timesteps, sigmas, negative Euler increments and old velocities
  are replayed alongside bit-preserved FP16/BF16 states. A failed reward skips
  the whole acquisition. A partially failed update requires checkpoint recovery.
- `ART_GROUP_OVERLAP=1` overlaps bounded CPU reward processing with later
  generation. `ART_GENERATION_PREFETCH=1` additionally overlaps one next
  generation with response transfer; it requires overlap, at least two pending
  groups and exactly eight samples/engines. It defaults off.
- Lossless mask packing, trajectory transport and copy reductions are automatic
  on the patched runtime. Memory preparation is agreed across all training ranks
  before restoring inference weights.

The CPU suite covers mathematics, orchestration, transport and DCP/RNG recovery.
GPU end-to-end quality, memory and throughput must be revalidated for the exact
checkpoint, runtime image and reward service used in a deployment.
