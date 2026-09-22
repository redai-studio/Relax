# Diffusion Generative RL

Relax can post-train **diffusion / flow-matching generation models** with RL, not only autoregressive LLMs. This path runs [FlowGRPO](https://arxiv.org/abs/2505.05470)-style policy optimization over a denoising trajectory: the rollout engine samples images with a stochastic (SDE) sampler and returns the trajectory, a reward model scores the generated media, and the actor replays a subset of the denoising steps to compute a clipped PPO objective.

The training backend is **FSDP2** (`--train-backend fsdp`) instead of Megatron, and the rollout engine is **SGLang's native diffusion server** instead of the text `srt` engine. Everything else — the Ray Serve controller, TransferQueue data plane, colocate GPU time-sharing, metrics and checkpointing — is the same Relax machinery described in [Architecture](./architecture.md).

## Scope

::: warning Release scope
The generative path is intentionally narrow. The following are **enforced at launch preflight** (`validate_generative_config` in `relax/backends/fsdp/arguments.py`) and abort the job with a collected error list; see [Preflight validation](#preflight-validation) for the complete set.

- **Text-to-image only.** `GENERATION_TASKS` is `("t2i",)`. Image-edit and video / audio-video are not on this branch.
- **Synchronous colocate only.** `--colocate` is required; `--fully-async` and `--hybrid` are rejected.
- **Full fine-tune or LoRA.** `--fsdp-trainable-mode` is `full` or `lora`, and it must agree with `--lora-rank` (see [LoRA](#lora)).
- **No KL-to-reference.** FlowGRPO here has no reference model, so a nonzero `--kl-coef` or `--use-kl-loss` is rejected rather than silently ignored.
- **No CFG.** `guidance_scale` must be `1.0`; the actor replays a single positive-conditioned forward, so a CFG rollout would train against a policy it never sampled from.
- **No trained SDE step 0** under `sde_type="sde"` (see [Trained SDE steps](#trained-sde-steps)).
  :::

### Tasks and adapters

`--generation-task` accepts `t2i`. Which model you run is decided by the adapter you point `--model-adapter-path` at:

| Adapter | Dotpath | Declared tasks | Status |
|---|---|---|---|
| Qwen-Image | `relax.models.qwen_image.adapter.QwenImageAdapter` | `t2i` | **Validated end-to-end**, with two ready-to-run launch scripts |

::: tip Bringing your own model
The runtime depends only on the `GenerativeModelAdapter` protocol in `relax/models/generative.py` (`load_train_model`, `build_rollout_request`, `validate_rollout_response`, `pack_trajectory`, `replay_transition`, `artifact_tracks`, `weight_name_map`). A new family is a new dotpath — plus its task string added to `GENERATION_TASKS`, so the vocabulary can never advertise a task with no working replay path.
:::

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    Ray cluster — 8 GPUs, synchronous colocate             │
│                                                                           │
│  ┌──────────────────────────┐            ┌───────────────────────────┐    │
│  │  FSDPTrainRayActor       │  weight    │  SGLangNativeGeneration   │    │
│  │  relax/backends/fsdp     │  sync      │  Engine (one per GPU)     │    │
│  │                          │  (bf16,    │                           │    │
│  │  transformer  (FSDP2)    │──buckets,─►│  DiT + VAE + text encoder │    │
│  │  AdamW (offloaded)       │  CUDA IPC) │  POST /rollout/generate   │    │
│  └──────────┬───────────────┘            └─────────────┬─────────────┘    │
│             │                                          │                  │
│             │                          trajectory +    │                  │
│             │                          decoded media   ▼                  │
│             │                          ┌──────────────────────────────┐   │
│             │                          │ artifact_root/<task>/        │   │
│             │                          │   rollout_*/group_*.safeten. │   │
│             │                          │   rollout_*/..._sample_*.png │   │
│             │                          └─────────────┬────────────────┘   │
│             │                                        │                    │
│             │                                        ▼                    │
│             │                          ┌──────────────────────────────┐   │
│             │                          │ Reward scorer, e.g. PickScore│   │
│             │                          │ group-normalized advantages  │   │
│             │                          └─────────────┬────────────────┘   │
│             │        TransferQueue train rows        │                    │
│             └────────────────────────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### One training step

1. **Rollout** — `relax.engine.rollout.native_generation.generate_rollout` resolves the rollout's trained SDE steps once (so every candidate and every engine agree on one set), reaps stale artifact directories, then dispatches a wave of prompt groups across all engines. Every candidate in a group gets a distinct `sample_id`, which drives its initial latent and per-step noise, so the group has real intra-group reward variance.
2. **Trajectory sidecar** — the engine returns the DiT trajectory latents, timesteps, per-step rollout log-probs and the frozen conditioning (`denoising_env`), echoing the requested `height` / `width` so the trainer can rebuild the true latent grid. The adapter packs them and the driver writes one safetensors sidecar per group under `--artifact-root`; the decoded image is written next to it as a PNG. Large tensors never enter the TransferQueue.
3. **Reward** — `relax.engine.rewards.generative.post_process` (wired via `--custom-reward-post-process-path`) fires once per rollout, batch-scores the media through the configured scorer, normalizes each reward component **within its prompt group**, and weight-combines the components into a single advantage.
4. **Update** — the actor rehydrates the sidecars, splits its local shard into `--num-updates-per-batch` disjoint updates, replays the trained SDE steps through the transformer, recomputes the Flow-SDE transition log-prob, and applies the clipped PPO objective from `relax.models.flow_grpo`.
5. **Weight sync** — the transformer is gathered and streamed to every engine over CUDA IPC (rank *j* → engine *j*, same physical GPU, matched by device UUID), verified against a manifest, committed, and then the actor sleeps again so the engine owns the card for the next rollout.

The FlowGRPO math (`flow_sde_transition_moments`, `flow_sde_log_prob`, `replay_transition_logp`, `grpo_clip_loss`, `normalize_grouped`) lives in `relax/models/flow_grpo.py` as pure `torch` with no framework imports, and is the single source of truth shared by the rollout engine and the actor replay.

## Prerequisites

### Docker image

The generative path runs in the standard Relax training image. That image starts from `lmsysorg/sglang:v0.5.15.post1-cu129`, applies `docker/patch/sglang/v0.5.15.post1.patch`, and installs the diffusion runtime dependencies from `requirements.txt`.

```bash
docker build -f docker/Dockerfile -t relax:latest .
```

The image includes `diffusers>=0.37.0`, `imageio[ffmpeg]`, `soundfile`, and `peft>=0.20.0,<0.21.0`; no separate overlay image is required.

::: tip What the patch adds
Rollout itself is upstream: SGLang's diffusion server already exposes `POST /rollout/generate` returning the trajectory, per-step log-probs and frozen conditions. The combined Relax SGLang patch adds the **in-memory diffusion weight-update path** (`/update_weights_from_tensor`), the LoRA adapter endpoint (`/set_lora_from_tensor`), and a guard that rejects a full-weight tensor sync after SGLang has converted the DiT to LoRA layers. The patch also wires the driver's sigmas, per-step seeds and `x_T` recipe through the SGLang request path, and includes the Qwen-Image text truncation window plus the `rollout_sde_type="dance"` log-prob path. Without the weight-update path the only option is a disk reload per step; without the consistency pieces replay log-probs will not match the rollout trajectory. See `examples/diffusion/README.md` for the patch layout, hash and verification commands.

`SGLangNativeGenerationEngine` now performs a read-only static contract check before launching the server to confirm that `docker/patch/sglang/v0.5.15.post1.patch` has been applied; it no longer installs monkey patches inside worker processes.
:::

### Models

```bash
# Policy model
hf download Qwen/Qwen-Image --local-dir ${MODEL_DIR}/Qwen-Image

# Reward model (PickScore v1, a fine-tuned CLIP-H)
hf download yuvalkirstain/PickScore_v1 --local-dir ${MODEL_DIR}/PickScore_v1
```

`QwenImageAdapter.load_train_model` loads only the `transformer` subfolder with `diffusers.QwenImageTransformer2DModel` — the VAE and text encoder are loaded frozen by the rollout engine, never by the FSDP actor.

## Data preparation

Native generation uses a **unified JSONL** schema, one JSON object per line:

```json
{"prompt": "a red panda in a teacup", "metadata": {"task": "t2i", "sample_id": "pickapic_0000001"}}
```

Three scripts in `examples/diffusion/` build and gate this file. The commands below produce exactly the filenames the launch scripts default to.

### 1. Convert a public dataset

```bash
python3 examples/diffusion/prepare_data.py \
  --source pickapic \
  --input /path/to/pick_a_pic_prompts \
  --task t2i \
  --output ${DATA_DIR}/raw/t2i/pickapic.jsonl
```

`--source` selects a converter from `CONVERTERS`: `pickapic`, or `prompts` (a generic parquet `prompt`/`caption` column, a JSONL with a `prompt` field, or one prompt per line).

### 2. Dedup, filter and split

```bash
python3 examples/diffusion/curate_data.py \
  --input ${DATA_DIR}/raw/t2i/pickapic.jsonl \
  --out-dir ${DATA_DIR}/processed/t2i \
  --prefix pickapic_ --min-prompt-words 6 \
  --eval-size 2048 --eval-subset-sizes 64 256 --no-check-media
```

This writes `pickapic_train.jsonl`, `pickapic_eval.jsonl` and the `pickapic_eval64.jsonl` / `pickapic_eval256.jsonl` subsets — the exact files `PROMPT_SET` and `EVAL_SET` default to in both launch scripts.

`--min-prompt-words 6` and the 2048-prompt holdout are the Pick-a-Pic alignment preset used by the launch scripts: unfiltered one- and two-word captions score with almost no intra-group variance and starve GRPO of signal. Every transform is deterministic — prompts are deduped by a stable content hash over prompt + sorted media paths, records with missing media are dropped (skip that check with `--no-check-media`), and the train/eval assignment comes from hashing `sample_id`, never RNG — so a rerun reproduces the identical partition. Without `--prefix` / `--eval-size` / `--eval-subset-sizes` you get plain `train.jsonl` / `eval.jsonl` and an `--eval-fraction`-based split instead.

### 3. Validate before burning GPUs

```bash
python3 examples/diffusion/inspect_data.py \
  --train ${DATA_DIR}/processed/t2i/pickapic_train.jsonl \
  --eval  ${DATA_DIR}/processed/t2i/pickapic_eval256.jsonl \
  --decode-media
```

Checks required keys, a valid `metadata.task`, `<image>` / `<video>` placeholder ↔ media consistency, and train/eval split leakage. `--decode-media` additionally opens every referenced file to confirm it decodes.

## Quick Start

The validated recipe is Qwen-Image T2I **LoRA** on 8 GPUs, in adapter-sync mode:

```bash
cd /path/to/Relax
export EXP_DIR=/your/shared/fs/native-generation   # models/, data/, runs/ live here

bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
```

The script expects `${EXP_DIR}/models/Qwen-Image`, `${EXP_DIR}/models/PickScore_v1` and the curated prompt files under `${EXP_DIR}/data/processed/t2i/`; override `MODEL_DIR`, `DATA_DIR`, `ARTIFACT_ROOT`, `SAVE_DIR` or `EXP_DIR` to relocate them. It submits a Ray job with:

```bash
--resource '{"actor": [1, 8], "rollout": [1, 8]}'   # colocate: same 8 GPUs for both roles
--colocate
--train-backend fsdp
--generation-task t2i
--model-adapter-path      relax.models.qwen_image.adapter.QwenImageAdapter
--rollout-engine-class-path relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine
--rollout-function-path   relax.engine.rollout.native_generation.generate_rollout
--custom-convert-samples-to-train-data-path relax.engine.rollout.native_generation.convert_samples_to_train_data
--custom-reward-post-process-path           relax.engine.rewards.generative.post_process
```

The full flag list for both shipped recipes, with the reasoning behind each value, is in [Reference configuration](#reference-configuration). The geometry that matters most:

| Setting | LoRA (validated) | Full FT | Note |
|---|---|---|---|
| `--rollout-batch-size` × `--n-samples-per-prompt` | 32 × 8 = 256 | 8 × 8 = 64 | The GRPO group geometry; `--global-batch-size` is their product |
| `--num-updates-per-batch` | 2 | 2 | Splits the batch into 2 **disjoint** optimizer updates — see the warning below |
| `--sampling-config` | 384×384, 12 steps, `eta=0.7`, 3 trained SDE steps redrawn per rollout from `[1,2,3,4,5]` | same | 12 denoising steps are generated, only 3 are trained |
| `--generative-advantage-std-mode` | `group` | `group` | The default uses canonical Relax/Text GRPO per-group std; set `batch` explicitly only when reproducing the experimental reference-diffusion scale |
| `--lr` | `3e-4` | `3e-5` | An adapter starting from `B = 0` needs ~10× the full-FT rate |
| `--eps-clip` | `1e-4` | `1e-4` | FlowGRPO uses a tiny PPO clip range, not the text-RL `0.2` |
| `--reward-runtime` | `colocate` | `colocate` | PickScore scores in-process on the rollout GPU while the actor is offloaded |

::: warning `--num-updates-per-batch` is a split, not a replay
The rollout batch is partitioned into N **disjoint** equal-size optimizer updates; every sample contributes to exactly one step. The optimizer-step boundary is therefore `global_batch_size / num_updates_per_batch`, not `global_batch_size`.

With `--num-updates-per-batch 1` the importance ratio is always exactly 1, so with group-centered advantages the clipped loss is 0 by construction and clipping never engages. Use `2` or more.
:::

### Launch Scripts

Use the checked-in launch scripts under `scripts/training/diffusion/`. They pass the validated `train.py` arguments directly and are the source of truth for runs on this branch:

```bash
export EXP_DIR=...

# Full fine-tune reference
bash scripts/training/diffusion/run-qwen-image-t2i-8xgpu.sh

# Validated LoRA recipe
bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
```

::: warning
The YAML profiles under `examples/diffusion/` are reference config fragments, not the launch path. Start from the shipped launch scripts in `scripts/training/diffusion/`, which carry the settings that actually ran on 8×96 GB.
:::

## Configuration Reference

Complete reference for the arguments that are active when `--train-backend fsdp` is selected. These flags are registered by `add_generative_arguments` / `add_fsdp_arguments` in `relax/backends/fsdp/arguments.py` and wired into the main parser in `relax/utils/arguments.py`. For the shared non-generative flags see [Configuration](./configuration.md).

::: tip Defaults are inert
Every flag below defaults to a value that leaves the standard Megatron token-RL path untouched, so adding the generative group to the parser has no effect until you pass `--train-backend fsdp`.
:::

### Backend selection

| Flag | Type | Default | Description |
|---|---|---|---|
| `--train-backend` | `megatron` \| `fsdp` | `megatron` | Selects the training actor. `fsdp` activates `FSDPTrainRayActor` and the generative preflight validation. |

### Task and model

| Flag | Type | Default | Description |
|---|---|---|---|
| `--generation-task` | `t2i` | `None` | The generation task. **Required** for `fsdp`. Also names the checkpoint and artifact subdirectory. |
| `--model-path` | str | `None` | HF-format model directory handed to the adapter's `load_train_model`. **Required** for `fsdp`. |
| `--model-revision` | str | `None` | Optional revision. Contributes to the base-model hash and is forwarded to the rollout engine. |
| `--model-adapter-path` | str | `None` | Dotpath to a `GenerativeModelAdapter` implementation. **Required** for `fsdp`. |

`GENERATION_TASKS` is `("t2i",)`. Image-edit and video / audio-video, plus their Wan and LTX adapters, are not on this branch; a caller who brings their own adapter via `--model-adapter-path` must add its task to `GENERATION_TASKS` too, so the vocabulary can never advertise a task with no working replay path.

Shipped adapter: `relax.models.qwen_image.adapter.QwenImageAdapter` (`family = "qwen_image"`, `supported_tasks = ("t2i",)`).

### Rollout wiring

The generative rollout replaces four pluggable hooks. Three of them are the framework-level `--custom-*` / `--rollout-function-path` flags; the fourth is generative-specific.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rollout-engine-class-path` | str | `None` | Dotpath to the rollout engine class. Empty means the default text `SGLangEngine`. Set to `relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine`. |
| `--rollout-function-path` | str | — | Set to `relax.engine.rollout.native_generation.generate_rollout`. |
| `--custom-convert-samples-to-train-data-path` | str | — | Set to `relax.engine.rollout.native_generation.convert_samples_to_train_data`. |
| `--custom-reward-post-process-path` | str | — | Set to `relax.engine.rewards.generative.post_process`. |
| `--rollout-num-gpus-per-engine` | int | `1` | GPUs per diffusion server. Must be ≥ 1, and the actor world size must be divisible by it — each engine's GPUs must tile the FSDP ranks for the CUDA-IPC weight sync. |

::: tip SGLang static patch check
`SGLangNativeGenerationEngine` checks the current SGLang source before launching the diffusion server: driver sampling fields must survive the request path, driver sigmas must not be shifted a second time, Qwen-Image must use the 512-token text window, per-step seeds / rollout variance noise / driver `x_T` support must exist, and `rollout_sde_type="dance"` must be accepted. A silent miss would train on log-probs that do not match the recorded trajectory, so a failed check aborts startup and points back to `docker/patch/sglang/v0.5.15.post1.patch`.
:::

### Artifacts

| Flag | Type | Default | Description |
|---|---|---|---|
| `--artifact-root` | str | `None` | Root directory for media and trajectory sidecars. Layout: `<root>/<task>/rollout_<id>/group_<idx>.safetensors` plus `..._sample_<slot>.png`; evaluation writes under `<root>/eval/<task>/`. Must be readable by the rollout driver **and** every trainer rank — the driver writes each sidecar from one process and all FSDP ranks read it back. |
| `--artifact-retention-rollouts` | int | `2` | How many rollouts of sidecars + generated media to keep. Older rollout directories are reaped at the start of each rollout by `prune_stale_artifacts`. `<= 0` keeps everything, which grows without bound (one safetensors sidecar per group plus one image per candidate, every step). |

### Sampling

| Flag | Type | Default | Description |
|---|---|---|---|
| `--sampling-config` | JSON dict | `None` | Per-task sampling geometry, parsed with `json.loads`. Read by the adapter, the rollout driver and the actor replay. |
| `--generation-seed` | int | `1234` | Base seed. See the seeding note under `driver_xt` below. |

#### `--sampling-config` keys

| Key | Default | Read by | Description |
|---|---|---|---|
| `height` / `width` | `384` / `384` | adapter | Output resolution. The engine echoes both back on the response, which is the only source of the true latent grid — a packed sequence length cannot recover a non-square one. |
| `num_inference_steps` | `12` | adapter | Denoising steps used for **generation**. |
| `guidance_scale` | `1.0` | adapter | **Must be `1.0`.** Anything else raises: `replay_transition` runs a single positive-conditioned forward, so a CFG rollout would train against a different policy than it sampled from. Two-forward CFG replay is not implemented. |
| `eta` | `0.7` (rollout) | adapter, actor | SDE noise level, sent as `rollout_noise_level`. |
| `sigma_max` | `0.99` | actor | Clamp used by the replay when `sigma == 1`. Leave it alone; see the SDE step-0 note below. |
| `sde_type` | `"sde"` | adapter, actor | `"sde"` is the FlowGRPO coefficient `sqrt(sigma / (1 - sigma)) * eta`; `"dance"` is the DanceGRPO constant `eta`. |
| `driver_sigmas` | `true` | adapter | Send the driver's own FlowMatch sigma schedule with the request instead of letting the server recompute it. |
| `driver_xt` | `true` | adapter, driver | The driver supplies the initial latent `x_T` recipe (`initial_noise_group_ids` / `initial_noise_latent_shape` / `initial_noise_seed` / `denoise_seeds`). Under `driver_xt` the per-request engine seed stays constant and candidate diversity comes from the per-candidate `sample_id`; with `driver_xt: false` the driver falls back to `generation_seed + group_index * group_size + slot` as the engine seed. |
| `init_noise_latent_shape` | derived | adapter | Override for the `x_T` latent shape; defaults to `[16, *latent_grid(height, width)]`. |
| `sample_id_mode` | `"metadata"` | driver | How a candidate's stable id is built. `metadata` uses `prompt:<prompt_id-or-sample_id>:sample:<slot>` (falling back to the group index); `positional` / `group_index` use `prompt:<group_index>:sample:<slot>`. Also settable via `RELAX_NATIVE_GENERATION_SAMPLE_ID_MODE`. |

::: warning `eta` has two different defaults
The adapter defaults `eta` to `0.7` when building the rollout request, but the actor's replay defaults it to `1.0`. Always set `eta` explicitly in `--sampling-config` — both shipped recipes do — otherwise the replayed Gaussian is not the one the sampler drew from.
:::

#### Trained SDE steps

The **trained** SDE step subset is resolved by `relax.models.generative.resolve_sde_indices`. It does double duty: these indices select where SDE noise is injected during generation *and* which steps are replayed for the gradient. Resolution order:

| Key | Default | Description |
|---|---|---|
| `sde_resample_per_rollout` | `false` | **Highest priority.** When true and a `rollout_id` is available, draw `num_sde_steps` indices from the pool with `numpy.random.default_rng(rollout_id)` — reproducible and identical on every engine and rank without any communication. Both shipped scripts enable this, so it overrides `sde_indices` during training. |
| `sde_pool` | the fraction window | Candidate steps for that per-rollout draw. |
| `sde_indices` | `None` | Explicit list, sorted and de-duplicated. Used verbatim when there is no per-rollout resample — which is the case for held-out evaluation (`rollout_id=None`), so this is the deterministic fallback that keeps the eval number comparable across steps. |
| `num_sde_steps` | `0` | Number of steps to train, picked stride-spaced (`floor(k * len(window) / num_sde)`) inside the fraction window. |
| `sde_timestep_fraction` | `[0.0, 1.0]` | `[lo, hi]` normalized window of the schedule. |

With no SDE configuration at all the resolved list is empty and the pipeline fails when packing the trajectory. Generation always runs all `num_inference_steps`; only the resolved subset is replayed and trained, which is what keeps the replay within a card's memory budget.

::: danger SDE step 0 is rejected under `sde_type="sde"`
At step 0 the schedule has `sigma == 1`, so the diffusion coefficient `sqrt(sigma / (1 - sigma)) * eta` is singular and only stays finite because of an arbitrary `sigma_max = 0.99` clamp — which is *not* what the sampler used. Measured against the SGLang diffusion engine, `std_dev_t` came out 7.0 vs ~2.92, which flips the sign of the `sample` coefficient in `prev_sample_mean` and inflates the variance 2.4x; every other step agrees to 1e-4. `_validate_sde_schedule` therefore rejects a trained step 0 at preflight, checking every route by which it can become trainable: an explicit `sde_indices`, the `sde_pool` a per-rollout resample draws from, and the derived `sde_timestep_fraction` window.

Use `sde_indices [1,3,5]` with `sde_pool [1,2,3,4,5]`, or a `sde_timestep_fraction` starting above `1/num_inference_steps`. Note that `sde_timestep_fraction: [0.0, 0.5]` over 12 steps resolves to `[0, 2, 4]`, which *would* train step 0. `sde_type="dance"` uses a constant coefficient with no division by `1 - sigma` and is unaffected.
:::

### FSDP2 training

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fsdp-trainable-mode` | `full` \| `lora` | `full` | Must agree with `--lora-rank`: `lora` requires `> 0`, `full` requires `0`. |
| `--fsdp-trainable-attr` | str | `transformer` | Attribute on the model bundle holding the trainable module. Everything else is frozen and lives in the rollout engine. |
| `--fsdp-param-dtype` | `bf16` \| `fp32` | `bf16` | FSDP2 mixed-precision parameter dtype (the all-gathered compute copy). |
| `--fsdp-reduce-dtype` | `bf16` \| `fp32` | `fp32` | Gradient reduction dtype. |
| `--fsdp-master-dtype` | `fp32` \| `bf16` | unset | Storage dtype for the **trainable** parameters only, i.e. the optimizer's master copy. Unset keeps the loaded dtype. LoRA runs should set `fp32`; the forward is unaffected. |
| `--no-fsdp-reshard-after-forward` | flag | resharding **on** | Opt out of resharding parameters after forward (trade memory for speed). There is no positive form — a `--fsdp-reshard-after-forward` would be a no-op. |
| `--fsdp-activation-checkpointing` | flag | `False` | Recompute each transformer block in backward. Effectively required: it collapses the SDE-replay activation cost to roughly the block inputs. |
| `--fsdp-cpu-offload` | flag | `False` | FSDP2 CPU parameter offload. See the warning below. Rejected under `--fsdp-trainable-mode lora`. |
| `--fsdp-load-wave-size` | int | `0` | Gate base-model loading to N ranks at a time with a barrier between waves, bounding peak host RAM and model-directory read contention at launch. `0` means all ranks load at once. |
| `--fsdp-lr-scheduler` | `constant` \| `linear` \| `cosine` | `constant` | Post-warmup learning-rate shape. See [Learning rate](#learning-rate). |
| `--fsdp-debug-fingerprint` | flag | `False` | Log an abs-sum fingerprint of the trainable weights before and after each step to detect no-op optimizer steps, and emit the `[logp parity]` line (replayed π_old anchor vs the engine's own sampling log-prob). Off by default — it is a full-parameter reduction plus a host sync every step, and it forces the explicit-anchor replay path. |

The optimizer is always `torch.optim.AdamW`; there is no flag to change it. Its hyperparameters come from the shared `--lr` / `--weight-decay` / `--adam-beta1` / `--adam-beta2` / `--adam-eps` flags via `_adamw_kwargs`.

::: warning `--fsdp-cpu-offload`
Enabling FSDP2 CPU offload makes the FSDP collectives on the offloaded DTensors (`clip_grad_norm_`) fail with *"No backend type associated with device type cpu"*. Both shipped Qwen-Image recipes leave it off and rely on `--offload-train` / `--offload-rollout` (the Relax colocate time-share) plus a small trained SDE subset instead.
:::

#### Colocate offload

These are framework-level flags, but the generative path depends on them:

| Flag | Description |
|---|---|
| `--colocate` | **Required.** Actor and rollout time-share the same GPUs. |
| `--offload-train` | Release the actor's parameter/buffer device storage and move the AdamW state to CPU during rollout; wake it for training and weight sync. `offload_module_to_cpu` resizes the DTensor's *local shard storage* to 0 and keeps a pinned host copy — reassigning `param.data` would have been a no-op under `fully_shard`, freeing zero bytes. |
| `--offload-rollout` | Offload the diffusion engine's modules during training. Weight sync onloads only the engine transformer (`tags=[WEIGHTS]`), then the engine is fully reloaded before the next rollout. A failed `release_memory_occupation` / `resume_memory_occupation` is **fatal**, not a warning: a swallowed failure leaves the pipeline resident and the colocated actor OOMs many steps later with nothing pointing back at the cause. |

#### Learning rate

`lr_at_step` computes the rate for each optimizer step (0-based):

- Warmup is linear over `--lr-warmup-iters` steps.
- The post-warmup shape is `--fsdp-lr-scheduler`; `constant` (the default) preserves the validated behaviour and ignores the decay settings.
- `linear` / `cosine` decay from `--lr` toward `--min-lr` over `--lr-decay-iters`.
- **All of these are counted in optimizer steps, not rollouts** — the actor takes `--num-updates-per-batch` steps per rollout.
- The step count round-trips through `trainer_state.json` as `lr_scheduler_steps`, so warmup does not restart on resume.
- Megatron's `--lr-decay-style` has no effect here; preflight warns when it is set alongside the default `constant` scheduler.

### FlowGRPO

| Flag | Type | Default | Description |
|---|---|---|---|
| `--num-updates-per-batch` | int | `1` | PPO optimizer updates per rollout batch. See below. |
| `--advantage-estimator` | str | — | Set to `grpo`. |
| `--eps-clip` | float | `0.2` | PPO clip range. FlowGRPO uses a much smaller value than text RL — both shipped recipes use `1e-4`. |
| `--eps-clip-high` | float | `None` | Optional asymmetric upper clip. `grpo_clip_loss` supports it; the actor currently passes only the symmetric `--eps-clip`. |
| `--rollout-batch-size` | int | — | Prompts per rollout (number of groups). |
| `--n-samples-per-prompt` | int | — | Candidates per prompt — the GRPO group size. Advantages are normalized within the group. |
| `--global-batch-size` | int | — | `rollout_batch_size * n_samples_per_prompt`. This is the rollout batch, **not** the optimizer-step boundary when `--num-updates-per-batch > 1`. |
| `--micro-batch-size` | int | `1` | How many candidates of a rank's slice are forwarded at once. See below. |
| `--clip-grad` | float | `1.0` | Gradient-norm clip applied before each optimizer step. |
| `--disable-grpo-std-normalization` | flag | normalization **on** | Stop dividing each group-centered component by that group's own std — the Dr.GRPO variant. `grpo_std_normalization` is also force-cleared when `n_samples_per_prompt == 1`. |
| `--generative-advantage-std-mode` | `group` \| `batch` \| `none` | legacy bool behaviour | Explicit std divisor after per-prompt centering. `group` matches Relax/Text GRPO, `batch` matches the reference diffusion recipe's global std, and `none` is Dr.GRPO. The LoRA script defaults to `group`. |

::: warning `--num-updates-per-batch` splits the batch; it does not replay it
`_plan_micro_batch_updates` partitions this rank's local train shard into **N disjoint, equal-sample-count updates** (CountPlanner-style). Every sample contributes to exactly one optimizer step; nothing is reused across mini-epochs. Consequently the optimizer-step boundary is `global_batch_size / num_updates_per_batch`, not `global_batch_size`.

The π_old anchor is still frozen once, before any optimizer step, for every micro-batch — that is what makes the first update's ratio exactly 1 while later updates, running after an optimizer step has moved the weights, diverge from the anchor and produce a nonzero clipped loss with real PPO clipping.

With `--num-updates-per-batch 1` the ratio is always exactly 1, so with group-centered advantages the clipped loss is 0 by construction and clipping never engages. Both shipped recipes use `2`.

The plan is a hard consistency check: `N` must evenly divide the local sample count, and no micro-batch may straddle an update boundary, or `train()` raises.
:::

::: tip `--micro-batch-size` and the DP shard
`hydrate_micro_batches` builds the same ordered group list on every rank (one micro-batch per group), so the FSDP collective sequence is identical everywhere. When `n_samples_per_prompt` divides the DP world size, each rank replays only the `[dp_rank::dp_world]` candidates of each group; otherwise the group falls back to full replay on every rank (correct, just redundant).

`--micro-batch-size` splits that per-rank slice further into equal chunks, each emitted as its own micro-batch — it only splits when the chunk size divides the slice evenly, because unequal chunks would silently reweight the mean. Keep it as **large** as memory allows: the FLOPs are identical either way, but every extra micro-batch is another FSDP all-gather and another reduce-scatter. The full-FT script uses `2`; the LoRA script uses `1` because at `n=8` over 8 ranks the slice is already a single candidate.
:::

The objective lives in `relax/models/flow_grpo.py` (`flow_sde_transition_std_dev_t`, `dance_sde_transition_std_dev_t`, `flow_sde_transition_std`, `flow_sde_transition_moments`, `flow_sde_log_prob`, `replay_transition_logp`, `grpo_clip_loss`, `normalize_grouped`, `combine_component_advantages`). It is pure `torch` with no framework imports and is shared by the rollout engine and the actor replay so the first on-policy update matches bit-for-bit.

The clipped objective is evaluated **per (sample, step)** and mean-reduced over both axes: `_replay_logp` returns a `[B, S]` matrix with one column per trained SDE step rather than summing the steps, so one step drifting out of the clip band does not zero the gradient of the others.

::: danger No KL-to-reference
There is no reference model and no KL term in the FlowGRPO update. Preflight rejects any nonzero `--kl-coef` or `--use-kl-loss` rather than silently ignoring it.
:::

### Reward

| Flag | Type | Default | Description |
|---|---|---|---|
| `--reward-runtime` | `colocate` \| `cpu` \| `remote` | `cpu` | Where the scorer runs. See the note below. |
| `--reward-scorer-path` | str | `None` | Dotpath to a `GenerativeRewardScorer`. Required unless `reward_runtime=remote` with an endpoint. |
| `--reward-model-path` | str | `None` | Local model directory for the scorer. |
| `--reward-required-components` | JSON list | `None` | Component names that must be present, e.g. `'["pickscore"]'`. A missing or non-finite component fails the whole group. |
| `--reward-component-weights` | JSON dict | `None` | `{component: weight}` used to combine normalized components into one advantage. Preflight fails if a required component has no weight. |
| `--reward-endpoint` | str | `None` | HTTP endpoint. Required when `reward_runtime=remote`. |

::: warning `cpu` and `colocate` both score in-process
`post_process` runs inside the rollout worker, so there are exactly two possibilities: score in this process (`cpu`, with CUDA disabled, or `colocate`, sharing the rollout GPU while the actor is offloaded), or POST to an external service (`remote`). There is no separate scorer actor pool — the old `dedicated` value was a synonym for `colocate` that additionally logged a warning about a Ray placement group it never got, and it has been removed. The manager is cached per scorer configuration, so the model is loaded once rather than once per rollout.

Both shipped scripts default to `--reward-runtime colocate`: PickScore needs ~5.6 GB, it runs while the actor is offloaded, and on CPU it dominates post-processing (measured ~104 s/rollout for reward + convert + TransferQueue at 256 samples). `reward/score_time` measures the scorer alone, so the choice is verifiable.
:::

#### Shipped scorers

| Scorer | Dotpath | Components | Notes |
|---|---|---|---|
| PickScore | `relax.engine.rewards.pickscore.PickScoreScorer` | `pickscore` | CLIP-H text-image alignment (`required_tracks = ("image",)`); the validated T2I reward. |

Only PickScore has been exercised in the validated T2I loop.

### Weight sync

| Flag | Type | Default | Description |
|---|---|---|---|
| `--weight-sync-mode` | `full` \| `adapter` | `full` | Derived, not configured: `--lora-adapter-mode` implies `adapter`, and an explicit value that contradicts it is a preflight error rather than something the validator silently overwrites. |
| `--weight-sync-wire-dtype` | `bf16` | `bf16` | Wire dtype for the streamed tensors. |
| `--weight-sync-bucket-size-mb` | int | `512` | Bucket size for the chunked full-tensor stream. Both shipped scripts raise it to `2048`. |

There is no sync-interval flag: the colocate transaction syncs every step by construction, at the end of `train()`.

Each step the actor gathers the full transformer with DTensor `full_tensor()`, builds a `FullWeightManifest` (tensor count, total bytes, `ordered_name_shape_hash`), and streams it bucket by bucket to every engine over CUDA IPC. The IPC topology is matched by CUDA device UUID, not GPU index, because the placement group reorders GPUs. After streaming, the count/bytes are verified against the manifest, rank 0 cross-checks `get_weights_checksum`, and only then is `commit_weight_version` called. Any failure raises `WeightSyncError`; `update_weights` rolls the policy version back so a retry rebuilds the same version, increments `train/weight_sync_failures`, and either recovers the engines (with `--use-fault-tolerance`) or re-raises on every rank.

::: tip Activation-checkpointing wrapper names
`apply_activation_checkpointing` inserts a `_checkpoint_wrapped_module` segment into every wrapped block's parameter names. `strip_transport_wrappers` removes it before the manifest and the wire name are computed. Without that, SGLang's weight loader skips the unrecognized names with a bare `continue` **while still reporting success** — every transformer-block weight was silently dropped and the sync reported a clean commit.
:::

Bucket size drives the RPC count, and a sync's cost is almost entirely per-RPC overhead: the measured `[wsync time]` breakdown for a 40.9 GB merge-mode sync was gather 0.5 s / serialize 0.3 s / transport 38.8 s, i.e. ~0.5 s per bucket to open an on-device CUDA-IPC handle. 512 MB is 80 buckets; 2048 MB is 20. The bucket is materialized on GPU, so a larger one also adds ~1.5 GB to the transient sync footprint.

### Preflight validation

`validate_generative_config` runs at launch when `train_backend == "fsdp"`, collects **every** violation, and raises a single `ValueError`. It is import-light (no torch, no model loads), and the launch scripts under `scripts/training/diffusion/` go through the same `train.py` entry point, so bad generative configs fail before training services are built.

**Errors:**

- `generation_task` is not in `GENERATION_TASKS` (`("t2i",)`).
- `model_adapter_path` or `model_path` is missing.
- `fsdp_trainable_mode` is neither `full` nor `lora`.
- `fsdp_trainable_mode='lora'` with `lora_rank <= 0`, or `lora_rank > 0` without `fsdp_trainable_mode lora`.
- Under `lora`: `lora_dropout != 0`, or `--fsdp-cpu-offload` on.
- Under `full`: `weight_sync_mode != 'full'`.
- `--lora-adapter-mode` with an explicit contradicting `--weight-sync-mode`, or `weight_sync_mode='adapter'` without `--lora-adapter-mode`.
- `sampling_config` can train SDE step 0 under `sde_type='sde'` (checked across `sde_indices`, `sde_pool` and the derived fraction window).
- `weight_sync_wire_dtype != 'bf16'`.
- `fully_async` or `hybrid` on; `colocate` off.
- `kl_coef != 0` or `use_kl_loss` on.
- `rollout_num_gpus_per_engine < 1`, or `actor_num_gpus_per_node * actor_num_nodes` not divisible by it.
- A `reward_required_components` entry with no `reward_component_weights` entry.
- `use_dynamic_batch_size` on, or `max_tokens_per_gpu` set (diffusion latents are fixed-shape; there is no token-budget batching).
- `autoscaler_config` set (the diffusion engine does not implement the elastic router / scale-out API).
- `save_hf` set (export offline with `examples/diffusion/export_checkpoint.py`).
- `load_debug_rollout_data` set (the generative actor reads its batch from the TransferQueue plus on-disk trajectory sidecars).

**Warnings** (logged, not fatal):

- `fsdp_trainable_mode='lora'` without `--fsdp-master-dtype fp32`.
- `async_save` (the DCP save is synchronous; this cannot be a hard error because `slime_validate_args` requires `--async-save` alongside `--rotate-ckpt`).
- `lr_decay_style` set while `--fsdp-lr-scheduler` is `constant`.
- `save_debug_train_data` (the generative actor emits numeric indices plus on-disk sidecars, not token sequences).

#### Runtime checks

Checks that need live objects run later:

| Check | Where |
|---|---|
| Trainable boundary (full FT: nothing frozen; LoRA: adapter present and base frozen) | `FSDPTrainRayActor._assert_trainable_boundary` |
| `guidance_scale == 1.0` | `QwenImageAdapter.build_rollout_request` |
| Response carries `trajectory_latents`, `timesteps`, `sde_indices`, `height`, `width` | `QwenImageAdapter.validate_rollout_response` |
| `actor_world == sum(engine_gpu_counts)` | first weight sync (`_assert_sync_plan_aligned`) |
| Equal micro-batch count across DP ranks | `_assert_microbatch_count_aligned` (MAX-reduce, raises instead of deadlocking) |
| Adapter contract match on resume (mode, base-model hash, LoRA rank/alpha/dropout/targets) | `_assert_resumable` |

### Reference configuration

Both recipes live in `scripts/training/diffusion/`. Every value below is an environment-overridable default in the script; the script is the source of truth, this is a reading aid.

#### Validated LoRA recipe

`scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh` has a completed 100-rollout adapter-sync alignment run against the Qwen-Image / PickScore reference baseline (final eval 0.8643 vs the baseline's 0.8620).

```bash
--resource '{"actor": [1, 8], "rollout": [1, 8]}'
--colocate

# backend / task / hooks
--train-backend fsdp
--generation-task t2i
--model-path ${MODEL_DIR}/Qwen-Image
--model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter
--rollout-engine-class-path relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine
--rollout-function-path relax.engine.rollout.native_generation.generate_rollout
--custom-convert-samples-to-train-data-path relax.engine.rollout.native_generation.convert_samples_to_train_data
--custom-reward-post-process-path relax.engine.rewards.generative.post_process

# group geometry: 32 prompts x 8 candidates = a 256-sample rollout batch,
# split into 2 disjoint optimizer updates of 128 samples each
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256
--micro-batch-size 1
--rollout-seed 42

# eval
--eval-prompt-data pickscore ${DATA_DIR}/processed/t2i/pickapic_eval256.jsonl
--eval-interval 10
--n-samples-per-eval-prompt 2

# sampling: generate 12 steps, train 3 of them, redrawn per rollout from a pool
# that excludes the singular step 0
--sampling-config '{"height":384,"width":384,"num_inference_steps":12,"guidance_scale":1.0,"eta":0.7,"sde_type":"sde","sde_indices":[3,4,5],"sde_resample_per_rollout":true,"num_sde_steps":3,"sde_pool":[1,2,3,4,5],"driver_xt":true,"sample_id_mode":"metadata"}'
--generation-seed 42

# LoRA
--lora-rank 64
--lora-alpha 128
--lora-dropout 0.0
--lora-adapter-mode          # script default; set ADAPTER_MODE=0 for merge-mode sync validation

# FSDP2
--fsdp-trainable-mode lora
--fsdp-trainable-attr transformer
--fsdp-param-dtype bf16
--fsdp-reduce-dtype fp32
--fsdp-master-dtype fp32
--fsdp-activation-checkpointing
--fsdp-load-wave-size 2
--offload-train
--offload-rollout

# FlowGRPO + optimizer (default LoRA recipe)
--advantage-estimator grpo
--eps-clip 1e-4
--generative-advantage-std-mode group
--num-updates-per-batch 2
--lr 3e-4
--lr-warmup-iters 0
--adam-eps 1e-8
--weight-decay 0.0

# reward
--reward-runtime colocate
--reward-scorer-path relax.engine.rewards.pickscore.PickScoreScorer
--reward-model-path ${MODEL_DIR}/PickScore_v1
--reward-required-components '["pickscore"]'
--reward-component-weights '{"pickscore":1.0}'

# weight sync + engine layout
--weight-sync-wire-dtype bf16
--weight-sync-bucket-size-mb 2048
--rollout-num-gpus-per-engine 1

# checkpoints
--save ${EXP_DIR}/runs/qwen-image-t2i-lora/
--save-interval 20
--max-actor-ckpt-to-keep 10
```

At launch the script also writes
`ARTIFACT_ROOT/recipes/qwen-image-t2i-lora-*.json`, recording the prompt/eval
sets, DataLoader-order flag, sampling config, `advantage_std_mode`, LoRA sync
mode, git revision and SGLang patch version used for the validation run.

#### Full fine-tune reference

`scripts/training/diffusion/run-qwen-image-t2i-8xgpu.sh` is where the empirically bracketed full-FT settings are documented. It has not reproduced the reference curve; prefer the LoRA script unless you specifically need full-parameter updates. It differs from the block above in:

| Setting | Full FT | Why |
|---|---|---|
| `--fsdp-trainable-mode` | `full` (no `--fsdp-master-dtype`, no `--lora-*`) | Full-parameter updates. |
| `--rollout-batch-size` / `--n-samples-per-prompt` / `--global-batch-size` | `8` / `8` / `64` | `n` is capped by memory without the ~25 GB/rank the frozen base's grads and AdamW moments cost. At the bare GRPO minimum of 2, `reward/group_std_mean` was only ~0.01 — too weak to learn from. |
| `--micro-batch-size` | `2` | At `n=8` over 8 ranks with `1` candidate per group per rank the split is skipped anyway; `2` is the documented "as large as memory allows" default (measured `mem_peak_gb` 63.6 of 96 at `B=1`). |
| `sde_indices` | `[1,3,5]` | Same pool `[1,2,3,4,5]` and same per-rollout resample. |
| `--lr` / `--lr-warmup-iters` / `--adam-eps` | `3e-5` / `20` / `1e-15` | Bracketed against a held-out 256-prompt PickScore eval: `1e-5`+`1e-8` froze (129 rollouts, eval flat), `1e-4`+`1e-15` overshot (eval −0.06 by rollout 19). The tiny `adam-eps` restores AdamW's scale invariance — the mean-reduced flow-SDE log-prob yields per-element grads ~1e-8, so the default `1e-8` dominates the denominator. |
| eval set / retention | `pickapic_eval64.jsonl` / `--max-actor-ckpt-to-keep 2` | A full-FT 20B DCP checkpoint (weights + fp32 AdamW state) is ~115 GB. |

## LoRA

Set `--fsdp-trainable-mode lora` together with `--lora-rank`. The two must agree — preflight rejects one without the other rather than silently training the wrong parameter set. The flags are the same `--lora-*` group the Megatron text path uses; the FSDP/diffusers implementation is `relax/backends/fsdp/lora.py`.

```bash
bash scripts/training/diffusion/run-qwen-image-t2i-lora-8xgpu.sh
```

### What it buys

**Memory first; adapter sync is the validated rollout path.** Freezing the base removes its gradients and AdamW moments — roughly 25 GB/rank on a 20B DiT — which is what lets you raise `--n-samples-per-prompt` and the number of trained SDE steps. It does *not* shrink activations. The script defaults to adapter sync so only LoRA tensors move each step; merge mode still streams the whole transformer and is intended for explicit transport validation.

### Rollout paths

| Mode | Flag | Sync per step | SGLang requirement |
|---|---|---|---|
| Adapter | `--lora-adapter-mode` (script default) | Adapter only — 1440 tensors, ~377 MB for the validated rank-64 run | `docker/patch/sglang/v0.5.15.post1.patch` |
| Merge | `--lora-merge-mode` (`ADAPTER_MODE=0`) | Full transformer, with `B @ A` folded into each base weight | `docker/patch/sglang/v0.5.15.post1.patch` |

The two must never be mixed on one engine. Once SGLang's LoRA conversion runs, the DiT parameters are renamed to `*.base_layer.weight`, and its weight loader skips names it does not recognize with a bare `continue` **while still reporting success** — a full-weight sync into a LoRA-converted engine silently drops every tensor. Adapter mode therefore never performs a full sync at all (LoRA training does not change the base), and the patch turns the mixed case into an explicit error.

### Target modules

Unset, `--lora-target-modules` falls back to the model adapter's own `lora_target_modules`. `QwenImageAdapter` supplies the eight attention projections of each transformer block — `attn.to_q/to_k/to_v/to_out.0` (image stream) and `attn.add_q_proj/add_k_proj/add_v_proj/to_add_out` (text stream). SGLang's Qwen-Image DiT names these identically, so an adapter trained here loads there without a rename.

Names are matched as module-name **suffixes**.

### Settings that matter

| Flag | Value | Why |
|---|---|---|
| `--fsdp-master-dtype` | `fp32` | The optimizer's master copy for the adapter. With bf16 masters the ~1e-4 relative RL updates round off the mantissa: grad norm and loss stay healthy while the policy stops moving, which reads exactly like a too-low learning rate. Preflight warns when it is unset under LoRA. The forward is unaffected — `--fsdp-param-dtype` still governs the all-gathered compute copy. |
| `--lora-dropout` | `0.0` (enforced) | The FlowGRPO π_old anchor is a replay through the live model under `model.train()`, so a stochastic forward breaks the `ratio == 1` invariant of the first update and, at `--eps-clip 1e-4`, clips essentially everything. |
| `--lr` | `3e-4` | Roughly 10× the full-FT `3e-5`; a rank-64 adapter starts from `B = 0` and has far fewer degrees of freedom. |
| `--fsdp-cpu-offload` | rejected | Gradient clipping over CPU-offloaded adapter DTensors has no collective backend. |

### Checkpoints and export

A LoRA checkpoint stores **only** the adapter (`ignore_frozen_params`), so it is well under 1 GB instead of ~115 GB. The base is reloaded from `--model-path` on every resume, which is why `adapter_contract.json` records the rank, alpha, dropout, target modules, task type and base-model hash — `_assert_resumable` refuses on any mismatch. Alpha in particular cannot be recovered from the weights, and a wrong one silently rescales the policy.

Two export forms:

```bash
# Fold the adapter into the base -> a directly loadable diffusers pipeline
python3 examples/diffusion/export_checkpoint.py --form merged \
  --save-dir ... --task t2i --iteration 100 --output-dir ... \
  --model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter \
  --base-model-path ${MODEL_DIR}/Qwen-Image --verify

# Portable HF-PEFT adapter directory
python3 examples/diffusion/export_checkpoint.py --form adapter \
  --save-dir ... --task t2i --iteration 100 --output-dir ... --verify
```

`--base-model-path` is **required** for `--form merged` on a LoRA checkpoint: without a base there is nothing to fold into, and the export would otherwise write a "transformer" made entirely of `lora_A`/`lora_B` tensors — non-empty and finite, so it would pass a naive sanity check.

## Outputs

### Artifacts

Written under `--artifact-root`, one directory per rollout:

```
${ARTIFACT_ROOT}/t2i/
└── rollout_0000000/
    ├── group_00000000.safetensors          # trajectory sidecar for the whole group
    ├── group_00000000_sample_0000.png      # decoded candidate images
    └── group_00000000_sample_0001.png
```

Evaluation passes write under `${ARTIFACT_ROOT}/eval/t2i/` with the same layout.

The sidecar holds the flattened `common.*` (sigmas, SDE indices, sample indices, policy versions, seed hashes), `conditions.*` (frozen text conditioning) and `tracks.*` (per-candidate latents) namespaces, written `.tmp → fsync → atomic rename`. Each candidate also carries a small JSON manifest in its `train_metadata` recording the policy version, the sampling fingerprint and the weight-manifest hash.

`--artifact-root` must be on storage every trainer rank can read: the driver writes each sidecar from one process and all FSDP ranks read it back. `--artifact-retention-rollouts` (default `2`) reaps older rollout directories at the start of each rollout; without it a few hundred steps silently fill the volume.

### Checkpoints

Saved with PyTorch Distributed Checkpoint (DCP) under `<--save>/<task>/iter_<n>/`:

```
${SAVE}/t2i/iter_0000020/
├── fsdp/                      # DCP shards (model + optimizer)
├── trainer_state.json         # rollout id, policy version, task, adapter, sampling fp, lr_scheduler_steps
├── weight_sync_manifest.json
├── adapter_contract.json
├── rng_rank0.pt ...           # per-rank RNG, best effort
└── COMMITTED                  # written last; the only marker resume trusts
${SAVE}/latest_checkpointed_iteration.txt
${SAVE}/t2i/latest_checkpointed_iteration.txt
```

The write order is fixed — shards → RNG → JSON sidecars → `COMMITTED` → the `latest_checkpointed_iteration.txt` markers — so a crash mid-write never leaves a half-checkpoint that resume would trust. The `COMMITTED` marker is only written when a MIN all-reduce confirms **every** rank wrote successfully.

`--rotate-ckpt` / `--max-actor-ckpt-to-keep` apply; rotation runs on rank 0 before the new write and its outcome is broadcast, so a rotation failure raises on every rank instead of hanging the others at a barrier. A failed DCP write is logged loudly and counted in `train/checkpoint_failures` but does not kill the run.

::: warning Resume needs `--load`
`_maybe_resume` reads `--load`, not `--save`. Neither shipped script sets it, so a restart begins from rollout 0 unless you pass `--load <same dir as --save>`. Resume then restores the model, optimizer, `policy_version`, `lr_scheduler_steps` (so warmup does not restart) and per-rank RNG, and returns `latest + 1` as the starting rollout id.
:::

Export a committed checkpoint back to HF safetensors:

```bash
python3 examples/diffusion/export_checkpoint.py \
  --save-dir ${SAVE_DIR} \
  --task t2i \
  --iteration 20 \
  --output-dir /path/to/export \
  --model-adapter-path relax.models.qwen_image.adapter.QwenImageAdapter \
  --verify
```

## Monitoring

The actor initializes the tracking adapter in-process on rank 0, so the usual `--use-clearml` / `--use-metrics-service` / TensorBoard flags work. All series share the `rollout/step` x-axis. Generative-specific series:

| Metric | Meaning |
|---|---|
| `train/loss`, `train/ratio_{mean,std,min,max}`, `train/clip_fraction`, `train/approx_kl` | FlowGRPO clipped-objective health. `ratio_min` / `ratio_max` are MAX-reduced across ranks, the rest averaged |
| `train/grad_norm`, `train/lr`, `train/optimizer_steps`, `train/optimizer_skips`, `train/num_microbatches` | Optimizer health; `skips == num_microbatches` means the whole round was dropped as degenerate |
| `train/policy_version`, `train/weight_sync_tensors`, `train/weight_sync_gb`, `train/weight_sync_failures` | Weight-sync provenance — without a version series a reward regression cannot be tied to a rolled-back sync |
| `train/checkpoint_failures` | Non-fatal DCP save failures |
| `reward/<component>_{mean,std,min,max}` | Per-component raw reward (e.g. `reward/pickscore_mean`) |
| `reward/advantage_{mean,std}`, `reward/num_groups` | Combined advantage distribution |
| `reward/group_std_mean` | **The key diagnostic** — mean per-group reward std |
| `reward/degenerate` | 1 when the advantage variance collapsed below `1e-6` |
| `reward/score_time`, `reward/score_samples_per_s` | The scorer alone, separate from `perf/reward_time` |
| `perf/mem_{allocated,reserved,peak}_gb` | GPU watermarks, worst rank |
| `perf/train_transitions_per_s`, `perf/train_samples_per_s` | Throughput. A "transition" is one replayed SDE step for one candidate |

::: tip No MFU / TFLOPs
The shared `FlopsCounter` dispatches on an LLM `hf_config` and has no analytic model for a DiT, so `log_perf_data_raw` is called with `flops_counter=None`. The generative path reports replayed transitions and candidates per second instead — the quantities that actually bound it.
:::

## Troubleshooting

**`Generative RL config validation failed:` at launch.** Preflight collects every violation at once — read the whole list. The common ones are a missing `--model-adapter-path` / `--model-path`, `--fully-async` left on, a nonzero `--kl-coef`, a `sampling_config` that can train SDE step 0, or an actor world size that is not divisible by `--rollout-num-gpus-per-engine` (each engine's GPUs must tile the FSDP ranks for the CUDA-IPC sync).

**`reward/group_std_mean` near zero and `train/loss` stuck at 0.** The group has no reward variance, so every advantage is ~0 and the optimizer step is skipped. Increase `--n-samples-per-prompt`, check that your prompts are diverse enough for the scorer to separate candidates, and confirm `--num-updates-per-batch` is ≥ 2.

**OOM during the training step.** The card holds the sharded params, grads, AdamW state and the replay activations. In order: keep `--fsdp-activation-checkpointing` on, shrink the trained SDE-step subset (`num_sde_steps` / `sde_pool`), lower the resolution in `--sampling-config`, then reduce `--n-samples-per-prompt` — or switch to the LoRA recipe, which frees the frozen base's grads and AdamW moments (~25 GB/rank). Keep `--offload-train` and `--offload-rollout` on; colocate depends on the actor and the engine taking turns on the card.

**OOM at launch, before the first step.** Every FSDP rank materializes the full base model on CPU before sharding. Lower `--fsdp-load-wave-size` (both shipped scripts use `2`) to gate how many ranks load at once.

**Suspected no-op optimizer steps.** `--fsdp-debug-fingerprint` logs an abs-sum fingerprint of the trainable weights before and after each step plus a `[logp parity]` line comparing the replayed anchor against the engine's own sampling log-prob, isolating a broken step/offload path from broken reward math. It is off by default because it is a full-parameter reduction plus a host sync every step.

**GPU memory still held after a stopped run.** Check for orphan `sgl_diffusion::scheduler` processes before relaunching, even when `ray serve status` is empty.

## Acknowledgements

We thank the UniRL authors for the Qwen-Image / PickScore reference recipe and metrics used for alignment checks.

## Next Steps

- [Architecture](./architecture.md) — the controller / service / component model the generative path plugs into
- [OOM Troubleshooting](./oom-troubleshooting.md) — general memory-pressure playbook
