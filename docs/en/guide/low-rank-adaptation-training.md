# LoRA RL (Parameter-Efficient RL Post-Training)

Relax supports **LoRA** (Low-Rank Adaptation) for RL post-training: instead of updating every weight, only small low-rank adapter matrices are trained while the base model stays frozen. This shrinks the optimizer state and the per-step weight-sync payload, so much larger models fit on the same GPU budget.

Two end-to-end LoRA rollout paths are wired up, differing only in **how the trained adapter reaches the rollout engine**:

| Mode              | What is synced to rollout each step          | Rollout serves         | Reference launch script                                                |
| ----------------- | -------------------------------------------- | ---------------------- | --------------------------------------------------------------------- |
| **Merge mode**    | Full model (adapter folded into base)        | one merged model       | `scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh`               |
| **Adapter mode**  | Base once, then only the LoRA adapter        | base + runtime adapter | `scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh`       |

The two modes are **mutually exclusive**. If you enable LoRA (`--lora-rank > 0`) without picking a mode, Relax forces **merge mode** (the default, broadest-support path).

Both modes support **dense and MoE (grouped-expert)** models, in **colocate and fully-async**. Larger reference recipes:

| Recipe                       | Script                                                                     |
| ---------------------------- | -------------------------------------------------------------------------- |
| MoE, adapter mode, 2-node    | `scripts/training/text/run-qwen36-35B-A3B-lora-adapter-16xgpu-async.sh`     |
| MoE + VL, merge mode         | `scripts/training/multimodal/run-qwen36-35B-A3B-lora-8xgpu-image.sh`        |

## Overview

LoRA is applied on the training side through Megatron-Bridge's PEFT integration (`relax/backends/megatron/model_provider.py`): the base model is frozen and only the injected adapter parameters receive gradients. The difference between the two modes is entirely on the **weight-sync path** (`relax/backends/megatron/weight_update/`):

- **Merge mode** folds each adapter into its base weight at export time (`LoRAMerge`), so the rollout engine loads a single merged model — no LoRA awareness required on the inference side. This reuses the standard full-weight-sync path, so bandwidth per step is the same as a full-parameter run.
- **Adapter mode** syncs the frozen base **once**, then every step pushes **only the small adapter** to the rollout engine via SGLang's runtime LoRA API. Rollout requests select it with `lora_path`. This trades a bit of rollout-side LoRA overhead for a large drop in per-step sync bandwidth.

::: tip Which mode should I use?
- **Merge mode** — simplest, works with the widest set of models and both deployment layouts (including distributed/NCCL rollout engines). Rollout inference is plain full-model inference. Costs a full weight sync every step.
- **Adapter mode** — best when the per-step weight-sync bandwidth dominates (large base model, small adapter). Adds constraints (`--sglang-dp-size 1`, no distributed rollout engines, `--lora-scope language` on VL / omni models). Rollout runs through SGLang's LoRA kernels.
:::

## Architecture

Both modes share the same training side; only the payload on the sync arrow differs.

```
        ┌────────────────────────────────────────────────────────┐
        │                  Training side (Actor)                 │
        │  Megatron-LM + Megatron-Bridge PEFT                    │
        │  base weights FROZEN, only LoRA adapter has gradients  │
        │  (model_provider.wrap_model_provider_with_lora)        │
        └───────────────────────────┬────────────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
     MERGE mode: fold adapter                      ADAPTER mode: base once,
     into base, sync FULL model                    then push ADAPTER only
     every step (NCCL / IPC)                       (SGLang runtime LoRA API)
             │                                               │
             ▼                                               ▼
        ┌─────────────────────────┐              ┌─────────────────────────┐
        │  Rollout (SGLang)       │              │  Rollout (SGLang)       │
        │  one merged model       │              │  base + adapter         │
        │  plain inference        │              │  lora_path=             │
        │                         │              │   relax_policy_lora     │
        └─────────────────────────┘              └─────────────────────────┘
```

The weight-sync backend dispatches on the LoRA mode:

- **Colocate** — `UpdateWeightFromTensor` (`weight_update/update_weight_from_tensor.py`). Merge mode folds adapters during HF export; adapter mode uses `_update_weights_adapter_mode` (base-once + per-step in-memory adapter push via `load_lora_adapter_from_tensors`).
- **Fully-async** — `DeviceDirectBackend` (`distributed/checkpoint_service/backends/device_direct.py`). Merge mode folds adapters before the NCCL broadcast; adapter mode has rank 0 broadcast the adapter tensors over NCCL in buckets capped by `--update-weight-buffer-size`, with a metadata-only HTTP `/update_lora_from_distributed` fanned out to every engine.

The mode-independent adapter export/gather logic (build the HF export bridge, export the local adapter in SGLang naming, PP-gather, the collective delta-skip decision, and the adapter config dict) lives in the shared `LoraAdapterSync` helper (`weight_update/lora_adapter_sync.py`), which both backends compose. It is purely in-memory — the live adapter never touches disk.

## Configuration

All LoRA flags are defined in `relax/utils/arguments.py`.

| Flag                     | Type       | Default                       | Description                                                                                          |
| ------------------------ | ---------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--lora-rank`            | int        | `0`                           | LoRA rank. `0` disables LoRA; any value `> 0` enables it.                                             |
| `--lora-alpha`           | int        | `32`                          | LoRA alpha scaling factor.                                                                            |
| `--lora-target-modules`  | str (list) | `linear_qkv linear_proj`      | Megatron-style module names to adapt (see mapping below). Space-separated.                            |
| `--lora-scope`           | str        | `all`                         | Which model region gets adapters: `all` / `language` / `vision`. No effect on text-only models.       |
| `--lora-dropout`         | float      | `0.0`                         | Dropout probability applied inside the LoRA layers.                                                   |
| `--lora-merge-mode`      | flag       | `False`                       | Fold adapters into base weights before sync. Mutually exclusive with `--lora-adapter-mode`.           |
| `--lora-adapter-mode`    | flag       | `False`                       | Sync base once, then push only the adapter each step. Mutually exclusive with `--lora-merge-mode`.    |

### Target modules

`--lora-target-modules` takes **Megatron-style** names (the canonical form Megatron-Bridge's LoRA matcher walks). They are expanded to HF-style names automatically when exporting the adapter (`convert_megatron_to_hf_target_modules` in `relax/utils/megatron_peft_utils.py`):

| Megatron name          | HF projection(s)                                          |
| ---------------------- | --------------------------------------------------------- |
| `linear_qkv`           | `q_proj`, `k_proj`, `v_proj`                              |
| `linear_proj`          | `o_proj`                                                  |
| `linear_fc1`           | `gate_proj`, `up_proj`                                    |
| `linear_fc2`           | `down_proj`                                               |
| `router`               | `gate`                                                    |
| `in_proj` (GDN)        | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`      |
| `out_proj` (GDN)       | `out_proj`                                                |

(The full map, including split-QKV and MLA variants, is in `MEGATRON_TO_HF_MODULES`.)

The adapter that is pushed to the rollout engine uses a slightly different flavor (`convert_megatron_to_sglang_target_modules`, overrides in `MEGATRON_TO_SGLANG_MODULES`): SGLang fuses the GDN input projection, so `in_proj` maps to the single `in_proj_qkvz` it wraps. Everything else matches the HF names. The *checkpoint* adapter keeps HF-PEFT names so it stays loadable by `peft`.

### LoRA scope (VL / omni models)

Megatron-Bridge's LoRA matcher matches by **leaf name**, so a bare `linear_qkv` would be injected into the vision tower as well as the language backbone. `--lora-scope` resolves the requested region against the concrete module tree and wraps full-path names instead (`scope_target_modules_to_region`):

- `all` (default) — every matched module, including vision tower / projector / audio encoder.
- `language` — excludes the vision tower / projector / audio encoder. Typical for language-only RL, and **required by adapter mode** on a model that declares a `vision_config` or `audio_config`.
- `vision` — only those regions.

This controls adapter **injection**, not base-weight freezing.

### Validation rules

Enforced in `relax/utils/arguments.py` when `lora_rank > 0`:

- `--lora-merge-mode` and `--lora-adapter-mode` are **mutually exclusive** — pick one.
- `--lora-adapter-mode` requires `--sglang-dp-size 1` (SGLang dynamic LoRA loading does not support DP attention).
- `--lora-adapter-mode` with `--lora-scope all` is **rejected** on a model whose HF config declares `vision_config` or `audio_config`. SGLang hosts LoRA on language-model layers only, so a vision adapter would train and then be silently dropped before rollout — the actor would optimize a policy the rollout never runs. Pass `--lora-scope language`, or use `--lora-merge-mode`.
- `router` in `--lora-target-modules` requires `--lora-adapter-mode`. Merge mode folds adapters via `LoRAMerge`, which only transforms `LoRALinear`; the router uses `LoRATopKRouter`, so its adapter would be dropped at sync time.
- If neither mode flag is set, Relax **forces `--lora-merge-mode`** and logs a warning.

## Merge Mode Recipe

Reference: `scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh` (Qwen3-4B, 8-GPU colocate, GRPO on `dapo-math-17k`).

The LoRA block in that script:

```bash
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-merge-mode
)
```

Launch it like any other colocate script:

```bash
bash scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh
```

How it works each step: the actor exports its weights, `LoRAMerge` folds every adapter into its paired base weight during HF conversion, and the merged full model is synced to SGLang over the normal IPC/NCCL path. The rollout engine is completely LoRA-agnostic — it just serves a full model.

::: warning MoE expert folding requires `--expert-tensor-parallel-size 1`
Grouped-expert LoRA is folded into each expert's base weight locally on the owning EP rank (`tp_size=1`, no expert-TP collective), so a mismatched ETP would leave one rank alone in a collective and deadlock. This applies to **merge mode always**, and to **adapter mode whenever an `actor_fwd` / reference role is present** (off-policy) — that role has no adapter transport of its own, so its expert delta is folded into the base instead. EP (expert-model-parallel) may stay `> 1`; dense models are unaffected. The reference recipes already set `--expert-tensor-parallel-size 1`.
:::

## Adapter Mode Recipe

Reference: `scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh` (Qwen3-4B, 8-GPU fully-async, GRPO on `dapo-math-17k`).

The LoRA block in that script:

```bash
LORA_ARGS=(
   --lora-rank 128
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-adapter-mode
)
```

Note it also sets `--rollout-num-gpus-per-engine 1` (one GPU per engine → `sglang_dp_size == 1`, required by adapter mode). Launch it:

```bash
bash scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh
```

How it works:

1. **First sync** — push base-only weights to the engine (adapter params are pulled out of the conversion buckets, never merged), then register the LoRA adapter under the fixed name `relax_policy_lora`.
2. **Every subsequent step** — refresh **only** the adapter. The base stays put on the engine. Rollout requests automatically pass `lora_path=relax_policy_lora` (`relax/engine/rollout/sglang_rollout.py`) so generation runs through the trained adapter.
3. **Delta-skip** — if no adapter parameter changed beyond a `1e-6` threshold on any rank, the whole push is skipped. This is a collective decision across all ranks (a per-rank early return would desync the gather and hang).

The transport of the adapter itself differs by deployment:

- **Colocate** — the adapter is serialized and pushed in-memory via `load_lora_adapter_from_tensors` (SGLang ≥ 0.5.12, no disk IO).
- **Fully-async** — rank 0 assembles the full adapter and `broadcast`s it over NCCL on the weight-update group (`src=0`, the same group base weights use), with an HTTP `/update_lora_from_distributed` fanned out to every engine carrying metadata only (tensor names / dtypes / shapes / bucket boundaries / adapter config). The broadcast is bucketed by `--update-weight-buffer-size` (default 512 MiB) on both sides, so peak device memory is one bucket rather than the whole adapter — an A3B MoE adapter is multi-GB. Also no disk IO — the adapter never touches the network filesystem.

::: warning Adapter mode constraints
- `--sglang-dp-size 1` is required (enforced at arg-validation time).
- **Distributed (non-colocated NCCL) rollout engines are not supported** in colocate mode — the adapter is only pushed to colocated IPC engines. Use colocate or fully-async; for distributed rollout use merge mode instead.
- On a **VL / omni** model, `--lora-scope language` is required (enforced at arg-validation time): SGLang hosts LoRA on language-model layers only. Use merge mode if you need to train the vision tower.
- MoE (grouped-expert) LoRA **is supported** in both deployments. With an `actor_fwd` / reference role present, `--expert-tensor-parallel-size 1` is required (see the warning above).
- On **GDN** layers, the `in_proj` adapter's `b`/`a` (beta/alpha gate) rows are pinned to zero by a forward hook (`install_gdn_gate_mask_hooks`): SGLang carries an adapter on the fused `in_proj_qkvz` but has none for the gate slices, so a delta learned there could never be replayed at rollout. Masking the adapter *output* means those rows receive zero gradient, keeping training and rollout numerically identical. Merge mode does not need this — the gate slices fold into the base losslessly.
- Adapter mode disables next-step rollout prefetch in colocate (the per-step adapter update is ~1s, so prefetch would just be re-done).
:::

## MoE and VL Recipes

- **MoE, adapter mode, 2 nodes** — `scripts/training/text/run-qwen36-35B-A3B-lora-adapter-16xgpu-async.sh` (Qwen3.6-35B-A3B, 16-GPU fully-async). Adapts attention, GDN and both MLP projections, and scopes to the language backbone:

  ```bash
  LORA_ARGS=(
     --lora-rank 32
     --lora-alpha 64
     --lora-scope language
     --lora-target-modules linear_qkv linear_proj in_proj out_proj linear_fc1 linear_fc2
     --lora-dropout 0.0
     --lora-adapter-mode
  )
  ```

- **MoE + VL, merge mode** — `scripts/training/multimodal/run-qwen36-35B-A3B-lora-8xgpu-image.sh`. Uses `--lora-scope all` to train the vision tower too, which forces `--lora-merge-mode` (SGLang cannot host a vision adapter).

Both set `--expert-tensor-parallel-size 1`.

## Checkpointing and Export

By default, `--save` writes the regular full Megatron distributed checkpoint. Add `--save-lora-only` to write a smaller, resumable actor checkpoint instead. This mode stores the LoRA tensors together with optimizer, scheduler, RNG, iteration, and common state, while restoring frozen base weights from `--hf-checkpoint` on resume.

`--save-lora-only` is mutually exclusive with full checkpoint export: it requires synchronous BF16 `torch_dist` checkpointing and cannot be combined with `--async-save`, `--rotate-ckpt`, `--no-save-optim`, `--no-save-rng`, `--fp16`, `--fp8`, or `--save-hf`. It also fails if any trainable model parameter is not a LoRA adapter, preventing an incomplete resume artifact.

::: tip
The lightweight checkpoint is a sharded Megatron resume artifact, not a portable HF-PEFT adapter. Explicit `--save-hf` export can still write `lora_adapter/` with `adapter_config.json`, `adapter_model.safetensors`, and `relax_lora_meta.json`; that directory is for external or inference use and is not a resume source.
:::

### Offline merge into a standalone HF checkpoint

`scripts/tools/merge_lora_adapter_to_hf.py` folds an exported `lora_adapter/` back into its base HF model and writes a standalone merged checkpoint — no Ray / Megatron / GPU cluster required:

```bash
python scripts/tools/merge_lora_adapter_to_hf.py \
    --base-hf-dir /path/to/Qwen3.6-35B-A3B \
    --adapter-dir /path/to/save/iter_0000100/lora_adapter \
    --output-dir  /path/to/Qwen3.6-35B-A3B-merged
```

It merges at the **tensor** level rather than via `peft.merge_and_unload()`, on purpose: `AutoModelForCausalLM` silently resolves a multimodal checkpoint to its text-only class (dropping `vision_config` and the vision tower on re-save), and PEFT matches *modules* by name so it cannot see grouped MoE experts, which are 3-D `nn.Parameter` tensors — the bulk of the trained capacity on an A3B MoE. The tool streams one shard at a time (memory is bounded by the largest shard), copies `config.json` and auxiliary files verbatim, and treats an adapter tensor that fails to pair with a base tensor as a hard error rather than a silent drop.

## Troubleshooting

| Symptom                                                                        | Likely cause / fix                                                                                                                            |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lora-merge-mode and --lora-adapter-mode are mutually exclusive`             | Both flags set. Pick exactly one.                                                                                                            |
| `--lora-adapter-mode requires --sglang-dp-size 1`                             | Adapter mode with DP attention. Set `--rollout-num-gpus-per-engine 1` (or otherwise force `sglang_dp_size == 1`).                            |
| `--lora-adapter-mode does not yet support distributed ... rollout engines`     | Adapter mode with non-colocated rollout engines. Switch to colocate, or use `--lora-merge-mode` for distributed rollout.                    |
| `--lora-adapter-mode with --lora-scope all is not supported on a model that has a vision/audio encoder` | VL / omni model in adapter mode. Pass `--lora-scope language`, or switch to `--lora-merge-mode` to train the vision tower.       |
| `LoRA on 'router' is only supported in --lora-adapter-mode`                    | `router` in `--lora-target-modules` under merge mode. Drop `router`, or switch to adapter mode.                                             |
| `MoE LoRA expert folding requires --expert-tensor-parallel-size 1`             | MoE with ETP > 1. Set `--expert-tensor-parallel-size 1` (EP may stay > 1). Adapter mode only avoids this when fully on-policy (no `actor_fwd`). |
| `[lora-merge] NO adapter tensors in backup dict`                              | Adapter params missing from `weights_getter()` output — check `--lora-target-modules` names and that LoRA actually attached at model build.  |
| Rollout quality looks like the base model in adapter mode                      | The adapter push was skipped or `lora_path` not applied. Confirm requests carry `lora_path=relax_policy_lora`, and check the logs for adapter-push entries (delta-skip suppresses pushes when nothing changed). |
