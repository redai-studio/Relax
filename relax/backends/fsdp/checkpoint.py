# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Distributed checkpoint (DCP) save/load with the COMMITTED protocol.

Layout (design doc 12)::

    ${SAVE_DIR}/<task>/
      latest_checkpointed_iteration.txt
      iter_0000007/
        fsdp/                    # DCP sharded model + optimizer state
        trainer_state.json
        weight_sync_manifest.json
        adapter_contract.json
        COMMITTED                # written last; resume only from committed dirs

Write order is fixed — ``fsdp/`` shards → sidecar JSONs → ``COMMITTED`` marker →
``latest_checkpointed_iteration.txt`` — so a crash mid-write never leaves a
half-written checkpoint that resume would trust.

The model/optimizer state is read/written via ``torch.distributed.checkpoint``'s
``get_state_dict`` / ``set_state_dict`` helpers, which transparently handle FSDP2
``DTensor`` shards and plain modules alike, so the same code path serves both the
distributed actor and a single-process unit test.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = [
    "iter_dir_name",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_committed",
    "export_hf",
    "export_peft_adapter",
    "read_adapter_contract",
    "COMMITTED_MARKER",
]

COMMITTED_MARKER = "COMMITTED"
_LATEST = "latest_checkpointed_iteration.txt"


def iter_dir_name(iteration: int) -> str:
    return f"iter_{int(iteration):07d}"


def _all_ranks_ok(ok: int) -> int:
    """MIN-reduce a per-rank success flag so every rank agrees on the
    outcome."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return int(ok)
    if dist.get_world_size(dist.group.WORLD) <= 1:
        return int(ok)
    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    flag = torch.tensor([int(ok)], dtype=torch.long)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return int(flag.item())


def _rank() -> int:
    import torch.distributed as dist

    if not dist.is_initialized():
        return 0
    if dist.get_world_size(dist.group.WORLD) <= 1:
        return dist.get_rank(dist.group.WORLD)
    from relax.utils.distributed_utils import get_gloo_group

    return dist.get_rank(get_gloo_group())


def _save_rng(ckpt_dir: str) -> None:
    """Persist this rank's RNG state (best-effort) for deterministic resume.

    Each rank owns its own generator under pure DP, so RNG is written per rank.
    Best-effort: an RNG-write failure must not fail the checkpoint.
    """
    try:
        state: Dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        import random as _random

        state["python"] = _random.getstate()
        try:
            import numpy as _np

            state["numpy"] = _np.random.get_state()
        except Exception:
            pass
        torch.save(state, os.path.join(ckpt_dir, f"rng_rank{_rank()}.pt"))
    except Exception as e:
        logger.warning(f"RNG state save skipped: {e}")


def _load_rng(ckpt_dir: str) -> None:
    """Restore this rank's RNG state if present (best-effort).

    Skips silently when the file is absent (older checkpoint, or a resume with
    a different world size than the save) so resume never hard-fails on RNG.
    """
    path = os.path.join(ckpt_dir, f"rng_rank{_rank()}.pt")
    if not os.path.exists(path):
        logger.warning(f"No RNG state for rank {_rank()} at {path}; resume RNG not restored.")
        return
    try:
        state = torch.load(path, map_location="cpu")
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        import random as _random

        if "python" in state:
            _random.setstate(state["python"])
        if "numpy" in state:
            import numpy as _np

            _np.random.set_state(state["numpy"])
    except Exception as e:
        logger.warning(f"RNG state restore skipped: {e}")


def _task_root(save_dir: str, task: str) -> str:
    return os.path.join(save_dir, task)


def _is_committed(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, COMMITTED_MARKER))


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(
    save_dir: str,
    task: str,
    iteration: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    trainer_state: Dict[str, Any],
    weight_sync_manifest: Dict[str, Any],
    adapter_contract: Dict[str, Any],
    adapter_only: bool = False,
    is_rank0: bool = True,
) -> str:
    """Save a committed checkpoint and return its directory.

    ``trainer_state`` must carry the resume-critical fields (rollout/optimizer
    step, scheduler, RNG, dataset cursor, policy version, sampling fingerprint,
    adapter contract hash — design doc 12). The DCP shard write is collective;
    the JSON sidecars and the COMMITTED marker are written by rank 0 only.

    ``adapter_only`` (LoRA runs) drops the frozen base from the shard write —
    the base is reloaded from ``--model-path`` on resume, so persisting it every
    interval would multiply a sub-GB adapter into a >100 GB full-model snapshot.
    :func:`load_checkpoint` must be called with the matching flag.
    """
    task_root = _task_root(save_dir, task)
    ckpt_dir = os.path.join(task_root, iter_dir_name(iteration))
    staging_dir = os.path.join(task_root, f".{iter_dir_name(iteration)}.inprogress")
    fsdp_dir = os.path.join(staging_dir, "fsdp")
    decision: Dict[str, Any] = {"action": "save"}
    if is_rank0 and _is_committed(ckpt_dir):
        try:
            expected = {
                "trainer_state.json": trainer_state,
                "weight_sync_manifest.json": weight_sync_manifest,
                "adapter_contract.json": adapter_contract,
            }
            existing = {}
            for name in expected:
                with open(os.path.join(ckpt_dir, name), encoding="utf-8") as f:
                    existing[name] = json.load(f)
            if existing == expected:
                decision = {"action": "return"}
            else:
                decision = {
                    "action": "error",
                    "message": f"Refusing to overwrite committed checkpoint with different metadata: {ckpt_dir}",
                }
        except Exception as exc:
            decision = {
                "action": "error",
                "message": f"Cannot validate existing committed checkpoint {ckpt_dir}: {type(exc).__name__}: {exc}",
            }
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size(dist.group.WORLD) > 1:
        from relax.utils.distributed_utils import get_gloo_group

        group = get_gloo_group()
        payload = [decision if is_rank0 else None]
        dist.broadcast_object_list(payload, src=0, group=group)
        received_decision = payload[0]
        if received_decision is None:
            raise RuntimeError("Checkpoint decision broadcast returned no rank-0 payload.")
        decision = received_decision
    if decision["action"] == "return":
        logger.info(f"Checkpoint already committed with identical metadata: {ckpt_dir}")
        return ckpt_dir
    if decision["action"] == "error":
        raise RuntimeError(decision["message"])

    setup_error: Exception | None = None
    if is_rank0:
        try:
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
            os.makedirs(fsdp_dir, exist_ok=True)
        except Exception as exc:
            setup_error = exc
    # This outcome collective also makes the directory creation visible before
    # peers build/write their state; a rank-0 mkdir failure cannot strand peers
    # in get_state_dict or DCP.
    if not _all_ranks_ok(0 if setup_error is not None else 1):
        if setup_error is not None:
            raise setup_error
        raise RuntimeError(f"Checkpoint directory creation failed on rank 0: {fsdp_dir}.")

    state_error: Exception | None = None
    model_sd: Dict[str, Any] = {}
    optim_sd: Dict[str, Any] = {}
    try:
        model_sd, optim_sd = get_state_dict(
            model,
            optimizers=optimizer if optimizer is not None else [],
            options=StateDictOptions(cpu_offload=True, ignore_frozen_params=adapter_only),
        )
    except Exception as exc:
        state_error = exc
    if not _all_ranks_ok(0 if state_error is not None else 1):
        if state_error is not None:
            raise state_error
        raise RuntimeError(f"Checkpoint state-dict preparation failed on another rank: {staging_dir}.")
    state_dict: Dict[str, Any] = {"model": model_sd}
    if optimizer is not None:
        state_dict["optim"] = optim_sd

    # Collective commit agreement: only write the COMMITTED marker if EVERY rank
    # wrote its shards. Otherwise a shard write that fails on one rank only (e.g.
    # that rank's local FS is full) while rank 0 succeeds would still be marked
    # committed → an unresumable "committed" checkpoint. On failure every rank
    # raises the SAME error (the flag is MIN-reduced), so the caller's try/except
    # runs symmetrically across ranks.
    ok = 1
    try:
        dcp.save(state_dict, checkpoint_id=fsdp_dir)
    except Exception:
        ok = 0
        logger.exception(f"DCP shard write failed on rank for {fsdp_dir}")
    ok = _all_ranks_ok(ok)
    if not ok:
        raise RuntimeError(f"DCP save failed on at least one rank; not committing {ckpt_dir}.")

    # Each rank persists its own RNG state in the staging directory for
    # deterministic resume.
    _save_rng(staging_dir)
    # Do not let rank 0 rename staging while another rank still has its RNG file
    # open. The agreement collective also serves as the required barrier.
    _all_ranks_ok(1)

    publish_error: Exception | None = None
    if is_rank0:
        try:
            trainer_state_path = os.path.join(staging_dir, "trainer_state.json")
            weight_manifest_path = os.path.join(staging_dir, "weight_sync_manifest.json")
            adapter_contract_path = os.path.join(staging_dir, "adapter_contract.json")
            committed_path = os.path.join(staging_dir, COMMITTED_MARKER)
            _write_json(trainer_state_path, trainer_state)
            _write_json(weight_manifest_path, weight_sync_manifest)
            _write_json(adapter_contract_path, adapter_contract)
            required_paths = [
                os.path.join(fsdp_dir, ".metadata"),
                trainer_state_path,
                weight_manifest_path,
                adapter_contract_path,
            ]
            missing = [path for path in required_paths if not os.path.isfile(path)]
            if missing:
                raise RuntimeError(f"Checkpoint publication is missing required files: {missing}.")
            # COMMITTED is the last write inside staging. The directory is still
            # hidden from discovery; promotion below makes the complete snapshot
            # visible in one rename instead of authenticating a partial overwrite.
            with open(committed_path, "w", encoding="utf-8") as f:
                f.write(str(int(iteration)))
                f.flush()
                os.fsync(f.fileno())

            backup_dir = os.path.join(task_root, f".{iter_dir_name(iteration)}.backup")
            shutil.rmtree(backup_dir, ignore_errors=True)
            moved_old = False
            try:
                if os.path.exists(ckpt_dir):
                    os.replace(ckpt_dir, backup_dir)
                    moved_old = True
                os.replace(staging_dir, ckpt_dir)
            except Exception:
                if moved_old and not os.path.exists(ckpt_dir) and os.path.exists(backup_dir):
                    os.replace(backup_dir, ckpt_dir)
                raise
            if moved_old:
                shutil.rmtree(backup_dir, ignore_errors=True)
            _write_latest_markers(save_dir, task, iteration)
        except Exception as exc:
            publish_error = exc
            logger.exception(f"Checkpoint metadata publication failed for {ckpt_dir}")
    published = _all_ranks_ok(0 if publish_error is not None else 1)
    if not published:
        if publish_error is not None:
            raise publish_error
        raise RuntimeError(f"Checkpoint metadata publication failed on rank 0; not publishing {ckpt_dir}.")
    logger.info(f"Saved committed checkpoint: {ckpt_dir}")
    return ckpt_dir


def _write_latest_markers(save_dir: str, task: str, iteration: int) -> None:
    """Publish ``latest_checkpointed_iteration.txt`` at both levels.

    The task-scoped copy under ``<save>/<task>/`` records which iteration this
    task last committed. The copy at ``<save>/`` is the FRAMEWORK contract:
    :func:`relax.utils.utils.recovery_load_path` looks for exactly that path on
    restart and only then sets ``args.load = args.save``. Writing only the
    task-scoped copy -- as this backend originally did -- means the restart
    hook never fires, ``args.load`` stays unset, ``_maybe_resume`` returns 0,
    and a restarted job silently trains from scratch with a fully populated
    checkpoint sitting on disk.
    """
    for root in (_task_root(save_dir, task), save_dir):
        _write_atomic_latest(root, iteration)


def _write_atomic_latest(root: str, iteration: int) -> None:
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, _LATEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(iteration)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(root, _LATEST))


def find_latest_committed(save_dir: str, task: str) -> Optional[int]:
    """Return the highest iteration with a COMMITTED marker, or None."""
    task_root = _task_root(save_dir, task)
    if not os.path.isdir(task_root):
        return None
    candidates = []
    for entry in os.listdir(task_root):
        if entry.startswith("iter_") and _is_committed(os.path.join(task_root, entry)):
            try:
                candidates.append(int(entry.split("_", 1)[1]))
            except ValueError:
                continue
    return max(candidates) if candidates else None


def load_checkpoint(
    save_dir: str,
    task: str,
    iteration: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    adapter_only: bool = False,
) -> Dict[str, Any]:
    """Restore model/optimizer from a committed checkpoint; return
    trainer_state.

    Raises if the target directory is missing its COMMITTED marker — only
    committed checkpoints are resumable (design doc 12).

    ``adapter_only`` must match what :func:`save_checkpoint` was given. Asking
    DCP for the frozen base keys that an adapter-only save never wrote makes the
    load raise on the first missing key — and only at the *first resume*, long
    after a run has accumulated apparently-successful checkpoints.
    """
    ckpt_dir = os.path.join(_task_root(save_dir, task), iter_dir_name(iteration))
    if not _is_committed(ckpt_dir):
        raise FileNotFoundError(f"Refusing to resume from uncommitted checkpoint: {ckpt_dir}")
    fsdp_dir = os.path.join(ckpt_dir, "fsdp")

    options = StateDictOptions(ignore_frozen_params=adapter_only, strict=not adapter_only)
    model_sd, optim_sd = get_state_dict(
        model,
        optimizers=optimizer if optimizer is not None else [],
        options=options,
    )
    state_dict: Dict[str, Any] = {"model": model_sd}
    if optimizer is not None:
        state_dict["optim"] = optim_sd
    dcp.load(state_dict, checkpoint_id=fsdp_dir)
    set_state_dict(
        model,
        optimizers=optimizer if optimizer is not None else [],
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict.get("optim", {}),
        options=options,
    )

    # Restore this rank's RNG state (best-effort) for deterministic resume.
    _load_rng(ckpt_dir)

    with open(os.path.join(ckpt_dir, "trainer_state.json"), encoding="utf-8") as f:
        return json.load(f)


def read_adapter_contract(save_dir: str, task: str, iteration: int) -> Dict[str, Any]:
    """Read ``adapter_contract.json`` for a checkpoint; ``{}`` when absent."""
    path = os.path.join(_task_root(save_dir, task), iter_dir_name(iteration), "adapter_contract.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_safetensors_dir(directory: str) -> Dict[str, torch.Tensor]:
    """Load every ``*.safetensors`` shard in ``directory`` into one CPU
    dict."""
    from safetensors.torch import load_file

    shards = sorted(f for f in os.listdir(directory) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors found under {directory}.")
    out: Dict[str, torch.Tensor] = {}
    for shard in shards:
        out.update(load_file(os.path.join(directory, shard)))
    return out


def _fold_adapter_into_base(
    base: Dict[str, torch.Tensor],
    adapter: Dict[str, torch.Tensor],
    lora_meta: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Return ``base`` with every LoRA pair in ``adapter`` folded in.

    Raises on an orphan ``lora_A``/``lora_B`` or a missing base weight rather
    than dropping it: a silently skipped fold produces a checkpoint that loads
    fine and generates as if untrained.
    """
    from relax.backends.fsdp.lora import fold_lora_delta

    scaling = float(lora_meta.get("scaling") or 0.0)
    if scaling <= 0.0:
        rank, alpha = int(lora_meta.get("rank", 0)), float(lora_meta.get("alpha", 0))
        if rank <= 0:
            raise ValueError(f"Cannot fold LoRA: contract has no usable rank/alpha ({lora_meta!r}).")
        scaling = alpha / rank

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in adapter.items():
        for segment, kind in ((".lora_A.", "lora_A"), (".lora_B.", "lora_B")):
            if segment in name:
                pairs.setdefault(name.split(segment)[0], {})[kind] = tensor
                break

    merged = dict(base)
    for stem, pair in sorted(pairs.items()):
        if "lora_A" not in pair or "lora_B" not in pair:
            raise ValueError(f"LoRA module {stem!r} in the checkpoint is missing one of lora_A/lora_B.")
        target = f"{stem}.weight"
        if target not in merged:
            raise ValueError(f"LoRA module {stem!r} has no base weight {target!r} in the base model.")
        merged[target] = fold_lora_delta(merged[target], pair["lora_A"], pair["lora_B"], scaling)
    logger.info(f"Folded {len(pairs)} LoRA pairs into the base weights (scaling={scaling:.4f})")
    return merged


def _load_dcp_model_state(fsdp_dir: str) -> Dict[str, torch.Tensor]:
    """Read the ``model`` half of a DCP checkpoint into a flat CPU dict.

    ``dcp.load({"model": {}})`` is a silent no-op: DCP fills the tensors a
    destination dict already declares, so an empty one loads nothing and returns
    successfully. The destination therefore has to be pre-allocated from the
    checkpoint's own metadata.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    metadata = FileSystemReader(fsdp_dir).read_metadata()
    prefix = "model."
    destination: Dict[str, torch.Tensor] = {}
    for key, entry in metadata.state_dict_metadata.items():
        if not key.startswith(prefix) or not isinstance(entry, TensorStorageMetadata):
            continue
        destination[key[len(prefix) :]] = torch.empty(entry.size, dtype=entry.properties.dtype)
    if destination:
        dcp.load({"model": destination}, checkpoint_id=fsdp_dir)
    return destination


def export_hf(
    save_dir: str,
    task: str,
    iteration: int,
    output_dir: str,
    *,
    name_map: Optional[Any] = None,
    base_model_path: Optional[str] = None,
    subfolder: str = "transformer",
    max_shard_size_bytes: int = 5 * 1024**3,
) -> str:
    """Offline export of a committed DCP checkpoint to a loadable diffusers
    tree.

    Loads the DCP model state onto CPU (no rank-0 whole-model aggregation on
    the train hot path — this is an offline tool), renames parameters via the
    adapter ``name_map``, and writes **sharded** safetensors plus the
    ``model.safetensors.index.json`` weight map that ``from_pretrained``
    requires.

    When ``base_model_path`` is given, the frozen components the trainer never
    touches (VAE, text encoder, scheduler, tokenizer, ``model_index.json``, and
    the transformer's own ``config.json``) are copied across, so ``output_dir``
    is a directly loadable pipeline rather than a bare tensor dump. The trained
    transformer lands in ``<output_dir>/<subfolder>/``.

    For a LoRA (``save_mode == "adapter"``) checkpoint the DCP dict holds *only*
    adapter tensors, so ``base_model_path`` becomes mandatory and the adapter is
    folded into the base here. Without that check the export would cheerfully
    write a "transformer" made entirely of ``lora_A``/``lora_B`` tensors, which
    is non-empty and finite and therefore passes every downstream sanity check.

    Returns ``output_dir``.
    """

    ckpt_dir = os.path.join(_task_root(save_dir, task), iter_dir_name(iteration))
    if not _is_committed(ckpt_dir):
        raise FileNotFoundError(f"Refusing to export uncommitted checkpoint: {ckpt_dir}")
    fsdp_dir = os.path.join(ckpt_dir, "fsdp")

    # Read the sharded model state into a flat CPU state dict.
    model_state = _load_dcp_model_state(fsdp_dir)
    renamed = {(name_map(k) if callable(name_map) else k): v.contiguous().cpu() for k, v in model_state.items()}
    if not renamed:
        raise RuntimeError(f"No tensors loaded from {fsdp_dir}; refusing to write an empty export.")

    contract = read_adapter_contract(save_dir, task, iteration)
    if contract.get("save_mode") == "adapter":
        if not base_model_path:
            raise ValueError(
                f"{ckpt_dir} is a LoRA (adapter-only) checkpoint: base_model_path is required so the "
                "adapter can be folded into the base weights. Exporting without it would write a "
                "transformer containing only lora_A/lora_B tensors."
            )
        base = _read_safetensors_dir(os.path.join(base_model_path, subfolder))
        renamed = _fold_adapter_into_base(base, renamed, contract.get("lora") or {})

    # The trained transformer goes in its own subfolder so the result mirrors the
    # base model's layout (diffusers loads the DiT from <root>/transformer/).
    transformer_dir = os.path.join(output_dir, subfolder) if base_model_path else output_dir
    os.makedirs(transformer_dir, exist_ok=True)
    _write_sharded_safetensors(renamed, transformer_dir, max_shard_size_bytes)

    if base_model_path:
        _copy_frozen_pipeline_components(base_model_path, output_dir, subfolder)

    logger.info(f"Exported {len(renamed)} tensors to {transformer_dir}")
    return output_dir


def export_peft_adapter(save_dir: str, task: str, iteration: int, output_dir: str) -> str:
    """Export a LoRA checkpoint as a portable HF-PEFT adapter directory.

    Produces ``adapter_config.json`` + ``adapter_model.safetensors``, loadable
    by ``peft.PeftModel.from_pretrained``. Complements :func:`export_hf`, which
    bakes the adapter into a full diffusers tree instead.
    """
    from relax.backends.fsdp.lora import peft_adapter_state_dict
    from relax.utils.megatron_peft_utils import write_hf_peft_adapter

    ckpt_dir = os.path.join(_task_root(save_dir, task), iter_dir_name(iteration))
    if not _is_committed(ckpt_dir):
        raise FileNotFoundError(f"Refusing to export uncommitted checkpoint: {ckpt_dir}")
    contract = read_adapter_contract(save_dir, task, iteration)
    if contract.get("save_mode") != "adapter":
        raise ValueError(f"{ckpt_dir} is not a LoRA checkpoint (save_mode={contract.get('save_mode')!r}).")
    lora_meta = contract.get("lora") or {}

    model_state = _load_dcp_model_state(os.path.join(ckpt_dir, "fsdp"))
    # The DCP keys already carry PEFT's ``.lora_A.<adapter>.weight`` shape.
    renamed = peft_adapter_state_dict(
        [(k, v.contiguous().cpu()) for k, v in model_state.items()],
        adapter_name=str(lora_meta.get("adapter_name", "default")),
    )
    if not renamed:
        raise RuntimeError(f"No LoRA tensors found in {ckpt_dir}; refusing to write an empty adapter.")

    write_hf_peft_adapter(
        renamed,
        output_dir,
        lora_rank=int(lora_meta["rank"]),
        lora_alpha=int(lora_meta["alpha"]),
        target_modules=lora_meta.get("target_modules") or [],
        lora_dropout=float(lora_meta.get("dropout", 0.0)),
        task_type=str(lora_meta.get("task_type", "FEATURE_EXTRACTION")),
    )
    logger.info(f"Exported {len(renamed)} adapter tensors to {output_dir}")
    return output_dir


def _write_sharded_safetensors(tensors: Dict[str, torch.Tensor], out_dir: str, max_shard_size_bytes: int) -> None:
    """Write ``tensors`` as sharded safetensors + an index the loader can read.

    A single flat ``model.safetensors`` is not loadable for multi-shard-sized
    models and carries no weight map; ``from_pretrained`` looks for
    ``model.safetensors.index.json`` whenever more than one shard exists.
    """
    from safetensors.torch import save_file

    shards: List[Dict[str, torch.Tensor]] = [{}]
    shard_bytes = [0]
    for name in sorted(tensors):
        tensor = tensors[name]
        nbytes = tensor.numel() * tensor.element_size()
        if shards[-1] and shard_bytes[-1] + nbytes > max_shard_size_bytes:
            shards.append({})
            shard_bytes.append(0)
        shards[-1][name] = tensor
        shard_bytes[-1] += nbytes

    total_bytes = sum(shard_bytes)
    if len(shards) == 1:
        save_file(shards[0], os.path.join(out_dir, "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
        return

    weight_map: Dict[str, str] = {}
    total = len(shards)
    for i, shard in enumerate(shards, start=1):
        filename = f"diffusion_pytorch_model-{i:05d}-of-{total:05d}.safetensors"
        save_file(shard, os.path.join(out_dir, filename), metadata={"format": "pt"})
        for name in shard:
            weight_map[name] = filename
    _write_json(
        os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json"),
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
    )


def _copy_frozen_pipeline_components(base_model_path: str, output_dir: str, subfolder: str) -> None:
    """Copy the untrained pipeline pieces from the base model into
    ``output_dir``.

    The FSDP actor only ever trains the transformer, so VAE / text encoder /
    scheduler / tokenizer are byte-identical to the base model. Copying them
    (plus ``model_index.json`` and the transformer's ``config.json``) is what
    makes the export a runnable pipeline. Best-effort per entry: a missing
    component is warned about, not fatal.
    """
    import shutil

    if not os.path.isdir(base_model_path):
        logger.warning(f"base_model_path {base_model_path!r} is not a directory; skipping frozen-component copy.")
        return

    # The transformer's config.json must sit beside the exported weights.
    src_cfg = os.path.join(base_model_path, subfolder, "config.json")
    if os.path.exists(src_cfg):
        shutil.copy2(src_cfg, os.path.join(output_dir, subfolder, "config.json"))
    else:
        logger.warning(f"No transformer config.json at {src_cfg}; the export may not load.")

    for entry in sorted(os.listdir(base_model_path)):
        src = os.path.join(base_model_path, entry)
        dst = os.path.join(output_dir, entry)
        if entry == subfolder:
            continue  # trained weights already written
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif entry.endswith((".json", ".txt", ".model")):
            shutil.copy2(src, dst)
    logger.info(f"Copied frozen pipeline components from {base_model_path}")
