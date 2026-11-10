# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""DCP-sharded checkpoints for FSDPTurbo, with a legacy HF fallback.

Every rank writes and reads only its shards. ``RELAX_FSDPTURBO_CKPT_DCP=0``
selects rank-0 HF gather with per-rank VP n-gram shards;
``RELAX_FSDPTURBO_CKPT_EXPORT_HF=1`` adds an external-only HF export. The
resume marker preserves the framework ``recovery_load_path`` contract.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.distributed_utils import get_gloo_group
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_TURBO_PREFIX = "model."
_LATEST = "latest_checkpointed_iteration.txt"
_DCP_METADATA = ".metadata"

# DCP is optional on older torch builds; fallback is the legacy HF gather path.
# pylint: disable=possibly-used-before-assignment,ungrouped-imports
try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.stateful import Stateful

    _HAS_DCP = True
except ImportError:  # pragma: no cover - torch < 2.2
    _HAS_DCP = False


def _write_atomic_latest(root: str, rollout_id: int) -> None:
    """Atomically publish the marker required by ``recovery_load_path``."""
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, _LATEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(rollout_id)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(root, _LATEST))


def _strip_turbo_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """``FSDPTurbo.model.<HF name>`` -> ``<HF name>``."""
    return {
        (name[len(_TURBO_PREFIX) :] if name.startswith(_TURBO_PREFIX) else name): tensor
        for name, tensor in state_dict.items()
    }


def _get_full_state_dicts(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    # Gather full tensors into rank-0 host RAM; saving local shards here would
    # produce an unresumable file. Other ranks do not hold a full copy.
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    return _strip_turbo_prefix(model_state), optimizer_state


def _storage_probe_ok(tensor: torch.Tensor) -> bool:
    """Check both storage APIs used by safetensors.

    NPU forks may raise exceptions other than ``RuntimeError`` here.
    """
    try:
        tensor.untyped_storage().data_ptr()
        tensor.storage().data_ptr()
        return True
    except Exception:
        return False


def _materialize_fresh(tensor: torch.Tensor) -> torch.Tensor | None:
    """Copy ``tensor`` into fresh, valid storage.

    ``clone()`` may retain invalid Python storage on this stack; each fallback
    forces a fresh allocation and is validated before use.
    """

    def _via_copy() -> torch.Tensor:
        fresh = torch.empty_like(tensor)
        fresh.copy_(tensor)
        return fresh

    def _via_roundtrip() -> torch.Tensor:
        return tensor.to(torch.float32).to(tensor.dtype)

    def _via_pickle() -> torch.Tensor:
        buf = io.BytesIO()
        torch.save(tensor, buf)
        buf.seek(0)
        return torch.load(buf, weights_only=True)

    for make in (_via_copy, _via_roundtrip, _via_pickle):
        try:
            fresh = make()
        except Exception:
            continue
        if _storage_probe_ok(fresh):
            return fresh
    return None


def _to_valid_cpu_tensor(
    name: str, tensor: Any, materialized: list[str], unfixable: list[str], dropped_meta: list[str]
) -> torch.Tensor | None:
    """Normalize an entry for safetensors' storage-pointer accesses.

    Drop meta tensors and copy device tensors to CPU. Keep valid CPU tensors by
    reference to avoid duplicating the ~100 GB frozen PLE table; rebuild
    invalid storage when possible. Unfixable tensors remain for the torch.save
    fallback, which does not probe Python storage pointers.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    tensor = tensor.detach()
    if tensor.device.type == "meta":
        dropped_meta.append(name)
        return None
    if tensor.device.type != "cpu":
        tensor = tensor.to("cpu")
    if not _storage_probe_ok(tensor):
        fresh = _materialize_fresh(tensor)
        if fresh is not None:
            materialized.append(name)
            return fresh
        unfixable.append(name)
    return tensor


def _strip_param_prefix(name: str) -> str:
    """``model.language_model...`` -> ``language_model...`` (single key)."""
    return name[len(_TURBO_PREFIX) :] if name.startswith(_TURBO_PREFIX) else name


def _find_vp_table_params(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Find VP tables whose full gather would not fit on one device.

    The qwen4_exp table is ~95 GiB and must be saved as local shards. Test mesh
    membership rather than equality because a VP sub-mesh may expose the root
    mesh's dimension names.
    """
    vp_params: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        spec = getattr(param, "_spec", None)
        if spec is None:
            continue
        mesh_dim_names = tuple(getattr(spec.mesh, "mesh_dim_names", None) or ())
        if "vp" in mesh_dim_names:
            vp_params[name] = param
    return vp_params


def _find_frozen_ngram_param_names(model: torch.nn.Module) -> set[str]:
    """Find frozen, replicated PLE tables used without ``vp_plan``.

    Each ~102 GiB copy is rebuilt deterministically from HF, so checkpointing
    it would waste host memory on save and resume. VP-sharded tables still use
    the regular sharded DCP path.
    """
    names: set[str] = set()
    for name, param in model.named_parameters():
        if "ngram_embedding" in name and getattr(param, "_spec", None) is None and not param.requires_grad:
            names.add(name)
    return names


def _detach_vp_tables_for_gather(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, vp_params: dict[str, torch.nn.Parameter]
) -> tuple:
    """Temporarily hide VP tables from the full-gather view.

    Remove them from both optimizer groups and state for consistency. Their
    ~190 GiB of moments are not persisted and re-zero on resume. Return a
    receipt for exact restoration after the gather.
    """
    module_stash: dict[str, tuple[torch.nn.Module, str, torch.nn.Parameter]] = {}
    for name, param in vp_params.items():
        module_path, _, attr = name.rpartition(".")
        module = model.get_submodule(module_path)
        module_stash[name] = (module, attr, param)
        module._parameters[attr] = None
    vp_set = set(vp_params.values())
    state_stash = {}
    for p in vp_set:
        if p in optimizer.state:
            state_stash[p] = optimizer.state.pop(p)
    group_slots: list[tuple[int, int, torch.nn.Parameter]] = []
    for group_idx, group in enumerate(optimizer.param_groups):
        kept = []
        for position, p in enumerate(group["params"]):
            if p in vp_set:
                group_slots.append((group_idx, position, p))
            else:
                kept.append(p)
        group["params"] = kept
    return module_stash, state_stash, group_slots


def _restore_vp_tables_after_gather(model: torch.nn.Module, optimizer: torch.optim.Optimizer, receipt: tuple) -> None:
    """Put the detached VP tables back exactly where they came from."""
    module_stash, state_stash, group_slots = receipt
    for group_idx, position, p in reversed(group_slots):
        optimizer.param_groups[group_idx]["params"].insert(position, p)
    for p, state in state_stash.items():
        optimizer.state[p] = state
    for _name, (module, attr, param) in module_stash.items():
        module._parameters[attr] = param


# DCP sharded path (default).

if _HAS_DCP:

    class _AppState(Stateful):
        """Wrap model, optimizer and resume id for one DCP operation.

        DCP fills the sharded template through storage shared with live state,
        then ``load_state_dict`` formalizes the restore.
        """

        def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, rollout_id: int = 0):
            self.model = model
            self.optimizer = optimizer
            self.rollout_id = int(rollout_id)

        def state_dict(self) -> dict[str, Any]:
            from torch.distributed.checkpoint.state_dict import get_state_dict

            # Sharded defaults avoid materializing the ~95 GiB VP table in full.
            model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
            # Frozen replicated tables are rebuilt from HF and need no checkpoint copy.
            dropped = _find_frozen_ngram_param_names(self.model) & model_sd.keys()
            for key in dropped:
                del model_sd[key]
            if dropped:
                logger.info(
                    f"[fsdp_turbo] save_checkpoint: excluded {len(dropped)} frozen n-gram table(s) from "
                    "the DCP save (e.g. "
                    f"'{sorted(dropped)[0]}'); they are re-materialized from the HF dir on resume"
                )
            return {"model": model_sd, "optim": optim_sd, "rollout_id": self.rollout_id}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

            model_sd = state_dict.get("model", {})
            # Re-inject skipped frozen tables so strict restore sees a complete mapping.
            missing_ngram = _find_frozen_ngram_param_names(self.model) - model_sd.keys()
            if missing_ngram:
                params = dict(self.model.named_parameters())
                for name in missing_ngram:
                    model_sd[name] = params[name]
            try:
                set_state_dict(
                    self.model,
                    self.optimizer,
                    model_state_dict=model_sd,
                    optim_state_dict=state_dict.get("optim", {}),
                    options=StateDictOptions(strict=True),
                )
            except RuntimeError as e:
                if not _is_fork_set_failure(str(e)):
                    raise
                # This known fork refusal happens after DCP has written values
                # through shared storage, so the restored weights remain intact.
                logger.warning(
                    "[fsdp_turbo] load_checkpoint: set_state_dict refused with a known fork "
                    f"signature after the DCP load ({e}); values were already landed in place "
                    "by the DCP reader itself"
                )
            self.rollout_id = int(state_dict.get("rollout_id", 0))

    class _OptimStateTolerantLoadPlanner(DefaultLoadPlanner):
        """Drop absent lazy Adam state and frozen replicated n-gram reads.

        Parameters that never received gradients keep fresh zero state. The
        ~102 GiB frozen tables are identical to their HF-built values and would
        otherwise be read in full per rank. VP tables and all other keys remain
        strict so real mismatches still fail.
        """

        _OPTIM_STATE_KEY_PREFIX = "app.optim.state."
        _MODEL_KEY_PREFIX = "app.model."

        @staticmethod
        def _is_frozen_plain_ngram(key: str, value: Any) -> bool:
            return (
                key.startswith(_OptimStateTolerantLoadPlanner._MODEL_KEY_PREFIX)
                and "ngram_embedding" in key
                and isinstance(value, torch.Tensor)
                and not hasattr(value, "_local_tensor")  # not a DTensor (not VP-sharded)
                and not value.requires_grad  # frozen: identical to the build-time HF values
            )

        def create_local_plan(self):  # type: ignore[override]
            if self.metadata is not None and self.flatten_state_dict:
                saved_keys = set(self.metadata.state_dict_metadata.keys())
                dropped = [
                    key
                    for key, value in list(self.state_dict.items())
                    if (key not in saved_keys and key.startswith(self._OPTIM_STATE_KEY_PREFIX))
                    or self._is_frozen_plain_ngram(key, value)
                ]
                for key in dropped:
                    del self.state_dict[key]
                if dropped:
                    logger.info(
                        "[fsdp_turbo] load_checkpoint: "
                        f"{len(dropped)} key(s) excluded from the DCP load (fresh optimizer state / "
                        f"frozen n-gram tables identical to the HF materialization, e.g. '{dropped[0]}'); "
                        "keeping current values for them"
                    )
            return super().create_local_plan()

    def _save_dcp(
        output_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, rollout_id: int
    ) -> None:
        """Save local shards; published ``.metadata`` certifies all ranks
        finished."""
        os.makedirs(output_dir, exist_ok=True)
        dcp.save({"app": _AppState(model, optimizer, rollout_id)}, checkpoint_id=output_dir)

    def _load_dcp(load_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> int:
        """Resume DCP shards in place and return ``saved_rollout_id + 1``.

        Host memory stays at one local shard per rank.
        """
        app = _AppState(model, optimizer, 0)
        dcp.load({"app": app}, checkpoint_id=load_dir, planner=_OptimStateTolerantLoadPlanner())
        logger.info(
            f"[fsdp_turbo] load_checkpoint: resumed DCP sharded checkpoint from {load_dir} "
            f"(next rollout_id={app.rollout_id + 1})"
        )
        return app.rollout_id + 1


def _save_vp_shards(output_dir: str, vp_params: dict[str, torch.nn.Parameter], rollout_id: int) -> None:
    """Write per-rank VP shards without a full gather.

    Rank 0 publishes ``meta.json`` after a barrier to certify all shard files;
    loading rejects mismatched VP configurations.
    """
    from safetensors.torch import save_file

    rank = dist.get_rank()
    shard_dir = os.path.join(output_dir, "ngram_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard = {
        _strip_param_prefix(name): param.to_local().detach().cpu().contiguous() for name, param in vp_params.items()
    }
    save_file(shard, os.path.join(shard_dir, f"rank-{rank:05d}.safetensors"), metadata={"format": "pt"})
    # Rank 0 publishes metadata only after every shard file is complete.
    dist.barrier(group=get_gloo_group())
    if rank == 0:
        first_mesh = next(iter(vp_params.values()))._spec.mesh
        meta = {
            "rollout_id": int(rollout_id),
            "vp_size": int(first_mesh.size(0)),
            "params": {
                _strip_param_prefix(name): {
                    "global_shape": list(param.shape),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "local_shape": list(param.to_local().shape),
                }
                for name, param in vp_params.items()
            },
        }
        meta_path = os.path.join(shard_dir, "meta.json")
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp_path, meta_path)
        logger.info(
            f"[fsdp_turbo] save_checkpoint: wrote {len(vp_params)} VP table param(s) as "
            f"{dist.get_world_size()} per-rank shards under {shard_dir} (optimizer moments NOT persisted)"
        )


def _write_hf_model_file(output_dir: str, model_state: dict[str, Any]) -> None:
    """Write a fully gathered model state as safetensors.

    Rebuild invalid Python storage and retry once. Fall back to ``model.bin``
    because torch.save does not probe Python storage pointers.
    """
    cleaned_state: dict[str, torch.Tensor] = {}
    materialized: list[str] = []
    unfixable: list[str] = []
    dropped_meta: list[str] = []
    for name, tensor in model_state.items():
        valid_tensor = _to_valid_cpu_tensor(name, tensor, materialized, unfixable, dropped_meta)
        if valid_tensor is not None:
            cleaned_state[name] = valid_tensor
    model_state = cleaned_state
    if dropped_meta:
        logger.warning(
            f"[fsdp_turbo] save_checkpoint: dropped {len(dropped_meta)} meta-device tensor(s) — they are "
            f"NOT saved; resume refills them from current values (e.g. {dropped_meta[:3]})"
        )
    if materialized:
        logger.warning(
            f"[fsdp_turbo] save_checkpoint: re-materialized {len(materialized)} state-dict tensors with "
            f"invalid python storage before safetensors save (e.g. '{materialized[0]}' .. "
            f"'{materialized[-1]}')"
        )
    if unfixable:
        logger.warning(
            f"[fsdp_turbo] save_checkpoint: {len(unfixable)} tensor(s) could not be re-materialized and "
            f"need the torch.save fallback (e.g. {unfixable[:3]})"
        )
    from safetensors.torch import save_file

    output_path = os.path.join(output_dir, "model.safetensors")
    safetensors_ok = False
    try:
        save_file(model_state, output_path, metadata={"format": "pt"})
        safetensors_ok = True
    except RuntimeError as e:
        if "invalid python storage" not in str(e):
            raise
        # Storage may become invalid after the pre-pass; rebuild and retry once.
        still_bad = [name for name, t in model_state.items() if not _storage_probe_ok(t)]
        unfixed: list[str] = []
        for name in still_bad:
            fresh = _materialize_fresh(model_state[name])
            if fresh is None:
                unfixed.append(name)
            else:
                model_state[name] = fresh
        logger.warning(
            f"[fsdp_turbo] save_checkpoint: save_file raised 'invalid python storage' after the pre-pass; "
            f"re-materialized {len(still_bad) - len(unfixed)}/{len(still_bad)} offender(s), retrying once"
        )
        if not unfixed:
            try:
                save_file(model_state, output_path, metadata={"format": "pt"})
                safetensors_ok = True
            except Exception as retry_err:
                logger.warning(f"[fsdp_turbo] save_checkpoint: save_file retry failed too: {retry_err}")
        if not safetensors_ok:
            # Remove partial safetensors output before the torch.save fallback.
            try:
                os.remove(output_path)
            except OSError:
                pass
            bin_path = os.path.join(output_dir, "model.bin")
            logger.warning(f"[fsdp_turbo] save_checkpoint: falling back to torch.save at {bin_path}")
            torch.save(model_state, bin_path)


def _export_hf_weights(
    output_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, rollout_id: int
) -> None:
    """Export HF weights beside DCP for external consumers only.

    This rank-0 gather uses host RAM and bypasses VP tables into shards; keep
    it disabled for large models. Resume always reads DCP.
    """
    vp_params = _find_vp_table_params(model)
    detach_receipt = _detach_vp_tables_for_gather(model, optimizer, vp_params)
    try:
        model_state, _optimizer_state = _get_full_state_dicts(model, optimizer)
    finally:
        _restore_vp_tables_after_gather(model, optimizer, detach_receipt)
    if vp_params:
        _save_vp_shards(output_dir, vp_params, rollout_id)
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        _write_hf_model_file(output_dir, model_state)
        logger.info(
            "[fsdp_turbo] save_checkpoint: HF weight export written beside the DCP checkpoint "
            "(RELAX_FSDPTURBO_CKPT_EXPORT_HF); resume does NOT read it"
        )


def save_checkpoint(
    args,
    rollout_id: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    hf_config,
    tokenizer,
) -> None:
    """Save model and optimizer state under ``args.save/<step>``.

    DCP shards are the default; ``RELAX_FSDPTURBO_CKPT_DCP=0`` selects the
    legacy rank-0 HF gather.
    """
    if args.save is None:
        return
    output_dir = os.path.join(args.save, str(rollout_id))
    rank = dist.get_rank()

    use_dcp = Envs.RELAX_FSDPTURBO_CKPT_DCP and _HAS_DCP
    if Envs.RELAX_FSDPTURBO_CKPT_DCP and not _HAS_DCP and rank == 0:
        logger.warning(
            "[fsdp_turbo] save_checkpoint: RELAX_FSDPTURBO_CKPT_DCP is on but "
            "torch.distributed.checkpoint is unavailable in this build; falling back to "
            "the legacy HF gather path"
        )
    if use_dcp:
        _save_dcp(output_dir, model, optimizer, rollout_id)
        if Envs.RELAX_FSDPTURBO_CKPT_EXPORT_HF:
            _export_hf_weights(output_dir, model, optimizer, rollout_id)
        # Publish the resume marker only after every rank finishes writing.
        dist.barrier(group=get_gloo_group())
        if rank == 0:
            # Config/tokenizer keep the step recognizable to HF tooling; resume reads DCP shards.
            hf_config.save_pretrained(output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            logger.info(
                f"[fsdp_turbo] saved DCP sharded checkpoint for rollout_id={rollout_id} to {output_dir} "
                f"({dist.get_world_size()} per-rank shard files; model+optimizer in sharded form, "
                "VP tables included with their optimizer moments)"
            )
            # Step marker supports direct --load; root marker supports crash recovery.
            for root in (output_dir, args.save):
                _write_atomic_latest(root, rollout_id)
        dist.barrier(group=get_gloo_group())
        return

    # ---- legacy HF-gather path (RELAX_FSDPTURBO_CKPT_DCP=0) ----
    # Detach ~95 GiB VP tables from the single-card gather and shard them directly.
    vp_params = _find_vp_table_params(model)
    detach_receipt = _detach_vp_tables_for_gather(model, optimizer, vp_params)
    try:
        model_state, optimizer_state = _get_full_state_dicts(model, optimizer)
    finally:
        _restore_vp_tables_after_gather(model, optimizer, detach_receipt)
    if vp_params:
        _save_vp_shards(output_dir, vp_params, rollout_id)

    if rank == 0:
        logger.info(
            f"[fsdp_turbo] save_checkpoint: VP table param(s) found: {len(vp_params)} "
            "(non-zero means the shard bypass is active; zero means the tables ride the full gather)"
        )
        os.makedirs(output_dir, exist_ok=True)
        _write_hf_model_file(output_dir, model_state)
        hf_config.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(
            {"optimizer": optimizer_state, "rollout_id": rollout_id},
            os.path.join(output_dir, "optimizer.pt"),
        )
        logger.info(f"[fsdp_turbo] saved checkpoint for rollout_id={rollout_id} to {output_dir}")
        # Step marker supports direct --load; root marker supports crash recovery.
        for root in (output_dir, args.save):
            _write_atomic_latest(root, rollout_id)
    dist.barrier(group=get_gloo_group())


def _shard_view(file_t: torch.Tensor, param: torch.Tensor) -> torch.Tensor | None:
    """Return this rank's slice of a full tensor for a DTensor parameter.

    Match FSDP2's uneven ``torch.chunk`` layout, including zero padding for
    tail ranks. Return None when the exact layout cannot be derived so the
    caller can fall back.
    """
    spec = getattr(param, "_spec", None)
    if spec is None:
        return None
    try:
        mesh = spec.mesh
        coords = mesh.get_coordinate()
        if coords is None:
            return None
        local_shape = tuple(getattr(param, "_local_tensor", param).shape)
        view = file_t
        plan: list[tuple[int, int, int]] = []  # (dim, start, data rows)
        padded = False
        for mesh_idx, placement in enumerate(spec.placements):
            dim = getattr(placement, "dim", None)
            if dim is None:
                continue
            size = mesh.size(mesh_idx)
            if size <= 1:
                continue
            base, rem = divmod(view.shape[dim], size)
            rank = coords[mesh_idx]
            if rank < rem:
                start, data_rows = rank * (base + 1), base + 1
            else:
                start, data_rows = rem * (base + 1) + (rank - rem) * base, base
            if data_rows > local_shape[dim]:
                return None
            padded = padded or data_rows < local_shape[dim]
            plan.append((dim, start, data_rows))
            view = view.narrow(dim, start, data_rows)
        if not padded:
            return view
        # Match the padded local shard; only unshard trims the zero rows.
        out = file_t.new_zeros(local_shape)
        dst = out
        for dim, _, data_rows in plan:
            if data_rows < local_shape[dim]:
                dst = dst.narrow(dim, 0, data_rows)
        if dst.shape != view.shape:
            return None
        dst.copy_(view)
        return out
    except Exception:
        return None


def _is_fork_set_failure(msg: str) -> bool:
    """Return whether ``msg`` is a recoverable fork set_state_dict failure."""
    return "Sharding propagation failed" in msg or "mixed torch.Tensor and DTensor" in msg


def _describe_placement(param: torch.Tensor) -> str:
    """Describe mesh placement for landing-refusal diagnostics."""
    spec = getattr(param, "_spec", None)
    if spec is None:
        return "no _spec (not a standard DTensor)"
    try:
        mesh = spec.mesh
        return (
            f"placements={spec.placements} mesh={tuple(mesh.size(i) for i in range(mesh.ndim))} "
            f"coords={mesh.get_coordinate()}"
        )
    except Exception:
        return f"unusable spec ({spec!r})"


def _load_vp_shards(load_dir: str, vp_params: dict[str, torch.nn.Parameter]) -> None:
    """Restore VP tables from per-rank shards.

    Without ``meta.json``, the legacy gather path already restored them or
    their current materialized values must be kept.
    """
    shard_dir = os.path.join(load_dir, "ngram_shards")
    meta_path = os.path.join(shard_dir, "meta.json")
    if not os.path.isfile(meta_path):
        logger.info(
            f"[fsdp_turbo] load_checkpoint: no ngram_shards/meta.json under {load_dir}; "
            "VP table params keep whatever the gather path landed"
        )
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    saved_params: dict[str, Any] = meta["params"]
    current_names = {_strip_param_prefix(name) for name in vp_params}
    if set(saved_params) != current_names:
        raise RuntimeError(
            "[fsdp_turbo] load_checkpoint: VP table param set mismatch — checkpoint has "
            f"{sorted(saved_params)}, current model has {sorted(current_names)}"
        )
    first_param = next(iter(vp_params.values()))
    current_vp_size = int(first_param._spec.mesh.size(0))
    if int(meta["vp_size"]) != current_vp_size:
        raise RuntimeError(
            f"[fsdp_turbo] load_checkpoint: refusing {load_dir}: saved with vp_size={meta['vp_size']} "
            f"but the current vp_size={current_vp_size} — cross-configuration resharding is "
            "unsupported; resume with the same vp_parallel_size the checkpoint was saved with"
        )
    for name, param in vp_params.items():
        entry = saved_params[_strip_param_prefix(name)]
        if list(param.shape) != list(entry["global_shape"]):
            raise RuntimeError(
                f"[fsdp_turbo] load_checkpoint: VP table param '{name}' shape mismatch — checkpoint has "
                f"{entry['global_shape']}, current model has {list(param.shape)}"
            )
    from safetensors.torch import load_file

    rank = dist.get_rank()
    shard = load_file(os.path.join(shard_dir, f"rank-{rank:05d}.safetensors"))
    with torch.no_grad():
        for name, param in vp_params.items():
            local = getattr(param, "_local_tensor", param)
            value = shard[_strip_param_prefix(name)]
            if tuple(value.shape) != tuple(local.shape):
                raise RuntimeError(
                    f"[fsdp_turbo] load_checkpoint: VP table param '{name}' local shard is "
                    f"{tuple(local.shape)} but the saved shard for rank {rank} is {tuple(value.shape)}"
                )
            local.copy_(value.to(local.dtype))
    logger.warning(
        f"[fsdp_turbo] load_checkpoint: restored {len(vp_params)} VP table param(s) from ngram_shards "
        f"(rank {rank}); their optimizer moments are NOT persisted and re-zero on resume"
    )


def load_checkpoint(
    args,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    hf_config,
    tokenizer,
) -> int:
    """Resume a DCP or legacy HF checkpoint from ``--load``.

    DCP is detected by ``.metadata``; other directories use the legacy reader.
    Return the next rollout id, or 0 when no optimizer state exists.
    """
    if args.load is None or not os.path.isdir(args.load):
        return 0

    # Accept either a step dir or a save root with a resume marker. Check step
    # signatures first so a DCP step's own marker does not append its id again.
    load_dir = args.load
    if (
        not os.path.isfile(os.path.join(load_dir, _DCP_METADATA))
        and not os.path.isfile(os.path.join(load_dir, "model.safetensors"))
        and not os.path.isfile(os.path.join(load_dir, "model.bin"))
    ):
        marker_path = os.path.join(load_dir, _LATEST)
        if os.path.isfile(marker_path):
            with open(marker_path, encoding="utf-8") as f:
                load_dir = os.path.join(load_dir, f.read().strip())
    if not os.path.isdir(load_dir):
        logger.warning(f"[fsdp_turbo] load_checkpoint: no checkpoint found under {args.load}, training from scratch")
        return 0

    # DCP directories use the sharded reader instead of the legacy full-file path.
    if os.path.isfile(os.path.join(load_dir, _DCP_METADATA)):
        if not _HAS_DCP:
            raise RuntimeError(
                f"[fsdp_turbo] load_checkpoint: {load_dir} is a DCP sharded checkpoint but "
                "torch.distributed.checkpoint is unavailable in this build; cannot resume it"
            )
        return _load_dcp(load_dir, model, optimizer)

    # Broadcast only the path and step; every rank reads the multi-GB files
    # directly through mmap/page cache.
    rank = dist.get_rank()
    meta: list[Any] = [load_dir, 0]
    if rank == 0:
        rollout_id = 0
        marker_path = os.path.join(load_dir, _LATEST)
        if os.path.isfile(marker_path):
            with open(marker_path, encoding="utf-8") as f:
                rollout_id = int(f.read().strip())
        else:
            try:
                rollout_id = int(os.path.basename(load_dir))
            except ValueError:
                rollout_id = 0
        meta = [load_dir, rollout_id + 1]
    dist.broadcast_object_list(meta, src=0, group=get_gloo_group())
    load_dir, resumed_rollout_id = meta

    model_path = os.path.join(load_dir, "model.safetensors")
    if os.path.isfile(model_path):
        from safetensors.torch import load_file

        model_state: dict[str, Any] = load_file(model_path)
    else:
        model_state = torch.load(os.path.join(load_dir, "model.bin"), map_location="cpu", weights_only=False)
    optimizer_path = os.path.join(load_dir, "optimizer.pt")
    optimizer_state: dict[str, Any] | None = None
    if os.path.exists(optimizer_path):
        payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        optimizer_state = payload.get("optimizer")
    else:
        resumed_rollout_id = 0  # no optimizer state: train a fresh step count
    if rank == 0:
        logger.info(f"[fsdp_turbo] resuming from {load_dir} (next rollout_id={resumed_rollout_id})")

    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    # Build the exact key skeleton required by the current model and optimizer;
    # checkpoint data fills it without missing-key failures.
    model_skeleton, optim_skeleton = get_state_dict(model, optimizer)

    # Map HF checkpoint names to FSDPTurbo keys; missing keys keep current
    # values and surplus checkpoint keys are dropped.
    missing_fill: list[str] = []
    aligned_state: dict[str, Any] = {}
    held_model: dict[str, Any] = {}
    for name, tensor in model_skeleton.items():
        stripped = name[len(_TURBO_PREFIX) :] if name.startswith(_TURBO_PREFIX) else name
        if stripped in model_state:
            aligned_state[name] = model_state[stripped]
            # Keep original file tensors for the force-land pass.
            held_model[name] = model_state[stripped]
        else:
            aligned_state[name] = tensor
            missing_fill.append(name)
    model_state = aligned_state
    if missing_fill:
        logger.warning(
            f"[fsdp_turbo] load_checkpoint: {len(missing_fill)} model key(s) missing from the checkpoint; "
            f"keeping resume-time values for them (e.g. {missing_fill[:3]})"
        )

    # Reject rank-local shards instead of loading a different wrong slice per rank.
    rank_local = [
        name
        for name, value in held_model.items()
        if hasattr(value, "_local_tensor") and tuple(value._local_tensor.shape) != tuple(value.shape)
    ]
    if rank_local:
        raise RuntimeError(
            f"[fsdp_turbo] load_checkpoint: refusing {load_dir}: {len(rank_local)}/{len(held_model)} values "
            f"carry rank-local DTensor shards (e.g. '{rank_local[0]}') — only 1/N of each tensor was saved; "
            f"re-save the checkpoint from a healthy run before resuming"
        )

    # Missing optimizer entries keep fresh zero moments; surplus saved keys are dropped.
    if optimizer_state is not None:
        skeleton_state = optim_skeleton.get("state", {})
        saved_state = optimizer_state.get("state", {})
        restored = sum(1 for k in skeleton_state if k in saved_state)
        for fqn, entry in skeleton_state.items():
            if fqn in saved_state:
                skeleton_state[fqn] = saved_state[fqn]
        optimizer_state["state"] = skeleton_state
        logger.info(
            f"[fsdp_turbo] load_checkpoint: restored {restored}/{len(skeleton_state)} optimizer "
            f"state entries from checkpoint ({len(saved_state)} saved, {len(saved_state) - restored} unused)"
        )

    # This fork can reject EP sharding or plain-to-DTensor values after the
    # optimizer restore. Pop named model offenders, retry non-strictly, then
    # land them manually below.
    manual: dict[str, Any] = {}
    bad_keys: list[str] = []
    params = dict(model.named_parameters())
    try:
        set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optimizer_state)
    except RuntimeError as e:
        msg = str(e)
        if not _is_fork_set_failure(msg):
            raise
        import re

        bad_keys = sorted(set(re.findall(r'parameter named "([^"]+)"', msg)))
        manual = {k: model_state.pop(k) for k in bad_keys if k in model_state}
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        # Retries may surface more offenders. After two failures, force-land
        # model weights from held file tensors; optimizer state already restored.
        set_ok = False
        for _attempt in range(2):
            try:
                set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=model_state,
                    optim_state_dict=optimizer_state,
                    options=StateDictOptions(strict=False),
                )
                set_ok = True
                break
            except RuntimeError as e2:
                if not _is_fork_set_failure(str(e2)):
                    raise
                more = [k for k in re.findall(r'parameter named "([^"]+)"', str(e2)) if k in model_state]
                bad_keys.extend(k for k in more if k not in bad_keys)
                manual.update({k: model_state.pop(k) for k in more})
        if not set_ok:
            logger.warning(
                "[fsdp_turbo] load_checkpoint: set_state_dict's model path keeps failing on fork "
                "placements; the force-land pass below is the landing mechanism for model weights"
            )
        with torch.no_grad():
            for k, v in manual.items():
                p = params.get(k)
                if p is None:
                    continue
                # Copy into raw local storage to avoid mixed Tensor/DTensor
                # errors. Use an exact spec-driven slice when shapes differ;
                # reject unsupported trainable layouts, while frozen parameters
                # keep their identical current values.
                local = getattr(p, "_local_tensor", p)
                v_plain = getattr(v, "_local_tensor", v)
                if local.shape == v_plain.shape:
                    local.copy_(v_plain.to(local.dtype))
                    continue
                view = _shard_view(v_plain, p)
                if view is not None and view.shape == local.shape:
                    local.copy_(view.to(local.dtype))
                    continue
                if not p.requires_grad:
                    continue
                raise RuntimeError(
                    f"[fsdp_turbo] load_checkpoint: param {k} local shard {tuple(local.shape)} cannot "
                    f"accept checkpoint value {tuple(v_plain.shape)} — unsupported shard layout "
                    f"({_describe_placement(p)}; shard_view={'None' if view is None else tuple(view.shape)}); "
                    "report this layout so the slicer can cover it"
                )
        if manual:
            logger.warning(
                f"[fsdp_turbo] load_checkpoint: copied {len(manual)} set-refused param(s) into local "
                f"shards manually (e.g. {bad_keys[:3]})"
            )
    # This fork's model restore may not stick. Copy every held file tensor into
    # local storage; count unsupported layouts and leave those to set_state_dict.
    landed = 0
    unsupported = 0
    with torch.no_grad():
        for name, file_t in held_model.items():
            p = params.get(name)
            if p is None:
                continue
            # Unwrap file DTensors and require the global shape so rank-local
            # shards cannot be tiled onto every rank.
            file_plain = getattr(file_t, "_local_tensor", file_t)
            expected = tuple(model_skeleton[name].shape) if name in model_skeleton else tuple(file_plain.shape)
            if tuple(file_plain.shape) != expected:
                unsupported += 1
                continue
            local = getattr(p, "_local_tensor", p)
            if local.shape == file_plain.shape:
                local.copy_(file_plain.to(local.dtype))
                landed += 1
                continue
            view = _shard_view(file_plain, p)
            if view is None or view.shape != local.shape:
                unsupported += 1
                continue
            local.copy_(view.to(local.dtype))
            landed += 1
    logger.warning(
        f"[fsdp_turbo] load_checkpoint: force-landed {landed} param(s) from the checkpoint file "
        f"({unsupported} left to set_state_dict: unsupported shard layout)"
    )
    if unsupported > len(held_model) // 2:
        logger.error(
            "[fsdp_turbo] load_checkpoint: most checkpoint values carry rank-local shard shapes — "
            "this file was saved without full gathering and cannot resume; re-save with a fixed save path"
        )
    # VP tables bypassed the gather and must be restored from per-rank shards.
    vp_params = _find_vp_table_params(model)
    if vp_params:
        _load_vp_shards(load_dir, vp_params)
    # Rebuild restored Adam moments with zeros_like(param), the layout this
    # fork's optimizer accepts. Skip mismatched value copies; momentum
    # re-converges while the restored step counter preserves bias correction.
    if optimizer is not None:
        rebuilt = 0
        skipped = 0
        with torch.no_grad():
            for p, entry in optimizer.state.items():
                if not isinstance(entry, dict):
                    continue
                for m in ("exp_avg", "exp_avg_sq"):
                    if m not in entry:
                        continue
                    fresh = torch.zeros_like(p)
                    fresh_local = getattr(fresh, "_local_tensor", fresh)
                    src = getattr(entry[m], "_local_tensor", entry[m])
                    if src.shape == fresh_local.shape:
                        fresh_local.copy_(src.to(fresh_local.dtype))
                    else:
                        skipped += 1
                    entry[m] = fresh
                    rebuilt += 1
        if rebuilt:
            logger.warning(
                f"[fsdp_turbo] load_checkpoint: rebuilt {rebuilt} Adam moment tensor(s) in the "
                f"from-scratch zeros_like(param) form ({skipped} value copy skipped on shape mismatch)"
            )
    return int(resumed_rollout_id)
