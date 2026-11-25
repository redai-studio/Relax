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
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from relax.backends.fsdp_turbo.checkpoint_publication import checkpoint_publication
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


@contextmanager
def _checkpoint_write_phase():
    """Propagate file-write errors before any rank enters the next phase."""
    error = None
    try:
        yield
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(get_gloo_group())
    dist.all_gather_object(errors, error, group=get_gloo_group())
    if any(item is not None for item in errors):
        raise RuntimeError(f"Checkpoint write failed across ranks: {errors}")


def validate_checkpoint_publisher(args) -> None:
    with _checkpoint_write_phase():
        with checkpoint_publication(args.save, args._turbo_checkpoint_generation):
            pass


def _get_full_state_dicts(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    # Gather full tensors into rank-0 host RAM; saving local shards here would
    # produce an unresumable file. Other ranks do not hold a full copy.
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    if optimizer is None or not optimizer.state:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        model_state = get_model_state_dict(model, options=options)
        optimizer_state = optimizer.state_dict() if optimizer is not None else None
    else:
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
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
            self.initialize_optimizer = False
            self.saved_optimizer_keys: set[str] | None = None

        def state_dict(self) -> dict[str, Any]:
            from torch.distributed.checkpoint.state_dict import get_state_dict

            # Sharded defaults avoid materializing the ~95 GiB VP table in full.
            if self.optimizer is None or not self.optimizer.state and not self.initialize_optimizer:
                from torch.distributed.checkpoint.state_dict import get_model_state_dict

                model_sd = get_model_state_dict(self.model)
                optim_sd = self.optimizer.state_dict() if self.optimizer is not None else None
            else:
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
            state = {"model": model_sd, "rollout_id": self.rollout_id}
            if self.optimizer is not None:
                state["optim"] = optim_sd
            return state

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                set_model_state_dict,
                set_optimizer_state_dict,
            )

            model_sd = state_dict.get("model", {})
            optim_sd = state_dict.get("optim", {})
            if self.saved_optimizer_keys is not None:
                optim_sd["state"] = {
                    name: entry
                    for name, entry in optim_sd.get("state", {}).items()
                    if name in self.saved_optimizer_keys
                }
            if not state_dict.get("optim", {}).get("state"):
                set_model_state_dict(self.model, model_sd, options=StateDictOptions(strict=False))
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(state_dict["optim"])
                self.rollout_id = int(state_dict.get("rollout_id", 0))
                return
            # Re-inject skipped frozen tables so strict restore sees a complete mapping.
            missing_ngram = _find_frozen_ngram_param_names(self.model) - model_sd.keys()
            if missing_ngram:
                params = dict(self.model.named_parameters())
                for name in missing_ngram:
                    model_sd[name] = params[name]
            try:
                set_model_state_dict(
                    self.model,
                    model_sd,
                    options=StateDictOptions(strict=True),
                )
            except RuntimeError as e:
                if not _is_fork_set_failure(str(e)):
                    raise
                # This known fork refusal happens after DCP has written values
                # through shared storage, so the restored weights remain intact.
                logger.warning(
                    "[fsdp_turbo] load_checkpoint: model state restore refused with a known fork "
                    f"signature after the DCP load ({e}); values were already landed in place "
                    "by the DCP reader itself"
                )
            if self.optimizer is not None:
                set_optimizer_state_dict(self.model, self.optimizer, optim_sd, options=StateDictOptions(strict=False))
            self.rollout_id = int(state_dict.get("rollout_id", 0))

    class _OptimStateTolerantLoadPlanner(DefaultLoadPlanner):
        """Drop absent lazy Adam state and frozen replicated n-gram reads.

        Parameters that never received gradients remain uninitialized. The ~102
        GiB frozen tables are identical to their HF-built values and would
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
                        f"{len(dropped)} key(s) excluded from the DCP load (uninitialized optimizer state / "
                        f"frozen n-gram tables identical to the HF materialization, e.g. '{dropped[0]}')"
                    )
            return super().create_local_plan()

    def _save_dcp(
        output_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, rollout_id: int
    ) -> None:
        """Save local shards; published ``.metadata`` certifies all ranks
        finished."""
        with _checkpoint_write_phase():
            os.makedirs(output_dir, exist_ok=True)
        dcp.save(
            {"app": _AppState(model, optimizer, rollout_id)},
            checkpoint_id=output_dir,
            process_group=get_gloo_group(),
        )

    def _load_dcp(load_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> int:
        """Resume DCP shards in place and return ``saved_rollout_id + 1``.

        Host memory stays at one local shard per rank.
        """
        app = _AppState(model, optimizer, 0)
        metadata = dcp.FileSystemReader(load_dir).read_metadata()
        if optimizer is not None and not any(path[:2] == ("app", "optim") for path in metadata.planner_data.values()):
            raise ValueError("Checkpoint has no optimizer state; use --no-load-optim to restore weights only")
        app.saved_optimizer_keys = {
            path[3] for path in metadata.planner_data.values() if path[:3] == ("app", "optim", "state")
        }
        app.initialize_optimizer = bool(app.saved_optimizer_keys)
        dcp.load(
            {"app": app},
            checkpoint_id=load_dir,
            planner=_OptimStateTolerantLoadPlanner(),
            process_group=get_gloo_group(),
        )
        logger.info(
            f"[fsdp_turbo] load_checkpoint: resumed DCP sharded checkpoint from {load_dir} "
            f"(next rollout_id={app.rollout_id + 1})"
        )
        return app.rollout_id + 1


def _save_vp_shards(output_dir: str, vp_params: dict[str, torch.nn.Parameter], rollout_id: int) -> None:
    """Publish VP metadata only after all ranks have written their shards."""
    from safetensors.torch import save_file

    rank = dist.get_rank()
    shard_dir = os.path.join(output_dir, "ngram_shards")
    with _checkpoint_write_phase():
        os.makedirs(shard_dir, exist_ok=True)
        shard = {
            _strip_param_prefix(name): param.to_local().detach().cpu().contiguous()
            for name, param in vp_params.items()
        }
        save_file(shard, os.path.join(shard_dir, f"rank-{rank:05d}.safetensors"), metadata={"format": "pt"})
    with _checkpoint_write_phase():
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
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        model_state = _strip_turbo_prefix(
            get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        )
    finally:
        _restore_vp_tables_after_gather(model, optimizer, detach_receipt)
    if vp_params:
        _save_vp_shards(output_dir, vp_params, rollout_id)
    with _checkpoint_write_phase():
        if dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            _write_hf_model_file(output_dir, model_state)
            logger.info(
                "[fsdp_turbo] save_checkpoint: HF weight export written beside the DCP checkpoint "
                "(RELAX_FSDPTURBO_CKPT_EXPORT_HF); resume does NOT read it"
            )


def _prune_checkpoints(args, rollout_id: int) -> None:
    max_keep = args.max_actor_ckpt_to_keep
    rotate = args.rotate_ckpt
    if max_keep is None and not rotate:
        return
    checkpoints = sorted(
        (
            path
            for path in Path(args.save).iterdir()
            if path.name.isdecimal()
            and int(path.name) <= rollout_id
            and not path.is_symlink()
            and path.is_dir()
            and (path / _LATEST).is_file()
            and (path / _LATEST).read_text(encoding="utf-8").strip() == path.name
        ),
        key=lambda path: int(path.name),
        reverse=True,
    )
    if rotate:
        periodic = [path for path in checkpoints if (int(path.name) + 1) % args.save_interval == 0]
        keep = set(periodic if max_keep is None else periodic[:max_keep])
        keep.add(Path(args.save) / str(rollout_id))
    else:
        keep = set(checkpoints[: max(1, max_keep)])
    for path in checkpoints:
        if path not in keep:
            shutil.rmtree(path)


def _publish_checkpoint(output_dir: str, args, rollout_id: int) -> None:
    """Make a completed step visible before updating the recovery pointer."""
    with checkpoint_publication(args.save, args._turbo_checkpoint_generation):
        _write_atomic_latest(output_dir, rollout_id)
        os.rename(output_dir, os.path.join(args.save, str(rollout_id)))
        _write_atomic_latest(args.save, rollout_id)
        try:
            _prune_checkpoints(args, rollout_id)
        except OSError as error:
            logger.warning(f"Checkpoint {rollout_id} published, but old checkpoint cleanup failed: {error}")


def save_checkpoint(
    args,
    rollout_id: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    hf_config,
    tokenizer,
    lr_scheduler=None,
) -> None:
    """Save model and optimizer state under ``args.save/<step>``.

    DCP shards are the default; ``RELAX_FSDPTURBO_CKPT_DCP=0`` selects the
    legacy rank-0 HF gather.
    """
    if args.save is None:
        return
    hf_config = getattr(getattr(model, "model", model), "config", hf_config)
    final_dir = os.path.join(args.save, str(rollout_id))
    rank = dist.get_rank()
    staging_dir = [""]

    with _checkpoint_write_phase():
        if rank == 0:
            if not hasattr(args, "_turbo_checkpoint_generation"):
                raise ValueError("Checkpoint publisher has no startup generation")
            if os.path.lexists(final_dir):
                raise FileExistsError(f"Refusing to overwrite checkpoint: {final_dir}")
            os.makedirs(args.save, exist_ok=True)
            staging_dir[0] = os.path.join(args.save, f".{rollout_id}.{uuid.uuid4().hex}")
            os.makedirs(staging_dir[0], exist_ok=False)
    dist.broadcast_object_list(staging_dir, src=0, group=get_gloo_group())
    output_dir = staging_dir[0]

    with _checkpoint_write_phase():
        if rank == 0:
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "lr_scheduler.pt"))

    use_dcp = Envs.RELAX_FSDPTURBO_CKPT_DCP and _HAS_DCP
    saved_optimizer = None if args.no_save_optim else optimizer
    if Envs.RELAX_FSDPTURBO_CKPT_DCP and not _HAS_DCP and rank == 0:
        logger.warning(
            "[fsdp_turbo] save_checkpoint: RELAX_FSDPTURBO_CKPT_DCP is on but "
            "torch.distributed.checkpoint is unavailable in this build; falling back to "
            "the legacy HF gather path"
        )
    if use_dcp:
        _save_dcp(output_dir, model, saved_optimizer, rollout_id)
        if Envs.RELAX_FSDPTURBO_CKPT_EXPORT_HF:
            _export_hf_weights(output_dir, model, optimizer, rollout_id)
        # Publish the resume marker only after every rank finishes writing.
        dist.barrier(group=get_gloo_group())
        with _checkpoint_write_phase():
            if rank == 0:
                # Config/tokenizer keep the step recognizable to HF tooling; resume reads DCP shards.
                hf_config.save_pretrained(output_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(output_dir)
                _publish_checkpoint(output_dir, args, rollout_id)
                logger.info(
                    f"[fsdp_turbo] saved DCP sharded checkpoint for rollout_id={rollout_id} to {final_dir} "
                    f"({dist.get_world_size()} per-rank shard files; optimizer saved={saved_optimizer is not None})"
                )
        dist.barrier(group=get_gloo_group())
        return

    # ---- legacy HF-gather path (RELAX_FSDPTURBO_CKPT_DCP=0) ----
    # Detach ~95 GiB VP tables from the single-card gather and shard them directly.
    vp_params = _find_vp_table_params(model)
    detach_receipt = _detach_vp_tables_for_gather(model, optimizer, vp_params)
    try:
        model_state, optimizer_state = _get_full_state_dicts(model, saved_optimizer)
    finally:
        _restore_vp_tables_after_gather(model, optimizer, detach_receipt)
    if vp_params:
        _save_vp_shards(output_dir, vp_params, rollout_id)

    with _checkpoint_write_phase():
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
            if saved_optimizer is not None:
                torch.save(
                    {"optimizer": optimizer_state, "rollout_id": rollout_id},
                    os.path.join(output_dir, "optimizer.pt"),
                )
            _publish_checkpoint(output_dir, args, rollout_id)
            logger.info(f"[fsdp_turbo] saved checkpoint for rollout_id={rollout_id} to {final_dir}")
    dist.barrier(group=get_gloo_group())


def _is_fork_set_failure(msg: str) -> bool:
    """Return whether ``msg`` is a recoverable fork set_state_dict failure."""
    return "Sharding propagation failed" in msg or "mixed torch.Tensor and DTensor" in msg


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


def _localize_full_tensor(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convert a full CPU checkpoint tensor to the target's local layout."""
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        local = value.to_local()
        if local.shape != value.shape:
            raise ValueError("Legacy checkpoint contains rank-local tensors instead of full weights")
        value = local
    if value.shape != target.shape:
        raise ValueError(f"Checkpoint shape mismatch: {value.shape} != {target.shape}")
    if not isinstance(target, DTensor):
        return value.to(device=target.device)

    from relax.backends.fsdp_turbo.model_utils import _compose_dtensor_local_indices

    for dim, indices in enumerate(_compose_dtensor_local_indices(target.shape, target.device_mesh, target.placements)):
        if isinstance(indices, range):
            slices = [slice(None)] * value.ndim
            slices[dim] = slice(indices.start, indices.stop, indices.step)
            value = value[tuple(slices)]
        else:
            value = value.index_select(dim, torch.tensor(indices, dtype=torch.long, device=value.device))
    result = torch.empty_like(target, dtype=value.dtype)
    with torch.no_grad():
        result.to_local().copy_(value)
    return result


def find_latest_complete_checkpoint(save_root: str) -> str | None:
    """Find a published step even if the root recovery pointer is stale."""
    if not os.path.isdir(save_root):
        return None
    with os.scandir(save_root) as entries:
        candidates = sorted(
            (
                (int(entry.name), entry.path)
                for entry in entries
                if entry.name.isdecimal() and entry.is_dir(follow_symlinks=False)
            ),
            reverse=True,
        )
    for step, path in candidates:
        marker = os.path.join(path, _LATEST)
        if not os.path.isfile(marker):
            continue
        with open(marker, encoding="utf-8") as file:
            if file.read().strip() != str(step):
                raise ValueError(f"Checkpoint completion marker does not match its step: {path}")
        has_model = any(os.path.isfile(os.path.join(path, name)) for name in ("model.safetensors", "model.bin"))
        has_state = os.path.isfile(os.path.join(path, _DCP_METADATA)) or has_model
        if not has_state or not os.path.isfile(os.path.join(path, "config.json")):
            raise ValueError(f"Published checkpoint is incomplete: {path}")
        return path
    return None


def _resolve_checkpoint_directory(path: str) -> str:
    if not any(os.path.isfile(os.path.join(path, name)) for name in (_DCP_METADATA, "model.safetensors", "model.bin")):
        latest = find_latest_complete_checkpoint(path)
        if latest is not None:
            path = latest
        else:
            marker = os.path.join(path, _LATEST)
            if os.path.isfile(marker):
                with open(marker, encoding="utf-8") as file:
                    path = os.path.join(path, file.read().strip())
    return path


def remember_initial_load_policy(args) -> None:
    if hasattr(args, "_turbo_initial_load_policy"):
        return
    args._turbo_initial_load_policy = {}
    if args.load is not None:
        selected = _resolve_checkpoint_directory(args.load)
        args._turbo_initial_load_policy[os.path.realpath(selected)] = args.no_load_optim


def checkpoint_recovery_load_policy(args, path: str) -> tuple[str, bool]:
    # Match aliases in the driver; only the logical path travels to other nodes.
    selected = _resolve_checkpoint_directory(path)
    return selected, args._turbo_initial_load_policy.get(os.path.realpath(selected), args.no_save_optim)


def load_checkpoint(
    args,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    hf_config,
    tokenizer,
    lr_scheduler=None,
) -> int:
    """Resume a DCP or legacy HF checkpoint from ``--load``.

    DCP is detected by ``.metadata``; other directories use the legacy reader.
    Return the next rollout id; ``--no-load-optim`` keeps a fresh optimizer.
    """
    if args.load is None:
        return 0
    if not os.path.isdir(args.load):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.load}")

    # Accept either a step dir or a save root with a resume marker. Check step
    # signatures first so a DCP step's own marker does not append its id again.
    selection = [getattr(args, "_turbo_recovery_load_path", args.load), None]
    if dist.get_rank(group=get_gloo_group()) == 0:
        try:
            selection[0] = _resolve_checkpoint_directory(selection[0])
        except (OSError, ValueError) as error:
            selection[1] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(selection, src=0, group=get_gloo_group())
    if selection[1] is not None:
        raise ValueError(f"Checkpoint discovery failed: {selection[1]}")
    load_dir = selection[0]
    if args.no_load_optim:
        optimizer = None
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Checkpoint marker points to missing directory: {load_dir}")
    if not any(
        os.path.isfile(os.path.join(load_dir, name)) for name in (_DCP_METADATA, "model.safetensors", "model.bin")
    ):
        raise ValueError(f"No DCP metadata or legacy model weights found in checkpoint: {load_dir}")
    if lr_scheduler is not None and not os.path.isfile(os.path.join(load_dir, "lr_scheduler.pt")):
        raise ValueError(f"Checkpoint has no LR scheduler state: {load_dir}")

    # DCP directories use the sharded reader instead of the legacy full-file path.
    if os.path.isfile(os.path.join(load_dir, _DCP_METADATA)):
        if not _HAS_DCP:
            raise RuntimeError(
                f"[fsdp_turbo] load_checkpoint: {load_dir} is a DCP sharded checkpoint but "
                "torch.distributed.checkpoint is unavailable in this build; cannot resume it"
            )
        resumed_rollout_id = _load_dcp(load_dir, model, optimizer)
        _restore_lr_scheduler(load_dir, lr_scheduler)
        return resumed_rollout_id

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
                rollout_id = int(os.path.basename(os.path.normpath(load_dir)))
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
    if optimizer is not None and os.path.exists(optimizer_path):
        payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        optimizer_state = payload.get("optimizer")
        if optimizer_state is None:
            raise ValueError(f"Checkpoint has no optimizer state: {optimizer_path}")
    elif optimizer is not None:
        raise FileNotFoundError(f"Checkpoint has no optimizer state file: {optimizer_path}; use --no-load-optim")
    if rank == 0:
        logger.info(f"[fsdp_turbo] resuming from {load_dir} (next rollout_id={resumed_rollout_id})")

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    skeleton = get_model_state_dict(model)
    aligned = {}
    for name, tensor in skeleton.items():
        stripped = name[len(_TURBO_PREFIX) :] if name.startswith(_TURBO_PREFIX) else name
        if stripped in model_state:
            value = model_state[stripped]
            if tuple(value.shape) != tuple(tensor.shape):
                raise ValueError(f"Checkpoint shape mismatch for {name}: {value.shape} != {tensor.shape}")
            aligned[name] = _localize_full_tensor(value, tensor)
        elif "ngram_embedding" not in name:
            raise ValueError(f"Checkpoint is missing model parameter {name}")
    options = StateDictOptions(strict=False)
    set_model_state_dict(model, aligned, options=options)
    if optimizer is not None and optimizer_state is not None:
        if optimizer_state.get("state"):
            for name, entry in optimizer_state["state"].items():
                for key, value in entry.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        entry[key] = _localize_full_tensor(value, skeleton[name])
            set_optimizer_state_dict(model, optimizer, optimizer_state, options=options)
        else:
            optimizer.load_state_dict(optimizer_state)
    # VP tables bypassed the gather and must be restored from per-rank shards.
    vp_params = _find_vp_table_params(model)
    if vp_params:
        _load_vp_shards(load_dir, vp_params)
    _restore_lr_scheduler(load_dir, lr_scheduler)
    return int(resumed_rollout_id)


def _restore_lr_scheduler(load_dir: str, lr_scheduler) -> None:
    if lr_scheduler is None:
        return
    path = os.path.join(load_dir, "lr_scheduler.pt")
    if not os.path.isfile(path):
        raise ValueError(
            "Checkpoint has no LR scheduler state; exact resume is unavailable. "
            "Use the exported weights as a fresh training checkpoint instead."
        )
    lr_scheduler.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    if hasattr(lr_scheduler, "get_last_lr"):
        for group, lr in zip(lr_scheduler.optimizer.param_groups, lr_scheduler.get_last_lr(), strict=True):
            group["lr"] = lr
