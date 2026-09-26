# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import hashlib
import json
import os
import re
from argparse import Namespace
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint_megatron
from megatron.training.global_vars import get_args

from relax.utils import megatron_bridge_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.hf_page_cache import warm_hf_checkpoint_page_cache
from relax.utils.logging_utils import get_logger
from relax.utils.model_source import is_model_source_alias
from relax.utils.training.ppo_utils import (
    use_critic_lm_head_for_hf_load,
    use_sequence_classification_lm_head_for_hf_load,
)


try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = get_logger(__name__)

__all__ = ["save_checkpoint"]

_LORA_CHECKPOINT_METADATA_ATTR = "relax_lora_checkpoint_metadata"
_LORA_CHECKPOINT_FORMAT_VERSION = 1


def _filter_lora_checkpoint_state_dict(state_dict: dict, *, metadata: dict | None = None) -> dict:
    """Keep LoRA model shards while retaining resumable common/optimizer
    state."""
    from relax.utils.megatron_peft_utils import is_lora_adapter_param

    for key in tuple(state_dict):
        if re.fullmatch(r"model\d*", key):
            model_state = state_dict[key]
            if not isinstance(model_state, dict):
                raise TypeError(f"Expected a dict for checkpoint entry {key!r}, got {type(model_state).__name__}.")
            state_dict[key] = {name: value for name, value in model_state.items() if is_lora_adapter_param(str(name))}

    if metadata is not None:
        checkpoint_args = copy.copy(state_dict["args"])
        setattr(checkpoint_args, _LORA_CHECKPOINT_METADATA_ATTR, metadata)
        state_dict["args"] = checkpoint_args
    return state_dict


@contextmanager
def _patch_lora_checkpoint_state_dict(*, metadata: dict | None = None):
    """Temporarily filter MCore's generated model state to LoRA tensors.

    The original generator runs first so the distributed optimizer can derive
    its shard mapping from the complete model state. The optimizer itself only
    owns trainable parameters, so its state remains resumable while frozen
    base-model tensors are removed afterwards.
    """
    import megatron.training.checkpointing as checkpointing

    original_generate_state_dict = checkpointing.generate_state_dict

    def generate_lora_state_dict(*args, **kwargs):
        state_dict = original_generate_state_dict(*args, **kwargs)
        return _filter_lora_checkpoint_state_dict(state_dict, metadata=metadata)

    checkpointing.generate_state_dict = generate_lora_state_dict
    try:
        yield
    finally:
        checkpointing.generate_state_dict = original_generate_state_dict


def _global_trainable_lora_parameter_names(model) -> tuple[str, ...]:
    import torch.distributed as dist
    from megatron.core.utils import unwrap_model

    from relax.utils.megatron_peft_utils import is_lora_adapter_param

    chunks = unwrap_model(model)
    if not isinstance(chunks, list):
        chunks = [chunks]

    local_names: list[str] = []
    invalid_names: list[str] = []
    for chunk in chunks:
        for name, parameter in chunk.named_parameters():
            if not parameter.requires_grad:
                continue
            (local_names if is_lora_adapter_param(name) else invalid_names).append(name)

    gathered: list[tuple[list[str], list[str]]]
    if dist.is_initialized():
        group = get_gloo_group()
        gathered = [([], []) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather_object(gathered, (local_names, invalid_names), group=group)
    else:
        gathered = [(local_names, invalid_names)]

    all_lora = sorted({name for names, _ in gathered for name in names})
    all_invalid = sorted({name for _, names in gathered for name in names})
    if all_invalid:
        preview = ", ".join(all_invalid[:8])
        raise RuntimeError(
            "--save-lora-only requires every trainable model parameter to be a LoRA adapter; "
            f"found {len(all_invalid)} non-LoRA trainable parameter(s): {preview}"
        )
    if not all_lora:
        raise RuntimeError("--save-lora-only found no trainable LoRA adapter parameters.")
    return tuple(all_lora)


def _lora_checkpoint_metadata(args, trainable_names: tuple[str, ...]) -> dict:
    return {
        "format_version": _LORA_CHECKPOINT_FORMAT_VERSION,
        "base_hf_checkpoint": _hf_checkpoint_identity(args.hf_checkpoint),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": tuple(args.lora_target_modules or ()),
        "lora_scope": args.lora_scope,
        "lora_merge_mode": args.lora_merge_mode,
        "lora_adapter_mode": getattr(args, "lora_adapter_mode", False),
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "context_parallel_size": args.context_parallel_size,
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "expert_tensor_parallel_size": getattr(args, "expert_tensor_parallel_size", 1),
        "virtual_pipeline_model_parallel_size": getattr(args, "virtual_pipeline_model_parallel_size", None),
        "pipeline_model_parallel_layout": str(getattr(args, "pipeline_model_parallel_layout", None)),
        "decoder_first_pipeline_num_layers": getattr(args, "decoder_first_pipeline_num_layers", None),
        "decoder_last_pipeline_num_layers": getattr(args, "decoder_last_pipeline_num_layers", None),
        "num_layers_per_virtual_pipeline_stage": getattr(args, "num_layers_per_virtual_pipeline_stage", None),
        "world_size": args.world_size,
        "data_parallel_size": getattr(args, "data_parallel_size", None),
        "trainable_parameter_names": trainable_names,
    }


@lru_cache(maxsize=8)
def _hf_checkpoint_identity(path: str) -> dict[str, str | None]:
    root = Path(path)

    def digest(filename: str) -> str | None:
        candidate = root / filename
        if not candidate.is_file():
            return None
        return hashlib.sha256(candidate.read_bytes()).hexdigest()

    index_path = next(
        (
            candidate
            for candidate in (
                root / "model.safetensors.index.json",
                root / "pytorch_model.bin.index.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    identity = {
        "config_sha256": digest("config.json"),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path else None,
    }
    shard_names: list[str] = []
    if index_path is not None:
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    elif root.is_dir():
        shard_names = sorted(
            candidate.name for pattern in ("*.safetensors", "pytorch_model*.bin") for candidate in root.glob(pattern)
        )

    shard_stats = []
    for name in shard_names:
        shard = root / name
        if not shard.is_file():
            raise FileNotFoundError(f"HF base checkpoint shard is missing: {shard}")
        stat = shard.stat()
        shard_stats.append((name, stat.st_size, stat.st_mtime_ns))
    if shard_stats:
        encoded = json.dumps(shard_stats, separators=(",", ":"), ensure_ascii=True).encode()
        identity["shard_stat_sha256"] = hashlib.sha256(encoded).hexdigest()
        identity["shard_count"] = str(len(shard_stats))
    if not shard_stats:
        identity["path"] = str(root.resolve())
    return identity


@contextmanager
def _strict_lora_checkpoint_dcp_load(args):
    previous = args.dist_ckpt_strictness
    args.dist_ckpt_strictness = "raise_all"
    try:
        yield
    finally:
        args.dist_ckpt_strictness = previous


@contextmanager
def _preserve_hybrid_optimizer_steps_on_load():
    """Keep Adam steps when MCore rebuilds CPU-offload sub-optimizers.

    MCore reconstructs the common HybridDeviceOptimizer step from checkpoint
    param groups, but writes it into the pre-load state.  ``load_state_dict``
    then replaces that state with the distributed shards, which do not carry
    the scalar step.  Inject the reconstructed value into those incoming shards
    so the HybridDeviceOptimizer post-load hook can propagate it to the newly
    built CPU/GPU sub-optimizers.
    """
    try:
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        yield
        return

    original_hybrid_load = HybridDeviceOptimizer.load_state_dict
    original_distributed_load = DistributedOptimizer.load_state_dict
    loaded_steps: list[tuple[HybridDeviceOptimizer, int]] = []

    def load_state_dict_with_step(optimizer, state_dict):
        steps = {group["step"] for group in state_dict.get("param_groups", ()) if "step" in group}
        checkpoint_step = None
        if len(steps) == 1:
            checkpoint_step = next(iter(steps))
            step = torch.tensor(checkpoint_step, dtype=torch.float32, device="cpu")
            for state in state_dict.get("state", {}).values():
                if isinstance(state, dict):
                    # ``dummy_step()`` may already have populated this incoming
                    # state with step 1.  The checkpoint param group is the
                    # authoritative value and must replace that bootstrap step.
                    state["step"] = step.detach().clone()
        result = original_hybrid_load(optimizer, state_dict)
        if checkpoint_step is not None:
            step = torch.tensor(checkpoint_step, dtype=torch.float32, device="cpu")
            for state in optimizer.state.values():
                if isinstance(state, dict):
                    state["step"] = step.detach().clone()
            for sub_optimizer in optimizer.sub_optimizers:
                for state in sub_optimizer.state.values():
                    if isinstance(state, dict):
                        state["step"] = step.detach().clone()
                for group in sub_optimizer.param_groups:
                    group["step"] = int(checkpoint_step)
            for group in optimizer.param_groups:
                group["step"] = int(checkpoint_step)
        return result

    def load_distributed_state_dict_with_step(optimizer, state_dict):
        groups = state_dict.get("optimizer", {}).get("param_groups", ())
        steps = {group["step"] for group in groups if "step" in group}
        result = original_distributed_load(optimizer, state_dict)
        if isinstance(optimizer.optimizer, HybridDeviceOptimizer) and len(steps) == 1:
            loaded_steps.append((optimizer.optimizer, int(next(iter(steps)))))
        return result

    HybridDeviceOptimizer.load_state_dict = load_state_dict_with_step
    DistributedOptimizer.load_state_dict = load_distributed_state_dict_with_step
    try:
        yield
    finally:
        HybridDeviceOptimizer.load_state_dict = original_hybrid_load
        DistributedOptimizer.load_state_dict = original_distributed_load
        # Parameter-state loading happens after ``DistributedOptimizer.load_state_dict``
        # and may rebuild HDO's sub-optimizers once more.  Apply the checkpoint
        # step only after the complete DCP load has returned.
        for optimizer, checkpoint_step in loaded_steps:
            step = torch.tensor(checkpoint_step, dtype=torch.float32, device="cpu")
            for state in optimizer.state.values():
                if isinstance(state, dict):
                    state["step"] = step.detach().clone()
            for sub_optimizer in optimizer.sub_optimizers:
                for state in sub_optimizer.state.values():
                    if isinstance(state, dict):
                        state["step"] = step.detach().clone()
                for group in sub_optimizer.param_groups:
                    group["step"] = checkpoint_step
            for group in optimizer.param_groups:
                group["step"] = checkpoint_step


def _sync_hybrid_optimizer_checkpoint_steps(optimizer) -> None:
    """Copy the real Adam step back to HybridDeviceOptimizer param groups."""
    try:
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    except ImportError:
        return

    pending = [optimizer]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, HybridDeviceOptimizer):
            steps = {
                int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
                for state in current.state.values()
                if isinstance(state, dict) and "step" in state
            }
            if len(steps) > 1:
                raise RuntimeError(f"HybridDeviceOptimizer has inconsistent checkpoint steps: {sorted(steps)}")
            if steps:
                step = next(iter(steps))
                for group in current.param_groups:
                    group["step"] = step
                for sub_optimizer in current.sub_optimizers:
                    for group in sub_optimizer.param_groups:
                        group["step"] = step
        pending.extend(getattr(current, "chained_optimizers", ()))
        nested = getattr(current, "optimizer", None)
        if nested is not None:
            pending.append(nested)


@contextmanager
def _validate_lora_model_state_load(model):
    """Allow omitted frozen base keys, but never omit an adapter key
    silently."""
    from megatron.core.utils import unwrap_model

    from relax.utils.megatron_peft_utils import is_lora_adapter_param

    chunks = unwrap_model(model)
    chunks = chunks if isinstance(chunks, (list, tuple)) else [chunks]
    original_loaders = []
    for chunk in chunks:
        original = chunk.load_state_dict
        original_loaders.append((chunk, original))

        def checked_load_state_dict(state_dict, strict=True, _original=original):
            result = _original(state_dict, strict=False)
            missing_adapters = [key for key in result.missing_keys if is_lora_adapter_param(key)]
            if missing_adapters or result.unexpected_keys:
                raise RuntimeError(
                    "Lightweight LoRA model-state load mismatch: "
                    f"missing_adapters={missing_adapters[:8]}, unexpected={result.unexpected_keys[:8]}"
                )
            return result

        chunk.load_state_dict = checked_load_state_dict
    try:
        yield
    finally:
        for chunk, original in original_loaders:
            chunk.load_state_dict = original


def _resolve_checkpoint_iteration_dir(load_path: str | Path) -> Path | None:
    path = Path(load_path)
    if re.fullmatch(r"iter_\d{7}", path.name):
        return path
    tracker = path / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    value = tracker.read_text().strip()
    if not value.isdigit():
        return None
    return path / f"iter_{int(value):07d}"


def _read_lora_checkpoint_metadata(load_path: str | Path) -> dict | None:
    checkpoint_dir = _resolve_checkpoint_iteration_dir(load_path)
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    from megatron.core import dist_checkpointing

    common_state = dist_checkpointing.load_common_state_dict(str(checkpoint_dir))
    checkpoint_args = common_state.get("args")
    return getattr(checkpoint_args, _LORA_CHECKPOINT_METADATA_ATTR, None)


def _validate_lora_checkpoint_metadata(args, model, metadata: dict) -> None:
    if metadata.get("format_version") != _LORA_CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported lightweight LoRA checkpoint format version: {metadata.get('format_version')!r}."
        )
    expected = _lora_checkpoint_metadata(args, _global_trainable_lora_parameter_names(model))
    mismatches = {
        key: (metadata.get(key), expected.get(key)) for key in expected if metadata.get(key) != expected.get(key)
    }
    if mismatches:
        details = ", ".join(f"{key}: checkpoint={old!r}, current={new!r}" for key, (old, new) in mismatches.items())
        raise RuntimeError(f"Lightweight LoRA checkpoint is incompatible with the current run: {details}")


def save_checkpoint(*args, lora_only: bool = False, **kwargs):
    """Save either the regular full DCP or a resumable LoRA-only DCP."""
    if not lora_only:
        return _save_checkpoint_megatron(*args, **kwargs)

    runtime_args = get_args()
    model = args[1] if len(args) > 1 else kwargs["model"]
    optimizer = args[2] if len(args) > 2 else kwargs.get("optimizer")
    _sync_hybrid_optimizer_checkpoint_steps(optimizer)
    trainable_names = _global_trainable_lora_parameter_names(model)
    metadata = _lora_checkpoint_metadata(runtime_args, trainable_names)
    with _patch_lora_checkpoint_state_dict(metadata=metadata):
        return _save_checkpoint_megatron(*args, **kwargs)


def _alias_renamed_transfer_queue_enum() -> None:
    """Back-compat for checkpoints saved with transfer_queue < 0.1.10.dev0.

    The streaming dataloader persists a ``ZMQServerInfo`` (with a ``Role`` enum
    field) into Megatron's ``common.pt``. transfer_queue >= 0.1.10.dev0 (now the
    minimum required version, see ``relax/utils/arguments.py``) renamed that enum
    ``TransferQueueRole`` -> ``Role`` while keeping the member values identical.
    Older checkpoints pickle the old class path, so ``torch.load`` raises
    ``AttributeError: Can't get attribute 'TransferQueueRole'``. Re-expose the old
    name as an alias so those checkpoints unpickle to the correct ``Role`` member.
    The restored connection info is stale and gets re-initialized on resume, so
    this only affects deserialization.
    """
    try:
        from transfer_queue.utils import enum_utils
    except ImportError:
        return
    if not hasattr(enum_utils, "TransferQueueRole") and hasattr(enum_utils, "Role"):
        enum_utils.TransferQueueRole = enum_utils.Role


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    exist = Path(load_path).exists() and _is_dir_nonempty(load_path)

    if exist and _is_megatron_checkpoint(load_path):
        _alias_renamed_transfer_queue_enum()
        lora_metadata = _read_lora_checkpoint_metadata(load_path)
        try:
            if lora_metadata is None:
                return _load_checkpoint_megatron(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    opt_param_scheduler=opt_param_scheduler,
                    checkpointing_context=checkpointing_context,
                    skip_load_to_model_and_opt=skip_load_to_model_and_opt,
                )

            _validate_lora_checkpoint_metadata(args, ddp_model, lora_metadata)
            if not skip_load_to_model_and_opt:
                _load_checkpoint_hf(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    args=args,
                    load_path=args.hf_checkpoint,
                )
            with (
                _patch_lora_checkpoint_state_dict(),
                _strict_lora_checkpoint_dcp_load(args),
                _preserve_hybrid_optimizer_steps_on_load(),
                _validate_lora_model_state_load(ddp_model),
            ):
                return _load_checkpoint_megatron(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    opt_param_scheduler=opt_param_scheduler,
                    checkpointing_context=checkpointing_context,
                    skip_load_to_model_and_opt=skip_load_to_model_and_opt,
                    strict=True,
                )
        except AssertionError as e:
            if "OptimizerParamScheduler" in str(e):
                raise RuntimeError(_format_opt_param_scheduler_error(args, e)) from e
            raise
    else:
        if not exist:
            load_path = None
            logger.warning(f"{args.load=} does not exist or is an empty directory. use args.hf_checkpoint")
        elif not _is_hf_checkpoint(load_path):
            logger.warning(
                f"{args.load=} exists but is not a valid HF checkpoint (no config.json). "
                "Falling back to args.hf_checkpoint"
            )
            load_path = None
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _format_opt_param_scheduler_error(args, original: AssertionError) -> str:
    # lr_decay_steps = num_rollout * rollout_batch_size * n_samples_per_prompt
    # (see relax/backends/megatron/model.py:get_optimizer_param_scheduler).
    # When any of those args change vs. the saved checkpoint, Megatron's
    # OptimizerParamScheduler refuses to load. Tell the user exactly what to do.
    return (
        f"Resume failed: {original}\n\n"
        f"Megatron's OptimizerParamScheduler rejects mismatched LR/WD schedule values "
        f"between the current run and the loaded checkpoint. This is almost always caused "
        f"by changing one of the args that feed into `lr_decay_steps` / `wd_incr_steps`:\n"
        f"    lr_decay_steps = num_rollout * rollout_batch_size * n_samples_per_prompt\n"
        f"Current values:\n"
        f"    --num-rollout            {getattr(args, 'num_rollout', None)}\n"
        f"    --rollout-batch-size     {getattr(args, 'rollout_batch_size', None)}\n"
        f"    --n-samples-per-prompt   {getattr(args, 'n_samples_per_prompt', None)}\n"
        f"    --global-batch-size      {getattr(args, 'global_batch_size', None)}\n"
        f"    --lr-decay-iters         {getattr(args, 'lr_decay_iters', None)}\n"
        f"    --lr-warmup-iters        {getattr(args, 'lr_warmup_iters', None)}\n"
        f"    --lr-warmup-fraction     {getattr(args, 'lr_warmup_fraction', None)}\n\n"
        f"Pick one:\n"
        f"  (a) Revert the changed arg to match the checkpoint, OR\n"
        f"  (b) Add `--override-opt_param-scheduler` to keep the NEW schedule "
        f"(class values overwrite checkpoint values), OR\n"
        f"  (c) Add `--use-checkpoint-opt_param-scheduler` to keep the OLD schedule "
        f"(checkpoint values overwrite class values)."
    )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _is_hf_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "config.json").is_file()


@contextmanager
def _patch_scatter_dtype_cast():
    """Temporarily patch torch.distributed.scatter to auto-cast scatter_list
    tensors to match output dtype.

    Megatron Bridge's scatter_to_tp_ranks creates `output` with the Megatron
    model dtype (e.g. fp16) and `scatter_list` from HF weights (e.g. bf16). Not
    all mapping types (e.g. GatedMLPMapping) cast HF weights to the target
    dtype before scatter, causing a ValueError from PyTorch's dtype consistency
    check. This patch ensures scatter_list tensors are cast to match output's
    dtype.
    """
    import torch.distributed as dist

    original_scatter = dist.scatter

    def _scatter_with_dtype_cast(output, scatter_list=None, **kwargs):
        if scatter_list is not None and output is not None:
            target_dtype = output.dtype
            scatter_list = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in scatter_list]
        return original_scatter(output, scatter_list=scatter_list, **kwargs)

    dist.scatter = _scatter_with_dtype_cast
    try:
        yield
    finally:
        dist.scatter = original_scatter


def _select_hf_load_source(args: Namespace, load_path: str | None) -> str:
    """Choose an HF source after Megatron resume has been ruled out."""
    # Prefer ref_load (if it's an HF dir) over hf_checkpoint on fallback. INT4 QAT
    # runs set --hf-checkpoint to a compressed-tensors packed dir that the bridge
    # cannot read; --ref-load points at the BF16 HF dir that it can. Mirrors the
    # `args.load = args.ref_load or args.hf_checkpoint` remap in arguments.py.
    if load_path is not None:
        return args.hf_checkpoint if is_model_source_alias(args, load_path) else load_path
    if args.ref_load and _is_hf_checkpoint(args.ref_load):
        return args.ref_load
    return args.hf_checkpoint


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str | None):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    source_path = _select_hf_load_source(args, load_path)
    logger.info(
        f"Load checkpoint from HuggingFace model into Megatron (requested_path={load_path}, source_path={source_path})"
    )

    if getattr(args, "warm_hf_checkpoint_page_cache", False):
        warm_hf_checkpoint_page_cache(source_path)

    with use_critic_lm_head_for_hf_load(ddp_model):
        with use_sequence_classification_lm_head_for_hf_load(ddp_model):
            with megatron_bridge_utils.patch_megatron_model(ddp_model):
                bridge = AutoBridge.from_hf_pretrained(source_path, trust_remote_code=True)
                with _patch_scatter_dtype_cast():
                    bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)


def _save_lora_to_checkpoint(model, checkpoint_dir: str, args, bridge=None) -> None:
    """Save the LoRA adapter as a standard HF-PEFT directory under
    ``checkpoint_dir``.

    This is a portable *export* artifact (HF ``adapter_model.safetensors`` +
    ``adapter_config.json``) for external / inference use — e.g. loading with
    ``peft.PeftModel.from_pretrained``. It is NOT the resume source: LoRA params are
    ordinary model parameters, so the native Megatron torch_dist checkpoint already
    persists and restores them on resume.

    Every rank exports its owned (TP-gathered) adapter params via Megatron-Bridge, the
    shards are gathered to global rank 0, and rank 0 writes one consolidated adapter.

    Collective: the export + ``gather_object`` MUST run in lockstep on every rank, so
    nothing before the gather is gated behind a rank check.

    Args:
        model: The training model (sequence of VPP chunks) with LoRA.
        checkpoint_dir: Directory where ``lora_adapter/`` will be written.
        args: Training arguments.
        bridge: Optional pre-built ``AutoBridge`` (reused by ``save_hf_model``); built
            from ``args.hf_checkpoint`` when not supplied.
    """

    import torch.distributed as dist

    from relax.utils.megatron_peft_utils import convert_megatron_to_hf_target_modules, write_hf_peft_adapter

    gloo = get_gloo_group()
    is_dst = dist.get_rank() == 0
    local_error = None
    local_adapter = {}
    try:
        if bridge is None:
            from megatron.bridge import AutoBridge

            bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # Export this rank's owned adapter params (TP already gathered by the bridge);
        # cpu=True so the tensors can travel over the gloo gather below.
        with megatron_bridge_utils.patch_megatron_model(model):
            local_adapter = {
                item.param_name: item.weight.detach().cpu() for item in bridge.export_adapter_weights(model, cpu=True)
            }
    except Exception as exc:
        local_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"

    errors = [None] * dist.get_world_size(group=gloo)
    dist.all_gather_object(errors, local_error, group=gloo)
    errors = [error for error in errors if error]
    if errors:
        raise RuntimeError("LoRA adapter export failed before gather: " + "; ".join(errors))

    gathered = [None] * dist.get_world_size(group=gloo) if is_dst else None
    dist.gather_object(local_adapter, object_gather_list=gathered, dst=0, group=gloo)

    write_error = None
    if is_dst:
        try:
            merged: dict[str, torch.Tensor] = {}
            for shard in gathered:
                if shard:
                    merged.update(shard)
            if not merged:
                raise RuntimeError("LoRA enabled but no adapter parameters were gathered.")

            adapter_dir = Path(checkpoint_dir) / "lora_adapter"
            # args.lora_target_modules holds canonical Megatron names; the on-disk HF-PEFT
            # adapter_config.json must carry HF-style names so standard PEFT loaders can match
            # them against the HF module tree.
            write_hf_peft_adapter(
                merged,
                adapter_dir,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=convert_megatron_to_hf_target_modules(args.lora_target_modules),
                lora_dropout=args.lora_dropout,
            )

            # Sidecar metadata for resume-time mode-mismatch diagnostics. Kept separate from
            # adapter_config.json so it can never confuse load_peft_adapter.
            import json

            metadata = {
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_target_modules": list(args.lora_target_modules),
                "lora_dropout": args.lora_dropout,
                "lora_merge_mode": args.lora_merge_mode,
                "lora_adapter_mode": getattr(args, "lora_adapter_mode", False),
            }
            with open(adapter_dir / "relax_lora_meta.json", "w") as f:
                json.dump(metadata, f)

            if metadata["lora_adapter_mode"]:
                mode_str = "adapter"
            elif metadata["lora_merge_mode"]:
                mode_str = "merge"
            else:
                mode_str = "standard"
            logger.info(f"Saved LoRA adapter to {adapter_dir} ({len(merged)} tensors, mode={mode_str})")
        except Exception as exc:
            write_error = f"{type(exc).__name__}: {exc}"

    write_status = [write_error]
    dist.broadcast_object_list(write_status, src=0, group=gloo)
    if write_status[0] is not None:
        raise RuntimeError(f"Failed to write LoRA adapter: {write_status[0]}")
