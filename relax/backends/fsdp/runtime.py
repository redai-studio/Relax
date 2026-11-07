# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FSDP2 runtime helpers: sharding, offload/onload, trajectory sidecar I/O.

Kept separate from the actor so the wrapping policy and the sidecar contract
can be unit-tested without a full Ray training loop. Design references: full-FT
FSDP state (10.1), owner state machine offload/onload (3.2), group trajectory
sidecar write order (5.3).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = [
    "resolve_block_classes",
    "resolve_fine_grained_modules",
    "resolve_module_groups",
    "resolve_replicated_modules",
    "fsdp2_wrap",
    "upcast_trainable_params",
    "offload_module_to_cpu",
    "onload_module_to_device",
    "write_trajectory_sidecar",
    "hydrate_trajectory_sidecar",
]

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_block_classes(model: torch.nn.Module) -> List[Type[torch.nn.Module]]:
    """Discover transformer block classes to shard, via ``_no_split_modules``.

    HF-style transformers declare ``_no_split_modules`` (e.g.
    ``("QwenImageTransformerBlock",)``); each named class found on any
    submodule is a shard boundary. Falls back to the model's direct
    ``ModuleList`` children when the attribute is absent.
    """
    names = set(getattr(model, "_no_split_modules", None) or [])
    classes: List[Type[torch.nn.Module]] = []
    seen = set()
    for module in model.modules():
        cls = type(module)
        if cls.__name__ in names and cls not in seen:
            seen.add(cls)
            classes.append(cls)
    if not classes:
        for module in model.modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                cls = type(module[0])
                if cls not in seen:
                    seen.add(cls)
                    classes.append(cls)
    return classes


def resolve_fine_grained_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Resolve explicitly marked inner FSDP boundaries.

    Model policies may set ``_fsdp_fine_grained`` on selected submodules. These
    modules are nested inside the transformer block boundary and must be
    sharded before their parents. Instance markers avoid sharding every module
    of a broad class such as ``Linear``.
    """
    return [module for module in model.modules() if getattr(module, "_fsdp_fine_grained", False)]


def resolve_module_groups(model: torch.nn.Module) -> List[List[torch.nn.Module]]:
    """Validate optional adapter-declared sibling communication groups.

    Members must be called through ``Module.__call__``. Grouping siblings does
    not move/register them again, so checkpoint names remain unchanged. Groups
    and existing fine-grained boundaries must have disjoint module/parameter
    ownership; their common parents are sharded afterwards.
    """
    declared = getattr(model, "_fsdp_module_groups", ())
    if not isinstance(declared, (list, tuple)):
        raise ValueError("_fsdp_module_groups must be a sequence of module sequences")
    available = set(model.modules()) - {model}
    owned_modules: set[torch.nn.Module] = set()
    owned_parameters: set[torch.nn.Parameter] = set()
    for module in resolve_fine_grained_modules(model):
        owned_modules.update(module.modules())
        owned_parameters.update(module.parameters())
    groups = []
    for members in declared:
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError("Each FSDP module group must be a nonempty module sequence")
        group = []
        for module in members:
            if not isinstance(module, torch.nn.Module) or module not in available:
                raise ValueError("FSDP group members must be non-root modules belonging to the model")
            descendants, parameters = set(module.modules()), set(module.parameters())
            if owned_modules.intersection(descendants) or owned_parameters.intersection(parameters):
                raise ValueError("FSDP module groups must have disjoint module and parameter ownership")
            owned_modules.update(descendants)
            owned_parameters.update(parameters)
            group.append(module)
        groups.append(group)
    return groups


def _parameter_storage_identity(parameter: torch.nn.Parameter) -> tuple[torch.device, int] | None:
    """Identify shared dense storage without reading data or synchronizing
    CUDA."""
    if parameter.device.type == "meta" or parameter.layout != torch.strided:
        return None
    storage = parameter.untyped_storage()
    return (parameter.device, storage.data_ptr()) if storage.nbytes() else None


def resolve_replicated_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Validate optional frozen subtrees excluded from every FSDP boundary.

    The adapter keeps modules registered and must place their parameters on the
    correct device/dtype before wrapping: native ``ignored_params`` neither
    moves nor casts them and never reduces their gradients. Sharing within one
    frozen subtree is allowed; aliases outside it or into a sharded boundary
    are rejected before any parameter or module mutation.
    """
    declared = getattr(model, "_fsdp_replicated_modules", ())
    if not isinstance(declared, (list, tuple)):
        raise ValueError("_fsdp_replicated_modules must be a sequence of modules")
    if not declared:
        return []
    available = set(model.modules()) - {model}
    owned_modules: set[torch.nn.Module] = set()
    owned_parameters: set[torch.nn.Parameter] = set()
    owned_storage: set[tuple[torch.device, int]] = set()
    replicated = []
    for module in declared:
        if not isinstance(module, torch.nn.Module) or module not in available:
            raise ValueError("FSDP replicated modules must be non-root modules belonging to the model")
        descendants, parameters = set(module.modules()), set(module.parameters())
        if any(parameter.requires_grad for parameter in parameters):
            raise ValueError("FSDP replicated modules must contain only frozen parameters")
        storage = {identity for parameter in parameters if (identity := _parameter_storage_identity(parameter))}
        if (
            owned_modules.intersection(descendants)
            or owned_parameters.intersection(parameters)
            or owned_storage.intersection(storage)
        ):
            raise ValueError("FSDP replicated modules must have disjoint module and parameter ownership")
        owned_modules.update(descendants)
        owned_parameters.update(parameters)
        owned_storage.update(storage)
        replicated.append(module)

    blocks = tuple(resolve_block_classes(model))
    boundaries = [
        *resolve_fine_grained_modules(model),
        *(module for group in resolve_module_groups(model) for module in group),
        *(module for module in model.modules() if isinstance(module, blocks)),
    ]
    for boundary in boundaries:
        if owned_modules.intersection(boundary.modules()) or owned_parameters.intersection(boundary.parameters()):
            raise ValueError("FSDP replicated modules overlap a fine-grained, grouped or block sharding boundary")

    try:
        registrations = list(model.named_modules(remove_duplicate=False))
    except RecursionError as error:
        raise ValueError("FSDP replicated modules require acyclic module registration") from error
    prefixes = []
    for module in replicated:
        names = [name for name, candidate in registrations if candidate is module]
        if len(names) != 1:
            raise ValueError("FSDP replicated modules must not have multiple registration aliases")
        prefixes.append(names[0])
    for name, module in registrations:
        inside = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        if inside:
            continue
        if module in owned_modules:
            raise ValueError("FSDP replicated module has an alias outside its declared subtree")
        for parameter in module.parameters(recurse=False):
            if parameter in owned_parameters or _parameter_storage_identity(parameter) in owned_storage:
                raise ValueError("FSDP replicated parameter has an alias outside its declared subtree")
    return replicated


@torch.no_grad()
def upcast_trainable_params(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """Promote only the TRAINABLE parameters to ``dtype``; return the count.

    Under LoRA the base is frozen and only the adapter carries gradients, so
    this raises the precision of the optimizer's master copy without touching
    the frozen bulk. It must run before any ``fully_shard`` (DTensors are
    skipped, and they cannot be re-cast this way afterwards).

    This is not a micro-optimization: with a bf16 master, the ~1e-4 relative
    updates a diffusion RL run produces round off the mantissa on every step,
    so the policy stops moving while grad norm and loss still look healthy.
    ``MixedPrecisionPolicy.param_dtype`` is deliberately left alone, so the copy
    that participates in the forward is still bf16 and the arithmetic -- hence
    the GRPO ratio -- matches full FT exactly.
    """
    from torch.distributed.tensor import DTensor

    count = 0
    for param in model.parameters():
        if isinstance(param, DTensor) or not param.dtype.is_floating_point or not param.requires_grad:
            continue
        if param.dtype != dtype:
            param.data = param.data.to(dtype)
            count += 1
    return count


def fsdp2_wrap(
    model: torch.nn.Module,
    *,
    device_mesh=None,
    param_dtype: str = "bf16",
    reduce_dtype: str = "fp32",
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    master_dtype: Optional[str] = None,
) -> torch.nn.Module:
    """Apply FSDP2 ``fully_shard`` per transformer block, then to the root.

    RL passes its trainable transformer; image SFT passes the complete forward
    root. Each block is sharded first so the root wrap only holds the leftover
    parameters. An adapter may declare frozen ``_fsdp_replicated_modules`` to
    exclude those parameters from the root as well. The adapter must put them
    on the correct device/dtype; FSDP does not manage ignored parameters.

    When ``activation_checkpointing`` is set, each transformer block is wrapped
    with a non-reentrant checkpoint FIRST (so the block's forward activations are
    recomputed in backward instead of stored) and then sharded — this is the
    dominant lever for the FlowGRPO replay, whose grad graph otherwise holds the
    full-resolution activations of every trained SDE step.

    ``master_dtype`` (LoRA runs: ``fp32``) upcasts the trainable parameters
    before sharding; see :func:`upcast_trainable_params`. ``None`` -- the
    default -- leaves full FT bit-identical.

    FSDP2 shards each parameter individually, so a block holding trainable LoRA
    adapters alongside frozen base weights is fine (unlike FSDP1's FlatParameter,
    which required uniform ``requires_grad`` per unit).
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(
        param_dtype=_DTYPES[param_dtype],
        reduce_dtype=_DTYPES[reduce_dtype],
    )
    offload_policy = CPUOffloadPolicy() if cpu_offload else None
    kwargs = {"mp_policy": mp_policy, "reshard_after_forward": reshard_after_forward}
    if device_mesh is not None:
        kwargs["mesh"] = device_mesh
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy

    replicated_modules = resolve_replicated_modules(model)
    if master_dtype is not None:
        promoted = upcast_trainable_params(model, _DTYPES[master_dtype])
        logger.info(f"fsdp2_wrap: promoted {promoted} trainable param(s) to {master_dtype} optimizer masters")

    block_classes = tuple(resolve_block_classes(model))
    fine_grained_modules = resolve_fine_grained_modules(model)
    module_groups = resolve_module_groups(model)
    shard_targets: tuple = block_classes
    if block_classes and activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            CheckpointWrapper,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        def _wrap(module: torch.nn.Module) -> torch.nn.Module:
            return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=_wrap, check_fn=lambda m: isinstance(m, block_classes)
        )
        # After AC the blocks are wrapped, so shard the CheckpointWrappers.
        shard_targets = (CheckpointWrapper,)

    for module in fine_grained_modules:
        fully_shard(module, **kwargs)
    for modules in module_groups:
        fully_shard(modules, **kwargs)
    if block_classes:
        for module in model.modules():
            if isinstance(module, shard_targets):
                fully_shard(module, **kwargs)
    if replicated_modules:
        ignored_params = {parameter for module in replicated_modules for parameter in module.parameters()}
        fully_shard(model, ignored_params=ignored_params, **kwargs)
    else:
        fully_shard(model, **kwargs)
    logger.info(
        f"FSDP2 wrapped model; fine-grained modules: {len(fine_grained_modules)} "
        f"({sorted({type(m).__name__ for m in fine_grained_modules})}); "
        f"module groups: {len(module_groups)}; "
        f"block classes: {[c.__name__ for c in block_classes]}; "
        f"activation_checkpointing={activation_checkpointing}, cpu_offload={cpu_offload}, "
        f"master_dtype={master_dtype}"
    )
    return model


_PIN_MEMORY = os.environ.get("RELAX_FSDP_OFFLOAD_PIN_MEMORY", "1") == "1"


def _storage_carrier(t: torch.Tensor) -> torch.Tensor:
    """The tensor that actually owns the bytes.

    For an FSDP2 parameter this is the DTensor's local shard; for a plain
    tensor it is the tensor itself.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - torch built without DTensor
        return t
    return t._local_tensor if isinstance(t, DTensor) else t


@torch.no_grad()
def _offload_storage(t: torch.Tensor) -> None:
    """Free ``t``'s device storage, keeping a pinned CPU copy of the bytes.

    The tensor object, its device and its metadata are left untouched — only
    the underlying allocation is released. That distinction is the whole point:
    FSDP2's ``FSDPParam._sharded_param_data`` (the buffer ``all_gather_inputs``
    reads) ALIASES the same storage as the parameter's ``_local_tensor``, so
    resizing that one storage moves both views together. Rebinding
    ``_local_tensor`` to a CPU tensor instead would desynchronize them and the
    forward would silently keep reading the stale pre-offload weights forever.

    Reassigning ``param.data`` does not work either: ``.data`` on a DTensor
    parameter returns a fresh alias whose ``_local_tensor`` is a different
    object, so the write never reaches the parameter and the offload is a
    no-op (the previous implementation freed exactly zero bytes under
    ``fully_shard``).

    The host copy is pinned and reused across sleep/wake cycles: pinned memory
    lets the driver DMA straight from the device instead of staging through a
    bounce buffer, and reusing the buffer avoids re-paying the page-locking
    cost plus the transient 2x host peak a reallocation would cause. Mirrors
    :class:`relax.backends.megatron.weight_update.train_offload._SelectiveOffloadStrategy`.
    """
    local = _storage_carrier(t)
    if local.device.type == "cpu":
        return
    storage = local.untyped_storage()
    if storage.size() == 0:  # already offloaded
        return
    cpu_data = getattr(local, "_relax_cpu_data", None)
    if cpu_data is None or cpu_data.shape != local.shape or cpu_data.dtype != local.dtype:
        cpu_data = torch.empty_like(local, device=torch.device("cpu"), pin_memory=_PIN_MEMORY)
        local._relax_cpu_data = cpu_data
    # Synchronous D2H: the copy must land before resize_(0) frees the source.
    cpu_data.copy_(local, non_blocking=False)
    local._relax_offload_bytes = storage.size()
    storage.resize_(0)


@torch.no_grad()
def _onload_storage(t: torch.Tensor) -> None:
    """Reallocate ``t``'s device storage and restore it from the CPU copy."""
    local = _storage_carrier(t)
    nbytes = getattr(local, "_relax_offload_bytes", None)
    if nbytes is None:  # never offloaded by us
        return
    storage = local.untyped_storage()
    if storage.size() == 0:
        storage.resize_(nbytes)
    local.copy_(local._relax_cpu_data, non_blocking=True)
    del local._relax_offload_bytes


@torch.no_grad()
def offload_module_to_cpu(module: torch.nn.Module) -> None:
    """Release a module's parameter/buffer device memory (ROLLOUT / REWARD
    phases).

    Pairs with :func:`onload_module_to_device`. Between the two the module is
    unusable — every parameter has zero-sized storage — which is exactly the
    contract the colocate owner state machine wants.
    """
    for p in module.parameters(recurse=True):
        _offload_storage(p)
    for b in module.buffers(recurse=True):
        _offload_storage(b)
    torch.cuda.empty_cache()


@torch.no_grad()
def onload_module_to_device(module: torch.nn.Module, device: torch.device) -> None:
    """Restore a module's parameters and buffers onto ``device``.

    ``device`` is accepted for symmetry and validated against where the
    storages actually live: offload never moved the tensors, so they come back
    exactly where they were.
    """
    for p in module.parameters(recurse=True):
        _onload_storage(p)
    for b in module.buffers(recurse=True):
        _onload_storage(b)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Trajectory sidecar I/O (safetensors, atomic write order)
# ---------------------------------------------------------------------------


def write_trajectory_sidecar(
    path: str,
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Write a group trajectory sidecar with the fixed atomic order.

    ``.tmp -> fsync -> atomic rename`` (design doc 5.3) so a manifest that
    references the sidecar is only committed after the file is durable. Tensors
    are moved to CPU and made contiguous for safetensors.
    """
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cpu_tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    tmp = path + ".tmp"
    save_file(cpu_tensors, tmp, metadata=metadata)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return path


def hydrate_trajectory_sidecar(
    path: str,
    keys: Optional[Iterable[str]] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Load a trajectory sidecar; optionally only the requested ``keys``.

    Returns ``(tensors, metadata)``. Uses safetensors lazy loading so a replay
    step can hydrate just the tensors it needs.
    """
    from safetensors import safe_open

    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=device) as f:
        metadata = dict(f.metadata() or {})
        wanted = list(keys) if keys is not None else list(f.keys())
        for k in wanted:
            tensors[k] = f.get_tensor(k)
    return tensors, metadata
