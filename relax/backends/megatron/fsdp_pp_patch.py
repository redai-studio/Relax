# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Make Megatron-FSDP's DeviceMesh construction pipeline-parallel aware (#14).

``megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`` builds the FSDP device
mesh from ``torch.arange(dist.get_world_size())`` with the einops pattern
``"(dp_cp ep tp) -> ep dp_cp tp"``. That assumes ``world_size == dp_cp * ep * tp``,
but the real relation is ``world_size == dp_cp * ep * tp * pp_size``: the pattern
has no ``pp`` axis. With ``--use-megatron-fsdp --pipeline-model-parallel-size 2``
on 16 GPUs the einops call raises ``Shape mismatch, 16 != 8`` before the model is
even built. Upstream simply never tested the combination — ``--use-torch-fsdp2``
carries an explicit ``pipeline_model_parallel_size == 1`` assertion
(``megatron/training/arguments.py``) while the ``use_megatron_fsdp`` branch does not.

Megatron's default rank order is ``'tp-cp-ep-dp-pp'``, i.e. ``pp`` is the slowest
varying dimension, so **each pipeline stage occupies a contiguous block of ranks**.
The fix is therefore to feed the mesh builder this rank's stage block instead of
the whole world. Verified against Megatron's own ``RankGenerator`` over 1110 valid
``(world, tp, cp, pp, ep, etp)`` combinations, every rank, both the dense and the
expert mesh path: zero mismatches, and byte-identical output when ``pp_size == 1``.

Installed as a Relax-side monkeypatch rather than an edit to ``/root/Megatron-LM``
so it survives Megatron reinstalls (see debug doc #13).

A second, independent PP blocker lives here too: see
:func:`patch_dtensor_meta_check_for_pp` (#15), which clears the DTensor metadata
check that Megatron-FSDP runs on the global process group once per parameter.
"""

import re
import types
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_PP_MESH_MARKER = "_relax_pp_stage_mesh"
_META_CHECK_MARKER = "_relax_pp_meta_check_noop"
_EMBD_BCAST_MARKER = "_relax_pp_embd_broadcast_local"
_EMBD_GRAD_MARKER = "_relax_pp_embd_grad_reshard"
_PP_CKPT_MARKER = "_relax_pp_ckpt_global_layers"


def pp_stage_window(world_size: int, rank: int, pp_ranks: Sequence[int]) -> tuple[int, int]:
    """Return ``(stage_base, stage_size)`` for the pipeline stage that owns
    ``rank``.

    Pure arithmetic — no ``torch.distributed``, no Megatron — so it is unit testable
    on CPU. ``pp_ranks`` is the global rank list of *this* rank's pipeline-model-parallel
    group, i.e. one rank per stage.

    Raises:
        ValueError: on a malformed ``pp_ranks`` (empty, not covering ``rank``, or a
            pipeline size that does not divide ``world_size``).
        RuntimeError: when the pipeline stages are not contiguous equal-size rank
            blocks, which means the rank order is not Megatron's default and the
            whole stage-window approach does not apply.
    """
    pp_ranks = list(pp_ranks)
    if not pp_ranks:
        raise ValueError("[Relax] Empty pipeline-model-parallel rank list.")
    pp_size = len(pp_ranks)
    if world_size % pp_size:
        raise ValueError(
            f"[Relax] world_size={world_size} is not divisible by pipeline size {pp_size} (pp group ranks {pp_ranks})."
        )
    if rank not in pp_ranks:
        raise ValueError(f"[Relax] Rank {rank} is not a member of its own pp group {pp_ranks}.")

    stage_size = world_size // pp_size
    ordered = sorted(pp_ranks)
    expected = [ordered[0] + i * stage_size for i in range(pp_size)]
    if ordered != expected:
        raise RuntimeError(
            f"[Relax] The Megatron-FSDP PP mesh patch requires pipeline parallelism to be the "
            f"outermost rank dimension (Megatron default order 'tp-cp-ep-dp-pp'), so that each "
            f"stage occupies a contiguous block of {stage_size} ranks. Got pp group ranks "
            f"{ordered}, expected {expected}. This almost always means --use-tp-pp-dp-mapping "
            f"(order 'tp-cp-ep-pp-dp') is combined with --use-megatron-fsdp and "
            f"--pipeline-model-parallel-size > 1, which is not supported."
        )

    # The check above forces ordered[0] < stage_size, so rank // stage_size is exactly
    # this rank's pipeline stage index.
    return (rank // stage_size) * stage_size, stage_size


def stage_local_arange(world_size: int, rank: int, pp_ranks: Sequence[int]) -> torch.Tensor:
    """``torch.arange(world_size)`` narrowed to this rank's pipeline stage
    block.

    Elementwise identical to ``torch.arange(world_size)`` when ``pp_size ==
    1``.
    """
    base, size = pp_stage_window(world_size, rank, pp_ranks)
    return torch.arange(size) + base


def _current_pp_ranks() -> list[int]:
    """Global ranks of this rank's pipeline-model-parallel group."""
    from megatron.core import parallel_state

    group = parallel_state.get_pipeline_model_parallel_group()
    if isinstance(group, (list, tuple)):
        # parallel_state binds a *list* when a rank belongs to several pp groups
        # (encoder/decoder split). Every entry has the same stride, so the first
        # one is enough to derive the stage window.
        group = group[0]
    return sorted(dist.get_process_group_ranks(group))


class _StageLocalTorch:
    """Pass-through proxy for the ``torch`` module used by the FSDP mesh
    builders.

    Narrows a full-world ``arange(world_size)`` to the calling rank's pipeline
    stage block and forwards everything else untouched. ``torch.arange``
    appears exactly twice in ``mcore_fsdp_adapter`` — the two buggy mesh
    builders — so the intercept is precise. The ``args[0] == world_size``
    predicate also means the shim steps aside automatically if upstream ever
    fixes the arange itself.
    """

    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(torch, name)

    def arange(self, *args, **kwargs):
        world_size = dist.get_world_size(group=dist.group.WORLD)
        if len(args) == 1 and not kwargs and args[0] == world_size:
            return stage_local_arange(world_size, dist.get_rank(group=dist.group.WORLD), _current_pp_ranks())
        return torch.arange(*args, **kwargs)


def patch_fsdp_mesh_for_pp() -> None:
    """Make Megatron-FSDP's device-mesh builders pipeline-parallel aware.

    Rebinds ``torch`` to :class:`_StageLocalTorch` inside a copy of each mesh
    builder's globals. Copying the globals (rather than mutating ``adapter.torch``)
    keeps the patch stateless — no ``try``/``finally``, no pollution of a
    third-party namespace, and safe under VPP where the builders run once per
    model chunk. Delegating to the untouched upstream bytecode keeps the ~110
    lines of einops patterns and consistency assertions in one place.

    Idempotent. Only the defining module needs patching: nothing in Megatron
    imports these two names, and both call sites resolve them from module globals.
    """
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter as adapter

    if getattr(adapter._get_dp_tp_mesh, _PP_MESH_MARKER, False):
        return

    shim = _StageLocalTorch()
    for name in ("_get_dp_tp_mesh", "_get_hsdp_tp_mesh"):
        orig = getattr(adapter, name)
        patched = types.FunctionType(
            orig.__code__,
            {**orig.__globals__, "torch": shim},
            orig.__name__,
            orig.__defaults__,
            orig.__closure__,
        )
        patched.__kwdefaults__ = orig.__kwdefaults__
        patched.__qualname__ = orig.__qualname__
        patched.__module__ = orig.__module__
        patched.__doc__ = orig.__doc__
        setattr(patched, _PP_MESH_MARKER, True)
        setattr(adapter, name, patched)

    logger.info("Installed Relax PP-aware DeviceMesh patch for Megatron-FSDP (#14).")


def patch_dtensor_meta_check_for_pp() -> None:
    """Disable ``DTensor.from_local``'s cross-rank metadata check under PP
    (#15).

    ``torch.distributed.tensor._collective_utils.check_tensor_meta`` all-gathers each
    parameter's metadata on the **default (global) process group** — it never looks at
    the DeviceMesh it is supposedly validating::

        gathered_metadata = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_metadata, local_metadata)

    Megatron-FSDP calls it once per TP-sharded parameter, with ``run_check=True``
    hardcoded (``param_and_grad_buffer.py`` ``_init_distributed_params``). Under
    pipeline parallelism each stage owns a *different* parameter list, so the two
    stages issue a different number of these world-wide collectives. They desync, and
    calls for unrelated parameters get paired up — producing both a bogus
    ``ValueError: Inconsistent tensor metadata`` and, further downstream, corrupted
    payloads for genuinely unrelated ``all_gather_object`` calls (``_get_param_groups``
    receives this function's metadata *dict* instead of a list of param-group keys).

    Neutering the check — rather than passing ``run_check=False`` — is deliberate:
    ``run_check`` also drives ``mesh_broadcast`` for ``Replicate()`` placements
    (``_api.py``), which is a real data sync, not a validation. Turning ``run_check``
    off would silently leave TP-duplicated parameters unsynchronized. This patch
    removes only the broken collective and leaves the broadcast intact.

    The check is pure validation (it returns ``None`` or raises), so a no-op is
    behaviourally exact minus the check itself.

    Idempotent. Patches both the defining module and the ``from ... import`` rebinding
    in ``_api``, which is the one actually called (see debug doc #13).
    """
    from torch.distributed.tensor import _api as dtensor_api
    from torch.distributed.tensor import _collective_utils

    if getattr(dtensor_api.check_tensor_meta, _META_CHECK_MARKER, False):
        return

    def _noop_check_tensor_meta(local_tensor, check_shape_stride=False):
        # Upstream returns None on success; nothing reads the return value.
        return None

    setattr(_noop_check_tensor_meta, _META_CHECK_MARKER, True)
    # _api.py does `from ._collective_utils import check_tensor_meta`, binding the name
    # in its own namespace — that binding is the live call site, so it must be replaced.
    # Patch the defining module too, so any later importer gets the no-op as well.
    dtensor_api.check_tensor_meta = _noop_check_tensor_meta
    _collective_utils.check_tensor_meta = _noop_check_tensor_meta

    logger.info("Disabled DTensor cross-rank metadata check for Megatron-FSDP + PP (#15).")


@contextmanager
def _dtensor_transparent_broadcast():
    """Temporarily let ``torch.distributed.broadcast`` accept DTensor
    arguments.

    c10d collectives have no DTensor sharding strategy registered, so handing a
    DTensor to ``dist.broadcast`` raises ``NotImplementedError: Operator
    c10d.broadcast_.default does not have a sharding strategy registered``.
    Unwrapping to the local shard is safe for an in-place receive because
    ``to_local()`` aliases the DTensor's own storage rather than copying it.

    Captures whatever ``dist.broadcast`` is bound to at *call* time, so it composes
    with ``relax.utils.reloadable_process_group.monkey_patch_torch_dist``, which has
    already wrapped it by then.
    """
    original_broadcast = dist.broadcast

    def _broadcast_local(tensor, *args, **kwargs):
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return original_broadcast(tensor, *args, **kwargs)

    dist.broadcast = _broadcast_local
    try:
        yield
    finally:
        dist.broadcast = original_broadcast


def patch_bridge_shared_embedding_broadcast_for_pp() -> None:
    """Let the bridge sync tied embeddings across PP stages under FSDP (#16).

    With tied embeddings and ``PP > 1`` the first stage holds
    ``embedding.word_embeddings`` and the last stage holds ``output_layer``; the two
    must carry identical weights. ``MegatronModelBridge._broadcast_shared_embeddings``
    reconciles them with a plain ``torch.distributed.broadcast`` over the embedding
    group, which explodes under Megatron-FSDP because the weight is a DTensor (same
    failure family as debug doc #4).

    Broadcasting the *local shards* instead is equivalent here by construction rather
    than by luck: Megatron-FSDP puts shared-embedding parameters in their own bucket
    precisely so "the first and last pipeline stage partition optimizer state for the
    shared embedding parameters the same way across DP replicas"
    (``param_and_grad_buffer.py::_does_param_require_new_bucket``). Matching layouts
    mean shard *i* on the first stage corresponds to shard *i* on the last stage.

    Upstream's body is left untouched — only ``dist.broadcast`` is made DTensor-aware
    for the duration of the call. The receive path then works unmodified:
    ``torch.empty_like(<DTensor>)`` yields a DTensor whose ``to_local()`` aliases its
    storage, and the trailing ``output_layer.weight.data.copy_(embd_weights)`` is a
    DTensor-to-DTensor copy, which DTensor does implement.

    Idempotent. Pointless at PP == 1 (upstream returns early), so it is installed
    behind the same ``pipeline_model_parallel_size > 1`` gate as the other PP patches.
    """
    from megatron.bridge.models.conversion import model_bridge

    bridge_cls = model_bridge.MegatronModelBridge
    if getattr(bridge_cls._broadcast_shared_embeddings, _EMBD_BCAST_MARKER, False):
        return

    _orig = bridge_cls._broadcast_shared_embeddings

    def _patched(self, megatron_model):
        with _dtensor_transparent_broadcast():
            return _orig(self, megatron_model)

    setattr(_patched, _EMBD_BCAST_MARKER, True)
    bridge_cls._broadcast_shared_embeddings = _patched

    logger.info("Installed Relax DTensor-aware shared-embedding broadcast for Megatron-FSDP + PP (#16).")


def patch_embedding_grad_reshard_for_pp() -> None:
    """Fix the tied-embedding gradient all-reduce under Megatron-FSDP + PP
    (#21).

    With tied embeddings and PP>1, ``finalize_model_grads.py::_allreduce_embedding_grad``
    syncs the first stage's ``word_embeddings`` gradient with the last stage's
    ``output_layer`` gradient. For Megatron-FSDP it deliberately skips the usual
    ``full_tensor()`` gather and all-reduces just the raw FSDP-local shard
    (``orig_grad._local_tensor``) — a valid bandwidth optimization, since Megatron-FSDP
    puts tied-embedding parameters in a matched bucket so shard *i* on the first stage
    lines up with shard *i* on the last stage (same reasoning as #16). The bug is what
    happens *after*: it reassigns that raw local shard straight to ``weight.grad``
    via ``_reshard_if_dtensor(grad, orig_grad)`` — but by then ``orig_grad`` has
    already been overwritten with the unwrapped ``._local_tensor``, so
    ``_reshard_if_dtensor``'s ``isinstance(reference_tensor, DTensor)`` check is
    False and it returns the raw shard untouched. PyTorch's native ``.grad`` setter
    then rejects it: ``weight`` (a DTensor) declares a much larger shape than the
    lone FSDP shard —

        RuntimeError: attempting to assign a gradient of size '[37984, 2560]' to a
        tensor of size '[75968, 2560]'. Please ensure that the gradient and the
        tensor are the same size

    Fixed by keeping a reference to the *original* (still-wrapped) gradient DTensor
    and, for the Megatron-FSDP branch, rewrapping the all-reduced local shard with
    ``DTensor.from_local(..., shape=dtensor_grad.shape, stride=dtensor_grad.stride())``
    before the ``setattr`` — mirroring exactly how ``make_fsdp_dtensor`` itself
    declares a DTensor's shape (explicit ``shape=``/``stride=``, not auto-computed
    from placements, since the mesh also carries the TP dim and auto-computing would
    double-count it). Verified against the real ``make_fsdp_dtensor`` in a 4-rank
    (TP=2 x FSDP-shard=2) run: reconstructing this way reproduces the original
    DTensor's exact ``.shape``/``.stride()``, and ``weight.grad = <reconstructed>``
    succeeds where assigning the raw local shard directly raises the exact error above.

    Non-FSDP models fall through to the untouched original implementation.

    Idempotent. Only the defining module needs patching: both call sites
    (``_allreduce_word_embedding_grads`` and ``_allreduce_position_embedding_grads``)
    resolve ``_allreduce_embedding_grad`` from module globals at call time.

    Fetched via ``importlib.import_module`` rather than
    ``from megatron.core.distributed import finalize_model_grads`` — the package's
    own ``__init__.py`` does ``from .finalize_model_grads import finalize_model_grads``,
    which rebinds the *attribute* ``megatron.core.distributed.finalize_model_grads``
    to that function, shadowing the submodule of the same name. Both ``from ... import``
    and ``import a.b.c as x`` resolve via that (shadowed) attribute and would silently
    hand back the function instead of the module, blowing up on
    ``fmg._allreduce_embedding_grad`` with ``AttributeError: 'function' object has no
    attribute '_allreduce_embedding_grad'``. The submodule itself is still registered
    in ``sys.modules`` regardless of what the package attribute points to, which is
    exactly what ``importlib.import_module`` reads from.
    """
    import importlib

    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    if getattr(fmg._allreduce_embedding_grad, _EMBD_GRAD_MARKER, False):
        return

    original = fmg._allreduce_embedding_grad

    def _patched_allreduce_embedding_grad(model, embd_group, pp_group, weight_getter, skip_if_none=True, config=None):
        if not (
            fmg.get_pg_size(embd_group) > 1
            and dist.get_rank(group=dist.group.WORLD) in dist.get_process_group_ranks(embd_group)
        ):
            return

        if fmg.is_pp_first_stage(pp_group):
            model_module = model[0]
        elif fmg.is_pp_last_stage(pp_group):
            model_module = model[-1]
        elif getattr(config, "mtp_num_layers", None) is not None and config.mtp_num_layers > 0:
            model_module = model[-1]
        else:
            model_module = model[0]

        ddp_config = model_module.ddp_config
        if not getattr(ddp_config, "use_megatron_fsdp", False):
            return original(model, embd_group, pp_group, weight_getter, skip_if_none=skip_if_none, config=config)

        model_module = fmg.get_attr_wrapped_model(model_module, "pre_process", return_model_obj=True)
        weight = weight_getter(model_module)
        if weight is None and skip_if_none:
            return

        grad_attr = "decoupled_grad" if hasattr(weight, "decoupled_grad") else fmg._get_main_grad_attr(weight)
        dtensor_grad = getattr(weight, grad_attr)
        if dtensor_grad is None:
            if skip_if_none:
                return
            local_grad = None
        else:
            local_grad = dtensor_grad._local_tensor

        if local_grad is None and skip_if_none:
            return

        dist.all_reduce(local_grad, group=embd_group)

        from torch.distributed.tensor import DTensor

        setattr(
            weight,
            grad_attr,
            DTensor.from_local(
                local_grad,
                device_mesh=dtensor_grad.device_mesh,
                placements=dtensor_grad.placements,
                shape=dtensor_grad.shape,
                stride=dtensor_grad.stride(),
                run_check=False,
            ),
        )

    setattr(_patched_allreduce_embedding_grad, _EMBD_GRAD_MARKER, True)
    fmg._allreduce_embedding_grad = _patched_allreduce_embedding_grad

    logger.info("Installed Relax DTensor-aware embedding-gradient reshard for Megatron-FSDP + PP (#21).")


def _is_megatron_checkpoint_path(path: Optional[Union[str, Path]]) -> bool:
    if not path:
        return False
    path = Path(path)
    return (path / "latest_checkpointed_iteration.txt").is_file() or bool(re.fullmatch(r"iter_\d{7}", path.name))


def check_fsdp_pp_checkpoint_support(args, role: str = "actor") -> None:
    """Refuse ``fsdp_dtensor`` + PP combinations this repo's P2 fix doesn't
    cover.

    :func:`patch_checkpoint_layer_numbering_for_pp` fixes the general PP>1 case
    (stage-local ``decoder.layers.0...`` collisions across pipeline stages, see its
    docstring). Virtual pipeline parallelism is not covered by that fix and is refused
    here instead of silently mis-numbering:

    - Virtual pipeline parallelism (multiple model chunks per rank): the patched
      ``preprocess_fsdp_dtensor_state_dict`` only ever receives ``model[0]``, so a
      second-or-later VPP chunk on the same rank would get the *first* chunk's layer
      offset.

    The check applies when the role will save a checkpoint or resume a native Megatron
    checkpoint. HuggingFace initialization is unaffected because it does not use the
    unsupported ``fsdp_dtensor`` preprocessing path.
    """
    if not getattr(args, "use_megatron_fsdp", False):
        return
    if getattr(args, "pipeline_model_parallel_size", 1) <= 1:
        return
    if getattr(args, "ckpt_format", None) != "fsdp_dtensor":
        return
    if role == "critic":
        load_path = getattr(args, "critic_load", None)
        save_path = getattr(args, "critic_save", None)
    elif role in ("reference", "actor_fwd"):
        load_path = getattr(args, "ref_load", None)
        save_path = None
    else:
        load_path = getattr(args, "load", None)
        save_path = getattr(args, "save", None)
    if not save_path and not _is_megatron_checkpoint_path(load_path):
        return
    if (getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) > 1:
        raise RuntimeError(
            "[Relax] --ckpt-format fsdp_dtensor with --use-megatron-fsdp, "
            "--pipeline-model-parallel-size > 1, and virtual pipeline parallelism "
            "(--num-virtual-stages-per-pipeline-rank / --virtual-pipeline-model-parallel-size) "
            "is not supported: the global-layer-numbering patch only handles one model chunk "
            "per rank. Drop VPP, or disable native Megatron checkpoint load/save for a smoke test."
        )


def _pp_global_layer_offset(config) -> int:
    """Global (0-indexed) layer offset for this rank's pipeline stage.

    Thin wrapper around Megatron's own ``get_transformer_layer_offset`` — the exact
    function every live ``TransformerLayer.layer_number`` is derived from
    (``transformer_layer.py``: ``self.layer_number = local_1_based_idx +
    get_transformer_layer_offset(self.config, vp_stage, pp_rank)``), and the same one
    Relax's own (already-tested) ``weight_update/common.py::_named_params_and_buffers_global``
    uses for the equivalent rename on the live weight-sync path. ``vp_stage``/``pp_rank``
    are left at their defaults (auto-derived from ``parallel_state``): this patch only
    supports one model chunk per rank (see :func:`check_fsdp_pp_checkpoint_support`).
    """
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    return get_transformer_layer_offset(config)


_LAYER_INDEX_RE = re.compile(r"layers\.(\d+)")


def _rename_layers_to_global(state_dict: dict, offset: int) -> None:
    """Rewrite ``layers.<local>`` keys to ``layers.<local + offset>`` in place.

    Mutates ``state_dict`` in place via a two-phase remove/re-insert under the new
    key, so overlapping source/destination keys cannot overwrite values that have
    not been moved yet. The tensor *object* is untouched, only its dict key changes.
    That matters: Megatron's
    ``fsdp_dtensor`` load path (``_load_base_checkpoint``) snapshots
    ``state_dict["model"]``/``["optimizer"]`` *before* calling
    ``preprocess_fsdp_dtensor_state_dict`` (which this function is hooked into), lets
    ``torch.distributed.checkpoint.load_state_dict`` fill the renamed dict's tensors
    in place, then restores the snapshot as the dict Megatron's own ``model.load_state_dict()``
    consumes. Since the snapshot and the renamed dict share the same tensor objects,
    restoring the original (local-named) keys afterward automatically un-does this
    rename with no separate "reverse on load" step needed — the loaded *values* are
    already sitting in the right place regardless of which key currently points at them.

    MTP has its own layer numbering. Megatron requires all MTP layers on one
    stage, so its MTP offset is zero even when the decoder offset is not. Preserve
    both ``mtp.layers.<index>`` and nested ``mtp_model_layer.layers.<index>`` keys.
    """
    if offset == 0:
        return
    renames = []
    for key, value in list(state_dict.items()):
        if "mtp" in key:
            continue
        match = _LAYER_INDEX_RE.search(key)
        if match is None:
            continue
        local_idx = int(match.group(1))
        new_key = _LAYER_INDEX_RE.sub(f"layers.{local_idx + offset}", key, count=1)
        renames.append((key, new_key, value))

    for key, _, _ in renames:
        del state_dict[key]
    for _, new_key, value in renames:
        state_dict[new_key] = value


def patch_checkpoint_layer_numbering_for_pp() -> None:
    """Make ``fsdp_dtensor`` checkpoints use globally-unique layer numbers
    under PP (#22).

    Every pipeline stage's ``TransformerBlock`` numbers its own layers starting from 0
    (``decoder.layers.0``, ``decoder.layers.1``, ...) — that is correct for the stage's
    own forward pass, but ``megatron/training/checkpointing.py::generate_state_dict``
    saves ``fsdp_dtensor`` model weights via the plain ``model[i].state_dict_for_save_checkpoint()``
    (unlike ``torch_dist``, which routes through the model's own PP-aware
    ``sharded_state_dict()``). ``torch.distributed.checkpoint.save`` is a single global
    keyspace and dedups identical keys, so every stage's "layer 0" collides and only one
    stage's copy survives — the others are silently dropped on save, and on load, every
    stage loads that *one* surviving stage's weights into its own local layer slots
    (a silent duplication, not just a loss).

    The same bug hits optimizer state: ``DistributedOptimizer.sharded_param_state_fsdp_dtensor``
    keys ``exp_avg``/``exp_avg_sq`` via ``self._param_name(param)``
    (``distrib_optimizer.py``), which already rewrites expert indices
    (``handle_experts_in_state_dict``) but does no layer-offset rewriting at all — same
    stage-local collision, independent of the model-weight bug.

    Single patch, one hook point, for both: wrap
    ``checkpointing.preprocess_fsdp_dtensor_state_dict`` — already the hook point for
    swiglu-split and expert-index rewriting, called identically at save and at load's
    skeleton-prep step (see :func:`_rename_layers_to_global`'s docstring for why one
    direction of rename is enough for both save and load) — to additionally rename
    ``state_dict["model"]`` and, if present, ``state_dict["optimizer"]["state"]`` keys
    to global layer indices, using ``get_transformer_layer_offset`` (cross-checked
    against ``TransformerLayer.layer_number``'s own construction and against Relax's
    already-tested ``weight_update/common.py::_named_params_and_buffers_global``, see
    :func:`_pp_global_layer_offset`): ``global_idx = local_idx + get_transformer_layer_offset(config)``.

    **Only patch here, deliberately not ``DistributedOptimizer._param_name``.** An
    earlier version of this patch *also* wrapped ``_param_name`` to rename its output
    directly, reasoning that Megatron's own ``param ↔ name`` bookkeeping
    (``self.param_to_name``) is internal and renaming its output would be transparent.
    That reasoning missed that ``_param_name`` fires *inside* ``generate_state_dict``,
    well *before* ``preprocess_fsdp_dtensor_state_dict`` runs — so by the time
    ``handle_swiglu_in_state_dict`` (called from the *unpatched* original body this
    function delegates to first) cross-references ``state_dict["model"]`` (still
    local-named at that point) against ``state_dict["optimizer"]["state"]`` (already
    globally-renamed by the ``_param_name`` patch), the two disagree, and
    ``model.get_parameter(f"module.{key}")`` fails with
    ``AttributeError: ModuleList has no attribute '18'`` (a real crash hit on a live
    Qwen3-4B PP=2 run — the two dicts must stay in the *same* (local) naming scheme all
    the way through Megatron's own preprocessing, and only get renamed at the very end,
    after that original body has already run and returned).

    MTP retains its independent layer numbering (all MTP layers are on one stage).
    MTP optimizer parameter-group metadata uses the same keys as its tensor state;
    retain the existing metadata layout for non-MTP checkpoint compatibility.
    Keep the original outer dicts local-named for Megatron's load-time snapshots;
    share their values so DCP writes tensors and group metadata back into them.
    Scope: single model chunk per rank (VPP is refused at startup).

    Idempotent. Pointless (and not installed) at PP == 1 — gated behind the same
    ``pipeline_model_parallel_size > 1`` condition as the other PP patches.
    """
    import importlib

    checkpointing = importlib.import_module("megatron.training.checkpointing")

    if getattr(checkpointing.preprocess_fsdp_dtensor_state_dict, _PP_CKPT_MARKER, False):
        return

    _orig_preprocess = checkpointing.preprocess_fsdp_dtensor_state_dict

    def _patched_preprocess(args, raw_state_dict, model):
        state_dict = _orig_preprocess(args, raw_state_dict, model)
        offset = _pp_global_layer_offset(model.config)
        state_dict["model"] = state_dict["model"].copy()
        _rename_layers_to_global(state_dict["model"], offset)
        if "optimizer" in state_dict:
            state_dict["optimizer"] = state_dict["optimizer"].copy()
            fields = ("state", "param_to_group_meta") if getattr(args, "mtp_num_layers", None) else ("state",)
            for field in fields:
                if field in state_dict["optimizer"]:
                    values = state_dict["optimizer"][field].copy()
                    _rename_layers_to_global(values, offset)
                    state_dict["optimizer"][field] = values
        return state_dict

    setattr(_patched_preprocess, _PP_CKPT_MARKER, True)
    checkpointing.preprocess_fsdp_dtensor_state_dict = _patched_preprocess

    logger.info("Installed Relax global-layer-numbering patch for fsdp_dtensor checkpoints + PP (#22).")
