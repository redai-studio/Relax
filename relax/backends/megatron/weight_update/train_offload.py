# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Train-state offload / reload for Megatron actors during colocate sleep/wake.
#
# Two interchangeable implementations are provided and selected at construction
# time by whether the ``torch_memory_saver`` (TMS) hook ``.so`` is LD_PRELOAD'ed
# — NOT by the accelerator vendor:
#
#   * TMS path (:class:`_TmsOffloadStrategy`): ``torch_memory_saver.pause()`` /
#     ``resume()``. TMS is semantically blind and backs up the *entire* runtime
#     memory pool via VMM. Works only when the hook ``.so`` is preloaded and the
#     allocator supports it.
#   * Selective path (:class:`_SelectiveOffloadStrategy`): application-level selective
#     offload — only the *live* train state (model param flat buffers + optimizer
#     master params / Adam state) is copied to CPU; gradients and recomputable
#     activations are just released, and ``empty_cache()`` returns the freed pages
#     to the driver. Pure PyTorch storage ops, hardware-neutral, and the only safe
#     option when TMS is unavailable (e.g. when expandable_segments is enabled).
#
# The public :class:`MegatronTrainStateOffloader` hides the choice behind a
# uniform ``offload()`` / ``reload()`` / ``disable_during_update()`` interface, so
# the actor's ``sleep()`` / ``wake_up()`` / ``update_weights()`` never touch TMS
# nor the Megatron buffer internals directly.
#
# The tensor-storage manipulation mirrors Megatron-LM / verl
# (``verl/utils/megatron_utils.py``): typed ``.storage()`` API plus a ``cpu_data``
# attribute attached to each flat buffer.

import os
from contextlib import nullcontext
from typing import ContextManager, Iterable, Iterator, List, Optional, Protocol

import torch

from relax.utils import device as device_utils
from relax.utils.device import device_module
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# Host-side backup tensors may use pinned memory / async copies only where the backend
# supports it — on Kunlunxin both segfault. Mirrors relax/utils/training/tensor_backper.py.
_PIN_MEMORY = device_utils.use_pinned_host_memory()
_NON_BLOCKING = device_utils.use_non_blocking_copy()


# MoE token-dispatcher cache attributes to drop before offload. Each MoE layer's
# DeepEP dispatcher / _DeepepManager stashes these on itself during the forward and
# never resets them after combine(). ``token_probs`` (torch.topk -> TopkBackward0)
# and ``dispatched_probs`` carry grad_fn, so they keep the whole MoE forward autograd
# graph — and its per-layer hidden-state activations — resident on GPU. They are
# rebuilt on the next forward, so clearing before offload is safe and reclaims that
# memory. Selective offload never touched these; TMS's whole-pool pause happened to hide
# them, which is why selective offload showed a larger residual on MoE models.
_MOE_DISPATCHER_CACHE_ATTRS = (
    "token_probs",
    "token_indices",
    "dispatched_probs",
    "dispatched_indices",
    "dispatched_routing_map",
    "dispatched_input",
    "handle",
    "tokens_per_expert",
    "routing_map",
    "probs",
    "global_local_map",
    "input_splits",
    "output_splits",
    "output_splits_tp",
    "hidden_shape",
    "hidden_shape_before_permute",
    "reversed_local_input_permutation_mapping",
    "reversed_global_input_permutation_mapping",
    "global_input_tokens_local_experts_indices",
    "local_input_tokens_global_experts_indices",
    "reversed_mapping_for_combine",
    "pad_offsets",
    "num_global_tokens_per_local_expert",
    "local_map",
    "local_probs",
)

# A MoE layer may keep several dispatcher handles: the active one, plus a saved
# training dispatcher and an inference dispatcher when it toggles modes for rollout
# (see MoELayer._saved_token_dispatcher / _inference_token_dispatcher). The real
# training-activation cache can sit on any of them, so clear all.
_MOE_DISPATCHER_ATTRS = ("token_dispatcher", "_saved_token_dispatcher", "_inference_token_dispatcher")


def torch_memory_saver_preloaded() -> bool:
    """Return True when the ``torch_memory_saver`` hook ``.so`` is
    LD_PRELOAD'ed.

    Only in that case may we touch ``torch_memory_saver`` — otherwise even
    reading ``memory_margin_bytes`` triggers ``_ensure_initialized()`` ->
    AssertionError ("invalid LD_PRELOAD"). When it is not preloaded we fall
    back to the selective offload, which needs no ``.so``.
    """
    return "torch_memory_saver" in os.environ.get("LD_PRELOAD", "")


def warm_cpu_gradient_allocator(optimizer) -> None:
    """Prepare native HDO grad allocations without binding grads or state."""
    if not _PIN_MEMORY:
        return
    from megatron.core.optimizer import ChainedOptimizer
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    wrappers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    # Hold the whole batch until return so equal-sized parameters get distinct
    # cached allocations. Native HDO still creates and binds the real gradients.
    buffers = [
        torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=True)
        for wrapper in wrappers
        if isinstance(inner := getattr(wrapper, "optimizer", None), HybridDeviceOptimizer) and inner.pin_cpu_grads
        for cpu in inner.cpu_optimizers
        for group in cpu.param_groups
        for param in group["params"]
        if param not in inner.cpu_copy_map_grad
    ]
    del buffers


class _OffloadStrategy(Protocol):
    """Structural interface implemented by the TMS and selective offload
    strategies.

    A ``Protocol`` (not an ABC) keeps this composition-over-inheritance: the
    two strategies don't subclass anything, they just satisfy this shape, and
    the facade holds one via delegation.
    """

    def offload(self) -> None: ...

    def reload(self) -> None: ...

    def disable_during_update(self) -> ContextManager: ...


class _TmsOffloadStrategy:
    """Offload via ``torch_memory_saver`` VMM ``pause()`` / ``resume()``."""

    def offload(self) -> None:
        from torch_memory_saver import torch_memory_saver

        torch_memory_saver.pause()

    def reload(self) -> None:
        from torch_memory_saver import torch_memory_saver

        torch_memory_saver.resume()

    def disable_during_update(self) -> ContextManager:
        # Weight update allocates temp buffers that must NOT be tracked by TMS
        # (they'd be double-counted / paused on the next sleep).
        from torch_memory_saver import torch_memory_saver

        return torch_memory_saver.disable()


class _SelectiveOffloadStrategy:
    """Selective, application-level CPU offload of the live Megatron train
    state."""

    def __init__(self, model: Optional[List], optimizer, args) -> None:
        self.model = model
        self.optimizer = optimizer
        # When Megatron's --optimizer-cpu-offload is active, HybridDeviceOptimizer
        # manages its own CPU/GPU tensor placement; moving optimizer state ourselves
        # would break its invariants (_fused_adam device mismatch).
        self._skip_optimizer: bool = bool(getattr(args, "optimizer_cpu_offload", False))

    def _iter_optimizers(self) -> Iterator:
        """Yield every Megatron DistributedOptimizer (unwrapping
        ChainedOptimizer)."""
        from megatron.core.optimizer import ChainedOptimizer

        opt = self.optimizer
        if isinstance(opt, ChainedOptimizer):
            yield from opt.chained_optimizers
        else:
            yield opt

    def _iter_ddp_buffers(self) -> Iterator:
        """Yield every DDP flat buffer (params + expert-parallel) across model
        chunks."""
        if self.model is None:
            return
        for ddp_model in self.model:
            for buffers in [
                getattr(ddp_model, "buffers", []) or [],
                getattr(ddp_model, "expert_parallel_buffers", []) or [],
            ]:
                for buffer in buffers:
                    yield buffer

    @staticmethod
    def _param_index_map(buffer) -> dict:
        return getattr(buffer, "param_index_map", None) or getattr(buffer, "param_to_index", {})

    @torch.no_grad()
    def _clear_moe_dispatcher_caches(self) -> None:
        """Drop MoE token-dispatcher tensor caches so the pinned forward graph
        frees.

        See :data:`_MOE_DISPATCHER_CACHE_ATTRS`. No-op for non-MoE models (the
        ``token_dispatcher`` attribute is absent, so nothing is touched).
        """
        if self.model is None:
            return

        def _clear_on(obj) -> None:
            if obj is None:
                return
            for attr in _MOE_DISPATCHER_CACHE_ATTRS:
                if getattr(obj, attr, None) is not None:
                    setattr(obj, attr, None)

        for ddp_model in self.model:
            modules = ddp_model.modules() if hasattr(ddp_model, "modules") else []
            for module in modules:
                for name in _MOE_DISPATCHER_ATTRS:
                    dispatcher = getattr(module, name, None)
                    if dispatcher is None:
                        continue
                    # Cache lives on the dispatcher and on its DeepEP/HybridEP comm manager.
                    _clear_on(dispatcher)
                    _clear_on(getattr(dispatcher, "_comm_manager", None))

    def _offload_training_state(self, *, retain_parameters: bool = False) -> None:
        """Offload each buffer before the optimizer, optionally retaining GPU
        parameters for sync."""
        for buffer in self._iter_ddp_buffers():
            if not retain_parameters:
                self._copy_parameters_to_cpu((buffer,), non_blocking=False)
                self._release_parameter_storage((buffer,))
            if buffer.grad_data is not None and buffer.grad_data.storage().size() > 0:
                buffer.grad_data_size = buffer.grad_data.storage().size()
                buffer.grad_data.storage().resize_(0)

        if not self._skip_optimizer:
            # 2. Optimizer fp32 master params (shard_fp32_from_float16_groups).
            for opt in self._iter_optimizers():
                for group in getattr(opt, "shard_fp32_from_float16_groups", []) or []:
                    if isinstance(group, list):
                        for p in group:
                            if p is not None and p.device.type != "cpu":
                                p.data = p.data.to("cpu", non_blocking=False)
                    elif group is not None and group.device.type != "cpu":
                        group.data = group.data.to("cpu", non_blocking=False)

            # 3. Optimizer Adam state (exp_avg, exp_avg_sq) — lazily allocated, not recomputable.
            for opt in self._iter_optimizers():
                if getattr(opt, "optimizer", None) is None:
                    continue
                for state in opt.optimizer.state.values():
                    if "exp_avg" in state:
                        state["exp_avg"] = state["exp_avg"].to("cpu", non_blocking=False)
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to("cpu", non_blocking=False)

    def can_release_initial_gradients(self, weights: dict[str, torch.Tensor]) -> bool:
        """Qualify gradient-only release before the initial native export."""
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

        if (
            not self.model
            or not self._skip_optimizer
            or not all(
                isinstance(getattr(wrapper, "optimizer", None), HybridDeviceOptimizer)
                for wrapper in self._iter_optimizers()
            )
        ):
            return False
        for model in self.model:
            config = model.ddp_config
            if not config.use_distributed_optimizer or getattr(config, "reuse_grad_buf_for_mxfp8_param_ag", False):
                return False
            for group in (*model.bucket_groups, *model.expert_parallel_bucket_groups):
                if group.param_gather_handle is not None or group.grad_reduce_handle is not None:
                    return False

        buffers = list(self._iter_ddp_buffers())
        if not buffers or any(
            buffer.param_data is None
            or buffer.param_data.dtype != torch.bfloat16
            or not buffer.param_data.is_cuda
            or not buffer.param_data.untyped_storage().nbytes()
            for buffer in buffers
        ):
            return False
        grads = [buffer.grad_data.untyped_storage() for buffer in buffers if buffer.grad_data is not None]
        grads = [storage for storage in grads if storage.nbytes()]
        if not grads or any(storage.device.type != "cuda" for storage in grads):
            return False
        grad_ids = {storage._cdata for storage in grads}
        ranges = [(storage.device, storage.data_ptr(), storage.data_ptr() + storage.nbytes()) for storage in grads]
        for tensor in (*weights.values(), *(buffer.param_data for buffer in buffers)):
            storage = tensor.untyped_storage()
            start = storage.data_ptr()
            end = start + storage.nbytes()
            if storage._cdata in grad_ids or any(
                storage.device == device and start < right and left < end for device, left, right in ranges
            ):
                return False
        return True

    def validate_optimizer_storage_aliases(self) -> str | None:
        """Check storage aliases before retaining weights and releasing
        gradients.

        Inspect only tensors moved by ordinary offload. Optimizer configuration
        and state handling remain the responsibility of the original path.
        """
        if self._skip_optimizer:
            return None

        model_storages = {
            (tensor.device, tensor.untyped_storage().data_ptr())
            for b in self._iter_ddp_buffers()
            for tensor in (b.param_data, b.grad_data)
            if tensor is not None and tensor.untyped_storage().nbytes()
        }
        for opt in self._iter_optimizers():
            tensors = [
                p
                for group in getattr(opt, "shard_fp32_from_float16_groups", []) or []
                for p in (group if isinstance(group, list) else [group])
                if p is not None
            ]
            inner = getattr(opt, "optimizer", None)
            if inner is not None:
                for state in inner.state.values():
                    tensors.extend(state[name] for name in ("exp_avg", "exp_avg_sq") if name in state)
            if any(
                isinstance(tensor, torch.Tensor)
                and (tensor.device, tensor.untyped_storage().data_ptr()) in model_storages
                for tensor in tensors
            ):
                return "optimizer state aliases model parameters/gradients"
        return None

    def _copy_parameters_to_cpu(self, buffers: Iterable, *, non_blocking: bool) -> None:
        """Copy DDP parameter buffers to reusable CPU storage."""
        for buffer in buffers:
            param = buffer.param_data
            if param is None or param.storage().size() == 0:
                continue
            cpu = getattr(param, "cpu_data", None)
            if cpu is None:
                cpu = torch.empty_like(param, device="cpu", pin_memory=_PIN_MEMORY)
                param.cpu_data = cpu
            if cpu.shape != param.shape or cpu.dtype != param.dtype:
                raise RuntimeError("Train-state CPU backing no longer matches the parameter buffer")
            cpu.copy_(param, non_blocking=non_blocking)
            buffer.param_data_size = param.storage().size()

    def _release_parameter_storage(self, buffers: Iterable) -> None:
        """Expose completed CPU copies, then free the DDP buffers' GPU
        storage."""
        for buffer in buffers:
            param = buffer.param_data
            if param is None or param.storage().size() == 0:
                continue
            for tensor, (start, end, *_rest) in self._param_index_map(buffer).items():
                if end - start == tensor.numel():
                    tensor._relax_cpu_offload_data = param.cpu_data[start:end].view(tensor.shape)
            param.storage().resize_(0)

    @torch.no_grad()
    def offload(self) -> None:
        import gc

        if getattr(self, "_actor_weights_pending_release", False):
            raise RuntimeError("A source weight transfer still owns the training parameters")
        self._clear_moe_dispatcher_caches()
        self._offload_training_state()
        gc.collect()
        device_module.empty_cache()

    @torch.no_grad()
    def initial_weight_sync_supported(self, weights: dict, snapshot: dict) -> bool:
        """Qualify a sleeping source before restoring any initial GPU bytes."""
        buffers = list(self._iter_ddp_buffers())
        if (
            getattr(self, "_actor_weights_pending_release", False)
            or not buffers
            or weights.keys() != snapshot.keys()
            or any(
                b.param_data is None
                or b.param_data.device.type != "cuda"
                or b.param_data.storage().size() != 0
                or not getattr(b, "param_data_size", 0)
                or not hasattr(b.param_data, "cpu_data")
                or b.param_data.cpu_data.device.type != "cpu"
                or b.param_data.cpu_data.shape != b.param_data.shape
                or b.param_data.cpu_data.dtype != b.param_data.dtype
                or b.grad_data is not None
                and b.grad_data.storage().size() != 0
                for b in buffers
            )
        ):
            return False
        parameter_storages = {b.param_data.untyped_storage()._cdata for b in buffers}
        if any(
            b.grad_data is not None and b.grad_data.untyped_storage()._cdata in parameter_storages for b in buffers
        ):
            return False
        backing = {b.param_data.untyped_storage()._cdata: b.param_data for b in buffers}
        for name, tensor in weights.items():
            cpu = snapshot[name]
            if (
                tensor.device.type != "cuda"
                or cpu.device.type != "cpu"
                or (tensor.shape, tensor.dtype)
                != (
                    cpu.shape,
                    cpu.dtype,
                )
            ):
                return False
            flat = backing.get(tensor.untyped_storage()._cdata)
            if flat is None:
                if tensor.untyped_storage().nbytes() == 0:
                    return False
            else:
                view = getattr(tensor, "_relax_cpu_offload_data", None)
                if (
                    view is None
                    or view.untyped_storage()._cdata != flat.cpu_data.untyped_storage()._cdata
                    or view.storage_offset() != tensor.storage_offset() - flat.storage_offset()
                    or (view.shape, view.stride(), view.dtype) != (tensor.shape, tensor.stride(), tensor.dtype)
                ):
                    return False
        return True

    @torch.no_grad()
    def _begin_initial_weight_sync(self, weights: dict, snapshot: dict) -> None:
        # The caller proves version 0 and no intervening training/model switch.
        # Reuse the completed initial CPU snapshot; there is no second D2H.
        if not self.initial_weight_sync_supported(weights, snapshot):
            raise RuntimeError("Initial source allocations no longer match the CPU snapshot")
        buffers = list(self._iter_ddp_buffers())
        backing = {b.param_data.untyped_storage()._cdata: b.param_data for b in buffers}
        cpu_weights = {
            name: tensor._relax_cpu_offload_data if tensor.untyped_storage()._cdata in backing else snapshot[name]
            for name, tensor in weights.items()
        }
        try:
            for buffer in buffers:
                buffer.param_data.storage().resize_(buffer.param_data_size)
                buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=_NON_BLOCKING)
            for name, tensor in weights.items():
                if tensor.untyped_storage()._cdata not in backing:
                    tensor.copy_(snapshot[name], non_blocking=_NON_BLOCKING)
            done = torch.cuda.Event()
            done.record()
            done.synchronize()
        except BaseException:
            # No receiver has been seeded or submitted yet. Drain H2D before
            # returning the flat allocations to sleep; never swallow the error.
            device_module.synchronize()
            self._release_parameter_storage(buffers)
            raise
        for buffer in buffers:
            for tensor in self._param_index_map(buffer):
                if hasattr(tensor, "_relax_cpu_offload_data"):
                    del tensor._relax_cpu_offload_data
        self._cpu_backup_done = done
        self._cpu_weight_snapshot = cpu_weights
        self._actor_weights_pending_release = True

    @torch.no_grad()
    def begin_weight_sync(
        self, weights: dict[str, torch.Tensor], *, overlap: bool, initial_snapshot: dict | None = None
    ) -> torch.cuda.Event:
        """Back up Actor weights to CPU while keeping their GPU storage
        readable.

        The caller must first wait for optimizer and parameter-gather writes.
        This method releases gradients and offloads optimizer state, but does
        not send weights. With overlap=False it waits for the CPU backup before
        returning. In both modes, keep GPU weights unchanged until the caller
        receives every receiver ACK and calls finish_weight_sync.
        """
        if initial_snapshot is not None:
            self._begin_initial_weight_sync(weights, initial_snapshot)
            return self._cpu_backup_done
        if getattr(self, "_actor_weights_pending_release", False):
            raise RuntimeError("A source weight transfer is already pending")
        buffers = list(self._iter_ddp_buffers())
        if not buffers or any(b.param_data is None or b.param_data.storage().size() == 0 for b in buffers):
            raise RuntimeError("Source weight sync requires resident DDP parameter allocations")
        parameter_storages = {b.param_data.untyped_storage().data_ptr() for b in buffers}
        if any(
            b.grad_data is not None and b.grad_data.untyped_storage().data_ptr() in parameter_storages for b in buffers
        ):
            raise RuntimeError("Source weight sync cannot release gradients that alias model parameters")
        if any(t.device.type != "cuda" for t in weights.values()):
            raise RuntimeError("Source weight sync requires all exported parameters/buffers on CUDA")
        reason = self.validate_optimizer_storage_aliases()
        if reason is not None:
            raise RuntimeError(f"Optimizer state can no longer be offloaded safely: {reason}")
        self._clear_moe_dispatcher_caches()
        # TODO(agent): overlap optimizer-state D2H with weight sync once pinned
        # buffers, completion events, storage aliases and memory headroom are
        # qualified. For now finish its blocking offload before sending weights.
        self._offload_training_state(retain_parameters=True)
        device_module.empty_cache()
        self._actor_weights_pending_release = True
        self._cpu_backup_stream = torch.cuda.Stream()
        self._cpu_backup_stream.wait_stream(torch.cuda.current_stream())
        snapshot = {}
        extras = getattr(self, "_standalone_cpu_weights", {})
        with torch.cuda.stream(self._cpu_backup_stream):
            self._copy_parameters_to_cpu(buffers, non_blocking=True)
            by_storage = {b.param_data.untyped_storage().data_ptr(): b.param_data for b in buffers}
            for name, tensor in weights.items():
                flat = by_storage.get(tensor.untyped_storage().data_ptr())
                if flat is not None:
                    if tensor.dtype != flat.dtype:
                        raise RuntimeError("A parameter has a different dtype from its backing allocation")
                    offset = tensor.storage_offset() - flat.storage_offset()
                    snapshot[name] = flat.cpu_data.as_strided(tensor.shape, tensor.stride(), offset)
                else:
                    cpu = extras.get(name)
                    if cpu is None:
                        cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                        extras[name] = cpu
                    if cpu.shape != tensor.shape or cpu.dtype != tensor.dtype:
                        raise RuntimeError("Standalone parameter layout changed during source weight sync")
                    cpu.copy_(tensor.detach(), non_blocking=True)
                    snapshot[name] = cpu
            self._cpu_backup_done = torch.cuda.Event()
            self._cpu_backup_done.record()
        self._standalone_cpu_weights = extras
        self._cpu_weight_snapshot = snapshot
        if not overlap:
            self._cpu_backup_done.synchronize()
        return self._cpu_backup_done

    @torch.no_grad()
    def finish_weight_sync(self) -> dict[str, torch.Tensor]:
        """Wait for CPU backup, free GPU parameter storage, and return CPU
        views.

        The caller must already have all receiver ACKs, including completion of
        CUDA reads and closure of IPC mappings. This method does not wait for
        receivers itself.
        """
        if not getattr(self, "_actor_weights_pending_release", False):
            raise RuntimeError("No source weight transfer is pending")
        self._cpu_backup_done.synchronize()
        self._release_parameter_storage(self._iter_ddp_buffers())
        self._actor_weights_pending_release = False
        snapshot, self._cpu_weight_snapshot = self._cpu_weight_snapshot, None
        device_module.empty_cache()
        return snapshot

    @torch.no_grad()
    def reload(self) -> None:
        import gc

        if getattr(self, "_actor_weights_pending_release", False):
            raise RuntimeError("Cannot reload while a source weight transfer is pending")
        device = device_utils.make_current_torch_device()
        # 1. Model DDP flat buffers.
        for buffer in self._iter_ddp_buffers():
            # grad_data: reallocate and zero (contents were discarded on offload).
            if (
                buffer.grad_data is not None
                and hasattr(buffer, "grad_data_size")
                and buffer.grad_data.storage().size() == 0
            ):
                buffer.grad_data.storage().resize_(buffer.grad_data_size)
                buffer.grad_data.zero_()
                del buffer.grad_data_size
            # param_data: reallocate and copy back from the CPU backup.
            if (
                buffer.param_data is not None
                and hasattr(buffer, "param_data_size")
                and buffer.param_data.storage().size() == 0
            ):
                buffer.param_data.storage().resize_(buffer.param_data_size)
                buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=_NON_BLOCKING)
                for param in self._param_index_map(buffer):
                    if hasattr(param, "_relax_cpu_offload_data"):
                        del param._relax_cpu_offload_data
                # param_data_size and the pinned cpu_data deliberately outlive this call:
                # the next offload() reuses the same host buffer (see offload()).
        if _NON_BLOCKING:
            # The H2D copies above are async; make them observable to non-stream code
            # (and to the optimizer-state moves below) before returning.
            device_module.synchronize()

        if not self._skip_optimizer:
            # 2. Optimizer fp32 master params.
            for opt in self._iter_optimizers():
                for group in getattr(opt, "shard_fp32_from_float16_groups", []) or []:
                    if isinstance(group, list):
                        for p in group:
                            if p is not None and p.device.type == "cpu":
                                p.data = p.data.to(device, non_blocking=False)
                    elif group is not None and group.device.type == "cpu":
                        group.data = group.data.to(device, non_blocking=False)

            # 3. Optimizer Adam state.
            for opt in self._iter_optimizers():
                if getattr(opt, "optimizer", None) is None:
                    continue
                for state in opt.optimizer.state.values():
                    if "exp_avg" in state:
                        state["exp_avg"] = state["exp_avg"].to(device, non_blocking=False)
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(device, non_blocking=False)

        gc.collect()
        device_module.empty_cache()

    def disable_during_update(self) -> ContextManager:
        # Selective offload doesn't hook the allocator, so nothing to disable.
        #
        # This is also deliberately NOT the place to toggle expandable_segments. Enabling
        # it is supported on this path (no torch_memory_saver clash) and does help
        # offload()'s resize_(0) + empty_cache() hand whole segments back to the driver,
        # but it belongs to the user via --train-env-vars, not as a hidden side effect of
        # --selective-offload. The IPC bucket allocator separately scopes ordinary
        # allocation to transport buffers, restoring the training configuration before
        # conversion and communication continue.
        return nullcontext()


class MegatronTrainStateOffloader:
    """Facade over the TMS and selective offload strategies for colocate
    sleep/wake.

    Strategy selection:
      * Default: ``torch_memory_saver`` (when its hook ``.so`` is LD_PRELOAD'ed and
        the allocator supports it).
      * ``--selective-offload``: force the application-level selective offload,
        regardless of TMS availability.
      * Safety fallback: if TMS is requested (default) but not preloaded/usable,
        fall back to the selective path with a warning.

    When ``offload_train`` is disabled :attr:`enabled` is ``False`` and every method
    is a no-op.
    """

    def __init__(self, model: Optional[List], optimizer, args) -> None:
        self.enabled: bool = bool(getattr(args, "offload_train", False))
        self.uses_tms: bool = False

        # Default is torch_memory_saver; --selective-offload forces the selective path.
        force_selective = bool(getattr(args, "selective_offload", False))
        if self.enabled and not force_selective:
            if torch_memory_saver_preloaded():
                self.uses_tms = self._init_torch_memory_saver(args)
            else:
                logger.warning(
                    "torch_memory_saver hook is not LD_PRELOAD'ed; falling back to selective "
                    "offload. Pass --selective-offload to select it explicitly and silence this warning."
                )

        if self.enabled and not self.uses_tms and getattr(args, "optimizer", "adam") == "sgd":
            raise ValueError(
                "SGD does not support selective train-state offload; use a working "
                "torch_memory_saver configuration or disable train-state offload."
            )

        self._strategy: _OffloadStrategy = (
            _TmsOffloadStrategy() if self.uses_tms else _SelectiveOffloadStrategy(model, optimizer, args)
        )

        if self.enabled:
            logger.info("Train-state offload enabled (strategy=%s)", "tms" if self.uses_tms else "selective")

    @staticmethod
    def _init_torch_memory_saver(args) -> bool:
        """Configure torch_memory_saver; return True iff it is usable for
        offload."""
        from torch_memory_saver import torch_memory_saver

        margin = max(int(getattr(args, "train_memory_margin_bytes", 0)), 0)
        try:
            torch_memory_saver.memory_margin_bytes = margin
            if margin > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {margin}")
            return True
        except (RuntimeError, NotImplementedError, AssertionError) as e:
            logger.warning(
                f"torch_memory_saver is unavailable in the current allocator mode ({e}); "
                "falling back to selective offload."
            )
            return False

    def offload(self) -> None:
        """Offload the train state off the device (sleep).

        No-op when disabled.
        """
        if self.enabled:
            self._strategy.offload()

    def reload(self) -> None:
        """Reload the train state back onto the device (wake_up).

        No-op when disabled.
        """
        if self.enabled:
            self._strategy.reload()

    def initial_weight_sync_supported(self, weights: dict, snapshot: dict) -> bool:
        return self.enabled and not self.uses_tms and self._strategy.initial_weight_sync_supported(weights, snapshot)

    def can_release_initial_gradients(self, weights: dict[str, torch.Tensor]) -> bool:
        if not self.enabled or self.uses_tms:
            return False
        try:
            return self._strategy.can_release_initial_gradients(weights)
        except (AttributeError, NotImplementedError, RuntimeError, TypeError):
            # A different DDP/optimizer layout keeps the ordinary sleep path.
            return False

    def release_initial_gradients(self) -> None:
        """Run after all ranks qualified the storage and before the initial
        export."""
        torch.cuda.synchronize()
        self._strategy._offload_training_state(retain_parameters=True)
        device_module.empty_cache()

    def abort_initial_weight_sync(self) -> None:
        """Undo initial H2D only before any receiver seed or source reader."""
        if self.enabled and not self.uses_tms and getattr(self._strategy, "_actor_weights_pending_release", False):
            self._strategy.finish_weight_sync()

    def begin_weight_sync(
        self, weights: dict[str, torch.Tensor], *, overlap: bool, initial_snapshot: dict | None = None
    ) -> torch.cuda.Event:
        """Prepare the CPU backup and retain Actor GPU weights for the
        sender."""
        if not self.enabled or self.uses_tms:
            raise RuntimeError("Source weight sync requires selective train-state offload")
        return self._strategy.begin_weight_sync(weights, overlap=overlap, initial_snapshot=initial_snapshot)

    def validate_actor_weight_retention(self) -> str | None:
        """Check selective offload, CUDA and optimizer support without changing
        state."""
        if not self.enabled or self.uses_tms:
            return "requires selective train-state offload"
        if not torch.cuda.is_available() or torch.version.hip is not None:
            return "source weight transfer requires CUDA"
        return self._strategy.validate_optimizer_storage_aliases()

    def finish_weight_sync(self) -> dict[str, torch.Tensor]:
        """Release Actor GPU weights after receiver ACKs; return the CPU
        backup."""
        if not self.enabled or self.uses_tms:
            raise RuntimeError("Source weight sync requires selective train-state offload")
        return self._strategy.finish_weight_sync()

    def disable_during_update(self) -> ContextManager:
        """Context manager wrapping weight update so offload bookkeeping stays
        consistent."""
        if not self.enabled:
            return nullcontext()
        return self._strategy.disable_during_update()
