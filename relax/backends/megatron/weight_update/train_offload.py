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
from typing import ContextManager, Iterator, List, Optional, Protocol

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

    @torch.no_grad()
    def offload(self) -> None:
        import gc

        # 0. Drop MoE token-dispatcher caches (token_probs/dispatched_probs carry
        # grad_fn and pin the MoE forward graph + its activations on GPU).
        self._clear_moe_dispatcher_caches()

        # 1. Model DDP flat buffers.
        for buffer in self._iter_ddp_buffers():
            # param_data: copy to CPU, then free GPU storage (must survive sleep).
            if buffer.param_data is not None and buffer.param_data.storage().size() > 0:
                # Reuse one persistent pinned host buffer per DDP buffer instead of
                # allocating a fresh pageable one every sleep. Pinned memory lets the
                # driver DMA straight from the device rather than staging through its
                # internal bounce buffer (~2x on D2H, and reload() gets the same on H2D).
                # Reusing it also avoids re-paying the page-locking cost, and avoids the
                # transient 2x host-memory peak that reallocating would cause: Python
                # evaluates the new buffer before dropping the reference to the old one.
                cpu_data = getattr(buffer.param_data, "cpu_data", None)
                if cpu_data is None:
                    cpu_data = torch.empty_like(buffer.param_data, device=torch.device("cpu"), pin_memory=_PIN_MEMORY)
                    buffer.param_data.cpu_data = cpu_data
                else:
                    # A mismatch means something rebuilt param_data behind our back;
                    # surface it rather than silently reallocating (2x peak again).
                    assert cpu_data.shape == buffer.param_data.shape, (
                        f"cpu_data shape {tuple(cpu_data.shape)} != param_data shape {tuple(buffer.param_data.shape)}"
                    )
                    assert cpu_data.dtype == buffer.param_data.dtype, (
                        f"cpu_data dtype {cpu_data.dtype} != param_data dtype {buffer.param_data.dtype}"
                    )
                # Synchronous D2H: the copy must land before resize_(0) frees the source.
                cpu_data.copy_(buffer.param_data.data, non_blocking=False)
                buffer.param_data_size = buffer.param_data.storage().size()
                # Give each contiguous param a CPU-backed view so code that reads
                # param.data while offloaded doesn't hit the resized-to-0 storage.
                # These views alias cpu_data, which the next offload() overwrites — safe
                # only because reload() drops them before returning.
                for param, (start, end, *_rest) in self._param_index_map(buffer).items():
                    if end - start == param.numel():
                        param._relax_cpu_offload_data = cpu_data[start:end].view(param.shape)
                buffer.param_data.storage().resize_(0)
            # grad_data: just free GPU storage (contents are recomputed on wake).
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

        gc.collect()
        device_module.empty_cache()

    @torch.no_grad()
    def reload(self) -> None:
        import gc

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
        # --selective-offload. Scoping it here would not work anyway:
        # _set_allocator_settings only affects newly created segments, so turning it off
        # around a region cannot stop allocations from being served out of expandable
        # blocks that already exist. See the OOM troubleshooting guide for the supported
        # configuration.
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

    def disable_during_update(self) -> ContextManager:
        """Context manager wrapping weight update so offload bookkeeping stays
        consistent."""
        if not self.enabled:
            return nullcontext()
        return self._strategy.disable_during_update()
