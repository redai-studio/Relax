# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey-patch SGLang's ``_RoutedExpertsCapturerReal`` for fully async Device-
to-Host copy of routed-experts buffers.

Problem
-------
The original ``_sync_fwd_experts_buffer_DtoH`` performs two synchronous
``.cpu()`` calls after every forward pass.  The first call — on
``out_cache_loc`` — implicitly synchronises the *entire* default CUDA
stream, blocking the CPU thread until the forward pass completes and
destroying the overlap scheduler's CPU–GPU pipeline.  The second call
copies the routing-expert data itself.

Together they add ~20 % overhead to rollout latency.

Solution
--------
**All GPU→CPU copies are fully asynchronous; no stream sync occurs in
``on_forward_end``.**

1. GPU→GPU snapshot of ``device_cache.buffer`` → *staging buffer*
   (default stream).  Staging is never overwritten by ``capture()``.
2. On a dedicated *copy stream*:

   a. ``wait_stream(default)`` to order after step 1.
   b. ``non_blocking`` copy of the staging slice → *pinned staging*
      (CPU, pinned memory).
   c. ``non_blocking`` copy of ``out_cache_loc`` → *pinned loc buffer*
      (CPU, pinned memory).
   d. Record a CUDA event.

3. **No synchronisation on the CPU thread.**  The function returns
   immediately and the CPU can proceed to schedule the next batch.
4. Before reading ``host_cache.buffer`` (in ``get_routed_experts``
   or the next ``_sync_fwd_experts_buffer_DtoH``), the event is
   synchronised and the pinned buffers are scattered into
   ``host_cache.buffer``.

Extra GPU memory: ~12 MB staging buffer (Qwen3-30B-A3B config).
Extra CPU pinned memory: ~12 MB staging + ~8 KB loc buffer.

Injection
---------
This module is imported inside the SGLang *scheduler subprocess*
(spawned via ``multiprocessing.Process`` with ``spawn`` start method).
``apply_patch`` replaces methods on ``_RoutedExpertsCapturerReal``
*before* the capturer is instantiated by ``ModelRunner``.
"""

import logging

import torch

from relax.utils.device import device_module


logger = logging.getLogger(__name__)


def apply_patch():
    """Monkey-patch ``_RoutedExpertsCapturerReal`` for fully async D→H copy.

    Must be called inside the SGLang scheduler subprocess, before the model
    runner initialises the routed-experts capturer.
    """
    from sglang.srt.layers.moe.routed_experts_capturer import (
        _RoutedExpertsCapturerReal,
    )

    _orig_init = _RoutedExpertsCapturerReal.__init__
    _orig_get_routed_experts = _RoutedExpertsCapturerReal.get_routed_experts

    # ------------------------------------------------------------------ #
    # Patched __init__                                                    #
    # ------------------------------------------------------------------ #
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)

        dev_buf = self.device_cache.buffer
        topk = self.num_experts_per_tok

        # GPU staging buffer — same shape as device_cache.buffer.
        self._staging_buffer = torch.zeros_like(dev_buf)

        # CPU pinned staging for routing data (topk slice only).
        self._pinned_staging = torch.zeros(
            (dev_buf.shape[0], dev_buf.shape[1], topk),
            dtype=dev_buf.dtype,
            device="cpu",
            pin_memory=True,
        )

        # CPU pinned buffer for out_cache_loc indices.
        # max_running_requests is a safe upper bound for batch size.
        max_batch = dev_buf.shape[0]
        self._pinned_loc = torch.zeros(max_batch, dtype=torch.int64, device="cpu", pin_memory=True)

        # Dedicated copy stream + event.
        self._copy_stream = device_module.Stream(device=dev_buf.device)
        self._copy_event = device_module.Event()

        # Pending scatter state.
        self._pending_n = 0  # 0 means nothing pending

        staging_mb = self._staging_buffer.nelement() * self._staging_buffer.element_size() / (1024 * 1024)
        pinned_mb = self._pinned_staging.nelement() * self._pinned_staging.element_size() / (1024 * 1024)
        logger.info(
            "Routing-replay async D→H patch: GPU staging %.2f MB, CPU pinned staging %.2f MB, stream %s",
            staging_mb,
            pinned_mb,
            self._copy_stream,
        )

    # ------------------------------------------------------------------ #
    # Patched _sync_fwd_experts_buffer_DtoH — fully async, zero sync      #
    # ------------------------------------------------------------------ #
    def _patched_sync(self, forward_batch, can_run_graph, cuda_graph_batch):
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            get_dp_local_info,
            is_dp_attention_enabled,
        )
        from sglang.srt.layers.moe import get_moe_a2a_backend

        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = forward_batch.out_cache_loc.shape[0]

        n_tok = local_end_pos - local_start_pos
        topk = self.num_experts_per_tok

        # Flush previous pending scatter so pinned buffers can be reused.
        self._flush_pending_scatter()

        # Capture the active stream BEFORE entering the copy-stream context.
        # In overlap-scheduler mode this is the *forward_stream*; without
        # overlap it is the default stream.  We need this reference so that
        # copy_stream can order itself after the GPU→GPU staging copy below.
        active_stream = device_module.current_stream(self.device_cache.buffer.device)

        # 1) GPU→GPU snapshot on the active stream — fast, no sync.
        self._staging_buffer[:n_tok].copy_(self.device_cache.buffer[local_start_pos:local_end_pos])

        # 2) On copy stream: async copies to pinned CPU buffers.
        #    copy_stream waits on active_stream so the staging snapshot
        #    above completes before we start reading it.
        with device_module.stream_context(self._copy_stream):
            self._copy_stream.wait_stream(active_stream)
            # 2a) Routing data: staging[:n_tok, :, :topk] → pinned_staging
            self._pinned_staging[:n_tok, :, :topk].copy_(self._staging_buffer[:n_tok, :, :topk], non_blocking=True)
            # 2b) Location indices: out_cache_loc → pinned_loc
            self._pinned_loc[:n_tok].copy_(forward_batch.out_cache_loc, non_blocking=True)

        # 3) Record event — no sync, returns immediately.
        self._copy_event.record(self._copy_stream)

        # 4) Mark pending.
        self._pending_n = n_tok

    # ------------------------------------------------------------------ #
    # Flush: event.sync → scatter pinned data into host_cache             #
    # ------------------------------------------------------------------ #
    def _flush_pending_scatter(self):
        if self._pending_n == 0:
            return
        self._copy_event.synchronize()
        n = self._pending_n
        topk = self.num_experts_per_tok
        loc = self._pinned_loc[:n]
        self.host_cache.buffer[loc] = self._pinned_staging[:n, :, :topk]
        self._pending_n = 0

    # ------------------------------------------------------------------ #
    # Patched get_routed_experts: flush first                             #
    # ------------------------------------------------------------------ #
    def _patched_get_routed_experts(self, req_pool_idx, seqlen, req_to_token_pool):
        self._flush_pending_scatter()
        return _orig_get_routed_experts(self, req_pool_idx, seqlen, req_to_token_pool)

    # Apply all patches.
    _RoutedExpertsCapturerReal.__init__ = _patched_init
    _RoutedExpertsCapturerReal._sync_fwd_experts_buffer_DtoH = _patched_sync
    _RoutedExpertsCapturerReal._flush_pending_scatter = _flush_pending_scatter
    _RoutedExpertsCapturerReal.get_routed_experts = _patched_get_routed_experts

    logger.info("Routing-replay async D→H patch applied to _RoutedExpertsCapturerReal")
