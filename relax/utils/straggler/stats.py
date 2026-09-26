# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixed layout of the per-rank straggler statistics vector.

Every training rank accumulates one ``WindowStats`` over ``report_interval``
steps and ships it as a flat ``float64[NUM_FIELDS]`` vector, so the cross-rank
gather is a single small collective regardless of model or parallelism.
"""

from __future__ import annotations

from typing import Mapping


# Compute-stream time (ms, summed over the window) per coarse segment. The
# first block is fed by Megatron's own ``config.timers`` call sites (see
# MEGATRON_TIMER_SEGMENTS); the ``lp_*`` block is the same call sites during
# ``forward_only`` (log-prob passes). Adding a segment (e.g. for MoE dispatch
# hooks) is one entry here plus one mapping below.
GPU_SEGMENTS: tuple[str, ...] = (
    "fwd",
    "bwd",
    "optim",
    "pp_recv",
    "pp_send",
    "dp_grad_sync",
    "dp_param_gather",
    "lp_fwd",
    "lp_pp_recv",
    "lp_pp_send",
)
# CPU wall time (ms) of the same fwd / bwd brackets, and Python GC pauses that
# happened while a step was in flight.
CPU_FIELDS: tuple[str, ...] = ("cpu_fwd", "cpu_bwd", "gc")
# Counters: tokens processed by this rank, number of fwd brackets, number of
# steps in the window, event pairs dropped because the pending queue was full,
# and the collector's own CPU cost (ms) spent draining events.
COUNT_FIELDS: tuple[str, ...] = ("tokens", "num_fwd", "num_steps", "dropped", "overhead")

FIELDS: tuple[str, ...] = GPU_SEGMENTS + CPU_FIELDS + COUNT_FIELDS
FIELD_INDEX: Mapping[str, int] = {name: index for index, name in enumerate(FIELDS)}
NUM_FIELDS: int = len(FIELDS)

# Segments whose sum is "time this rank spent computing on its own" versus
# "time this rank spent waiting for a peer". Used by the detector.
SELF_SEGMENTS: tuple[str, ...] = ("fwd", "bwd", "optim")
WAIT_SEGMENTS: tuple[str, ...] = ("pp_recv", "dp_grad_sync", "dp_param_gather")
# Segment whose per-rank asymmetry inside a grad-sync group exposes late arrival.
LATE_ARRIVAL_SEGMENT = "dp_grad_sync"

# Megatron timer name -> segment, for the training phase. Names come from
# megatron/core/pipeline_parallel/schedules.py, p2p_communication.py,
# distributed/finalize_model_grads.py, optimizer/optimizer.py and
# optimizer/distrib_optimizer.py. ``forward-backward`` wraps a whole step and is
# deliberately absent (it duplicates Relax's ``actor_train`` timer).
MEGATRON_TIMER_SEGMENTS: Mapping[str, str] = {
    "forward-compute": "fwd",
    "backward-compute": "bwd",
    "forward-recv": "pp_recv",
    "backward-recv": "pp_recv",
    "forward-send-backward-recv": "pp_recv",
    "backward-send-forward-recv": "pp_recv",
    "forward-send-forward-recv": "pp_recv",
    "backward-send-backward-recv": "pp_recv",
    "forward-backward-send-forward-backward-recv": "pp_recv",
    "forward-send": "pp_send",
    "backward-send": "pp_send",
    "all-grads-sync": "dp_grad_sync",
    "embedding-grads-all-reduce": "dp_grad_sync",
    "non-tensor-parallel-grads-all-reduce": "dp_grad_sync",
    "conditional-embedder-grads-all-reduce": "dp_grad_sync",
    "params-all-gather": "dp_param_gather",
    "optimizer-copy-to-main-grad": "optim",
    "optimizer-unscale-and-check-inf": "optim",
    "optimizer-clip-main-grad": "optim",
    "optimizer-count-zeros": "optim",
    "optimizer-inner-step": "optim",
    "optimizer-copy-main-to-model-params": "optim",
}

# Same call sites during ``forward_only`` land in their own buckets so a slow
# log-prob pass does not pollute the training-step numbers.
FORWARD_ONLY_TIMER_SEGMENTS: Mapping[str, str] = {
    name: {"fwd": "lp_fwd", "pp_recv": "lp_pp_recv", "pp_send": "lp_pp_send"}.get(segment, segment)
    for name, segment in MEGATRON_TIMER_SEGMENTS.items()
    if segment in ("fwd", "pp_recv", "pp_send")
}


class WindowStats:
    """Accumulate one rank's window in the ``FIELDS`` layout.

    ``add`` resolves the field name; hot paths that run per event use
    ``add_index`` with an index looked up once at import time.
    """

    __slots__ = ("values",)

    def __init__(self) -> None:
        self.values: list[float] = [0.0] * NUM_FIELDS

    def add(self, field: str, value: float) -> None:
        self.values[FIELD_INDEX[field]] += value

    def add_index(self, index: int, value: float) -> None:
        self.values[index] += value

    def get(self, field: str) -> float:
        return self.values[FIELD_INDEX[field]]

    def reset(self) -> None:
        self.values = [0.0] * NUM_FIELDS

    def as_list(self) -> list[float]:
        return list(self.values)
