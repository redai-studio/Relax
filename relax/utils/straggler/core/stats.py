# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""What the collection hands to the analysis: the fixed layout of the per-rank
statistics vector, each rank's static identity, the records of a sampled step
and the profiler levels.

Every training rank accumulates one ``WindowStats`` over ``report_interval``
steps and ships it as a flat ``float64[NUM_FIELDS]`` vector, so the cross-rank
gather is a single small collective regardless of model or parallelism.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any, Mapping

from relax.utils.straggler.core.health import HEALTH_DISABLE


# Compute-stream time (ms, summed over the window) per coarse segment. The
# first block is fed by the training framework's timer call sites (the Megatron
# adapter maps its timer names onto them); the ``lp_*`` block is the same call
# sites during forward-only passes (log-probs). Adding a segment is one entry
# here plus one in the adapter's mapping.
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
# The training-phase ``pp_recv`` split by the 1F1B phase of each bracket (ms,
# summed over the window); ``pp_recv`` keeps the total. The adapter tells the
# phase from the timer name (see its PP_PHASE_TIMERS).
PP_PHASE_FIELDS: tuple[str, ...] = ("pp_wait_fill", "pp_wait_up", "pp_wait_down", "pp_wait_drain")
# CPU wall time (ms) of the same fwd / bwd brackets, and Python GC pauses that
# happened while a step was in flight.
CPU_FIELDS: tuple[str, ...] = ("cpu_fwd", "cpu_bwd", "gc")
# Counters: tokens processed by this rank, number of fwd brackets, number of
# steps in the window, event pairs dropped because the pending queue was full,
# and the collector's own CPU cost (ms) spent draining events.
COUNT_FIELDS: tuple[str, ...] = ("tokens", "num_fwd", "num_steps", "dropped", "overhead")
# Window-level extremes over single training micro-batches, not summed per
# step: the slowest and fastest fwd / bwd bracket (ms), which forward of its
# step the slowest one was (from 0) and its token count (-1 when the training
# forward_step did not report one), and the largest and smallest token counts.
# A minimum, or a token extreme, of 0 means no sample.
MICRO_BATCH_FIELDS: tuple[str, ...] = (
    "mb_fwd_max",
    "mb_fwd_min",
    "mb_bwd_max",
    "mb_bwd_min",
    "mb_fwd_argmax",
    "mb_fwd_argmax_tokens",
    "mb_tokens_max",
    "mb_tokens_min",
)
# Window-level extremes over single steps, not summed: the largest one-step
# total of each of these segments (ms), which the window's averages hide when
# a rank slowed down for one step or only late in the window.
STEP_PEAK_SEGMENTS: tuple[str, ...] = ("fwd", "bwd", "optim", "pp_recv", "dp_grad_sync")
STEP_PEAK_FIELDS: tuple[str, ...] = tuple(f"{name}_step_max" for name in STEP_PEAK_SEGMENTS)
# Window-level status, not summed per step: event pairs still unread when the
# window closed (their time lands in the next window), exceptions the profiler
# caught on this rank during the window, this rank's health code
# (``health.HEALTH_*``; 2 asks every rank to switch the profiler off), and the
# level the primary rank asks every rank to switch to, plus one (0: no request).
STATUS_FIELDS: tuple[str, ...] = ("unread", "errors", "health", "control")

FIELDS: tuple[str, ...] = (
    GPU_SEGMENTS + PP_PHASE_FIELDS + CPU_FIELDS + COUNT_FIELDS + MICRO_BATCH_FIELDS + STEP_PEAK_FIELDS + STATUS_FIELDS
)
FIELD_INDEX: Mapping[str, int] = {name: index for index, name in enumerate(FIELDS)}
NUM_FIELDS: int = len(FIELDS)


def disabled_row() -> list[float]:
    """The vector a switched-off rank keeps sending: all zeros except the
    ``health`` field, which asks every rank to switch the profiler off.

    Same length and shape as a normal window, so the gather is unchanged; a
    rank that stops recording time must still join it, or the remaining ranks
    block until the group times out.
    """
    values = [0.0] * NUM_FIELDS
    values[FIELD_INDEX["health"]] = float(HEALTH_DISABLE)
    return values


# Segments whose sum is "time this rank spent computing on its own" versus
# "time this rank spent waiting for a peer". Used by ``analysis.detector``.
SELF_SEGMENTS: tuple[str, ...] = ("fwd", "bwd", "optim")
WAIT_SEGMENTS: tuple[str, ...] = ("pp_recv", "dp_grad_sync", "dp_param_gather")

# The phase slot of a ``dp_grad_sync`` bracket says which gradient sync it is:
# its position among the sync names the adapter gives the trace. The
# data-parallel reduction, the only one that ends at the same time on DP peers,
# is 0.
DP_REDUCTION_PHASE = 0


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

    def max_index(self, index: int, value: float) -> bool:
        """Keep the larger value; True when ``value`` is the new maximum."""
        if value > self.values[index]:
            self.values[index] = value
            return True
        return False

    def min_index(self, index: int, value: float) -> None:
        """Keep the smaller value, a stored 0 meaning no sample yet."""
        if self.values[index] == 0.0 or value < self.values[index]:
            self.values[index] = value

    def get(self, field: str) -> float:
        return self.values[FIELD_INDEX[field]]

    def set(self, field: str, value: float) -> None:
        self.values[FIELD_INDEX[field]] = value

    def reset(self) -> None:
        self.values = [0.0] * NUM_FIELDS

    def as_list(self) -> list[float]:
        return list(self.values)


@dataclass(frozen=True)
class RankMeta:
    """Static identity of one training rank."""

    rank: int
    dp: int
    tp: int
    pp: int
    cp: int = 0
    ep: int = 0
    host: str = ""
    device: int = -1
    # The rank that logs metrics and receives the sampled-step traces.
    primary: bool = False

    @property
    def tag(self) -> str:
        return f"rank{self.rank}_dp{self.dp}_tp{self.tp}_pp{self.pp}"


# One bracket of a sampled step: segment, phase slot (see ``EventSink.push``; -1
# if none), micro-batch (-1), tokens (-1), start and end in ms after the rank's
# anchor event.
RECORD_WIDTH = 6


@dataclass
class RankTrace:
    """One rank's sampled step: the wall-clock time of its anchor event, its
    brackets as flat ``RECORD_WIDTH``-float rows, how many brackets of the step
    are missing, and on the primary rank its own CPU timeline events."""

    wall: float
    records: array = field(default_factory=lambda: array("f"))
    lost: int = 0
    cpu_events: list[dict[str, Any]] = field(default_factory=list)


# The levels an operator can switch the profiler to while training runs: 0
# paused (no brackets recorded, windows not analyzed), 1 coarse (the per-window
# statistics), 2 detail (plus the sampled-step trace), 3 modules (plus the
# forward and backward of the blocks of every layer in that trace; never the
# default).
LEVELS: tuple[str, ...] = ("paused", "coarse", "detail", "modules")
