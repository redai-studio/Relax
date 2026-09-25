# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The training-loop context the straggler profiler attaches to its evidence.

This module is the **only** place where a training-loop integer (the rollout id
and the optimizer-step index) enters the profiler. The training thread calls
:func:`record_optimizer_step` once per optimizer step; it is a counted no-op
unless the profiler is enabled, and nothing here performs I/O, opens a socket,
synchronises a device or takes a collective.

A background thread (the observer/collector/sender) may read the value at any
time, so the slot is guarded by a lock and the stored value is a frozen
dataclass: a reader sees either the previous context or the next one, never a
half-written one. Writing allocates exactly one small value object and swaps a
single module-level slot, so a long run cannot grow memory here; reading
allocates nothing.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TrainingContext:
    """One immutable snapshot of where the training loop is.

    Attributes:
        rollout_id: Rollout the optimizer step belongs to.
        optimizer_step: Index of the step inside that rollout.
        sample_seq: Optional sample sequence index, when the caller knows one.
        global_step: ``rollout_id * num_steps_per_rollout + optimizer_step``
            when the rollout length is known, otherwise the bare optimizer-step
            index.
        updated_at: ``time.monotonic()`` at publication, for staleness checks.
    """

    rollout_id: int
    optimizer_step: int
    sample_seq: Optional[int]
    global_step: int
    updated_at: float
    #: LOCAL per-rank work for this optimizer step, when the data path published
    #: it. These are advisory transport metadata: the detector reports the
    #: peer-relative difference next to the timing gap and never divides it out.
    tokens: Optional[int] = None
    sequences: Optional[int] = None
    microbatches: Optional[int] = None


_CONTEXT_LOCK = threading.Lock()
_CURRENT: Optional[TrainingContext] = None
_UPDATES = 0
_FAILURES = 0

#: Per-rollout ``(tokens, sequences, microbatches)`` triples published by the
#: data path, keyed by rollout so a prefetch that runs ahead cannot hand a step
#: another rollout's work. Bounded: the oldest rollout is evicted past the cap.
_STEP_WORKLOADS: "OrderedDict[int, Tuple[Tuple[int, int, int], ...]]" = OrderedDict()
MAX_ROLLOUT_WORKLOADS = 8
_WORKLOAD_EVICTIONS = 0


def _count_failure() -> None:
    """Count one swallowed failure; the counter is a bounded scalar."""
    global _FAILURES
    with _CONTEXT_LOCK:
        _FAILURES += 1


def publish_step_workload(rollout_id: int, steps: Sequence[Sequence[int]]) -> None:
    """Publish this rank's LOCAL work per optimizer step for one rollout.

    Called once per rollout from the data path with
    ``(tokens, sequences, microbatches)`` per step, all computed from values the
    caller already holds: pure Python ``sum`` over the existing
    ``total_lengths`` list and its step slices. It adds no tensor-to-Python
    conversion, no device synchronisation, no collective and no Megatron-schedule
    change, and the publication is an O(1) swap of one bounded dict entry.

    Publishing the *local* counts is the point: a rank-invariant figure such as
    ``step_global_batch_size`` would make ``workload_delta`` identically zero and
    leave the comparability gate inert forever. Failures are counted and the
    previous publication is kept, because the profiler must not perturb the loop
    it observes.
    """
    global _WORKLOAD_EVICTIONS
    try:
        normalised = tuple((int(step[0]), int(step[1]), int(step[2])) for step in steps)
    except Exception:
        _count_failure()
        return
    with _CONTEXT_LOCK:
        _STEP_WORKLOADS[int(rollout_id)] = normalised
        while len(_STEP_WORKLOADS) > MAX_ROLLOUT_WORKLOADS:
            _STEP_WORKLOADS.popitem(last=False)
            _WORKLOAD_EVICTIONS += 1


def _step_workload(rollout_id: int, optimizer_step: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return this step's published local work, or ``(None, None, None)``.

    A rollout that was never published, or evicted, reads as absent rather than
    as another rollout's numbers.
    """
    rollout = _STEP_WORKLOADS.get(int(rollout_id))
    if not rollout or not 0 <= int(optimizer_step) < len(rollout):
        return None, None, None
    return rollout[int(optimizer_step)]


def _build_context(
    rollout_id: int,
    optimizer_step: int,
    sample_seq: Optional[int],
    num_steps_per_rollout: Optional[int],
) -> TrainingContext:
    """Assemble the frozen value object from the raw loop integers.

    Kept separate from :func:`set_training_context` so a test can force a
    failure without touching the public entry point.
    """
    if num_steps_per_rollout is not None:
        global_step = int(rollout_id) * int(num_steps_per_rollout) + int(optimizer_step)
    else:
        global_step = int(optimizer_step)
    tokens, sequences, microbatches = _step_workload(rollout_id, optimizer_step)
    return TrainingContext(
        rollout_id=int(rollout_id),
        optimizer_step=int(optimizer_step),
        sample_seq=None if sample_seq is None else int(sample_seq),
        global_step=global_step,
        updated_at=time.monotonic(),
        tokens=tokens,
        sequences=sequences,
        microbatches=microbatches,
    )


def set_training_context(
    rollout_id: int,
    optimizer_step: int,
    sample_seq: Optional[int] = None,
    num_steps_per_rollout: Optional[int] = None,
) -> None:
    """Publish the current training position; O(1) and never raises.

    ``global_step`` is ``rollout_id * num_steps_per_rollout + optimizer_step``
    when the rollout length is known and falls back to the bare optimizer-step
    index otherwise. A failure is counted and the previous context is kept,
    because a profiler must not perturb the loop it observes.
    """
    global _CURRENT, _UPDATES
    try:
        context = _build_context(rollout_id, optimizer_step, sample_seq, num_steps_per_rollout)
    except Exception:
        _count_failure()
        return
    with _CONTEXT_LOCK:
        _CURRENT = context
        _UPDATES += 1


def get_training_context() -> Optional[TrainingContext]:
    """Return the current context, or ``None`` before the first update.

    Allocation-free so the background reader can poll it without feeding the
    allocator on the training thread's behalf.
    """
    with _CONTEXT_LOCK:
        return _CURRENT


def snapshot() -> Dict[str, Any]:
    """Return a JSON-friendly mapping for the wire protocol.

    Keys: ``rollout_id``, ``optimizer_step``, ``sample_seq``, ``global_step``.
    An empty mapping means no context has been published yet.
    """
    context = get_training_context()
    if context is None:
        return {}
    payload: Dict[str, Any] = {
        "rollout_id": context.rollout_id,
        "optimizer_step": context.optimizer_step,
        "sample_seq": context.sample_seq,
        "global_step": context.global_step,
    }
    # Workload rides the wire only when this rank's data path published it, so a
    # vehicle that does not publish keeps the original lean payload.
    for field in ("tokens", "sequences", "microbatches"):
        value = getattr(context, field)
        if value is not None:
            payload[field] = value
    return payload


def record_optimizer_step(rollout_id: int, optimizer_step: int, num_steps_per_rollout: Optional[int] = None) -> None:
    """Cheap hook the Megatron training loop calls once per optimizer step.

    ``num_steps_per_rollout`` is the rollout length the caller is iterating
    (``len(num_microbatches)`` in the Megatron backend). It lets
    ``global_step`` be the run-wide step index instead of the bare in-rollout
    index; when it is omitted the arithmetic in :func:`set_training_context`
    falls back to ``optimizer_step``.

    The master switch is checked lazily on every call (no cached boolean), so
    enabling the profiler after import, or re-importing this module, behaves
    consistently. When the profiler is off this returns immediately; any
    failure is counted and swallowed.
    """
    try:
        from relax.utils.straggler import is_straggler_profiler_enabled
    except Exception:
        _count_failure()
        return
    try:
        enabled = bool(is_straggler_profiler_enabled())
    except Exception:
        _count_failure()
        return
    if not enabled:
        return
    set_training_context(rollout_id, optimizer_step, num_steps_per_rollout=num_steps_per_rollout)


def training_context_stats() -> Dict[str, int]:
    """Return this module's bounded counters (test/diagnostic surface).

    ``stored`` is 0 or 1 by construction: the context lives in a single slot
    and no call appends to a list or dict, so ``updates`` growing over a run
    does not mean memory grows.
    """
    with _CONTEXT_LOCK:
        return {
            "updates": _UPDATES,
            "failures": _FAILURES,
            "stored": 0 if _CURRENT is None else 1,
            "workload_rollouts": len(_STEP_WORKLOADS),
            "workload_evictions": _WORKLOAD_EVICTIONS,
        }


def reset_training_context_for_tests() -> None:
    """Drop the stored context and zero the counters."""
    global _CURRENT, _UPDATES, _FAILURES, _WORKLOAD_EVICTIONS
    with _CONTEXT_LOCK:
        _CURRENT = None
        _UPDATES = 0
        _FAILURES = 0
        _STEP_WORKLOADS.clear()
        _WORKLOAD_EVICTIONS = 0


__all__ = [
    "MAX_ROLLOUT_WORKLOADS",
    "TrainingContext",
    "get_training_context",
    "publish_step_workload",
    "record_optimizer_step",
    "reset_training_context_for_tests",
    "set_training_context",
    "snapshot",
    "training_context_stats",
]
