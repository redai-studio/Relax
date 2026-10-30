# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""This rank's brackets, from the timer call sites to the window statistics.

Hot path (the framework's timer call sites, through ``timers``):
``record_event`` / ``push`` record events on the current stream and append
them to a queue. Once per step ``end_step`` reads the completed pairs with
``event.query()`` (never ``synchronize``) into the window's ``WindowStats``,
which the collector gathers when the window closes.

A "step" in this package is one ``end_step`` call, i.e. one rollout, which may
contain several optimizer steps.

When the collector asks for a sample, the first step of the next window is
also placed on a real time axis (see ``analysis.trace``): it starts with one
device synchronize and an anchor event, and its brackets keep their offsets
from that anchor. With ``modules_on`` that step also carries module brackets
(see ``modules``), which are attached only while it runs.
"""

from __future__ import annotations

import gc
import threading
from array import array
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter, time
from typing import Any, Callable, Sequence

from relax.utils.straggler.core.health import ProfilerHealth
from relax.utils.straggler.core.modules import ModuleTimers
from relax.utils.straggler.core.stats import (
    FIELD_INDEX,
    NUM_FIELDS,
    RECORD_WIDTH,
    STEP_PEAK_FIELDS,
    STEP_PEAK_SEGMENTS,
    RankTrace,
    WindowStats,
)
from relax.utils.straggler.core.timers import EventPool


_FWD_INDEX = FIELD_INDEX["fwd"]
_BWD_INDEX = FIELD_INDEX["bwd"]
_LP_FWD_INDEX = FIELD_INDEX["lp_fwd"]
_PP_RECV_INDEX = FIELD_INDEX["pp_recv"]
# Segments whose brackets are numbered within the step (micro-batch order).
_NUMBERED = (_FWD_INDEX, _BWD_INDEX, _LP_FWD_INDEX)
_CPU_FWD_INDEX = FIELD_INDEX["cpu_fwd"]
_CPU_BWD_INDEX = FIELD_INDEX["cpu_bwd"]
_NUM_FWD_INDEX = FIELD_INDEX["num_fwd"]
_TOKENS_INDEX = FIELD_INDEX["tokens"]
_GC_INDEX = FIELD_INDEX["gc"]
_DROPPED_INDEX = FIELD_INDEX["dropped"]
_MB_FWD_MAX_INDEX = FIELD_INDEX["mb_fwd_max"]
_MB_FWD_MIN_INDEX = FIELD_INDEX["mb_fwd_min"]
_MB_BWD_MAX_INDEX = FIELD_INDEX["mb_bwd_max"]
_MB_BWD_MIN_INDEX = FIELD_INDEX["mb_bwd_min"]
_MB_FWD_ARGMAX_INDEX = FIELD_INDEX["mb_fwd_argmax"]
_MB_FWD_ARGMAX_TOKENS_INDEX = FIELD_INDEX["mb_fwd_argmax_tokens"]
_MB_TOKENS_MAX_INDEX = FIELD_INDEX["mb_tokens_max"]
_MB_TOKENS_MIN_INDEX = FIELD_INDEX["mb_tokens_min"]
# (window total, one-step maximum) of each segment whose slowest step the window keeps.
_STEP_PEAK_INDICES = tuple(
    (FIELD_INDEX[name], FIELD_INDEX[peak]) for name, peak in zip(STEP_PEAK_SEGMENTS, STEP_PEAK_FIELDS)
)

# (start, end) wall-clock seconds -> the primary rank's CPU timeline events in that span.
CpuEventsFn = Callable[[float, float], list[dict[str, Any]]]


@dataclass
class _TraceCapture:
    """The sampled step being read out: its anchor event and wall-clock time,
    the sequence numbers of its first window pair and one past its last (set
    when the step ends), its brackets so far (``modules`` of them module
    brackets) and how many were lost."""

    anchor: Any
    wall: float
    first: int
    end: int | None = None
    records: array = field(default_factory=lambda: array("f"))
    lost: int = 0
    modules: int = 0

    def covers(self, seq: int) -> bool:
        return self.first <= seq and (self.end is None or seq < self.end)


class EventRecorder:
    """Queue this rank's timing events and fold them lazily into the window
    statistics and, during a sampled step, its trace: the ``timers.EventSink``
    that the timers and module brackets report to.

    Torch-free: device events and stream-capture detection are injected. The
    collector closes the windows and steers the recorder through
    ``recording`` (off while paused or switched off), ``modules_on`` (module
    brackets in the sampled step) and ``reset_window`` (whether the next
    window's first step is sampled).
    """

    def __init__(
        self,
        *,
        event_factory: Callable[[], Any],
        is_capturing: Callable[[], bool],
        health: ProfilerHealth,
        pool_size: int = 4096,
        max_pending: int = 16384,
        clock: Callable[[], float] = perf_counter,
        wall_clock: Callable[[], float] = time,
        synchronize: Callable[[], None] = lambda: None,
        cpu_events: CpuEventsFn | None = None,
        max_trace_brackets: int = 4096,
        max_module_brackets: int = 8192,
        interleaved: bool = False,
        register_gc_callback: bool = True,
    ) -> None:
        self._health = health
        self.window = WindowStats()
        # Set by the collector; a plain attribute because ``record_event`` checks it on every bracket.
        self.recording = True
        self.modules_on = False
        self._pool = EventPool(pool_size, event_factory)
        self._max_pending = max_pending
        # Window brackets: (segment, PP phase field or -1, start event, end event, CPU ms, micro-batch or -1,
        # tokens or -1); tokens only for a training forward.
        self._pending: deque[tuple[int, int, Any, Any, float, int, float]] = deque()
        # Window pairs ever pushed and ever popped (folded or dropped): the sequence number of the queue head.
        self._pushed = 0
        self._popped = 0
        # Module brackets of the sampled step: (segment, layer, start event, end event). They share
        # ``max_pending`` with the window brackets, take at most half of it and exist only while a sample is read out.
        self._modules: deque[tuple[int, int, Any, Any]] = deque()
        # Brackets of each numbered segment pushed in this step, and the token
        # count reported for the training forward bracket that is open (-1 when none was).
        self._step_numbers = dict.fromkeys(_NUMBERED, 0)
        self._next_tokens = -1.0
        # Under interleaved PP the n-th forward / backward of a step is not micro-batch n: ``note_schedule`` names
        # the brackets of each forward-backward call, and the rest carry -1.
        self._interleaved = interleaved
        # The noted forward-backward call: bracket counts of its segments at its start, its table, model chunks.
        self._schedule: tuple[dict[int, int], Sequence[tuple[int, int]], int] | None = None
        # The window's totals of the step-peak segments at the end of its previous step.
        self._step_base = [0.0] * len(_STEP_PEAK_INDICES)
        self._clock = clock
        self._wall = wall_clock
        self._synchronize = synchronize
        self._is_capturing = is_capturing
        self._max_trace_brackets = max_trace_brackets
        self._max_module_brackets = max_module_brackets
        # Set by the integration once the model exists; attached only during a sampled step with ``modules_on``.
        self.module_timers: ModuleTimers | None = None
        self.model_chunks = 1
        # The first window has no sample: its first step carries the warm-up.
        self._sample_next = False
        self._trace: _TraceCapture | None = None
        self._trace_done: RankTrace | None = None
        # Primary rank only: the CPU events of the step traced last, read when it ends.
        self._cpu_events_source = cpu_events
        self._cpu_events: list[dict[str, Any]] = []
        # True from the first bracket of a step until ``end_step``; GC pauses are
        # only attributed to the step while it is set (not during train_wait,
        # offload / clear_memory, etc.).
        self._step_open = False
        self._gc_start: float | None = None
        # Thread that first calls record_event; only that thread's GC pauses
        # are attributed to the training window (gc.callbacks fire on the
        # triggering thread, so a background GC must not corrupt _gc_start).
        self._train_thread: int | None = None
        if register_gc_callback:
            gc.callbacks.append(self.on_gc)

    # ---- hot path -----------------------------------------------------------------------------------------------

    def record_event(self) -> Any:
        """Record a pooled event on the current stream and open the step for GC
        attribution.

        Returns ``None`` (the bracket is skipped) while not ``recording``, and
        during CUDA graph capture, where ``elapsed_time`` would be meaningless
        on replay.
        """
        if not self.recording or self._is_capturing():
            return None
        if not self._step_open:
            self._step_open = True
            if self._sample_next:
                self._begin_trace()
        if self._train_thread is None:
            self._train_thread = threading.get_ident()
        event = self._pool.acquire()
        event.record()
        return event

    def discard_event(self, event: Any) -> None:
        """Return the start event of a bracket that could not be closed."""
        self._pool.release(event)

    def on_error(self, where: str, exc: BaseException) -> None:
        """Count an exception caught on the hot path (see ``ProfilerHealth``);
        in the sampled step it also counts in the sample's ``lost``, since the
        bracket it skipped may be one that the other ranks kept."""
        self._health.record_error(where, exc)
        trace = self._trace
        if trace is not None and trace.end is None:
            trace.lost += 1

    def on_degraded(self, reason: str) -> None:
        """Record a lasting gap in what this rank measures (see
        ``ProfilerHealth.mark_degraded``)."""
        self._health.mark_degraded(reason)

    def push(self, segment_index: int, phase_index: int, start_event: Any, end_event: Any, cpu_ms: float) -> None:
        """Queue one bracket for ``drain``; when the queue is full drop the
        oldest pair, so a rank that stops draining cannot grow without bound.

        Module brackets give way to window brackets: one that finds the queue
        full, or half of it taken by module brackets, is dropped itself, and a
        window bracket that finds the queue full drops the oldest module
        bracket before it drops a window bracket. A dropped window bracket
        counts in ``dropped``; a dropped module bracket only in the sample's
        ``lost``.

        A fwd / bwd / log-prob fwd bracket takes the next micro-batch number of
        its segment in the step, under interleaved PP the micro-batch and model
        chunk that ``note_schedule`` gives it; a training forward also takes the
        token count reported while it was open.
        """
        trace = self._trace
        full = len(self._pending) + len(self._modules) >= self._max_pending
        if segment_index >= NUM_FIELDS:
            if trace is not None and not full and 2 * len(self._modules) < self._max_pending:
                self._modules.append((segment_index, phase_index, start_event, end_event))
                return
            self._pool.release(start_event, end_event)
            if trace is not None:
                trace.lost += 1
            return
        if full and trace is not None and self._modules:
            _, _, old_start, old_end = self._modules.popleft()
            self._pool.release(old_start, old_end)
            trace.lost += 1
        elif full:
            _, _, old_start, old_end, _, _, _ = self._pending.popleft()
            self._pool.release(old_start, old_end)
            self.window.add_index(_DROPPED_INDEX, 1.0)
            if trace is not None and trace.covers(self._popped):
                trace.lost += 1
            self._popped += 1
        micro_batch = self._step_numbers.get(segment_index, -1)
        tokens = -1.0
        if micro_batch >= 0:
            self._step_numbers[segment_index] = micro_batch + 1
            if segment_index == _FWD_INDEX:
                tokens, self._next_tokens = self._next_tokens, -1.0
            if self._interleaved:
                # The phase slot of a micro-batch bracket holds its model chunk.
                micro_batch, phase_index = self._scheduled(segment_index, micro_batch)
        self._pending.append((segment_index, phase_index, start_event, end_event, cpu_ms, micro_batch, tokens))
        self._pushed += 1

    def note_schedule(self, build_table: Callable[[], Sequence[tuple[int, int]]], forward_only: bool) -> None:
        """Take the schedule table of the forward-backward call about to run:
        the (micro-batch, model chunk) of each forward step in call order.

        Megatron runs the backward steps in the same order with the model
        chunks reversed; an empty table names no bracket. ``build_table`` is
        only called under interleaved PP.
        """
        if not self._interleaved:
            return
        table = self._health.guarded("schedule", build_table, default=())
        segments = (_LP_FWD_INDEX,) if forward_only else (_FWD_INDEX, _BWD_INDEX)
        chunks = 1 + max((chunk for _, chunk in table), default=-1)
        self._schedule = ({segment: self._step_numbers[segment] for segment in segments}, table, chunks)

    def _scheduled(self, segment_index: int, count: int) -> tuple[int, int]:
        """Micro-batch and model chunk of the ``count``-th bracket of a
        numbered segment in the step; (-1, -1) outside the noted call."""
        if self._schedule is None:
            return -1, -1
        starts, table, chunks = self._schedule
        start = starts.get(segment_index)
        if start is None or count - start >= len(table):
            return -1, -1
        micro_batch, chunk = table[count - start]
        if segment_index == _BWD_INDEX:
            chunk = chunks - 1 - chunk
        # Each earlier call of the step ran ``chunks`` brackets per micro-batch.
        return start // chunks + micro_batch, chunk

    def _begin_trace(self) -> None:
        """Anchor the sampled step: an event recorded on an idle device marks
        the wall-clock time that every bracket of the step is placed
        against."""
        self._sample_next = False
        self._synchronize()
        anchor = self._pool.acquire()
        anchor.record()
        self._trace = _TraceCapture(anchor=anchor, wall=self._wall(), first=self._pushed)
        if self.modules_on and self.module_timers is not None:
            try:
                self.module_timers.attach()
            except Exception as exc:
                self._detach_modules()
                self.on_error("modules", exc)

    def _detach_modules(self) -> None:
        if self.module_timers is not None:
            self._health.guarded("modules", self.module_timers.detach)

    def note_tokens(self, tokens: int) -> None:
        """Take the token count of the training micro-batch whose forward
        bracket is open; that bracket's ``push`` claims it, and the window
        counts it once per micro-batch (each model chunk runs its own
        forward)."""
        self._next_tokens = float(tokens)
        self.window.add_index(_TOKENS_INDEX, tokens / self.model_chunks)

    def on_gc(self, phase: str, info: dict[str, Any]) -> None:
        if self._train_thread is not None and threading.get_ident() != self._train_thread:
            return
        if phase == "start":
            self._gc_start = self._clock() if self._step_open else None
        elif self._gc_start is not None:
            self.window.add_index(_GC_INDEX, (self._clock() - self._gc_start) * 1e3)
            self._gc_start = None

    # ---- once per step ------------------------------------------------------------------------------------------

    @property
    def pending_events(self) -> int:
        return len(self._pending) + len(self._modules)

    @property
    def unread(self) -> int:
        """Window pairs still queued; their time lands in the next window."""
        return len(self._pending)

    def drain(self) -> int:
        """Fold completed event pairs from the front of each queue into the
        window or the sample; stop at the first pair that is still in flight.

        Pairs are queued in stream order, so an incomplete pair implies every
        later pair of its queue is incomplete too. Each pair is popped only
        after it has been folded, so an exception leaves the queues consistent
        and no event is released twice. Returns the number of pairs folded.
        """
        started = self._clock()
        folded = 0
        pending = self._pending
        window = self.window
        trace = self._trace
        while pending:
            segment_index, phase_index, start_event, end_event, cpu_ms, micro_batch, tokens = pending[0]
            if not (end_event.query() and start_event.query()):
                break
            ms = start_event.elapsed_time(end_event)
            # Read before anything is folded, so a failing readout leaves no partial update.
            traced = trace is not None and trace.covers(self._popped)
            if traced:
                offsets = (trace.anchor.elapsed_time(start_event), trace.anchor.elapsed_time(end_event))
            window.add_index(segment_index, ms)
            # Only a PP wait's phase is a field; a gradient sync's names its timer, for the trace.
            if segment_index == _PP_RECV_INDEX and phase_index >= 0:
                window.add_index(phase_index, ms)
            if segment_index == _FWD_INDEX:
                window.add_index(_CPU_FWD_INDEX, cpu_ms)
                window.add_index(_NUM_FWD_INDEX, 1.0)
                if window.max_index(_MB_FWD_MAX_INDEX, ms):
                    window.values[_MB_FWD_ARGMAX_INDEX] = float(micro_batch)
                    window.values[_MB_FWD_ARGMAX_TOKENS_INDEX] = tokens
                window.min_index(_MB_FWD_MIN_INDEX, ms)
                if tokens >= 0:
                    window.max_index(_MB_TOKENS_MAX_INDEX, tokens)
                    window.min_index(_MB_TOKENS_MIN_INDEX, tokens)
            elif segment_index == _BWD_INDEX:
                window.add_index(_CPU_BWD_INDEX, cpu_ms)
                window.max_index(_MB_BWD_MAX_INDEX, ms)
                window.min_index(_MB_BWD_MIN_INDEX, ms)
            if traced:
                if len(trace.records) // RECORD_WIDTH - trace.modules < self._max_trace_brackets:
                    trace.records.extend((segment_index, phase_index, micro_batch, tokens, *offsets))
                else:
                    trace.lost += 1
            pending.popleft()
            self._pool.release(start_event, end_event)
            self._popped += 1
            folded += 1
        modules = self._modules
        while modules and trace is not None:
            segment_index, layer, start_event, end_event = modules[0]
            if not (end_event.query() and start_event.query()):
                break
            offsets = (trace.anchor.elapsed_time(start_event), trace.anchor.elapsed_time(end_event))
            if trace.modules < self._max_module_brackets:
                # The phase slot holds the layer number.
                trace.records.extend((segment_index, layer, -1, -1.0, *offsets))
                trace.modules += 1
            else:
                trace.lost += 1
            modules.popleft()
            self._pool.release(start_event, end_event)
            folded += 1
        self._finish_trace()
        window.add("overhead", (self._clock() - started) * 1e3)
        return folded

    def end_step(self) -> None:
        """Close one training step: read what has completed, end a sampled step
        (its pairs still in flight are read by later steps) and keep the step's
        peaks; never raises."""
        self._health.guarded("drain", self.drain)
        self._step_open = False
        self._step_numbers = dict.fromkeys(_NUMBERED, 0)
        self._schedule = None
        if self._trace is not None and self._trace.end is None:
            self._detach_modules()
            self._trace.end = self._pushed
            if self._cpu_events_source is not None:
                # Read now, before the window can close, so the sample leaves with the events of its own step.
                self._cpu_events = self._health.guarded(
                    "trace", self._cpu_events_source, self._trace.wall, self._wall(), default=[]
                )
            self._finish_trace()
        self._note_step_peaks()
        self.window.add("num_steps", 1.0)

    def _note_step_peaks(self) -> None:
        """Keep the largest one-step total of each step-peak segment; a bracket
        still unread at ``end_step`` counts towards the step that reads it."""
        values = self.window.values
        for k, (total, peak) in enumerate(_STEP_PEAK_INDICES):
            self.window.max_index(peak, values[total] - self._step_base[k])
            self._step_base[k] = values[total]

    def _finish_trace(self) -> None:
        """Hand the sampled step over once every pair of it has been read."""
        trace = self._trace
        if trace is None or trace.end is None or self._popped < trace.end or self._modules:
            return
        self._pool.release(trace.anchor)
        self._trace = None
        self._trace_done = RankTrace(wall=trace.wall, records=trace.records, lost=trace.lost)

    # ---- once per window ----------------------------------------------------------------------------------------

    def take_sample(self) -> RankTrace | None:
        """The complete sampled step of this window, with the CPU events of its
        step on the primary rank, or ``None``; a sample still in flight (only
        with very short windows) is dropped."""
        self._drop_trace()
        sample, self._trace_done = self._trace_done, None
        if sample is not None:
            sample.cpu_events = self._cpu_events
        self._cpu_events = []
        return sample

    def reset_window(self, sample_next: bool) -> None:
        """Start the next window's statistics; with ``sample_next`` its first
        step is sampled."""
        self.window.reset()
        self._step_base = [0.0] * len(_STEP_PEAK_INDICES)
        self._sample_next = sample_next

    def stop(self) -> None:
        """Record nothing from now on and return every queued event to the
        pool."""
        self.recording = False
        self._release_all(self._pending)
        self._drop_trace()
        self._detach_modules()

    def close(self) -> None:
        """Unregister the GC callback; safe to call more than once."""
        if self.on_gc in gc.callbacks:
            gc.callbacks.remove(self.on_gc)

    def _drop_trace(self) -> None:
        """Abandon the sampled step and its module brackets."""
        if self._trace is None:
            return
        self._pool.release(self._trace.anchor)
        self._trace = None
        self._release_all(self._modules)

    def _release_all(self, queue: deque) -> None:
        """Empty ``queue`` and return its events to the pool."""
        while queue:
            _, _, start_event, end_event, *_ = queue.popleft()
            self._pool.release(start_event, end_event)
