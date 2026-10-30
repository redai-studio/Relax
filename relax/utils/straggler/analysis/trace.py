# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Sampled-step trace: every rank's GPU brackets of one step on a real time
axis, gathered to the primary rank and written as a Chrome trace for Perfetto.

Each rank is one process row, ordered by (dp, pp, tp) so the stages of a
pipeline sit next to each other. Within a row the slices nest by time:

* level 1 (PP): busy runs and the PP waits between them (``bubble <phase>``),
  plus ``untimed`` gaps that no bracket covers (host-side time);
* level 2 (fwd / bwd): inside a busy run, ``F{i}`` / ``B{i}`` per micro-batch
  (``F{i} chunk {c}`` under interleaved PP, with the model chunk), sends, the
  gradient syncs (``grad sync`` is the DP reduction, the others keep their
  timer names) and the optimizer;
* level 3 (modules, only at the ``modules`` level): inside ``F{i}`` / ``B{i}``,
  the forward / backward of the blocks the adapter's module specs name in
  every layer, e.g. ``attn fwd L3``, ``dispatch bwd L3`` (``attn fwd MTP L1``
  for an MTP layer); a forward inside a ``B{i}`` is an activation recompute.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from relax.utils.straggler.core.modules import module_block
from relax.utils.straggler.core.stats import (
    DP_REDUCTION_PHASE,
    FIELD_INDEX,
    GPU_SEGMENTS,
    NUM_FIELDS,
    PP_PHASE_FIELDS,
    RECORD_WIDTH,
    RankMeta,
    RankTrace,
)


logger = logging.getLogger(__name__)

# Above any real pid (kernel.pid_max <= 2**22) so the straggler rows never merge
# with a live process's track in Perfetto.
TIMELINE_PID_BASE = 1 << 30
FILE_PREFIX = "straggler_trace_rollout"
# Gaps between brackets shorter than this are launch latency, not host stalls.
_UNTIMED_MS = 0.5

_FWD = FIELD_INDEX["fwd"]
_BWD = FIELD_INDEX["bwd"]
_LP_FWD = FIELD_INDEX["lp_fwd"]
_OPTIM = FIELD_INDEX["optim"]
_GRAD_SYNC = FIELD_INDEX["dp_grad_sync"]
_RECV = FIELD_INDEX["pp_recv"]
_LP_RECV = FIELD_INDEX["lp_pp_recv"]
_SEGMENT_NAMES = {FIELD_INDEX[name]: name for name in GPU_SEGMENTS}
_PHASE_NAMES = {FIELD_INDEX[name]: name.removeprefix("pp_wait_") for name in PP_PHASE_FIELDS}
_WAITS_FOR = {"fill": "previous stage", "up": "previous stage", "down": "next stage", "drain": "next stage"}
_MICRO_BATCH_NAMES = {_FWD: "F", _BWD: "B", _LP_FWD: "lp F"}
_LEVEL2_NAMES = {
    "pp_send": "send",
    "lp_pp_send": "lp send",
    "dp_param_gather": "param gather",
    "optim": "optimizer",
}


@dataclass
class _Bracket:
    segment: int
    phase: int
    micro_batch: int
    tokens: int
    start: float
    end: float

    @property
    def is_wait(self) -> bool:
        return self.segment in (_RECV, _LP_RECV)


def _brackets(trace: RankTrace) -> list[_Bracket]:
    values = trace.records
    rows = (values[i : i + RECORD_WIDTH] for i in range(0, len(values), RECORD_WIDTH))
    return sorted(
        (
            _Bracket(int(seg), int(phase), int(mb), int(tokens), start, end)
            for seg, phase, mb, tokens, start, end in rows
        ),
        key=lambda bracket: bracket.start,
    )


def _micro_batch_args(bracket: _Bracket) -> dict[str, Any]:
    """The micro-batch of a fwd / bwd / log-prob fwd bracket and, under
    interleaved PP, its model chunk (held in the phase slot); empty for other
    brackets."""
    if bracket.segment not in _MICRO_BATCH_NAMES:
        return {}
    args: dict[str, Any] = {"micro_batch": bracket.micro_batch} if bracket.micro_batch >= 0 else {}
    if bracket.phase >= 0:
        args["model_chunk"] = bracket.phase
    return args


def _name(bracket: _Bracket, sync_names: Sequence[str]) -> str:
    name = _MICRO_BATCH_NAMES.get(bracket.segment)
    if name is not None:
        if bracket.micro_batch >= 0:
            name += str(bracket.micro_batch)
        return name if bracket.phase < 0 else f"{name} chunk {bracket.phase}"
    if bracket.segment == _LP_RECV:
        return "lp bubble"
    if bracket.segment == _RECV:
        return f"bubble {_PHASE_NAMES.get(bracket.phase, 'recv')}"
    if bracket.segment == _GRAD_SYNC:
        if bracket.phase == DP_REDUCTION_PHASE:
            return "grad sync"
        return sync_names[bracket.phase] if 0 <= bracket.phase < len(sync_names) else f"grad sync {bracket.phase}"
    return _LEVEL2_NAMES[_SEGMENT_NAMES[bracket.segment]]


def _slice(
    name: str, pid: int, wall_us: float, start: float, end: float, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A slice from ``start`` to ``end`` ms after the anchor at ``wall_us``.

    In whole microseconds, like the CPU timeline: Perfetto reads a fractional
    ``ts`` as a double in ns, which this far from the epoch is off by up to 256
    ns, enough to push a slice past the end of the one it nests in.
    """
    ts = round(wall_us + start * 1e3)
    event = {
        "name": name,
        "ph": "X",
        "pid": pid,
        "tid": 0,
        "ts": ts,
        "dur": round(wall_us + end * 1e3) - ts,
    }
    if args:
        event["args"] = args
    return event


def _rank_events(
    brackets: list[_Bracket], pid: int, wall_us: float, sync_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Level-1 busy runs, waits and untimed gaps, with the level-2 brackets of
    each busy run."""
    events: list[dict[str, Any]] = []
    runs: list[tuple[float, float, list[_Bracket]]] = []
    for bracket in brackets:
        if not bracket.is_wait and runs and runs[-1][2] and bracket.start - runs[-1][1] <= _UNTIMED_MS:
            start, end, members = runs[-1]
            members.append(bracket)
            runs[-1] = (start, max(end, bracket.end), members)
        else:
            runs.append((bracket.start, bracket.end, [] if bracket.is_wait else [bracket]))
            if bracket.is_wait:
                phase = _PHASE_NAMES.get(bracket.phase)
                args = {"waits_for": _WAITS_FOR[phase]} if phase else None
                events.append(_slice(_name(bracket, sync_names), pid, wall_us, bracket.start, bracket.end, args))
    previous_end = None
    for start, end, members in runs:
        if previous_end is not None and start - previous_end > _UNTIMED_MS:
            events.append(_slice("untimed", pid, wall_us, previous_end, start))
        previous_end = end if previous_end is None else max(previous_end, end)
        if not members:
            continue
        events.append(_slice("busy", pid, wall_us, start, end))
        for bracket in members:
            args = _micro_batch_args(bracket)
            if bracket.tokens >= 0:
                args["tokens"] = bracket.tokens
            events.append(_slice(_name(bracket, sync_names), pid, wall_us, bracket.start, bracket.end, args))
    return events


def _module_events(
    modules: list[_Bracket], brackets: list[_Bracket], pid: int, wall_us: float, module_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Level-3 slices.

    Each module bracket takes the micro-batch of the ``F{i}`` / ``B{i}`` / ``lp
    F{i}`` bracket that contains it; both lists are sorted by start, and
    brackets of one stream do not overlap.
    """
    outers = [b for b in brackets if b.segment in (_FWD, _BWD, _LP_FWD)]
    events: list[dict[str, Any]] = []
    j = 0
    for module in modules:
        while j < len(outers) and outers[j].end < module.start:
            j += 1
        outer = (
            outers[j] if j < len(outers) and outers[j].start <= module.start <= module.end <= outers[j].end else None
        )
        layer, mtp = abs(module.phase), module.phase < 0
        block, backward = module_block(module_names, module.segment)
        name = f"{block} {'bwd' if backward else 'fwd'} {'MTP ' if mtp else ''}L{layer}"
        args: dict[str, Any] = {"layer": layer, "mtp": True} if mtp else {"layer": layer}
        if outer is not None:
            args.update(_micro_batch_args(outer))
            if outer.segment == _BWD and not backward:
                name += " (recompute)"
            elif outer.segment == _LP_FWD:
                name = f"lp {name}"
        events.append(_slice(name, pid, wall_us, module.start, module.end, args))
    return events


def _expected_bubble(brackets: list[_Bracket], stages: int) -> float | None:
    """Idle time a balanced pipeline would have in this sample.

    Non-interleaved 1F1B idles (stages - 1) x (mean fwd + mean bwd) per
    optimizer step while it fills and drains; ``None`` without PP or without
    fwd / bwd brackets.
    """
    fwd = [b.end - b.start for b in brackets if b.segment == _FWD]
    bwd = [b.end - b.start for b in brackets if b.segment == _BWD]
    if stages < 2 or not fwd or not bwd:
        return None
    optimizer_steps, in_optimizer = 0, False
    for bracket in brackets:
        if bracket.segment == _OPTIM and not in_optimizer:
            optimizer_steps += 1
        if bracket.segment in (_OPTIM, _FWD, _BWD):
            in_optimizer = bracket.segment == _OPTIM
    return max(optimizer_steps, 1) * (stages - 1) * (sum(fwd) / len(fwd) + sum(bwd) / len(bwd))


def _dp_sync_end(brackets: list[_Bracket]) -> float | None:
    """End of the last DP reduction, the one sync that ends at the same time on
    DP peers."""
    ends = [b.end for b in brackets if b.segment == _GRAD_SYNC and b.phase == DP_REDUCTION_PHASE]
    return ends[-1] if ends else None


def _add_edge(edges: dict[tuple[str, str], list[float]], a: str, b: str, delta: float) -> None:
    edges.setdefault((a, b), []).append(delta)
    edges.setdefault((b, a), []).append(-delta)


def _propagate(edges: dict[tuple[str, str], list[float]], offsets: dict[str, float]) -> None:
    neighbours: dict[str, list[tuple[str, float]]] = {}
    for (a, b), deltas in edges.items():
        neighbours.setdefault(a, []).append((b, median(deltas)))
    pending = list(offsets)
    while pending:
        a = pending.pop()
        for b, delta in neighbours.get(a, ()):
            if b not in offsets:
                offsets[b] = offsets[a] + delta
                pending.append(b)


def _clock_offsets(ranks: Sequence[tuple[RankMeta, float, list[_Bracket]]], reference: str) -> dict[str, float]:
    """Each host's wall clock minus the ``reference`` host's, in ms, from
    ``(meta, anchor wall time in ms, brackets)`` per rank; hosts that nothing
    links to the reference are left out.

    DP peers end the DP reduction together, so in each DP group the median end
    per host differs by the clock offset. Hosts no DP group links fall back to
    the cross-host PP hops: stage s + 1 starts F{i} only after stage s finished
    it, and stage s starts B{i} only after stage s + 1 finished it, so with the
    same latency both ways the offset is (smallest forward gap - smallest
    backward gap) / 2.
    """
    dp_ends: dict[tuple[int, int], dict[str, list[float]]] = {}
    hosts: dict[tuple[int, int, int], str] = {}
    micro_batches: dict[tuple[int, int, int], dict[tuple[int, int, int], tuple[float, float]]] = {}
    for rank_meta, wall_ms, brackets in ranks:
        end = _dp_sync_end(brackets)
        if end is not None:
            dp_ends.setdefault((rank_meta.pp, rank_meta.tp), {}).setdefault(rank_meta.host, []).append(wall_ms + end)
        position = (rank_meta.pp, rank_meta.dp, rank_meta.tp)
        hosts[position] = rank_meta.host
        micro_batches[position] = {
            (b.segment, b.micro_batch, b.phase): (wall_ms + b.start, wall_ms + b.end)
            for b in brackets
            if b.segment in (_FWD, _BWD) and b.micro_batch >= 0
        }
    dp_edges: dict[tuple[str, str], list[float]] = {}
    for by_host in dp_ends.values():
        first, *others = by_host
        for host in others:
            _add_edge(dp_edges, first, host, median(by_host[host]) - median(by_host[first]))
    pp_edges: dict[tuple[str, str], list[float]] = {}
    for position, up in micro_batches.items():
        pp, dp, tp = position
        below = (pp + 1, dp, tp)
        if hosts.get(below, hosts[position]) == hosts[position]:
            continue
        down = micro_batches[below]
        shared = up.keys() & down.keys()
        fwd = [down[key][0] - up[key][1] for key in shared if key[0] == _FWD]
        bwd = [up[key][0] - down[key][1] for key in shared if key[0] == _BWD]
        if fwd and bwd:
            _add_edge(pp_edges, hosts[position], hosts[below], (min(fwd) - min(bwd)) / 2)
    offsets = {reference: 0.0}
    _propagate(dp_edges, offsets)
    _propagate({**pp_edges, **dp_edges}, offsets)
    return offsets


def build_trace(
    traces: Sequence[RankTrace | None],
    meta: Sequence[RankMeta],
    sync_names: Sequence[str] = (),
    module_names: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, float]]:
    """The Chrome trace of one sampled step and its ``straggler/trace/*``
    scalars; ``traces[i]`` belongs to ``meta[i]`` (``None`` when that rank has
    no complete sample).

    Every host's ranks are shifted by its clock offset to the primary's host
    (``_clock_offsets``), so slices on different hosts line up. A rank that
    lost brackets may end on a different reduction than its peers, so it takes
    no part in the offsets or ``align_spread_ms``; hosts left without an offset
    keep their own clock and are counted in ``unaligned_hosts``.

    ``sync_names`` names the gradient syncs by phase slot and ``module_names``
    the blocks by index (the adapter's ``GRAD_SYNC_TIMERS`` and
    ``MODULE_NAMES``); a slot or index beyond them gets a numbered name.
    """
    stages = 1 + max((m.pp for m in meta), default=0)
    events: list[dict[str, Any]] = []
    grad_sync_ends: dict[tuple[int, int], list[float]] = {}
    excess: dict[int, float] = {}
    module_brackets = 0
    brackets_of = {i: _brackets(trace) for i, trace in enumerate(traces) if trace is not None}
    complete = [i for i in brackets_of if not traces[i].lost]
    reference = next((m.host for m in meta if m.primary), meta[0].host if meta else "")
    offsets = _clock_offsets([(meta[i], traces[i].wall * 1e3, brackets_of[i]) for i in complete], reference)
    order = sorted(range(len(meta)), key=lambda i: (meta[i].dp, meta[i].pp, meta[i].tp))
    for sort_index, i in enumerate(order):
        trace, rank_meta = traces[i], meta[i]
        if trace is None:
            continue
        pid = TIMELINE_PID_BASE + rank_meta.rank
        wall_ms = trace.wall * 1e3 - offsets.get(rank_meta.host, 0.0)
        wall_us = wall_ms * 1e3
        brackets = brackets_of[i]
        modules = [b for b in brackets if b.segment >= NUM_FIELDS]
        if modules:
            brackets = [b for b in brackets if b.segment < NUM_FIELDS]
            module_brackets += len(modules)
        bubble = sum(b.end - b.start for b in brackets if b.segment == _RECV)
        expected = _expected_bubble(brackets, stages)
        name = f"pp{rank_meta.pp} dp{rank_meta.dp} tp{rank_meta.tp} · rank {rank_meta.rank}"
        if expected is not None:
            name += f" · bubble {bubble:.1f} ms, balanced 1F1B {expected:.1f} ms"
            excess[rank_meta.rank] = bubble - expected
        events += [
            {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}},
            {"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": sort_index}},
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": 0, "args": {"name": "GPU compute stream"}},
        ]
        events += _rank_events(brackets, pid, wall_us, sync_names)
        events += _module_events(modules, brackets, pid, wall_us, module_names)
        sync_end = _dp_sync_end(brackets)
        if sync_end is not None and not trace.lost:
            grad_sync_ends.setdefault((rank_meta.pp, rank_meta.tp), []).append(wall_ms + sync_end)
        events += trace.cpu_events
        events += [
            {"ph": "M", "name": "process_name", "pid": pid_, "args": {"name": f"primary CPU (pid {pid_})"}}
            for pid_ in sorted({event["pid"] for event in trace.cpu_events})
        ]
    traced = [trace for trace in traces if trace is not None]
    metrics = {"straggler/trace/ranks": float(len(traced)), "straggler/trace/lost": float(sum(t.lost for t in traced))}
    if module_brackets:
        metrics["straggler/trace/module_brackets"] = float(module_brackets)
    spreads = [max(ends) - min(ends) for ends in grad_sync_ends.values() if len(ends) > 1]
    if spreads:
        metrics["straggler/trace/align_spread_ms"] = float(max(spreads))
    if len(offsets) > 1:
        metrics["straggler/trace/clock_offset_ms_max"] = float(max(abs(offset) for offset in offsets.values()))
    unaligned = {meta[i].host for i in brackets_of} - offsets.keys()
    if unaligned:
        metrics["straggler/trace/unaligned_hosts"] = float(len(unaligned))
    if excess:
        worst = max(excess, key=excess.__getitem__)
        metrics["straggler/trace/bubble_excess_ms_max"] = float(excess[worst])
        metrics["straggler/trace/bubble_excess_rank"] = float(worst)
    trace = {"traceEvents": events, "displayTimeUnit": "ms", "otherData": {"source": "relax straggler profiler"}}
    return trace, metrics


def write_window_trace(
    trace_dir: str,
    keep: int,
    traces: Sequence[RankTrace | None],
    meta: Sequence[RankMeta],
    first_rollout: int,
    sync_names: Sequence[str] = (),
    module_names: Sequence[str] = (),
) -> dict[str, float]:
    """Write the sampled step of the window starting at ``first_rollout`` and
    keep the newest ``keep`` (at least one) trace files; returns the trace
    scalars (names as in ``build_trace``)."""
    trace, metrics = build_trace(traces, meta, sync_names, module_names)
    if not metrics["straggler/trace/ranks"]:
        return metrics
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"{FILE_PREFIX}{first_rollout:06d}.json")
    with open(f"{path}.tmp", "w") as f:
        json.dump(trace, f)
    os.replace(f"{path}.tmp", path)
    names = sorted(name for name in os.listdir(trace_dir) if name.startswith(FILE_PREFIX) and name.endswith(".json"))
    for name in names[: max(len(names) - max(keep, 1), 0)]:
        os.remove(os.path.join(trace_dir, name))
    logger.info("[straggler] trace of rollout %d written to %s", first_rollout, path)
    return metrics
