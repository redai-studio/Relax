# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the sampled-step trace: layout, nesting, scalars, files."""

import json
from array import array
from dataclasses import replace
from functools import partial

import pytest

from relax.utils.straggler.analysis import trace as trace_module
from relax.utils.straggler.analysis.trace import FILE_PREFIX, TIMELINE_PID_BASE
from relax.utils.straggler.core.modules import module_segment
from relax.utils.straggler.core.stats import FIELD_INDEX, RankTrace
from relax.utils.straggler.megatron_adapter import GRAD_SYNC_TIMERS, MODULE_NAMES
from tests.utils.straggler_helpers import meta as _meta


# The names the profiler passes in.
build_trace = partial(trace_module.build_trace, sync_names=GRAD_SYNC_TIMERS, module_names=MODULE_NAMES)
write_window_trace = partial(trace_module.write_window_trace, sync_names=GRAD_SYNC_TIMERS, module_names=MODULE_NAMES)


def _bracket(segment, start, end, phase=None, micro_batch=-1, tokens=-1):
    return (FIELD_INDEX[segment], FIELD_INDEX[phase] if phase else -1, micro_batch, tokens, start, end)


def _sync(start, end, timer="all-grads-sync"):
    return (FIELD_INDEX["dp_grad_sync"], GRAD_SYNC_TIMERS.index(timer), -1, -1, start, end)


# Stage 0 of a two-stage 1F1B step: two forwards, a drain wait, two backwards,
# an 8 ms host-side gap, then the gradient sync and the optimizer.
_FIRST_STAGE = [
    _bracket("fwd", 0, 10, micro_batch=0, tokens=300),
    _bracket("pp_send", 10, 11),
    _bracket("fwd", 11, 21, micro_batch=1, tokens=500),
    _bracket("pp_send", 21, 22),
    _bracket("pp_recv", 22, 40, phase="pp_wait_drain"),
    _bracket("bwd", 40, 60, micro_batch=0),
    _bracket("pp_recv", 60, 62, phase="pp_wait_drain"),
    _bracket("bwd", 62, 82, micro_batch=1),
    _sync(90, 95),
    _bracket("optim", 95, 100),
]
_LAST_STAGE = [
    _bracket("pp_recv", 0, 11, phase="pp_wait_fill"),
    _bracket("fwd", 11, 21, micro_batch=0),
    _bracket("bwd", 21, 41, micro_batch=0),
    _bracket("pp_recv", 41, 42, phase="pp_wait_up"),
    _bracket("fwd", 42, 52, micro_batch=1),
    _bracket("bwd", 52, 72, micro_batch=1),
    _sync(90, 95),
    _bracket("optim", 95, 100),
]


def _trace(brackets, wall=1000.0, **kwargs):
    return RankTrace(wall=wall, records=array("f", [value for bracket in brackets for value in bracket]), **kwargs)


def _slices(trace, rank, name=None):
    pid = TIMELINE_PID_BASE + rank
    return [e for e in trace["traceEvents"] if e["ph"] == "X" and e["pid"] == pid and name in (None, e["name"])]


def _meta_name(trace, rank, kind="process_name"):
    pid = TIMELINE_PID_BASE + rank
    (event,) = [e for e in trace["traceEvents"] if e["ph"] == "M" and e["name"] == kind and e["pid"] == pid]
    return event["args"]


def _two_stages(rank1_lag_s=0.0005):
    # Ranks 0, 1: stage 0 (dp 0, 1); ranks 2, 3: stage 1.
    meta = _meta(4, pp_size=2)
    traces = [
        _trace(_FIRST_STAGE),
        _trace(_FIRST_STAGE, wall=1000.0 + rank1_lag_s),
        _trace(_LAST_STAGE),
        _trace(_LAST_STAGE),
    ]
    return traces, meta


def test_rows_follow_dp_then_pp_and_name_the_measured_and_balanced_bubble():
    trace, _ = build_trace(*_two_stages())
    assert [_meta_name(trace, rank, "process_sort_index")["sort_index"] for rank in range(4)] == [0, 2, 1, 3]
    # Stage 0 waits 18 + 2 ms; a balanced two-stage 1F1B idles (2 - 1) x (10 + 20) ms per optimizer step.
    assert _meta_name(trace, 0)["name"] == "pp0 dp0 tp0 · rank 0 · bubble 20.0 ms, balanced 1F1B 30.0 ms"
    assert _meta_name(trace, 2)["name"] == "pp1 dp0 tp0 · rank 2 · bubble 12.0 ms, balanced 1F1B 30.0 ms"
    json.dumps(trace)


def test_waits_untimed_gaps_and_busy_runs_nest_the_brackets():
    trace, _ = build_trace(*_two_stages())
    wall_us = 1000.0 * 1e6
    names = [e["name"] for e in sorted(_slices(trace, 0), key=lambda e: (e["ts"], -e["dur"]))]
    assert names == [
        "busy",
        "F0",
        "send",
        "F1",
        "send",
        "bubble drain",
        "busy",
        "B0",
        "bubble drain",
        "busy",
        "B1",
        "untimed",
        "busy",
        "grad sync",
        "optimizer",
    ]
    (gap,) = _slices(trace, 0, "untimed")
    assert (gap["ts"], gap["dur"]) == (wall_us + 82e3, 8e3)
    first_busy = min(_slices(trace, 0, "busy"), key=lambda e: e["ts"])
    assert (first_busy["ts"], first_busy["dur"]) == (wall_us, 22e3)
    (f1,) = _slices(trace, 0, "F1")
    assert f1["args"] == {"micro_batch": 1, "tokens": 500}
    assert _slices(trace, 0, "bubble drain")[0]["args"] == {"waits_for": "next stage"}
    assert _slices(trace, 2, "bubble fill")[0]["args"] == {"waits_for": "previous stage"}
    assert _slices(trace, 2, "bubble up")[0]["args"] == {"waits_for": "previous stage"}


def test_slices_are_in_whole_microseconds_so_a_run_and_its_last_member_end_together():
    trace, _ = build_trace([_trace(_FIRST_STAGE, wall=1792661892.6237913)], _meta(1))
    assert all(isinstance(e["ts"], int) and isinstance(e["dur"], int) for e in _slices(trace, 0))
    busy = min(_slices(trace, 0, "busy"), key=lambda e: e["ts"])
    last_send = max(_slices(trace, 0, "send"), key=lambda e: e["ts"])
    assert busy["ts"] + busy["dur"] == last_send["ts"] + last_send["dur"]


def test_scalars_report_ranks_alignment_and_the_largest_extra_bubble():
    _, metrics = build_trace(*_two_stages(rank1_lag_s=0.0005))
    assert metrics["straggler/trace/ranks"] == 4.0
    assert metrics["straggler/trace/lost"] == 0.0
    # Ranks 0 and 1 end the same gradient sync, 0.5 ms apart on their wall clocks.
    assert metrics["straggler/trace/align_spread_ms"] == pytest.approx(0.5, abs=1e-3)
    assert (metrics["straggler/trace/bubble_excess_ms_max"], metrics["straggler/trace/bubble_excess_rank"]) == (
        pytest.approx(-10.0),
        0.0,
    )


def test_ranks_align_on_the_dp_gradient_sync_and_the_other_syncs_keep_their_timer_names():
    # Both ranks end the DP sync together; rank 1's embedding sync, which waits for its own pipeline, ends 3 ms later.
    steps = [
        [*_FIRST_STAGE[:-2], _sync(90, 95), _sync(95, end, "embedding-grads-all-reduce"), _bracket("optim", end, 100)]
        for end in (96, 99)
    ]
    trace, metrics = build_trace([_trace(steps[0]), _trace(steps[1], wall=1000.0005)], _meta(2))
    assert metrics["straggler/trace/align_spread_ms"] == pytest.approx(0.5, abs=1e-3)
    (dp_sync,) = _slices(trace, 1, "grad sync")
    (embedding_sync,) = _slices(trace, 1, "embedding-grads-all-reduce")
    assert (dp_sync["dur"], embedding_sync["dur"]) == (5000, 4000)


def test_a_dp_group_across_hosts_shifts_each_host_by_its_clock_offset_to_the_primary_host():
    # DP16 on hosts h0 (ranks 0-7) and h1 (8-15); h0's clock runs 3 ms ahead and rank 8 is the primary.
    meta = [replace(m, primary=m.rank == 8) for m in _meta(16)]
    traces = [_trace(_FIRST_STAGE, wall=1000.003 if m.host == "h0" else 1000.0) for m in meta]
    trace, metrics = build_trace(traces, meta)
    assert metrics["straggler/trace/clock_offset_ms_max"] == pytest.approx(3.0, abs=1e-3)
    assert metrics["straggler/trace/align_spread_ms"] == pytest.approx(0.0, abs=1e-3)
    assert {_slices(trace, rank, "grad sync")[0]["ts"] for rank in (0, 8)} == {round(1000.0 * 1e6 + 90e3)}


def test_ranks_that_lost_brackets_take_no_part_in_the_clock_offsets_or_the_alignment():
    # As above, but every h1 rank lost brackets and kept a later reduction as its last one.
    meta = [replace(m, primary=m.rank == 8) for m in _meta(16)]
    later = [*_FIRST_STAGE[:-2], _sync(1090, 1095)]
    traces = [_trace(_FIRST_STAGE, wall=1000.003) if m.host == "h0" else _trace(later, lost=1) for m in meta]
    _, metrics = build_trace(traces, meta)
    assert "straggler/trace/clock_offset_ms_max" not in metrics
    assert metrics["straggler/trace/unaligned_hosts"] == 1.0
    assert metrics["straggler/trace/align_spread_ms"] == pytest.approx(0.0, abs=1e-3)


# A two-stage 1F1B step with 1 ms between a stage finishing a micro-batch and the next stage starting it.
_UPSTREAM = [
    _bracket("fwd", 0, 10, micro_batch=0),
    _bracket("fwd", 10, 20, micro_batch=1),
    _bracket("bwd", 39, 59, micro_batch=0),
    _bracket("bwd", 66, 86, micro_batch=1),
    _sync(86, 90),
]
_DOWNSTREAM = [
    _bracket("fwd", 11, 21, micro_batch=0),
    _bracket("bwd", 21, 38, micro_batch=0),
    _bracket("fwd", 38, 48, micro_batch=1),
    _bracket("bwd", 48, 65, micro_batch=1),
    _sync(66, 90),
]


def test_hosts_linked_only_by_a_pp_hop_are_aligned_by_the_forward_and_backward_order():
    # Stage 0 on h0, stage 1 on h1 (each DP group on one host); h1's clock runs 3 ms ahead.
    meta = _meta(16, pp_size=2)
    traces = [_trace(_UPSTREAM) if m.pp == 0 else _trace(_DOWNSTREAM, wall=1000.003) for m in meta]
    trace, metrics = build_trace(traces, meta)
    assert metrics["straggler/trace/clock_offset_ms_max"] == pytest.approx(3.0, abs=1e-3)
    assert _slices(trace, 8, "F0")[0]["ts"] == round(1000.0 * 1e6 + 11e3)


def test_hosts_nothing_links_keep_their_own_clocks():
    meta = _meta(16, pp_size=2)
    traces = [_trace([_bracket("optim", 0, 5)], wall=1000.0 + m.pp * 0.003) for m in meta]
    trace, metrics = build_trace(traces, meta)
    assert "straggler/trace/clock_offset_ms_max" not in metrics
    assert metrics["straggler/trace/unaligned_hosts"] == 1.0
    assert _slices(trace, 8, "optimizer")[0]["ts"] == round(1000.003 * 1e6)


def test_syncs_and_blocks_the_adapter_did_not_name_get_numbered_names():
    brackets = [
        *_FIRST_STAGE[:-2],
        _sync(90, 95),
        _sync(95, 96, "embedding-grads-all-reduce"),
        _module("attn_fwd", 1, 1, 4),
    ]
    trace, _ = trace_module.build_trace([_trace(brackets)], _meta(1))
    names = {e["name"] for e in _slices(trace, 0)}
    assert {"grad sync", "grad sync 3", "block 0 fwd L1"} <= names


def test_missing_samples_are_skipped_and_cpu_events_get_their_own_row():
    traces, meta = _two_stages()
    cpu_event = {"name": "get_data_batch_1", "ph": "X", "ts": 1000.0 * 1e6, "dur": 5e3, "pid": 4242, "tid": 7}
    traces[1] = None
    traces[3] = _trace(_LAST_STAGE, lost=3, cpu_events=[cpu_event])
    trace, metrics = build_trace(traces, meta)
    assert (metrics["straggler/trace/ranks"], metrics["straggler/trace/lost"]) == (3.0, 3.0)
    assert not _slices(trace, 1)
    assert cpu_event in trace["traceEvents"]
    assert {"ph": "M", "name": "process_name", "pid": 4242, "args": {"name": "primary CPU (pid 4242)"}} in trace[
        "traceEvents"
    ]


def _module(name, layer, start, end):
    block, direction = name.rsplit("_", 1)
    return (module_segment(MODULE_NAMES, block, direction == "bwd"), layer, -1, -1, start, end)


def test_module_brackets_nest_in_their_micro_batch_and_mark_recompute_and_log_prob_forwards():
    modules = [
        _module("attn_fwd", 1, 1, 4),
        _module("moe_fwd", 1, 4, 9),
        # Activation recompute: the forward runs again inside B0, before its backward.
        _module("attn_fwd", 1, 41, 44),
        _module("moe_bwd", 1, 45, 50),
        _module("attn_bwd", 1, 50, 58),
        _module("mlp_fwd", 2, 101, 103),
        _module("attn_fwd", 3, 85, 86),
    ]
    log_prob = [_bracket("lp_fwd", 100, 104, micro_batch=5)]
    plain, _ = build_trace([_trace(_FIRST_STAGE + log_prob)], _meta(1))
    trace, metrics = build_trace([_trace(_FIRST_STAGE + log_prob + modules)], _meta(1))
    assert metrics["straggler/trace/module_brackets"] == 7.0
    # Busy runs, untimed gaps and the level-2 slices are those of the same step without modules.
    assert [e for e in _slices(trace, 0) if "layer" not in e.get("args", {})] == _slices(plain, 0)
    wall_us = 1000.0 * 1e6
    named = {
        (e["name"], (e["ts"] - wall_us) / 1e3): e["args"] for e in _slices(trace, 0) if "layer" in e.get("args", {})
    }
    assert named == {
        ("attn fwd L1", 1.0): {"layer": 1, "micro_batch": 0},
        ("moe fwd L1", 4.0): {"layer": 1, "micro_batch": 0},
        ("attn fwd L1 (recompute)", 41.0): {"layer": 1, "micro_batch": 0},
        ("moe bwd L1", 45.0): {"layer": 1, "micro_batch": 0},
        ("attn bwd L1", 50.0): {"layer": 1, "micro_batch": 0},
        ("lp mlp fwd L2", 101.0): {"layer": 2, "micro_batch": 5},
        ("attn fwd L3", 85.0): {"layer": 3},
    }


def test_brackets_without_a_micro_batch_number_are_named_without_one():
    brackets = [_bracket("fwd", 0, 10, tokens=300), _bracket("bwd", 10, 30), _module("attn_fwd", 1, 1, 4)]
    trace, _ = build_trace([_trace(brackets)], _meta(1))
    named = {e["name"]: e.get("args", {}) for e in _slices(trace, 0) if e["name"] != "busy"}
    assert named == {"F": {"tokens": 300}, "B": {}, "attn fwd L1": {"layer": 1}}


def test_under_interleaved_pp_micro_batch_brackets_and_their_modules_name_the_model_chunk():
    fwd, bwd = FIELD_INDEX["fwd"], FIELD_INDEX["bwd"]
    brackets = [(fwd, 1, 3, 300, 0, 10), (bwd, 0, 3, -1, 10, 30), _module("attn_fwd", 8, 1, 4)]
    trace, _ = build_trace([_trace(brackets)], _meta(1))
    named = {e["name"]: e.get("args", {}) for e in _slices(trace, 0) if e["name"] != "busy"}
    assert named == {
        "F3 chunk 1": {"micro_batch": 3, "model_chunk": 1, "tokens": 300},
        "B3 chunk 0": {"micro_batch": 3, "model_chunk": 0},
        "attn fwd L8": {"layer": 8, "micro_batch": 3, "model_chunk": 1},
    }


def test_an_mtp_layer_is_named_apart_from_the_decoder_layer_with_the_same_number():
    modules = [_module("attn_fwd", 1, 1, 4), _module("attn_fwd", -1, 5, 8)]
    trace, _ = build_trace([_trace(_FIRST_STAGE + modules)], _meta(1))
    named = {e["name"]: e["args"] for e in _slices(trace, 0) if "layer" in e.get("args", {})}
    assert named == {
        "attn fwd L1": {"layer": 1, "micro_batch": 0},
        "attn fwd MTP L1": {"layer": 1, "mtp": True, "micro_batch": 0},
    }


def test_without_pp_there_is_no_bubble_estimate():
    trace, metrics = build_trace([_trace(_FIRST_STAGE[:4])], _meta(1))
    assert _meta_name(trace, 0)["name"] == "pp0 dp0 tp0 · rank 0"
    assert "straggler/trace/bubble_excess_ms_max" not in metrics


def test_files_are_written_per_window_and_only_the_newest_are_kept(tmp_path):
    traces, meta = _two_stages()
    for first_rollout in (10, 20, 30):
        metrics = write_window_trace(str(tmp_path), 2, traces, meta, first_rollout)
    assert metrics["straggler/trace/ranks"] == 4.0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{FILE_PREFIX}000020.json",
        f"{FILE_PREFIX}000030.json",
    ]
    assert json.loads((tmp_path / f"{FILE_PREFIX}000030.json").read_text())["traceEvents"]

    write_window_trace(str(tmp_path / "empty"), 2, [None] * 4, meta, 40)
    assert not (tmp_path / "empty").exists()
