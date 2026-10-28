# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep Relax decode deadlines inside SGLang's native failure consensus."""

import ast
import os
from importlib import metadata
from types import MethodType, SimpleNamespace

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


try:
    _sglang_version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(Version(_sglang_version) < Version("0.5.21"), reason="SGLang 0.5.21 PD queues")


def _load(relative_path: str, class_name: str, member_name: str | None = None, **namespace):
    path = sglang_source_path(relative_path)
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    if member_name is not None:
        node = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == member_name)
        node.decorator_list = []
    # The source methods use runtime-only types; defer their annotations.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


def _transport(status, states):
    transport = SimpleNamespace(status=status, abort_count=0)
    transport.poll = lambda: transport.status

    def abort():
        transport.abort_count += 1
        transport.status = states.Failed

    transport.abort = abort
    return transport


def test_prefill_one_rank_deadline_retires_request_on_all_ranks(monkeypatch):
    states = _load("srt/disaggregation/base/conn.py", "KVPoll")
    monkeypatch.setenv("SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT", "100")
    logger = SimpleNamespace(error=lambda *_: None, warning_once=lambda *_: None)
    poller = _load("srt/disaggregation/prefill.py", "_PrefillTransferDeadlinePoller", KVPoll=states, logger=logger)
    abort_type = type("Abort", (), {})
    requests = [
        SimpleNamespace(
            rid="request",
            disagg_kv_sender=_transport(states.Transferring, states),
            pending_bootstrap=False,
            time_stats=SimpleNamespace(prefill_transfer_queue_entry_time=entry, set_completion_time=lambda: None),
            finished_reason=abort_type(),
            return_logprob=False,
        )
        for entry in (1.0, 150.0)
    ]
    local = [poller(req, 200.0, 100.0).poll() for req in requests]
    assert local == [states.Failed, states.Transferring]
    failed, released, streamed = [], [], []

    def collective(pollers, *_):
        assert pollers[0].poll() in local
        # SGLang's MIN reduction propagates Failed from either CP/TP rank.
        return [min(local)]

    process = _load(
        "srt/disaggregation/prefill.py",
        "SchedulerDisaggregationPrefillMixin",
        "process_disagg_prefill_inflight_queue",
        os=os,
        time=SimpleNamespace(perf_counter=lambda: 200.0),
        KVPoll=states,
        _PrefillTransferDeadlinePoller=poller,
        poll_and_all_reduce_attn_cp_tp_group=collective,
        FINISH_ABORT=abort_type,
        maybe_release_metadata_buffer=lambda req, _: released.append(req),
        logger=logger,
    )
    poll_statuses = _load(
        "srt/disaggregation/prefill.py",
        "SchedulerDisaggregationPrefillMixin",
        "poll_prefill_transfer_statuses",
        os=os,
        time=SimpleNamespace(perf_counter=lambda: 200.0),
        _PrefillTransferDeadlinePoller=poller,
        poll_and_all_reduce_attn_cp_tp_group=collective,
    )
    for req in requests:
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[req],
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            handle_inflight_transfer_failure=failed.append,
            output_streamer=SimpleNamespace(stream_output=lambda reqs, *_: streamed.extend(reqs)),
            req_to_metadata_buffer_idx_allocator=object(),
        )
        scheduler.poll_prefill_transfer_statuses = MethodType(poll_statuses, scheduler)
        assert process(scheduler, rids_to_check=[req.rid]) == [req]
        assert scheduler.disagg_prefill_inflight_queue == []
    assert failed == released == streamed == requests


def test_pp_transfer_deadlines_enter_terminal_intersection_before_release(monkeypatch):
    states = _load("srt/disaggregation/base/conn.py", "KVPoll")
    monkeypatch.setenv("SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT", "100")
    logger = SimpleNamespace(error=lambda *_: None, warning_once=lambda *_: None)
    poller = _load("srt/disaggregation/prefill.py", "_PrefillTransferDeadlinePoller", KVPoll=states, logger=logger)
    abort_type = type("Abort", (), {})
    clock = SimpleNamespace(now=200.0)
    requests = [
        [
            SimpleNamespace(
                rid="request",
                disagg_kv_sender=_transport(states.Transferring, states),
                pending_bootstrap=False,
                finished_reason=abort_type(),
                return_logprob=False,
                time_stats=SimpleNamespace(
                    prefill_transfer_queue_entry_time=1.0 if (pp, tp) == (0, 0) else 150.0,
                    set_completion_time=lambda: None,
                ),
            )
            for tp in range(2)
        ]
        for pp in range(2)
    ]

    def collective(pollers, cp_group, tp_group):
        local = [poller(req, clock.now, 100.0).poll() for req in requests[tp_group]]
        assert pollers[0].poll() in local
        return [min(local)]

    namespace = dict(
        os=os,
        time=SimpleNamespace(perf_counter=lambda: clock.now),
        KVPoll=states,
        _PrefillTransferDeadlinePoller=poller,
        poll_and_all_reduce_attn_cp_tp_group=collective,
    )
    poll_statuses = _load(
        "srt/disaggregation/prefill.py",
        "SchedulerDisaggregationPrefillMixin",
        "poll_prefill_transfer_statuses",
        **namespace,
    )
    transferred = _load(
        "srt/managers/scheduler_pp_mixin.py",
        "SchedulerPPMixin",
        "_pp_pd_get_prefill_transferred_ids",
        KVPoll=states,
    )
    failed, released, streamed = [], [], []
    process = _load(
        "srt/disaggregation/prefill.py",
        "SchedulerDisaggregationPrefillMixin",
        "process_disagg_prefill_inflight_queue",
        **namespace,
        FINISH_ABORT=abort_type,
        logger=logger,
        maybe_release_metadata_buffer=lambda req, _: released.append(req),
    )
    schedulers = []
    for pp, ranks in enumerate(requests):
        stage = []
        for req in ranks:
            scheduler = SimpleNamespace(
                disagg_prefill_inflight_queue=[req],
                attn_cp_cpu_group=pp,
                attn_tp_cpu_group=pp,
                pp_group=SimpleNamespace(is_first_rank=pp == 0),
                handle_inflight_transfer_failure=failed.append,
                output_streamer=SimpleNamespace(stream_output=lambda reqs, *_: streamed.extend(reqs)),
                req_to_metadata_buffer_idx_allocator=object(),
            )
            scheduler.poll_prefill_transfer_statuses = MethodType(poll_statuses, scheduler)
            stage.append(scheduler)
        schedulers.append(stage)

    for now, expected in ((200.0, []), (300.0, ["request"])):
        clock.now = now
        consensus = []
        for stage in schedulers:
            results = []
            for scheduler in stage:
                scheduler._pp_recv_pyobj_from_prev_stage = lambda: consensus
                results.append(transferred(scheduler))
            assert results[0] == results[1]
            consensus = results[0]
        assert consensus == expected
        for stage in schedulers:
            for scheduler in stage:
                req = scheduler.disagg_prefill_inflight_queue[0]
                assert process(scheduler, rids_to_check=consensus) == ([req] if expected else [])
                assert scheduler.disagg_prefill_inflight_queue == ([] if expected else [req])
        if not expected:
            assert failed == released == streamed == []
    assert failed == released == streamed == [req for stage in requests for req in stage]
