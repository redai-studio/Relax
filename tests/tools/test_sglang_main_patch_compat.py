# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep main's PD and HiCache fixes when upgrading the SGLang patch."""

from __future__ import annotations

import ast
import os
from enum import IntEnum
from types import MethodType, SimpleNamespace

import pytest

from tests.backends.sglang.source_utils import sglang_source_path
from tests.tools.sglang_patch_source import extract_functions, patched_functions, patched_source


def _load(target: str, names: set[str], namespace: dict) -> dict:
    nodes = patched_functions(f"python/sglang/srt/{target}", names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), target, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("peer_failed", [False, True])
def test_bootstrap_deadline_is_reduced_before_queue_removal(monkeypatch, pp_size, peer_failed):
    class Poll(IntEnum):
        Failed = 0
        Bootstrapping = 1
        WaitingForInput = 2

    target = "disaggregation/prefill.py"
    source = ast.parse(patched_source(f"python/sglang/srt/{target}"))
    poller = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_BootstrapDeadlinePoller"
    )
    namespace = {"KVPoll": Poll, "os": os, "time": SimpleNamespace(perf_counter=lambda: 1000)}
    exec(compile(ast.Module(body=[poller], type_ignores=[]), target, "exec"), namespace)
    requests = [
        SimpleNamespace(
            rid=str(index),
            disagg_kv_sender=SimpleNamespace(poll=lambda: Poll.Bootstrapping),
            time_stats=SimpleNamespace(prefill_bootstrap_queue_entry_time=entry),
            prefill_attempt_count=0,
            finished_reason=None,
        )
        for index, entry in enumerate([899, 950])
    ]
    failures, local_votes = [], []
    cp_group, tp_group = object(), object()

    def consensus(pollers, cp, tp):
        assert (cp, tp) == (cp_group, tp_group)
        votes = [poller.poll() for poller in pollers]
        local_votes.extend(votes)
        # With PP, inject the remote failure through the previous stage's bad
        # RIDs below; without PP it arrives through the CP/TP collective.
        return [Poll.Failed] * len(votes) if peer_failed and pp_size == 1 else votes

    namespace.update(
        poll_and_all_reduce_attn_cp_tp_group=consensus,
        get_disagg=lambda: SimpleNamespace(optimistic_prefill_attempts=0),
    )
    queue = SimpleNamespace(
        pp_size=pp_size,
        queue=requests[:],
        scheduler=SimpleNamespace(
            attn_cp_cpu_group=cp_group, attn_tp_cpu_group=tp_group, handle_bootstrap_failure=failures.append
        ),
    )
    monkeypatch.setenv("SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT", "100")
    _load(target, {"poll_bootstrap_statuses", "pop_bootstrapped"}, namespace)
    queue.poll_bootstrap_statuses = MethodType(namespace["poll_bootstrap_statuses"], queue)
    queue.pop_bootstrapped = MethodType(namespace["pop_bootstrapped"], queue)
    expected = requests if peer_failed else requests[:1]
    if pp_size == 1:
        bootstrapped, failed = queue.pop_bootstrapped(return_failed_reqs=True)
        assert not bootstrapped and failed == expected
    else:
        # Execute the real PP collection, bad-union and application stages.
        # Only the previous rank's message and CP/TP collective are stubbed.
        utils_path = sglang_source_path("srt/disaggregation/utils.py")
        nodes = extract_functions(utils_path.read_text(), str(utils_path), {"poll_and_all_reduce_pp"})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(utils_path), "exec"), namespace)
        namespace["FINISH_ABORT"] = type("FinishAbort", (), {})
        _load(
            "managers/scheduler_pp_mixin.py",
            {"_pp_pd_get_bootstrapped_ids", "process_bootstrapped_queue", "_route_aborts_to_bad"},
            namespace,
        )
        received = []

        def receive_previous():
            received.append(True)
            return [[], [requests[1].rid] if peer_failed else []]

        scheduler = SimpleNamespace(
            disagg_prefill_bootstrap_queue=queue,
            pp_group=SimpleNamespace(is_first_rank=False),
            _pp_recv_pyobj_from_prev_stage=receive_previous,
            _route_aborts_to_bad=namespace["_route_aborts_to_bad"],
            waiting_queue=[],
        )
        consensus_ids = namespace["_pp_pd_get_bootstrapped_ids"](scheduler)
        assert consensus_ids[0] == []
        assert set(consensus_ids[1]) == {req.rid for req in expected}
        assert queue.queue == requests and failures == []
        result = namespace["process_bootstrapped_queue"](scheduler, consensus_ids)
        assert result == [[], [req.rid for req in expected]]
        assert scheduler.waiting_queue == [] and received == [True]
    assert failures == expected
    assert local_votes == [Poll.Failed, Poll.Bootstrapping]
    assert queue.queue == ([] if peer_failed else requests[1:])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("idle", [False, True])
def test_paused_scheduler_completes_pending_hicache_transfers(enabled, idle):
    events = []
    scheduler = SimpleNamespace(
        enable_hierarchical_cache=enabled,
        tree_cache=SimpleNamespace(check_hicache_events=lambda: events.append("dma")),
        is_fully_idle=lambda: idle,
        metrics_reporter=SimpleNamespace(
            record_scheduler_idle=lambda: events.append("idle"),
            record_scheduler_active=lambda: events.append("active"),
        ),
    )
    function = _load("managers/scheduler.py", {"_record_scheduler_state_for_paused_engine"}, {})[
        "_record_scheduler_state_for_paused_engine"
    ]
    function(scheduler)
    assert events == (["dma"] if enabled else []) + ["idle" if idle else "active"]


@pytest.mark.parametrize("blocked", [None, "disabled", "missing", "root", "backed", "pending", "unlocked", "no_mamba"])
def test_deferred_chunked_backup_keeps_only_a_live_locked_checkpoint(blocked):
    component = SimpleNamespace(FULL="full", MAMBA="mamba")
    node = SimpleNamespace(
        backuped=blocked == "backed",
        write_through_pending_id=1 if blocked == "pending" else None,
        component_data={
            "full": SimpleNamespace(lock_ref=0 if blocked == "unlocked" else 1),
            "mamba": SimpleNamespace(value=None if blocked == "no_mamba" else object()),
        },
    )
    hits = []
    cache = SimpleNamespace(
        _node_arena={} if blocked == "missing" else {5: node},
        root_node=node if blocked == "root" else object(),
        enable_hicache=blocked != "disabled",
        _inc_hit_count_and_check=lambda value: hits.append(value) or True,
        _build_backup_kv_action=lambda value: ("backup", value),
    )
    function = _load(
        "mem_cache/unified_cache/unified_tree_core.py",
        {"finish_deferred_chunked_backup"},
        {"ComponentType": component},
    )["finish_deferred_chunked_backup"]
    assert function(cache, 5) == ([("backup", node)] if blocked is None else [])
    assert hits == ([node] if blocked is None else [])
