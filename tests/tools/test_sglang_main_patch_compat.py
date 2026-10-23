# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep main's PD and HiCache fixes when upgrading the SGLang patch."""

from __future__ import annotations

import ast
import os
from enum import IntEnum
from types import SimpleNamespace

import pytest

from tests.tools.sglang_patch_source import patched_functions, patched_source


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
        )
        for index, entry in enumerate([899, 950])
    ]
    failures, local_votes = [], []
    cp_group, tp_group = object(), object()

    def consensus(pollers, cp, tp):
        assert (cp, tp) == (cp_group, tp_group)
        votes = [poller.poll() for poller in pollers]
        local_votes.extend(votes)
        return [Poll.Failed] * len(votes) if peer_failed else votes

    namespace.update(
        poll_and_all_reduce_attn_cp_tp_group=consensus,
        poll_and_all_reduce_pp=lambda ids, *_: [None for _ in ids],
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
    function = _load(target, {"pop_bootstrapped"}, namespace)["pop_bootstrapped"]
    bootstrapped, failed = function(queue, return_failed_reqs=True)
    expected = requests if peer_failed else requests[:1]
    assert not bootstrapped and failed == failures == expected
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
