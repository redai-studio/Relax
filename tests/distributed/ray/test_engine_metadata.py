# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine discovery contract tests using the real rollout module and
entrypoint.

No Ray runtime is started. Actor handles and init ObjectRefs are controlled
fakes: metadata reads must preserve timeout=0 and must never submit an actor
getter. Import failures are collection errors, not a successful all-skipped
test run.
"""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ray.core.generated.gcs_pb2 import ActorTableData
from ray.exceptions import GetTimeoutError

from relax.distributed.ray import rollout


ActorState = ActorTableData.ActorState


class InitResult:
    def __init__(self, value=None, *, pending=False, error=None, on_read=None):
        self.value = value
        self.pending = pending
        self.error = error
        self.reads = 0
        self.on_read = on_read

    def __await__(self):
        async def resolve():
            if self.error is not None:
                raise self.error
            assert not self.pending, "pending init requires an explicit completion event"
            return self.value

        return resolve().__await__()


def engine_metadata(url="http://[2001:db8::1]:30000", pid=123, node_id="node-1"):
    return {"url": url, "pid": pid, "node_id": node_id}


def make_engine(state=ActorState.ALIVE):
    engine = MagicMock()
    engine._get_local_state.return_value = state
    engine.get_url.remote.side_effect = lambda: pytest.fail("discovery submitted get_url RPC")
    engine.get_pid_and_node_id.remote.side_effect = lambda: pytest.fail("discovery submitted PID RPC")
    return engine


@pytest.fixture
def zero_wait_get(monkeypatch):
    calls = []

    def get(ref, *, timeout):
        calls.append((ref, timeout))
        assert timeout == 0, "discovery must not wait for engine init"
        assert isinstance(ref, InitResult), "discovery must read individual original init refs"
        ref.reads += 1
        if ref.on_read is not None:
            ref.on_read()
        if ref.pending:
            raise GetTimeoutError("init is still running")
        if ref.error is not None:
            raise ref.error
        return ref.value

    monkeypatch.setattr(rollout.ray, "get", get)
    return calls


def make_manager(engines, refs):
    group = rollout.EngineGroup(
        args=SimpleNamespace(num_gpus_per_node=8),
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=1,
        num_new_engines=0,
    )
    group.engine_init_records = {slot: SimpleNamespace(value=ref) for slot, ref in enumerate(refs)}
    server = rollout.RolloutServer(engine_groups=[group], router_ip="127.0.0.1", router_port=3000, model_name="actor")
    manager = object.__new__(rollout.RolloutManager.__ray_metadata__.modified_class)
    manager._engine_actor_states = ActorState
    manager.servers = {"actor": server}
    manager.args = SimpleNamespace(
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        scale_out_partial_success_policy="keep_partial",
        use_fault_tolerance=False,
    )
    return manager, group


def engine_rows(result, model_name="actor", group_index=0):
    return result["models"][model_name]["engine_groups"][group_index]["engines"]


def test_get_engines_info_never_submits_getters_while_init_pending(zero_wait_get):
    engine = make_engine()
    manager, _ = make_manager([engine], [InitResult(pending=True)])

    result = manager.get_engines_info()

    engine.get_url.remote.assert_not_called()
    engine.get_pid_and_node_id.remote.assert_not_called()
    assert result["observation"] == {
        "complete": False,
        "issues": [{"scope": "actor/0/0", "reason": "init_pending"}],
    }


def test_get_engines_info_preserves_ready_peer_when_same_group_init_pending(zero_wait_get):
    ready, pending = make_engine(), make_engine()
    metadata = engine_metadata()
    ready_ref, pending_ref = InitResult(metadata), InitResult(pending=True)
    manager, _ = make_manager([ready, pending], [ready_ref, pending_ref])

    result = manager.get_engines_info()

    assert engine_rows(result)[0].get("url") == metadata["url"]
    assert engine_rows(result)[0]["pid"] == metadata["pid"]
    assert "url" not in engine_rows(result)[1]
    assert result["observation"] == {
        "complete": False,
        "issues": [{"scope": "actor/0/1", "reason": "init_pending"}],
    }
    assert zero_wait_get == [(ready_ref, 0), (pending_ref, 0)]


def assert_issue(result, reason, *, scope="actor/0/0"):
    assert result["observation"] == {"complete": False, "issues": [{"scope": scope, "reason": reason}]}


def assert_no_metadata(row):
    assert not {"url", "pid", "node_id"}.intersection(row)


@pytest.mark.parametrize(
    ("state", "state_name", "reason"),
    [
        (ActorState.DEAD, "DEAD", "actor_dead_unreconciled"),
        (ActorState.RESTARTING, "RESTARTING", "actor_state:RESTARTING"),
        (ActorState.PENDING_CREATION, "PENDING_CREATION", "actor_state:PENDING_CREATION"),
        (None, "UNKNOWN", "actor_state_unknown"),
        (9999, "UNKNOWN", "actor_state_unknown"),
    ],
)
def test_get_engines_info_omits_metadata_for_non_alive_state(zero_wait_get, state, state_name, reason):
    engine = make_engine(state)
    manager, _ = make_manager([engine], [InitResult(engine_metadata())])

    result = manager.get_engines_info()

    assert engine_rows(result) == [{"rank": 0, "status": "active", "actor_state": state_name}]
    assert_issue(result, reason)
    assert zero_wait_get == []
    engine.get_url.remote.assert_not_called()
    engine.get_pid_and_node_id.remote.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "exception"])
def test_get_engines_info_unavailable_local_state_never_falls_back_to_rpc(zero_wait_get, failure):
    engine = make_engine()
    if failure == "missing":
        del engine._get_local_state
    else:
        engine._get_local_state.side_effect = RuntimeError("local state unavailable")
    manager, _ = make_manager([engine], [InitResult(engine_metadata())])

    result = manager.get_engines_info()

    assert_issue(result, "actor_state_unavailable")
    assert engine_rows(result) == [{"rank": 0, "status": "active", "actor_state": "UNKNOWN"}]
    assert zero_wait_get == []
    engine.get_url.remote.assert_not_called()
    engine.get_pid_and_node_id.remote.assert_not_called()


def test_get_engines_info_none_slot_is_complete_even_with_stale_record(zero_wait_get):
    manager, _ = make_manager([None], [InitResult(engine_metadata())])

    result = manager.get_engines_info()

    assert engine_rows(result) == [{"rank": 0, "status": "dead", "actor_state": "ABSENT"}]
    assert result["observation"] == {"complete": True, "issues": []}
    assert zero_wait_get == []


def test_get_engines_info_missing_record_is_incomplete(zero_wait_get):
    manager, _ = make_manager([make_engine()], [])

    result = manager.get_engines_info()

    assert_issue(result, "missing_record")
    assert_no_metadata(engine_rows(result)[0])
    assert zero_wait_get == []


def test_get_engines_info_unknown_state_can_recover_without_consuming_init_ref(zero_wait_get):
    engine = make_engine(None)
    ref = InitResult(engine_metadata())
    manager, _ = make_manager([engine], [ref])

    assert_issue(manager.get_engines_info(), "actor_state_unknown")
    assert ref.reads == 0
    engine._get_local_state.return_value = ActorState.ALIVE
    result = manager.get_engines_info()

    assert result["observation"]["complete"] is True
    assert engine_rows(result)[0]["url"] == ref.value["url"]
    assert ref.reads == 1


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {},
        {"url": "http://engine", "pid": 1},
        engine_metadata(url=""),
        engine_metadata(url=12),
        engine_metadata(pid=True),
        engine_metadata(pid="123"),
        engine_metadata(node_id=None),
    ],
)
def test_get_engines_info_rejects_invalid_init_metadata_without_caching(zero_wait_get, metadata):
    ref = InitResult(metadata)
    manager, group = make_manager([make_engine()], [ref])

    result = manager.get_engines_info()

    assert_issue(result, "metadata_invalid")
    assert_no_metadata(engine_rows(result)[0])
    assert group.engine_init_records[0].value is ref


@pytest.mark.parametrize("error", [GetTimeoutError("pending"), RuntimeError("init result lost")])
def test_get_engines_info_init_read_failure_is_retryable_and_preserves_peer(zero_wait_get, error):
    failed_ref = InitResult(error=error)
    ready_ref = InitResult(engine_metadata(pid=456))
    manager, group = make_manager([make_engine(), make_engine()], [failed_ref, ready_ref])

    failed = manager.get_engines_info()

    reason = "init_pending" if isinstance(error, GetTimeoutError) else "init_unreadable"
    assert_issue(failed, reason)
    assert engine_rows(failed)[1]["pid"] == 456
    assert group.engine_init_records[0].value is failed_ref
    failed_ref.error = None
    failed_ref.value = engine_metadata(pid=789)
    recovered = manager.get_engines_info()
    assert recovered["observation"]["complete"] is True
    assert [row["pid"] for row in engine_rows(recovered)] == [789, 456]
    assert ready_ref.reads == 1


def test_get_engines_info_caches_copies_and_drops_extra_fields(zero_wait_get):
    original = engine_metadata()
    original["extra"] = ["mutable"]
    ref = InitResult(original)
    engine = make_engine()
    manager, group = make_manager([engine], [ref])

    first = manager.get_engines_info()

    assert group.engine_init_records[0].value == engine_metadata()
    assert group.engine_init_records[0].value is not original
    engine_rows(first)[0]["url"] = "http://response-was-mutated"
    original["url"] = "http://init-result-was-mutated"
    ref.error = RuntimeError("ref became unreadable after cache")
    second = manager.get_engines_info()
    assert engine_rows(second)[0]["url"] == engine_metadata()["url"]
    assert second["observation"]["complete"] is True
    assert ref.reads == 1
    assert engine._get_local_state.call_count == 4
    json.dumps(second)


@pytest.mark.parametrize("cached", [False, True])
def test_get_engines_info_rechecks_alive_after_metadata_read(zero_wait_get, cached):
    engine = make_engine()
    engine._get_local_state.side_effect = [ActorState.ALIVE, ActorState.DEAD]
    value = engine_metadata() if cached else InitResult(engine_metadata())
    manager, _ = make_manager([engine], [value])

    result = manager.get_engines_info()

    assert_issue(result, "actor_dead_unreconciled")
    assert engine_rows(result) == [{"rank": 0, "status": "active", "actor_state": "DEAD"}]


def test_get_engines_info_filters_models_and_counts_multinode_rows(zero_wait_get):
    manager, group = make_manager(
        [make_engine(), make_engine(), make_engine(), make_engine()],
        [
            InitResult(engine_metadata(pid=1)),
            InitResult(engine_metadata(url=None, pid=2)),
            InitResult(engine_metadata(url="http://engine-2", pid=3)),
            InitResult(engine_metadata(url=None, pid=4, node_id="")),
        ],
    )
    group.rank_offset = 7
    group.num_gpus_per_engine = 16
    group.worker_type = "decode"
    other, _ = make_manager([make_engine()], [InitResult(pending=True)])
    manager.servers["reward"] = other.servers["actor"]

    result = manager.get_engines_info(model_name="actor")

    assert list(result["models"]) == ["actor"]
    assert result["observation"] == {"complete": True, "issues": []}
    assert result["total_engines"] == result["models"]["actor"]["total_engines"] == 4
    assert [row["rank"] for row in engine_rows(result)] == [7, 8, 9, 10]
    assert len([row for row in engine_rows(result) if "url" in row]) == 2
    assert engine_rows(result)[3]["node_id"] == ""
    assert len(zero_wait_get) == 4
    assert manager.get_engines_info(model_name="missing") == {
        "models": {},
        "total_engines": 0,
        "observation": {"complete": True, "issues": []},
    }


@pytest.mark.parametrize("replacement", ["engine", "record", "clear", "replace_both"])
@pytest.mark.parametrize("outcome", ["ready", "pending", "error"])
def test_get_engines_info_generation_change_wins_over_every_ref_outcome(zero_wait_get, replacement, outcome):
    old_engine, new_engine = make_engine(), make_engine()
    ref = InitResult(engine_metadata(pid=1))
    if outcome == "pending":
        ref.pending = True
    elif outcome == "error":
        ref.error = RuntimeError("init result unavailable")
    manager, group = make_manager([old_engine], [ref])
    old_record = group.engine_init_records[0]
    new_record = SimpleNamespace(value=InitResult(engine_metadata(pid=2)))

    def replace():
        if replacement in {"engine", "replace_both"}:
            group.all_engines[0] = new_engine
        elif replacement == "clear":
            group.all_engines[0] = None
        if replacement in {"record", "replace_both"}:
            group.engine_init_records[0] = new_record

    ref.on_read = replace
    result = manager.get_engines_info()

    assert_issue(result, "generation_changed")
    assert_no_metadata(engine_rows(result)[0])
    assert engine_rows(result)[0]["status"] == "active"
    assert new_record.value.reads == 0
    assert new_record.value.value["pid"] == 2
    if outcome == "ready":
        assert old_record.value == engine_metadata(pid=1)
    else:
        assert old_record.value is ref


@pytest.mark.parametrize("state", [ActorState.DEAD, None, ActorState.RESTARTING])
def test_get_engines_info_generation_check_also_covers_state_early_exit(zero_wait_get, state):
    engine = make_engine()
    manager, group = make_manager([engine], [InitResult(engine_metadata())])

    def read_state():
        group.all_engines[0] = None
        return state

    engine._get_local_state.side_effect = read_state
    result = manager.get_engines_info()

    assert_issue(result, "generation_changed")
    assert_no_metadata(engine_rows(result)[0])
    assert engine_rows(result)[0]["status"] == "active"
    assert zero_wait_get == []


def test_get_engines_info_a_to_none_to_b_never_reuses_a_metadata(zero_wait_get):
    manager, group = make_manager([make_engine()], [InitResult(engine_metadata(pid=1))])
    assert engine_rows(manager.get_engines_info())[0]["pid"] == 1
    group.all_engines[0] = None
    removed = manager.get_engines_info()
    assert removed["observation"]["complete"] is True
    assert_no_metadata(engine_rows(removed)[0])
    group.engine_init_records.pop(0)
    group.all_engines[0] = make_engine()
    assert_issue(manager.get_engines_info(), "missing_record")
    new_ref = InitResult(engine_metadata(pid=2), pending=True)
    group.engine_init_records[0] = SimpleNamespace(value=new_ref)
    assert_issue(manager.get_engines_info(), "init_pending")
    new_ref.pending = False
    recovered = manager.get_engines_info()
    assert recovered["observation"]["complete"] is True
    assert engine_rows(recovered)[0]["pid"] == 2


def test_get_engines_info_late_cache_write_only_mutates_captured_record(zero_wait_get):
    old_engine, new_engine = make_engine(), make_engine()
    old_ref, new_ref = InitResult(engine_metadata(pid=1)), InitResult(engine_metadata(pid=2))
    manager, group = make_manager([old_engine], [old_ref])
    new_record = SimpleNamespace(value=new_ref)

    class ReplaceDuringCacheWrite:
        value = property(lambda self: self.cached_value)

        @value.setter
        def value(self, metadata):
            group.engine_init_records[0] = new_record
            group.all_engines[0] = new_engine
            self.cached_value = metadata

    old_record = ReplaceDuringCacheWrite()
    old_record.cached_value = old_ref
    group.engine_init_records[0] = old_record
    result = manager.get_engines_info()

    assert_issue(result, "generation_changed")
    assert old_record.value == engine_metadata(pid=1)
    assert new_record.value is new_ref
    assert engine_rows(manager.get_engines_info())[0]["pid"] == 2


@pytest.mark.parametrize("initially_absent", [False, True])
def test_get_engines_info_does_not_mix_status_if_slot_changes_after_helper(
    monkeypatch, zero_wait_get, initially_absent
):
    engine = None if initially_absent else make_engine()
    manager, group = make_manager([engine], [InitResult(engine_metadata(pid=1))])
    real_helper = rollout._read_engine_metadata

    def read_then_replace(*args, **kwargs):
        row_data = real_helper(*args, **kwargs)
        group.all_engines[0] = make_engine() if initially_absent else None
        group.engine_init_records[0] = SimpleNamespace(value=InitResult(engine_metadata(pid=2)))
        return row_data

    monkeypatch.setattr(rollout, "_read_engine_metadata", read_then_replace)
    result = manager.get_engines_info()

    assert result["observation"]["complete"] is True
    row = engine_rows(result)[0]
    if initially_absent:
        assert row == {"rank": 0, "status": "dead", "actor_state": "ABSENT"}
    else:
        assert row == {"rank": 0, "status": "active", "actor_state": "ALIVE", **engine_metadata(pid=1)}


def test_get_engines_info_without_local_state_capability_fails_closed(zero_wait_get):
    engine = make_engine()
    manager, _ = make_manager([engine], [InitResult(engine_metadata())])
    manager._engine_actor_states = None

    result = manager.get_engines_info()

    assert_issue(result, "actor_state_unavailable")
    assert_no_metadata(engine_rows(result)[0])
    assert zero_wait_get == []


@pytest.mark.parametrize("missing", ["method", "enum"])
def test_local_actor_state_capability_failure_is_logged_and_nonfatal(monkeypatch, missing):
    from ray.actor import ActorHandle
    from ray.core.generated import gcs_pb2

    error_log = MagicMock()
    monkeypatch.setattr(rollout.logger, "error", error_log)
    if missing == "method":
        monkeypatch.setattr(ActorHandle, "_get_local_state", None)
    else:
        monkeypatch.setattr(gcs_pb2, "ActorTableData", SimpleNamespace())

    assert rollout._get_engine_actor_states() is None
    error_log.assert_called_once()


def test_get_engines_info_concurrent_successes_share_only_the_captured_record(zero_wait_get):
    both_reading = threading.Barrier(2)
    ref = InitResult(engine_metadata(), on_read=lambda: both_reading.wait(timeout=5))
    manager, group = make_manager([make_engine()], [ref])
    record = group.engine_init_records[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(manager.get_engines_info)
        second = pool.submit(manager.get_engines_info)
        results = [first.result(timeout=10), second.result(timeout=10)]

    assert all(result["observation"]["complete"] is True for result in results)
    assert group.engine_init_records[0] is record
    assert record.value == engine_metadata()
    assert ref.reads == 2
    engine_rows(results[0])[0]["url"] = "http://mutated-response"
    assert engine_rows(results[1])[0]["url"] == engine_metadata()["url"]
    assert engine_rows(manager.get_engines_info())[0]["url"] == engine_metadata()["url"]
    assert ref.reads == 2


def configure_actor_creation(monkeypatch, engines):
    actor_class = MagicMock()
    actor_class.options.return_value.remote.side_effect = engines
    monkeypatch.setattr(rollout.ray, "remote", MagicMock(return_value=actor_class))
    monkeypatch.setattr(rollout, "get_ray_accelerator_kwargs", lambda num_gpus: {"num_gpus": num_gpus})
    return actor_class


@pytest.mark.parametrize("mode", ["initial", "initial_external", "recover", "native_scale_out", "native_recover"])
def test_start_engines_records_original_refs_and_preserves_live_slots(monkeypatch, zero_wait_get, mode):
    live, first, second = make_engine(), make_engine(), make_engine()
    manager, group = make_manager([live, None, None], [engine_metadata(pid=1)])
    live_record = group.engine_init_records[0]
    group.engine_init_records[1] = SimpleNamespace(value=engine_metadata(pid=99))
    group.engine_init_records[2] = SimpleNamespace(value=engine_metadata(pid=98))
    group.rank_offset = 5
    group.is_scaled_out = mode in ("native_scale_out", "native_recover")
    group.skip_dcs_registration = mode == "native_scale_out"
    group.skip_router_registration = group.is_scaled_out
    if mode == "native_recover":
        group.sglang_overrides = {"host": "::1", "port": 32000}
    group.pg = (object(), list(range(8)), list(range(8)))
    group.args = SimpleNamespace(
        num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        debug_train_only=False,
        rollout_external=mode == "initial_external",
        hf_checkpoint="/test/model",
    )
    first_ref, second_ref = InitResult(pending=True), InitResult(pending=True)

    def create_engine(*args, rank, **kwargs):
        slot = rank - group.rank_offset
        assert slot not in group.engine_init_records, "old metadata must be removed before replacement creation"
        return {1: first, 2: second}[slot]

    actor_class = configure_actor_creation(monkeypatch, create_engine)
    monkeypatch.setattr(rollout, "PlacementGroupSchedulingStrategy", MagicMock())
    monkeypatch.setattr(rollout, "build_runai_streamer_env_for_load", lambda *args: {})

    def allocate(**kwargs):
        assert group.all_engines == [live, first, second], "handles must remain visible before init for cleanup"
        observed = manager.get_engines_info()
        assert observed["observation"]["issues"] == [
            {"scope": "actor/0/6", "reason": "missing_record"},
            {"scope": "actor/0/7", "reason": "missing_record"},
        ]
        return {6: {"host": "worker", "port": 30006}, 7: {"host": "worker", "port": 30007}}

    monkeypatch.setattr(rollout, "_allocate_rollout_engine_addr_and_ports_external", allocate)
    monkeypatch.setattr(
        rollout, "_allocate_rollout_engine_addr_and_ports_normal", lambda **kwargs: (allocate(**kwargs), {0: 31000})
    )
    first.init.remote.return_value = first_ref

    def submit_second(**kwargs):
        assert group.engine_init_records[1].value is first_ref, "record each ref immediately after submission"
        assert 2 not in group.engine_init_records
        return second_ref

    second.init.remote.side_effect = submit_second
    handles, cursors = group.start_engines(port_cursors={0: 30000})

    assert handles == [first_ref, second_ref]
    assert group.engine_init_records[0] is live_record
    assert group.engine_init_records[1].value is first_ref
    assert group.engine_init_records[2].value is second_ref
    assert isinstance(group.engine_init_records[1], rollout._EngineInitRecord)
    assert group.engine_init_records[1].pending_router_registration == group.is_scaled_out
    assert group.engine_init_records[2].pending_router_registration == group.is_scaled_out
    assert group.engine_init_records[1].router_url == (
        "http://[::1]:32000" if mode == "native_recover" else "http://worker:30006"
    )
    assert group.engine_init_records[2].router_url == (
        "http://[::1]:32000" if mode == "native_recover" else "http://worker:30007"
    )
    assert group.num_new_engines == 2
    assert actor_class.options.return_value.remote.call_count == 2
    assert first_ref.reads == second_ref.reads == 0
    assert cursors == ({0: 30000} if mode == "initial_external" else {0: 31000})
    for engine in [live, first, second]:
        engine.get_url.remote.assert_not_called()
        engine.get_pid_and_node_id.remote.assert_not_called()


def test_start_engines_preserves_submitted_record_on_later_synchronous_failure(monkeypatch):
    first, second = make_engine(), make_engine()
    _, group = make_manager([None, None], [])
    group.args = SimpleNamespace(
        num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        debug_train_only=False,
        rollout_external=True,
        hf_checkpoint="/test/model",
    )
    group.pg = (object(), list(range(8)), list(range(8)))
    first_ref = InitResult(pending=True)
    first.init.remote.return_value = first_ref
    second.init.remote.side_effect = RuntimeError("second init submission failed")
    configure_actor_creation(monkeypatch, [first, second])
    monkeypatch.setattr(rollout, "PlacementGroupSchedulingStrategy", MagicMock())
    monkeypatch.setattr(rollout, "build_runai_streamer_env_for_load", lambda *args: {})
    monkeypatch.setattr(rollout, "_allocate_rollout_engine_addr_and_ports_external", lambda **kwargs: {0: {}, 1: {}})

    with pytest.raises(RuntimeError, match="second init submission failed"):
        group.start_engines()

    assert group.all_engines == [first, second]
    assert group.engine_init_records[0].value is first_ref
    assert 1 not in group.engine_init_records


def prepare_external_manager(monkeypatch, *, policy="keep_partial", finalize_success=True, creation_error=False):
    manager, initial_group = make_manager([make_engine()], [engine_metadata(pid=100)])
    manager.args.scale_out_partial_success_policy = policy
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=finalize_success)
    manager._rollback_engines = AsyncMock()
    engines = [make_engine(), make_engine(), make_engine()]
    refs = [InitResult(engine_metadata(pid=i)) for i in range(3)]
    refs[1].error = RuntimeError("B init failed")
    for engine, ref in zip(engines, refs):
        engine.init.remote.return_value = ref
        engine.register_dcs.remote.side_effect = lambda: InitResult(None)
        engine.register_to_router.remote.side_effect = lambda: InitResult(True)
    if creation_error:
        configure_actor_creation(monkeypatch, [engines[0], engines[1], RuntimeError("C creation failed")])
    else:
        configure_actor_creation(monkeypatch, engines)
    request = rollout.ScaleOutRequest(
        request_id="metadata-test",
        status=rollout.ScaleOutStatus.PENDING,
        model_name="actor",
        engine_urls=["http://192.0.2.1:30000", "http://192.0.2.2:30000", "http://192.0.2.3:30000"],
    )
    return manager, request, initial_group, engines, refs


def test_external_partial_success_publishes_matching_compressed_refs(monkeypatch, zero_wait_get):
    manager, request, initial_group, engines, refs = prepare_external_manager(monkeypatch)
    real_finalize = manager._finalize_engine_group_registration
    manager._finalize_engine_group_registration = AsyncMock(wraps=real_finalize)

    asyncio.run(manager._scale_out_external(request))

    assert request.status == rollout.ScaleOutStatus.ACTIVE
    call = manager._finalize_engine_group_registration.call_args.kwargs
    assert "engines" not in call, "external must derive handles and records from one success sequence"
    assert call["engine_init_pairs"] == [(engines[0], refs[0]), (engines[2], refs[2])]
    assert manager.servers["actor"].engine_groups[0] is initial_group
    new_group = manager.servers["actor"].engine_groups[1]
    assert new_group.all_engines == [engines[0], engines[2]]
    assert list(new_group.engine_init_records) == [0, 1]
    assert new_group.engine_init_records[0].value is refs[0]
    assert new_group.engine_init_records[1].value is refs[2]
    manager._rollback_engines.assert_awaited_once_with([engines[1]])
    observed = manager.get_engines_info()
    assert observed["observation"]["complete"] is True
    assert [row["rank"] for row in engine_rows(observed, group_index=1)] == [1, 2]
    assert [row["pid"] for row in engine_rows(observed, group_index=1)] == [0, 2]
    engines[1].register_dcs.remote.assert_not_called()
    engines[1].register_to_router.remote.assert_not_called()


@pytest.mark.parametrize("failure", ["rollback_all", "finalize", "creation"])
def test_external_failure_cleans_only_owned_success_and_failure_actors(monkeypatch, failure):
    manager, request, initial_group, engines, _ = prepare_external_manager(
        monkeypatch,
        policy="rollback_all" if failure == "rollback_all" else "keep_partial",
        finalize_success=failure != "finalize",
        creation_error=failure == "creation",
    )

    asyncio.run(manager._scale_out_external(request))

    assert request.status == rollout.ScaleOutStatus.FAILED
    assert manager.servers["actor"].engine_groups == [initial_group]
    cleaned = [engine for call in manager._rollback_engines.await_args_list for engine in call.args[0]]
    expected = engines[:2] if failure == "creation" else engines
    assert len(cleaned) == len(expected)
    assert {id(engine) for engine in cleaned} == {id(engine) for engine in expected}
    assert initial_group.all_engines[0] not in cleaned
    for engine in engines:
        engine.register_to_router.remote.assert_not_called()


def test_native_finalizer_keeps_existing_init_record_identity():
    engine = make_engine()
    ref = InitResult(engine_metadata())
    manager, group = make_manager([engine], [ref])
    record = group.engine_init_records[0]
    server = manager.servers["actor"]
    server.engine_groups = []
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=True)
    engine.register_dcs.remote.side_effect = lambda: InitResult(None)
    engine.register_to_router.remote.side_effect = lambda: InitResult(True)
    request = rollout.ScaleOutRequest(
        request_id="native-metadata-test", status=rollout.ScaleOutStatus.CREATING, model_name="actor"
    )

    result = asyncio.run(
        manager._finalize_engine_group_registration(request, server, engines=[engine], engine_group=group)
    )

    assert result.success is True
    assert result.group is group
    assert server.engine_groups == [group]
    assert group.engine_init_records[0] is record
    assert record.value is ref


def prepare_sglang_init(monkeypatch, *, external=False, node_rank=0):
    from relax.backends.sglang import sglang_engine

    args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=3000, rollout_external=external)
    engine = sglang_engine.SGLangEngine(args, rank=node_rank)
    events = []

    def compute_server_args(args, rank, dist_init_addr, nccl_port, host, port, *unused_args, **unused_kwargs):
        assert host == "[2001:db8::1]"
        assert dist_init_addr == "[2001:db8::1]:30001"
        return {"node_rank": node_rank, "host": host, "port": port}, []

    monkeypatch.setattr(sglang_engine, "_compute_server_args", compute_server_args)
    monkeypatch.setattr(
        sglang_engine.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "ray-node")
    )
    engine._init_normal = MagicMock(side_effect=lambda *args, **kwargs: events.append("normal_started"))
    engine._init_external = MagicMock(side_effect=lambda *args, **kwargs: events.append("external_started"))
    engine.register_dcs = MagicMock(side_effect=lambda: events.append("dcs_registered"))
    return engine, events


@pytest.mark.parametrize("skip_dcs", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("node_rank", [0, 1])
def test_sglang_init_returns_metadata_after_success_for_both_dcs_paths(monkeypatch, skip_dcs, external, node_rank):
    engine, events = prepare_sglang_init(monkeypatch, external=external, node_rank=node_rank)

    result = engine.init("2001:db8::1:30001", 30000, None, host="2001:db8::1", skip_dcs_registration=skip_dcs)

    assert result == {
        "url": "http://[2001:db8::1]:30000" if node_rank == 0 else None,
        "pid": os.getpid(),
        "node_id": "ray-node",
    }
    assert events == ["external_started" if external else "normal_started"] + ([] if skip_dcs else ["dcs_registered"])
    assert engine.register_dcs.call_count == int(not skip_dcs)
    assert engine._init_external.call_count == int(external)
    assert engine._init_normal.call_count == int(not external)


@pytest.mark.parametrize("failure_stage", ["startup", "dcs"])
def test_sglang_init_failure_propagates_without_publishing_success_metadata(monkeypatch, failure_stage):
    engine, _ = prepare_sglang_init(monkeypatch)
    failing_method = engine._init_normal if failure_stage == "startup" else engine.register_dcs
    failing_method.side_effect = RuntimeError(f"{failure_stage} failed")
    engine.get_url = MagicMock(wraps=engine.get_url)
    engine.get_pid_and_node_id = MagicMock(wraps=engine.get_pid_and_node_id)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        engine.init("2001:db8::1:30001", 30000, None, host="2001:db8::1")

    engine.get_url.assert_not_called()
    engine.get_pid_and_node_id.assert_not_called()
    assert engine.register_dcs.call_count == int(failure_stage == "dcs")
