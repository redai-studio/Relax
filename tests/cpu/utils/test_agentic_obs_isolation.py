# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only fault injection: no Ray cluster, torch, Megatron or GPU
required."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import subprocess
import sys
import threading
import time
import weakref
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from relax.agentic.observability import SessionObservation, ShardTrajectoryRecorder, shard_generation_args
from relax.agentic.stage_recorder import StageRecorder
from relax.utils.agentic_obs import AsyncJsonlAppender, merge_train_updates, writers
from relax.utils.agentic_obs import train_hooks as hooks
from relax.utils.agentic_obs.records import UNASSIGNED_SAMPLE_EXPORT_ID, ExportStageRecord, encode_group_ids
from tests.cpu.agentic_obs_args import make_agentic_obs_args


@pytest.fixture(autouse=True)
def drain():
    yield
    hooks.close_train_appenders()
    writers.close_observation_writers()


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("failure", [None, "validation", "compaction", "routing"])
def test_session_export_compacts_incrementally_before_publishing(observe, failure):
    path = Path(__file__).resolve().parents[3] / "relax/agentic/session/service.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgenticSessionShard")
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_session_export_transport"
    ]
    cls.bases, cls.decorator_list = [], []
    hashes = iter(["leaf", "missing" if failure == "validation" else "leaf", "leaf", "leaf"])
    scope = {
        "copy": copy,
        "cast": lambda typ, value: value,
        "SessionForest": object,
        "SessionExportTransport": NS,
        "_NonFinalizableExportError": ValueError,
        "_merge_export_metadata": lambda target, patch: target.update(patch),
        "_messages_tools_template_state_hash": lambda *args: next(hashes),
        "normalize_template_kwargs": lambda value: value,
    }
    exec("from __future__ import annotations\n" + ast.unparse(cls), scope)
    shard = scope["AgenticSessionShard"]()
    shard.args = NS(use_rollout_routing_replay=failure == "routing", num_layers=1, moe_router_topk=1)
    shard._generation_backend = NS(compiler=NS(apply_chat_template_kwargs={}))
    live_samples, payloads, recorded = [], [], []

    class Sample:
        def __init__(self, metadata):
            self.metadata = metadata

    def build_sample(**kwargs):
        # Assignment can still own the preceding Sample while constructing this one.
        assert sum(ref() is not None for ref in live_samples) <= 1
        sample = Sample(kwargs["metadata"])
        live_samples.append(weakref.ref(sample))
        return sample

    def compact(*, name, sample):
        if failure == "compaction" and payloads:
            raise ValueError("compaction failed")
        payload = {
            "name": name,
            "sample_payload": {
                "sample_export_id": sample.sample_export_id,
                "metadata": copy.deepcopy(sample.metadata),
                "tokens": [1, 2],
                "rollout_routed_experts": None,
            },
        }
        payloads.append(payload)
        return payload

    def record(**kwargs):
        assert len(payloads) == 4  # All exports must succeed before the first record.
        assert not any(ref() is not None for ref in live_samples)
        if observe:
            recorded.append(kwargs)

    shard._build_sample, shard._export_payload = build_sample, compact
    shard._trajectory_recorder = NS(next_sample_export_id=lambda: len(payloads), record=record)
    session = NS(
        forest=NS(nodes_by_hash={"leaf": None}), group=NS(group_id="g", rollout_mode="train"), observation=NS()
    )
    records = tuple({"name": str(i), "metadata": {"export": i}, "messages": []} for i in range(4))

    def export():
        return shard._build_session_export_transport(
            session=session, reward=None, metadata={}, output_records=records, finalize_events={}
        )

    if failure is not None:
        with pytest.raises(ValueError):
            export()
        assert not recorded
    else:
        transport = export()
        assert len(transport.exports) == 4
        assert session.observation.export_ids == list(range(4))
        assert [row["metadata"]["export"] for row in recorded] == (list(range(4)) if observe else [])


@pytest.mark.parametrize("cancelled", [False, True])
def test_group_reward_observation_preserves_results_and_cancellation(args, cancelled):
    from contextlib import nullcontext

    path = Path(__file__).resolve().parents[3] / "relax/agentic/pipeline/reward.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardDomain")
    cls.body = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "score_group"]
    cls.bases, cls.decorator_list = [], []

    async def scenario():
        entered = asyncio.Event()

        async def score(rm_args, samples):
            entered.set()
            if cancelled:
                await asyncio.Event().wait()
            return [1.0, 2.0]

        scope = {
            "time": time,
            "nullcontext": nullcontext,
            "batched_async_rm": score,
            "mark_sample_agentic_event": lambda *args: None,
        }
        exec("from __future__ import annotations\n" + ast.unparse(cls), scope)
        domain = scope["RewardDomain"]()
        domain.args = args
        group = NS(samples=[NS(sample_export_id=export_id, reward=None) for export_id in (7, 8)])
        recorder = StageRecorder(args)
        task = asyncio.create_task(
            domain.score_group(
                group,
                observe_reward=lambda samples: recorder.observe(
                    samples, rollout_id=1, group_id="g", stage="reward", operation="rm"
                ),
            )
        )
        await entered.wait()
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert [sample.reward for sample in group.samples] == ([None, None] if cancelled else [1.0, 2.0])
        await recorder.aclose()

    asyncio.run(scenario())
    rows = [
        json.loads(line)
        for line in (Path(args.dump_agentic_obs) / "run/stages/rollout_00000001.jsonl").read_text().splitlines()
    ]
    ends = [row for row in rows if row["status"] != "start"]
    assert len(ends) == 2
    assert {row["status"] for row in ends} == {"cancelled" if cancelled else "completed"}


@pytest.fixture
def args(tmp_path):
    return make_agentic_obs_args(tmp_path, run_id="run")


@pytest.mark.parametrize("fleet_rebuild", [False, True])
def test_recreated_shard_recorder_keeps_export_ids_unique_within_run(args, monkeypatch, fleet_rebuild):
    initial_args = replacement_args = args
    if fleet_rebuild:
        initial_args = shard_generation_args(args)
        replacement_args = shard_generation_args(args)
        # The actor workers must use the generation assigned by the driver.
        monkeypatch.setattr("relax.agentic.observability._SHARD_GENERATIONS", iter(()))

    first = ShardTrajectoryRecorder(initial_args, shard_ordinal=0)
    before = {first.next_sample_export_id() for _ in range(3)}
    asyncio.run(first.aclose())

    replacement = ShardTrajectoryRecorder(replacement_args, shard_ordinal=0)
    after = {replacement.next_sample_export_id() for _ in range(3)}
    asyncio.run(replacement.aclose())

    assert len(before | after) == 6
    assert all(0 <= export_id < 2**63 for export_id in before | after)


@pytest.fixture
def mpu(monkeypatch):
    state = NS(rank=0, size=2)
    api = NS(
        get_tensor_model_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda **kw: True,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_world_size=lambda: 1,
        get_data_parallel_rank=lambda **kw: state.rank,
        get_data_parallel_world_size=lambda **kw: state.size,
    )
    monkeypatch.setitem(sys.modules, "megatron", NS())
    monkeypatch.setitem(sys.modules, "megatron.core", NS(mpu=api))
    monkeypatch.setattr(hooks, "_UPDATE_SEQUENCES", {})
    monkeypatch.setattr(hooks, "_PRODUCER_STATUS", {})
    return state


def read_rows(path):
    return [json.loads(line) for file in path.glob("*.jsonl") for line in file.read_text().splitlines()]


def test_worker_bounds_inflight_bytes_and_drops_tail_at_deadline():
    entered, release = threading.Event(), threading.Event()
    discarded, written = [], []
    worker = writers.BoundedWorker(capacity=3, max_bytes=10)

    def blocked():
        entered.set()
        assert release.wait(5)

    try:
        assert worker.submit(blocked, size_bytes=6)
        assert entered.wait(1)
        assert not worker.submit(lambda: None, size_bytes=5)
        assert worker.submit(written.append, "tail", size_bytes=4, on_discard=lambda: discarded.append(True))
        before = time.monotonic()
        assert not worker.close(0.02)
        assert time.monotonic() - before < 0.25
        assert discarded == [True] and worker._pending == 1 and worker._bytes == 6
        assert not worker.submit(lambda: None)
    finally:
        release.set()
        assert worker.close(1)
    assert written == []


def test_process_worker_limit_bounds_repeated_stalled_recorders(monkeypatch):
    monkeypatch.setattr(writers, "_WORKER_SLOTS", threading.BoundedSemaphore(1))
    entered, release = threading.Event(), threading.Event()
    first, second = writers.BoundedWorker(), writers.BoundedWorker()
    try:
        assert first.submit(lambda: (entered.set(), release.wait(5)))
        assert entered.wait(1)
        assert not second.submit(lambda: None)
        assert second._thread is None and second._pending == 0
    finally:
        release.set()
        first.close()
        second.close()


@pytest.mark.parametrize("phase", ["write", "status"])
def test_stalled_writer_cannot_hold_process_exit(phase):
    script = """
import threading
from relax.utils.agentic_obs.writers import BoundedWorker
entered = threading.Event()
release = threading.Event()
def block():
    entered.set()
    release.wait()
worker = BoundedWorker(on_close=block if PHASE == 'status' else None)
worker.submit(block if PHASE == 'write' else lambda: None)
if PHASE == 'status':
    worker.request_close()
assert entered.wait(2)
""".replace("PHASE", repr(phase))
    result = subprocess.run([sys.executable, "-c", script], timeout=4, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_immutable_snapshot_and_write_failure_are_isolated(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    appender = AsyncJsonlAppender(tmp_path / "records.jsonl")
    assert appender._worker.submit(lambda: (entered.set(), release.wait(5)))
    assert entered.wait(1)
    payload = {"sample_export_id": 2**60 + 1, "nested": {"mask": [1, 0]}}
    assert appender.submit(payload)
    payload["nested"]["mask"][0] = 0
    release.set()
    appender.close()
    row = json.loads((tmp_path / "records.jsonl").read_text())
    assert row["nested"]["mask"] == [1, 0] and row["sample_export_id"] == str(2**60 + 1)
    monkeypatch.setattr(writers.JsonlAppender, "append_many", lambda *a: (_ for _ in ()).throw(OSError("disk failed")))
    failing = AsyncJsonlAppender(tmp_path / "failed.jsonl")
    assert failing.submit({"id": 1})
    failing.close()
    assert failing._worker.error_count == 1


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
def test_stage_error_or_cancellation_survives_stalled_disk(args, monkeypatch, failure):
    entered, release = threading.Event(), threading.Event()
    recorder = StageRecorder(args)
    sample = NS(sample_export_id=1, reward=1.0)
    error = failure("business failure")
    original = writers.JsonlAppender.append_many

    def blocked(self, rows):
        entered.set()
        assert release.wait(5)
        original(self, rows)

    monkeypatch.setattr(writers.JsonlAppender, "append_many", blocked)

    async def scenario():
        with pytest.raises(failure) as caught:
            async with recorder.observe([sample], rollout_id=1, group_id="g", stage="reward"):
                assert await asyncio.to_thread(entered.wait, 1)
                raise error
        assert caught.value is error
        await asyncio.wait_for(recorder._worker.aclose(0.02), 0.2)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        recorder.close()


def test_record_construction_error_does_not_mask_business_exception(args, monkeypatch):
    recorder = StageRecorder(args)
    monkeypatch.setattr(
        ExportStageRecord, "to_dict", lambda self: (_ for _ in ()).throw(ValueError("bad observation"))
    )
    error = RuntimeError("reward failed")

    async def scenario():
        with pytest.raises(RuntimeError) as caught:
            async with recorder.observe(
                [NS(sample_export_id=1, reward=1)], rollout_id=1, group_id="g", stage="reward"
            ):
                raise error
        assert caught.value is error

    asyncio.run(scenario())
    recorder.close()
    shard = ShardTrajectoryRecorder(args, shard_ordinal=0)
    shard.record_session(
        SessionObservation(started_at_utc="bad timestamp"),
        session_id="s",
        group_id="g",
        rollout_mode="train",
        rollout_id=1,
        stage="rollout",
        status="completed",
        is_terminal=True,
    )
    asyncio.run(shard.aclose())


def test_log_persistence_never_delays_cleanup_and_owns_snapshot(tmp_path, monkeypatch):
    from relax.agentic.runner import ipc

    entered, release = threading.Event(), threading.Event()
    source, target = tmp_path / "temporary.log", tmp_path / "logs/s.log"
    source.write_bytes(b"x" * (1024 * 1024 + 7) + b"tail")
    original = ipc._persist_log_tail

    def blocked(payload, path):
        entered.set()
        assert release.wait(5)
        original(payload, path)

    monkeypatch.setattr(ipc, "_persist_log_tail", blocked)
    process = object.__new__(ipc.ManagedAgentProcess)
    process._observation_log_path, process.observation_log_ref = target, None

    async def cleanup():
        await process._persist_observation_log(source)
        source.unlink()  # TemporaryDirectory cleanup is free to run immediately.

    try:
        asyncio.run(asyncio.wait_for(cleanup(), 0.2))
        assert entered.wait(1) and not source.exists() and not target.exists()
        assert process.observation_log_ref == "logs/s.log"
    finally:
        release.set()
        writers.close_observation_writers()
    assert target.stat().st_size == 1024 * 1024 and target.read_bytes().endswith(b"tail")


def test_train_local_facts_survive_blocked_inputs_and_cleanup_deadline(args, mpu, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = hooks._write_update

    def blocked(state):
        entered.set()
        assert release.wait(5)
        original(state)

    monkeypatch.setattr(hooks, "_write_update", blocked)
    try:
        for rank in range(2):
            mpu.rank = rank
            with hooks.observe_train_step(args, 1, 0):
                hooks._CURRENT.get().group_ids.add(f"g{rank}")
                hooks.record_optimizer_result(True)
        assert entered.wait(1)
        stage_drained = threading.Event()
        assert hooks._STAGE_EXECUTOR.submit(stage_drained.set)
        assert stage_drained.wait(1)
        before = time.monotonic()
        hooks.close_train_appenders(0.03)
        assert time.monotonic() - before < 0.3
        rows = read_rows(Path(args.dump_agentic_obs) / "run/stages")
        (merged,) = merge_train_updates(rows, [])
        assert merged["status"] == "committed" and merged["group_ids"] == ["g0", "g1"]
        assert not merged["input_complete"]
    finally:
        release.set()
        hooks.close_train_appenders()
    # Queued input was discarded; both its reservation and the inflight one return.
    assert hooks._PENDING.acquire(blocking=False)
    hooks._PENDING.release()


def test_train_observer_failure_preserves_business_exception_and_next_update(args, mpu, monkeypatch):
    error = RuntimeError("optimizer failed")
    monkeypatch.setattr(ExportStageRecord, "to_dict", lambda self: (_ for _ in ()).throw(ValueError("bad event")))
    with pytest.raises(RuntimeError) as caught:
        with hooks.observe_train_step(args, 1, 0):
            raise error
    assert caught.value is error and hooks._CURRENT.get() is None
    with hooks.observe_train_step(args, 1, 1):
        hooks.record_optimizer_result(True)


def boundaries(update="actor:1:0", sequence=0, size=2):
    rows = []
    for rank in range(size):
        for index, status in enumerate(("start", "completed"), 1):
            rows.append(
                {
                    "run_id": "run",
                    "train_generation_id": "generation",
                    "update_id": update,
                    "scope": "update_fragment",
                    "stage": "train",
                    "rollout_id": 1,
                    "step_id": 0,
                    "update_sequence": sequence,
                    "dp_rank": rank,
                    "dp_world_size": size,
                    "status": status,
                    "stage_id": f"{update}:{rank}",
                    "event_seq": index,
                    "optimizer_successful": status != "start",
                    "input_complete": True,
                    "groups_finalized": True,
                    "group_ids": [f"g{rank}"],
                }
            )
    return rows


@pytest.mark.parametrize("damage", ["missing", "conflict", "identity", "sequence", "unknown_outcome"])
def test_merge_requires_consistent_complete_dp_outcomes(damage):
    rows = boundaries()
    if damage == "missing":
        rows.pop()
    elif damage == "conflict":
        rows.append({**rows[-1], "stage_id": "another-attempt"})
    elif damage == "identity":
        for row in rows[2:]:
            row["step_id"] = 9
    elif damage == "sequence":
        rows[-1]["update_sequence"] = 2
    else:
        rows[-1].pop("optimizer_successful")
    (result,) = merge_train_updates(rows, [])
    assert result["status"] == "unknown" and result["optimizer_step"] is None


def test_merge_missing_input_and_sequence_gap_never_invent_completeness():
    rows = boundaries() + boundaries("actor:2:0", sequence=2)
    result = merge_train_updates(rows, [])
    assert [r["optimizer_step"] for r in result] == [1, None]
    assert all(not r["input_complete"] for r in result)
    rows += boundaries("conflicting-update", sequence=0)[:-1]
    assert all(r["optimizer_step"] is None for r in merge_train_updates(rows, []))


def test_merge_idempotent_success_skip_and_legacy():
    rows = boundaries() + boundaries("actor:1:1", sequence=1) + boundaries("actor:2:0", sequence=2)
    rows[7]["optimizer_successful"] = False
    rows[7]["status"] = "failed"
    inputs = [
        {
            "run_id": "run",
            "update_id": update,
            "train_generation_id": "generation",
            "stage_id": f"{update}:{rank}",
            "dp_rank": rank,
            "dp_world_size": 2,
            "input_complete": True,
        }
        for update in ("actor:1:0", "actor:1:1", "actor:2:0")
        for rank in range(2)
    ]
    result = merge_train_updates(rows + copy.deepcopy(rows), inputs + copy.deepcopy(inputs))
    assert [r["status"] for r in result] == ["committed", "failed", "committed"]
    assert [r["optimizer_step"] for r in result] == [1, None, 2]
    assert all(r["input_complete"] for r in result)
    legacy = {"scope": "update", "status": "committed", "optimizer_step": 9}
    assert merge_train_updates([legacy], []) == [legacy]


def test_dropping_protected_group_progress_and_release_wake_cleanup():
    # Extract the actual methods to isolate Ray imports; no copied implementation.
    path = Path(__file__).resolve().parents[3] / "relax/agentic/pipeline/runtime.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RuntimeDomain")
    names = {
        "_apply_shard_progress",
        "_wait_for_no_protected_groups",
        "_drop_group_and_release",
        "_release_shard_group",
    }
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    cls.bases, cls.decorator_list = [], []
    scope = {"asyncio": asyncio, "cast": lambda typ, value: value}
    exec("from __future__ import annotations\n" + ast.unparse(cls), scope)

    async def scenario():
        runtime = scope["RuntimeDomain"]()
        runtime._state_changed = asyncio.Event()
        runtime._resident_group_permits = asyncio.Semaphore(0)
        runtime._start_shard_observers = lambda: None
        observer = asyncio.create_task(asyncio.sleep(10))
        shard = NS(group_streams={}, revision_changed=asyncio.Event(), observer_task=observer)
        stream = NS(
            group_id="g",
            shard=shard,
            protected=True,
            dropping=False,
            stop_reason=None,
            session_results=(),
            first_request_barrier=asyncio.get_running_loop().create_future(),
        )
        shard.group_streams["g"] = stream
        runtime._shards = [shard]

        async def remote_drop(*args, **kw):
            progress = NS(
                group_id="g", ready_for_lease=True, interrupted=False, protected=False, takeable_session_ids=()
            )
            runtime._apply_shard_progress(shard, NS(revision=1, groups=[progress]))
            assert not stream.first_request_barrier.done()
            await asyncio.sleep(0)
            runtime._apply_shard_progress(shard, NS(revision=2, groups=[]))
            await asyncio.sleep(0)

        shard.handle = NS(drop_group=NS(remote=remote_drop))
        waiter = asyncio.create_task(runtime._wait_for_no_protected_groups())
        await asyncio.sleep(0)
        await runtime._drop_group_and_release(stream)
        await asyncio.wait_for(waiter, 0.2)
        assert not shard.group_streams and runtime._state_changed.is_set()
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)

    asyncio.run(scenario())


def test_six_streams_keep_prompt_session_export_and_update_links(args, mpu):
    from relax.agentic.inventory import InventoryRecorder

    mpu.size = 1
    export_id = (15 << 56) | 1
    sample = NS(
        sample_export_id=export_id,
        prompt="question",
        metadata={},
        group_index=2,
        index=5,
        agentic_prompt_source=None,
        reward=0.8,
        session_id="s",
    )
    node = NS(
        messages_delta=[{"role": "assistant", "content": "answer"}],
        train_token_delta=[1, 2],
        kind="resp",
        state_hash="leaf",
        weight_version_delta=["v1"],
    )
    forest = NS(session_id="s", group_index=2, index=5, label="gold", lineage=lambda state: [node])
    inventory = InventoryRecorder(args, "train")
    shard = ShardTrajectoryRecorder(args, shard_ordinal=15)
    stage = StageRecorder(args)
    inventory.record_group("g", [sample], ["s"])
    shard.record(
        forest=forest,
        state_hash="leaf",
        sample_export_id=sample.sample_export_id,
        metadata=sample.metadata,
        export_name=None,
        group_id="g",
        rollout_mode="train",
    )

    async def observe():
        await stage.record_queued([sample], rollout_id=1, partition_id=0, group_ids={export_id: "g"})
        shard.record_session(
            SessionObservation(stage_id="session:s:rollout", export_ids=[export_id]),
            session_id="s",
            group_id="g",
            rollout_mode="train",
            rollout_id=1,
            stage="rollout",
            status="completed",
            is_terminal=True,
        )
        await stage.aclose()
        await inventory.aclose()
        await shard.aclose()

    asyncio.run(observe())
    with hooks.observe_train_step(args, 1, 0):
        state = hooks._CURRENT.get()
        state.group_ids.add("g")
        state.snapshots.append(
            hooks._Snapshot([(export_id, "g", 2, 3)], NS(tolist=lambda: [0.5, 0.5, 0.8, 1, 0]), None, None, True, True)
        )
        hooks.record_optimizer_result(True)
    hooks.close_train_appenders()
    writers.close_observation_writers()
    root = Path(args.dump_agentic_obs) / "run"
    rows = {
        name: read_rows(root / name) for name in ("prompts", "groups", "sessions", "trajectories", "stages", "train")
    }
    assert all(rows.values())
    assert rows["prompts"][0]["prompt_id"] == rows["groups"][0]["prompt_id"] == rows["sessions"][0]["prompt_id"]
    assert rows["sessions"][0]["session_id"] == rows["trajectories"][0]["session_id"] == "s"
    assert (
        rows["trajectories"][0]["sample_export_id"]
        == rows["train"][0]["samples"][0]["sample_export_id"]
        == str(export_id)
    )
    assert rows["train"][0]["samples"][0]["loss_mask"] == [1, 0]
    (update,) = merge_train_updates(rows["stages"], rows["train"])
    assert update["status"] == "committed" and update["input_complete"] and update["group_ids"] == ["g"]
    assert update["started_at"] <= update["occurred_at"]


@pytest.mark.parametrize("ids", [None, [7], [2**70, 7], [7.0, 7], [99, 7], [8, 7]])
def test_custom_converter_identity_failure_keeps_business_rows(ids):
    # Run the real converter branch without importing its unrelated GPU stack.
    path = Path(__file__).resolve().parents[3] / "relax/utils/utils.py"
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "convert_samples_to_train_data"
    )
    converted = {"total_lengths": [2, 3], "sample_indices": [5, 6], "sample_export_ids": ids}
    scope = {
        "torch": NS(Tensor=type("Tensor", (), {})),
        "TensorDict": type("TensorDict", (), {}),
        "UNASSIGNED_SAMPLE_EXPORT_ID": UNASSIGNED_SAMPLE_EXPORT_ID,
        "encode_group_ids": encode_group_ids,
        "agentic_obs_enabled": lambda args: True,
        "load_function": lambda path: lambda args, samples: converted,
        "logger": writers.logger,
    }
    exec("from __future__ import annotations\n" + ast.unparse(function), scope)
    samples = [NS(sample_export_id=7, agentic_group_id="a"), NS(sample_export_id=8, agentic_group_id="b")]
    result = scope["convert_samples_to_train_data"](
        Namespace(custom_convert_samples_to_train_data_path="custom"), samples
    )
    assert result is converted and result["total_lengths"] == [2, 3] and result["sample_indices"] == [5, 6]
    assert result["sample_export_ids"] == ([8, 7] if ids == [8, 7] else [-1, -1])
    assert result["agentic_group_ids"] == ([[98, 0], [97, 0]] if ids == [8, 7] else [[0], [0]])


def test_writer_setup_failure_does_not_leak_train_reservations(args, mpu, monkeypatch):
    hooks.close_train_appenders()
    pending = threading.BoundedSemaphore(1)
    monkeypatch.setattr(hooks, "_PENDING", pending)
    monkeypatch.setattr(hooks, "BoundedWorker", lambda **kw: (_ for _ in ()).throw(OSError("cannot start writer")))
    for step in range(3):
        with hooks.observe_train_step(args, 1, step):
            assert hooks._CURRENT.get().reserved
            hooks.record_optimizer_result(True)
    assert pending.acquire(blocking=False)
    pending.release()


def test_input_capture_failure_keeps_local_optimizer_outcome(args, mpu, monkeypatch):
    mpu.size = 1
    monkeypatch.setattr(hooks, "_capture", lambda *a: (_ for _ in ()).throw(ValueError("bad observation identity")))
    with hooks.observe_train_step(args, 1, 0):
        hooks.capture_training_input({})
        hooks.record_optimizer_result(True)
    hooks.close_train_appenders()
    root = Path(args.dump_agentic_obs) / "run"
    (result,) = merge_train_updates(read_rows(root / "stages"), read_rows(root / "train"))
    assert result["status"] == "committed"
    assert not result["groups_finalized"] and not result["input_complete"]


@pytest.mark.parametrize("input_attempt", [None, "previous-attempt"])
def test_replayed_update_never_reuses_previous_input(input_attempt):
    stages = boundaries(size=1)
    inputs = [
        {
            "run_id": "run",
            "train_generation_id": "generation",
            "update_id": "actor:1:0",
            "dp_rank": 0,
            "dp_world_size": 1,
            "stage_id": input_attempt,
            "input_complete": True,
        }
    ]
    (update,) = merge_train_updates(stages, inputs)
    assert update["status"] == "committed"
    assert not update["input_complete"]


def test_restarted_dp_generations_cannot_be_combined_into_commit():
    old = [{**row, "train_generation_id": "old"} for row in boundaries()[:2]]
    new = [{**row, "train_generation_id": "new"} for row in boundaries()[2:]]
    updates = merge_train_updates(old + new, [])
    assert updates and all(update["status"] == "unknown" for update in updates)
    assert all(update["optimizer_step"] is None for update in updates)


def test_actor_group_distributes_one_generation_without_changing_training_args():
    from relax.utils.agentic_obs.config import train_generation_args

    path = Path(__file__).resolve().parents[3] / "relax/distributed/ray/actor_group.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayTrainGroup")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "async_init")
    scope = {"train_generation_args": train_generation_args}
    exec(ast.unparse(method), scope)
    seen = []

    def initialize(args, role, **kwargs):
        seen.append(args)
        assert role == "actor" and kwargs == {"with_ref": True, "with_opd_teacher": False}
        return len(seen)

    group = NS(_actor_handlers=[NS(init=NS(remote=initialize)) for _ in range(2)])
    args = Namespace(dump_agentic_obs="output", learning_rate=0.01, resource={"actor": [1, 2]})
    original = copy.deepcopy(vars(args))
    assert scope["async_init"](group, args, "actor", with_ref=True) == [1, 2]
    assert seen[0] is seen[1] and seen[0] is not args
    generation = seen[0].agentic_obs_train_generation_id
    assert generation
    assert {key: value for key, value in vars(seen[0]).items() if key != "agentic_obs_train_generation_id"} == original
    assert vars(args) == original and group.args is args
    scope["async_init"](group, args, "actor", with_ref=True)
    assert seen[2].agentic_obs_train_generation_id != generation
    assert seen[2] is seen[3]
    args.dump_agentic_obs = None
    scope["async_init"](group, args, "actor", with_ref=True)
    assert seen[4] is seen[5] is args


def test_merge_restarted_generations_keep_independent_counts_and_inputs():
    stages, inputs = [], []
    for generation in ("old", "new"):
        for row in boundaries(size=1):
            stages.append({**row, "train_generation_id": generation, "stage_id": generation})
        inputs.append(
            {
                "run_id": "run",
                "train_generation_id": generation,
                "update_id": "actor:1:0",
                "dp_rank": 0,
                "dp_world_size": 1,
                "stage_id": generation,
                "input_complete": True,
            }
        )
    updates = merge_train_updates(stages, inputs)
    assert {update["train_generation_id"] for update in updates} == {"old", "new"}
    assert all(update["status"] == "committed" and update["input_complete"] for update in updates)
    assert all(update["optimizer_step"] == 1 for update in updates)
    # A legacy/input fragment without a generation never completes a v3 update.
    inputs[1].pop("train_generation_id")
    updates = merge_train_updates(stages, inputs)
    assert {update["train_generation_id"]: update["input_complete"] for update in updates} == {
        "old": True,
        "new": False,
    }


def test_merge_without_shared_generation_is_unknown():
    stages = boundaries(size=1)
    for record in stages:
        record.pop("train_generation_id")
    (update,) = merge_train_updates(stages, [])
    assert update["status"] == "unknown" and update["optimizer_step"] is None


@pytest.mark.parametrize("finished_worker", [False, True])
def test_background_concurrent_submissions_share_one_worker(monkeypatch, finished_worker):
    asyncio.run(writers.close_background())
    monkeypatch.setattr(writers, "_BACKGROUND", None)
    if finished_worker:
        writers.submit_background(lambda: None)
        asyncio.run(writers.close_background())
    original = writers.BoundedWorker
    created, accepted = [], []
    ready = threading.Barrier(5)
    entered, overlap, release = threading.Event(), threading.Event(), threading.Event()

    def create(**kwargs):
        worker = original(**kwargs)
        created.append(worker)
        entered.set()
        if len(created) > 1:
            overlap.set()
        assert release.wait(2)
        return worker

    def submit():
        ready.wait(2)
        accepted.append(writers.submit_background(lambda: None))

    monkeypatch.setattr(writers, "BoundedWorker", create)
    threads = [threading.Thread(target=submit) for _ in range(4)]
    try:
        for thread in threads:
            thread.start()
        ready.wait(2)
        assert entered.wait(1)
        overlap.wait(0.1)  # Let competing callers reach the paused constructor.
    finally:
        release.set()
        for thread in threads:
            thread.join(2)
        asyncio.run(writers.close_background())
        for worker in created:
            worker.close()
    assert accepted == [True] * 4
    assert len(created) == 1
    assert all(not worker._thread.is_alive() for worker in created)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("rejection", ["bytes", "capacity", "snapshot"])
def test_appender_persists_status_when_every_record_is_dropped(tmp_path, asynchronous, rejection):
    path = tmp_path / "records.jsonl"
    kwargs = {"max_bytes": 1} if rejection == "bytes" else {"capacity": 0} if rejection == "capacity" else {}
    appender = AsyncJsonlAppender(path, **kwargs)

    class BadRecord:
        def to_dict(self):
            raise ValueError("cannot snapshot")

    assert not appender.submit(BadRecord() if rejection == "snapshot" else {"id": 1})
    if asynchronous:
        asyncio.run(appender.aclose())
    else:
        appender.close()
    assert not path.exists()
    assert json.loads(path.with_suffix(".status.json").read_text()) == {
        "records_written": 0,
        "write_errors": 0,
        "records_dropped": 1,
    }
    appender.close()
    assert json.loads(path.with_suffix(".status.json").read_text())["records_dropped"] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dropped_only_appender_close_does_not_wait_for_storage(tmp_path, monkeypatch, asynchronous):
    original = writers.JsonlAppender
    entered, release = threading.Event(), threading.Event()
    producer = threading.get_ident()
    path = tmp_path / "records.jsonl"

    def blocked(*args, **kwargs):
        assert threading.get_ident() != producer
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(writers, "JsonlAppender", blocked)
    appender = AsyncJsonlAppender(path, max_bytes=1)
    assert not appender.submit({"id": 1})
    try:
        before = time.monotonic()
        if asynchronous:
            asyncio.run(asyncio.wait_for(appender.aclose(0.02), 0.2))
        else:
            appender.close(0.02)
        assert time.monotonic() - before < 0.3
        assert entered.wait(1)
        assert not appender._worker.done.is_set()
    finally:
        release.set()
        appender.close()
    assert json.loads(path.with_suffix(".status.json").read_text())["records_dropped"] == 1


def test_close_only_worker_respects_process_limit(monkeypatch):
    monkeypatch.setattr(writers, "_WORKER_SLOTS", threading.BoundedSemaphore(0))
    closed = threading.Event()
    worker = writers.BoundedWorker(on_close=closed.set)
    assert not worker.submit(lambda: None)
    assert worker.close(0.02)
    assert worker._thread is None and not closed.is_set()


def test_unused_appender_does_not_create_status(tmp_path):
    path = tmp_path / "unused.jsonl"
    appender = AsyncJsonlAppender(path)
    appender.close()
    assert not path.exists() and not path.with_suffix(".status.json").exists()


def test_session_sampling_config_preserves_falsy_explicit_values() -> None:
    """None is dropped, but explicit False / 0 / empty-list must survive.

    Regression guard: the filter in session_sampling_config is `value is not
    None`; a future change to a truthiness check would silently drop False and
    empty-list overrides, changing sampling semantics.
    """
    from relax.agentic.inventory import session_sampling_config

    args = Namespace(
        rollout_temperature=0.0,
        rollout_top_p=1.0,
        rollout_top_k=0,  # explicit greedy top-1, not a missing value
        rollout_max_response_len=None,  # unset -> must be dropped
        rollout_stop=[],  # explicit empty list -> must be kept
        rollout_stop_token_ids=[],
        rollout_skip_special_tokens=False,  # explicit False -> must be kept
        sglang_enable_deterministic_inference=False,
    )
    cfg = session_sampling_config(args, None, position=0)

    assert cfg["temperature"] == 0.0  # explicit float 0 kept
    assert cfg["top_k"] == 0
    assert cfg["skip_special_tokens"] is False
    assert cfg["stop"] == []
    assert cfg["stop_token_ids"] == []
    assert "max_new_tokens" not in cfg
