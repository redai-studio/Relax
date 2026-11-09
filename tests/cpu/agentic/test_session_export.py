# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SessionShard export: off-loop assembly, content equivalence, lifecycle."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from relax.agentic.pipeline import (
    _TRAINING_ARTIFACT_ARRAY_FIELDS,
    RuntimeGroupError,
    SessionSpec,
    TrainingFieldArtifact,
)
from relax.agentic.pipeline.runtime import BackendGenerateResult, EncodedMessages
from relax.agentic.runner.ipc import SessionOutput
from relax.agentic.session import service
from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.types import Sample


_SYSTEM = {"role": "system", "content": "test agent"}


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _task(session_id: str) -> dict[str, Any]:
    return {"role": "user", "content": f"task for {session_id}"}


def _observation(session_id: str, turn: int) -> dict[str, Any]:
    return {"role": "user", "content": f"observation {turn} for {session_id}"}


def _response_text(backend_request_id: str) -> str:
    return f"answer to {backend_request_id}"


def _logprobs(count: int) -> list[float]:
    return [-(i + 1) / 64 for i in range(count)]


class _CharTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class _FakeCompiler:
    """Message delta -> char ids of its sorted JSON; no chat template."""

    def __init__(self) -> None:
        self.apply_chat_template_kwargs: dict[str, Any] = {}
        self.processor = None

    async def encode_messages(self, messages, **_):
        ids = _chars(json.dumps(messages, sort_keys=True))
        return EncodedMessages(train_prompt_ids=ids, backend_prompt_ids=list(ids))

    encode_observation_delta = encode_messages


class _FakeBackend:
    def __init__(self) -> None:
        self.tokenizer = _CharTokenizer()
        self.compiler = _FakeCompiler()

    async def generate(self, *, request_id, **_):
        tokens = _chars(_response_text(request_id))
        return BackendGenerateResult(
            new_tokens=tokens,
            new_log_probs=_logprobs(len(tokens)),
            finish_type="stop",
            meta_info={"weight_version": "3", "finish_reason": {"type": "stop"}},
            elapsed=0.0,
        )

    async def abort_request(self, request_id):
        del request_id

    async def close_session(self, session_id, *, timeout_s):
        del session_id, timeout_s
        return True

    async def shutdown(self):
        return None


class _FakeProcess:
    def __init__(self) -> None:
        self.output = SessionOutput()
        self._done = asyncio.Event()

    def finish(self, output: SessionOutput) -> None:
        self.output = output
        self._done.set()

    def configure_observability(self, log_path) -> None:
        del log_path

    async def wait(self) -> None:
        await self._done.wait()

    async def set_timeout_active(self, active: bool) -> None:
        del active

    async def terminate_and_join(self) -> None:
        self._done.set()


class _FakeLauncher:
    def __init__(self) -> None:
        self.processes: dict[str, _FakeProcess] = {}

    def start_agent(self, session_input) -> _FakeProcess:
        process = _FakeProcess()
        self.processes[session_input.session_id] = process
        return process


class _Granted:
    def __init__(self, value: Any) -> None:
        self._value = value

    def remote(self, *_):
        future = asyncio.get_running_loop().create_future()
        future.set_result(self._value)
        return future


class _FakeLimiter:
    acquire_request_permit = _Granted(True)
    release_request_permit = _Granted(None)
    cancel_request_permit = _Granted(None)


def _shard_args() -> SimpleNamespace:
    return SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=64,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=0,
        rollout_max_context_len=None,
        eval_max_context_len=None,
        agentic_session_lifecycle=False,
        agentic_program_admission=False,
        agentic_admission_scope="train",
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        partial_rollout=False,
        partial_rollout_max_aborted_count=None,
        fully_async=False,
        use_slime_router=False,
        use_rollout_routing_replay=False,
    )


@pytest.fixture
async def shard(monkeypatch):
    """A real SessionShard with fake SGLang, agent processes and permits."""

    launcher = _FakeLauncher()
    monkeypatch.setattr(service.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_name=lambda: None))
    monkeypatch.setattr(service, "SGLangBackendAdapter", lambda _args: _FakeBackend())
    monkeypatch.setattr(service, "ManagedAgentLauncher", lambda *_: launcher)
    monkeypatch.setattr(service, "load_agent_app_spec_from_args", lambda _args: None)
    monkeypatch.setattr(service, "resolve_chat_api_base_url", lambda: "http://127.0.0.1:1/")
    shard = service.AgenticSessionShard.__ray_metadata__.modified_class(_shard_args(), _FakeLimiter())
    shard.test_launcher = launcher
    shard.test_release = threading.Event()
    shard.test_started: list[str] = []
    shard.test_threads: list[str] = []
    yield shard
    shard.test_release.set()
    await shard.shutdown()


def _gate_exports(shard) -> None:
    """Park export jobs on the worker until ``shard.test_release`` is set."""

    build = shard._prepare_session_exports

    def gated(**kwargs):
        shard.test_started.append(kwargs["metadata"].get("session"))
        shard.test_threads.append(threading.current_thread().name)
        # Every test releases the gate itself; the bound only stops a broken
        # implementation (assembling on the loop thread) from hanging forever.
        if not shard.test_release.wait(timeout=10):
            raise TimeoutError("export gate was never released")
        return build(**kwargs)

    shard._prepare_session_exports = gated


async def _until_exporting(shard, session_ids: list[str]) -> None:
    """Wait until these exports start assembly; check it was off the loop."""

    await _until(lambda: shard.test_started == session_ids)
    assert all(name.startswith("agentic-session-export") for name in shard.test_threads), shard.test_threads


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


async def _run_group(shard, group_id: str, session_ids: list[str], turns: int) -> dict[str, list[dict]]:
    """Start, cross the first-request barrier, lease, and play every turn."""

    specs = tuple(SessionSpec(session_id=sid, group_index=0, index=i) for i, sid in enumerate(session_ids))
    await shard.start_group("train", group_id, specs)
    await shard.resume_generation(1)

    async def agent(sid: str) -> list[dict]:
        history = [_SYSTEM, _task(sid)]
        for turn in range(turns):
            reply = await shard.chat(
                session_id=sid,
                messages=list(history),
                tools=[],
                chat_template_kwargs={},
                max_completion_tokens=64,
                stop=None,
                seed=None,
            )
            assert not reply.get("error"), reply
            history.append(reply["message"])
            if turn + 1 < turns:
                history.append(_observation(sid, turn))
        return history

    agents = [asyncio.ensure_future(agent(sid)) for sid in session_ids]
    await _until(lambda: shard._groups[group_id].first_request_barrier_crossed)
    await shard.lease_group(group_id, 1)
    return dict(zip(session_ids, await asyncio.gather(*agents), strict=True))


def _finish(shard, sid: str, output: SessionOutput | None = None) -> None:
    shard.test_launcher.processes[sid].finish(output or SessionOutput(reward=1.0, metadata={"session": sid}))


async def _take(shard, group_id: str, sid: str):
    cell = shard._groups[group_id].result_cells[sid]
    await _until(lambda: cell.terminal)
    (result,) = await shard.take_session_results(group_id, (sid,))
    return result


def _expected_fields(sid: str, turns: int) -> dict[str, Any]:
    """Training fields derived only from agent inputs and backend outputs."""

    tokens = _chars(json.dumps([_SYSTEM, _task(sid)], sort_keys=True))
    prompt_len = len(tokens)
    loss_mask: list[int] = []
    logprobs: list[float] = []
    spans: list[list[int]] = []
    for turn in range(turns):
        response = _chars(_response_text(f"req_{sid}_{turn}:0"))
        spans.append([len(loss_mask), len(loss_mask) + len(response)])
        tokens += response
        loss_mask += [1] * len(response)
        logprobs += _logprobs(len(response))
        if turn + 1 < turns:
            observation = _chars(json.dumps([_observation(sid, turn)], sort_keys=True))
            tokens += observation
            loss_mask += [0] * len(observation)
            logprobs += [0.0] * len(observation)
    return {
        "tokens": tokens,
        "prompt_len": prompt_len,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "spans": spans,
    }


@pytest.mark.parametrize("export_mode", ["implicit", "explicit"])
async def test_concurrent_exports_match_committed_trajectory(shard, export_mode) -> None:
    turns = 3
    histories = await _run_group(shard, "g", ["s0", "s1"], turns)
    for sid, history in histories.items():
        if export_mode == "explicit":
            record = {"name": "main", "messages": history, "reward": 1.0, "metadata": {"session": sid}}
            _finish(shard, sid, SessionOutput.from_records([record]))
        else:
            _finish(shard, sid)

    for sid in histories:
        transport = await _take(shard, "g", sid)
        (export,) = transport.exports
        payload = export["sample_payload"]
        assert export["name"] == ("main" if export_mode == "explicit" else None)
        assert payload["tokens"].dtype == np.int32 and payload["_rollout_tokens_shared"] is True
        sample = TrainingFieldArtifact(sample_payload=payload).to_sample()
        expected = _expected_fields(sid, turns)
        assert sample.tokens == expected["tokens"]
        assert sample.rollout_tokens == expected["tokens"]
        assert sample.loss_mask == expected["loss_mask"]
        assert sample.rollout_log_probs == expected["logprobs"]
        assert sample.custom_advantage == expected["spans"]
        assert sample.response_length == len(expected["loss_mask"])
        assert sample.prompt == "".join(map(chr, expected["tokens"][: expected["prompt_len"]]))
        assert sample.response == "".join(map(chr, expected["tokens"][expected["prompt_len"] :]))
        assert sample.weight_versions == ["3"] * turns
        assert sample.reward == 1.0 and sample.status is Sample.Status.COMPLETED
        assert sample.metadata["session"] == sid and sample.metadata["rollout_turns"] == turns
        events = sample.metadata["agentic_trace"]["events"]
        assert events["finalize_lock_acquired_at"] <= events["finalize_start_at"] <= events["finalize_end_at"]
    assert not shard._session_records


def _deepcopy_compaction(payload: dict[str, Any]) -> dict[str, Any]:
    """Reference: the pre-change compaction, which deep-copied the payload."""

    compacted = copy.deepcopy(payload)
    tokens, rollout_tokens = compacted.get("tokens"), compacted.get("rollout_tokens")
    if isinstance(tokens, list) and isinstance(rollout_tokens, list) and rollout_tokens == tokens:
        compacted["rollout_tokens"] = None
        compacted["_rollout_tokens_shared"] = True
    for name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = compacted.get(name)
        if isinstance(value, list) and value:
            compacted[name] = np.asarray(value, dtype=dtype)
    return compacted


def _assert_same(actual: Any, expected: Any, path: str = "payload") -> None:
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_same(a, e, f"{path}[{index}]")
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected), path
    else:
        assert actual == expected, path


def _rich_sample() -> Sample:
    """Two turns with top-k candidates, R3 routes and multimodal inputs."""

    forest = SessionForest.create_empty(session_id="s", group_index=1, index=2, metadata={"origin": {"k": [1]}})
    first = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=check_messages([_SYSTEM, _task("s")]),
        train_token_delta=[10, 11, 12],
        rollout_token_delta=[10, 11, 12],
        tools=[],
        chat_template_kwargs={},
    )
    resp = forest.append_resp(
        parent_state_hash=first.state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=check_messages([{"role": "assistant", "content": "ab"}]),
        token_delta=[97, 98],
        logprob_delta=[-0.5, -0.25],
        topk_token_ids_delta=np.array([[97, 1], [98, 2]], dtype=np.int32),
        topk_logprob_delta=np.array([[-0.5, -2.0], [-0.25, -3.0]], dtype=np.float32),
        weight_version_delta=["7"],
        rollout_routed_experts=np.arange(4 * 3 * 2, dtype=np.int32).reshape(4, 3, 2),
        export_metadata_patch={"agentic_trace": {"events": {"chat_end_at": 1.0}}, "request_id": "r0"},
    )
    obs = forest.append_obs(
        parent_state_hash=resp.state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=check_messages([_observation("s", 0)]),
        train_token_delta=[20, 21],
        rollout_token_delta=[20, 21],
        multimodal_train_inputs_delta={"pixel_values": torch.ones(2, 3)},
    )
    leaf = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=2,
        abort_count=1,
        messages_delta=check_messages([{"role": "assistant", "content": "c"}]),
        token_delta=[99, 100],
        logprob_delta=[-0.125, -0.0625],
        topk_token_ids_delta=np.array([[99, 3], [100, 4]], dtype=np.int32),
        topk_logprob_delta=np.array([[-0.125, -4.0], [-0.0625, -5.0]], dtype=np.float32),
        weight_version_delta=["8"],
        rollout_routed_experts=np.arange(8 * 3 * 2, dtype=np.int32).reshape(8, 3, 2),
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_CharTokenizer())
    sample.reward = {"score": 0.5}
    return sample


def test_compaction_matches_deepcopy_reference_and_leaves_sample_intact() -> None:
    sample = _rich_sample()
    before = copy.deepcopy(sample.to_dict())
    expected = _deepcopy_compaction(sample.to_dict())

    payload = TrainingFieldArtifact.from_sample(sample).sample_payload

    _assert_same(payload, expected)
    _assert_same(sample.to_dict(), before, "sample")
    _assert_same(TrainingFieldArtifact(sample_payload=payload).to_sample().to_dict(), before, "round-trip")


def test_compaction_does_not_copy_unconverted_fields() -> None:
    """Performance guard: assembly must not deep-copy the payload again."""

    sample = _rich_sample()
    payload = TrainingFieldArtifact.from_sample(sample).sample_payload
    assert payload["rollout_routed_experts"] is sample.rollout_routed_experts
    assert payload["multimodal_train_inputs"] is sample.multimodal_train_inputs
    assert payload["metadata"] is sample.metadata


async def test_export_runs_off_the_event_loop_and_freezes_the_forest(shard, tmp_path) -> None:
    shard.args.dump_agentic_obs = tmp_path
    shard._trajectory_recorder = service.ShardTrajectoryRecorder(shard.args, shard_ordinal=0)
    _gate_exports(shard)
    await _run_group(shard, "g", ["s0"], turns=2)
    session = shard._session_records["s0"]
    request_id = next(iter(session.resp_state_hash_by_request_id))
    node = session.forest.nodes_by_hash[session.resp_state_hash_by_request_id[request_id]]
    patch_before = copy.deepcopy(node.export_metadata_patch)
    static_before = copy.deepcopy(session.forest.static_metadata)
    _finish(shard, "s0")
    await _until_exporting(shard, ["s0"])

    # The Shard loop keeps serving while the job is parked on the worker.
    export_future = session.export_future
    assert (await shard.health())["ok"]
    assert not export_future.done()
    assert (await shard.debug_state())["pending_export_count"] == 1
    marked = await shard.mark_chat_service_response_ready(
        session_id="s0", request_id=request_id, remote_return_at=1.0, response_ready_at=1.0, http_return_at=1.0
    )
    assert marked is False and node.export_metadata_patch == patch_before
    shard._mark_group_timing(session.group, "group_late_milestone_at")
    assert session.forest.static_metadata == static_before

    shard.test_release.set()
    (export,) = (await _take(shard, "g", "s0")).exports
    payload = export["sample_payload"]
    assert payload["tokens"].tolist() == _expected_fields("s0", 2)["tokens"]
    assert payload["agentic_group_id"] == "g"
    assert session.observation.export_ids == [payload["sample_export_id"]]
    await shard._trajectory_recorder.aclose()
    records = (tmp_path / "run_unknown/trajectories/shard_00.jsonl").read_text().splitlines()
    (record,) = map(json.loads, records)
    assert record["sample_export_id"] == str(payload["sample_export_id"])
    assert (await shard.debug_state())["pending_export_count"] == 0


async def test_drop_discards_a_queued_export_and_waits_for_a_running_one(shard) -> None:
    _gate_exports(shard)
    await _run_group(shard, "g", ["s0", "s1"], turns=1)
    group = shard._groups["g"]
    _finish(shard, "s0")
    await _until_exporting(shard, ["s0"])
    _finish(shard, "s1")
    await _until(lambda: shard._session_records["s1"].export_future is not None)
    queued = shard._session_records["s1"].export_future

    drop = asyncio.ensure_future(shard.drop_group("g"))
    await _until(queued.cancelled)
    await asyncio.sleep(0.05)
    assert not drop.done(), "drop must wait for the export the worker is running"
    assert (await shard.debug_state())["pending_export_count"] == 1
    shard.test_release.set()
    await asyncio.wait_for(drop, timeout=5.0)

    assert shard.test_started == ["s0"]
    assert (await shard.debug_state())["pending_export_count"] == 0
    assert all(cell.dropped for cell in group.result_cells.values())
    assert "g" not in shard._groups and not shard._session_records


async def test_shutdown_waits_for_a_running_export_and_stops_the_worker(shard) -> None:
    _gate_exports(shard)
    await _run_group(shard, "g", ["s0"], turns=1)
    _finish(shard, "s0")
    await _until_exporting(shard, ["s0"])

    shutdown = asyncio.ensure_future(shard.shutdown())
    await asyncio.sleep(0.05)
    assert not shutdown.done()
    shard.test_release.set()
    await asyncio.wait_for(shutdown, timeout=5.0)

    assert not shard._groups and not shard._session_records
    assert not any(thread.is_alive() for thread in shard._export_executor._threads)
    with pytest.raises(RuntimeError):
        shard._export_executor.submit(lambda: None)


async def test_shutdown_waits_for_an_export_whose_session_was_retired(shard, tmp_path) -> None:
    """An infra failure retires a Session while its export still runs."""

    shard.args.dump_agentic_obs = tmp_path
    shard._trajectory_recorder = service.ShardTrajectoryRecorder(shard.args, shard_ordinal=0)
    _gate_exports(shard)
    await _run_group(shard, "g", ["s0"], turns=1)
    session = shard._session_records["s0"]
    watcher = session.resources.watcher_task
    cell = shard._groups["g"].result_cells["s0"]
    _finish(shard, "s0")
    await _until_exporting(shard, ["s0"])

    await shard._handle_infra_failure(session, RuntimeGroupError("permit release failed"))
    assert "s0" not in shard._session_records and not shard._groups["g"].sessions
    assert isinstance(cell.payload, RuntimeGroupError)
    assert (await shard.debug_state())["pending_export_count"] == 1

    shutdown = asyncio.ensure_future(shard.shutdown())
    await asyncio.sleep(0.05)
    assert not shutdown.done(), "shutdown must not join the worker while a retired Session's export runs"
    shard.test_release.set()
    await asyncio.wait_for(shutdown, timeout=5.0)
    await asyncio.wait_for(watcher, timeout=5.0)

    # The late export result is discarded; the cell keeps its single publication.
    assert isinstance(cell.payload, RuntimeGroupError) and "permit release failed" in str(cell.payload)
    assert session.observation.export_ids == []
    assert not (tmp_path / "run_unknown/trajectories/shard_00.jsonl").exists()
    assert not shard._inflight_exports
    assert not any(thread.is_alive() for thread in shard._export_executor._threads)


@pytest.mark.parametrize(
    ("error", "dropped"),
    [(service._NonFinalizableExportError("no committed response"), True), (ValueError("boom"), False)],
)
async def test_export_failures_keep_their_outcome_mapping(shard, error, dropped) -> None:
    await _run_group(shard, "g", ["s0"], turns=1)

    def failing(**_):
        raise error

    shard._prepare_session_exports = failing
    _finish(shard, "s0")
    group = shard._groups["g"]
    cell = group.result_cells["s0"]
    await _until(lambda: cell.terminal)

    assert cell.dropped is dropped
    if not dropped:
        assert isinstance(cell.payload, RuntimeGroupError) and "boom" in str(cell.payload)


async def test_repeated_export_cycles_leave_no_resident_state(shard) -> None:
    async def cycle(index: int) -> None:
        group_id = f"g{index}"
        session_ids = [f"{group_id}-s{i}" for i in range(2)]
        await _run_group(shard, group_id, session_ids, turns=2)
        for sid in session_ids:
            _finish(shard, sid)
        for sid in session_ids:
            assert (await _take(shard, group_id, sid)).exports
        await shard.release_group(group_id)

    await cycle(0)  # starts any lazily created worker thread
    threads = threading.active_count()
    for index in range(1, 4):
        await cycle(index)

    assert not shard._groups and not shard._session_records
    assert not shard._permit_cleanup_tasks
    assert threading.active_count() == threads
