# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Session export: training content, worker thread, cancellation, shutdown."""

from __future__ import annotations

import asyncio
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from relax.agentic.pipeline import _TRAINING_ARTIFACT_ARRAY_FIELDS, TrainingFieldArtifact, _compact_sample_payload
from relax.agentic.pipeline.runtime import finish_before_cancellation
from relax.agentic.session import service
from relax.agentic.session.state import SessionForest, _decode_tokens
from relax.utils.types import Sample


class _CharTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


def _forest(session_id: str = "s", turns: int = 3) -> tuple[SessionForest, str]:
    forest = SessionForest.create_empty(session_id=session_id, train_metadata={"loss": "grpo"}, metadata={"m": [1]})
    parent = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "hi"}],
        train_token_delta=[104, 105],
        rollout_token_delta=[104, 105],
    ).state_hash
    for turn in range(turns):
        tokens = [97 + turn] * 4
        parent = forest.append_resp(
            parent_state_hash=parent,
            rollout_id=turn,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": f"t{turn}"}],
            token_delta=tokens,
            logprob_delta=[-0.5 * (turn + 1)] * 4,
            weight_version_delta=[f"v{turn}"] * 4,
            export_metadata_patch={"request_id": f"r{turn}"},
        ).state_hash
        if turn + 1 < turns:
            parent = forest.append_obs(
                parent_state_hash=parent,
                rollout_id=turn,
                abort_count=0,
                messages_delta=[{"role": "tool", "content": "o"}],
                train_token_delta=[111, 112],
                rollout_token_delta=[111, 112],
            ).state_hash
    return forest, parent


def _reference_compact(payload: dict) -> dict:
    """The pre-optimization implementation: deep-copy, then convert."""

    compacted = copy.deepcopy(payload)
    tokens, rollout_tokens = compacted.get("tokens"), compacted.get("rollout_tokens")
    if isinstance(tokens, list) and isinstance(rollout_tokens, list) and rollout_tokens == tokens:
        compacted["rollout_tokens"] = None
        compacted["_rollout_tokens_shared"] = True
    for field_name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = compacted.get(field_name)
        if isinstance(value, list) and value:
            compacted[field_name] = np.asarray(value, dtype=dtype)
    return compacted


def _assert_same_payload(actual: dict, expected: dict) -> None:
    assert list(actual) == list(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert actual[key].dtype == value.dtype and np.array_equal(actual[key], value), key
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("shared_rollout_tokens", [True, False])
def test_compaction_matches_deepcopy_reference_and_isolates_source(shared_rollout_tokens):
    forest, leaf = _forest()
    sample = forest.build_sample(leaf_state_hash=leaf, tokenizer=_CharTokenizer())
    sample.teacher_log_probs = [-1.0, -2.0]
    sample.teacher_topk_token_ids = [[1, 2], [3, 4]]
    sample.metadata["shared"] = sample.metadata["m"]  # Aliasing must survive like in deepcopy.
    if not shared_rollout_tokens:
        sample.rollout_tokens = sample.tokens[:-1]
    payload = sample.to_dict()

    compacted = _compact_sample_payload(payload)
    _assert_same_payload(compacted, _reference_compact(payload))
    assert compacted.get("_rollout_tokens_shared", False) is shared_rollout_tokens
    assert compacted["metadata"]["shared"] is compacted["metadata"]["m"]

    # Nothing in the payload may alias the Sample it came from.
    snapshot = copy.deepcopy(compacted)
    sample.tokens.append(0)
    sample.loss_mask[0] = 9
    sample.weight_versions[0] = "mutated"
    sample.metadata["m"].append(2)
    sample.train_metadata["loss"] = "mutated"
    sample.custom_advantage[0][0] = 99
    _assert_same_payload(compacted, snapshot)
    assert all(value.flags.owndata for value in compacted.values() if isinstance(value, np.ndarray))


def test_compacted_payload_round_trips_training_fields():
    forest, leaf = _forest()
    sample = forest.build_sample(leaf_state_hash=leaf, tokenizer=_CharTokenizer())
    restored = TrainingFieldArtifact.from_sample(sample).to_sample()
    for name in ("tokens", "rollout_tokens", "loss_mask", "rollout_log_probs", "weight_versions", "response"):
        assert getattr(restored, name) == getattr(sample, name), name
    assert restored.metadata == sample.metadata and restored.custom_advantage == sample.custom_advantage


def _hf_tokenizer(clean_up_tokenization_spaces: bool):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=None))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|im_end|>"],
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
    )
    backend.train_from_iterator(["def f(x):\n    return x , y !", "你好，世界 hello world ."] * 20, trainer)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, clean_up_tokenization_spaces=clean_up_tokenization_spaces
    )


@pytest.mark.parametrize("clean_up_tokenization_spaces", [False, True])
def test_decode_matches_tokenizer_decode(clean_up_tokenization_spaces, monkeypatch):
    tokenizer = _hf_tokenizer(clean_up_tokenization_spaces)
    text = "def f(x):\n    return x , y !<|im_end|>你好，世界 hello world ."
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    # Cut through a multi-byte character so replacement handling is covered too.
    for ids in (token_ids, token_ids[:-3], token_ids[1:]):
        assert _decode_tokens(tokenizer=tokenizer, token_ids=ids) == tokenizer.decode(ids, skip_special_tokens=False)

    calls = []
    monkeypatch.setattr(
        type(tokenizer),
        "decode",
        lambda self, *args, **kwargs: calls.append(args) or "",
    )
    _decode_tokens(tokenizer=tokenizer, token_ids=token_ids)
    # The GIL-free backend batch entry point is used only when it is equivalent.
    assert bool(calls) is clean_up_tokenization_spaces


def test_decode_without_backend_uses_tokenizer_decode():
    assert _decode_tokens(tokenizer=_CharTokenizer(), token_ids=[104, 105]) == "hi"
    assert _decode_tokens(tokenizer=_CharTokenizer(), token_ids=[]) == ""


# ----------------------------------------------------------------- Shard export


def _shard():
    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(agentic_session_lifecycle=False, use_rollout_routing_replay=False)
    shard._groups, shard._session_records = {}, {}
    shard._train_generation_open = True
    shard._permit_cleanup_tasks = set()
    shard._notify_state_change = lambda group=None: None
    shard._trajectory_recorder = service.ShardTrajectoryRecorder(SimpleNamespace(), shard_ordinal=0)

    async def shutdown_backend():
        return None

    shard._generation_backend = SimpleNamespace(
        tokenizer=_CharTokenizer(), compiler=SimpleNamespace(apply_chat_template_kwargs={}), shutdown=shutdown_backend
    )
    return shard


def _session(shard, session_id: str):
    async def terminate_and_join():
        return None

    forest, _ = _forest(session_id)
    cell = service._SessionResultCell()
    group = service.ResidentGroup("train", f"g-{session_id}", {session_id: cell}, [], ownership="runtime")
    output = SimpleNamespace(reward=1.0, metadata={}, records=())
    session = service._SessionRecord(group, session_id, {}, cell, forest=forest)
    session.resources = SimpleNamespace(
        process=SimpleNamespace(output=output, terminate_and_join=terminate_and_join), process_wait=None
    )
    group.sessions.append(session)
    shard._groups[group.group_id] = group
    shard._session_records[session_id] = session
    return session


class _GatedMaterialize:
    """Wrap the real materialization with a gate and thread bookkeeping."""

    def __init__(self, shard):
        self.real = shard._materialize_session_exports
        self.gate = threading.Event()
        self.started = threading.Semaphore(0)
        self.active = self.max_active = 0
        self.threads: set[str] = set()
        self.lock = threading.Lock()

    def __call__(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.threads.add(threading.current_thread().name)
        self.started.release()
        try:
            assert self.gate.wait(10)
            return self.real(**kwargs)
        finally:
            with self.lock:
                self.active -= 1


async def _wait_started(gated: _GatedMaterialize) -> None:
    while not gated.started.acquire(blocking=False):
        await asyncio.sleep(0.001)


def test_exports_run_on_one_worker_while_the_loop_keeps_serving():
    shard = _shard()
    gated = _GatedMaterialize(shard)
    shard._materialize_session_exports = gated
    sessions = [_session(shard, f"s{i}") for i in range(3)]

    async def scenario():
        finals = [asyncio.create_task(shard._finalize_terminal_session(session)) for session in sessions]
        await _wait_started(gated)
        # The export is blocked in the worker, yet the Shard still answers.
        health = await asyncio.wait_for(shard.health(), 1)
        assert health["active_sessions"] == 3
        assert not any(session.result_cell.terminal for session in sessions)
        gated.gate.set()
        await asyncio.wait_for(asyncio.gather(*finals), 10)

    asyncio.run(scenario())
    assert gated.max_active == 1
    assert len(gated.threads) == 1 and threading.main_thread().name not in gated.threads
    exports = [session.result_cell.payload.exports[0]["sample_payload"] for session in sessions]
    assert sorted(payload["sample_export_id"] for payload in exports) == sorted(
        {payload["sample_export_id"] for payload in exports}
    )
    assert all(payload["agentic_group_id"] == session.group.group_id for payload, session in zip(exports, sessions))
    assert not shard._session_records


def test_cancelled_finalize_finishes_export_before_propagating():
    shard = _shard()
    gated = _GatedMaterialize(shard)
    shard._materialize_session_exports = gated
    session = _session(shard, "s0")

    async def scenario():
        task = asyncio.create_task(
            finish_before_cancellation(shard._finalize_terminal_session(session), "session-finalize:s0")
        )
        await _wait_started(gated)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()  # Cancellation waits for the running export.
        gated.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)

    asyncio.run(scenario())
    assert isinstance(session.result_cell.payload, service.SessionExportTransport)
    assert session.forest is None and not shard._session_records


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (service._NonFinalizableExportError("no committed response"), "export_validation_failed"),
        (RuntimeError("boom"), "export_failed"),
    ],
)
def test_worker_errors_keep_their_failure_mapping(error, expected):
    shard = _shard()
    session = _session(shard, "s0")
    failures = []
    real_finish = shard._finish_session

    def materialize(**kwargs):
        raise error

    async def finish(session, outcome, *, failure=None):
        failures.append((outcome, failure))
        return await real_finish(session, outcome, failure=failure)

    shard._materialize_session_exports = materialize
    shard._finish_session = finish
    asyncio.run(shard._finalize_terminal_session(session))
    ((outcome, failure),) = failures
    assert failure[0] == expected
    if expected == "export_failed":
        assert isinstance(outcome, service.RuntimeGroupError) and "boom" in str(outcome)
    else:
        assert outcome is None


def test_shutdown_stops_the_export_worker():
    shard = _shard()
    session = _session(shard, "s0")

    async def scenario():
        await shard._finalize_terminal_session(session)
        worker = shard._export_executor
        threads = set(worker._threads)
        assert threads and all(thread.is_alive() for thread in threads)
        await shard.shutdown()
        return threads

    threads = asyncio.run(scenario())
    assert not any(thread.is_alive() for thread in threads)
    assert isinstance(session.result_cell.payload, service.SessionExportTransport)


def test_shutdown_before_any_export_starts_no_worker():
    shard = _shard()

    asyncio.run(shard.shutdown())
    assert shard._export_executor is None


def test_concurrent_exports_match_serial_exports():
    def run(concurrent: bool) -> list[dict]:
        shard = _shard()
        sessions = [_session(shard, f"s{i}") for i in range(4)]

        async def scenario():
            if concurrent:
                await asyncio.gather(*(shard._finalize_terminal_session(session) for session in sessions))
            else:
                for session in sessions:
                    await shard._finalize_terminal_session(session)
            await shard.shutdown()

        asyncio.run(scenario())
        return [
            TrainingFieldArtifact(sample_payload=session.result_cell.payload.exports[0]["sample_payload"]).to_sample()
            for session in sessions
        ]

    fields = ("tokens", "rollout_tokens", "loss_mask", "rollout_log_probs", "weight_versions", "response", "prompt")
    for concurrent_sample, serial_sample in zip(run(True), run(False)):
        assert isinstance(concurrent_sample, Sample)
        for name in fields:
            assert getattr(concurrent_sample, name) == getattr(serial_sample, name), name
