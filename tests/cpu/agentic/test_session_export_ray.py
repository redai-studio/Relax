# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""End-to-end smoke: a real Ray Shard actor keeps answering RPCs while it
exports long trajectories.

The actor is the production ``AgenticSessionShard`` class with a test-only
constructor that fills real Forests (standing in for the inference backend)
and uses a real fast tokenizer built offline. Short ``health()`` RPCs travel
through Ray while several long Sessions finalize concurrently through the
production ``_finalize_terminal_session`` path.

The first export is held inside the export worker while the driver sends
``health()`` RPCs, and each RPC must return before the export is released. An
export that ran on the event loop would hold the loop instead, so those RPCs
would time out.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest


ray = pytest.importorskip("ray")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from relax.agentic.session import service  # noqa: E402
from relax.agentic.session.state import SessionForest  # noqa: E402


_TURNS = 48
_RESPONSE_TOKENS = 2048
_OBSERVATION_TOKENS = 512
_SESSIONS = 4


def _tokenizer():
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=None))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=2000, initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet()
    )
    corpus = [f"def handler_{i}(request):\n    return {{'status': {i}, 'body': '你好 world'}}\n" for i in range(400)]
    backend.train_from_iterator(corpus, trainer)
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)


def _forest(session_id: str, vocab_size: int, seed: int) -> SessionForest:
    def ids(count: int, salt: int) -> list[int]:
        return [(seed * 7919 + salt * 104729 + i * 31) % vocab_size for i in range(count)]

    forest = SessionForest.create_empty(session_id=session_id)
    parent = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "task"}],
        train_token_delta=ids(128, -1),
        rollout_token_delta=[],
    ).state_hash
    for turn in range(_TURNS):
        tokens = ids(_RESPONSE_TOKENS, 2 * turn)
        parent = forest.append_resp(
            parent_state_hash=parent,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": f"turn {turn}"}],
            token_delta=tokens,
            logprob_delta=[-(token % 97) / 97 for token in tokens],
            weight_version_delta=["1"] * len(tokens),
            export_metadata_patch={"request_id": f"r{turn}"},
        ).state_hash
        if turn + 1 < _TURNS:
            parent = forest.append_obs(
                parent_state_hash=parent,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "tool", "content": f"obs {turn}"}],
                train_token_delta=ids(_OBSERVATION_TOKENS, 2 * turn + 1),
                rollout_token_delta=[],
            ).state_hash
    return forest


class _SmokeShard(service.AgenticSessionShard.__ray_metadata__.modified_class):
    def __init__(self) -> None:
        self.args = SimpleNamespace(agentic_session_lifecycle=False, use_rollout_routing_replay=False)
        self._groups, self._session_records = {}, {}
        self._closed, self._train_generation_open = False, True
        self._state_revision = 0
        self._state_changed = asyncio.Event()
        self._permit_cleanup_tasks, self._agent_cleanup_tasks = set(), set()
        self._pending_backend_cleanups = {}
        self._export_started, self._release_export = threading.Event(), threading.Event()
        self._export_started.set()
        self._release_export.set()  # No hold until start_exports() arms it.
        self._trajectory_recorder = service.ShardTrajectoryRecorder(SimpleNamespace(), shard_ordinal=0)

        async def shutdown_backend() -> None:
            return None

        self._tokenizer = _tokenizer()
        self._generation_backend = SimpleNamespace(
            tokenizer=self._tokenizer,
            compiler=SimpleNamespace(apply_chat_template_kwargs={}),
            shutdown=shutdown_backend,
        )

    def _add_session(self, index: int):
        async def terminate_and_join() -> None:
            return None

        session_id = f"s{index}"
        forest = _forest(session_id, len(self._tokenizer), index)
        cell = service._SessionResultCell()
        group = service.ResidentGroup("train", f"g{index}", {session_id: cell}, [], ownership="runtime")
        process = SimpleNamespace(
            output=SimpleNamespace(reward=1.0, metadata={}, records=()), terminate_and_join=terminate_and_join
        )
        session = service._SessionRecord(group, session_id, {}, cell, forest=forest)
        idle = SimpleNamespace(cancel=lambda: None)
        session.resources = SimpleNamespace(process=process, process_wait=idle, watcher_task=idle)
        group.sessions.append(session)
        self._groups[group.group_id] = group
        self._session_records[session_id] = session
        return session

    def _materialize_session_exports(self, **kwargs):
        # Hold the first real export in the worker until the driver has seen RPCs answered.
        if not self._export_started.is_set():
            self._export_started.set()
            assert self._release_export.wait(60)
        return super()._materialize_session_exports(**kwargs)

    async def start_exports(self) -> None:
        await self._finalize_terminal_session(self._add_session(-1))  # Warm-up.
        self._export_started, self._release_export = threading.Event(), threading.Event()
        self._sessions = [self._add_session(i) for i in range(_SESSIONS)]
        self._exports = asyncio.gather(*(self._finalize_terminal_session(session) for session in self._sessions))

    async def wait_export_started(self) -> None:
        while not self._export_started.is_set():
            await asyncio.sleep(0.001)

    async def health(self) -> dict:
        reply = await super().health()
        reply["export_in_flight"] = self._export_started.is_set() and not self._release_export.is_set()
        return reply

    async def finish_exports(self) -> dict:
        self._release_export.set()
        await self._exports
        exports = [session.result_cell.payload.exports[0]["sample_payload"] for session in self._sessions]
        return {
            "token_counts": [len(payload["tokens"]) for payload in exports],
            "export_ids": [payload["sample_export_id"] for payload in exports],
            "remaining_sessions": len(self._session_records),
        }


@pytest.fixture(scope="module")
def ray_runtime():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    if started_here:
        ray.shutdown()


def test_shard_actor_answers_rpcs_while_exporting_long_trajectories(ray_runtime):
    actor_cls = ray.remote(max_concurrency=service._AGENTIC_SHARD_MAX_CONCURRENCY)(_SmokeShard)
    shard = actor_cls.options(num_cpus=0).remote()
    try:
        ray.get(shard.health.remote(), timeout=60)
        ray.get(shard.start_exports.remote(), timeout=60)
        ray.get(shard.wait_export_started.remote(), timeout=60)
        # The export is held in the worker; each RPC must return before it is released.
        replies = [ray.get(shard.health.remote(), timeout=10) for _ in range(3)]
        result = ray.get(shard.finish_exports.remote(), timeout=120)
        ray.get(shard.shutdown.remote(), timeout=60)
    finally:
        ray.kill(shard)

    assert [reply["export_in_flight"] for reply in replies] == [True] * 3
    expected_tokens = 128 + _TURNS * _RESPONSE_TOKENS + (_TURNS - 1) * _OBSERVATION_TOKENS
    assert result["token_counts"] == [expected_tokens] * _SESSIONS
    assert len(set(result["export_ids"])) == _SESSIONS
    assert result["remaining_sessions"] == 0
