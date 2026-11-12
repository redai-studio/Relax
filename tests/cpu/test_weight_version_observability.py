# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weight version span serialization into the trajectory observation stream.

Self-contained copy of the six-stream join scenario from
``tests/cpu/test_agentic_observability.py`` (helpers, fixtures, and the
scenario itself), extended to assert that the ``weight_version_spans``
captured on SessionForest resp nodes survive the trajectory JSONL boundary.
"""

from __future__ import annotations

import asyncio
import json
import sys
from argparse import Namespace
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch

from relax.agentic.observability import ShardTrajectoryRecorder
from relax.agentic.pipeline import GroupExport, SampleExport, SessionExport, TrainingFieldArtifact
from relax.agentic.session.state import SessionForest, check_messages
from relax.agentic.stage_recorder import STATUS_QUEUED, StageRecorder
from relax.utils.agentic_obs import merge_train_updates, resolve_agentic_obs_config
from relax.utils.agentic_obs.writers import close_observation_writers
from relax.utils.types import Sample, WeightVersionSpan


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _obs_args(tmp_path: Path, **overrides: Any) -> Namespace:
    args = {
        "dump_agentic_obs": str(tmp_path),
        "agentic_obs_run_id": "run_test",
        "agentic_obs_train_generation_id": "generation",
        "dump_agentic_obs_media": False,
        "reward_key": None,
        "use_rollout_routing_replay": False,
    }
    args.update(overrides)
    return Namespace(**args)


def _two_turn_forest() -> tuple[SessionForest, str]:
    """Build a forest whose export lineage is root -> obs -> resp -> obs ->
    resp."""

    forest = SessionForest.create_empty(
        session_id="shard_3.session-abcdef",
        group_index=2,
        index=5,
        label="gold",
        train_metadata={"loss": "grpo"},
        metadata={},
    )
    first_obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    first_resp = forest.append_resp(
        parent_state_hash=first_obs.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "a"}]}],
        token_delta=_chars("a"),
        logprob_delta=[-0.1],
        status="completed",
        export_metadata_patch={"request_id": "req-1"},
    )
    second_obs = forest.append_obs(
        parent_state_hash=first_resp.state_hash,
        rollout_id=8,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "more"}]}]),
        train_token_delta=_chars("more"),
        rollout_token_delta=_chars("more"),
    )
    leaf = forest.append_resp(
        parent_state_hash=second_obs.state_hash,
        rollout_id=8,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "bb"}]}],
        token_delta=_chars("bb"),
        logprob_delta=[-0.2, -0.3],
        status="completed",
        export_metadata_patch={"request_id": "req-2"},
    )
    return forest, leaf.state_hash


class _FakeMpu:
    """Minimal megatron.core.mpu stand-in for the emit path."""

    def __init__(
        self, *, tp_rank: int = 0, last_stage: bool = True, cp_size: int = 1, cp_rank: int = 0, dp_rank: int = 0
    ) -> None:
        self._tp_rank = tp_rank
        self._last_stage = last_stage
        self._cp_size = cp_size
        self._cp_rank = cp_rank
        self._dp_rank = dp_rank

    def get_tensor_model_parallel_rank(self) -> int:
        return self._tp_rank

    def is_pipeline_last_stage(self, ignore_virtual=False) -> bool:
        return self._last_stage

    def get_context_parallel_world_size(self) -> int:
        return self._cp_size

    def get_context_parallel_rank(self) -> int:
        return self._cp_rank

    def get_data_parallel_rank(self, with_context_parallel=False) -> int:
        return self._dp_rank

    def get_data_parallel_world_size(self, with_context_parallel=False) -> int:
        return 1


@pytest.fixture
def fake_mpu(monkeypatch):
    """Install a fake ``megatron.core`` so the emit path runs off-GPU."""

    def install(**kwargs) -> _FakeMpu:
        mpu = _FakeMpu(**kwargs)
        megatron = sys.modules.get("megatron") or SimpleNamespace()
        core = SimpleNamespace(mpu=mpu)
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setattr(megatron, "core", core, raising=False)
        return mpu

    from relax.utils.agentic_obs import train_hooks

    train_hooks.close_train_appenders()
    close_observation_writers()
    monkeypatch.setattr(train_hooks, "_UPDATE_SEQUENCES", {})
    yield install
    train_hooks.close_train_appenders()
    close_observation_writers()


def _train_args(tmp_path: Path, **overrides: Any) -> Namespace:
    args = {
        "dump_agentic_obs": str(tmp_path),
        "agentic_obs_run_id": "run_test",
        "agentic_obs_train_generation_id": "generation",
        "dump_agentic_obs_media": False,
        "dynamic_context_parallel": False,
        "context_parallel_size": 1,
        "n_samples_per_prompt": 1,
        "rollout_batch_size": 1,
        "advantage_estimator": "grpo",
        "agentic_custom_advantage_path": None,
        "reward_key": None,
    }
    args.update(overrides)
    return Namespace(**args)


def _rollout_data(export_ids: list[int], **overrides: Any) -> dict[str, Any]:
    count = len(export_ids)
    data = {
        "sample_export_ids": [torch.tensor(export_id) for export_id in export_ids],
        "agentic_group_ids": [f"group-{export_id}" for export_id in export_ids],
        "response_lengths": [4] * count,
        "total_lengths": [6] * count,
        "loss_masks": [torch.ones(4, dtype=torch.int) for _ in range(count)],
        "advantages": [torch.full((4,), 0.5) for _ in range(count)],
        "returns": [torch.full((4,), 2.0) for _ in range(count)],
    }
    data.update(overrides)
    return data


def _emit_update(args, rollout_id, data, step_id=0, successful=True):
    from relax.utils.agentic_obs import train_hooks

    with train_hooks.observe_train_step(args, rollout_id, step_id):
        train_hooks.capture_training_input(data)
        train_hooks.record_optimizer_result(successful)


@pytest.fixture
def rollout_module(monkeypatch):
    """Exercise real orchestration with its unrelated GPU logging import
    stubbed."""
    import importlib
    from types import ModuleType

    logging_module = ModuleType("relax.distributed.ray.rollout")
    logging_module._log_rollout_data = lambda *args, **kwargs: None
    logging_module.get_local_rollout_manager = lambda: SimpleNamespace(training_weight_updating=False)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", logging_module)
    monkeypatch.delitem(sys.modules, "relax.agentic.rollout", raising=False)
    module = importlib.import_module("relax.agentic.rollout")
    yield module
    sys.modules.pop("relax.agentic.rollout", None)


def _obs_pipeline(module, args):
    pipeline = object.__new__(module.AgenticResidentPipeline)
    pipeline.args = args
    pipeline._data_system_client = None
    pipeline._active_step = module._StepContext(rollout_id=7, final_backfill=False)
    pipeline._active_group_tasks = []
    pipeline._group_observations = {}
    pipeline._finalized_groups = deque()
    pipeline._stage_recorder = StageRecorder(args)
    pipeline._record_rollout = AsyncMock()  # These fixtures exercise trajectory ownership, not rollout membership.
    pipeline.prepare_domain = SimpleNamespace(close_step=lambda: None, shutdown=AsyncMock())
    pipeline.runtime_domain = SimpleNamespace(
        pause_generation=AsyncMock(), trim_memory=AsyncMock(), shutdown=AsyncMock()
    )
    pipeline.transfer_domain = SimpleNamespace(
        flush_batches=lambda: (),
        release_finished_partitions=lambda: None,
        pending_groups=lambda: (),
        clear=lambda: None,
    )
    return pipeline


async def _lease_observed_group(pipeline, module, stream, reward_domain, collect_group):
    pipeline._changed = asyncio.Event()
    pipeline._opd_manager = None
    pipeline.reward_domain = reward_domain
    # Upstream take_ready yields (stream, teacher_id, response_keys) triples; this
    # scenario has no MOPD teacher, so both side fields are empty.
    pipeline.prepare_domain.take_ready = lambda count: [(stream, None, ())]
    pipeline.runtime_domain.lease_group = AsyncMock()
    pipeline.runtime_domain.drop_group = AsyncMock()
    pipeline.runtime_domain.collect_group = collect_group
    pipeline._runtime_group_gap = lambda context: 1
    await pipeline._lease_prepared_groups(pipeline._active_step)
    assert len(pipeline._group_observations) == 1
    return pipeline._active_group_tasks[0]


def _session_observation_fixture(tmp_path):
    from relax.agentic.observability import SessionObservation
    from relax.agentic.session import service

    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = _obs_args(
        tmp_path,
        agentic_session_lifecycle=False,
        partial_rollout=True,
        partial_rollout_max_aborted_count=3,
        fully_async=False,
        use_slime_router=False,
    )
    shard._trajectory_recorder = ShardTrajectoryRecorder(shard.args, shard_ordinal=0)
    shard._generation_backend = SimpleNamespace(abort_request=AsyncMock(), tokenizer=_FakeTokenizer())
    shard._notify_state_change = lambda group=None: None
    shard._train_generation_open = True
    cell = service._SessionResultCell()
    group = service.ResidentGroup("train", "g", {"s": cell}, [], ownership=service.GroupOwnership.RUNTIME)
    session = service._SessionRecord(group, "s", {}, cell, observation=SessionObservation())
    group.sessions.append(session)
    shard._session_records = {"s": session}
    shard._groups = {"g": group}
    return shard, session


def test_trajectory_stream_carries_weight_version_spans(tmp_path, rollout_module, fake_mpu):
    from relax.agentic.inventory import InventoryRecorder
    from relax.agentic.pipeline.reward import RewardDomain
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    args = _train_args(tmp_path, group_rm=False, reward_key="score")

    async def scenario():
        inventory = InventoryRecorder(args, "train")
        shard, session = _session_observation_fixture(tmp_path)
        shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=15)
        shard._trajectory_recorder.next_sample_export_id()  # Exercise a nonzero low bit across the artifact boundary.
        forest, leaf_hash = _two_turn_forest()
        forest.lineage(leaf_hash)[2].weight_version_delta.append("v1")
        # forest.lineage includes the synthetic root at index 0, so the two resp
        # nodes sit at forest-side indices 2 and 4.
        forest.lineage(leaf_hash)[2].weight_version_spans.append(
            WeightVersionSpan(version="v1", token_start=0, token_end=1, attempt_seq=0)
        )
        # The second span exercises the unknown-version (None) path through the JSONL boundary.
        forest.lineage(leaf_hash)[4].weight_version_spans.append(
            WeightVersionSpan(version=None, token_start=0, token_end=2, attempt_seq=0)
        )
        forest.session_id = session.session_id
        session.forest = forest
        inventory.record_group("g", [Sample(prompt="hi", group_index=2, index=5)], ["s"])
        transport = shard._build_session_export_transport(
            session=session,
            reward={"score": 0.8, "judge": 0.9},
            metadata={},
            output_records=(),
            finalize_events={},
        )
        await shard._finish_session(session, transport)
        sample = TrainingFieldArtifact(sample_payload=transport.exports[0]["sample_payload"]).to_sample()
        group = GroupExport("g", (SessionExport(exports=(SampleExport(name=None, sample=sample),)),))
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(group.sessions[0])
        stream = SimpleNamespace(
            group_id="g", session_results=(SimpleNamespace(session_id="s", completion=completion),)
        )
        pipeline = _obs_pipeline(rollout_module, args)
        task = await _lease_observed_group(
            pipeline, rollout_module, stream, RewardDomain(args, "train", None), AsyncMock(return_value=group)
        )
        await task
        pipeline._collect_finalized_groups()
        assert not pipeline._active_group_tasks and not pipeline._group_observations
        assert list(pipeline._finalized_groups) == [group]
        collected = pipeline._finalized_groups.popleft()
        await pipeline._stage_recorder.record_queued(
            collected.samples, rollout_id=7, partition_id=6, group_ids={sample.sample_export_id: "g"}
        )
        pipeline._stage_recorder.close()
        await inventory.aclose()
        await shard._trajectory_recorder.aclose()
        return sample, forest

    sample, forest = asyncio.run(scenario())
    response_length = sample.response_length
    _emit_update(
        args,
        7,
        _rollout_data(
            [sample.sample_export_id],
            agentic_group_ids=["g"],
            response_lengths=[response_length],
            total_lengths=[len(sample.tokens)],
            loss_masks=[torch.tensor(sample.loss_mask)],
            advantages=[torch.full((response_length,), 0.5)],
            returns=[torch.full((response_length,), 0.8)],
        ),
    )
    train_hooks.close_train_appenders()
    close_observation_writers()
    config = resolve_agentic_obs_config(args)
    root = config.root
    assert not config.stage_path(6).exists()
    assert json.loads(config.manifest_path.read_text())["cp_size"] == 1
    rows = {
        stream: [row for path in (root / stream).glob("*.jsonl") for row in _read_jsonl(path)]
        for stream in ("prompts", "groups", "sessions", "trajectories", "stages", "train")
    }
    assert all(rows.values())
    assert rows["prompts"][0]["prompt_id"] == rows["groups"][0]["prompt_id"] == rows["sessions"][0]["prompt_id"]
    assert (
        rows["groups"][0]["group_id"] == rows["sessions"][0]["group_id"] == rows["trajectories"][0]["group_id"] == "g"
    )
    assert rows["sessions"][0]["session_id"] == rows["trajectories"][0]["session_id"] == "s"
    trajectory = rows["trajectories"][0]
    export_id = trajectory["sample_export_id"]
    assert export_id == str(sample.sample_export_id)
    assert int(export_id) >> 56 == 15 and int(export_id) & 1
    assert trajectory["schema_version"] == "3.2.0" and trajectory["run_id"] == "run_test"
    assert trajectory["sample_index"] == 5 and "index" not in trajectory
    assert trajectory["group_index"] == 2 and trajectory["label"] == "gold"
    assert trajectory["export_name"] is None and trajectory["exported_at"].endswith("Z")
    lineage = trajectory["lineage"]
    assert [node["kind"] for node in lineage] == ["obs", "resp", "obs", "resp"]
    assert [node["token_count"] for node in lineage] == [2, 1, 4, 2]
    assert len(forest.lineage(trajectory["leaf_state_hash"])) == len(lineage) + 1
    assert forest.root_state_hash not in {node["state_hash"] for node in lineage}
    assert lineage[1]["weight_versions"] == ["v1"]
    assert trajectory["agentic_trace"]["turn_count"] == 2
    assert [turn["resp_state_hash"] for turn in trajectory["agentic_trace"]["turns"]] == [
        lineage[1]["state_hash"],
        lineage[3]["state_hash"],
    ]
    assert rows["train"][0]["samples"][0]["sample_export_id"] == export_id
    transferred = [event for event in rows["stages"] if event["stage"] == "transfer" and event["status"] != "start"]
    assert [(event["status"], event["partition_id"]) for event in transferred] == [(STATUS_QUEUED, 6)]
    for event in rows["stages"]:
        if event["stage"] in ("reward", "transfer") and event["status"] != "start":
            assert event["reward_value"] == 0.8 and event["reward_detail"] == {"score": 0.8, "judge": 0.9}
    committed = [
        event for event in merge_train_updates(rows["stages"], rows["train"]) if event["status"] == "committed"
    ]
    assert committed[0]["update_id"] == rows["train"][0]["update_id"]
    assert committed[0]["status"] == "committed" and committed[0]["group_ids"] == ["g"]
    session_terminal = [event for event in rows["stages"] if event["scope"] == "session" and event["is_terminal"]]
    assert len(session_terminal) == 1 and session_terminal[0]["export_ids"] == [export_id]

    # The resp nodes carry their captured spans (the None version round-trips as
    # JSON null); obs nodes carry the empty default.
    assert [node["weight_version_spans"] for node in lineage if node["kind"] == "resp"] == [
        [{"version": "v1", "token_start": 0, "token_end": 1, "attempt_seq": 0}],
        [{"version": None, "token_start": 0, "token_end": 2, "attempt_seq": 0}],
    ]
    obs_nodes = [node for node in lineage if node["kind"] == "obs"]
    assert all(node["weight_version_spans"] == [] for node in obs_nodes)
