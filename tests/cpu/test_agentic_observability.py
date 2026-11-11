# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Integration tests for agentic observability against real producers.

``tests/cpu/utils/test_agentic_obs.py`` covers the record and writer layer in
isolation. This file exercises the layer against the types it actually consumes
— a real ``SessionForest``, real ``GroupExport``s, a real ``TransferDomain`` —
plus the two boundaries the join key has to survive: the TransferQueue column
and the actor's on-device reduction.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from argparse import Namespace
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch

from relax.agentic.observability import (
    ShardTrajectoryRecorder,
    resolve_shard_ordinal,
    sample_reward_fields,
)
from relax.agentic.pipeline import GroupExport, SampleExport, SessionExport, TrainingFieldArtifact
from relax.agentic.pipeline.transfer import TransferDomain
from relax.agentic.session.state import SessionForest, check_messages
from relax.agentic.stage_recorder import (
    STAGE_FILTER,
    STAGE_OVERSAMPLING,
    STAGE_SHUTDOWN,
    STATUS_QUEUED,
    StageRecorder,
)
from relax.utils.agentic_obs import agentic_obs_enabled, merge_train_updates, resolve_agentic_obs_config
from relax.utils.agentic_obs.writers import close_observation_writers
from relax.utils.training.data_fields import build_data_fields
from relax.utils.types import Sample

from tests.cpu.agentic_obs_args import make_agentic_obs_args


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


def _read_stage_ends(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    for stage_id in {row["stage_id"] for row in rows if row["scope"] != "session"}:
        pair = [row for row in rows if row["stage_id"] == stage_id]
        assert len(pair) == 2
        start, end = sorted(pair, key=lambda row: row["event_seq"])
        assert start["status"] == "start" and not start["is_terminal"]
        assert end["status"] != "start" and end["event_seq"] > start["event_seq"]
        assert start["occurred_at"] <= end["occurred_at"]
        assert start["record_id"] != end["record_id"]
        assert (start["scope"], start["stage"]) == (end["scope"], end["stage"])
    return [row for row in rows if row["status"] != "start"]


def _obs_args(tmp_path: Path, **overrides: Any) -> Namespace:
    # Factory defaults live in one dict so callers can still override them via
    # **overrides ({**defaults, **overrides}: caller wins).
    return make_agentic_obs_args(
        tmp_path,
        run_id="run_test",
        **{"use_rollout_routing_replay": False, **overrides},
    )


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


# --------------------------------------------------------------------------
# Shard side: the trajectory record
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dump_media", [False, True], ids=["references-only", "with-media"])
def test_trajectory_media_respects_sidecar_switch(tmp_path, dump_media) -> None:
    forest, leaf_hash = _two_turn_forest()
    message = forest.lineage(leaf_hash)[1].messages_delta[0]
    content = message["content"] = [
        {"type": "text", "text": message["content"]},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        {"type": "image_url", "image_url": {"url": "https://example/a.png"}},
    ]
    sample = forest.build_sample(leaf_state_hash=leaf_hash, tokenizer=_FakeTokenizer())
    args = _obs_args(tmp_path, dump_agentic_obs_media=dump_media)
    config = resolve_agentic_obs_config(args)
    recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)

    async def scenario():
        recorder.record(
            forest=forest,
            state_hash=leaf_hash,
            sample_export_id=sample.sample_export_id,
            metadata=sample.metadata,
            export_name="answer",
        )
        # Snapshotting and enqueueing do not touch disk.
        assert not config.media_dir.exists()
        await recorder.aclose()

    asyncio.run(scenario())

    (record,) = _read_jsonl(config.trajectory_path(0))
    assert record["export_name"] == "answer"
    serialized = json.dumps(record)
    assert "QUJDRA" not in serialized
    assert "<media:sha256:" in serialized
    assert content[1]["image_url"]["url"] == "data:image/png;base64,QUJDRA=="
    assert record["lineage"][0]["messages"][0]["content"][0]["text"] == "hi"
    assert record["lineage"][0]["messages"][0]["content"][2] == content[2]
    written = list(config.media_dir.glob("*.bin"))
    assert len(written) == int(dump_media)
    if dump_media:
        assert written[0].read_bytes() == b"ABCD"
    else:
        assert not config.media_dir.exists()


def test_resolve_shard_ordinal_falls_back_outside_a_ray_actor() -> None:
    # Driver-side unit tests construct the recorder with no actor context; that
    # must degrade to shard 0 rather than raise into the export path.
    assert resolve_shard_ordinal() == 0
    assert resolve_shard_ordinal(default=4) == 4


# --------------------------------------------------------------------------
# Driver side: stage records
# --------------------------------------------------------------------------


def _group(group_id: str, *, export_ids: list[int], reward: Any = 1.0) -> GroupExport:
    exports = tuple(
        SampleExport(
            name=None,
            sample=Sample(response=f"r{export_id}", reward=reward, sample_export_id=export_id, metadata={}),
        )
        for export_id in export_ids
    )
    return GroupExport(group_id=group_id, sessions=(SessionExport(exports=exports),))


def test_stage_recorder_keeps_the_open_file_count_flat(tmp_path) -> None:
    # A long run must not accumulate one descriptor per step until it hits its
    # ulimit; older steps are evicted and reopened on demand.
    args = _obs_args(tmp_path)
    recorder = StageRecorder(args)

    async def scenario() -> None:
        await recorder.record_group_dropped(
            _group("unassigned", export_ids=[-1, -1]), rollout_id=99, stage=STAGE_SHUTDOWN
        )
        for rollout_id in range(12):
            await recorder.record_group_dropped(
                _group(f"g{rollout_id}", export_ids=[rollout_id]),
                rollout_id=rollout_id,
                stage=STAGE_FILTER,
            )
        assert len(recorder._appenders) <= 4
        recorder.close()

    asyncio.run(scenario())

    config = resolve_agentic_obs_config(args)
    assert not config.stage_path(99).exists()
    for rollout_id in range(12):
        (row,) = _read_stage_ends(config.stage_path(rollout_id))
        assert row["sample_export_id"] == str(rollout_id)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("stage", [STAGE_FILTER, STAGE_OVERSAMPLING, STAGE_SHUTDOWN])
def test_stage_recorder_preserves_drop_reason_defaults_and_overrides(tmp_path, explicit, stage):
    recorder = StageRecorder(_obs_args(tmp_path))

    async def scenario():
        await recorder.record_group_dropped(
            _group("g", export_ids=[7]),
            rollout_id=7,
            stage=stage,
            **({"reason_code": "custom_failure", "category": "infra"} if explicit else {}),
        )
        recorder.close()

    asyncio.run(scenario())
    (event,) = _read_stage_ends(resolve_agentic_obs_config(_obs_args(tmp_path)).stage_path(7))
    assert event["reason_code"] == ("custom_failure" if explicit else f"{stage}_rejected")
    assert event["category"] == ("infra" if explicit else "selection")
    assert event["status"] == ("failed" if explicit else "cancelled" if stage == STAGE_SHUTDOWN else "filtered")


def test_stage_recorder_distinguishes_queue_acknowledgement_from_reward_completion(tmp_path):
    recorder = StageRecorder(_obs_args(tmp_path))
    sample = Sample(sample_export_id=7, reward=1.0)

    async def scenario():
        await recorder.record_reward(sample, rollout_id=7, group_id="g")
        await recorder.record_queued([sample], rollout_id=7, partition_id=6, group_ids={7: "g"})
        recorder.close()

    asyncio.run(scenario())
    reward, transfer = _read_stage_ends(resolve_agentic_obs_config(_obs_args(tmp_path)).stage_path(7))
    assert (reward["stage"], reward["status"]) == ("reward", "completed")
    assert (transfer["stage"], transfer["status"]) == ("transfer", "queued_for_train")
    assert not reward["is_terminal"] and not transfer["is_terminal"]


def test_sample_reward_fields_handles_every_reward_shape() -> None:
    args = Namespace(reward_key=None)
    assert sample_reward_fields(args, Sample(reward=None)) == (None, None)
    assert sample_reward_fields(args, Sample(reward=1.5)) == (1.5, None)
    # Without --reward-key there is no defensible projection of a dict reward,
    # but the detail is still worth keeping.
    assert sample_reward_fields(args, Sample(reward={"a": 1.0})) == (None, {"a": 1.0})


# --------------------------------------------------------------------------
# Train side: the TransferQueue column and the on-device reduction
# --------------------------------------------------------------------------


def test_export_id_column_is_gated_on_the_flag(tmp_path) -> None:
    # Producer and consumer must agree exactly: a column the whitelist requests
    # but nothing writes makes async_get_meta wait forever.
    from relax.utils.utils import convert_samples_to_train_data, dict_to_tensordict

    base = {
        "custom_convert_samples_to_train_data_path": None,
        "custom_reward_post_process_path": None,
        "agentic_custom_advantage_path": None,
        "advantage_estimator": "grpo",
        "rewards_normalization": False,
        "reward_key": None,
        "multimodal_keys": None,
        "use_opd": False,
        "debug_train_only": True,
        "n_samples_per_prompt": 1,
        "loss_type": "grpo",
    }
    export_ids = [(15 << 56) | 1, (15 << 56) | 2]
    samples = [
        Sample(index=i, group_index=i, tokens=[1, 2, 3], response_length=2, reward=1.0, sample_export_id=export_id)
        for i, export_id in enumerate(export_ids)
    ]

    off = Namespace(**base, dump_agentic_obs=None)
    assert agentic_obs_enabled(off) is False
    assert "sample_export_ids" not in convert_samples_to_train_data(off, samples)
    assert "sample_export_ids" not in build_data_fields(off)
    recorder = StageRecorder(off)
    assert not recorder.enabled
    asyncio.run(recorder.record_group_dropped(_group("off", export_ids=[1]), rollout_id=0, stage=STAGE_FILTER))
    recorder.close()
    _emit_update(off, 8, _rollout_data([5]))
    assert not list(tmp_path.iterdir())

    on = Namespace(**base, dump_agentic_obs=str(tmp_path))
    assert agentic_obs_enabled(on) is True
    converted = convert_samples_to_train_data(on, samples)
    assert converted["sample_export_ids"] == export_ids
    assert "sample_export_ids" in build_data_fields(on)
    batch = dict_to_tensordict({"sample_export_ids": converted["sample_export_ids"]}, len(samples))
    assert batch["sample_export_ids"].dtype == torch.int64
    assert batch["sample_export_ids"].tolist() == export_ids


@pytest.mark.parametrize("missing_value", ["absent", None])
def test_custom_converter_missing_ids_preserves_rows_with_unknown_identity(tmp_path, monkeypatch, missing_value):
    from relax.utils import utils

    args = _obs_args(tmp_path, custom_convert_samples_to_train_data_path="custom.convert")
    converted = {"tokens": [[1, 2]]}
    if missing_value is None:
        converted["sample_export_ids"] = None
    monkeypatch.setattr(utils, "load_function", lambda path: lambda args, samples: converted)
    result = utils.convert_samples_to_train_data(args, [Sample(sample_export_id=7)])
    assert result["tokens"] == [[1, 2]]
    assert result["sample_export_ids"] == [-1]
    assert result["agentic_group_ids"] == [[0]]
    args.dump_agentic_obs = None
    assert utils.convert_samples_to_train_data(args, [Sample()]) is converted


@pytest.mark.parametrize("as_tensordict", [False, True])
def test_custom_converter_preserves_output_export_order(tmp_path, monkeypatch, as_tensordict):
    from relax.utils import utils

    args = _obs_args(tmp_path, custom_convert_samples_to_train_data_path="custom.convert")
    export_ids = [(15 << 56) | 1, (15 << 56) | 2]
    # A custom converter can reorder and expand rows: it owns the output mapping.
    expected_ids = [export_ids[1], export_ids[0], export_ids[1]]
    converted = {"sample_export_ids": expected_ids, "total_lengths": [2, 2, 2]}
    if as_tensordict:
        converted = utils.dict_to_tensordict(converted, 3)
    monkeypatch.setattr(utils, "load_function", lambda path: lambda args, samples: converted)
    result = utils.convert_samples_to_train_data(args, [Sample(sample_export_id=value) for value in export_ids])
    assert result is converted
    assert [int(value) for value in result["sample_export_ids"]] == expected_ids


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


def _train_args(tmp_path: Path, **overrides: Any) -> Namespace:
    # Factory defaults live in one dict so callers can still override them via
    # **overrides ({**defaults, **overrides}: caller wins).
    return make_agentic_obs_args(
        tmp_path,
        run_id="run_test",
        **{
            "dynamic_context_parallel": False,
            "context_parallel_size": 1,
            "n_samples_per_prompt": 1,
            "rollout_batch_size": 1,
            "advantage_estimator": "grpo",
            "agentic_custom_advantage_path": None,
            **overrides,
        },
    )


def _emit_update(args, rollout_id, data, step_id=0, successful=True):
    from relax.utils.agentic_obs import train_hooks

    with train_hooks.observe_train_step(args, rollout_id, step_id):
        train_hooks.capture_training_input(data)
        train_hooks.record_optimizer_result(successful)


def _read_train_samples(path):
    return [sample for record in _read_jsonl(path) for sample in record["samples"]]


@pytest.mark.parametrize("kwargs", [{"tp_rank": 1}, {"last_stage": False}])
def test_train_scalars_only_written_by_one_rank_per_dp_shard(tmp_path, fake_mpu, kwargs) -> None:
    # Tensor- and pipeline-parallel peers hold the same broadcast rollout_data;
    # letting them all write would multiply the output by tp * pp.
    from relax.utils.agentic_obs.train_hooks import close_train_appenders

    fake_mpu(**kwargs)
    args = _train_args(tmp_path)
    _emit_update(args, 5, _rollout_data([5]))
    close_train_appenders()

    assert not resolve_agentic_obs_config(args).train_debug_dir.exists()


def test_train_scalars_never_raise_into_the_training_step(tmp_path, fake_mpu) -> None:
    # Losing an observability line must not fail a step that has already run.
    from relax.utils.agentic_obs.train_hooks import close_train_appenders

    fake_mpu()
    args = _train_args(tmp_path)
    broken = _rollout_data([5, 6])
    broken["response_lengths"] = [4]

    _emit_update(args, 7, broken)
    close_train_appenders()

    (record,) = _read_jsonl(resolve_agentic_obs_config(args).train_scalar_path(7, 0))
    assert not record["input_complete"]
    assert record["samples"] == []


@pytest.mark.parametrize("dynamic_cp, cp_size", [(False, 2), (True, 4)], ids=["static-cp", "dynamic-cp"])
def test_train_scalars_preserve_cp_statistics_without_duplicates(tmp_path, fake_mpu, dynamic_cp, cp_size):
    from relax.utils.agentic_obs.train_hooks import close_train_appenders

    args = _train_args(tmp_path, dynamic_context_parallel=dynamic_cp, context_parallel_size=cp_size)
    for cp_rank in range(cp_size):
        fake_mpu(cp_size=cp_size, cp_rank=cp_rank, dp_rank=2)
        _emit_update(args, 1, _rollout_data([7, 8]))
    close_train_appenders()
    close_observation_writers()
    config = resolve_agentic_obs_config(args)
    rows = _read_train_samples(config.train_scalar_path(1, 2))
    assert [row["sample_export_id"] for row in rows] == ["7", "8"]
    # Static CP sees partial sequences; dynamic CP captures the full input.
    for row in rows:
        assert row["dp_rank"] == 2
        assert row["effective_tokens"] == 4
        assert row["response_length"] == 4 and row["total_length"] == 6
        if dynamic_cp:
            assert row["advantage_mean"] == pytest.approx(0.5)
            assert row["advantage_absmax"] == pytest.approx(0.5)
            assert row["returns_mean"] == pytest.approx(2.0)
        else:
            assert row["advantage_mean"] is None
            assert row["advantage_absmax"] is None
            assert row["returns_mean"] is None
    manifest = json.loads(config.manifest_path.read_text())
    assert manifest["cp_size"] == cp_size
    assert manifest["dynamic_context_parallel"] is dynamic_cp


def test_trajectory_media_write_leaves_event_loop_free(tmp_path, monkeypatch):
    from relax.utils.agentic_obs import writers

    args = _obs_args(tmp_path, dump_agentic_obs_media=True)
    config = resolve_agentic_obs_config(args)
    forest, leaf_hash = _two_turn_forest()
    message = forest.lineage(leaf_hash)[1].messages_delta[0]
    message["content"] = [
        {"type": "text", "text": message["content"]},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
    ]
    sample = forest.build_sample(leaf_state_hash=leaf_hash, tokenizer=_FakeTokenizer())
    sample.sample_export_id = 42
    entered = threading.Event()
    release = threading.Event()
    original_write = writers.write_media

    def blocked_media_write(media_dir, media):
        assert threading.current_thread() is not threading.main_thread()
        entered.set()
        assert release.wait(5)
        original_write(media_dir, media)

    monkeypatch.setattr(writers, "write_media", blocked_media_write)

    async def scenario():
        recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
        recorder.record(
            forest=forest,
            state_hash=leaf_hash,
            sample_export_id=sample.sample_export_id,
            metadata=sample.metadata,
            export_name=None,
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            # This coroutine can run while media persistence is still blocked.
            assert not config.trajectory_path(0).exists()
        finally:
            release.set()
            await recorder.aclose()

    asyncio.run(scenario())
    assert next(config.media_dir.iterdir()).read_bytes() == b"ABCD"
    assert _read_jsonl(config.trajectory_path(0))[0]["sample_export_id"] == "42"


@pytest.fixture
def rollout_module(monkeypatch):
    """Exercise real orchestration with its unrelated GPU logging import
    stubbed."""
    import importlib
    from types import ModuleType

    logging_module = ModuleType("relax.distributed.ray.rollout")
    logging_module._log_rollout_data = lambda *args, **kwargs: None
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
    pipeline.prepare_domain.take_ready = lambda count: [stream]
    pipeline.runtime_domain.lease_group = AsyncMock()
    pipeline.runtime_domain.drop_group = AsyncMock()
    pipeline.runtime_domain.collect_group = collect_group
    pipeline._runtime_group_gap = lambda context: 1
    await pipeline._lease_prepared_groups(pipeline._active_step)
    assert len(pipeline._group_observations) == 1
    return pipeline._active_group_tasks[0]


@pytest.mark.parametrize("shutdown", [False, True])
@pytest.mark.parametrize("reward_mode", ["sample", "group", "pending_sibling"])
def test_cancel_records_exports_while_reward_is_pending(tmp_path, rollout_module, monkeypatch, shutdown, reward_mode):
    from relax.agentic.pipeline import reward as reward_module

    args = _obs_args(
        tmp_path,
        group_rm=reward_mode == "group",
        agentic_custom_advantage_path=None,
        use_dynamic_global_batch_size=False,
        partial_rollout=False,
        fully_async=False,
    )

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        rm_started = asyncio.Event()

        async def slow_rm(*args):
            rm_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(reward_module, "async_rm", slow_rm)
        monkeypatch.setattr(reward_module, "batched_async_rm", slow_rm)
        group = _group("in-reward", export_ids=[42, 43], reward=None)
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(group.sessions[0])
        results = [SimpleNamespace(session_id="s0", completion=completion)]
        if reward_mode == "pending_sibling":
            results.append(SimpleNamespace(session_id="s1", completion=asyncio.get_running_loop().create_future()))
        stream = SimpleNamespace(group_id=group.group_id, session_results=tuple(results))

        async def collect_group(stream):
            sessions = await asyncio.gather(*(asyncio.shield(result.completion) for result in stream.session_results))
            return GroupExport(stream.group_id, tuple(sessions))

        task = await _lease_observed_group(
            pipeline, rollout_module, stream, reward_module.RewardDomain(args, "train", None), collect_group
        )
        await asyncio.wait_for(rm_started.wait(), 5)
        if shutdown:
            await pipeline._shutdown()
        else:
            await pipeline._close_rollout_step(pipeline._active_step)
        assert task.cancelled()
        assert not pipeline._active_group_tasks
        assert not pipeline._group_observations
        pipeline._stage_recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert sorted(row["sample_export_id"] for row in rows) == ["42", "43"]
    assert {row["stage"] for row in rows} == {"reward"}
    assert {row["status"] for row in rows} == {"cancelled"}


@pytest.mark.parametrize("drop_stage", ["filter", "custom_advantage"])
def test_cancel_during_drop_record_does_not_duplicate_stage(tmp_path, rollout_module, drop_stage):
    from relax.agentic.pipeline.reward import RewardDomain

    args = _obs_args(tmp_path, group_rm=False, agentic_custom_advantage_path=None)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        group = _group("already-dropped", export_ids=[42])
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(group.sessions[0])
        stream = SimpleNamespace(
            group_id=group.group_id, session_results=(SimpleNamespace(session_id="s0", completion=completion),)
        )
        drop_decided = asyncio.Event()

        def reject(samples):
            drop_decided.set()
            return False

        reward = RewardDomain(args, "train", reject if drop_stage == "filter" else None)
        if drop_stage == "custom_advantage":

            def custom_advantage(payload):
                drop_decided.set()
                return None

            reward.custom_advantage_func = custom_advantage
        # Let the real start be written, then cancel while the decided end is pending.
        release_end = asyncio.Event()
        writing_end = asyncio.Event()
        original_record = pipeline._stage_recorder._record_samples

        async def delayed_record(*a, **kw):
            await original_record(*a, **kw)
            if kw.get("status") == "filtered":
                writing_end.set()
                await release_end.wait()

        pipeline._stage_recorder._record_samples = delayed_record
        task = await _lease_observed_group(pipeline, rollout_module, stream, reward, AsyncMock(return_value=group))
        try:
            await asyncio.wait_for(writing_end.wait(), 5)
            assert drop_decided.is_set()
            close_task = asyncio.create_task(pipeline._cancel_active_group_tasks(rollout_id=7, stage="oversampling"))
            await asyncio.wait_for(close_task, 0.2)
        finally:
            release_end.set()
        await asyncio.wait_for(close_task, 5)
        assert task.cancelled()
        assert not pipeline._group_observations
        pipeline._stage_recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert [row["status"] for row in rows] == ["completed", "filtered"]
    assert rows[-1]["stage"] == "filter"
    assert rows[-1]["operation"] == ("dynamic_filter" if drop_stage == "filter" else "custom_advantage")


@pytest.mark.parametrize("completion_point", ["tq", "pause", "cancel"])
def test_step_close_records_groups_completed_during_barriers(tmp_path, rollout_module, completion_point):
    args = _obs_args(tmp_path, use_dynamic_global_batch_size=False, partial_rollout=False, fully_async=False)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        gate = asyncio.Event()
        started = asyncio.Event()

        async def finish_group():
            started.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                if completion_point != "cancel":
                    raise
            return _group("late", export_ids=[42])

        task = asyncio.create_task(finish_group())
        pipeline._active_group_tasks.append(task)
        await started.wait()

        async def unblock():
            gate.set()
            await task

        if completion_point == "pause":
            pipeline.runtime_domain.pause_generation = unblock
        elif completion_point == "tq":
            pipeline._active_step.tq_put_tasks.append(asyncio.create_task(unblock()))
        await pipeline._close_rollout_step(pipeline._active_step)
        assert not pipeline._active_group_tasks
        assert not pipeline._finalized_groups
        pipeline._stage_recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert [(row["sample_export_id"], row["stage"]) for row in rows] == [("42", "filter")]
    assert rows[0]["operation"] == "oversampling"


def test_shutdown_records_all_completed_owners_before_propagating_error(tmp_path, rollout_module):
    args = _obs_args(tmp_path, use_dynamic_global_batch_size=False, fully_async=False, rollout_batch_size=4)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        pipeline._finalized_groups.append(_group("finalized", export_ids=[1]))
        pipeline.transfer_domain = TransferDomain(args=args)
        pipeline.transfer_domain.open_partition(6, 4)
        assert not pipeline.transfer_domain.pending_groups()
        pipeline.transfer_domain.accept(_group("pending-2", export_ids=[2]))
        pipeline.transfer_domain.accept(_group("pending-4", export_ids=[4]))
        assert [group.group_id for group in pipeline.transfer_domain.pending_groups()] == ["pending-2", "pending-4"]

        async def completed():
            return _group("active", export_ids=[3])

        async def failed():
            raise RuntimeError("group failure")

        pipeline._active_group_tasks = [asyncio.create_task(completed()), asyncio.create_task(failed())]
        await asyncio.gather(*pipeline._active_group_tasks, return_exceptions=True)
        with pytest.raises(RuntimeError, match="group failure"):
            await pipeline._shutdown()
        pipeline.runtime_domain.shutdown.assert_awaited_once()
        assert not pipeline.transfer_domain.pending_groups()
        assert not pipeline._stage_recorder._appenders

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert sorted(row["sample_export_id"] for row in rows) == ["1", "2", "3", "4"]
    assert {row["stage"] for row in rows} == {"filter"}
    assert {row["operation"] for row in rows} == {"shutdown"}


def test_step_close_does_not_wait_for_drop_record_persistence(tmp_path, rollout_module, monkeypatch):
    from relax.utils.agentic_obs import JsonlAppender

    args = _obs_args(tmp_path, use_dynamic_global_batch_size=False, partial_rollout=False, fully_async=False)
    started = threading.Event()
    release = threading.Event()
    original_append = JsonlAppender.append_many

    def blocked_append(self, records):
        started.set()
        assert release.wait(5)
        original_append(self, records)

    monkeypatch.setattr(JsonlAppender, "append_many", blocked_append)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        pipeline._finalized_groups.extend([_group("first", export_ids=[1, 2, -1]), _group("second", export_ids=[3])])
        close_task = asyncio.create_task(pipeline._close_rollout_step(pipeline._active_step))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await asyncio.wait_for(close_task, 0.2)
        finally:
            release.set()
        pipeline._stage_recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert [row["sample_export_id"] for row in rows] == ["1", "2", "3"]
    assert [row["group_id"] for row in rows] == ["first", "first", "second"]
    assert {row["status"] for row in rows} == {"filtered"}
    assert {row["stage"] for row in rows} == {STAGE_FILTER}
    assert {row["reward_value"] for row in rows} == {1.0}
    assert {row["partition_id"] for row in rows} == {None}
    assert {row["reason"] for row in rows} == {"step closed with partition debt already met"}


def test_partial_step_close_keeps_resident_tail(tmp_path, rollout_module):
    args = _obs_args(tmp_path, use_dynamic_global_batch_size=False, partial_rollout=True, fully_async=False)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        pipeline._finalized_groups.append(_group("resident", export_ids=[1]))
        task = asyncio.create_task(asyncio.Event().wait())
        pipeline._active_group_tasks.append(task)
        await pipeline._close_rollout_step(pipeline._active_step)
        assert not task.done()
        assert len(pipeline._finalized_groups) == 1
        assert not resolve_agentic_obs_config(args).stage_path(7).exists()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pipeline._stage_recorder.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("rollout_mode", ["train", "eval"])
def test_shard_export_preserves_rollout_identity(tmp_path, rollout_module, rollout_mode):
    from relax.agentic.session.service import AgenticSessionShard

    args = _obs_args(tmp_path)
    forest, _ = _two_turn_forest()
    shard_type = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_type)
    shard.args = args
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=3)
    shard._generation_backend = SimpleNamespace(tokenizer=_FakeTokenizer())
    group_id = f"{rollout_mode}:dataset_a:7:prepare_group_0"
    session = SimpleNamespace(
        forest=forest, group=SimpleNamespace(group_id=group_id, rollout_mode=rollout_mode), observation=None
    )

    async def scenario():
        transport = shard._build_session_export_transport(
            session=session, reward=1.0, metadata={}, output_records=(), finalize_events={}
        )
        await shard._trajectory_recorder.aclose()
        return transport

    transport = asyncio.run(scenario())
    (record,) = _read_jsonl(resolve_agentic_obs_config(args).trajectory_path(3))
    assert record["group_id"] == group_id
    assert record["rollout_mode"] == rollout_mode
    assert record["sample_export_id"] == str(transport.exports[0]["sample_payload"]["sample_export_id"])


def test_eval_rollout_records_dataset_and_eval_reward(tmp_path, rollout_module, monkeypatch):
    datasets = [SimpleNamespace(name=name, n_samples_per_eval_prompt=1) for name in ("dataset_a", "dataset_b")]
    args = _obs_args(
        tmp_path,
        eval_datasets=datasets,
        reward_key="train",
        eval_reward_key="eval",
        group_rm=False,
        agentic_custom_advantage_path=None,
    )
    samples = {dataset.name: Sample(index=i, group_index=0) for i, dataset in enumerate(datasets)}
    monkeypatch.setattr(rollout_module, "_build_eval_samples", lambda args, dataset: [samples[dataset.name]])
    monkeypatch.setattr(rollout_module, "_agentic_eval_runtime_concurrency", lambda *args: 1)
    runtime = SimpleNamespace(collect_agentic_kv_metrics=AsyncMock(return_value={}), shutdown=AsyncMock())
    monkeypatch.setattr(rollout_module.RuntimeDomain, "connect", AsyncMock(return_value=runtime))

    async def scored_groups(**kwargs):
        index = kwargs["groups"][0][0].index
        group = _group(
            f"{kwargs['scope_id']}:prepare_group_0", export_ids=[index], reward={"train": -1, "eval": index + 0.5}
        )
        group.samples[0].index = index
        await kwargs["stage_recorder"].record_evaluated(group, rollout_id=kwargs["eval_id"])
        return (group,)

    monkeypatch.setattr(rollout_module, "_eval_groups", scored_groups)
    result = asyncio.run(rollout_module.eval_rollout(args, 7))
    config = resolve_agentic_obs_config(args)
    assert not config.stage_path(7).exists()
    rows = _read_stage_ends(config.stage_path(7, rollout_mode="eval"))
    rollout = [row for row in rows if row["scope"] == "rollout"]
    assert len(rollout) == 1 and rollout[0]["status"] == "completed"
    rows = [row for row in rows if row["scope"] == "trajectory"]
    assert {row["stage"] for row in rows} == {"filter"}
    assert [row["group_id"] for row in rows] == [f"eval:{dataset.name}:7:prepare_group_0" for dataset in datasets]
    assert [row["reward_value"] for row in rows] == [0.5, 1.5]
    assert {row["operation"] for row in rows} == {"eval"}
    assert {row["status"] for row in rows} == {"passed"}
    assert result.data["dataset_b"]["rewards"] == [1.5]
    assert runtime.shutdown.await_count == 2


def test_update_membership_masks_and_optimizer_skip(tmp_path, fake_mpu):
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    args = _train_args(tmp_path)
    with train_hooks.observe_train_step(args, 8, 0):
        batch = _rollout_data(
            [2**60 + 1],
            loss_masks=[torch.tensor([1, 0, 1, 0])],
            advantages=[torch.tensor([1.0, 100.0, 3.0, 100.0])],
        )
        train_hooks.capture_training_input(batch)
        batch["loss_masks"][0].zero_()  # immutable input snapshot
        train_hooks.capture_training_input({**_rollout_data([99]), "__is_dummy__": True})
        train_hooks.capture_training_input(_rollout_data([2], loss_masks=[torch.zeros(4)]))
        assert Sample().sample_export_id == -1
        train_hooks.capture_training_input(_rollout_data([Sample().sample_export_id] * 2))
        train_hooks.record_optimizer_result(True)
    _emit_update(args, 8, _rollout_data([3]), step_id=1, successful=False)
    _emit_update(args, 9, _rollout_data([4]), successful=True)
    train_hooks.close_train_appenders()
    close_observation_writers()
    config = resolve_agentic_obs_config(args)
    first, second = _read_jsonl(config.train_scalar_path(8, 0))
    assert first["update_id"] != second["update_id"]
    assert first["schema_version"] == "3.0.0"
    assert first["run_id"] == "run_test"
    assert first["capture_phase"] == "training_input"
    assert [row["sample_export_id"] for row in first["samples"]] == [str(2**60 + 1), "2"]
    assert first["samples"][0]["loss_mask"] == [1, 0, 1, 0]
    assert first["samples"][0]["effective_tokens"] == 2
    assert first["samples"][0]["advantage_mean"] == pytest.approx(2.0)
    assert first["samples"][0]["advantage_absmax"] == pytest.approx(3.0)
    assert first["samples"][1]["effective_tokens"] == 0
    assert first["samples"][1]["advantage_mean"] == pytest.approx(0.0)
    stages = [r for path in config.stages_dir.glob("actor_*.jsonl") for r in _read_jsonl(path)]
    inputs = [r for path in config.train_debug_dir.glob("*.jsonl") for r in _read_jsonl(path)]
    merged = {r["update_id"]: r for r in merge_train_updates(stages, inputs)}
    committed, skipped = merged["actor:8:0"], merged["actor:8:1"]
    assert committed["status"] == "committed" and committed["optimizer_step"] == 1
    assert skipped["status"] == "failed" and skipped["optimizer_step"] is None
    assert committed["group_ids"] == ["group-1152921504606846977", "group-2"]
    assert not committed["groups_finalized"]  # Unassigned rows cannot establish membership.
    assert not committed["input_complete"]
    assert merged["actor:9:0"]["optimizer_step"] == 2


def test_update_joins_remote_dp_members_even_when_local_input_is_empty(tmp_path, fake_mpu, monkeypatch):
    from relax.utils.agentic_obs import train_hooks

    mpu = fake_mpu()
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda *a, **kw: pytest.fail("unexpected gather"))
    args = _train_args(tmp_path)
    with train_hooks.observe_train_step(args, 1, 0):
        train_hooks.capture_training_input({**_rollout_data([99]), "__is_dummy__": True})
        train_hooks.record_optimizer_result(True)
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda **kwargs: 1)
    _emit_update(args, 1, _rollout_data([2], agentic_group_ids=["remote-group"]))
    train_hooks.close_train_appenders()
    close_observation_writers()
    config = resolve_agentic_obs_config(args)
    stages = [r for path in config.stages_dir.glob("actor_*.jsonl") for r in _read_jsonl(path)]
    inputs = [r for path in config.train_debug_dir.glob("*.jsonl") for r in _read_jsonl(path)]
    (event,) = merge_train_updates(stages, inputs)
    assert event["status"] == "committed"
    assert event["group_ids"] == ["remote-group"] and event["groups_finalized"]
    assert _read_jsonl(config.train_scalar_path(1, 0))[0]["samples"] == []


def test_update_exception_does_not_run_a_collective_or_commit(tmp_path, fake_mpu, monkeypatch):
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda *a, **kw: pytest.fail("unexpected gather"))
    args = _train_args(tmp_path)
    with pytest.raises(RuntimeError, match="forward failed"):
        with train_hooks.observe_train_step(args, 1, 0):
            train_hooks.capture_training_input(_rollout_data([1]))
            raise RuntimeError("forward failed")
    train_hooks.close_train_appenders()
    close_observation_writers()
    config = resolve_agentic_obs_config(args)
    assert _read_jsonl(config.train_scalar_path(1, 0))[0]["samples"]
    (event,) = _read_stage_ends(config.stages_dir / "actor_00000001_dp_000.jsonl")
    assert event["status"] == "failed" and event["reason_code"] == "train_step_failed"
    assert not event["groups_finalized"] and not event["input_complete"]
    assert train_hooks._CURRENT.get() is None


def test_train_snapshot_backpressure_preserves_optimizer_event(tmp_path, fake_mpu, monkeypatch):
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    monkeypatch.setattr(train_hooks, "_PENDING", threading.BoundedSemaphore(1))
    entered, release = threading.Event(), threading.Event()
    original_write = train_hooks._write_update

    def delayed(state):
        if state.reserved:
            entered.set()
            assert release.wait(5)
        original_write(state)

    monkeypatch.setattr(train_hooks, "_write_update", delayed)
    args = _train_args(tmp_path)
    try:
        _emit_update(args, 1, _rollout_data([1]))
        assert entered.wait(5)
        _emit_update(args, 1, _rollout_data([2]), step_id=1)
    finally:
        release.set()
        train_hooks.close_train_appenders()
    close_observation_writers()
    config = resolve_agentic_obs_config(args)
    overflow = next(
        row
        for row in _read_stage_ends(config.stages_dir / "actor_00000001_dp_000.jsonl")
        if row["update_id"] == "actor:1:1"
    )
    assert overflow["status"] == "completed" and overflow["optimizer_successful"]
    assert not overflow["input_complete"]
    assert overflow["group_ids"] == ["group-2"]
    status = json.loads((config.train_debug_dir / "producer_dp_000.status.json").read_text())
    assert status["updates_dropped"] == 1


def test_train_metadata_submission_failure_releases_slot_and_finishes_stage(tmp_path, fake_mpu, monkeypatch):
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    monkeypatch.setattr(train_hooks, "_PENDING", threading.BoundedSemaphore(0))
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(train_hooks, "_METADATA_PENDING", slots)

    class ClosedExecutor:
        def submit(self, *args, **kwargs):
            raise RuntimeError("executor is shut down")

    args = _train_args(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(train_hooks, "_METADATA_EXECUTOR", ClosedExecutor())
        _emit_update(args, 7, _rollout_data([7]))
        assert slots.acquire(blocking=False)
        slots.release()
    train_hooks.close_train_appenders()
    (event,) = _read_stage_ends(resolve_agentic_obs_config(args).stages_dir / "actor_00000007_dp_000.jsonl")
    assert event["status"] == "completed" and event["optimizer_successful"] and not event["input_complete"]
    assert event["group_ids"] == ["group-7"]


def test_train_optimizer_outcome_does_not_call_membership_collective(tmp_path, fake_mpu, monkeypatch):
    from relax.utils.agentic_obs import train_hooks

    mpu = fake_mpu()
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    group = object()
    monkeypatch.setattr(mpu, "get_data_parallel_group_gloo", lambda **kwargs: group, raising=False)
    calls = []

    def gather(*args, **kwargs):
        calls.append(kwargs["group"])
        raise RuntimeError("gather failed")

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    args = _train_args(tmp_path)
    _emit_update(args, 7, _rollout_data([7]))
    train_hooks.close_train_appenders()
    assert calls == []
    (event,) = _read_stage_ends(resolve_agentic_obs_config(args).stages_dir / "actor_00000007_dp_000.jsonl")
    assert event["status"] == "completed" and event["optimizer_successful"]
    assert event["groups_finalized"] and event["input_complete"]
    assert event["group_ids"] == ["group-7"] and "optimizer_step" not in event


def test_train_stage_concurrent_initialization_drains_every_record(tmp_path, fake_mpu, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from relax.utils.agentic_obs import ExportStageRecord, train_hooks
    from relax.utils.agentic_obs.writers import BoundedWorker

    entered = threading.Barrier(9)
    release_factory = threading.Event()
    factory_started = threading.Event()
    executors = []

    def create_executor(**kwargs):
        factory_started.set()
        assert release_factory.wait(5)
        executor = BoundedWorker(**kwargs)
        executors.append(executor)
        return executor

    args = _train_args(tmp_path)
    state = train_hooks._Update(args, resolve_agentic_obs_config(args), 7, "actor:7:0", 0, 1, 1)
    monkeypatch.setattr(train_hooks, "BoundedWorker", create_executor)

    def submit(index):
        entered.wait(timeout=5)
        train_hooks._submit_stage(state, ExportStageRecord(index, "train", "start", "g"))

    try:
        with ThreadPoolExecutor(max_workers=8) as callers:
            futures = [callers.submit(submit, i) for i in range(8)]
            entered.wait(timeout=5)
            try:
                assert factory_started.wait(5)
            finally:
                release_factory.set()
            for future in futures:
                future.result(timeout=5)
        train_hooks.close_train_appenders()
        assert len(executors) == 1
        rows = _read_jsonl(state.config.stages_dir / "actor_00000007_dp_000.jsonl")
        assert {row["sample_export_id"] for row in rows} == {str(i) for i in range(8)}
    finally:
        release_factory.set()
        for executor in executors:
            executor.close()


@pytest.mark.parametrize("layout", ["dense", "jagged"])
def test_runtime_group_ids_survive_tq_shapes(layout):
    from relax.utils.agentic_obs.records import decode_group_ids
    from relax.utils.utils import dict_to_tensordict

    groups = ["train:组:1", "eval:dataset:group:2000"]
    rows = [list(group.encode("utf-8")) + [0] for group in groups]
    if layout == "dense":
        values = dict_to_tensordict({"agentic_group_ids": rows}, 2)["agentic_group_ids"]
    else:
        values = torch.nested.as_nested_tensor(
            [torch.tensor(row, dtype=torch.uint8) for row in rows], layout=torch.jagged
        )
    assert decode_group_ids(values) == groups


@pytest.mark.parametrize("put_started", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_transfer_failure_distinguishes_rejection_from_unknown(
    tmp_path, rollout_module, monkeypatch, put_started, cancelled
):
    args = _obs_args(tmp_path)

    async def transfer(**kwargs):
        if put_started:
            kwargs["on_put_started"]()
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("transfer unavailable")

    monkeypatch.setattr(rollout_module, "_transfer_batch_to_data_system", transfer)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await pipeline._transfer_and_record(pipeline._active_step, 6, (_group("g", export_ids=[7, 8]),), False)
        pipeline._stage_recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert {row["sample_export_id"] for row in rows} == {"7", "8"}
    assert all(row["is_terminal"] == (not put_started) for row in rows)
    assert {row["reason_code"] for row in rows} == {
        "transfer_outcome_unknown" if put_started else "transfer_not_submitted"
    }
    assert {row["category"] for row in rows} == {"infra"}


def test_controlled_sibling_failure_retires_existing_exports(tmp_path, rollout_module):
    from relax.agentic.pipeline.reward import RewardDomain

    args = _obs_args(tmp_path, group_rm=True, agentic_custom_advantage_path=None)

    async def scenario():
        group = _group("g", export_ids=[7, 8])
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(group.sessions[0])
        stream = SimpleNamespace(
            group_id="g", session_results=(SimpleNamespace(session_id="s", completion=completed),)
        )
        runtime = SimpleNamespace(collect_group=AsyncMock(return_value=None), drop_group=AsyncMock())
        recorder = StageRecorder(args)
        result = await rollout_module._run_group(
            runtime, RewardDomain(args, "train", None), stream, stage_recorder=recorder, rollout_id=7
        )
        assert result is None
        recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert {row["sample_export_id"] for row in rows} == {"7", "8"}
    assert all(row["reason_code"] == "group_member_failed" and row["is_terminal"] for row in rows)


def test_completed_group_rewards_are_observed(tmp_path, rollout_module, monkeypatch):
    from relax.agentic.pipeline import reward as reward_module

    args = _obs_args(tmp_path, group_rm=True, agentic_custom_advantage_path=None)
    monkeypatch.setattr(reward_module, "batched_async_rm", AsyncMock(return_value=[1.0, 2.0]))

    async def scenario():
        group = _group("g", export_ids=[7, 8], reward=None)
        recorder = StageRecorder(args)

        def observed(samples):
            return recorder.observe(samples, rollout_id=7, group_id="g", stage="reward", operation="rm")

        await reward_module.RewardDomain(args, "train", None).score_group(group, observe_reward=observed)
        assert [sample.reward for sample in group.samples] == [1.0, 2.0]
        recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert [row["reward_value"] for row in rows] == [1.0, 2.0]
    assert all(row["stage"] == "reward" and not row["is_terminal"] for row in rows)


@pytest.mark.parametrize("failure_phase", ["validation", "compaction", "routing"])
def test_multiple_exports_publish_only_after_all_payloads_succeed(
    tmp_path, rollout_module, monkeypatch, failure_phase
):
    from relax.agentic.session import service

    args = _obs_args(tmp_path, use_rollout_routing_replay=failure_phase == "routing", num_layers=2, moe_router_topk=1)
    forest, leaf_hash = _two_turn_forest()
    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=3)
    shard._generation_backend = SimpleNamespace(
        tokenizer=_FakeTokenizer(), compiler=SimpleNamespace(apply_chat_template_kwargs={})
    )
    session = SimpleNamespace(forest=forest, group=SimpleNamespace(group_id="g", rollout_mode="train"))
    hashes = iter([leaf_hash, "missing" if failure_phase == "validation" else leaf_hash])
    monkeypatch.setattr(service, "_messages_tools_template_state_hash", lambda *a: next(hashes))
    original_payload = shard._export_payload
    calls = []

    def payload(**kwargs):
        calls.append(kwargs)
        if failure_phase == "compaction" and len(calls) == 2:
            raise RuntimeError("compaction failed")
        export = original_payload(**kwargs)
        if failure_phase == "routing" and len(calls) == 1:
            sample_payload = export["sample_payload"]
            sample_payload["rollout_routed_experts"] = torch.zeros(
                (len(sample_payload["tokens"]) - 1, args.num_layers, args.moe_router_topk)
            )
        return export

    monkeypatch.setattr(shard, "_export_payload", payload)

    async def scenario():
        with pytest.raises((RuntimeError, service._NonFinalizableExportError)):
            shard._build_session_export_transport(
                session=session,
                reward=1.0,
                metadata={},
                finalize_events={},
                output_records=tuple({"name": name, "metadata": {}, "messages": []} for name in ("one", "two")),
            )
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    assert not resolve_agentic_obs_config(args).trajectory_path(3).exists()
    assert len(calls) == (1 if failure_phase == "validation" else 2)


def test_shard_drop_retires_only_exports_still_owned_by_shard(tmp_path, rollout_module):
    from relax.agentic.pipeline import SessionExportTransport
    from relax.agentic.session import service

    args = _obs_args(tmp_path)
    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=3)
    unclaimed = service._SessionResultCell()
    unclaimed.publish(SessionExportTransport(exports=({"sample_payload": {"sample_export_id": 7}},)))
    delivered = service._SessionResultCell()
    delivered.publish(SessionExportTransport(exports=({"sample_payload": {"sample_export_id": 8}},)))
    delivered.take()
    group = service.ResidentGroup("train", "g", {"unclaimed": unclaimed, "delivered": delivered}, [])
    shard._groups = {"g": group}
    shard._notify_state_change = lambda: None

    async def scenario():
        await shard._drop_group(group)
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    (event,) = _read_stage_ends(resolve_agentic_obs_config(args).stages_dir / "shard_03.jsonl")
    assert event["sample_export_id"] == "7"
    assert event["reason_code"] == "group_discarded_before_delivery" and event["is_terminal"]


@pytest.mark.parametrize("cancelled", [False, True])
def test_runtime_drop_joins_inflight_export_delivery(tmp_path, cancelled):
    from relax.agentic.pipeline import runtime as runtime_module

    async def scenario():
        completion = asyncio.get_running_loop().create_future()
        result = runtime_module._SessionResultRef("s", completion)
        barrier = asyncio.get_running_loop().create_future()
        barrier.set_result(True)
        take_started, release = asyncio.Event(), asyncio.Event()
        group = _group("g", export_ids=[7])
        shard = SimpleNamespace(
            group_streams={}, handle=SimpleNamespace(drop_group=SimpleNamespace(remote=AsyncMock()))
        )
        stream = runtime_module.RuntimeGroupStream("g", shard, barrier, (result,))
        shard.group_streams["g"] = stream
        runtime = object.__new__(runtime_module.RuntimeDomain)
        runtime._resident_group_permits = asyncio.Semaphore(0)
        runtime._state_changed = asyncio.Event()

        async def take():
            take_started.set()
            await release.wait()
            completion.set_result(group.sessions[0])

        result.take_task = asyncio.create_task(take())
        await take_started.wait()
        task = asyncio.create_task(runtime.drop_group(stream))
        while not stream.dropping:
            await asyncio.sleep(0)
        assert not task.done()
        if cancelled:
            task.cancel()
        release.set()
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        shard.handle.drop_group.remote.assert_awaited_once_with("g", reason=None)
        assert completion.result() is group.sessions[0]
        assert not result.take_task.cancelled()

    asyncio.run(scenario())


def test_rollout_membership_cannot_reopen_after_terminal(tmp_path, rollout_module):
    args = _obs_args(tmp_path)

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        record = rollout_module.AgenticResidentPipeline._record_rollout
        context = pipeline._active_step
        await record(pipeline, context)
        context.observed_group_ids.update(("g1", "g2"))
        await record(pipeline, context)  # Membership changes are finalized in the terminal boundary.
        context.terminal_recorded = True
        await record(pipeline, context, status="failed", groups_finalized=True)
        await record(pipeline, context)
        pipeline._stage_recorder.close()

    asyncio.run(scenario())
    rows = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    rows = _read_jsonl(resolve_agentic_obs_config(args).stage_path(7))
    assert [row["status"] for row in rows] == ["start", "failed"]
    assert [row["event_seq"] for row in rows] == [1, 2]
    assert rows[-1]["groups_finalized"] and rows[-1]["group_ids"] == ["g1", "g2"]


def test_eval_success_terminal_survives_cancellation(tmp_path, rollout_module, monkeypatch):
    args = _obs_args(tmp_path)

    async def scenario():
        group = _group("eval:g", export_ids=[7])
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(group.sessions[0])
        stream = SimpleNamespace(group_id=group.group_id, session_results=(SimpleNamespace(completion=completion),))
        runtime = SimpleNamespace(prepare_group=AsyncMock(return_value=stream), lease_group=AsyncMock())
        recorder = StageRecorder(args, rollout_mode="eval")
        entered = asyncio.Event()
        original_record = recorder.record_evaluated

        async def record(*a, **kw):
            await original_record(*a, **kw)
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(recorder, "record_evaluated", record)
        monkeypatch.setattr(rollout_module, "_run_group", AsyncMock(return_value=group))
        task = asyncio.create_task(
            rollout_module._eval_groups(
                7, "eval:dataset", ((Sample(),),), runtime, SimpleNamespace(group_rm=True), SimpleNamespace(), recorder
            )
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await task
        recorder.close()

    asyncio.run(scenario())
    (event,) = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7, rollout_mode="eval"))
    assert (
        event["stage"] == "filter"
        and event["operation"] == "eval"
        and event["status"] == "passed"
        and event["is_terminal"]
    )


def test_cleanup_failure_retires_published_export(tmp_path, rollout_module, monkeypatch):
    from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport
    from relax.agentic.session import service

    args = _obs_args(tmp_path, agentic_session_lifecycle=False)
    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=3)
    shard._generation_backend = SimpleNamespace(abort_request=AsyncMock(side_effect=RuntimeError("abort failed")))
    shard._notify_state_change = lambda group=None: None
    cell = service._SessionResultCell()
    group = service.ResidentGroup("train", "g", {"s": cell}, [])
    session = service._SessionRecord(group=group, session_id="s", session_sampling_params={}, result_cell=cell)
    group.sessions.append(session)
    shard._session_records = {"s": session}
    monkeypatch.setattr(service, "_backend_request_id", lambda ir: "request")

    async def scenario():
        runner = asyncio.create_task(asyncio.sleep(0))
        session.live_irs = [
            SimpleNamespace(
                runner_task=runner, backend_started=True, waiter=SimpleNamespace(set_result=lambda value: None)
            )
        ]
        outcome = SessionExportTransport(exports=({"sample_payload": {"sample_export_id": 7}},))
        error = await shard._finish_session_once(session, outcome)
        assert str(error) == "abort failed" and isinstance(cell.payload, RuntimeGroupError)
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    (event,) = _read_stage_ends(resolve_agentic_obs_config(args).stages_dir / "shard_03.jsonl")
    assert event["sample_export_id"] == "7"
    assert event["reason_code"] == "session_cleanup_failed" and event["is_terminal"]


@pytest.mark.parametrize("extension", ["jsonl", "parquet"])
def test_prompt_source_survives_slices_shuffle_and_replication(tmp_path, extension):
    from relax.utils.data.data import Dataset
    from relax.utils.data.streaming_dataset import StreamingDataset

    path = tmp_path / f"questions.{extension}"
    rows = [{"text": "same", "metadata": {"dataset_id": "questions", "dataset_version": "v1"}} for _ in range(4)]
    if extension == "jsonl":
        path.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n")
        expected_rows = ["3", "5"]
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
        expected_rows = ["1", "2"]
    options = dict(path=f"{path}@[1:3]", tokenizer=None, processor=None, max_length=None, record_source=True)
    eager = Dataset(**options)
    streaming = StreamingDataset(**options)
    eager_sources = [sample.agentic_prompt_source for sample in eager.samples]
    stream_sources = [streaming[i].agentic_prompt_source for i in range(len(streaming))]
    assert eager_sources == stream_sources
    assert [source["source_row_id"] for source in eager_sources] == expected_rows
    assert eager_sources[0]["prompt_id"] != eager_sources[1]["prompt_id"]
    for epoch in (0, 1, 4):
        eager.shuffle(epoch)
        assert {sample.agentic_prompt_source["prompt_id"] for sample in eager.samples} == {
            source["prompt_id"] for source in eager_sources
        }
    assert eager_sources[0]["input_payload"] == rows[0]


@pytest.mark.parametrize(
    ("row_slice", "expected_rows"),
    [("[-3:]", [2, 3, 4]), ("[-4:-1]", [1, 2, 3]), ("[-10:]", [0, 1, 2, 3, 4])],
)
def test_streaming_parquet_slice_preserves_prompt_source(tmp_path, row_slice, expected_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from relax.utils.agentic_obs.source import snapshot_source
    from relax.utils.data.streaming_dataset import StreamingReader

    path = tmp_path / "questions.parquet"
    rows = [{"text": str(i)} for i in range(5)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    unsliced = StreamingReader(str(path), with_source=True)
    reader = StreamingReader(f"{path}@{row_slice}", with_source=True)
    expected_sources = [snapshot_source(unsliced[i], metadata_key="metadata", split="train") for i in expected_rows]

    assert len(reader) == len(expected_rows)
    assert [reader[i].source_row_id for i in range(len(reader))] == [str(i) for i in expected_rows]
    assert [snapshot_source(reader[i], metadata_key="metadata", split="train") for i in range(len(reader))] == (
        expected_sources
    )
    indices = list(reversed(range(len(reader)))) + [0]
    batch = list(reader.iter_batch(indices))
    assert [i for i, _ in batch] == indices
    assert [snapshot_source(row, metadata_key="metadata", split="train") for _, row in batch] == [
        expected_sources[i] for i in indices
    ]

    plain_reader = StreamingReader(f"{path}@{row_slice}")
    plain_rows = [plain_reader[i] for i in range(len(plain_reader))]
    assert plain_rows == [rows[i] for i in expected_rows]
    assert all(type(row) is dict for row in plain_rows)
    assert list(plain_reader.iter_batch(indices)) == [(i, plain_rows[i]) for i in indices]


def test_streaming_parquet_per_file_slices_preserve_prompt_source(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from relax.utils.agentic_obs.source import snapshot_source
    from relax.utils.data.data import Dataset
    from relax.utils.data.streaming_dataset import StreamingDataset

    paths = [tmp_path / f"questions_{i}.parquet" for i in range(2)]
    for path in paths:
        pq.write_table(pa.Table.from_pylist([{"text": "same"} for _ in range(5)]), path, row_group_size=2)
    options = dict(tokenizer=None, processor=None, max_length=None, record_source=True)
    unsliced = Dataset(path=[str(path) for path in paths], **options)
    expected_sources = [unsliced[i].agentic_prompt_source for i in [2, 3, 4, 5, 6, 7, 8, 9]]
    streaming = StreamingDataset(path=[f"{paths[0]}@[-3:]", f"{paths[1]}@[-10:]"], **options)

    assert [streaming[i].agentic_prompt_source for i in range(len(streaming))] == expected_sources
    indices = [7, 0, 3, 2, 0]
    batch = list(streaming.reader.iter_batch(indices))
    assert [i for i, _ in batch] == indices
    assert [snapshot_source(row, metadata_key="metadata", split="train") for _, row in batch] == [
        expected_sources[i] for i in indices
    ]


def test_streaming_prompt_source_survives_reused_line_index(tmp_path, monkeypatch):
    import ray

    from relax.utils.data.streaming_dataset import StreamingReader

    path = tmp_path / "questions.jsonl"
    first = '{"text":"first"}\n'
    path.write_text("\n" + first + '\n{"text":"second"}\n')
    cached_offsets = [1, len(first.encode()) + 2]
    monkeypatch.setattr(ray, "get", lambda _ref: cached_offsets)
    reader = StreamingReader(str(path), with_source=True, line_offsets_cache={str(path): object()})

    assert [reader[i]["text"] for i in range(len(reader))] == ["first", "second"]
    assert [reader[i].source_row_id for i in range(len(reader))] == ["1", "3"]


def test_inventory_replicates_and_resampling_join_original_prompt(tmp_path):
    import copy

    from relax.agentic.inventory import InventoryRecorder
    from relax.utils.data.data import Dataset

    path = tmp_path / "questions.jsonl"
    path.write_text('{"text":"original", "metadata":{"task":"math"}}\n')
    dataset = Dataset(
        str(path), None, None, None, record_source=True, custom_prompt_func=lambda _args, row: "transformed"
    )
    samples = [copy.copy(dataset[0]) for _ in range(2)]
    for i, sample in enumerate(samples):
        sample.group_index, sample.index = 41, 82 + i
    samples[1].sampling_params = {"temperature": 0.2, "max_new_tokens": 17}
    args = _obs_args(tmp_path, rollout_temperature=1.0, rollout_top_p=0.9)

    async def scenario():
        recorder = InventoryRecorder(args, "train")
        recorder.record_group("g1", samples, ["s1", "s2"])
        recorder.record_group("g2", samples, ["s3", "s4"])
        await recorder.aclose()

    asyncio.run(scenario())
    root = resolve_agentic_obs_config(args).root
    (prompt,) = _read_jsonl(next((root / "prompts").glob("*.jsonl")))
    groups = _read_jsonl(next((root / "groups").glob("*.jsonl")))
    sessions = _read_jsonl(next((root / "sessions").glob("*.jsonl")))
    assert prompt["input_payload"]["text"] == "original" and prompt["prompt_preview"] == "transformed"
    assert {row["prompt_id"] for row in groups + sessions} == {prompt["prompt_id"]}
    assert len(groups) == 2 and len(sessions) == 4
    assert [row["replicate_index"] for row in sessions] == [0, 1, 0, 1]
    assert sessions[1]["sampling_config"]["temperature"] == 0.2
    assert sessions[1]["sampling_config"]["top_p"] == 0.9
    assert sessions[1]["sampling_config"]["max_new_tokens"] == 17
    assert set(json.loads((root / "manifest.json").read_text())["streams"]) == {
        "prompts",
        "groups",
        "sessions",
        "trajectories",
        "stages",
        "train",
    }


@pytest.mark.parametrize("remote_unknown", [False, True])
def test_runtime_inventory_covers_prepare_failure_before_any_export(
    tmp_path, rollout_module, monkeypatch, remote_unknown
):
    from relax.agentic.pipeline import GroupInput
    from relax.agentic.pipeline import runtime as runtime_module

    args = _obs_args(tmp_path, rollout_global_dataset=False)

    async def scenario():
        handle = SimpleNamespace(
            start_group=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("prepare broke")))
        )
        binding = runtime_module._SessionShardBinding("shard", handle)
        runtime = runtime_module.RuntimeDomain(args, "eval", [binding], 1)
        runtime._start_shard_observers = lambda: None
        if not remote_unknown:
            monkeypatch.setattr(
                runtime_module,
                "_build_session_specs",
                lambda *a, **kw: (_ for _ in ()).throw(ValueError("invalid input")),
            )
        with pytest.raises((RuntimeError, ValueError)):
            await runtime.prepare_group(GroupInput("g", [Sample(prompt="q") for _ in range(3)]))
        await runtime._inventory.aclose()
        assert runtime._resident_group_permits._value == 1

    asyncio.run(scenario())
    root = resolve_agentic_obs_config(args).root
    sessions = _read_jsonl(next((root / "sessions").glob("*.jsonl")))
    events = _read_jsonl(next((root / "stages").glob("*.jsonl")))
    assert len(_read_jsonl(next((root / "groups").glob("*.jsonl")))) == 1
    assert len(sessions) == 3
    assert len(events) == (3 if remote_unknown else 6)
    assert {s["session_id"] for s in sessions} == {e["session_id"] for e in events}
    for session in sessions:
        pair = sorted((e for e in events if e["session_id"] == session["session_id"]), key=lambda e: e["event_seq"])
        assert pair[0]["status"] == "start" and not pair[0]["is_terminal"]
        assert pair[0]["event_seq"] == 1
        if not remote_unknown:
            assert pair[1]["event_seq"] == 2
            assert pair[1]["status"] == "failed" and pair[1]["is_terminal"]
            assert pair[0]["stage_id"] == pair[1]["stage_id"]
            assert pair[1]["llm_calls"] == [] and pair[1]["llm_calls_complete"]
            assert pair[1]["wall_ms"] >= 0 and pair[1]["token_count_complete"]
    assert not (root / "trajectories").exists()


@pytest.mark.parametrize(
    "prompt,expected",
    [
        (None, []),
        ({"role": "user", "content": "q"}, [{"role": "user", "content": "q"}]),
        ([{"role": "assistant", "content": None}], None),
    ],
    ids=["empty", "user-dict", "empty-assistant"],
)
def test_runtime_sample_messages_validates_each_prompt(prompt, expected):
    from relax.agentic.pipeline.runtime import _sample_messages

    if expected is None:
        with pytest.raises(ValueError, match="content must not be empty"):
            _sample_messages(Sample(prompt=prompt))
    else:
        assert _sample_messages(Sample(prompt=prompt)) == expected


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


@pytest.mark.parametrize("failure_code", ["agent_timeout", "agent_exit_nonzero", "agent_output_invalid"])
def test_session_failure_has_single_terminal_without_export(tmp_path, rollout_module, failure_code):
    from relax.agentic.runner import AgentExecutionError

    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        session.observation.request_count = 3
        session.observation.attempt_count = 4
        session.observation.generated_tokens = 936
        error = AgentExecutionError("agent failed", reason_code=failure_code)
        await asyncio.gather(
            shard._handle_controlled_agent_failure(session, error),
            shard._handle_controlled_agent_failure(session, error),
        )
        shard._observe_session(session, "rollout", "running")
        await shard._trajectory_recorder.aclose()
        assert session.forest is None and not shard._session_records

    asyncio.run(scenario())
    (event,) = _read_jsonl(tmp_path / "run_test/stages/shard_00.jsonl")
    assert event["scope"] == "session" and event["reason_code"] == failure_code
    assert event["is_terminal"] and event["status"] == "failed" and event["category"] == "execution"
    assert (event["request_count"], event["attempt_count"], event["generated_tokens"]) == (3, 4, 936)
    assert event["token_count_complete"] and event["export_ids"] == []
    assert event["log_ref"] is None and event["wall_ms"] >= 0 and event["active_ms"] == 0
    assert not (tmp_path / "run_test/trajectories/shard_00.jsonl").exists()


@pytest.mark.parametrize("late_result", [False, True])
@pytest.mark.parametrize("timing_failure", [False, True])
def test_session_attempt_tokens_include_abort_and_cleanup_result(
    tmp_path, rollout_module, late_result, timing_failure, monkeypatch
):
    from contextlib import asynccontextmanager

    from relax.agentic.pipeline.runtime import BackendGenerateResult
    from relax.agentic.session.state import InflightRequest, RequestKind

    if timing_failure:
        from relax.agentic import observability

        def fail(value):
            raise RuntimeError("timing observation failed")

        monkeypatch.setattr(observability, "math", SimpleNamespace(isfinite=fail))

    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        forest, _ = _two_turn_forest()
        session.forest = forest
        session.observation.begin_call("req")
        session.rollout_id = 7
        shard._terminal_response_locked = lambda **kw: {"ok": True}

        @asynccontextmanager
        async def permit(*args):
            yield

        shard._admission_lease = permit
        shard._sglang_request_permit = permit
        ir = InflightRequest(
            "req",
            forest.root_state_hash,
            7,
            RequestKind.FRESH,
            0,
            asyncio.get_running_loop().create_future(),
            0.0,
            sampling_params={"max_new_tokens": 20},
        )
        session.live_irs.add(ir)
        cleanup = []

        async def generate(**kw):
            if late_result:
                cleanup.append(asyncio.create_task(shard._finish_session(session, None)))
                while session.finish_task is None:
                    await asyncio.sleep(0)
                return BackendGenerateResult([3, 4, 5], [-1.0] * 3, "stop", {"queue_time": 0.03}, 0.1)
            shard._train_generation_open = False
            return BackendGenerateResult([1, 2], [-1.0] * 2, "abort", {"queue_time": 0.02}, 0.1)

        shard._generation_backend.generate = generate
        ir.runner_task = asyncio.create_task(shard._run_ir(session, ir))
        await ir.runner_task
        if late_result:
            await asyncio.gather(*cleanup)
            assert session.result_cell.terminal
        else:
            assert ir.abort_count == 1 and not session.result_cell.terminal
            session.rollout_id = 8
            shard._train_generation_open = True
            shard._generation_backend.generate = AsyncMock(
                return_value=BackendGenerateResult([3, 4, 5], [-1.0] * 3, "stop", {"queue_time": 0.03}, 0.1)
            )
            shard._dispatch_queued_irs_locked(session)
            await ir.runner_task
            await shard._finish_session(session, None)
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    events = _read_jsonl(tmp_path / "run_test/stages/shard_00.jsonl")
    terminal = events[-1]
    assert terminal["is_terminal"] and terminal["token_count_complete"]
    assert terminal["request_count"] == 1
    assert terminal["attempt_count"] == (1 if late_result else 2)
    assert terminal["generated_tokens"] == (3 if late_result else 5)
    assert len(events) == 1 and events[0]["event_seq"] == 2
    (call,) = terminal["llm_calls"]
    assert call["request_id"] == "req" and terminal["llm_calls_complete"]
    assert [attempt["rollout_id"] for attempt in call["attempts"]] == ([7] if late_result else [7, 8])
    assert [attempt["status"] for attempt in call["attempts"]] == (
        ["completed"] if late_result else ["aborted", "completed"]
    )
    assert call["status"] == ("cancelled" if late_result else "completed")
    assert call["started_at"] <= call["finished_at"]
    assert all(attempt["started_at"] <= attempt["finished_at"] for attempt in call["attempts"])

    expected = [0.03] if late_result else [0.02, 0.03]
    for attempt, queue_time in zip(call["attempts"], expected, strict=True):
        assert attempt.get("backend_timing") == (None if timing_failure else {"queue_time": queue_time})


@pytest.mark.parametrize("mode", ["success", "nonzero", "invalid", "timeout"])
@pytest.mark.parametrize("log_failure", [None, "exception", "cancelled"])
def test_managed_agent_logs_and_structured_failure_survive_temp_cleanup(tmp_path, monkeypatch, mode, log_failure):
    from relax.agentic.runner import ipc
    from relax.agentic.runner.ipc import AgentExecutionError, ManagedAgentProcess, ManagedCommandAppSpec, SessionInput

    if log_failure == "exception":

        def fail_copy(*args):
            raise ValueError("log writer failed")

        monkeypatch.setattr(ipc, "_persist_log_tail", fail_copy)
    elif log_failure == "cancelled":
        original_persist = ManagedAgentProcess._persist_observation_log

        async def cancelled_copy(self, source):
            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            await original_persist(self, source)
            await asyncio.sleep(0)  # Deliver cancellation after the nonblocking enqueue.

        monkeypatch.setattr(ManagedAgentProcess, "_persist_observation_log", cancelled_copy)

    async def scenario():
        terminated = asyncio.Event()
        workdirs = []

        async def wait():
            if mode == "timeout":
                await terminated.wait()
            return 1 if mode == "nonzero" else 0

        async def launch(**kw):
            workdirs.append(Path(kw["session_dir"]))
            Path(kw["log_path"]).write_text("agent diagnostics\n")
            if mode == "invalid":
                (workdirs[-1] / "session_output.json").write_text("{bad json")
            return SimpleNamespace(wait=wait, close=AsyncMock(), terminate=AsyncMock(side_effect=terminated.set))

        process = ManagedAgentProcess(
            SimpleNamespace(launch=launch),
            ManagedCommandAppSpec(command="unused", cwd=str(tmp_path), timeout_s=0.01),
            SessionInput(session_id="s", group_id="g", rollout_mode="train"),
        )
        process.configure_observability(tmp_path / "logs/s.log")
        await process.set_timeout_active(True)
        if mode == "success":
            if log_failure == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await process.wait()
            else:
                await process.wait()  # Missing output remains a valid implicit export.
            assert not process.output.records
        else:
            with pytest.raises(AgentExecutionError) as caught:
                await process.wait()
            assert (
                caught.value.reason_code
                == {"nonzero": "agent_exit_nonzero", "invalid": "agent_output_invalid", "timeout": "agent_timeout"}[
                    mode
                ]
            )
        assert process.observation_log_ref == "logs/s.log"
        assert process.active_ms >= 0 and all(not path.exists() for path in workdirs)

    asyncio.run(scenario())
    close_observation_writers()
    if log_failure != "exception":
        assert (tmp_path / "logs/s.log").read_text() == "agent diagnostics\n"


def test_deferred_prepare_does_not_create_failed_or_duplicate_sessions(tmp_path, rollout_module):
    from relax.agentic.pipeline import GroupInput
    from relax.agentic.pipeline import runtime as runtime_module

    args = _obs_args(tmp_path, rollout_global_dataset=False)

    async def scenario():
        binding = runtime_module._SessionShardBinding("shard", None)
        runtime = runtime_module.RuntimeDomain(args, "train", [binding], 1)
        runtime._start_shard_observers = lambda: None
        group_input = GroupInput("g", [Sample(prompt="q"), Sample(prompt="q")])
        for _ in range(2):
            decision = asyncio.get_running_loop().create_future()
            decision.set_result(False)
            assert await runtime.prepare_group(group_input, decision) is None

        async def start_group(mode, group_id, specs):
            binding.group_streams[group_id].first_request_barrier.set_result(True)

        binding.handle = SimpleNamespace(start_group=SimpleNamespace(remote=start_group))
        stream = await runtime.prepare_group(group_input)
        assert stream.group_id == "g"
        await runtime._inventory.aclose()

    asyncio.run(scenario())
    root = resolve_agentic_obs_config(args).root
    assert len(_read_jsonl(next((root / "groups").glob("*.jsonl")))) == 1
    assert len(_read_jsonl(next((root / "sessions").glob("*.jsonl")))) == 2
    starts = _read_jsonl(next((root / "stages").glob("*.jsonl")))
    assert len(starts) == 2 and all(row["status"] == "start" for row in starts)


@pytest.mark.parametrize("dump_media", [False, True])
def test_prompt_binary_media_respects_sidecar_switch(tmp_path, dump_media):
    from relax.agentic.inventory import InventoryRecorder
    from relax.utils.agentic_obs.source import SourceRow, snapshot_source

    data = {"prompt": "image", "image": {"bytes": b"raw-image-bytes", "path": None}}
    sample = Sample(
        prompt="image",
        agentic_prompt_source=snapshot_source(
            SourceRow(data, str(tmp_path / "data.parquet"), 0), metadata_key="metadata", split="train"
        ),
    )

    async def scenario():
        recorder = InventoryRecorder(_obs_args(tmp_path, dump_agentic_obs_media=dump_media), "train")
        recorder.record_group("g", [sample], ["s"])
        await recorder.aclose()

    asyncio.run(scenario())
    root = tmp_path / "run_test"
    (record,) = _read_jsonl(next((root / "prompts").glob("*.jsonl")))
    assert record["input_payload"]["image"]["bytes"].startswith("<media:sha256:")
    media = list((root / "media").glob("*.bin"))
    assert bool(media) == dump_media
    if dump_media:
        assert media[0].read_bytes() == b"raw-image-bytes"


def test_log_copy_does_not_delay_cleanup_or_depend_on_deleted_source(tmp_path, monkeypatch):
    from relax.agentic.runner import ipc

    entered = threading.Event()
    release = threading.Event()
    source = tmp_path / "command.log"
    source.write_text("diagnostics")
    original = ipc._persist_log_tail

    def copy_log(source_path, target):
        entered.set()
        assert release.wait(5)
        return original(source_path, target)

    monkeypatch.setattr(ipc, "_persist_log_tail", copy_log)

    async def scenario():
        process = object.__new__(ipc.ManagedAgentProcess)
        process._observation_log_path = tmp_path / "logs/s.log"
        process.observation_log_ref = None
        task = asyncio.create_task(process._persist_observation_log(source))
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.wait_for(task, 0.1)
        source.unlink()
        release.set()
        close_observation_writers()
        assert process.observation_log_ref == "logs/s.log"

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert (tmp_path / "logs/s.log").read_text() == "diagnostics"


def test_session_cleanup_diagnostics_do_not_change_success(tmp_path, rollout_module):
    from relax.agentic.pipeline import SessionExportTransport

    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        shard.args.agentic_session_lifecycle = True
        shard._lifecycle_close_count = shard._lifecycle_close_failure_count = 0
        shard._generation_backend.close_session = AsyncMock(return_value=False)
        session.observation.export_ids = [(3 << 56) + 1]
        outcome = SessionExportTransport(exports=({"sample_payload": {"sample_export_id": (3 << 56) + 1}},))
        await shard._finish_session(session, outcome)
        assert session.result_cell.payload is outcome
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    (terminal,) = _read_jsonl(tmp_path / "run_test/stages/shard_00.jsonl")
    assert terminal["diagnostics"][0]["reason_code"] == "session_close_failed"
    assert terminal["status"] == "completed" and terminal["is_terminal"]
    assert terminal["export_ids"] == [str((3 << 56) + 1)]


def test_all_six_streams_join_from_prompt_to_committed_update(tmp_path, rollout_module, fake_mpu):
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
    assert trajectory["schema_version"] == "3.0.0" and trajectory["run_id"] == "run_test"
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


def test_inventory_drains_even_when_runtime_cleanup_fails(tmp_path):
    from relax.agentic.pipeline import GroupInput
    from relax.agentic.pipeline.runtime import RuntimeDomain

    async def scenario():
        runtime = RuntimeDomain(_obs_args(tmp_path), "train", [], 1)
        runtime.observe_group_input(GroupInput("g", [Sample(prompt="q")]))
        runtime.trim_memory = AsyncMock(side_effect=RuntimeError("trim failed"))
        with pytest.raises(RuntimeError, match="trim failed"):
            await runtime.shutdown()

    asyncio.run(scenario())
    assert len(_read_jsonl(next((tmp_path / "run_test/groups").glob("*.jsonl")))) == 1


def test_admitted_calls_rejected_before_generation_remain_in_session_timeline(tmp_path, rollout_module):
    from relax.agentic.pipeline import SessionExportTransport

    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        shard._generation_backend.compiler = SimpleNamespace(apply_chat_template_kwargs={})
        session.forest, _ = _two_turn_forest()
        shard._match_parent_state = lambda **kw: (session.forest.lineage(session.forest.root_state_hash)[0], [])
        shard._append_observation = AsyncMock(side_effect=ValueError("encoding rejected"))
        for _ in range(2):
            with pytest.raises(ValueError, match="encoding rejected"):
                await shard.chat(
                    session_id="s",
                    messages=[],
                    tools=[],
                    chat_template_kwargs={},
                    max_completion_tokens=None,
                    stop=None,
                    seed=None,
                )
        assert not session.live_irs
        # An agent can recover from failed calls and finish without exporting them.
        await shard._finish_session(session, SessionExportTransport(exports=()))
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    (event,) = _read_jsonl(tmp_path / "run_test/stages/shard_00.jsonl")
    assert event["status"] == "completed" and event["llm_calls_complete"]
    assert event["request_count"] == 2 and event["attempt_count"] == 0
    assert event["generated_tokens"] == 0 and event["export_ids"] == []
    assert [call["request_id"] for call in event["llm_calls"]] == ["req_s_0", "req_s_1"]
    for index, call in enumerate(event["llm_calls"]):
        assert call["call_index"] == index and call["status"] == "failed"
        assert call["attempts"] == [] and call["started_at"] <= call["finished_at"]
        assert "retry_of_request_id" not in call
    assert "progress" not in event and "phase" not in event


@pytest.mark.parametrize("before_backend", [True, False])
def test_request_infra_failure_closes_call_as_failed(tmp_path, rollout_module, before_backend):
    from contextlib import asynccontextmanager

    from relax.agentic.pipeline import RuntimeGroupError
    from relax.agentic.session.state import InflightRequest, RequestKind

    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        session.observation.begin_call("req")

        @asynccontextmanager
        async def permit(*args):
            if before_backend:
                raise RuntimeGroupError("admission failed")
            yield

        shard._admission_lease = shard._sglang_request_permit = permit
        shard._generation_backend.generate = AsyncMock(side_effect=RuntimeGroupError("backend failed"))
        ir = InflightRequest(
            "req",
            "root",
            7,
            RequestKind.FRESH,
            0,
            asyncio.get_running_loop().create_future(),
            0.0,
            sampling_params={"max_new_tokens": 10},
        )
        session.live_irs.add(ir)
        ir.runner_task = asyncio.create_task(shard._run_ir(session, ir))
        await ir.runner_task
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    (event,) = _read_jsonl(tmp_path / "run_test/stages/shard_00.jsonl")
    (call,) = event["llm_calls"]
    assert event["status"] == call["status"] == "failed"
    assert event["llm_calls_complete"] and call["reason_code"] == "request_infra_failed"
    assert len(call["attempts"]) == (0 if before_backend else 1)
    if not before_backend:
        assert call["attempts"][0]["status"] == "failed"
        assert event["generated_tokens"] is None and not event["token_count_complete"]


@pytest.mark.parametrize("stage", ["reward", "filter"])
@pytest.mark.parametrize("cancelled", [True, False])
def test_stage_failure_is_paired_with_actual_start(tmp_path, stage, cancelled):
    args = _obs_args(tmp_path)
    sample = Sample(sample_export_id=7, reward=123.0)
    recorder = StageRecorder(args)

    async def scenario():
        error_type = asyncio.CancelledError if cancelled else ValueError
        with pytest.raises(error_type):
            async with recorder.observe([sample], rollout_id=7, group_id="g", stage=stage):
                drained = threading.Event()
                assert recorder._worker.submit(drained.set)
                assert await asyncio.to_thread(drained.wait, 5)
                (start,) = _read_jsonl(resolve_agentic_obs_config(args).stage_path(7))
                assert start["status"] == "start"
                raise error_type("operation stopped")
        recorder.close()

    asyncio.run(scenario())
    (end,) = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert end["stage"] == stage and end["status"] == ("cancelled" if cancelled else "failed")
    assert end["is_terminal"]
    if stage == "reward":
        assert end["reward_value"] is None  # failed RM cannot advertise an old score


def test_train_start_does_not_wait_for_tensor_writer(tmp_path, fake_mpu, monkeypatch):
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    args = _train_args(tmp_path)
    path = resolve_agentic_obs_config(args).stages_dir / "actor_00000007_dp_000.jsonl"
    with train_hooks.observe_train_step(args, 7, 0):
        # Drain only the small boundary writer, before training or any tensor submission.
        drained = threading.Event()
        assert train_hooks._STAGE_EXECUTOR.submit(drained.set)
        assert drained.wait(5)
        (start,) = _read_jsonl(path)
        assert start["status"] == "start"
        train_hooks.capture_training_input(_rollout_data([7]))
        train_hooks.record_optimizer_result(True)
    train_hooks.close_train_appenders()
    close_observation_writers()
    (end,) = _read_stage_ends(path)
    assert end["status"] == "completed" and end["optimizer_successful"]
    assert end["stage_id"] == start["stage_id"]


def test_registration_error_after_shard_execution_has_only_one_session_end(tmp_path, rollout_module):
    from relax.agentic.pipeline import GroupInput
    from relax.agentic.pipeline import runtime as runtime_module

    async def scenario():
        shard, _ = _session_observation_fixture(tmp_path)
        shard._groups.clear()
        shard._session_records.clear()
        shard._start_session = lambda *a: (_ for _ in ()).throw(RuntimeError("launch failed"))
        binding = runtime_module._SessionShardBinding(
            "shard", SimpleNamespace(start_group=SimpleNamespace(remote=shard.start_group))
        )
        runtime = runtime_module.RuntimeDomain(
            _obs_args(tmp_path, rollout_global_dataset=False), "train", [binding], 1
        )
        runtime._start_shard_observers = lambda: None
        with pytest.raises(RuntimeError, match="launch failed"):
            await runtime.prepare_group(GroupInput("g", [Sample(prompt="q"), Sample(prompt="q")]))
        assert not shard._groups and not shard._session_records
        await runtime._inventory.aclose()
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())
    events = [row for path in (tmp_path / "run_test/stages").glob("*.jsonl") for row in _read_jsonl(path)]
    assert len(events) == 4
    for session_id in {row["session_id"] for row in events}:
        pair = sorted((row for row in events if row["session_id"] == session_id), key=lambda row: row["event_seq"])
        assert [row["status"] for row in pair] == ["start", "failed"]
        assert pair[0]["stage_id"] == pair[1]["stage_id"]
        assert pair[1]["llm_calls"] == [] and pair[1]["llm_calls_complete"]
        terminal = pair[1]
        assert terminal["is_terminal"] and terminal["reason_code"] == "session_prepare_failed"
        assert terminal["request_count"] == terminal["attempt_count"] == terminal["generated_tokens"] == 0


@pytest.mark.parametrize("reason_code", ["oversampling", "shutdown", "group_member_failed"])
def test_group_drop_preserves_cancellation_reason_and_finishes_once(tmp_path, rollout_module, reason_code):
    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        reason = (reason_code, "selection", "group no longer needed")
        await asyncio.gather(shard.drop_group("g", reason=reason), shard.drop_group("g", reason=reason))
        await shard._trajectory_recorder.aclose()
        assert session.result_cell.terminal

    asyncio.run(scenario())
    (event,) = _read_jsonl(tmp_path / "run_test/stages/shard_00.jsonl")
    assert event["status"] == "cancelled" and event["reason_code"] == reason_code
    assert event["event_seq"] == 2 and event["llm_calls"] == []


def test_call_cancelled_during_compilation_keeps_its_actual_end(tmp_path, rollout_module):
    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        shard._generation_backend.compiler = SimpleNamespace(apply_chat_template_kwargs={})
        session.forest, _ = _two_turn_forest()
        shard._match_parent_state = lambda **kw: (session.forest.lineage(session.forest.root_state_hash)[0], [])
        compiling = asyncio.Event()

        async def compile(**kw):
            compiling.set()
            await asyncio.Event().wait()

        shard._append_observation = compile
        task = asyncio.create_task(
            shard.chat(
                session_id="s",
                messages=[],
                tools=[],
                chat_template_kwargs={},
                max_completion_tokens=None,
                stop=None,
                seed=None,
            )
        )
        await compiling.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        call = session.observation.calls["req_s_0"]
        assert call["status"] == "cancelled" and call["attempts"] == []
        call_end = call["finished_at"]
        assert call_end is not None
        await shard._finish_session(session, None)
        assert call["finished_at"] == call_end
        await shard._trajectory_recorder.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["transfer", "rollout"])
def test_cancellation_does_not_wait_for_stage_disk(tmp_path, rollout_module, monkeypatch, scope):
    args = _obs_args(tmp_path)
    entered, release = threading.Event(), threading.Event()

    async def scenario():
        pipeline = _obs_pipeline(rollout_module, args)
        recorder = pipeline._stage_recorder
        recorder._worker.submit(lambda: (entered.set(), release.wait(5)))
        assert await asyncio.to_thread(entered.wait, 1)
        business_started = asyncio.Event()

        async def transfer(*a, **kw):
            kw["on_put_started"]()
            business_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(rollout_module, "_transfer_batch_to_data_system", transfer)

        async def run_rollout():
            record = rollout_module.AgenticResidentPipeline._record_rollout
            try:
                await record(pipeline, pipeline._active_step)
                business_started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await record(pipeline, pipeline._active_step, status="cancelled", groups_finalized=True)
                raise

        task = asyncio.create_task(
            pipeline._transfer_and_record(pipeline._active_step, 6, (_group("g", export_ids=[7]),), False)
            if scope == "transfer"
            else run_rollout()
        )
        await asyncio.wait_for(business_started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.2)
        release.set()
        await recorder.aclose()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    (end,) = _read_stage_ends(resolve_agentic_obs_config(args).stage_path(7))
    assert end["status"] == "cancelled"
    if scope == "transfer":
        assert end["reason_code"] == "transfer_outcome_unknown" and not end["is_terminal"]
    else:
        assert end["is_terminal"]


@pytest.mark.parametrize("failure", [False, True])
def test_first_request_group_boundaries_preserve_gating_and_cleanup(tmp_path, rollout_module, monkeypatch, failure):
    from relax.agentic.session import service

    async def scenario():
        shard, session = _session_observation_fixture(tmp_path)
        forest, leaf = _two_turn_forest()
        session.forest = forest
        group = session.group
        group.ownership = service.GroupOwnership.PREPARE
        shard._train_generation_open = False
        shard._state_revision = 0
        shard._state_changed = asyncio.Event()
        shard._notify_state_change = type(shard)._notify_state_change.__get__(shard)
        dispatched = []
        shard._dispatch_queued_irs_locked = lambda item: dispatched.append(item.session_id)
        if failure:

            def fail(*args):
                raise OSError("timing write failed")

            monkeypatch.setattr(service, "mark_agentic_event_once", fail)
        shard._notify_state_change(group)
        assert "agentic_trace" not in forest.static_metadata
        # Stand in for the first IR; scheduling/cleanup methods are real.
        session.live_irs.add(object())
        shard._notify_state_change(group)
        await shard.lease_group(group.group_id, 7)
        assert group.ownership is service.GroupOwnership.RUNTIME
        assert session.rollout_id == 7 and not dispatched
        events = forest.static_metadata.get("agentic_trace", {}).get("events", {})
        assert "group_generation_open_at" not in events
        shard._train_generation_open = True
        await shard._open_group(group)
        assert dispatched == [session.session_id]
        if not failure:
            assert (
                events["group_first_requests_ready_at"]
                <= events["group_runtime_leased_at"]
                <= events["group_generation_open_at"]
            )
            before = dict(events)
            await shard._open_group(group)
            assert events == before  # resume must not overwrite first-start boundaries
            sample = forest.build_sample(leaf_state_hash=leaf, tokenizer=_FakeTokenizer())
            assert all(sample.metadata["agentic_trace"]["events"][key] == value for key, value in before.items())
        else:
            assert not events
        session.live_irs.clear()
        await shard._finish_session(session, None)
        await shard._trajectory_recorder.aclose()
        assert session.forest is None and not shard._session_records

    asyncio.run(scenario())


def test_backend_timing_omits_missing_invalid_and_unrelated_fields():
    from relax.agentic.observability import SessionObservation

    observation = SessionObservation()
    attempt = {}
    observation.record_backend_timing(attempt, {})
    assert "backend_timing" not in attempt
    observation.record_backend_timing(
        attempt,
        {
            "queue_time": 0.0,
            "forward_entry_time": 10.0,
            "prefill_finished_time": 0,
            "inference_time": float("nan"),
            "prefill_launch_latency": -1,
            "prefill_waiting_latency": "1",
            "request_finished_ts": True,
            "output_token_logprobs": [1, 2],
        },
    )
    assert attempt["backend_timing"] == {"queue_time": 0.0, "forward_entry_time": 10.0}
