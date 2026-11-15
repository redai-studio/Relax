# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the agentic observability record and writer layer."""

from __future__ import annotations

import asyncio
import json
import threading
from argparse import Namespace

import pytest

from relax.utils.agentic_obs import (
    AsyncJsonlAppender,
    ExportStageRecord,
    ExportTrajectoryRecord,
    JsonlAppender,
    agentic_obs_enabled,
    pack_sample_export_id,
    resolve_agentic_obs_config,
    unpack_sample_export_id,
)
from relax.utils.agentic_obs.records import encode_group_ids
from relax.utils.agentic_obs.writers import close_observation_writers


@pytest.mark.parametrize("groups", [[], [""], ["group-a", "组-b"]])
def test_group_ids_encode_utf8_with_a_terminator(groups):
    rows = encode_group_ids(groups)
    assert all(row[-1] == 0 for row in rows)
    assert [bytes(row[:-1]).decode("utf-8") for row in rows] == groups


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pack_sample_export_id_roundtrips_per_shard():
    seen = set()
    for shard_ordinal in range(16):
        for counter in (*range(50), 123456, (1 << 40) - 1):
            packed = pack_sample_export_id(shard_ordinal, counter)
            assert unpack_sample_export_id(packed) == (shard_ordinal, counter)
            assert 0 <= packed < 2**63 - 1
            assert packed not in seen
            seen.add(packed)


def test_sample_export_ids_reject_negative():
    with pytest.raises(ValueError):
        pack_sample_export_id(-1, 0)
    with pytest.raises(ValueError):
        pack_sample_export_id(0, -1)
    with pytest.raises(ValueError):
        unpack_sample_export_id(-1)


def test_resolve_agentic_obs_config_disabled_without_a_dump_dir():
    # The path doubles as the switch, matching --dump-details.
    assert agentic_obs_enabled(Namespace(dump_agentic_obs=None)) is False
    assert resolve_agentic_obs_config(Namespace(dump_agentic_obs=None)) is None
    assert resolve_agentic_obs_config(Namespace(dump_agentic_obs="")) is None
    assert resolve_agentic_obs_config(None) is None


def test_resolve_agentic_obs_config_paths(tmp_path):
    args = Namespace(
        dump_agentic_obs=str(tmp_path),
        agentic_obs_run_id="run_x",
        dump_agentic_obs_media=False,
    )
    assert agentic_obs_enabled(args) is True
    config = resolve_agentic_obs_config(args)
    assert config is not None
    assert config.trajectory_path(3).name == "shard_03.jsonl"
    assert config.stage_path(100).name == "rollout_00000100.jsonl"
    # dp_ rather than rank_: tensor/pipeline peers are filtered out upstream.
    assert config.train_scalar_path(100, 2).name == "step_00000100_dp_002.jsonl"
    assert config.root == tmp_path / "run_x"


def test_resolve_agentic_obs_config_falls_back_when_run_id_is_missing(tmp_path):
    # run_id is derived in slime_validate_args, never parsed, so it is only
    # absent when a caller builds a Namespace by hand. Must not crash a run.
    config = resolve_agentic_obs_config(Namespace(dump_agentic_obs=str(tmp_path)))
    assert config is not None
    assert config.root == tmp_path / "run_unknown"
    assert config.dump_media is False


def test_jsonl_appender_survives_unwritable_path(tmp_path):
    # A blocked path must degrade to a logged warning, never raise into training.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    appender = JsonlAppender(blocker / "stages.jsonl")
    appender.append(ExportStageRecord(sample_export_id=1, stage="filter", status="dropped", group_id="g"))
    assert appender.error_count == 1
    appender.close()


@pytest.mark.parametrize("batch", [False, True])
def test_jsonl_appender_warns_once_after_restoring_write_errors(tmp_path, monkeypatch, batch):
    from relax.utils.agentic_obs import writers

    path = tmp_path / "events.jsonl"
    path.mkdir()  # A directory cannot be opened as an append-only JSONL file.
    status_path = path.with_suffix(".status.json")
    status_path.write_text(json.dumps({"write_errors": 3}))
    warnings = []
    monkeypatch.setattr(writers.logger, "warning", lambda *args: warnings.append(args))
    appender = JsonlAppender(path)
    for _ in range(2):
        if batch:
            appender.append_many([{"status": "start"}])
        else:
            appender.append({"status": "start"})
    appender.close()
    assert appender.error_count == 5
    assert json.loads(status_path.read_text())["write_errors"] == 5
    assert len(warnings) == 1


@pytest.mark.parametrize("stream", ["trajectory", "stage"])
def test_jsonl_export_ids_preserve_adjacent_int64_values(tmp_path, stream):
    export_ids = [pack_sample_export_id(15, counter) for counter in (1, 2)]
    records = []
    for export_id in export_ids:
        if stream == "trajectory":
            record = ExportTrajectoryRecord(export_id, "session", None, 0, 0, None, "leaf")
        else:
            record = ExportStageRecord(
                export_id, "transfer", "queued_for_train", "group", partition_id=None if not records else 3
            )
        records.append(record)
    path = tmp_path / f"{stream}.jsonl"
    appender = JsonlAppender(path)
    appender.append(records[0])
    # AsyncJsonlAppender also hands pre-built dictionaries to append_many.
    payload = records[1].to_dict()
    appender.append_many([payload])
    appender.close()
    # parse_int=float models a JSON reader using IEEE 754 numbers. Strings must
    # preserve both the exact values and their distinction across every stream.
    rows = [json.loads(line, parse_int=float) for line in path.read_text().splitlines()]
    assert [row["sample_export_id"] for row in rows] == [str(value) for value in export_ids]
    assert records[0].sample_export_id == export_ids[0]
    assert payload["sample_export_id"] == export_ids[1]
    if stream == "stage":
        assert rows[0]["partition_id"] is None
        assert rows[1]["partition_id"] == 3


def test_async_appender_drops_when_queue_is_full(tmp_path):
    entered, release = threading.Event(), threading.Event()
    appender = AsyncJsonlAppender(tmp_path / "full.jsonl", capacity=2)
    try:
        assert appender._worker.submit(lambda: (entered.set(), release.wait(5)))
        assert entered.wait(1)
        assert appender.submit({"sample_export_id": 1})
        assert not appender.submit({"sample_export_id": 2})
        assert appender.dropped_count == 1
    finally:
        release.set()
        appender.close()
    assert _read_jsonl(tmp_path / "full.jsonl") == [{"sample_export_id": "1"}]


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_async_appender_close_waits_for_inflight_write_once(tmp_path, monkeypatch, cancel_waiter):
    path = tmp_path / "inflight.jsonl"
    started = threading.Event()
    release = threading.Event()
    original_append = JsonlAppender.append_many
    writes = []

    def blocked_append(self, records):
        started.set()
        assert release.wait(5)
        writes.extend(records)
        original_append(self, records)

    monkeypatch.setattr(JsonlAppender, "append_many", blocked_append)

    async def scenario():
        appender = AsyncJsonlAppender(path, capacity=2)
        assert appender.submit({"sample_export_id": 1})
        try:
            assert await asyncio.to_thread(started.wait, 5)
            assert appender.submit({"sample_export_id": 2})
            close_waiter = asyncio.create_task(appender.aclose())
            await asyncio.sleep(0)
            assert not close_waiter.done()
            assert not appender.submit({"sample_export_id": 3})
            if cancel_waiter:
                close_waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await close_waiter
        finally:
            release.set()
        if not cancel_waiter:
            await close_waiter
        await appender.aclose()
        assert appender._appender._handle is None

    asyncio.run(scenario())
    assert [row["sample_export_id"] for row in writes] == ([1] if cancel_waiter else [1, 2])
    assert _read_jsonl(path) == [{"sample_export_id": str(value)} for value in ([1] if cancel_waiter else [1, 2])]


def test_train_jsonl_preserves_nested_ids_and_strict_numbers(tmp_path):
    from relax.utils.agentic_obs import TrainRecord

    path = tmp_path / "train.jsonl"
    writer = JsonlAppender(path)
    writer.append(
        TrainRecord(
            run_id="run-1",
            update_id="actor:3:2",
            dp_rank=0,
            dp_world_size=1,
            samples=[{"sample_export_id": 2**60 + 1, "advantage_mean": float("nan"), "returns_mean": float("inf")}],
        )
    )
    writer.close()
    (row,) = _read_jsonl(path)
    assert row["schema_version"] == "3.1.0"
    assert row["samples"] == [{"sample_export_id": str(2**60 + 1), "advantage_mean": None, "returns_mean": None}]
    assert "NaN" not in path.read_text()
    assert json.loads(path.with_suffix(".status.json").read_text())["records_written"] == 1


@pytest.mark.parametrize("steps_per_epoch, expected_epochs", [(8, 2.5), (None, None), (0, None)])
def test_startup_preserves_unknown_epochs_and_configured_resume(tmp_path, steps_per_epoch, expected_epochs):
    from relax.utils.agentic_obs.startup import record_run_startup

    args = Namespace(
        dump_agentic_obs=str(tmp_path),
        agentic_obs_run_id="run-1",
        num_rollout=20,
        num_rollout_per_epoch=steps_per_epoch,
        start_rollout_id=7,
        api_key="must-not-be-recorded",
    )
    record_run_startup(args)
    record_run_startup(args)  # Service reinitialization creates a new interval.
    close_observation_writers()
    path = tmp_path / "run-1/stages/startup.jsonl"
    first, second, restart, _ = _read_jsonl(path)
    assert first["stage_id"] == second["stage_id"] != restart["stage_id"]
    assert first["record_id"] != second["record_id"]
    assert first["status"] == "start" and second["status"] == "completed"
    assert first["event_seq"] < second["event_seq"]
    assert first["occurred_at"] <= second["occurred_at"]
    assert second["scope"] == "run" and second["stage"] == "startup"
    assert not first["is_terminal"] and not second["is_terminal"]
    assert "sample_export_id" not in second and "rollout_id" not in second
    plan = second["run_config"]
    assert plan["num_epoch"] is None
    assert plan["num_rollout_per_epoch"] == steps_per_epoch
    assert plan["planned_epoch_equivalents"] == expected_epochs
    assert plan["configured_start_rollout_id"] == 7
    assert plan["step_unit"] == "rollout"
    assert "must-not-be-recorded" not in path.read_text()


def test_startup_disabled_and_write_failure_do_not_affect_training(tmp_path):
    from relax.utils.agentic_obs.startup import record_run_startup

    record_run_startup(Namespace())
    assert not list(tmp_path.iterdir())
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    record_run_startup(Namespace(dump_agentic_obs=str(blocker)))


@pytest.fixture(autouse=True)
def drain_observers():
    yield
    close_observation_writers()
