# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only ClearML startup snapshots; no Ray, SDK network or GPU."""

import importlib.util
import json
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.utils.agentic_obs import tracking, writers
from relax.utils.agentic_obs.startup import record_run_startup


@pytest.fixture
def adapter_class(monkeypatch):
    monkeypatch.setitem(sys.modules, "relax.utils.misc", SimpleNamespace(SingletonMeta=type))
    path = Path(__file__).resolve().parents[3] / "relax/utils/metrics/adapters/clearml.py"
    spec = importlib.util.spec_from_file_location("_clearml_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module._ClearMLAdapter
    monkeypatch.setattr(cls, "_get_task_info", lambda self, args: ("project", "experiment", []))
    monkeypatch.setattr(cls, "_connect_git_metadata", Mock())
    return cls


@pytest.fixture
def args(tmp_path):
    return Namespace(use_clearml=True, dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run-1")


@pytest.fixture(autouse=True)
def drain():
    writers.close_observation_writers()
    yield
    writers.close_observation_writers()


def install_task(monkeypatch, task_id="task-1"):
    task = Mock()
    task.id = task_id
    task.get_output_log_web_page.return_value = f"https://clearml.example/experiments/{task_id}"
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=SimpleNamespace(init=Mock(return_value=task))))
    return task


def test_clearml_binding_uses_existing_startup_stream(args, tmp_path, monkeypatch, adapter_class):
    install_task(monkeypatch)
    adapter_class(args)
    writers.close_observation_writers()
    record_run_startup(Namespace(**vars(args)))
    writers.close_observation_writers()
    records = [json.loads(line) for line in (tmp_path / "run-1/stages/startup.jsonl").read_text().splitlines()]
    assert records[1]["run_config"]["clearml_task"] == {
        "task_id": "task-1",
        "url": "https://clearml.example/experiments/task-1",
    }
    assert not (tmp_path / "run-1/run_metadata").exists()
    manifest = json.loads((tmp_path / "run-1/manifest.json").read_text())
    assert "run_metadata" not in manifest["streams"]


def test_disabled_observation_does_not_access_sdk_link(args, monkeypatch, adapter_class):
    args.dump_agentic_obs = None
    task = install_task(monkeypatch)
    adapter_class(args)
    task.get_output_log_web_page.assert_not_called()


@pytest.mark.parametrize("failure", ["url", "enqueue", "queue_full"])
def test_metadata_failure_preserves_metrics(args, monkeypatch, adapter_class, failure):
    task = install_task(monkeypatch)
    if failure == "url":
        task.get_output_log_web_page.side_effect = OSError("unavailable")
    elif failure == "enqueue":
        monkeypatch.setattr(tracking, "submit_background", Mock(side_effect=OSError("unavailable")))
    else:
        monkeypatch.setattr(tracking, "submit_background", Mock(return_value=False))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(floating=float, integer=int))
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(DataFrame=type("DataFrame", (), {})))
    adapter = adapter_class(args)
    adapter.log({"train/loss": 0.5}, step=7)
    adapter.finish()
    task.close.assert_called_once()
    writers.close_observation_writers()


@pytest.mark.parametrize("stall", ["sdk", "disk"])
def test_stalled_observation_does_not_block_adapter_or_cleanup(args, monkeypatch, adapter_class, stall):
    task = install_task(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def get_url(*unused):
        entered.set()
        release.wait(5)
        return "https://clearml.example/experiments/task-1"

    if stall == "sdk":
        task.get_output_log_web_page.side_effect = get_url
    else:
        monkeypatch.setattr(tracking, "atomic_write_json", get_url)
    try:
        started = time.monotonic()
        adapter = adapter_class(args)
        assert entered.wait(1)
        adapter.finish()
        task.close.assert_called_once()
        writers.close_observation_writers()
        assert time.monotonic() - started < 2
    finally:
        release.set()


def test_restart_snapshots_keep_previous_startup_links(args, tmp_path, monkeypatch, adapter_class):
    for task_id in ("first", "second"):
        install_task(monkeypatch, task_id)
        adapter_class(args)
        writers.close_observation_writers()
        record_run_startup(Namespace(**vars(args)))
        writers.close_observation_writers()
    records = [json.loads(line) for line in (tmp_path / "run-1/stages/startup.jsonl").read_text().splitlines()]
    assert [record["run_config"]["clearml_task"]["task_id"] for record in records[1::2]] == ["first", "second"]
    assert not (tmp_path / "run-1/stages/clearml.jsonl").exists()


@pytest.mark.parametrize("content", [None, "broken json", "[]", '{"task_id": "only-id"}'])
def test_missing_or_invalid_snapshot_preserves_startup(args, tmp_path, content):
    root = tmp_path / "run-1"
    root.mkdir()
    if content is not None:
        (root / "clearml_task.json").write_text(content)
    args.advantage_estimator = "grpo"
    record_run_startup(args)
    writers.close_observation_writers()
    records = [json.loads(line) for line in (root / "stages/startup.jsonl").read_text().splitlines()]
    assert records[1]["run_config"]["advantage_estimator"] == "grpo"
    assert "clearml_task" not in records[1]["run_config"]


def test_snapshot_disk_failure_preserves_metrics_and_cleanup(args, monkeypatch, adapter_class):
    task = install_task(monkeypatch)
    monkeypatch.setattr(writers.os, "replace", Mock(side_effect=OSError("disk unavailable")))
    adapter = adapter_class(args)
    adapter.finish()
    writers.close_observation_writers()
    task.close.assert_called_once()


def test_snapshot_read_failure_preserves_startup(args, tmp_path, monkeypatch):
    original = Path.read_text

    def read(path, *positional, **keywords):
        if path.name == "clearml_task.json":
            raise OSError("unavailable")
        return original(path, *positional, **keywords)

    monkeypatch.setattr(Path, "read_text", read)
    record_run_startup(args)
    writers.close_observation_writers()
    records = [json.loads(line) for line in (tmp_path / "run-1/stages/startup.jsonl").read_text().splitlines()]
    assert "clearml_task" not in records[1]["run_config"]
