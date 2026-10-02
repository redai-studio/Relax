# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/tools/ray_job_preflight.py"
spec = importlib.util.spec_from_file_location("ray_job_preflight", MODULE_PATH)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def idle_snapshot() -> Dict[str, Any]:
    return {
        "nodes": [{"state": "ALIVE", "local": True}],
        "jobs": [{"status": "SUCCEEDED"}],
        "placement_groups": [{"state": "REMOVED"}],
        "applications": {},
        "gpu_processes": [],
    }


def test_preflight_accepts_only_terminal_resources() -> None:
    preflight.assert_idle(idle_snapshot())


@pytest.mark.parametrize("status", ["RUNNING", "PENDING", "UNKNOWN", None])
def test_preflight_rejects_nonterminal_jobs(status: Optional[str]) -> None:
    snapshot = idle_snapshot()
    snapshot["jobs"] = [{"status": status}]
    with pytest.raises(ValueError, match="Ray job"):
        preflight.assert_idle(snapshot)


@pytest.mark.parametrize("field", ["nodes", "jobs", "placement_groups", "applications", "gpu_processes"])
def test_preflight_rejects_missing_inventory(field: str) -> None:
    snapshot = idle_snapshot()
    del snapshot[field]
    with pytest.raises(ValueError, match="missing"):
        preflight.assert_idle(snapshot)


@pytest.mark.parametrize("state", ["CREATED", "PENDING", "RESCHEDULING", None])
def test_preflight_rejects_live_or_unknown_groups(state: Optional[str]) -> None:
    snapshot = idle_snapshot()
    snapshot["placement_groups"] = [{"state": state}]
    with pytest.raises(ValueError, match="placement group"):
        preflight.assert_idle(snapshot)


def test_preflight_rejects_serve_and_gpu_ownership() -> None:
    for field, value in [("applications", {"other-app": {}}), ("gpu_processes", ["12345"])]:
        snapshot = idle_snapshot()
        snapshot[field] = value
        with pytest.raises(ValueError):
            preflight.assert_idle(snapshot)


def test_preflight_rejects_truncated_group_inventory() -> None:
    snapshot = idle_snapshot()
    snapshot["placement_groups"] *= 10000
    with pytest.raises(ValueError, match="truncated"):
        preflight.assert_idle(snapshot)


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [{"state": "ALIVE", "local": False}],
        [{"state": "DEAD", "local": True}],
        [{"state": "UNKNOWN"}],
        [{"state": "ALIVE", "local": True}] * 2,
    ],
)
def test_preflight_rejects_remote_multinode_or_unknown_cluster(nodes: list) -> None:
    snapshot = idle_snapshot()
    snapshot["nodes"] = nodes
    with pytest.raises(ValueError):
        preflight.assert_idle(snapshot)
