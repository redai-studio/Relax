# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP contract required by the pinned DP-aware SGLang router."""

import ast
import os
from importlib import metadata
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def legacy_load_service():
    if "SGLANG_TEST_SOURCE_ROOT" not in os.environ:
        try:
            version = metadata.version("sglang")
        except metadata.PackageNotFoundError:
            pytest.skip("Requires installed SGLang or SGLANG_TEST_SOURCE_ROOT")
        if Version(version) < Version("0.5.21"):
            pytest.skip("SGLang 0.5.21 migration")
    source = sglang_source_path("srt/entrypoints/http_server.py")
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_load")
    app = fastapi.FastAPI()
    manager = SimpleNamespace(get_loads=AsyncMock(return_value=[]))
    clock = Mock(return_value=123.5)
    namespace = {
        "app": app,
        "_global_state": SimpleNamespace(tokenizer_manager=manager),
        "time": SimpleNamespace(perf_counter=clock),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return app, manager, clock


def _snapshot(rank, running, waiting, total, used):
    return SimpleNamespace(
        dp_rank=rank,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_total_tokens=total,
        num_used_tokens=used,
    )


def test_legacy_load_preserves_router_array_and_dp_token_accounting(legacy_load_service):
    app, manager, clock = legacy_load_service
    manager.get_loads.return_value = [_snapshot(2, 2, 3, 70, 20), _snapshot(7, 0, 4, 100, 60)]
    with TestClient(app) as client:
        response = client.get("/get_load")
    assert response.status_code == 200
    assert response.json() == [
        {
            "dp_rank": 2,
            "num_reqs": 5,
            "num_waiting_reqs": 3,
            "num_tokens": 70,
            "num_pending_tokens": 50,
            "ts_tic": 123.5,
        },
        {
            "dp_rank": 7,
            "num_reqs": 4,
            "num_waiting_reqs": 4,
            "num_tokens": 100,
            "num_pending_tokens": 40,
            "ts_tic": 123.5,
        },
    ]
    # The pinned router sums this field directly from the top-level JSON array.
    assert sum(item["num_tokens"] for item in response.json()) == 170
    manager.get_loads.assert_awaited_once_with(include=["core"])
    clock.assert_called_once_with()


def test_legacy_load_reads_a_fresh_snapshot_for_each_request(legacy_load_service):
    app, manager, clock = legacy_load_service
    manager.get_loads.side_effect = [[_snapshot(0, 1, 2, 64, 32)], [_snapshot(0, 0, 0, 0, 0)]]
    clock.side_effect = [1.5, 9.5]
    with TestClient(app) as client:
        busy = client.get("/get_load").json()
        idle = client.get("/get_load").json()
    assert busy[0]["num_tokens"] == 64 and busy[0]["num_reqs"] == 3
    assert idle == [
        {"dp_rank": 0, "num_reqs": 0, "num_waiting_reqs": 0, "num_tokens": 0, "num_pending_tokens": 0, "ts_tic": 9.5}
    ]
    assert busy[0]["ts_tic"] == 1.5
    assert manager.get_loads.await_args_list == [call(include=["core"]), call(include=["core"])]


def test_legacy_load_allows_no_published_dp_snapshots(legacy_load_service):
    app, manager, _ = legacy_load_service
    with TestClient(app) as client:
        response = client.get("/get_load")
    assert response.status_code == 200
    assert response.json() == []
    manager.get_loads.assert_awaited_once_with(include=["core"])


def test_legacy_load_does_not_report_fake_idle_on_backend_error(legacy_load_service):
    app, manager, _ = legacy_load_service
    manager.get_loads.side_effect = RuntimeError("load snapshots unavailable")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/get_load")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
