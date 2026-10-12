# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.env import Envs, validate_env
from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


@pytest.fixture(autouse=True)
def cache_environment(sglang_engine_module, monkeypatch):
    monkeypatch.setattr(sglang_engine_module, "Envs", Envs)


def _args(**overrides):
    return SimpleNamespace(hicache_storage_backend="file", nnodes=1, node_rank=0, dp_size=1, **overrides)


@pytest.mark.parametrize("previous", [None, "existing-parent-setting"])
def test_prefill_cache_directory_isolation_and_recovery(sglang_engine_module, monkeypatch, tmp_path, previous):
    key = "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"
    monkeypatch.setenv("RELAX_HICACHE_FILE_ROOT", str(tmp_path))
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    directories = []
    process = object()
    monkeypatch.setattr(
        sglang_engine_module, "launch_server_process", lambda args: directories.append(os.environ[key]) or process
    )
    for rank in (0, 1, 0):
        assert sglang_engine_module._launch_server_with_file_cache_directory(_args(), "prefill", rank) is process
        assert os.environ.get(key) == previous
    assert directories == [str(tmp_path / f"prefill-engine-{rank:04d}") for rank in (0, 1, 0)]
    assert all(Path(directory).is_dir() for directory in directories)


def test_cache_directory_environment_restored_on_failed_launch(sglang_engine_module, monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_HICACHE_FILE_ROOT", str(tmp_path))
    monkeypatch.setenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", "previous")

    def launch(args):
        raise RuntimeError("failed to start engine")

    monkeypatch.setattr(sglang_engine_module, "launch_server_process", launch)
    with pytest.raises(RuntimeError, match="failed to start"):
        sglang_engine_module._launch_server_with_file_cache_directory(_args(), "prefill", 0)
    assert os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] == "previous"


@pytest.mark.parametrize("field,value", [("nnodes", 2), ("node_rank", 1), ("dp_size", 2)])
def test_cache_root_rejects_unsupported_topology(sglang_engine_module, monkeypatch, tmp_path, field, value):
    monkeypatch.setenv("RELAX_HICACHE_FILE_ROOT", str(tmp_path))
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="single-node DP1"):
        sglang_engine_module._launch_server_with_file_cache_directory(args, "prefill", 0)


@pytest.mark.parametrize("worker,backend", [("decode", "file"), ("regular", "file"), ("prefill", None)])
def test_non_file_prefill_launch_preserves_existing_behavior(
    sglang_engine_module, monkeypatch, tmp_path, worker, backend
):
    monkeypatch.setenv("RELAX_HICACHE_FILE_ROOT", str(tmp_path / "unused"))
    monkeypatch.setenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", "previous")
    args = _args()
    args.hicache_storage_backend = backend
    calls = []
    monkeypatch.setattr(sglang_engine_module, "launch_server_process", calls.append)
    sglang_engine_module._launch_server_with_file_cache_directory(args, worker, 0)
    assert calls == [args]
    assert os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] == "previous"
    assert not (tmp_path / "unused").exists()


def test_cache_root_is_accepted_by_strict_startup_validation(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("RELAX_", "SLIME_")):
            monkeypatch.delenv(name)
    validate_env({"RELAX_STRICT_ENV": "1", "RELAX_HICACHE_FILE_ROOT": str(tmp_path)})
