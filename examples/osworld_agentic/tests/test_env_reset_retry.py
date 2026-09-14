"""Exercise real reset/close methods without creating remote sandboxes."""

import fcntl
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def env_module():
    source = Path(__file__).resolve().parents[1] / "app" / "env_osworld.py"
    name = "examples.osworld_agentic.app._env_reset_test"
    desktop = ModuleType("desktop_env.desktop_env")
    desktop.DesktopEnv = object
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(os.environ, {"OSWORLD_REPO": "/tmp"}),
        patch.dict(sys.modules, {name: module, "desktop_env.desktop_env": desktop}),
    ):
        spec.loader.exec_module(module)
    return module


def hold_slot(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_SH)
    return fd


def assert_slot_held(path):
    other = os.open(path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(other)


def test_reset_retry_keeps_slot_until_successful_session_closes(env_module):
    env = env_module.OSWorldEnv(task_config={"id": "unrelated"})
    closed = []
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "task.lock")
        fd = hold_slot(path)
        env._env_slot_fd = fd
        calls = []

        def reset_once():
            assert_slot_held(path)
            calls.append(len(calls))
            attempt = len(calls)
            env._env = SimpleNamespace(close=lambda: closed.append(attempt))
            if attempt == 1:
                raise RuntimeError("temporary sandbox upload failure")
            return "observation"

        env._reset_once = reset_once
        with patch.object(env_module.time, "sleep"):
            assert env._reset_sync() == "observation"
        assert len(calls) == 2
        assert closed == [1]
        assert_slot_held(path)
        env._close_sync()
        assert closed == [1, 2]
        with pytest.raises(OSError):
            os.fstat(fd)
        env._close_sync()
        assert closed == [1, 2]


def test_exhausted_reset_releases_slot_and_every_sandbox(env_module):
    env = env_module.OSWorldEnv(task_config={"id": "unrelated"})
    closed = []
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "task.lock")
        fd = hold_slot(path)
        env._env_slot_fd = fd

        def reset_once():
            assert_slot_held(path)
            env._env = SimpleNamespace(close=lambda: closed.append(True))
            raise RuntimeError("persistent setup failure")

        env._reset_once = reset_once
        with patch.object(env_module.time, "sleep") as sleep:
            with pytest.raises(RuntimeError, match="after 3 attempts"):
                env._reset_sync()
        assert len(closed) == 3
        assert sleep.call_count == 2
        with pytest.raises(OSError):
            os.fstat(fd)
        assert env._env_slot_fd is None


def test_close_releases_slot_even_when_sandbox_close_fails(env_module):
    env = env_module.OSWorldEnv(task_config={"id": "unrelated"})
    with tempfile.TemporaryDirectory() as folder:
        fd = hold_slot(str(Path(folder) / "task.lock"))
        env._env_slot_fd = fd

        def close():
            raise RuntimeError("remote close failed")

        env._env = SimpleNamespace(close=close)
        env._close_sync()
        with pytest.raises(OSError):
            os.fstat(fd)
        assert env._env_slot_fd is None
