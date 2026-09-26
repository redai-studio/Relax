# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Smoke tests for the model upload hooks.

Verifies the fast-return-path & queue backpressure logic of
``push_to_quicksilver`` without touching the real ``model_tools.push_model``
(it's monkey-patched out). The ``push_to_huggingface`` variant is not covered
here.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_HOOK_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "scripts/tools/model_upload_hook.py"


@pytest.fixture
def hook_module(monkeypatch):
    """Load a fresh copy of the hook so per-test env vars take effect."""
    fake_calls: list = []

    def _fake_push_model(**kwargs):
        fake_calls.append(kwargs)

    fake_mt = ModuleType("model_tools")
    fake_mt.push_model = _fake_push_model
    fake_mt.__spec__ = importlib.util.spec_from_loader("model_tools", loader=None)
    monkeypatch.setitem(sys.modules, "model_tools", fake_mt)

    spec = importlib.util.spec_from_file_location("model_upload_hook_under_test", _HOOK_SOURCE)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    hook._fake_calls = fake_calls
    yield hook
    if hook._executor is not None:
        hook._executor.shutdown(wait=True)
    hook._executor = None
    hook._inflight_count = 0


def _args(**overrides):
    base = dict(experiment_name="my-exp")
    base.update(overrides)
    return SimpleNamespace(**base)


class TestFastReturnPath:
    def test_missing_credentials_returns_without_submit(self, monkeypatch, hook_module, caplog):
        monkeypatch.delenv("QS_USER", raising=False)
        monkeypatch.delenv("QS_TOKEN", raising=False)
        hook_module.push_to_quicksilver(_args(), "/tmp/hf/iter_1", 1, dtype="bf16", is_lora=False)
        assert hook_module._fake_calls == []

    def test_missing_model_tools_skips_upload(self, monkeypatch, hook_module):
        monkeypatch.setenv("QS_USER", "u")
        monkeypatch.setenv("QS_TOKEN", "t")
        monkeypatch.setenv("RELAX_QS_MODEL_NAME", "m")
        monkeypatch.setattr(hook_module, "_model_tools_available", lambda: False)
        hook_module.push_to_quicksilver(_args(), "/tmp/hf/iter_1", 1, dtype="bf16", is_lora=False)
        assert hook_module._fake_calls == []
        # Executor must not even be spun up when toolkit is missing.
        assert hook_module._executor is None

    def test_missing_model_name_returns_without_submit(self, monkeypatch, hook_module):
        monkeypatch.setenv("QS_USER", "u")
        monkeypatch.setenv("QS_TOKEN", "t")
        monkeypatch.delenv("RELAX_QS_MODEL_NAME", raising=False)
        args = SimpleNamespace()  # no experiment_name
        hook_module.push_to_quicksilver(args, "/tmp/hf/iter_1", 1, dtype="bf16", is_lora=False)
        assert hook_module._fake_calls == []

    def test_interval_gates_uploads(self, monkeypatch, hook_module, tmp_path):
        monkeypatch.setenv("QS_USER", "u")
        monkeypatch.setenv("QS_TOKEN", "t")
        monkeypatch.setenv("RELAX_QS_MODEL_NAME", "m")
        monkeypatch.setenv("RELAX_QS_UPLOAD_INTERVAL", "5")

        for rid in [1, 2, 3, 4, 5, 6, 10]:
            src = tmp_path / f"iter_{rid}"
            src.mkdir()
            (src / "x").write_text("")
            hook_module.push_to_quicksilver(_args(), str(src), rid, dtype="bf16", is_lora=False)
        _wait_for(hook_module, expected=2, timeout=5.0)
        submitted_rollouts = sorted(c["model_config"]["rollout_id"] for c in hook_module._fake_calls)
        assert submitted_rollouts == [5, 10]  # rid % 5 == 0


class TestUploadSubmission:
    def test_submits_with_expected_arguments(self, monkeypatch, hook_module, tmp_path):
        monkeypatch.setenv("QS_USER", "user-abc")
        monkeypatch.setenv("QS_TOKEN", "tok-xyz")
        monkeypatch.setenv("RELAX_QS_MODEL_NAME", "relax-fp8-actor")
        monkeypatch.setenv("RELAX_QS_ENV", "sandbox")
        monkeypatch.setenv("RELAX_QS_ZONE", "cn")
        monkeypatch.setenv("RELAX_QS_REGION", "tencent-ap-shanghai")

        src = tmp_path / "hf_out" / "iter_9"
        src.mkdir(parents=True)
        (src / "config.json").write_text("{}")

        hook_module.push_to_quicksilver(_args(), str(src), 9, dtype="fp8", is_lora=False)
        _wait_for(hook_module, expected=1, timeout=5.0)

        call = hook_module._fake_calls[0]
        # Upload goes through the snapshot, not the raw path.
        assert call["model_path"].endswith(f".uploading{os.sep}iter_9")
        assert call["model_name"] == "relax-fp8-actor"
        assert call["user"] == "user-abc"
        assert call["token"] == "tok-xyz"
        assert call["env"] == "sandbox"
        assert call["zone"] == "cn"
        assert call["region"] == "tencent-ap-shanghai"
        assert call["precision"] == "fp8"
        assert call["model_format"] == "huggingface"
        assert call["model_config"]["rollout_id"] == 9
        assert call["model_config"]["dtype"] == "fp8"
        # Snapshot must be cleaned up after upload finishes.
        assert not (src.parent / ".uploading" / "iter_9").exists()


class TestSnapshotIsolation:
    def test_link_snapshot_uses_hardlinks(self, hook_module, tmp_path):
        src = tmp_path / "iter_1"
        src.mkdir()
        (src / "config.json").write_text("hello")
        (src / "shard.safetensors").write_bytes(b"\x00" * 1024)

        snapshot_root = tmp_path / ".uploading"
        snap = hook_module._link_snapshot(str(src), str(snapshot_root), rollout_id=1)

        # Same inode (hardlink), not a copy.
        for name in ("config.json", "shard.safetensors"):
            assert os.stat(src / name).st_ino == os.stat(os.path.join(snap, name)).st_ino

    def test_snapshot_survives_source_rmtree(self, hook_module, tmp_path):
        src = tmp_path / "iter_2"
        src.mkdir()
        (src / "shard.safetensors").write_bytes(b"abc")

        snapshot_root = tmp_path / ".uploading"
        snap = hook_module._link_snapshot(str(src), str(snapshot_root), rollout_id=2)

        import shutil as _sh

        _sh.rmtree(src)
        # Snapshot content is still readable via the hardlink refcount.
        assert (Path(snap) / "shard.safetensors").read_bytes() == b"abc"

    def test_link_snapshot_overwrites_stale_dst(self, hook_module, tmp_path):
        src = tmp_path / "iter_3"
        src.mkdir()
        (src / "a.bin").write_bytes(b"new")

        snapshot_root = tmp_path / ".uploading"
        stale = snapshot_root / "iter_3"
        stale.mkdir(parents=True)
        (stale / "leftover.bin").write_bytes(b"garbage")

        snap = hook_module._link_snapshot(str(src), str(snapshot_root), rollout_id=3)
        assert set(os.listdir(snap)) == {"a.bin"}

    def test_cleanup_stale_snapshots_removes_all_iter_dirs(self, hook_module, tmp_path):
        snapshot_root = tmp_path / ".uploading"
        for rid in (1, 2, 5):
            (snapshot_root / f"iter_{rid}").mkdir(parents=True)
            (snapshot_root / f"iter_{rid}" / "x").write_text("")
        hook_module._cleanup_stale_snapshots(str(snapshot_root))
        assert not snapshot_root.exists() or not any(snapshot_root.iterdir())

    def test_snapshot_removed_even_on_upload_failure(self, monkeypatch, hook_module, tmp_path):
        monkeypatch.setenv("QS_USER", "u")
        monkeypatch.setenv("QS_TOKEN", "t")
        monkeypatch.setenv("RELAX_QS_MODEL_NAME", "m")

        src = tmp_path / "hf_out" / "iter_7"
        src.mkdir(parents=True)
        (src / "shard.safetensors").write_bytes(b"data")

        def _boom(**kwargs):
            hook_module._fake_calls.append(kwargs)
            raise RuntimeError("simulated cos failure")

        sys.modules["model_tools"].push_model = _boom

        hook_module.push_to_quicksilver(_args(), str(src), 7, dtype="bf16", is_lora=False)
        _wait_for(hook_module, expected=1, timeout=5.0)
        assert not (src.parent / ".uploading" / "iter_7").exists()


class TestQueueBackpressure:
    def test_queue_full_drops_new_work(self, monkeypatch, hook_module, tmp_path):
        monkeypatch.setenv("QS_USER", "u")
        monkeypatch.setenv("QS_TOKEN", "t")
        monkeypatch.setenv("RELAX_QS_MODEL_NAME", "m")
        monkeypatch.setenv("RELAX_QS_MAX_UPLOAD_WORKERS", "1")
        monkeypatch.setenv("RELAX_QS_UPLOAD_QUEUE_MAX", "2")

        # Replace fake push_model with a blocker so in-flight stays at cap.
        release = _install_blocking_push(hook_module)

        try:
            for rid in [1, 2, 3, 4, 5]:  # 2 accepted, 3 dropped
                src = tmp_path / f"iter_{rid}"
                src.mkdir()
                (src / "x").write_text("")
                hook_module.push_to_quicksilver(_args(), str(src), rid, dtype="bf16", is_lora=False)
            _wait_until(lambda: hook_module._inflight_count >= 2, timeout=2.0)
            assert hook_module._inflight_count == 2, "queue_max should cap in-flight at 2"
        finally:
            release()
        _wait_for(hook_module, expected=2, timeout=5.0)
        assert len(hook_module._fake_calls) == 2


# ---------- helpers -----------------------------------------------------------


class TestFlush:
    def test_flush_blocks_until_pending_uploads_finish(self, monkeypatch, hook_module, tmp_path):
        monkeypatch.setenv("QS_USER", "u")
        monkeypatch.setenv("QS_TOKEN", "t")
        monkeypatch.setenv("RELAX_QS_MODEL_NAME", "m")
        monkeypatch.setenv("RELAX_QS_MAX_UPLOAD_WORKERS", "1")

        # Install a blocking push that we can release manually.
        release = _install_blocking_push(hook_module)

        try:
            src = tmp_path / "iter_1"
            src.mkdir()
            (src / "x").write_text("")
            hook_module.push_to_quicksilver(_args(), str(src), 1, dtype="bf16", is_lora=False)
            _wait_until(lambda: hook_module._inflight_count >= 1, timeout=2.0)
            # Kick flush from a helper thread so we can watch it wait.
            import threading

            flush_returned = threading.Event()

            def _do_flush():
                hook_module.flush(timeout_sec=10.0)
                flush_returned.set()

            t = threading.Thread(target=_do_flush)
            t.start()
            # Flush must not return while push is still blocked.
            assert not flush_returned.wait(timeout=0.3)
            release()
            assert flush_returned.wait(timeout=5.0), "flush should return after upload completes"
        finally:
            release()
            t.join(timeout=5.0)
        assert len(hook_module._fake_calls) == 1
        assert hook_module._inflight_count == 0

    def test_flush_is_noop_when_no_executor(self, hook_module):
        # No push_to_quicksilver has been called yet — executor never initialized.
        assert hook_module._executor is None
        hook_module.flush(timeout_sec=1.0)  # must not raise


# ---------- helpers -----------------------------------------------------------


def _wait_for(hook_module, *, expected: int, timeout: float) -> None:
    """Block until the executor has completed `expected` uploads."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(hook_module._fake_calls) >= expected and hook_module._inflight_count == 0:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"timed out waiting for {expected} uploads; got {len(hook_module._fake_calls)}, "
        f"in-flight={hook_module._inflight_count}"
    )


def _wait_until(predicate, *, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for predicate")


def _install_blocking_push(hook_module):
    """Swap the fake push_model with one that blocks until released."""
    gate = __import__("threading").Event()

    def _blocking_push(**kwargs):
        hook_module._fake_calls.append(kwargs)
        gate.wait(timeout=5.0)

    sys.modules["model_tools"].push_model = _blocking_push
    return gate.set
