# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

import pytest

from relax.distributed.ray.kernel_cache import (
    KernelCacheAgent,
    KernelCacheConfig,
    LocalKernelCacheStore,
    _require_consistent_build_fingerprint,
    derive_local_cache_dir,
    prepare_local_kernel_cache,
)


def _config(shared: Path, local: Path, session: str) -> KernelCacheConfig:
    return KernelCacheConfig(
        shared_dir=str(shared),
        local_dir=str(local),
        session_id=session,
        cache_key="test-profile",
        build_fingerprint="test-build",
        sync_interval_sec=0,
    )


def _write_cache_file(local: Path, relative: str, content: bytes) -> Path:
    path = local / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_cluster_build_fingerprint_requires_identical_gpu_nodes() -> None:
    assert _require_consistent_build_fingerprint({"node-a": "same", "node-b": "same"}) == "same"

    with pytest.raises(RuntimeError, match="differs across GPU nodes"):
        _require_consistent_build_fingerprint({"node-a": "first", "node-b": "second"})

    with pytest.raises(RuntimeError, match="no alive GPU nodes"):
        _require_consistent_build_fingerprint({})


def test_kernel_cache_snapshot_restores_inductor_and_triton(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    source = tmp_path / "source"
    store = LocalKernelCacheStore(_config(shared, source, "session-a"), "node-a")
    _write_cache_file(source, "inductor/ab/kernel.so", b"inductor")
    _write_cache_file(source, "triton/cd/kernel.cubin", b"triton")

    result = store.snapshot("test")

    assert result.published
    assert result.changed_files == 2
    assert result.manifest_path is not None

    shutil.rmtree(source)
    restored = LocalKernelCacheStore(_config(shared, source, "session-b"), "node-b").restore()
    assert restored == 2
    assert (source / "inductor/ab/kernel.so").read_bytes() == b"inductor"
    assert (source / "triton/cd/kernel.cubin").read_bytes() == b"triton"


def test_kernel_cache_snapshot_only_publishes_incremental_changes(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    store = LocalKernelCacheStore(_config(shared, local, "session-a"), "node-a")
    _write_cache_file(local, "inductor/aa/one.py", b"one")

    first = store.snapshot("first")
    unchanged = store.snapshot("unchanged")
    _write_cache_file(local, "triton/bb/two.cubin", b"two")
    second = store.snapshot("second")

    assert first.changed_files == 1
    assert not unchanged.published
    assert second.changed_files == 1
    shard_dir = Path(first.manifest_path).parent
    assert len(list(shard_dir.glob("*.json.ready"))) == 2

    shutil.rmtree(local)
    restored = LocalKernelCacheStore(_config(shared, local, "session-b"), "node-b").restore()
    assert restored == 2
    assert (local / "inductor/aa/one.py").read_bytes() == b"one"
    assert (local / "triton/bb/two.cubin").read_bytes() == b"two"


def test_kernel_cache_restore_preserves_first_conflicting_artifact(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    relative = "triton/key/kernel.json"
    _write_cache_file(local, relative, b"first")
    LocalKernelCacheStore(_config(shared, local, "session-a"), "node-a").snapshot("first")
    shutil.rmtree(local)
    time.sleep(0.001)
    _write_cache_file(local, relative, b"second")
    LocalKernelCacheStore(_config(shared, local, "session-b"), "node-b").snapshot("second")
    shutil.rmtree(local)

    LocalKernelCacheStore(_config(shared, local, "session-c"), "node-c").restore()

    assert (local / relative).read_bytes() == b"first"


def test_kernel_cache_restore_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    config = _config(shared, local, "session-b")
    store = LocalKernelCacheStore(config, "node-b")
    shard = store.shared_dir / "incoming/session-a/node-a"
    shard.mkdir(parents=True)
    archive_path = shard / "000000.tar.ready"
    payload = tmp_path / "payload"
    payload.write_bytes(b"unsafe")
    with tarfile.open(archive_path, "w:") as archive:
        archive.add(payload, arcname="../../escape")
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache_key": config.cache_key,
        "build_fingerprint": config.build_fingerprint,
        "local_dir": config.local_dir,
        "session_id": "session-a",
        "node_id": "node-a",
        "sequence": 0,
        "created_ns": 1,
        "compression": "none",
        "archive": archive_path.relative_to(store.shared_dir).as_posix(),
        "archive_sha256": archive_hash,
        "files": [],
    }
    (shard / "000000.json.ready").write_text(json.dumps(manifest), encoding="utf-8")

    restored = store.restore()

    assert restored == 0
    assert not (tmp_path / "escape").exists()


def test_derive_local_cache_dir_is_stable_and_profile_scoped(tmp_path: Path) -> None:
    first = derive_local_cache_dir(str(tmp_path / "profile-a"))
    second = derive_local_cache_dir(str(tmp_path / "profile-a"))
    other = derive_local_cache_dir(str(tmp_path / "profile-b"))

    assert first == second
    assert first != other
    assert first.startswith("/tmp/relax-kernel-cache/")


def test_kernel_cache_restore_ignores_other_profile(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    _write_cache_file(local, "inductor/aa/kernel.so", b"profile-a")
    LocalKernelCacheStore(_config(shared, local, "session-a"), "node-a").snapshot("first")
    shutil.rmtree(local)

    other_config = KernelCacheConfig(
        shared_dir=str(shared),
        local_dir=str(local),
        session_id="session-b",
        cache_key="other-profile",
        build_fingerprint="test-build",
        sync_interval_sec=0,
    )
    restored = LocalKernelCacheStore(other_config, "node-b").restore()

    assert restored == 0
    assert not (local / "inductor/aa/kernel.so").exists()


def test_kernel_cache_restore_rejects_archive_path_escape(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    config = _config(shared, local, "session-b")
    store = LocalKernelCacheStore(config, "node-b")
    shard = store.shared_dir / "incoming/session-a/node-a"
    shard.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "cache_key": config.cache_key,
        "build_fingerprint": config.build_fingerprint,
        "local_dir": config.local_dir,
        "session_id": "session-a",
        "node_id": "node-a",
        "sequence": 0,
        "created_ns": 1,
        "compression": "none",
        "archive": "../../../../../../outside.tar.ready",
        "archive_bytes": 1,
        "files": [],
    }
    (shard / "000000.json.ready").write_text(json.dumps(manifest), encoding="utf-8")

    assert store.restore() == 0


def test_kernel_cache_snapshot_skips_only_incomplete_triton_group(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    incomplete = _write_cache_file(local, "triton/bad/__grp__kernel.json", b'{"child_paths":{"x":"/missing"}}')
    complete = _write_cache_file(local, "triton/good/kernel.cubin", b"complete")

    result = LocalKernelCacheStore(_config(shared, local, "session-a"), "node-a").snapshot("test")

    assert result.changed_files == 1
    assert incomplete.is_file()
    assert complete.is_file()


def test_kernel_cache_prepare_keeps_incomplete_group_dirty(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    group_dir = local / "triton/group"
    old_child = _write_cache_file(local, "triton/group/old.cubin", b"old")
    new_child = group_dir / "new.cubin"
    group = {
        "child_paths": {
            "old": str(old_child.resolve()),
            "new": str(new_child.resolve()),
        }
    }
    _write_cache_file(local, "triton/group/__grp__kernel.json", json.dumps(group).encode())
    store = LocalKernelCacheStore(_config(shared, local, "session-a"), "node-a")

    store.prepare()
    new_child.write_bytes(b"new")
    result = store.snapshot("complete")

    assert result.changed_files == 3
    shutil.rmtree(local)
    assert LocalKernelCacheStore(_config(shared, local, "session-b"), "node-b").restore() == 3
    assert (local / "triton/group/__grp__kernel.json").is_file()
    assert (local / "triton/group/old.cubin").read_bytes() == b"old"
    assert (local / "triton/group/new.cubin").read_bytes() == b"new"


def test_kernel_cache_restore_rejects_duplicate_tar_member(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    config = _config(shared, local, "session-b")
    store = LocalKernelCacheStore(config, "node-b")
    shard = store.shared_dir / "incoming/session-a/node-a"
    shard.mkdir(parents=True)
    payload = tmp_path / "kernel.so"
    payload.write_bytes(b"payload")
    archive_path = shard / "000000.tar.ready"
    with tarfile.open(archive_path, "w:") as archive:
        archive.add(payload, arcname="inductor/aa/kernel.so")
        archive.add(payload, arcname="inductor/aa/kernel.so")
    file_hash = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "cache_key": config.cache_key,
        "build_fingerprint": config.build_fingerprint,
        "local_dir": config.local_dir,
        "session_id": "session-a",
        "node_id": "node-a",
        "sequence": 0,
        "created_ns": 1,
        "compression": "none",
        "archive": archive_path.relative_to(store.shared_dir).as_posix(),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_bytes": archive_path.stat().st_size,
        "files": [
            {
                "path": "inductor/aa/kernel.so",
                "size": payload.stat().st_size,
                "mtime_ns": payload.stat().st_mtime_ns,
                "mode": 0o600,
                "sha256": file_hash,
            }
        ],
    }
    (shard / "000000.json.ready").write_text(json.dumps(manifest), encoding="utf-8")

    assert store.restore() == 0
    assert not (local / "inductor/aa/kernel.so").exists()


def test_prepare_local_kernel_cache_restores_before_ray_start(tmp_path: Path, monkeypatch) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    config = _config(shared, local, "session-a")
    _write_cache_file(local, "inductor/aa/kernel.so", b"seed")
    LocalKernelCacheStore(config, "source-node").snapshot("seed")
    shutil.rmtree(local)
    monkeypatch.setattr("relax.distributed.ray.kernel_cache.compute_build_fingerprint", lambda: "test-build")

    result = prepare_local_kernel_cache(config, node_id="local-node")

    assert result["restored_files"] == 1
    assert (local / "inductor/aa/kernel.so").read_bytes() == b"seed"
    assert LocalKernelCacheStore(config, "ray-node").verify_prepared()["prepared"]


def test_kernel_cache_attach_rejects_missing_prepared_marker(tmp_path: Path) -> None:
    store = LocalKernelCacheStore(_config(tmp_path / "shared", tmp_path / "local", "session-a"), "node-a")

    with pytest.raises(RuntimeError, match="not prepared successfully"):
        store.verify_prepared()


def test_kernel_cache_agent_rejects_claim_after_finalization_started(tmp_path: Path) -> None:
    config = _config(tmp_path / "shared", tmp_path / "local", "session-a")
    agent = KernelCacheAgent(config.as_dict(), "node-a")
    agent._prepared = True
    agent._stop.set()

    with pytest.raises(RuntimeError, match="not active"):
        agent.claim()
