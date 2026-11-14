import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from relax.distributed.weight_sync.checkpoint import validate_checkpoint, write_checkpoint
from relax.distributed.weight_sync.store import VersionStore, digest, stored_bytes
from relax.distributed.weight_sync.transport import SharedFSTransport, fetch


@pytest.fixture
def stores(tmp_path):
    return [
        VersionStore(tmp_path / name, disk_budget=8 << 20, max_weight_bytes=1 << 20, canonical_blocks=True)
        for name in ("source", "target")
    ]


def publish(store, version, value, **kwargs):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "w"
        path.write_bytes(value)
        return store.publish(
            "s",
            version,
            {"w": ("BF16", [len(value) // 2], path)},
            chunk_bytes=64,
            anchor_interval=2,
            model_config={"model_type": "test"},
            **kwargs,
        )


def content(store, manifest):
    return b"".join(store.iter_tensor(manifest, manifest["tensors"][0]))


def test_reader_opened_before_first_publication_adopts_block_format(tmp_path):
    reader = VersionStore(tmp_path, disk_budget=8 << 20, max_weight_bytes=1 << 20)
    writer = VersionStore(tmp_path, disk_budget=8 << 20, max_weight_bytes=1 << 20, canonical_blocks=True)
    manifest = publish(writer, 0, bytes(range(256)))
    assert reader.canonical_blocks
    assert reader.has_verified_version("s", 0, manifest["root"])
    assert content(reader, manifest) == bytes(range(256))


def test_blocks_reuse_unchanged_parts_of_a_changed_tensor(stores, tmp_path):
    source, target = stores
    original = bytes(range(256))
    changed = b"xx" + original[2:]
    first = publish(source, 0, original)
    before = stored_bytes(source.canonical_directory)
    second = publish(source, 1, changed)
    assert stored_bytes(source.canonical_directory) - before == 64
    assert content(source, first) == original
    assert content(source, second) == changed
    assert not list(source.canonical_directory.glob("*/*/*.bin"))
    received = fetch(SharedFSTransport(source), target, "s", 1)
    assert content(target, received) == changed
    checkpoint = write_checkpoint(target, "s", 1, tmp_path / "checkpoint")
    validate_checkpoint(checkpoint, second)
    assert source.ensure_full("s", 1)["root"] == second["root"]
    restarted = VersionStore(source.root, disk_budget=8 << 20, max_weight_bytes=1 << 20)
    assert restarted.canonical_blocks
    assert content(restarted, second) == changed


def test_scattered_changes_pay_for_every_changed_block(stores):
    source, _ = stores
    original = bytes(range(256))
    publish(source, 0, original)
    before = stored_bytes(source.canonical_directory)
    changed = bytearray(original)
    for offset in range(0, 256, 64):
        changed[offset] ^= 128
    m = publish(source, 1, changed)
    assert stored_bytes(source.canonical_directory) - before == len(original)
    assert content(source, m) == changed
    before = stored_bytes(source.canonical_directory)
    same = publish(source, 2, changed)
    assert stored_bytes(source.canonical_directory) == before
    assert same["root"] == m["root"]


def test_gc_preserves_shared_blocks_and_slow_subscriber(stores):
    source, _ = stores
    first = publish(source, 0, bytes(range(256)))
    source.register("slow", "s", 0)
    for version in range(1, 7):
        publish(source, version, bytes([version]) * 64 + bytes(range(64, 256)))
    source.collect()
    assert content(source, first) == bytes(range(256))
    source.unregister("slow")
    report = source.collect()
    assert report["removed_versions"] == 4
    assert report["raw_bytes_freed"] == 4 * 64
    for version in (4, 5, 6):
        assert content(source, source.manifest("s", version)) == bytes([version]) * 64 + bytes(range(64, 256))


def test_corrupt_block_repaired_and_same_target_full_is_self_contained(stores):
    source, target = stores
    publish(source, 0, bytes(range(256)))
    expected = b"aa" + bytes(range(2, 256))
    m = publish(source, 1, expected)
    full = source.ensure_full("s", 1)
    # Start without the delta base or any previous cache content.
    fetched = fetch(SharedFSTransport(source), target, "s", 1, full=True)
    assert fetched["root"] == m["root"] == full["root"]
    target.raw_block_path(fetched["tensors"][0]["chunks"][0]["output"]).write_bytes(b"corrupt")
    assert not target.has_verified_version("s", 1, m["root"])
    target.materialize("s", 1)
    assert content(target, fetched) == expected


def test_failed_publication_is_invisible_and_orphans_are_collected(stores):
    source, _ = stores
    publish(source, 0, bytes(range(256)))
    source.raw_block_path(digest(b"orphan")).write_bytes(b"orphan")
    before = stored_bytes(source.root)
    source.disk_budget = before + 10
    with pytest.raises(OSError):
        publish(source, 1, b"a" * 256)
    assert not source.manifest_path("s", 1).exists()
    source.disk_budget = 8 << 20
    source.collect()
    assert not source.raw_block_path(digest(b"orphan")).exists()
    assert content(source, source.manifest("s", 0)) == bytes(range(256))


def test_process_exit_during_collection_preserves_pinned_and_shared_blocks(stores):
    source, _ = stores
    for version in range(7):
        publish(source, version, bytes([version]) * 64 + bytes(range(64, 256)))
    source.pin("s", 0, "live-reader")
    code = """
import os, sys
from relax.distributed.weight_sync.store import VersionStore
store = VersionStore(sys.argv[1], disk_budget=8 << 20, max_weight_bytes=1 << 20)
delete = store._delete_version
def crash(path, manifest):
    delete(path, manifest)
    os._exit(23)
store._delete_version = crash
store.collect()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source.root)],
        timeout=20,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(Path.cwd()), os.environ.get("PYTHONPATH", "")])},
        capture_output=True,
    )
    assert result.returncode == 23, result.stderr
    reopened = VersionStore(source.root, disk_budget=8 << 20, max_weight_bytes=1 << 20)
    reopened.collect()
    for version in (0, 4, 5, 6):
        expected = bytes([version]) * 64 + bytes(range(64, 256))
        assert content(reopened, reopened.manifest("s", version)) == expected
    assert not list((source.root / "deletions").glob("*.json"))
