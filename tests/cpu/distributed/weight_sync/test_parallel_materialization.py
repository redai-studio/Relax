"""Bounded reconstruction preserves dependency order, repair and byte
budgets."""

import shutil

import pytest

from relax.distributed.weight_sync.store import VersionStore


def empty_cache(source, root):
    shutil.copytree(source.root, root, ignore=shutil.ignore_patterns("materialized"))
    return VersionStore(root, disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)


@pytest.mark.parametrize("compression", [False, True])
def test_parallel_chain_matches_serial_for_full_sparse_copy_and_dense(tmp_path, monkeypatch, compression):
    from relax.distributed.weight_sync import store as module

    source = VersionStore(tmp_path / "source", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    expected = []
    for version in range(4):
        table, contents = {}, {}
        for index in range(8):
            value = bytearray(bytes(range(256)) * 32)
            if version in (1, 2):
                value[index] ^= index + 1
            elif version == 3:
                value[:] = bytes([index + 1]) * len(value)
            path = tmp_path / f"input-{index}"
            path.write_bytes(value)
            table[str(index)] = ("BF16", [4096], path)
            contents[str(index)] = bytes(value)
        source.publish("s", version, table, chunk_bytes=1024, anchor_interval=100, compression=compression)
        expected.append(contents)
    serial = empty_cache(source, tmp_path / "serial")
    parallel = empty_cache(source, tmp_path / "parallel")
    with monkeypatch.context() as patch:
        patch.setattr(module, "ordered_map", lambda function, values, workers: map(function, values))
        reference = serial.materialize("s", 3)
    assert parallel.materialize("s", 3) == reference
    for version, contents in enumerate(expected):
        for store in (serial, parallel):
            manifest = store.manifest("s", version)
            assert store.has_verified_version("s", version, manifest["root"])
            for tensor in manifest["tensors"]:
                assert b"".join(store.iter_tensor(manifest, tensor)) == contents[tensor["name"]]


def test_reconstruction_repairs_corruption_without_rewriting_verified_blocks(tmp_path, monkeypatch):
    source = VersionStore(tmp_path / "source", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    expected = b"".join(bytes([index]) * 1024 for index in range(8))
    path.write_bytes(expected)
    manifest = source.publish("s", 0, {"w": ("U8", [8192], path)}, chunk_bytes=1024)
    broken = manifest["tensors"][0]["chunks"][3]
    source.raw_block_path(broken["output"]).write_bytes(b"x" * broken["size"])
    written, put = [], source._put_raw_block

    def record(value, output):
        written.append(output)
        return put(value, output)

    monkeypatch.setattr(source, "_put_raw_block", record)
    assert source.materialize("s", 0) == manifest
    assert written == [broken["output"]]
    assert b"".join(source.iter_tensor(manifest, manifest["tensors"][0])) == expected


def test_duplicate_reconstructed_blocks_share_one_budget_reservation(tmp_path):
    from relax.distributed.weight_sync.metrics import SyncMetrics
    from relax.distributed.weight_sync.traffic import capture

    source = VersionStore(tmp_path / "source", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    path.write_bytes(b"a" * 8192)
    manifest = source.publish("s", 0, {"w": ("U8", [8192], path)}, chunk_bytes=1024)
    target = empty_cache(source, tmp_path / "target")
    target._remaining_bytes = 1024
    metrics = SyncMetrics("duplicate-reconstruction")
    with capture(metrics):
        assert target.materialize("s", 0) == manifest
    assert target._remaining_bytes == 0
    assert metrics.report()["bytes"]["W"] == 1024
    assert target.has_verified_version("s", 0, manifest["root"])


def test_corrupt_object_fails_and_retained_partial_work_can_retry(tmp_path):
    source = VersionStore(tmp_path / "source", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    path.write_bytes(b"".join(bytes([index]) * 1024 for index in range(8)))
    manifest = source.publish("s", 0, {"w": ("U8", [8192], path)}, chunk_bytes=1024)
    target = empty_cache(source, tmp_path / "target")
    chunk = manifest["tensors"][0]["chunks"][3]
    encoded = target.object_path(chunk["object"]).read_bytes()
    target.object_path(chunk["object"]).write_bytes(b"x" * len(encoded))
    with pytest.raises(ValueError, match="Corrupt object"):
        target.materialize("s", 0)
    assert not target.has_verified_version("s", 0, manifest["root"])
    target.object_path(chunk["object"]).write_bytes(encoded)
    assert target.materialize("s", 0) == manifest
    assert target.has_verified_version("s", 0, manifest["root"])


def test_tensor_cache_fallback_is_converted_into_verified_blocks(tmp_path):
    source = VersionStore(tmp_path / "source", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    path.write_bytes(bytes(range(256)) * 32)
    # Force entry into reconstruction with one genuinely missing output, while
    # the other tensor can still be read from a retained tensor-format cache.
    second = tmp_path / "second"
    second.write_bytes(b"z" * 1024)
    table = {"w": ("U8", [8192], path), "z": ("U8", [1024], second)}
    full = source.publish("other", 0, table, chunk_bytes=1024)
    migrated = empty_cache(source, tmp_path / "migrated")
    old = migrated.tensor_path("other", 0, "w")
    old.parent.mkdir(parents=True)
    old.write_bytes(path.read_bytes())
    assert migrated.materialize("other", 0) == full
    for tensor in full["tensors"]:
        for chunk in tensor["chunks"]:
            assert migrated.raw_block_path(chunk["output"]).is_file()
