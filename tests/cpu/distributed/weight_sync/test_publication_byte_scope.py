"""Byte-local hash reuse never survives a write boundary or trusts disk."""

from pathlib import Path

import pytest

from relax.distributed.weight_sync import store as module
from relax.distributed.weight_sync.codec import decode, encode
from relax.distributed.weight_sync.source_patch import plan_replacements


def test_hash_evidence_is_exact_immutable_object_and_ends_with_worker(tmp_path, monkeypatch):
    store = module.VersionStore(
        tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True
    )
    source = tmp_path / "input"
    source.write_bytes(bytes(range(256)) * 4)
    calls = []
    real = module.digest

    def recorded(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(module, "digest", recorded)
    manifest = store.publish("s", 0, {"w": ("U8", [1024], source)}, chunk_bytes=1024)
    full_blocks = [v for v in calls if len(v) == 1024]
    assert len(full_blocks) == 2  # source bytes + independent durable object readback
    assert full_blocks[0] is not full_blocks[1]
    assert module._publication_bytes.get() == ()
    key = manifest["tensors"][0]["chunks"][0]["object"]
    # External insertion still hashes its new buffer and rejects mismatched bytes.
    with pytest.raises(ValueError, match="integrity"):
        store.put_object(key, b"x" * 1024)
    with pytest.raises(ValueError, match="content mismatch"):
        store._put_raw_block(b"x" * 1024, key)


def test_mutable_buffer_never_inherits_byte_evidence():
    value = bytearray(b"old")
    token = module._publication_bytes.set(((value, module.digest(value)),))
    try:
        value[:] = b"new"
        assert module._content_digest(value) == module.digest(b"new")
    finally:
        module._publication_bytes.reset(token)


def test_failed_write_resets_evidence_before_retry(tmp_path, monkeypatch):
    store = module.VersionStore(
        tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True
    )
    path = tmp_path / "input"
    path.write_bytes(b"x" * 1024)
    put = store.put_object
    observed = []

    def failed(key, value):
        observed.append(module._publication_bytes.get())
        raise OSError("injected byte-scope failure")

    monkeypatch.setattr(store, "put_object", failed)
    with pytest.raises(OSError, match="byte-scope failure"):
        store.publish("s", 0, {"w": ("U8", [1024], path)}, encode_workers=1)
    assert observed and observed[0]
    assert module._publication_bytes.get() == ()
    assert not store.manifest_path("s", 0).exists()
    monkeypatch.setattr(store, "put_object", put)
    assert store.publish("s", 0, {"w": ("U8", [1024], path)})["version"] == 0


def test_repeated_object_verified_once_and_corruption_still_rejected(tmp_path, monkeypatch):
    store = module.VersionStore(
        tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True
    )
    path = tmp_path / "input"
    path.write_bytes(b"x" * 1024)
    manifest = store.publish("s", 0, {f"w{i}": ("U8", [1024], path) for i in range(16)})
    key = manifest["tensors"][0]["chunks"][0]["object"]
    obj = store.object_path(key)
    opened = []
    real = Path.read_bytes

    def tracked(path):
        if path == obj:
            opened.append(path)
        return real(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    store.commit(manifest)
    assert len(opened) == 1
    obj.write_bytes(b"z" * 1024)
    with pytest.raises(ValueError, match="corrupt durable object"):
        store.commit(manifest)


@pytest.mark.parametrize("route", ["REUSE", "FULL"])
def test_no_frame_plan_does_not_create_scratch_directory(tmp_path, route):
    previous = tmp_path / "base"
    previous.write_bytes(b"x" * 1024)
    current = tmp_path / "current"
    current.write_bytes((b"x" if route == "REUSE" else b"y") * 1024)
    directory = tmp_path / "unused" / "frames"
    assert plan_replacements(current, previous, 2, directory)["route"] == route
    assert not directory.exists()


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("compression", [False, True])
def test_replacement_bytes_across_larger_comparison_window(width, compression):
    import struct

    window = 1 << 20
    base = bytes((window + 17) * width)
    value = bytearray(base)
    indices = [0, 65535, 65536, window - 1, window, window + 16]
    for index in indices:
        value[index * width] = 0x80
    current = bytes(value)
    kind, payload = encode(current, base, width, compression=compression)
    assert decode(kind, payload, base, width, len(base)) == current
    if not compression:
        expected = b"".join(
            struct.pack("<I", index) + current[index * width : (index + 1) * width] for index in indices
        )
        assert (kind, payload) == ("sparse", expected)
