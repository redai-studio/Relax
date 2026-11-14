"""Storage scan reduction preserves independent durable copies and repair."""

import os
from pathlib import Path

import pytest

from relax.distributed.weight_sync import checkpoint, seal
from relax.distributed.weight_sync.store import VersionStore


@pytest.fixture
def source(tmp_path):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    path.write_bytes(b"".join(bytes([i]) * 1024 for i in range(12)))
    return store, path, {"w": ("BF16", [6144], path)}


@pytest.mark.parametrize("change", ["rewrite_restore_mtime", "replace", "append", "truncate"])
def test_source_mutation_cannot_commit_or_leave_unreferenced_blocks(source, monkeypatch, change):
    store, path, table = source
    put = store._put_raw_block
    mutated = False

    def mutate(value, output):
        nonlocal mutated
        put(value, output)
        if mutated:
            return
        mutated = True
        status = path.stat()
        data = path.read_bytes()
        if change == "rewrite_restore_mtime":
            path.write_bytes(bytes([data[0] ^ 128]) + data[1:])
            os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
        elif change == "replace":
            replacement = path.with_suffix(".new")
            replacement.write_bytes(data)
            os.utime(replacement, ns=(status.st_atime_ns, status.st_mtime_ns))
            os.replace(replacement, path)
        elif change == "append":
            with path.open("ab") as handle:
                handle.write(b"extra")
        else:
            path.write_bytes(data[:-8])

    monkeypatch.setattr(store, "_put_raw_block", mutate)
    with pytest.raises(ValueError, match="Source changed"):
        store.publish("s", 0, table, chunk_bytes=1024, encode_workers=1, model_config={})
    assert not store.manifest_path("s", 0).exists()
    assert not list((store.root / "objects").iterdir())
    assert not list((store.canonical_directory / "blocks").iterdir())


def test_publisher_reads_source_and_verified_base_once(source, monkeypatch):
    store, path, table = source
    first = store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    value = bytearray(path.read_bytes())
    value[0] ^= 1
    path.write_bytes(value)
    opened, read = [], []
    original_open, original_read = Path.open, store.read_chunk

    def counted_open(self, *args, **kwargs):
        if self == path and args and args[0] == "rb":
            opened.append(self)
        return original_open(self, *args, **kwargs)

    def counted_read(stream, version, name, chunk):
        read.append((version, chunk["offset"]))
        return original_read(stream, version, name, chunk)

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(store, "read_chunk", counted_read)
    store.publish("s", 1, table, chunk_bytes=1024, model_config={})
    chunks = first["tensors"][0]["chunks"]
    assert len(opened) == len(chunks)
    assert sorted(read) == [(0, chunk["offset"]) for chunk in chunks]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_delta_publication_repairs_base_on_demand(source, damage):
    store, path, table = source
    original = path.read_bytes()
    first = store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    chunk = first["tensors"][0]["chunks"][5]
    raw = store.raw_block_path(chunk["output"])
    if damage == "missing":
        raw.unlink()
    else:
        raw.write_bytes(b"x" * chunk["size"])
    path.write_bytes(original[:-2] + b"zz")
    second = store.publish("s", 1, table, chunk_bytes=1024, model_config={})
    assert second["base_root"] == first["root"]
    assert b"".join(store.iter_tensor(first, first["tensors"][0])) == original
    assert b"".join(store.iter_tensor(second, second["tensors"][0])) == path.read_bytes()


def test_raw_and_canonical_remain_independently_repairable(source):
    store, _, table = source
    manifest = store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    chunk = manifest["tensors"][0]["chunks"][0]
    raw, encoded = store.raw_block_path(chunk["output"]), store.object_path(chunk["object"])
    assert raw.stat().st_ino != encoded.stat().st_ino
    encoded.write_bytes(b"x" * chunk["size"])
    assert store.read_chunk("s", 0, "w", chunk) == bytes(chunk["size"])
    assert store.ensure_full("s", 0) == manifest
    assert encoded.read_bytes() == raw.read_bytes()


@pytest.mark.parametrize("damage", ["none", "missing", "corrupt"])
def test_checkpoint_streams_verified_bytes_and_repairs_mid_tensor(source, tmp_path, monkeypatch, damage):
    store, path, table = source
    manifest = store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    chunk = manifest["tensors"][0]["chunks"][5]
    raw = store.raw_block_path(chunk["output"])
    if damage == "missing":
        raw.unlink()
    elif damage == "corrupt":
        raw.write_bytes(b"x" * chunk["size"])
    materialize = store.materialize
    repairs = []

    def count_repair(*args):
        repairs.append(args)
        return materialize(*args)

    monkeypatch.setattr(store, "materialize", count_repair)
    target = checkpoint.write_checkpoint(store, "s", 0, tmp_path / "checkpoint")
    checkpoint.validate_checkpoint(target, manifest)
    data = (target / "model.safetensors").read_bytes()
    assert data[8 + int.from_bytes(data[:8], "little") :] == path.read_bytes()
    assert len(repairs) == int(damage != "none")
    assert not list((store.root / "pins").glob("*.json"))


def test_unrepairable_checkpoint_has_no_destination_stage_or_pin(source, tmp_path):
    store, _, table = source
    manifest = store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    chunk = manifest["tensors"][0]["chunks"][5]
    for path in (store.raw_block_path(chunk["output"]), store.object_path(chunk["object"])):
        path.write_bytes(b"x" * chunk["size"])
    with pytest.raises(ValueError, match="Corrupt object"):
        checkpoint.write_checkpoint(store, "s", 0, tmp_path / "checkpoint")
    assert not (tmp_path / "checkpoint").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))
    assert not list((store.root / "pins").glob("*.json"))


@pytest.mark.parametrize("compression", [False, True])
def test_parallel_seal_matches_serial_and_checks_every_encoded_chunk(source, monkeypatch, compression):
    store, path, table = source
    store.publish("s", 0, table, chunk_bytes=1024, model_config={}, compression=compression)
    data = bytearray(path.read_bytes())
    data[12] ^= 4
    path.write_bytes(data)
    manifest = store.publish("s", 1, table, chunk_bytes=1024, model_config={}, compression=compression)
    parallel = seal.seal_versions(store, "s", [0, 1], "archive")
    with monkeypatch.context() as serial:
        serial.setattr(seal, "ordered_map", lambda f, values, workers: map(f, values))
        assert seal.seal_versions(store, "s", [0, 1], "archive") == parallel
    chunk = manifest["tensors"][0]["chunks"][-1]
    store.object_path(chunk["object"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Sealed object"):
        seal.seal_versions(store, "s", [0, 1], "damaged")
    assert not (store.root / "seals/damaged.json").exists()


def test_seal_from_encoded_objects_repairs_only_required_bases(source):
    import shutil

    store, path, table = source
    first = store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    path.write_bytes(b"zz" + path.read_bytes()[2:])
    second = store.publish("s", 1, table, chunk_bytes=1024, model_config={})
    shutil.rmtree(store.canonical_directory)
    # A FULL seal needs no canonical cache: every recorded output is decoded
    # and hashed from the actual durable object, not accepted from a marker.
    full = seal.seal_versions(store, "s", [0], "full")
    assert full["closure"][0]["root"] == first["root"]
    assert not store.canonical_directory.exists()
    delta = seal.seal_versions(store, "s", [1], "delta")
    assert delta["closure"][-1]["root"] == second["root"]
    assert store.has_verified_version("s", 0, first["root"])


def test_parallel_seal_deadline_prevents_publication(source, monkeypatch):
    import time

    from relax.distributed.weight_sync.deadline import time_budget

    store, _, table = source
    store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    read_bytes = Path.read_bytes

    def slow_read(self):
        if self.parent == store.root / "objects":
            time.sleep(0.025)
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    with time_budget(0.01), pytest.raises(TimeoutError):
        seal.seal_versions(store, "s", [0], "late")
    assert not (store.root / "seals/late.json").exists()


def test_checkpoint_writer_failure_drains_prefetch_before_unpin(source, tmp_path, monkeypatch):
    import threading
    import time

    store, _, table = source
    store.publish("s", 0, table, chunk_bytes=1024, model_config={})
    original_read, original_unpin, original_open = store.read_chunk, store.unpin, Path.open
    active, peak = 0, 0
    lock = threading.Lock()

    def delayed_read(stream, version, name, chunk):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.01 if chunk["offset"] == 0 else 0.1)
            return original_read(stream, version, name, chunk)
        finally:
            with lock:
                active -= 1

    def unpin(owner):
        assert active == 0
        return original_unpin(owner)

    class FailedWriter:
        def __init__(self, path):
            self.file = original_open(path, "wb")
            self.writes = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def __getattr__(self, name):
            return getattr(self.file, name)

        def write(self, data):
            self.writes += 1
            if self.writes == 3:
                raise OSError("injected checkpoint disk failure")
            return self.file.write(data)

    def opened(self, *args, **kwargs):
        if self.name == "model.safetensors" and args == ("wb",):
            return FailedWriter(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(store, "read_chunk", delayed_read)
    monkeypatch.setattr(store, "unpin", unpin)
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(OSError, match="injected checkpoint disk failure"):
        checkpoint.write_checkpoint(store, "s", 0, tmp_path / "checkpoint")
    assert peak == 4
    assert not list((store.root / "pins").glob("*.json"))
