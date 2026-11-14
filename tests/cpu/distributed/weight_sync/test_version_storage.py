"""Storage and codec contracts, independent of consumer and transport
adapters."""

import json
import random
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from relax.distributed.weight_sync.checkpoint import write_checkpoint
from relax.distributed.weight_sync.codec import decode, encode
from relax.distributed.weight_sync.store import VersionStore, canonical, digest


class VersionStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def store(self, name):
        return VersionStore(self.root / name, disk_budget=16 * 1024**2, max_weight_bytes=1024**2)

    def publish(self, store, version, value, **kwargs):
        path = self.root / "input.bin"
        path.write_bytes(value)
        return store.publish(
            "test", version, {"w": ("BF16", [len(value) // 2], path)}, chunk_bytes=64, model_config={}, **kwargs
        )

    def test_codec_arbitrary_bits(self):
        rng = random.Random(0)
        for width in (1, 2, 4, 8):
            for n in range(0, 400, 8):
                base = rng.randbytes(n)
                target = bytearray(base)
                for i in range(0, n, 32):
                    target[i] ^= 255
                kind, data = encode(bytes(target), base, width)
                self.assertEqual(decode(kind, data, base, width, n), target)
        # NaN payloads and +/-zero: byte identity, not floating point equality.
        base = bytes.fromhex("00000080017fc17f") * 16
        target = bytes.fromhex("00800000027fc27f") + base[8:]
        kind, payload = encode(target, base, 2)
        self.assertEqual(decode(kind, payload, base, 2, len(base)), target)

    def test_decoder_rejects_bad_indices(self):
        for indices in ([9], [2, 2], [3, 1]):
            data = b"".join(struct.pack("<I", i) + b"xx" for i in indices)
            with self.assertRaises(ValueError):
                decode("sparse", data, b"0" * 16, 2, 16)

    def test_compressed_bytes_and_bounded_decode(self):
        base = random.Random(0).randbytes(8192)
        target = bytes(value ^ (index % 2) for index, value in enumerate(base))
        kind, data = encode(target, base, 2, compression=True)
        self.assertEqual(kind, "xor_zlib")
        self.assertEqual(decode(kind, data, base, 2, len(base)), target)
        kind, data = encode(b"\0" * 8192, None, 2, compression=True)
        self.assertEqual(kind, "raw_zlib")
        self.assertEqual(decode(kind, data, None, 2, 8192), b"\0" * 8192)
        for bad in (data + b"trailing", data[:-1], zlib.compress(b"\0" * 65536)):
            with self.assertRaises(ValueError):
                decode("raw_zlib", bad, None, 2, 8192)

    def test_duplicate_conflict_and_schema(self):
        store = self.store("source")
        first = self.publish(store, 0, b"0" * 128)
        self.assertEqual(first, self.publish(store, 0, b"0" * 128))
        with self.assertRaises(ValueError):
            self.publish(store, 0, b"1" * 128)
        self.assertEqual(store.manifest("test", 0), first)
        changed = self.publish(store, 1, b"1" * 64)
        self.assertEqual(changed["kind"], "full")
        self.assertIsNone(changed["base_version"])

    def test_collection_keeps_pinned_chain_and_two_anchors(self):
        source = self.store("source")
        for version in range(7):
            self.publish(source, version, bytes([version]) * 128, anchor_interval=2)
        source.pin("test", 1, "offline-reader")
        result = source.collect()
        self.assertEqual(result["removed_versions"], 2)
        self.assertEqual(source.materialize("test", 1)["version"], 1)
        self.assertEqual(source.materialize("test", 6)["version"], 6)
        with self.assertRaises(FileNotFoundError):
            source.manifest("test", 3)
        source.unpin("offline-reader")
        source.collect()
        with self.assertRaises(FileNotFoundError):
            source.manifest("test", 1)

    def test_manifest_rejects_oversize_and_path(self):
        source = self.store("source")
        m = self.publish(source, 0, b"0" * 128)
        m["tensors"][0]["shape"] = [10**20]
        m["root"] = digest(canonical({k: v for k, v in m.items() if k != "root"}))
        with self.assertRaises(ValueError):
            source.commit(m)
        with self.assertRaises(ValueError):
            source.manifest_path("../escape", 0)

    def test_disk_budget_and_checkpoint_format(self):
        store = VersionStore(self.root / "small", disk_budget=40, max_weight_bytes=1024)
        with self.assertRaises(OSError):
            self.publish(store, 0, bytes(range(128)))
        self.assertFalse(store.manifest_path("test", 0).exists())
        store = self.store("source")
        self.publish(store, 0, bytes(range(128)))
        config = self.root / "config.json"
        config.write_text("{}")
        path = write_checkpoint(store, "test", 0, self.root / "checkpoint", config_source=config)
        with (path / "model.safetensors").open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
            self.assertEqual(header["w"]["data_offsets"], [0, 128])
            self.assertEqual(f.read(), bytes(range(128)))


def test_manifest_failure_can_retry_with_the_original_budget(tmp_path, monkeypatch):
    import pytest

    from relax.distributed.weight_sync import store as module

    source = tmp_path / "input"
    source.write_bytes(bytes(range(64)))
    reference = VersionStore(tmp_path / "reference", disk_budget=10_000, max_weight_bytes=64)
    reference.publish("s", 0, {"w": ("U8", [64], source)}, full_only=True)
    required = sum(path.stat().st_size for path in reference.root.rglob("*") if path.is_file())
    store = VersionStore(tmp_path / "store", disk_budget=required, max_weight_bytes=64)
    atomic_write = module.atomic_write

    def fail_manifest(path, data):
        if path == store.manifest_path("s", 0):
            raise OSError("injected manifest failure")
        atomic_write(path, data)

    monkeypatch.setattr(module, "atomic_write", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        store.publish("s", 0, {"w": ("U8", [64], source)}, full_only=True)
    monkeypatch.setattr(module, "atomic_write", atomic_write)
    result = store.publish("s", 0, {"w": ("U8", [64], source)}, full_only=True)
    assert store.materialize("s", 0)["root"] == result["root"]
    assert b"".join(store.iter_tensor(result, result["tensors"][0])) == source.read_bytes()


def test_failed_publication_preserves_committed_versions_and_full_repairs(tmp_path, monkeypatch):
    import pytest

    from relax.distributed.weight_sync import store as module

    for blocks in (False, True):
        store = VersionStore(
            tmp_path / str(blocks), disk_budget=100_000, max_weight_bytes=128, canonical_blocks=blocks
        )
        source = tmp_path / "input"
        source.write_bytes(bytes(range(128)))
        tensors = {"w": ("U8", [128], source)}
        first = store.publish("s", 0, tensors)
        source.write_bytes(b"x" + bytes(range(1, 128)))
        second = store.publish("s", 1, tensors)
        repaired = store.ensure_full("s", 1)
        store.pin("s", 0, "reader")
        atomic_write = module.atomic_write

        def fail_manifest(path, data):
            if path == store.manifest_path("s", 2):
                raise OSError("injected manifest failure")
            atomic_write(path, data)

        monkeypatch.setattr(module, "atomic_write", fail_manifest)
        source.write_bytes(b"z" * 128)
        with pytest.raises(OSError, match="injected manifest failure"):
            store.publish("s", 2, tensors)
        monkeypatch.setattr(module, "atomic_write", atomic_write)
        assert store.materialize("s", 0) == first
        assert store.manifest("s", 1) == second
        assert store.manifest("s", 1, prefer_full=True) == repaired
        assert b"".join(store.iter_tensor(first, first["tensors"][0])) == bytes(range(128))
        assert b"".join(store.iter_tensor(second, second["tensors"][0])) == b"x" + bytes(range(1, 128))
        result = store.publish("s", 2, tensors)
        assert b"".join(store.iter_tensor(result, result["tensors"][0])) == b"z" * 128


def test_post_rename_error_preserves_visible_publication(tmp_path, monkeypatch):
    import pytest

    from relax.distributed.weight_sync import store as module

    store = VersionStore(tmp_path / "store", disk_budget=10_000, max_weight_bytes=64)
    source = tmp_path / "input"
    source.write_bytes(bytes(range(64)))
    atomic_write = module.atomic_write

    def fail_after_manifest(path, data):
        atomic_write(path, data)
        if path == store.manifest_path("s", 0):
            raise OSError("injected directory fsync failure")

    monkeypatch.setattr(module, "atomic_write", fail_after_manifest)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        store.publish("s", 0, {"w": ("U8", [64], source)})
    monkeypatch.setattr(module, "atomic_write", atomic_write)
    assert store.materialize("s", 0)["version"] == 0
    assert store.publish("s", 0, {"w": ("U8", [64], source)}) == store.manifest("s", 0)
