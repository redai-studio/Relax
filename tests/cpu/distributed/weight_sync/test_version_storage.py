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
