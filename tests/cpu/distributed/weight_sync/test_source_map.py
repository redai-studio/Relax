import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from relax.distributed.weight_sync.source_map import TensorMapping, apply_replacements, sync_replacements
from relax.distributed.weight_sync.store import digest


def reconstruct(mapping, shards):
    outputs = {name: np.full(shape, -1, dtype=np.int64) for name, shape in mapping.outputs}
    for shard, values in enumerate(shards):
        for name, (positions, targets) in mapping.destinations(shard, np.arange(values.size)).items():
            destination = outputs[name].reshape(-1)
            if np.any(destination[targets] != -1):
                raise AssertionError("Overlapping owner writes")
            destination[targets] = values.reshape(-1)[positions]
    for values in outputs.values():
        if np.any(values == -1):
            raise AssertionError("Missing logical elements")
    return outputs


class SourceMapTests(unittest.TestCase):
    def test_qkv_boundaries(self):
        shards = [np.arange(rank * 64, (rank + 1) * 64).reshape(8, 8) for rank in range(2)]
        mapping = TensorMapping((8, 8), 2, 0, (("q", (8, 8)), ("k", (4, 8)), ("v", (4, 8))), "qkv", 2, 2, 2)
        actual = reconstruct(mapping, shards)
        grouped = np.concatenate(shards).reshape(2, 4, 2, 8)
        np.testing.assert_array_equal(actual["q"], grouped[:, :2].reshape(8, 8))
        np.testing.assert_array_equal(actual["k"], grouped[:, 2].reshape(4, 8))
        np.testing.assert_array_equal(actual["v"], grouped[:, 3].reshape(4, 8))

    def test_gate_up_rechunk_and_column_partition(self):
        shards = [np.arange(rank * 24, (rank + 1) * 24).reshape(6, 4) for rank in range(2)]
        mapping = TensorMapping((6, 4), 2, 0, (("gate", (6, 4)), ("up", (6, 4))), "gate_up")
        actual = reconstruct(mapping, shards)
        np.testing.assert_array_equal(actual["gate"], np.concatenate([part[:3] for part in shards]))
        np.testing.assert_array_equal(actual["up"], np.concatenate([part[3:] for part in shards]))
        columns = TensorMapping((6, 4), 2, 1, (("down", (6, 8)),))
        np.testing.assert_array_equal(reconstruct(columns, shards)["down"], np.concatenate(shards, axis=1))

    def test_vocabulary_padding_is_cropped_explicitly(self):
        shards = [np.arange(rank * 24, (rank + 1) * 24).reshape(6, 4) for rank in range(2)]
        with self.assertRaisesRegex(ValueError, "cover"):
            TensorMapping((6, 4), 2, 0, (("embedding", (10, 4)),))
        mapping = TensorMapping((6, 4), 2, 0, (("embedding", (10, 4)),), crop_vocabulary=True)
        np.testing.assert_array_equal(reconstruct(mapping, shards)["embedding"], np.concatenate(shards)[:10])
        with self.assertRaisesRegex(ValueError, "outside"):
            mapping.destinations(0, [24])

    def test_patches_detach_base_and_preserve_raw_bf16_payloads(self):
        mapping = TensorMapping((4, 2), 2, 0, (("gate", (4, 2)), ("up", (4, 2))), "gate_up")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            for name, _ in mapping.outputs:
                base = root / name
                base.write_bytes(bytes(16))
                os.link(base, staging / (digest(name.encode()) + ".bin"))
            # Sign bit of zero and a NaN payload are copied as opaque bytes.
            frame = struct.pack("<I", 3) + b"\0\x80" + struct.pack("<I", 4) + b"\xc1\x7f"
            result = apply_replacements(mapping, 1, 0, frame, 2, staging)
            self.assertEqual(result, {"elements_applied": 2, "elements_cropped": 0})
            self.assertEqual((root / "gate").read_bytes(), bytes(16))
            self.assertEqual((root / "up").read_bytes(), bytes(16))
            self.assertEqual((staging / (digest(b"gate") + ".bin")).read_bytes()[14:16], b"\0\x80")
            self.assertEqual((staging / (digest(b"up") + ".bin")).read_bytes()[8:10], b"\xc1\x7f")

    def test_group_sync_preserves_all_frames_and_original_base(self):
        mapping = TensorMapping((4, 2), 2, 0, (("gate", (4, 2)), ("up", (4, 2))), "gate_up")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for mode in ("per-frame", "group"):
                staging = root / mode
                staging.mkdir()
                for name, _ in mapping.outputs:
                    base = root / name
                    base.write_bytes(bytes(16))
                    os.link(base, staging / (digest(name.encode()) + ".bin"))
                for shard in range(2):
                    for index in range(8):
                        payload = struct.pack("<I", index) + struct.pack("<H", 0x7F80 + 8 * shard + index)
                        apply_replacements(mapping, shard, 0, payload, 2, staging, sync=mode == "per-frame")
                if mode == "group":
                    with patch("os.fsync", wraps=os.fsync) as drain:
                        sync_replacements(mapping, staging)
                    self.assertEqual(drain.call_count, 2)
                    with patch("os.fsync", side_effect=OSError("injected group sync error")):
                        with self.assertRaisesRegex(OSError, "group sync"):
                            sync_replacements(mapping, staging)
                outputs.append(
                    [(staging / (digest(name.encode()) + ".bin")).read_bytes() for name, _ in mapping.outputs]
                )
                for name, _ in mapping.outputs:
                    self.assertEqual((root / name).read_bytes(), bytes(16))
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
