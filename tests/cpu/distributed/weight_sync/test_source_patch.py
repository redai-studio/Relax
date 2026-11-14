import tempfile
import unittest
from pathlib import Path

from relax.distributed.weight_sync.codec import MAX_CHUNK_BYTES, decode
from relax.distributed.weight_sync.source_patch import file_digest, plan_replacements
from relax.distributed.weight_sync.store import digest


class SourcePatchTests(unittest.TestCase):
    def test_frames_cross_chunk_boundary_and_preserve_nan_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = bytes(MAX_CHUNK_BYTES + 1024)
            current = bytearray(base)
            current[20:22] = b"\xc1\x7f"
            current[MAX_CHUNK_BYTES + 10 : MAX_CHUNK_BYTES + 12] = b"\0\x80"
            (root / "base").write_bytes(base)
            (root / "current").write_bytes(current)
            plan = plan_replacements(root / "current", root / "base", 2, root / "frames")
            self.assertEqual(plan["route"], "PATCH")
            self.assertEqual(plan["payload_bytes"], 12)
            restored = bytearray(base)
            for frame in plan["frames"]:
                payload = (root / "frames" / f"{frame['sequence']}.bin").read_bytes()
                self.assertEqual(digest(payload), frame["sha256"])
                offset = frame["offset"] * 2
                block = base[offset : offset + MAX_CHUNK_BYTES]
                restored[offset : offset + len(block)] = decode("sparse", payload, block, 2, len(block))
            self.assertEqual(restored, current)
            self.assertEqual(file_digest(root / "current"), digest(bytes(current)))

    def test_unchanged_dense_and_missing_base_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base").write_bytes(bytes(4096))
            (root / "current").write_bytes(bytes(4096))
            self.assertEqual(plan_replacements(root / "current", root / "base", 2, root / "copy")["route"], "REUSE")
            (root / "current").write_bytes(b"\xff" * 4096)
            self.assertEqual(plan_replacements(root / "current", root / "base", 2, root / "dense")["route"], "FULL")
            self.assertEqual(
                plan_replacements(root / "current", root / "missing", 2, root / "missing-frames")["route"], "FULL"
            )

    def test_partial_frames_before_dense_fallback_fit_raw_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = bytes(MAX_CHUNK_BYTES + 1024)
            current = bytearray(base)
            current[:2] = b"\x01\x02"
            current[MAX_CHUNK_BYTES:] = b"\xff" * 1024
            (root / "base").write_bytes(base)
            (root / "current").write_bytes(current)
            plan = plan_replacements(root / "current", root / "base", 2, root / "frames")
            self.assertEqual(plan["route"], "FULL")
            retained = sum(path.stat().st_size for path in (root / "frames").iterdir())
            self.assertGreater(retained, 0)
            self.assertLessEqual(retained, len(current))

    def test_replacement_plan_checks_bytes_against_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = bytes(MAX_CHUNK_BYTES + 1024)
            current = bytearray(base)
            current[-2:] = b"\x80\x7f"
            (root / "base").write_bytes(base)
            path = root / "current"
            path.write_bytes(current)
            expected = digest(current)
            plan = plan_replacements(path, root / "base", 2, root / "frames", expected_current_hash=expected)
            self.assertEqual(plan["route"], "PATCH")
            # A candidate modified after its snapshot hash must not become a
            # valid replacement plan, even if the change remains sparse.
            current[8] ^= 1
            path.write_bytes(current)
            with self.assertRaisesRegex(ValueError, "frozen snapshot"):
                plan_replacements(path, root / "base", 2, root / "corrupt", expected_current_hash=expected)
            path.write_bytes(base)
            with self.assertRaisesRegex(ValueError, "frozen snapshot"):
                plan_replacements(path, root / "base", 2, root / "false-reuse", expected_current_hash=expected)


if __name__ == "__main__":
    unittest.main()
