import tempfile
import unittest
from pathlib import Path

from relax.distributed.weight_sync.store import VersionStore, canonical
from relax.distributed.weight_sync.transport import SharedFSTransport, fetch


class CanonicalPlacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.options = dict(disk_budget=1024**2, max_weight_bytes=4096)
        self.source = VersionStore(
            self.root / "packages", canonical_root=self.root / "canonical", canonical_budget=32768, **self.options
        )

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self, version, **kwargs):
        path = self.root / "input.bin"
        value = bytes([version]) + bytes(1023)
        path.write_bytes(value)
        return self.source.publish("s", version, {"w": ("BF16", [512], path)}, model_config={}, **kwargs)

    def test_restart_full_repair_without_anchor_and_gc(self):
        first = self.publish(0)
        delta = self.publish(1)
        self.assertEqual(delta["kind"], "delta")
        self.assertFalse((self.source.root / "materialized").exists())
        self.assertNotIn(str(self.root).encode(), canonical(delta))
        # The owner restarted. A FULL repair depends on the target's retained
        # canonical data, even if an old delivery object is damaged or lost.
        source = VersionStore(self.source.root, **self.options)
        for tensor in first["tensors"]:
            for chunk in tensor["chunks"]:
                source.object_path(chunk["object"]).unlink(missing_ok=True)
        full = source.ensure_full("s", 1)
        self.assertEqual(full["root"], delta["root"])
        destination = VersionStore(self.root / "destination", **self.options)
        fetch(SharedFSTransport(source), destination, "s", 1, full=True)
        self.assertEqual(destination.tensor_path("s", 1, "w").read_bytes(), bytes([1]) + bytes(1023))
        for version in range(2, 7):
            self.publish(version, anchor_interval=2)
        source.collect()
        self.assertFalse(source.tensor_path("s", 1, "w").exists())
        self.assertTrue(source.tensor_path("s", 6, "w").exists())

    def test_transport_opened_before_publisher_can_repair_external_canonical(self):
        early = VersionStore(self.root / "early-packages", **self.options)
        self.source = VersionStore(
            early.root,
            canonical_root=self.root / "late-canonical",
            canonical_budget=32768,
            canonical_blocks=True,
            **self.options,
        )
        first = self.publish(0)
        delta = self.publish(1)
        self.assertEqual(delta["kind"], "delta")
        for tensor in first["tensors"]:
            for chunk in tensor["chunks"]:
                early.object_path(chunk["object"]).unlink(missing_ok=True)
        destination = VersionStore(self.root / "early-destination", **self.options)
        fetch(SharedFSTransport(early), destination, "s", 1, full=True)
        self.assertEqual(destination.manifest("s", 1)["root"], delta["root"])
        self.assertEqual(destination.tensor_path("s", 1, "w").read_bytes(), bytes([1]) + bytes(1023))
        self.assertEqual(early.canonical_directory, self.source.canonical_directory)
        self.assertEqual(early.canonical_budget, 32768)

    def test_capacity_failure_retries_same_version_after_budget_increase(self):
        self.source.canonical_budget = 2048
        self.publish(0)
        with self.assertRaisesRegex(OSError, "Canonical disk budget"):
            self.publish(1)
        self.assertFalse(self.source.manifest_path("s", 1).exists())
        self.source = VersionStore(self.source.root, canonical_budget=4096, **self.options)
        self.assertEqual(self.publish(1)["kind"], "delta")
        self.assertEqual(self.source.tensor_path("s", 1, "w").stat().st_size, 1024)

    def test_unmounted_volume_preserves_publication_and_gc_state(self):
        for version in range(5):
            self.publish(version, anchor_interval=2)
        marker = self.source.canonical_directory / "placement.json"
        saved = marker.read_bytes()
        marker.unlink()
        with self.assertRaisesRegex(OSError, "Canonical volume is unavailable"):
            self.source.collect()
        self.assertEqual(len(list((self.source.root / "versions/s").glob("*.json"))), 5)
        with self.assertRaisesRegex(OSError, "Canonical volume is unavailable"):
            self.publish(5)
        marker.write_bytes(saved)
        self.assertEqual(self.publish(5)["version"], 5)

    def test_distinct_stores_share_volume_without_sharing_weights(self):
        other = VersionStore(
            self.root / "other", canonical_root=self.root / "canonical", canonical_budget=32768, **self.options
        )
        self.publish(0)
        self.assertNotEqual(other.canonical_directory, self.source.canonical_directory)
        self.assertFalse(other.tensor_path("s", 0, "w").exists())
        with self.assertRaisesRegex(ValueError, "differs from the retained store"):
            VersionStore(self.source.root, canonical_root=self.root / "elsewhere", **self.options)


if __name__ == "__main__":
    unittest.main()
