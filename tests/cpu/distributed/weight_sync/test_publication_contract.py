import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from relax.distributed.weight_sync.publication import VersionPublisher
from relax.distributed.weight_sync.store import VersionStore, canonical, load_json, validate_manifest
from relax.distributed.weight_sync.transport import ObjectServer, SharedFSTransport, TCPTransport, fetch


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = self.make_store("source")

    def tearDown(self):
        self.temp.cleanup()

    def make_store(self, name):
        return VersionStore(self.root / name, disk_budget=16 * 1024**2, max_weight_bytes=1024**2)

    def table(self, version):
        p = self.root / f"v{version}.bin"
        p.write_bytes(bytes([version]) + bytes(255))
        return {"层.weight": ("BF16", [128], p)}

    def publish(self, version, **kwargs):
        return self.store.publish("run", version, self.table(version), model_config={}, chunk_bytes=64, **kwargs)

    def test_strict_json_and_encoding(self):
        self.assertIn("层".encode(), canonical({"层": 1}))
        for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                load_json(data)
        m = self.publish(0)
        m["format"] = True
        with self.assertRaises(ValueError):
            validate_manifest(m, 1024)

    def test_full_repair_keeps_content_identity_and_default_package(self):
        self.publish(0)
        delta = self.publish(1)
        full = self.store.ensure_full("run", 1)
        self.assertEqual(full["root"], delta["root"])
        self.assertNotEqual(full["artifact_id"], delta["artifact_id"])
        self.assertEqual(self.store.manifest("run", 1), delta)
        self.assertEqual(self.store.ensure_full("run", 1), full)
        # A cold destination gets just the FULL target, with no predecessor.
        target = self.make_store("target")
        received = fetch(SharedFSTransport(self.store), target, "run", 1, full=True)
        self.assertEqual(received, full)
        self.assertFalse(target.manifest_path("run", 0).exists())
        self.assertEqual(target.tensor_path("run", 1, "层.weight").read_bytes(), bytes([1]) + bytes(255))
        repeated = fetch(SharedFSTransport(self.store), target, "run", 1)
        self.assertEqual(repeated["artifact_id"], full["artifact_id"])
        self.assertFalse(target.manifest_path("run", 0).exists())

    def test_same_target_full_repairs_missing_local_base(self):
        self.publish(0)
        self.publish(1)
        target = self.make_store("target")
        fetch(SharedFSTransport(self.store), target, "run", 1)
        target.manifest_path("run", 0).unlink()
        target.tensor_path("run", 0, "层.weight").unlink()
        target.tensor_path("run", 1, "层.weight").unlink()
        fetch(SharedFSTransport(self.store), target, "run", 1, full=True)
        self.assertEqual(target.materialize("run", 1)["root"], self.store.manifest("run", 1)["root"])

    def test_consumer_cursor_protects_future_versions_and_ack_is_idempotent(self):
        self.store.register("consumer", "run")
        manifests = [self.publish(v, anchor_interval=2) for v in range(12)]
        self.store.collect()
        self.assertEqual(len(list((self.store.root / "versions/run").glob("*.json"))), 12)
        for m in manifests[:8]:
            target = {k: m[k] for k in ("stream", "version", "root")}
            receipt = {"phase": "ACTIVE", "target": target, "serving": target}
            state = self.store.acknowledge("consumer", receipt)
            self.assertEqual(self.store.acknowledge("consumer", receipt), state)
        self.store.collect()
        self.assertTrue(self.store.manifest_path("run", 7).exists())
        self.assertFalse(self.store.manifest_path("run", 0).exists())
        self.assertTrue(self.store.manifest_path("run", 8).exists())
        with self.assertRaises(ValueError):
            self.store.register("late", "run", 0)

    def test_failed_target_does_not_advance_cursor(self):
        self.store.register("consumer", "run")
        m = self.publish(0)
        target = {k: m[k] for k in ("stream", "version", "root")}
        for receipt in (
            {"phase": "WAITING_ACTION", "target": target},
            {"phase": "ACTIVE", "target": target, "serving": {**target, "version": 1}},
        ):
            with self.assertRaises(ValueError):
                self.store.acknowledge("consumer", receipt)
        self.assertEqual(self.store.register("consumer", "run")["cursor"], -1)

    def test_latest_keeps_active_checkpoint_and_allows_collected_history(self):
        self.store.register("latest", "run", mode="latest")
        for version in range(12):
            self.publish(version, anchor_interval=2)
        manifest = self.store.manifest("run", 3)
        target = {key: manifest[key] for key in ("stream", "version", "root")}
        self.store.acknowledge("latest", {"phase": "ACTIVE", "target": target, "serving": target})
        self.store.collect()
        self.assertFalse(self.store.manifest_path("run", 0).exists())
        self.assertTrue(self.store.manifest_path("run", 3).exists())
        self.assertTrue(self.store.manifest_path("run", 2).exists())
        self.assertFalse(self.store.manifest_path("run", 4).exists())
        self.store.register("new-latest", "run", mode="latest")
        with self.assertRaises(ValueError):
            self.store.register("new-all", "run")
        with self.assertRaises(ValueError):
            self.store.register("latest", "run", mode="all_versions")

    def test_tcp_latest_subscription_can_skip_installations(self):
        self.publish(0)
        manifest = self.publish(1)
        target = {key: manifest[key] for key in ("stream", "version", "root")}
        with ObjectServer(("127.0.0.1", 0), self.store) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            thread.start()
            try:
                transport = TCPTransport(server.server_address)
                self.assertEqual(transport.register("latest", "run", mode="latest")["mode"], "latest")
                receipt = {"phase": "ACTIVE", "target": target, "serving": target}
                self.assertEqual(transport.acknowledge("latest", receipt)["cursor"], 1)
                self.assertEqual(transport.acknowledge("latest", receipt)["cursor"], 1)
            finally:
                server.shutdown()
                thread.join()

    def test_fetch_protects_old_chain_during_collection(self):
        for version in range(10):
            self.publish(version, anchor_interval=2)
        source = self.store

        class CollectingTransport(SharedFSTransport):
            def object(self, key):
                source.collect()
                return super().object(key)

        target = self.make_store("target")
        fetch(CollectingTransport(source), target, "run", 1)
        self.assertEqual(target.tensor_path("run", 1, "层.weight").read_bytes(), bytes([1]) + bytes(255))
        source.collect()
        self.assertFalse(source.manifest_path("run", 1).exists())

    def test_tcp_and_sharedfs_full_artifacts_are_identical(self):
        self.publish(0)
        self.publish(1)
        with ObjectServer(("127.0.0.1", 0), self.store) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                fs = fetch(SharedFSTransport(self.store), self.make_store("fs"), "run", 1, full=True)
                tcp = fetch(TCPTransport(server.server_address), self.make_store("tcp"), "run", 1, full=True)
                self.assertEqual(fs, tcp)
            finally:
                server.shutdown()
                thread.join()

    def test_publisher_freezes_inputs_and_explicitly_confirms(self):
        publisher = VersionPublisher(
            self.store, "run", self.root / "scratch", model_config={}, converter_revision="test"
        )
        try:
            table = self.table(0)
            handle = publisher.submit("optimizer-42", table)
            table["层.weight"][2].write_bytes(bytes([255]) * 256)
            with self.assertRaises(RuntimeError):
                publisher.submit("optimizer-43", self.table(1))
            published = publisher.wait_committed(handle)
            record = self.store.root / "publications/run/0.json"
            self.assertEqual(load_json(record.read_bytes())["phase"], "STORE_COMMITTED")
            self.assertEqual(self.store.tensor_path("run", 0, "层.weight").read_bytes(), bytes(256))
            self.assertEqual(publisher.confirm(handle, published), published)
            self.assertEqual(load_json(record.read_bytes())["phase"], "SOURCE_CONFIRMED")
        finally:
            publisher.close()

    def test_publication_retry_uses_same_candidate_after_disk_failure(self):
        publisher = VersionPublisher(
            self.store, "run", self.root / "scratch", model_config={}, converter_revision="test"
        )
        original = self.store.commit
        try:
            with patch.object(self.store, "commit", side_effect=OSError("injected")):
                handle = publisher.submit("u0", self.table(0))
                with self.assertRaises(OSError):
                    publisher.wait_committed(handle)
            self.assertFalse(self.store.manifest_path("run", 0).exists())
            publisher.retry(handle)
            published = publisher.finalize_pending(handle)
            self.assertEqual(published.update_id, "u0")
            self.assertEqual(published.version, 0)
            record = load_json((self.store.root / "publications/run/0.json").read_bytes())
            self.assertEqual(record["attempt"], 2)
            self.assertEqual(record["handle"]["candidate_id"], handle.candidate_id)
        finally:
            self.store.commit = original
            publisher.close()

    def test_close_finalizes_last_candidate(self):
        publisher = VersionPublisher(
            self.store, "run", self.root / "scratch", model_config={}, converter_revision="test"
        )
        publisher.submit("final-update", self.table(0))
        publisher.close()
        record = load_json((self.store.root / "publications/run/0.json").read_bytes())
        self.assertEqual(record["phase"], "SOURCE_CONFIRMED")

    def test_cleanup_preserves_committed_candidate_for_collective_confirmation(self):
        scratch = self.root / "scratch"
        publisher = VersionPublisher(self.store, "run", scratch, model_config={}, converter_revision="test")
        handle = publisher.submit("pending-update", self.table(0))
        expected = publisher.wait_committed(handle)
        publisher.close(finalize=False)
        record = self.store.root / "publications/run/0.json"
        self.assertEqual(load_json(record.read_bytes())["phase"], "STORE_COMMITTED")
        self.assertTrue((scratch / "publisher-run.json").exists())
        restored = VersionPublisher(self.store, "run", scratch, model_config={}, converter_revision="test")
        try:
            self.assertEqual(restored.pending, handle)
            self.assertEqual(restored.finalize_pending(handle), expected)
            self.assertEqual(load_json(record.read_bytes())["phase"], "SOURCE_CONFIRMED")
        finally:
            restored.close()


if __name__ == "__main__":
    unittest.main()
