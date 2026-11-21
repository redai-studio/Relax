import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from relax.distributed.weight_sync.consumer import VersionConsumer
from relax.distributed.weight_sync.publication import VersionPublisher
from relax.distributed.weight_sync.session import InstallSession
from relax.distributed.weight_sync.store import VersionStore, load_json
from relax.distributed.weight_sync.transport import SharedFSTransport


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.store("source")

    def tearDown(self):
        self.temp.cleanup()

    def store(self, name):
        return VersionStore(self.root / name, disk_budget=32 * 1024**2, max_weight_bytes=1024**2)

    def publish(self, version, **kwargs):
        path = self.root / f"input-{version}"
        path.write_bytes(bytes([version]) + bytes(255))
        return self.source.publish(
            "s", version, {"w": ("BF16", [128], path)}, model_config=kwargs.pop("model_config", {}), **kwargs
        )

    def consumer(self):
        return VersionConsumer(
            SharedFSTransport(self.source),
            self.store("cache"),
            self.root / "checkpoints",
            checkpoint_budget=32 * 1024**2,
        )

    def test_wire_normalization_keeps_identical_hf_config_as_delta(self):
        config = {"id2label": {0: "zero", 1: "one"}, "tuple": (1, 2)}
        first = self.publish(0, model_config=config)
        second = self.publish(1, model_config=config)
        self.assertEqual(second["kind"], "delta")
        self.assertEqual(first["schema_context"], second["schema_context"])

    def test_configuration_change_forces_full(self):
        self.publish(0)
        changed = self.publish(1, model_config={"rope_theta": 10000})
        self.assertEqual(changed["kind"], "full")
        self.assertIsNone(changed["base_root"])

    def test_live_lease_survives_gc_and_another_consumers_release(self):
        for version in range(10):
            self.publish(version, anchor_interval=2)
        consumer = self.consumer()
        first = consumer.prepare("s", 0, "one")
        second = consumer.prepare("s", 0, "two")
        for version in (2, 4, 6, 8):
            temporary = consumer.prepare("s", version, f"temporary-{version}")
            consumer.release(temporary, remove_checkpoint=True)
        consumer.cache.collect()
        self.assertTrue(consumer.cache.manifest_path("s", 0).exists())
        consumer.release(first, remove_checkpoint=True)
        self.assertTrue(second.checkpoint.is_dir())
        consumer.release(second, remove_checkpoint=True)
        self.assertFalse(second.checkpoint.exists())
        consumer.cache.collect()
        self.assertFalse(consumer.cache.manifest_path("s", 0).exists())

    def test_full_and_delta_share_valid_checkpoint_with_separate_leases(self):
        self.publish(0)
        self.publish(1)
        consumer = self.consumer()
        delta = consumer.prepare("s", 1, "delta")
        full = consumer.prepare("s", 1, "full", full=True)
        self.assertEqual(delta.checkpoint, full.checkpoint)
        self.assertNotEqual(delta.manifest["artifact_id"], full.manifest["artifact_id"])
        consumer.release(delta, remove_checkpoint=True)
        self.assertTrue(full.checkpoint.is_dir())
        consumer.release(full, remove_checkpoint=True)
        self.assertFalse(full.checkpoint.exists())

    def test_full_repair_does_not_reuse_corrupt_delta_checkpoint(self):
        from relax.distributed.weight_sync.checkpoint import validate_checkpoint

        self.publish(0)
        self.publish(1)
        consumer = self.consumer()
        delta = consumer.prepare("s", 1, "delta")
        path = delta.checkpoint / "model.safetensors"
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"x")
        full = consumer.prepare("s", 1, "full", full=True)
        self.assertNotEqual(delta.checkpoint, full.checkpoint)
        validate_checkpoint(full.checkpoint, full.manifest)
        self.assertTrue(delta.checkpoint.exists())

    def test_retention_owner_cannot_silently_move(self):
        self.publish(0)
        self.publish(1)
        self.source.pin("s", 0, "owner")
        with self.assertRaisesRegex(ValueError, "already protects"):
            self.source.pin("s", 1, "owner")
        self.assertEqual(load_json((self.source.root / "pins/owner.json").read_bytes())["version"], 0)

    def test_full_repair_skips_checkpoint_with_invalid_header_type(self):
        import struct

        self.publish(0)
        self.publish(1)
        consumer = self.consumer()
        delta = consumer.prepare("s", 1, "delta")
        with (delta.checkpoint / "model.safetensors").open("r+b") as handle:
            handle.write(struct.pack("<Q", 4) + b"null")
        full = consumer.prepare("s", 1, "full", full=True)
        self.assertNotEqual(delta.checkpoint, full.checkpoint)
        self.assertEqual(full.manifest["root"], delta.manifest["root"])

    def test_source_release_failure_keeps_prepared_result_and_reports_action(self):
        self.publish(0)

        class BrokenRelease(SharedFSTransport):
            def unpin(self, owner):
                raise ConnectionError("disconnected after complete download")

        consumer = self.consumer()
        consumer.transport = BrokenRelease(self.source)
        with self.assertLogs("relax.distributed.weight_sync.transport", level="WARNING") as messages:
            prepared = consumer.prepare("s", 0, "one")
        self.assertTrue(prepared.checkpoint.is_dir())
        self.assertIn("Retry unpin", messages.output[0])
        consumer.release(prepared, remove_checkpoint=True)

    def test_second_live_publisher_is_rejected(self):
        publisher = VersionPublisher(self.source, "s", self.root / "scratch", model_config={}, converter_revision="v1")
        try:
            with self.assertRaises(OSError):
                VersionPublisher(self.source, "s", self.root / "other", model_config={}, converter_revision="v1")
        finally:
            publisher.close()

    def test_process_exit_after_store_commit_resumes_same_version(self):
        script = """
import os, sys
from pathlib import Path
from relax.distributed.weight_sync.store import VersionStore
from relax.distributed.weight_sync.publication import VersionPublisher
r=Path(sys.argv[1]); p=r/'input'; p.write_bytes(bytes(256))
s=VersionStore(r/'source',disk_budget=32*1024**2,max_weight_bytes=1024**2,canonical_blocks=True)
v=VersionPublisher(s,'s',r/'scratch',model_config={},converter_revision='v1')
h=v.submit('step-42',{'w':('BF16',[128],p)})
v.wait_committed(h)
os._exit(0)
"""
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path.cwd()), os.environ.get("PYTHONPATH", "")]),
        }
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root)], env=environment, capture_output=True, timeout=20
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        publisher = VersionPublisher(self.source, "s", self.root / "scratch", model_config={}, converter_revision="v1")
        try:
            self.assertEqual(publisher.pending.version, 0)
            restored = publisher.finalize_pending(publisher.pending)
            self.assertEqual(restored.update_id, "step-42")
            self.assertEqual(publisher.next_version(), 1)
        finally:
            publisher.close()
        state = load_json((self.source.root / "publications/s/0.json").read_bytes())
        self.assertEqual(state["phase"], "SOURCE_CONFIRMED")
        self.assertFalse((self.root / "scratch/publisher-s.json").exists())

    def test_recovery_serves_old_version_without_acknowledging_failed_target(self):
        self.publish(0)
        self.publish(1)
        consumer = self.consumer()
        old, new = consumer.prepare("s", 0, "old"), consumer.prepare("s", 1, "new")

        class Engines:
            failing = False
            serving = False

            def identity(self):
                return "fixed-members"

            def close(self):
                self.serving = False

            def pause(self):
                pass

            def load(self, *args):
                if self.failing:
                    raise TimeoutError("worker install incomplete")

            def verify(self, *args):
                pass

            def flush(self):
                pass

            def resume(self):
                pass

            def open(self):
                self.serving = True

        engines = Engines()
        session = InstallSession(self.root / "session", engines)
        self.source.register("reader", "s")
        first = session.install(old.checkpoint, old.manifest, "initial")
        self.source.acknowledge("reader", first)
        engines.failing = True
        with self.assertRaises(TimeoutError):
            session.install(new.checkpoint, new.manifest, "update")
        self.assertFalse(engines.serving)
        engines.failing = False
        recovered = session.recover(old.checkpoint, old.manifest, "recover")
        self.assertTrue(engines.serving)
        self.assertEqual(recovered["serving"]["version"], 0)
        self.assertEqual(recovered["target"]["version"], 1)
        with self.assertRaises(ValueError):
            self.source.acknowledge("reader", recovered)
        final = session.install(new.checkpoint, new.manifest, "update-retry")
        self.assertEqual(self.source.acknowledge("reader", final)["cursor"], 1)


if __name__ == "__main__":
    unittest.main()
