"""Crash and concurrency regressions for publication and installation."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from relax.distributed.weight_sync.admission import InstallationFence
from relax.distributed.weight_sync.consumer import VersionConsumer
from relax.distributed.weight_sync.fence import WorkerFence
from relax.distributed.weight_sync.runtime import ConsumptionError, ConsumptionRunner
from relax.distributed.weight_sync.session import InstallSession
from relax.distributed.weight_sync.store import VersionStore, load_json, stored_bytes
from relax.distributed.weight_sync.transport import ObjectServer, SharedFSTransport, TCPTransport, fetch


class Engines:
    def __init__(self):
        self.serving, self.loads, self.member = False, 0, "worker-1"

    def identity(self):
        return self.member

    def preflight(self, checkpoint):
        if load_json((checkpoint / "config.json").read_bytes()).get("incompatible"):
            raise ValueError("configuration incompatible")

    def close(self):
        self.serving = False

    def pause(self):
        pass

    def load(self, checkpoint, version):
        self.loads += 1
        self.content = (checkpoint / "model.safetensors").read_bytes()

    def verify(self, checkpoint):
        assert self.content == (checkpoint / "model.safetensors").read_bytes()

    def flush(self):
        pass

    def resume(self):
        pass

    def open(self):
        self.serving = True


class ReviewRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = self.make_store("source")

    def tearDown(self):
        self.temp.cleanup()

    def make_store(self, name):
        return VersionStore(self.root / name, disk_budget=64 * 1024**2, max_weight_bytes=1024**2)

    def publish(self, version, **options):
        path = self.root / "input"
        path.write_bytes(bytes([version]) + bytes(1023))
        return self.store.publish("s", version, {"w": ("BF16", [512], path)}, **options)

    def runner(self):
        engines = Engines()
        consumer = VersionConsumer(
            SharedFSTransport(self.store),
            self.make_store("cache"),
            self.root / "checkpoints",
            checkpoint_budget=64 * 1024**2,
        )
        return ConsumptionRunner(consumer, InstallSession(self.root / "session", engines), "reader", "s"), engines

    def test_commit_decision_survives_stale_active_index_and_restart(self):
        for version in range(2):
            self.publish(version, model_config={})
        runner, engines = self.runner()
        runner.apply(0)
        old_index = (runner.directory / "active-checkpoint.json").read_bytes()
        with patch.object(engines, "open", side_effect=OSError("exit after commit")):
            with self.assertRaises(ConsumptionError):
                runner.apply(1)
        self.assertEqual((runner.directory / "active-checkpoint.json").read_bytes(), old_index)
        self.assertTrue(runner.session.status()["committed"])
        previous = load_json(old_index)
        with self.assertRaisesRegex(ValueError, "committed checkpoint"):
            runner.session.recover(runner.consumer.root / previous["checkpoint"], previous["manifest"], "old")
        restarted, replacement = self.runner()
        recovered = restarted.recover_active()
        self.assertEqual(recovered["cursor"], 1)
        self.assertEqual(recovered["active_version"]["version"], 1)
        self.assertTrue(replacement.serving)
        self.assertEqual(len(list((runner.consumer.root / ".leases").glob("*.json"))), 1)

    def test_active_index_failure_does_not_require_reloading_live_target(self):
        from relax.distributed.weight_sync import runtime

        self.publish(0, model_config={})
        runner, engines = self.runner()
        write = runtime.atomic_write

        def fail_index(path, data):
            if path.name == "active-checkpoint.json":
                raise OSError("index write interrupted")
            return write(path, data)

        with patch.object(runtime, "atomic_write", side_effect=fail_index):
            with self.assertRaises(ConsumptionError):
                runner.apply(0)
        self.assertTrue(engines.serving)
        self.assertEqual(runner.apply(0)["cursor"], 0)
        self.assertEqual(engines.loads, 1)

    def test_incompatible_preparation_preserves_previous_service(self):
        self.publish(0, model_config={})
        self.publish(1, model_config={"incompatible": True})
        runner, engines = self.runner()
        runner.apply(0)
        old = runner.session.status()
        with self.assertRaises(ConsumptionError):
            runner.apply(1)
        self.assertTrue(engines.serving)
        self.assertEqual(engines.loads, 1)
        self.assertEqual(runner.session.status(), old)
        self.assertEqual(self.store.consumer_status("reader")["cursor"], 0)

    def test_restart_cleans_checkpoint_created_before_prepare_returned(self):
        from relax.distributed.weight_sync import consumer as module

        self.publish(0, model_config={})
        runner, _ = self.runner()
        write = module.write_checkpoint

        def interrupted(*args, **kwargs):
            write(*args, **kwargs)
            raise OSError("exit after checkpoint rename")

        with patch.object(module, "write_checkpoint", side_effect=interrupted):
            with self.assertRaises(ConsumptionError):
                runner.apply(0)
        self.assertEqual(len(list(runner.consumer.root.glob("s-0-*"))), 1)
        staging = runner.consumer.root / ".checkpoint-interrupted"
        staging.mkdir()
        (staging / "partial").write_bytes(b"partial")
        restarted, _ = self.runner()
        self.assertFalse(list(restarted.consumer.root.glob("s-0-*")))
        self.assertFalse(staging.exists())
        self.assertEqual(restarted.apply(0)["cursor"], 0)

    def test_node_shared_checkpoint_survives_until_last_owner_releases(self):
        self.publish(0, model_config={})
        consumer = self.runner()[0].consumer
        first = consumer.prepare("s", 0, "engine-one")
        second = consumer.prepare("s", 0, "engine-two")
        self.assertEqual(first.checkpoint, second.checkpoint)
        consumer.release(first, remove_checkpoint=True)
        self.assertTrue(second.checkpoint.is_dir())
        self.assertEqual(self.store.manifest("s", 0)["root"], second.manifest["root"])
        consumer.release(second, remove_checkpoint=True)
        self.assertFalse(second.checkpoint.exists())

    def test_corrupt_delivery_automatically_requests_same_target_full_once(self):
        self.publish(0, model_config={})
        target = self.publish(1, model_config={})
        key = target["tensors"][0]["chunks"][0]["object"]
        self.store.object_path(key).write_bytes(b"corrupt")
        consumer = self.runner()[0].consumer
        with patch.object(self.store, "ensure_full", wraps=self.store.ensure_full) as repair:
            prepared = consumer.prepare("s", 1, "engine")
        self.assertEqual(repair.call_count, 1)
        self.assertEqual(prepared.manifest["root"], target["root"])
        self.assertEqual(prepared.manifest["kind"], "full")

    def test_missing_tcp_object_and_sharedfs_repair_use_publisher_owner(self):
        self.publish(0, model_config={})
        target = self.publish(1, model_config={})
        key = target["tensors"][0]["chunks"][0]["object"]
        with ObjectServer(("127.0.0.1", 0), self.store) as server:
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                for mode in ("tcp", "sharedfs"):
                    self.store.object_path(key).unlink(missing_ok=True)
                    transport = TCPTransport(server.server_address)
                    if mode == "sharedfs":
                        reader = self.make_store("source")
                        transport = SharedFSTransport(reader, repair=transport)
                        reader.ensure_full = lambda *args: self.fail("repair ran on consumer's store view")
                    consumer = VersionConsumer(
                        transport,
                        self.make_store("cache-" + mode),
                        self.root / ("prepared-" + mode),
                        checkpoint_budget=64 * 1024**2,
                    )
                    prepared = consumer.prepare("s", 1, "engine-" + mode)
                    self.assertEqual(prepared.manifest["root"], target["root"])
                    self.assertEqual(prepared.manifest["kind"], "full")
            finally:
                server.shutdown()
                worker.join()

    def test_pin_reply_loss_is_reconciled_with_persisted_identity(self):
        self.publish(0)
        target = self.make_store("cache")

        class Broken(SharedFSTransport):
            def pin(self, *args):
                super().pin(*args)
                raise ConnectionError("pin reply lost")

            def unpin(self, owner):
                raise ConnectionError("release disconnected")

        with self.assertLogs("relax.distributed.weight_sync.transport", level="WARNING"):
            with self.assertRaises(ConnectionError):
                fetch(Broken(self.store), target, "s", 0)
        pins = list((self.store.root / "pins").glob("*.json"))
        self.assertEqual(len(pins), 1)
        owner = pins[0].stem
        self.assertTrue((target.root / "fetch-operations" / f"{owner}.json").exists())
        fetch(SharedFSTransport(self.store), self.make_store("cache"), "s", 0)
        self.assertEqual(list((self.store.root / "pins").glob("*.json")), [])
        self.assertEqual(list((target.root / "fetch-operations").glob("*.json")), [])
        with self.assertRaisesRegex(ValueError, "already released"):
            self.store.pin("s", 0, owner)

    def test_full_repairs_original_anchor_and_generated_variant(self):
        for version in range(3):
            self.publish(version, anchor_interval=2, compression=True)
        for version in range(3):
            full = self.store.ensure_full("s", version)
            chunk = full["tensors"][0]["chunks"][0]
            for broken in (None, b"bad"):
                path = self.store.object_path(chunk["object"])
                if broken is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(broken)
                self.assertEqual(self.store.ensure_full("s", version), full)
                cached = fetch(
                    SharedFSTransport(self.store), self.make_store(f"copy-{version}"), "s", version, full=True
                )
                self.assertEqual(cached["root"], full["root"])
        full = self.store.ensure_full("s", 2)
        self.store.object_path(full["tensors"][0]["chunks"][0]["object"]).unlink()
        self.store.tensor_path("s", 2, "w").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "cannot repair"):
            self.store.ensure_full("s", 2)

    def test_collection_replays_manifest_delete_interruption(self):
        for version in range(6):
            self.publish(version, anchor_interval=2)
        original = Path.unlink

        def interrupted(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == self.store.manifest_path("s", 0):
                raise RuntimeError("power loss after manifest unlink")
            return result

        with patch.object(Path, "unlink", interrupted):
            with self.assertRaises(RuntimeError):
                self.store.collect()
        self.assertTrue(self.store.tensor_path("s", 0, "w").exists())
        restarted = self.make_store("source")
        restarted.collect()
        self.assertFalse(restarted.tensor_path("s", 0, "w").exists())
        self.assertFalse(list((restarted.root / "deletions").glob("*.json")))
        self.assertEqual(restarted.materialize("s", 5)["version"], 5)
        self.assertEqual(restarted.collect()["removed_versions"], 0)

    def test_failed_publication_reclaims_canonical_before_retry(self):
        self.publish(0)
        with patch.object(self.store, "commit", side_effect=OSError("precommit crash")):
            with self.assertRaises(OSError):
                self.publish(1)
        orphan = self.store.tensor_path("s", 1, "w")
        self.assertFalse(orphan.exists())
        self.publish(1)
        self.assertEqual(self.store.materialize("s", 1)["version"], 1)
        self.assertEqual(self.store.materialize("s", 0)["version"], 0)

    def test_unchanged_raw_storage_is_shared_across_anchors(self):
        path = self.root / "input"
        path.write_bytes(bytes(65536))
        for version in range(100):
            self.store.publish("s", version, {"w": ("BF16", [32768], path)}, anchor_interval=20)
        files = list(self.store.canonical_directory.glob("s/*/*.bin"))
        self.assertEqual(len(files), 100)
        self.assertEqual(stored_bytes(self.store.canonical_directory), 65536)
        self.store.collect()
        self.assertEqual(self.store.tensor_path("s", 99, "w").read_bytes(), bytes(65536))

    def test_consumers_progress_while_publisher_encodes(self):
        from relax.distributed.weight_sync import store

        first = self.publish(0)
        self.store.register("one", "s")
        entered, release = threading.Event(), threading.Event()
        encode, errors = store.encode, []

        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test release missing")
            return encode(*args, **kwargs)

        def publish():
            try:
                self.publish(1)
            except Exception as error:
                errors.append(error)

        with patch.object(store, "encode", delayed):
            thread = threading.Thread(target=publish)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.store.register("two", "s")
                self.store.pin("s", 0, "active")
                target = {k: first[k] for k in ("stream", "version", "root")}
                self.store.acknowledge("one", {"phase": "ACTIVE", "target": target, "serving": target})
                self.assertEqual([v["version"] for v in self.store.versions("s")], [0])
            finally:
                release.set()
                thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual([v["version"] for v in self.store.versions("s")], [0, 1])

    def test_busy_does_not_consume_network_retry_count(self):
        self.publish(0)
        with ObjectServer(("127.0.0.1", 0), self.store, max_workers=1) as server:
            server._slots.acquire()
            timer = threading.Timer(0.3, server._slots.release)
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            timer.start()
            try:
                result = TCPTransport(server.server_address, timeout=2, retries=0).versions("s")
                self.assertEqual(result[0]["version"], 0)
            finally:
                timer.join()
                server.shutdown()
                worker.join()


class IdentityTests(unittest.TestCase):
    def test_old_admission_claim_cannot_replace_new_installation(self):
        fence = InstallationFence()
        old = {"epoch": 1, "installation_id": "old", "root": "one", "workers": {"0": "worker"}}
        new = {**old, "epoch": 2, "installation_id": "new", "root": "two"}
        fence.claim(old)
        fence.claim(new)
        for operation in (fence.claim, fence.require):
            with self.assertRaises(ValueError):
                operation(old)
        self.assertEqual(fence.identity, new)
        fence.require(new)

    def test_worker_retry_binds_command_content_and_phase(self):
        fence = WorkerFence()
        identity = fence.begin(fence.instance, 0, "install", "root")
        calls = []
        with self.assertRaises(ValueError):
            fence.execute(identity, "verify", "verify", lambda: calls.append("premature"))
        result = fence.execute(identity, "load", "load", lambda: calls.append("load"), payload={"path": "one"})
        self.assertEqual(
            fence.execute(identity, "load", "load", lambda: calls.append("duplicate"), payload={"path": "one"}), result
        )
        with self.assertRaises(ValueError):
            fence.execute(identity, "load", "load", lambda: calls.append("bad"), payload={"path": "two"})
        self.assertEqual(calls, ["load"])


if __name__ == "__main__":
    unittest.main()
