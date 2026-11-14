import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from relax.distributed.weight_sync.budget import NodeBudget
from relax.distributed.weight_sync.consumer import VersionConsumer
from relax.distributed.weight_sync.runtime import ConsumptionError, ConsumptionRunner
from relax.distributed.weight_sync.session import InstallSession
from relax.distributed.weight_sync.store import VersionStore, canonical, load_json
from relax.distributed.weight_sync.transport import SharedFSTransport


class Engines:
    def __init__(self):
        self.serving, self.failing, self.loads = False, False, 0

    def identity(self):
        return "fixed-members"

    def close(self):
        self.serving = False

    def pause(self):
        pass

    def load(self, checkpoint, version):
        self.loads += 1
        if self.failing:
            raise OSError("incomplete installation")
        self.bytes = (checkpoint / "model.safetensors").read_bytes()

    def verify(self, checkpoint):
        assert self.bytes == (checkpoint / "model.safetensors").read_bytes()

    def flush(self):
        pass

    def resume(self):
        pass

    def open(self):
        self.serving = True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = VersionStore(self.root / "source", disk_budget=32 * 1024**2, max_weight_bytes=1024**2)
        for version in range(4):
            path = self.root / str(version)
            path.write_bytes(bytes([version]) + bytes(255))
            self.source.publish("s", version, {"w": ("BF16", [128], path)}, model_config={})

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, transport, *, mode="all_versions"):
        cache = VersionStore(self.root / "cache", disk_budget=32 * 1024**2, max_weight_bytes=1024**2)
        consumer = VersionConsumer(transport, cache, self.root / "checkpoints", checkpoint_budget=32 * 1024**2)
        engines = Engines()
        return ConsumptionRunner(
            consumer, InstallSession(self.root / "session", engines), "reader", "s", mode=mode
        ), engines

    def test_latest_installs_only_requested_target_and_confirms_it(self):
        runner, engines = self.runner(SharedFSTransport(self.source), mode="latest")
        self.assertEqual([item["version"] for item in runner.available()], [3])
        result = runner.apply(3)
        self.assertEqual(result["cursor"], 3)
        self.assertEqual(engines.loads, 1)
        self.assertTrue(engines.serving)
        runner.apply(3)
        self.assertEqual(engines.loads, 1)
        with self.assertRaises(ValueError):
            runner.apply(2)

    def test_capacity_shortage_explains_recovery_and_preserves_serving(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        runner.node_budget = NodeBudget(self.root / "resources", {"host": 100}, {"host": 50})
        runner.operation_resources = {"host": 60}
        with self.assertRaises(ConsumptionError):
            runner.apply(1)
        state = runner.status()
        self.assertEqual(state["stage"], "RESERVING")
        self.assertEqual(state["status"], "WAITING_ACTION")
        self.assertIn("increase_deployment_capacity", state["allowed_actions"])
        self.assertIn("'required': 110", state["reason"])
        self.assertTrue(engines.serving)
        self.assertEqual(engines.loads, 1)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)
        NodeBudget(self.root / "resources", {"host": 120}, {"host": 50})
        self.assertEqual(runner.apply(1)["cursor"], 1)
        self.assertFalse(list((self.root / "resources").glob("reservation-*.json")))

    def test_standalone_consumer_uses_shared_node_budget(self):
        from relax.distributed.weight_sync.cli_consume import consume_versions

        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        config = {
            "start_version": 0,
            "end_version": 0,
            "model_path": str(model),
            "transport": {"type": "sharedfs", "store": str(self.source.root), "disk_budget": 32 * 1024**2},
            "checkpoint_root": str(self.root / "checkpoints"),
            "checkpoint_budget": 32 * 1024**2,
            "session_directory": str(self.root / "session"),
            "consumer": "standalone",
            "stream": "s",
            "node_directory": str(self.root / "shared-node"),
            "node_budget": {"limits": {"host": 100}, "resident": {"host": 30}, "operation": {"host": 60}},
        }
        path = self.root / "consumer.json"
        path.write_bytes(canonical(config))
        cache = VersionStore(self.root / "cache", disk_budget=32 * 1024**2, max_weight_bytes=1024**2)
        engine, group = Mock(), Engines()
        module = "relax.distributed.weight_sync.cli_consume"
        with (
            patch(module + ".create_engine", return_value=engine),
            patch(module + ".EmbeddedSGLangGroup", return_value=group),
        ):
            consume_versions(cache, path)
        engine.shutdown.assert_called_once()
        self.assertTrue(group.serving)
        self.assertEqual(self.source.consumer_status("standalone")["cursor"], 0)
        resources = self.root / "shared-node/resources"
        self.assertEqual(load_json((resources / "configuration.json").read_bytes())["resident"], {"host": 30})
        self.assertFalse(list(resources.glob("reservation-*.json")))

    def test_preparation_deadline_preserves_active_engine_and_reuses_download(self):
        transport = SharedFSTransport(self.source)
        runner, engines = self.runner(transport)
        runner.apply(0)
        runner.operation_timeout = 1
        clock = [10.0]
        original = transport.object
        downloaded = []

        def delayed(key):
            value = original(key)
            downloaded.append(key)
            clock[0] += 2
            return value

        with patch("relax.distributed.weight_sync.deadline.time.monotonic", side_effect=lambda: clock[0]):
            with patch.object(transport, "object", side_effect=delayed), self.assertRaises(ConsumptionError):
                runner.apply(1)
        self.assertTrue(engines.serving)
        self.assertEqual(engines.loads, 1)
        self.assertEqual(runner.status()["stage"], "PREPARING")
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)
        self.assertEqual(list((self.source.root / "pins").glob("fetch-*.json")), [])
        self.assertTrue(downloaded)
        with patch.object(transport, "object", side_effect=AssertionError("completed object downloaded again")):
            self.assertEqual(runner.apply(1)["cursor"], 1)

    def test_installation_deadline_closes_gate_and_keeps_recovery_checkpoint(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        runner.operation_timeout = 1
        clock = [10.0]
        original = engines.load

        def delayed(*args):
            original(*args)
            clock[0] += 2

        with patch("relax.distributed.weight_sync.deadline.time.monotonic", side_effect=lambda: clock[0]):
            with patch.object(engines, "load", side_effect=delayed), self.assertRaises(ConsumptionError):
                runner.apply(1)
        self.assertFalse(engines.serving)
        self.assertEqual(runner.session.status()["phase"], "UNAVAILABLE")
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)
        self.assertEqual(runner.recover_active()["active_version"]["version"], 0)
        self.assertTrue(engines.serving)
        self.assertEqual(runner.apply(1, full=True)["cursor"], 1)

    def test_catch_up_gives_each_target_its_own_deadline(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.operation_timeout = 1
        clock = [10.0]
        original = engines.load

        def delayed(*args):
            original(*args)
            clock[0] += 0.6

        with patch("relax.distributed.weight_sync.deadline.time.monotonic", side_effect=lambda: clock[0]):
            with patch.object(engines, "load", side_effect=delayed):
                self.assertEqual(runner.apply_through(3)["cursor"], 3)
        self.assertTrue(engines.serving)
        self.assertEqual(engines.loads, 4)
        with self.assertRaisesRegex(ValueError, "Catch-up target precedes"):
            runner.apply_through(2)

    def test_catch_up_full_retry_resumes_failed_next_version(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply_through(0)
        engines.failing = True
        with self.assertRaises(ConsumptionError):
            runner.apply_through(3)
        self.assertEqual(runner.status()["target"], 1)
        engines.failing = False
        self.assertEqual(runner.apply_through(3, full=True)["cursor"], 3)
        self.assertTrue(engines.serving)

    def test_failed_suspend_cannot_reuse_stale_active_receipt(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        with patch.object(engines, "close", side_effect=TimeoutError("generation still active")):
            with self.assertRaises(TimeoutError):
                runner.session.suspend()
        self.assertEqual(runner.session.status()["phase"], "UNAVAILABLE")
        self.assertIsNone(runner.session.active_receipt("reader:s:0"))
        before = engines.loads
        runner.apply_through(0)
        self.assertEqual(engines.loads, before + 1)
        self.assertTrue(engines.serving)

    def test_latest_failed_install_does_not_advance_cursor(self):
        runner, engines = self.runner(SharedFSTransport(self.source), mode="latest")
        runner.apply(0)
        engines.failing = True
        with self.assertRaises(ConsumptionError):
            runner.apply(3)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)
        engines.failing = False
        runner.recover_active()
        self.assertTrue(engines.serving)
        self.assertEqual(runner.session.status()["serving"]["version"], 0)
        self.assertEqual(runner.apply(3, full=True)["cursor"], 3)

    def test_same_live_version_reuses_installation_without_rereading_checkpoint(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        with patch.object(runner.consumer, "prepare", side_effect=AssertionError("unexpected checkpoint read")):
            result = runner.apply(0)
        self.assertTrue(result["preparation"]["reused_active"])
        self.assertEqual(engines.loads, 1)
        self.assertTrue(engines.serving)

    def test_replaced_workers_invalidate_live_receipt(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        engines.identity = lambda: "replacement-members"
        engines.bytes = b"new worker initialization"
        result = runner.apply(0)
        self.assertEqual(engines.loads, 2)
        self.assertEqual(result["receipt"]["engines"], "replacement-members")
        self.assertNotEqual(engines.bytes, b"new worker initialization")

    def test_timing_write_failure_keeps_durably_activated_model_serving(self):
        from relax.distributed.weight_sync import session

        runner, engines = self.runner(SharedFSTransport(self.source))
        write = session.atomic_write

        def fail_after_open(path, content):
            if engines.serving:
                raise OSError("injected timing write failure")
            return write(path, content)

        with patch.object(session, "atomic_write", side_effect=fail_after_open):
            result = runner.apply(0)
        self.assertEqual(result["cursor"], 0)
        self.assertTrue(engines.serving)
        self.assertEqual(runner.session.status()["phase"], "ACTIVE")
        self.assertEqual(runner.session.status()["serving"]["version"], 0)
        runner.apply(1)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 1)

    def test_preparation_diagnostics_failure_keeps_checkpoint_installable(self):
        from relax.distributed.weight_sync import store as consumer

        runner, engines = self.runner(SharedFSTransport(self.source))
        write = consumer.atomic_write

        def fail_diagnostics(path, content):
            if path.name == "prepare-metrics.json":
                raise OSError("injected diagnostic write failure")
            return write(path, content)

        with patch.object(consumer, "atomic_write", side_effect=fail_diagnostics):
            result = runner.apply(0)
        self.assertEqual(result["cursor"], 0)
        self.assertTrue(engines.serving)
        self.assertEqual(runner.session.status()["phase"], "ACTIVE")
        self.assertTrue(list((runner.consumer.root / ".leases").glob("*.json")))

    def test_failed_install_recovers_old_then_full_retry_advances_cursor(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        prior_bytes = engines.bytes
        engines.failing = True
        with self.assertRaises(ConsumptionError) as caught:
            runner.apply(1)
        self.assertEqual(caught.exception.state["stage"], "INSTALLING")
        self.assertFalse(engines.serving)
        engines.failing = False
        recovered = runner.recover_active()
        self.assertEqual(recovered["target"], 1)
        self.assertEqual(recovered["active_version"]["version"], 0)
        self.assertEqual(engines.bytes, prior_bytes)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)
        final = runner.apply(1, full=True)
        self.assertEqual(final["cursor"], 1)
        self.assertNotEqual(engines.bytes, prior_bytes)
        self.assertEqual(len(list((runner.consumer.root / ".leases").glob("*.json"))), 1)

    def test_full_retry_reuses_verified_target_under_two_checkpoint_budget(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        engines.failing = True
        with self.assertRaises(ConsumptionError):
            runner.apply(1)
        retained = load_json((runner.directory / "retained-checkpoints.json").read_bytes())
        failed_target = next(record for record in retained.values() if record["manifest"]["version"] == 1)
        engines.failing = False
        runner.recover_active()
        used = sum(path.stat().st_size for path in runner.consumer.root.rglob("*") if path.is_file())
        runner.consumer.checkpoint_budget = used + 4096
        result = runner.apply(1, full=True)
        active = runner._active()
        self.assertEqual(result["cursor"], 1)
        self.assertEqual(active.checkpoint.name, failed_target["checkpoint"])
        self.assertEqual(active.manifest["kind"], "full")
        self.assertNotEqual(active.manifest["artifact_id"], failed_target["manifest"]["artifact_id"])
        self.assertEqual(len(list(runner.consumer.root.glob("*/model.safetensors"))), 1)
        self.assertTrue(engines.serving)

    def test_offload_requires_reinstallation_before_acknowledged_version_serves(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        loads = engines.loads
        runner.session.suspend()
        self.assertFalse(engines.serving)
        self.assertEqual(runner.session.status()["phase"], "SUSPENDED")
        engines.bytes = b"released device allocation"
        runner.apply(0)
        self.assertEqual(engines.loads, loads + 1)
        self.assertTrue(engines.serving)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)
        runner.apply(1)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 1)

    def test_lost_ack_retry_is_idempotent_without_reloading_live_engine(self):
        class LostReply(SharedFSTransport):
            lose = True

            def acknowledge(self, consumer, receipt):
                value = super().acknowledge(consumer, receipt)
                if self.lose:
                    self.lose = False
                    raise ConnectionError("acknowledgement reply was lost")
                return value

        runner, engines = self.runner(LostReply(self.source))
        with self.assertRaises(ConsumptionError) as caught:
            runner.apply(0)
        self.assertEqual(caught.exception.state["stage"], "ACKNOWLEDGING")
        self.assertTrue(engines.serving)
        with patch.object(runner.consumer, "prepare", side_effect=AssertionError("unexpected checkpoint read")):
            final = runner.apply(0)
        self.assertEqual(final["cursor"], 0)
        self.assertEqual(engines.loads, 1)

    def test_restart_requires_reloading_before_serving(self):
        runner, original = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        restarted, engines = self.runner(SharedFSTransport(self.source))
        self.assertFalse(engines.serving)
        restarted.apply(0)
        self.assertTrue(engines.serving)
        self.assertEqual(engines.loads, 1)
        self.assertEqual(engines.bytes, original.bytes)

    def test_admission_timeout_records_recovery_action_even_if_closing_again_fails(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)

        def timeout():
            raise TimeoutError("drain deadline exceeded")

        engines.close = timeout
        with self.assertRaises(ConsumptionError) as caught:
            runner.apply(1)
        state = runner.session.status()
        self.assertEqual(state["phase"], "UNAVAILABLE")
        self.assertEqual(state["error"], "drain deadline exceeded")
        self.assertIsNone(state["serving"])
        self.assertIn("restart_unresponsive_engine_then_recover", caught.exception.state["allowed_actions"])

    def test_diagnostic_status_write_failure_preserves_active_receipt(self):
        from relax.distributed.weight_sync import store

        runner, engines = self.runner(SharedFSTransport(self.source))
        original = store.atomic_write

        def write(path, payload):
            if path.name == "operation.json":
                raise PermissionError("diagnostic directory denied")
            return original(path, payload)

        with patch.object(store, "atomic_write", side_effect=write):
            result = runner.apply(0)
        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertEqual(runner.status()["stage"], "ACTIVE")
        self.assertTrue(engines.serving)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 0)

    def test_old_checkpoint_cleanup_failure_does_not_fail_active_installation(self):
        runner, engines = self.runner(SharedFSTransport(self.source))
        runner.apply(0)
        with patch.object(runner.consumer, "release", side_effect=PermissionError("busy old checkpoint")):
            result = runner.apply(1)
        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertEqual(result["cleanup"]["status"], "PENDING")
        self.assertTrue(engines.serving)
        self.assertEqual(self.source.consumer_status("reader")["cursor"], 1)
        self.assertEqual(len(load_json((runner.directory / "retained-checkpoints.json").read_bytes())), 2)
        result = runner.apply(1)
        self.assertEqual(result["cleanup"]["status"], "COMPLETE")
        self.assertEqual(len(load_json((runner.directory / "retained-checkpoints.json").read_bytes())), 1)
        self.assertEqual(engines.loads, 2)

    def test_two_consumer_groups_install_concurrently_with_shared_capacity(self):
        from concurrent.futures import ThreadPoolExecutor

        barrier = threading.Barrier(2)
        cache = VersionStore(self.root / "shared-cache", disk_budget=32 * 1024**2, max_weight_bytes=1024**2)
        budget = NodeBudget(self.root / "node-budget", {"host": 100}, {"host": 20})
        runners, groups = [], []
        for index in range(2):
            group = Engines()
            original_load = group.load

            def load(checkpoint, version, original=original_load):
                barrier.wait(timeout=3)
                original(checkpoint, version)

            group.load = load
            groups.append(group)
            runners.append(
                ConsumptionRunner(
                    VersionConsumer(
                        SharedFSTransport(self.source),
                        cache,
                        self.root / "shared-checkpoints",
                        checkpoint_budget=32 * 1024**2,
                    ),
                    InstallSession(self.root / f"parallel-session-{index}", group),
                    f"parallel-{index}",
                    "s",
                    node_budget=budget,
                    operation_resources={"host": 30},
                )
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(runner.apply, 0) for runner in runners]
            assert all(future.result(timeout=8)["status"] == "SUCCEEDED" for future in futures)
        assert all(group.serving for group in groups)
        assert not list(budget.directory.glob("reservation-*.json"))


if __name__ == "__main__":
    unittest.main()
