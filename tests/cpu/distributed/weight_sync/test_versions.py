import random
import tempfile
import threading
import unittest
from pathlib import Path

from relax.distributed.weight_sync.checkpoint import write_checkpoint
from relax.distributed.weight_sync.session import InstallSession
from relax.distributed.weight_sync.store import VersionStore
from relax.distributed.weight_sync.transport import ObjectServer, SharedFSTransport, TCPTransport, fetch


class VersionTests(unittest.TestCase):
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

    def test_compressed_version_replay(self):
        source, target = self.store("compressed"), self.store("receiver")
        value = bytearray(random.Random(0).randbytes(8192))
        path = self.root / "compressed.bin"
        for version in range(4):
            value[version] ^= 1
            path.write_bytes(value)
            source.publish("s", version, {"w": ("BF16", [4096], path)}, compression=True, model_config={})
            fetch(SharedFSTransport(source), target, "s", version)
            self.assertEqual(target.tensor_path("s", version, "w").read_bytes(), value)

    def test_offline_hundred_versions_both_transports(self):
        source = self.store("source")
        values = []
        value = bytearray(256)
        for version in range(101):
            value[version % 256] ^= 1
            values.append(bytes(value))
            self.publish(source, version, value)
        # Producer no longer exists: only durable files are used below.
        with ObjectServer(("127.0.0.1", 0), source) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for mode, transport in [
                    ("fs", SharedFSTransport(source)),
                    ("tcp", TCPTransport(server.server_address)),
                ]:
                    target = self.store(mode)
                    for version, expected in enumerate(values):
                        fetch(transport, target, "test", version)
                        self.assertEqual(target.tensor_path("test", version, "w").read_bytes(), expected)
            finally:
                server.shutdown()
                worker.join()

    def test_corrupt_object_and_cache(self):
        source, target = self.store("source"), self.store("target")
        m = self.publish(source, 0, b"0" * 128)
        fetch(SharedFSTransport(source), target, "test", 0)
        target.tensor_path("test", 0, "w").write_bytes(b"bad")
        target.materialize("test", 0)
        self.assertEqual(target.tensor_path("test", 0, "w").read_bytes(), b"0" * 128)
        key = m["tensors"][0]["chunks"][0]["object"]
        source.object_path(key).write_bytes(b"1" * 64)
        with self.assertRaises(ValueError):
            fetch(SharedFSTransport(source), self.store("fresh"), "test", 0)

    def test_resume_after_interruption(self):
        source, target = self.store("source"), self.store("target")
        m = self.publish(source, 0, bytes(range(128)))
        key = m["tensors"][0]["chunks"][0]["object"]
        target.put_object(key, source.object_path(key).read_bytes())

        class Resumed(SharedFSTransport):
            def object(self, requested):
                if requested == key:
                    raise AssertionError("Durable object retransmitted")
                return super().object(requested)

        fetch(Resumed(source), target, "test", 0)
        self.assertEqual(target.tensor_path("test", 0, "w").read_bytes(), bytes(range(128)))

    def test_partial_install_fail_closed_and_recovery(self):
        class Group:
            def __init__(self):
                self.serving, self.fail, self.member = True, True, "one"

            def identity(self):
                return self.member

            def close(self):
                self.serving = False

            def pause(self):
                pass

            def load(self, checkpoint, version):
                pass

            def verify(self, checkpoint):
                if self.fail:
                    raise ValueError("injected bit corruption")

            def flush(self):
                pass

            def resume(self):
                pass

            def open(self):
                self.serving = True

        group = Group()
        session = InstallSession(self.root / "session", group)
        m = self.publish(self.store("source"), 0, b"0" * 128)
        config = self.root / "config.json"
        config.write_text("{}")
        checkpoint = write_checkpoint(self.store("source"), "test", 0, self.root / "checkpoint", config_source=config)
        with self.assertRaises(ValueError):
            session.install(checkpoint, m, "i1")
        self.assertFalse(group.serving)
        self.assertEqual(session.status()["phase"], "UNAVAILABLE")
        group.fail = False
        session.install(checkpoint, m, "i1")
        self.assertTrue(group.serving)
        restarted = InstallSession(self.root / "session", group)
        self.assertFalse(group.serving)
        restarted.install(checkpoint, m, "i1")
        self.assertTrue(group.serving)


if __name__ == "__main__":
    unittest.main()
