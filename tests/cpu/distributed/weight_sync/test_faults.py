import socketserver
import struct
import tempfile
import threading
import unittest
from pathlib import Path

from relax.distributed.weight_sync.checkpoint import validate_checkpoint, write_checkpoint
from relax.distributed.weight_sync.session import InstallSession
from relax.distributed.weight_sync.store import VersionStore, writer_lock
from relax.distributed.weight_sync.transport import TCPTransport, receive_exact


def checkpoint_fixture(directory, version=1):
    root = Path(directory)
    store = VersionStore(root / "store", disk_budget=100000, max_weight_bytes=1024)
    (root / "weight.bin").write_bytes(b"\0" * 16)
    (root / "config.json").write_text("{}")
    m = store.publish("s", version, {"w": ("BF16", [8], root / "weight.bin")}, full_only=True, model_config={})
    p = write_checkpoint(store, "s", version, root / f"checkpoint-{version}", config_source=root / "config.json")
    return p, m


class FaultTests(unittest.TestCase):
    def test_checkpoint_config_is_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest = checkpoint_fixture(directory)
            (checkpoint / "config.json").write_text('{"rope_theta": 10000}')
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, manifest)

    def test_checkpoint_bytes_are_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest = checkpoint_fixture(directory)
            validate_checkpoint(checkpoint, manifest)
            path = checkpoint / "model.safetensors"
            value = bytearray(path.read_bytes())
            value[-1] ^= 1
            path.write_bytes(value)
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, manifest)

    def test_tcp_partial_frame_retried_and_accounted(self):
        calls = []

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(1)
                n = struct.unpack("!I", receive_exact(self.request, 4))[0]
                receive_exact(self.request, n)
                calls.append(1)
                self.request.sendall(struct.pack("!Q", 20))
                self.request.sendall(b"a" * (3 if len(calls) == 1 else 20))

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                transport = TCPTransport(server.server_address, timeout=1)
                self.assertEqual(transport.object("a" * 64), b"a" * 20)
                self.assertEqual(transport.bytes_received, 16 + 3 + 20)
                self.assertEqual(len(calls), 2)
            finally:
                server.shutdown()
                thread.join()

    def test_two_writers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".writer"
            with writer_lock(path):
                with self.assertRaises(OSError):
                    with writer_lock(path):
                        pass

    def test_commit_failure_does_not_open_gate_or_change_target(self):
        class Engines:
            serving = False
            failing = True
            incarnation = "a"

            def identity(self):
                return self.incarnation

            def close(self):
                self.serving = False

            def pause(self):
                pass

            def load(self, *args):
                pass

            def verify(self, *args):
                pass

            def flush(self):
                pass

            def resume(self):
                if self.failing:
                    raise TimeoutError("lost resume reply")

            def open(self):
                self.serving = True

        with tempfile.TemporaryDirectory() as directory:
            engines = Engines()
            session = InstallSession(directory, engines)
            checkpoint, m = checkpoint_fixture(directory)
            with self.assertRaises(TimeoutError):
                session.install(checkpoint, m, "one")
            self.assertFalse(session.status()["committed"])
            self.assertFalse(engines.serving)
            old_checkpoint, old = checkpoint_fixture(directory, 0)
            with self.assertRaises(ValueError):
                session.install(old_checkpoint, old, "old")
            engines.failing = False
            session.install(checkpoint, m, "one")
            self.assertTrue(engines.serving)

    def test_member_change_during_load_is_rejected(self):
        class Engines:
            incarnation = "old"
            serving = False

            def identity(self):
                return self.incarnation

            def close(self):
                self.serving = False

            def pause(self):
                pass

            def load(self, *args):
                self.incarnation = "new"

            def verify(self, *args):
                pass

            def flush(self):
                raise AssertionError("Must reject before commit")

            def resume(self):
                raise AssertionError("Must not resume")

            def open(self):
                self.serving = True

        with tempfile.TemporaryDirectory() as directory:
            engines = Engines()
            session = InstallSession(directory, engines)
            checkpoint, m = checkpoint_fixture(directory)
            with self.assertRaises(ValueError):
                session.install(checkpoint, m, "one")
            self.assertFalse(engines.serving)
            self.assertFalse(session.status()["committed"])


if __name__ == "__main__":
    unittest.main()
