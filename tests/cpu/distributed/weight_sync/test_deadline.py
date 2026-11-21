import asyncio
import socketserver
import struct
import threading
import time
import unittest
from types import SimpleNamespace

from relax.distributed.weight_sync.deadline import time_budget
from relax.distributed.weight_sync.sglang_loader import EmbeddedSGLangGroup
from relax.distributed.weight_sync.transport import TCPTransport, receive_exact


class DeadlineTests(unittest.TestCase):
    def test_multiple_tcp_objects_share_one_operation_budget(self):
        class Handler(socketserver.BaseRequestHandler):
            def handle(inner):
                inner.request.settimeout(1)
                size = struct.unpack("!I", receive_exact(inner.request, 4))[0]
                receive_exact(inner.request, size)
                time.sleep(0.16)
                try:
                    inner.request.sendall(struct.pack("!Q", 1) + b"x")
                except OSError:
                    pass

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            worker.start()
            try:
                transport = TCPTransport(server.server_address, timeout=2, retries=0)
                with time_budget(0.25):
                    self.assertEqual(transport.object("0" * 64), b"x")
                    with self.assertRaises(ConnectionError):
                        transport.object("1" * 64)
            finally:
                server.shutdown()
                worker.join()

    def test_embedded_probe_timeout_requires_engine_restart(self):
        cancelled, flushed = [], []

        async def generate(**kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

        group = EmbeddedSGLangGroup.__new__(EmbeddedSGLangGroup)
        group.identity = lambda: "fixed-worker"
        group._loaded_version, group._rpc_failed, group._serving = "s:1:root", False, False
        group.flush = lambda: flushed.append(True)
        loop = asyncio.new_event_loop()
        group.engine = SimpleNamespace(
            loop=loop,
            async_generate=generate,
            tokenizer_manager=SimpleNamespace(_update_weight_version_if_provided=lambda value: None),
        )
        try:
            with time_budget(0.05), self.assertRaisesRegex(TimeoutError, "restart the engine"):
                group.resume()
            self.assertTrue(cancelled)
            self.assertFalse(flushed)
            self.assertTrue(group._rpc_failed)
            self.assertFalse(group._serving)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
