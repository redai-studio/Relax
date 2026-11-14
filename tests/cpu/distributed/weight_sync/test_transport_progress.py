# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import socket
import socketserver
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

from relax.distributed.weight_sync.store import VersionStore, canonical, digest
from relax.distributed.weight_sync.transport import (
    BUSY_RESPONSE,
    ERROR_RESPONSE,
    ObjectServer,
    TCPTransport,
    receive_exact,
)


class ProgressTests(unittest.TestCase):
    def test_error_response_parts_share_the_object_deadline(self):
        detail = canonical({"type": "ValueError", "message": "remote failure"})

        class Handler(socketserver.BaseRequestHandler):
            def handle(inner):
                inner.request.settimeout(1)
                size = struct.unpack("!I", receive_exact(inner.request, 4))[0]
                receive_exact(inner.request, size)
                try:
                    for part in (struct.pack("!Q", ERROR_RESPONSE), struct.pack("!I", len(detail)), detail):
                        time.sleep(0.12)
                        inner.request.sendall(part)
                except OSError:
                    pass  # The client may close when the shared deadline expires.

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            thread.start()
            try:
                with self.assertRaises(ConnectionError):
                    TCPTransport(server.server_address, timeout=0.3, retries=0).object("0" * 64)
            finally:
                server.shutdown()
                thread.join()

    def test_partial_response_and_retry_are_both_counted(self):
        value = b"complete object"
        key = digest(value)
        response = struct.pack("!Q", len(value)) + value
        for cutoff in (3, 11):  # Interrupted header and interrupted body.
            with self.subTest(cutoff=cutoff):
                requests = []

                class Handler(socketserver.BaseRequestHandler):
                    def handle(inner):
                        inner.request.settimeout(1)
                        size = struct.unpack("!I", receive_exact(inner.request, 4))[0]
                        requests.append(receive_exact(inner.request, size))
                        inner.request.sendall(response[:cutoff] if len(requests) == 1 else response)

                with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
                    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
                    thread.start()
                    try:
                        transport = TCPTransport(server.server_address, timeout=2, retries=1)
                        self.assertEqual(transport.object(key), value)
                        self.assertEqual(transport.bytes_received, cutoff + len(response))
                        self.assertEqual(transport.bytes_sent, 2 * (4 + len(canonical({"op": "object", "key": key}))))
                        self.assertEqual(len(requests), 2)
                    finally:
                        server.shutdown()
                        thread.join()

    def test_slow_client_does_not_block_other_slot_and_overload_is_explicit(self):
        condition = threading.Condition()
        active = 0

        class ObservedServer(ObjectServer):
            def process_request_thread(self, request, client_address):
                nonlocal active
                with condition:
                    active += 1
                    condition.notify_all()
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    with condition:
                        active -= 1
                        condition.notify_all()

        def wait_active(count):
            with condition:
                self.assertTrue(condition.wait_for(lambda: active == count, timeout=2))

        with tempfile.TemporaryDirectory() as d:
            store = VersionStore(Path(d), disk_budget=1024**2, max_weight_bytes=1024**2)
            payload = b"independent fast request"
            key = digest(payload)
            store.put_object(key, payload)
            with ObservedServer(("127.0.0.1", 0), store, timeout=5.0, max_workers=2) as server:
                thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
                thread.start()
                slow = socket.create_connection(server.server_address)
                slow.sendall(b"\0")  # Incomplete request header holds one slot.
                wait_active(1)
                try:
                    self.assertEqual(TCPTransport(server.server_address, timeout=0.6).object(key), payload)
                    wait_active(1)  # The fast handler has released its worker slot.
                    second = socket.create_connection(server.server_address)
                    second.sendall(b"\0")
                    try:
                        wait_active(2)
                        with socket.create_connection(server.server_address, 0.6) as extra:
                            self.assertEqual(struct.unpack("!Q", receive_exact(extra, 8))[0], BUSY_RESPONSE)
                    finally:
                        second.close()
                    slow.close()
                    self.assertEqual(TCPTransport(server.server_address, timeout=1.0, retries=10).object(key), payload)
                finally:
                    slow.close()
                    server.shutdown()
                    thread.join()

    def test_slow_header_releases_worker_on_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            store = VersionStore(Path(d), disk_budget=1024**2, max_weight_bytes=1024**2)
            with ObjectServer(("127.0.0.1", 0), store, timeout=0.2, max_workers=1) as server:
                t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
                t.start()
                try:
                    with socket.create_connection(server.server_address, 1.0) as slow:
                        slow.sendall(b"\0")
                        self.assertEqual(slow.recv(1), b"")
                    value = b"after deadline"
                    key = digest(value)
                    store.put_object(key, value)
                    self.assertEqual(TCPTransport(server.server_address, timeout=1.0).object(key), value)
                finally:
                    server.shutdown()
                    t.join()

    def test_retry_budget_is_total_not_per_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            store = VersionStore(Path(d), disk_budget=1024**2, max_weight_bytes=1024**2)
            with ObjectServer(("127.0.0.1", 0), store, timeout=2.0, max_workers=1) as server:
                t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
                t.start()
                slow = socket.create_connection(server.server_address)
                slow.sendall(b"\0")
                started = time.monotonic()
                try:
                    with self.assertRaises(ConnectionError):
                        TCPTransport(server.server_address, timeout=0.15, retries=10).object("0" * 64)
                    self.assertLess(time.monotonic() - started, 0.65)
                finally:
                    slow.close()
                    server.shutdown()
                    t.join()


if __name__ == "__main__":
    unittest.main()
