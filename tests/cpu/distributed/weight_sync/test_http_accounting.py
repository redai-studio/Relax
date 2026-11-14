import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from relax.distributed.weight_sync.deadline import time_budget
from relax.distributed.weight_sync.sglang_loader import SGLangHTTPGroup


def test_actual_control_bodies_are_counted_on_success_and_error(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(body)
            value = json.loads(body)
            reply = b'{"ok":true}'
            self.send_response(409 if value["action"] == "fail" else 200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        group = SGLangHTTPGroup.__new__(SGLangHTTPGroup)
        group.url = f"http://127.0.0.1:{server.server_port}"
        group.owner, group.installation_identity = "test", None
        group.headers = {"Authorization": "Bearer synthetic-test-value"}
        group.audit_directory = tmp_path
        try:
            with time_budget(1800):
                assert group._control("status") == {"ok": True}
            assert json.loads(received[0])["timeout"] > 1700
            with pytest.raises(RuntimeError, match="409"):
                group._control("fail")
            recorded = next(tmp_path.glob("control-metrics-*.json"))
            raw = recorded.read_text()
            report = json.loads(raw)
            assert report["bytes_by_scope"]["http_body"]["C"] == sum(map(len, received)) + 2 * len(b'{"ok":true}')
            assert "synthetic-test-value" not in raw and "Authorization" not in raw
            assert not report["physical_link_coverage_complete"]
            # A fresh controller reuses the audit directory after a server
            # restart. Its counters must not erase the previous controller.
            del group.control_metrics
            group.owner = "restarted"
            with time_budget(1800):
                assert group._control("status") == {"ok": True}
            assert json.loads(received[0])["timeout"] > 1700
            assert recorded.read_text() == raw
            receipts = [json.loads(path.read_text()) for path in tmp_path.glob("control-metrics-*.json")]
            assert len(receipts) == 2
            assert sum(item["bytes_by_scope"]["http_body"]["C"] for item in receipts) == sum(
                map(len, received)
            ) + 3 * len(b'{"ok":true}')
        finally:
            server.shutdown()
            thread.join()
