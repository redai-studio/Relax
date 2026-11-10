# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP body accounting against real requests and server responses."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from task_delta_weight_sync.sync_metrics import record_http


def test_control_metrics_count_actual_http_bodies_once_per_engine_request():
    received = []
    reply = '{"success": true, "message": "权重"}'.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    counter = {}
    client = requests.Session()
    client.trust_env = False  # Loopback must not use a workspace HTTP proxy.
    try:
        for engine in range(4):
            response = client.post(
                f"http://127.0.0.1:{server.server_port}/update_weights_from_distributed",
                json={"engine": engine, "names": ["视觉参数"], "shapes": [[2, 4]], "flush_cache": False},
                timeout=3,
            )
            response.raise_for_status()
            record_http(counter, "update_weights_from_distributed", response)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    row = counter["update_weights_from_distributed"]
    assert len(received) == row["requests"] == 4  # Eight TP workers still receive only four HTTP requests.
    assert row["request_json_bytes"] == sum(len(body) for body in received)
    assert row["decoded_response_body_bytes"] == 4 * len(reply)
