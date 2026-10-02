# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from time import perf_counter

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app.search_config import RetrieverSearchConfig, SearchError  # noqa: E402
from app.search_http import request_search  # noqa: E402


@pytest.mark.parametrize("stage", ["headers", "body"])
def test_local_http_enforces_read_timeout(stage: str) -> None:
    started = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if stage == "body":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"[")
                self.wfile.flush()
            started.set()
            release.wait(2.0)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        config = RetrieverSearchConfig(
            backend="retriever",
            endpoint=f"http://127.0.0.1:{server.server_port}",
            timeout_s=0.1,
            max_retries=0,
        )
        requested = perf_counter()
        with pytest.raises(SearchError, match="^ReadTimeout$"):
            request_search(config, method="GET", parse_results=lambda payload: [])
        assert started.is_set()
        assert perf_counter() - requested >= config.timeout_s
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3.0)
        assert not worker.is_alive()
