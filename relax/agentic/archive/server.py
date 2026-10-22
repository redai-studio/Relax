# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Standalone single-writer archive endpoint; never starts with ordinary
training."""

from __future__ import annotations

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from relax.utils.logging_utils import get_logger

from .schema import encode
from .store import ArchiveStore


logger = get_logger(__name__)


def make_server(store: ArchiveStore, host: str, port: int, token: str = "") -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return  # Do not log payloads or authentication headers.

        def do_POST(self) -> None:
            self.connection.settimeout(10)
            try:
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                    raise ValueError("unauthorized")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= min(store.max_bytes * 2, 64 * 1024 * 1024):
                    raise ValueError("request exceeds archive ingress limit")
                payload = json.loads(self.rfile.read(length))
                identity = payload.get("trajectory_id", "")
                if self.path == "/begin":
                    result = {"trajectory_id": store.begin(payload["metadata"])}
                elif self.path == "/event":
                    store.append(identity, payload["event"])
                    result = {}
                elif self.path == "/finish":
                    result = store.finish(identity, payload["outcome"])
                elif self.path == "/abort":
                    store.abort(identity)
                    result = {}
                else:
                    raise ValueError("unknown operation")
                result = {"ok": True, **result}
            except Exception as exc:
                result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            body = encode(result)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # Client left; its staged bytes still count against the budget.

    return HTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--per-task-limit", default=5, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    token = os.environ.get("CUA_ARCHIVE_TOKEN", "")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("CUA_ARCHIVE_TOKEN is required for a non-loopback listener")
    store = ArchiveStore(args.archive_dir, args.max_bytes, args.per_task_limit)
    try:
        logger.info("Archive opened; existing staging remains charged until explicit offline recovery")
        server = make_server(store, args.host, args.port, token)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        store.close()


if __name__ == "__main__":
    main()
