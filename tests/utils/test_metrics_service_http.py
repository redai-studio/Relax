# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in CPU Serve smoke test, isolated from any existing Ray cluster.

Run with RUN_METRICS_SERVE_SMOKE=1 pytest
tests/utils/test_metrics_service_http.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


async def test_metrics_service_discovery_cancels_timed_out_serve_request(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    from relax.utils.metrics.service import MetricsService

    started = asyncio.Event()

    class PendingResponse:
        def __init__(self):
            self.cancel = Mock()

        def __await__(self):
            started.set()
            return asyncio.Event().wait().__await__()

    response = PendingResponse()
    handle = SimpleNamespace(get_engines=SimpleNamespace(remote=Mock(return_value=response)))
    monkeypatch.setattr("relax.utils.metrics.service.serve.get_app_handle", Mock(return_value=handle))
    instance = SimpleNamespace(_rollout_handle_task=None)
    discover = asyncio.create_task(MetricsService.func_or_class._discover_sglang_engines(instance))
    await asyncio.wait_for(started.wait(), 2)
    discover.cancel()
    with pytest.raises(asyncio.CancelledError):
        await discover
    response.cancel.assert_called_once()


@pytest.mark.skipif(os.environ.get("RUN_METRICS_SERVE_SMOKE") != "1", reason="Opt-in local CPU Ray Serve smoke test")
def test_metrics_service_http_scrapes_sglang_without_redirects():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--serve-smoke"],
        cwd=root,
        env={**os.environ, "RAY_ADDRESS": "local", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_serve_smoke() -> None:
    import socket
    import threading
    from argparse import Namespace
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx
    import ray
    from prometheus_client import CONTENT_TYPE_LATEST
    from prometheus_client.parser import text_string_to_metric_families
    from ray import serve

    from relax.utils.metrics.service import MetricsService

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/metrics"
            body = b'# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs{tp_rank="0"} 7\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    engine_url = f"http://127.0.0.1:{upstream.server_port}"

    @serve.deployment(ray_actor_options={"num_cpus": 0.1})
    class Rollout:
        async def get_engines(self) -> dict:
            return {
                "models": {
                    "default": {
                        "engine_groups": [
                            {"worker_type": "regular", "engines": [{"status": "active", "url": engine_url, "rank": 0}]}
                        ]
                    }
                },
                "observation": {"complete": True, "issues": []},
            }

    def metric_value(response: httpx.Response, name: str) -> float:
        return next(
            sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == name
        )

    # Select an unused local port; never connect to or shut down an existing cluster.
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    try:
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False)
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        serve.run(
            MetricsService.options(ray_actor_options={"num_cpus": 0.1}).bind(None, None, Namespace()),
            name="metrics",
            route_prefix="/metrics",
        )
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", follow_redirects=False, trust_env=False, timeout=15
        ) as client:
            # The service is deployed before Rollout during normal startup.
            response = client.get("/metrics")
            assert response.status_code == 200, response.text
            assert metric_value(response, "relax_sglang_discovery_success") == 0

            serve.run(Rollout.bind(), name="rollout", route_prefix=None)
            for path in ("/metrics", "/metrics/"):
                response = client.get(path)
                assert response.status_code == 200, response.text
                assert "location" not in response.headers
                assert response.headers["content-type"] == CONTENT_TYPE_LATEST
                assert metric_value(response, "sglang:num_running_reqs") == 7
                assert metric_value(response, "relax_sglang_scrape_success") == 1
                assert f'relax_engine="{engine_url}"' in response.text

            assert client.get("/metrics/health").json()["status"] == "healthy"
            response = client.post("/metrics/log_metric", json={"step": 1, "metric_name": "loss", "metric_value": 0.5})
            assert response.json()["status"] == "success"
            assert client.get("/metrics/unknown").status_code == 404
    finally:
        if ray.is_initialized():
            serve.shutdown()
            ray.shutdown()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


if __name__ == "__main__" and "--serve-smoke" in sys.argv:
    _run_serve_smoke()
