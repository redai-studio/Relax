# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP body accounting against real requests and server responses."""

import copy
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
import requests
import torch
from torch import nn

from task_delta_weight_sync.sync_metrics import (
    optimizer_boundary,
    parameter_sizes,
    record_http,
    validate_stream_receipts,
)


def test_optimizer_host_boundary_keeps_common_actor_cost_and_initialization_explicit():
    assert optimizer_boundary(SimpleNamespace(), 12.0) is None
    args = SimpleNamespace(_dws_optimizer_result=dict(observed_host_monotonic_s=10.0, updated=True))
    result = optimizer_boundary(args, 12.0, 20.0)
    assert result["actor_pre_updater_s"] == 2.0 and result["host_result_to_completion_s"] == 10.0
    with pytest.raises(ValueError, match="out of order"):
        optimizer_boundary(args, 9.0, 20.0)


def test_observed_stream_schemas_require_every_owner_member_version_and_nonce():
    workers = [dict(member=f"engine-{engine}:tp:{tp}", tp_rank=tp) for engine in range(4) for tp in range(2)]
    sources = [dict(streams={}) for _ in range(8)]
    for owner, source in enumerate(sources):
        names = [f"stream:expert:{owner}:tp:{tp}" for tp in range(2)]
        if owner == 0:
            names.append("stream:nonexpert")
        for stream in names:
            source["streams"][stream] = dict(
                schema="a" * 64,
                version=2,
                base=1,
                group_update="nonce",
                members=[
                    w["member"]
                    for w in workers
                    if stream.endswith(":nonexpert") or stream.endswith(f":tp:{w['tp_rank']}")
                ],
            )
    for worker in workers:
        worker["last_tcp_receiver_stages"] = dict(
            stream="stream",
            version=2,
            group_update="nonce",
            schemas={
                name: receipt["schema"]
                for s in sources
                for name, receipt in s["streams"].items()
                if worker["member"] in receipt["members"]
            },
        )
    arm = dict(stream="stream", version=2, source_ranks=sources)
    assert validate_stream_receipts(arm, workers)["status"] == "VERIFIED"
    with pytest.raises(ValueError, match="coverage"):
        validate_stream_receipts(arm, workers[:-1])
    for field, value in [("version", 1), ("group_update", "stale"), ("schemas", {})]:
        changed = copy.deepcopy(workers)
        changed[0]["last_tcp_receiver_stages"][field] = value
        with pytest.raises(ValueError, match="differs"):
            validate_stream_receipts(arm, changed)
    changed = copy.deepcopy(arm)
    changed["source_ranks"][1]["streams"]["stream:expert:1:tp:0"]["members"].pop()
    with pytest.raises(ValueError, match="member set"):
        validate_stream_receipts(changed, workers)


def test_native_parameter_denominator_preserves_fp32_vision_and_tied_objects():
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(3, dtype=torch.bfloat16))
    model.visual = nn.Linear(2, 1, bias=False, dtype=torch.float32)
    model.tied = model.weight
    sizes = parameter_sizes(model)
    assert sizes["named_parameters"] == 2 and sizes["named_parameter_bytes"] == 14
    assert sizes["parameter_groups"]["vision:torch.float32"] == dict(parameters=1, elements=2, bytes=8)


def test_control_metrics_count_actual_http_bodies_once_per_engine_request(monkeypatch):
    from relax.backends.sglang.sglang_engine import SGLangEngine

    received = []
    reply = '{"success": true, "message": "权重"}'.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.reply()

        def do_GET(self):
            self.reply()

        def reply(self):
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
    engine = object.__new__(SGLangEngine)
    engine.node_rank, engine.server_host, engine.server_port = 0, "127.0.0.1", server.server_port
    engine._dws_http_metrics = counter
    monkeypatch.setattr(requests, "post", client.post)
    try:
        for index in range(4):
            assert engine._make_request(
                "update_weights_from_distributed",
                {"engine": index, "names": ["视觉参数"], "shapes": [[2, 4]], "flush_cache": False},
                timeout=3,
            )["success"]
        assert isinstance(engine.pause_generation(timeout=3), requests.Response)
        assert isinstance(engine.continue_generation(timeout=3), requests.Response)
        response = client.get(f"http://127.0.0.1:{server.server_port}/server_info", timeout=3)
        record_http(counter, "server_info", response)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    row = counter["update_weights_from_distributed"]
    bodies = [body for path, body in received if path == "/update_weights_from_distributed"]
    assert len(bodies) == row["requests"] == 4  # Eight TP workers still receive only four HTTP requests.
    assert row["request_json_bytes"] == sum(len(body) for body in bodies)
    assert row["decoded_response_body_bytes"] == 4 * len(reply)
    for endpoint in ("pause_generation", "continue_generation"):
        assert counter[endpoint]["requests"] == 1 and counter[endpoint]["request_json_bytes"] == 2
    assert counter["server_info"]["request_json_bytes"] == 0
