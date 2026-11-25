# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise the built Rust router over HTTP; no model or GPU is required."""

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


pytest.importorskip("sglang_router")


def encode(values):
    return base64.b64encode(struct.pack(f"<{len(values)}i", *values)).decode()


def decode(value):
    data = base64.b64decode(value)
    return list(struct.unpack(f"<{len(data) // 4}i", data))


def response(prompt, completion, routes, *, index=None, finish="length"):
    value = {
        "text": "中文",
        "meta_info": {"prompt_tokens": prompt, "completion_tokens": completion, "finish_reason": {"type": finish}},
    }
    if routes is not None:
        value["meta_info"]["routed_experts"] = encode(routes)
    if index is not None:
        value["index"] = index
    return value


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        if getattr(self.server, "close_connections", False):
            self.send_header("Connection", "close")
            self.close_connection = True
        super().end_headers()

    def log_message(self, *_):
        pass

    def reply(self, body, status=200):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.reply(
            {
                "model_path": self.server.model_path,
                "is_generation": True,
                "dp_size": 1,
                "tp_size": 1,
                "enable_dp_attention": False,
                "disaggregation_mode": self.server.role,
                "max_total_num_tokens": 65536,
                "internal_states": [{}],
                "version": "0.5.21",
            }
        )

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/generate":
            if self.path == "/abort_request":
                with self.server.abort_condition:
                    self.server.abort_requests.append(request)
                    self.server.abort_condition.notify_all()
            self.reply({"success": True})
            return
        self.server.wire_attempts.append((self.client_address, request))
        previous = getattr(self, "last_generate_finished", None)
        if (
            self.server.stale_after_s is not None
            and previous is not None
            and time.monotonic() - previous >= self.server.stale_after_s
        ):
            # A controlled idle-close race: the old connection accepts the wire
            # request but resets before application dispatch. A new socket works.
            self.server.stale_resets.append(self.client_address)
            self.close_connection = True
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
            return
        self.server.received.append(request)
        if self.server.wait_for_abort:
            rids = request["rid"] if isinstance(request["rid"], list) else [request["rid"]]
            expected = [{"rid": rid, "abort_all": False} for rid in rids]
            with self.server.abort_condition:
                settled = self.server.abort_condition.wait_for(
                    lambda: all(control in self.server.abort_requests for control in expected), timeout=3
                )
            if not settled:
                self.reply({"error": {"message": "missing targeted sibling abort"}}, 400)
                return
        if self.server.response_gate is not None:
            self.server.response_gate.wait(timeout=8)
        if self.server.disconnect:
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if self.server.kv_error is not None:
            self.reply(self.server.kv_error(request), 500)
            self.server.kv_error_sent.set()
            return
        if self.server.response_status != 200:
            message = f"{self.server.role} rejected request"
            if self.server.response_status == 499:
                rid = request.get("rid", f"{self.server.role}-local-id")
                message = f"Request {rid[0] if isinstance(rid, list) else rid} was aborted"
            self.reply({"error": {"message": message}}, self.server.response_status)
            return
        body = self.server.result
        if isinstance(body, list) and body and isinstance(body[0], bytes):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in body:
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.reply(body)
        self.last_generate_finished = time.monotonic()


@pytest.fixture
def router(tmp_path, request):
    mode = getattr(request, "param", "pd")
    regular = mode == "regular"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "x": 1}, "unk_token": "[UNK]"},
            }
        )
    )
    workers = []
    threads = []
    for role in ("null", "null") if regular else ("prefill", "decode"):
        worker = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker.daemon_threads = True
        worker.close_connections = regular
        worker.role, worker.model_path, worker.received = role, str(tokenizer), []
        worker.response_status, worker.response_gate, worker.disconnect = 200, None, False
        worker.abort_condition, worker.abort_requests = threading.Condition(), []
        worker.wait_for_abort, worker.kv_error, worker.kv_error_sent = False, None, threading.Event()
        worker.stale_after_s, worker.wire_attempts, worker.stale_resets = None, [], []
        worker.result = response(3, 1, [1] * 12)
        worker.url = f"http://127.0.0.1:{worker.server_port}"
        thread = threading.Thread(target=worker.serve_forever, daemon=True)
        thread.start()
        workers.append(worker)
        threads.append(thread)
    with socket.socket() as reservation, socket.socket() as metrics_reservation:
        reservation.bind(("127.0.0.1", 0))
        metrics_reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        metrics_port = metrics_reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "router.log"
    for worker in workers:
        worker.router_log_path = log_path
    worker_args = [
        "--pd-disaggregation",
        "--prefill",
        workers[0].url,
        "--decode",
        workers[1].url,
        "--disable-circuit-breaker",
    ]
    if regular:
        worker_args = ["--worker-urls", workers[0].url, workers[1].url, "--cb-failure-threshold", "1"]
    elif mode != "retrying-pd":
        worker_args.append("--disable-retries")
    command = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(metrics_port),
        *worker_args,
        "--policy",
        "round_robin",
        "--worker-startup-timeout-secs",
        "8",
        "--worker-startup-check-interval",
        "1",
        "--request-timeout-secs",
        "10",
        "--disable-health-check",
    ]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_ENDPOINT": workers[0].url},
            stdin=subprocess.DEVNULL,
        )
        try:
            for _ in range(100):
                assert process.poll() is None, log_path.read_text()
                try:
                    with urlopen(url + "/workers", timeout=0.2) as request:
                        state = json.load(request)
                    if len(state.get("workers", [])) == 2 and all(w["is_healthy"] for w in state["workers"]):
                        break
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            else:
                pytest.fail(log_path.read_text())
            yield url, workers
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            for worker in workers:
                worker.shutdown()
                worker.server_close()
            for thread in threads:
                thread.join(timeout=2)


def generate(router, *, start=0, stream=False, batch=False, rid=None):
    url, _ = router
    payload = {
        "input_ids": [[1, 2, 3], [1, 2, 3, 4, 5]] if batch else [1, 2, 3],
        "sampling_params": {"max_new_tokens": 4},
        "return_routed_experts": True,
        "routed_experts_start_len": start,
        "stream": stream,
    }
    if rid is not None:
        payload["rid"] = rid
    return urlopen(
        Request(url + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}),
        timeout=12,
    )


@pytest.mark.parametrize("router", ["regular"], indirect=True)
@pytest.mark.parametrize("failure", ["connection-refused", "response-lost"])
def test_regular_router_only_retries_before_dispatch(router, failure):
    _, workers = router
    with generate(router, rid="warmup") as response:
        assert response.status == 200
    # Round robin selects the other worker next. All its earlier HTTP
    # connections are closed so shutting the listener gives a TCP refusal.
    target, healthy = sorted(workers, key=lambda worker: len(worker.received))
    assert len(target.received) == 0 and len(healthy.received) == 1
    if failure == "connection-refused":
        target.shutdown()
        target.server_close()
        with generate(router, rid="request") as response:
            assert response.status == 200
        assert [item["rid"] for item in healthy.received] == ["warmup", "request"]
        assert f"Failed to send typed request worker_url={target.url}" in target.router_log_path.read_text()
    else:
        target.disconnect = True
        with pytest.raises(HTTPError) as error:
            generate(router, rid="request")
        assert error.value.code in (500, 502, 503)
        assert b"no_available_workers" not in error.value.read()
        assert [item["rid"] for item in target.received] == ["request"]
        assert [item["rid"] for item in healthy.received] == ["warmup"]


def mooncake_error(request, stage, *, fault=None):
    rid, room = request["rid"], request["bootstrap_room"]
    if isinstance(rid, list):
        rid, room = rid[-1], room[-1]
    if fault == "other-rid":
        rid += "-other"
    if fault == "other-room":
        room += 1
    reason = "Aborted by AbortReq." if fault != "transport" else "connection lost"
    message = (
        f"Decode {stage} failed for request rank=0 decode_req.req.rid={rid!r} "
        f"decode_req.req.bootstrap_room={room} with exception KVTransferError(bootstrap_room={room}): {reason}"
    )
    return {"object": "error", "message": message, "type": "500", "param": None, "code": 500}


@pytest.mark.parametrize("router", ["retrying-pd"], indirect=True)
@pytest.mark.parametrize(
    "batch,stage,client_rids",
    [
        (False, "handshake", None),
        (True, "transfer", None),
        (True, None, None),
        pytest.param(False, "transfer", ("rid'with\\slash\t", "rid'\"with\\slash\n"), id="quoted-client-rids"),
    ],
)
def test_pd_abort_returns_worker_499(router, batch, stage, client_rids):
    _, (prefill, decode_worker) = router
    # A pre-dispatch P abort can catch D already waiting for KV. In the
    # reverse direction P may finish normally after receiving the abort.
    aborted, waiting = (prefill, decode_worker) if stage else (decode_worker, prefill)
    aborted.response_status = 499
    waiting.wait_for_abort = True
    if stage:
        waiting.kv_error = lambda request: mooncake_error(request, stage)
    seen = set()
    for client_rid in client_rids or (None, None):
        first_control = len(waiting.abort_requests)
        with pytest.raises(HTTPError) as error:
            generate(router, batch=batch, rid=client_rid)
        assert error.value.code == 499
        rid = prefill.received[-1]["rid"]
        assert rid == decode_worker.received[-1]["rid"]
        if client_rid is not None:
            assert rid == client_rid
        rids = rid if batch else [rid]
        assert len(rids) == (2 if batch else 1)
        assert all(isinstance(value, str) and value for value in rids)
        assert len(set(rids)) == len(rids) and seen.isdisjoint(rids)
        seen.update(rids)
        controls = waiting.abort_requests[first_control:]
        assert all(control["abort_all"] is False for control in controls)
        assert {control["rid"] for control in controls} == set(rids)
        assert json.load(error.value)["error"]["message"] == f"Request {rids[0]} was aborted"
    assert len(prefill.received) == len(decode_worker.received) == 2


@pytest.mark.parametrize("fault", ["other-rid", "other-room", "transport", "before-abort"])
def test_pd_prefill_abort_preserves_independent_decode_failure(router, fault):
    _, (prefill, decode_worker) = router
    prefill.response_status = 499
    if fault == "before-abort":
        # A local KV failure with the same message may predate the router's abort.
        prefill.response_gate = decode_worker.kv_error_sent
    else:
        decode_worker.wait_for_abort = True
    decode_worker.kv_error = lambda request: mooncake_error(request, "handshake", fault=fault)
    with pytest.raises(HTTPError) as error:
        generate(router)
    assert error.value.code == 500
    message = decode_worker.kv_error(decode_worker.received[0])["message"]
    assert message in json.load(error.value)["error"]["message"]


def test_pd_retires_idle_connections_before_backend_close(router):
    _, workers = router
    prefill, decode_worker = workers
    prefill.stale_after_s = 5
    prefill.result = response(3, 1, [1] * 12)
    decode_worker.result = response(3, 1, [0] * 12)
    with generate(router, rid="idle-first") as result:
        assert decode(json.load(result)["meta_info"]["routed_experts"]) == [1] * 12
    time.sleep(5.5)
    with generate(router, rid="idle-second") as result:
        assert decode(json.load(result)["meta_info"]["routed_experts"]) == [1] * 12
    assert prefill.stale_resets == []
    for worker in workers:
        assert len(worker.wire_attempts) == len(worker.received) == 2
        assert worker.wire_attempts[0][0] != worker.wire_attempts[1][0]
        assert [request["rid"] for request in worker.received] == ["idle-first", "idle-second"]
    rooms = [request["bootstrap_room"] for request in prefill.received]
    assert len(set(rooms)) == 2
    assert rooms == [request["bootstrap_room"] for request in decode_worker.received]


@pytest.mark.parametrize("completion,start,missing", [(4, 1, False), (1, 0, True), (0, 0, True)])
def test_pd_native_routes_cover_prompt_once(router, completion, start, missing):
    _, (prefill, worker) = router
    p = [10] * ((3 - start) * 4)
    d = [99] * (max(0, 3 + completion - 1 - start) * 4)
    if completion > 1:
        d[len(p) :] = [20] * ((completion - 1) * 4)
    prefill.result = response(3, 1, p)
    worker.result = response(3, completion, None if missing else d)
    with generate(router, start=start) as result:
        raw = result.read()
        assert int(result.headers["Content-Length"]) == len(raw)
        actual = decode(json.loads(raw)["meta_info"]["routed_experts"])
    expected = p[: max(0, 3 + completion - 1 - start) * 4] + ([20] * ((completion - 1) * 4) if completion > 1 else [])
    assert actual == expected
    assert prefill.received[-1]["return_routed_experts"] is True
    assert worker.received[-1]["routed_experts_start_len"] == start


@pytest.mark.parametrize("corruption", ["missing"])
def test_pd_router_rejects_incomplete_success_routes(router, corruption):
    _, (_, worker) = router
    worker.result = response(3, 3, [99] * 12 + [20] * 8)
    meta = worker.result["meta_info"]
    if corruption == "base64":
        meta["routed_experts"] = "!"
    elif corruption == "shape":
        meta["routed_experts"] = encode([1])
    elif corruption == "missing":
        del meta["routed_experts"]
    else:
        meta["prompt_tokens"] = 5
    with pytest.raises(HTTPError) as error:
        generate(router)
    assert error.value.code == 502
    assert "routing" in error.value.read().decode().lower()


@pytest.mark.parametrize("split", [True])
def test_pd_sse_handles_split_utf8_crlf_and_final_with_done(router, split):
    _, (prefill, worker) = router
    prefill.result = response(3, 1, [10] * 12)
    final = response(3, 3, [99] * 12 + [20] * 8)
    body = ("id: route\r\ndata: " + json.dumps(final, ensure_ascii=False) + "\r\n\r\ndata: [DONE]\r\n\r\n").encode()
    worker.result = [bytes([b]) for b in body] if split else [body]
    with generate(router, stream=True) as result:
        assert result.headers.get("Content-Length") is None
        raw = result.read().decode()
    assert "id: route" in raw and "中文" in raw and "[DONE]" in raw
    data = next(line[6:] for line in raw.splitlines() if line.startswith("data: {"))
    assert decode(json.loads(data)["meta_info"]["routed_experts"]) == [10] * 12 + [20] * 8
    assert prefill.received[-1]["stream"] is False
    assert worker.received[-1]["stream"] is True


@pytest.mark.parametrize("stream,prefill_status", [(True, 503)])
def test_pd_prefill_failure_is_not_hidden_by_decode_error(router, stream, prefill_status):
    _, (prefill, worker) = router
    prefill.disconnect = prefill_status is None
    prefill.response_status = prefill_status or 200
    worker.response_status, worker.response_gate = 500, threading.Event()
    outcome = {}

    def request():
        try:
            with generate(router, stream=stream) as result:
                outcome.update(code=result.status, body=result.read().decode())
        except HTTPError as error:
            outcome.update(code=error.code, body=error.read().decode())
        except Exception as error:
            outcome["exception"] = repr(error)

    client = threading.Thread(target=request, daemon=True)
    client.start()
    try:
        marker = "Prefill HTTP request failed" if prefill_status is None else "Prefill HTTP error response"
        deadline = time.monotonic() + 3
        while marker not in prefill.router_log_path.read_text() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker in prefill.router_log_path.read_text()
        assert client.is_alive(), "The pending D response must settle before the same rid can be retried"
    finally:
        worker.response_gate.set()
        client.join(timeout=5)
    assert not client.is_alive()
    assert outcome.get("code") == (prefill_status or 502), outcome
    body = outcome["body"].lower()
    assert "prefill" in body
    assert len(prefill.received) == len(worker.received) == 1
