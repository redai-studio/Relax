# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
import json

import httpx
import pytest

from relax.inference.admission import (
    AdmissionClosed,
    AdmissionGate,
    AdmissionServer,
    _AdmissionStream,
    create_admission_app,
)
from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


def test_closed_gate_refuses_new_work_and_requires_backend_ack_after_http_finish():
    gate = AdmissionGate()
    with pytest.raises(AdmissionClosed):
        gate.enter()
    gate.open()
    token = gate.enter()
    gate.close()
    gate.finish(token, completed=False)

    with pytest.raises(TimeoutError):
        gate.wait_drained(0.001)
    with pytest.raises(RuntimeError):
        gate.open()
    gate.confirm_backend_drained()
    gate.wait_drained(0.001)
    gate.open()
    gate.open()
    assert gate.status()["ready"]


def test_close_cancels_http_tasks_but_does_not_claim_gpu_idle():
    gate = AdmissionGate()
    gate.open()
    cancelled = []
    token = gate.enter(lambda: cancelled.append(True))

    gate.close()

    assert cancelled == [True]
    assert gate.status() == {"ready": False, "inflight": 1, "backend_drained": False}
    gate.finish(token, completed=True)
    assert not gate.status()["backend_drained"]


async def test_proxy_refuses_sleeping_generate_but_models_remain_available():
    calls = []
    gate = AdmissionGate()

    async def backend(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            response = await client.post("/generate", json={"input_ids": [1]})
            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
            assert (await client.get("/health_generate")).status_code == 503
            assert (await client.get("/v1/models")).status_code == 200
    assert calls == ["/v1/models"]


@pytest.mark.parametrize("path", ["/update_weights_from_tensor", "/resume_memory_occupation", "/pause_generation"])
async def test_public_guard_does_not_expose_backend_management(path):
    gate = AdmissionGate()
    gate.open()
    calls = []

    async def backend(request):
        calls.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            assert (await client.post(path, json={})).status_code == 404
    assert calls == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/abort_request", {"rid": "request-a"}),
        ("/abort_request", {"abort_all": True}),
        ("/close_session", {"session_id": "session-a"}),
    ],
)
async def test_cancellation_remains_available_when_admission_is_closed(path, payload):
    gate = AdmissionGate()
    calls = []

    async def backend(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"success": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            assert (await client.post(path, json=payload)).status_code == 200
            assert (await client.post("/update_weights_from_tensor", json={})).status_code == 404
    assert calls == [(path, payload)]
    assert gate.status() == {"ready": False, "inflight": 0, "backend_drained": True}


async def test_raw_request_preserves_token_multimodal_and_affinity_payload():
    gate = AdmissionGate()
    gate.open()
    payload = {
        "input_ids": [1, 2],
        "image_data": ["data:image/png;base64,AA=="],
        "return_logprob": True,
        "rid": "request-a",
    }
    captured = []

    async def backend(request):
        captured.append(request)
        return httpx.Response(200, json={"text": "result"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            response = await client.post("/generate", json=payload, headers={"X-SMG-Routing-Key": "session-a"})
    assert response.json() == {"text": "result"}
    assert json.loads(captured[0].content) == payload
    assert captured[0].headers["x-smg-routing-key"] == "session-a"
    assert gate.status()["inflight"] == 0


async def test_close_cancels_admitted_request_and_blocks_following_request():
    gate = AdmissionGate()
    gate.open()
    started = asyncio.Event()

    async def backend(request):
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            pending = asyncio.create_task(client.post("/generate", json={"input_ids": [1]}))
            await started.wait()
            gate.close()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert (await client.post("/generate", json={"input_ids": [2]})).status_code == 503
    assert gate.status()["inflight"] == 0
    assert not gate.status()["backend_drained"]


async def test_stream_disconnect_releases_http_count_without_claiming_gpu_drain():
    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"data: partial\n\n"
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    gate = AdmissionGate()
    gate.open()
    token = gate.enter()
    stream = Stream()
    response = _AdmissionStream(httpx.Response(200, stream=stream), gate, token)
    iterator = response.body_iterator
    assert await anext(iterator) == b"data: partial\n\n"

    await response.aclose()
    gate.close()

    assert stream.closed
    assert gate.status()["inflight"] == 0
    with pytest.raises(TimeoutError):
        gate.wait_drained(0.001)
    gate.confirm_backend_drained()
    gate.wait_drained(0.001)


def test_owned_guard_server_can_start_on_dynamic_port_and_stop_without_gpu():
    server = AdmissionServer("http://backend.example", "127.0.0.1", 0, timeout=3.0)
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.get(f"http://127.0.0.1:{server.port}/health_generate")
            assert response.status_code == 503
            server.gate.open()
            assert client.get(f"http://127.0.0.1:{server.port}/health_generate").status_code == 200
    finally:
        server.stop(timeout=3.0)
    assert not server._thread.is_alive()


@pytest.mark.parametrize("path", ["/server_info", "/get_server_info", "/model_info", "/get_model_info"])
async def test_router_metadata_available_while_sleeping_without_private_configuration(path):
    gate = AdmissionGate()
    calls = []

    async def backend(request):
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "served_model_name": "policy",
                "model_path": "policy",
                "dp_size": 2,
                "tp_size": 4,
                "architectures": ["TestModel"],
                "tokenizer_path": "policy",
                "is_generation": True,
                "host": "127.0.0.1",
                "port": 19001,
                "api_key": "test-only",
                "env": {"SECRET": "hidden"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            response = await client.get(path)
    assert response.status_code == 200
    assert not {"host", "port", "api_key", "env"}.intersection(response.json())
    if "server" in path:
        assert response.json()["dp_size"] == 2
        assert response.json()["served_model_name"] == "policy"
    else:
        assert response.json()["architectures"] == ["TestModel"]
    assert calls == ["/server_info" if "server" in path else "/model_info"]
    assert gate.status()["inflight"] == 0


async def test_router_bootstrap_liveness_does_not_require_open_generation():
    gate = AdmissionGate()
    calls = []

    async def backend(request):
        calls.append(request.url.path)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        app = create_admission_app("http://backend.example", gate, http_client=upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://public.example"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/health_generate")).status_code == 503
    assert calls == ["/health"]


def test_engine_router_publishes_guard_but_management_uses_private_backend(sglang_engine_module, monkeypatch):
    from types import SimpleNamespace

    module = sglang_engine_module
    engine = module.SGLangEngine.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host, engine.server_port = "127.0.0.1", 32101
    engine._strict_admission = True
    engine._ingress_host, engine._ingress_port = "192.0.2.1", 32102
    engine.router_ip, engine.router_port = "router.example", 32103
    engine.worker_type = "regular"
    engine.args = SimpleNamespace(use_slime_router=False)
    captured = []

    class Response:
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"worker_id": "guard-worker"}

    def post(url, **kwargs):
        captured.append((url, kwargs))
        return Response()

    monkeypatch.setattr(module.requests, "post", post)

    assert engine.get_url() == "http://192.0.2.1:32102"
    assert engine.register_to_router()
    engine._make_request("flush_cache", timeout=1)

    assert captured[0][1]["json"]["url"] == engine.get_url()
    assert captured[1][0] == "http://127.0.0.1:32101/flush_cache"


def test_engine_management_accepts_successful_empty_response(sglang_engine_module, monkeypatch):
    module = sglang_engine_module
    engine = module.SGLangEngine.__new__(module.SGLangEngine)
    engine.engine_spec = None
    engine.node_rank = 0
    engine.server_host, engine.server_port = "127.0.0.1", 32101

    response = module.requests.Response()
    response.status_code = 200
    response._content = b""
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: response)

    assert engine._make_request("abort_request", {"abort_all": True}, timeout=1) is None


def test_engine_management_rejects_nonempty_invalid_json(sglang_engine_module, monkeypatch):
    module = sglang_engine_module
    engine = module.SGLangEngine.__new__(module.SGLangEngine)
    engine.engine_spec = None
    engine.node_rank = 0
    engine.server_host, engine.server_port = "127.0.0.1", 32101

    response = module.requests.Response()
    response.status_code = 200
    response._content = b"not-json"
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(module.requests.exceptions.JSONDecodeError):
        engine._make_request("abort_request", {"abort_all": True}, timeout=1)


def test_dynamic_guard_waits_for_weight_continue_and_full_resident_tags(sglang_engine_module, monkeypatch):
    from types import SimpleNamespace

    module = sglang_engine_module
    engine = module.SGLangEngine.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host, engine.server_port = "127.0.0.1", 32101
    engine._strict_admission = True
    engine._admission_server = SimpleNamespace(gate=AdmissionGate())
    engine._resident_tags = {"weights"}
    engine._resume_generation_requested = False
    monkeypatch.setattr(engine, "_make_request", lambda *args, **kwargs: {})
    monkeypatch.setattr(engine, "flush_cache", lambda **kwargs: None)
    monkeypatch.setattr(
        module.requests, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None)
    )

    engine.continue_generation(timeout=1)
    assert not engine._admission_server.gate.status()["ready"]
    engine.resume_memory_occupation(tags=["kv_cache", "cuda_graph"])
    assert engine._admission_server.gate.status()["ready"]


def test_defer_backup_overrides_parser_default_but_rejects_explicit_group_false(sglang_engine_module, monkeypatch):
    import dataclasses
    from types import SimpleNamespace

    module = sglang_engine_module
    names = (
        "model_path trust_remote_code random_seed enable_memory_saver host port nccl_port nnodes node_rank "
        "dist_init_addr gpu_id_step base_gpu_id tp_size dp_size pp_size ep_size skip_server_warmup "
        "enable_draft_weights_cpu_backup enable_metrics enable_weights_cpu_backup"
    ).split()
    server_args = dataclasses.make_dataclass("ServerArgs", [(name, object, None) for name in names])
    monkeypatch.setattr(module, "ServerArgs", server_args)
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda value: value)
    monkeypatch.setattr(module, "_enable_draft_weights_cpu_backup", lambda *args: False)
    monkeypatch.setattr(module, "is_lora_enabled", lambda args: False)
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=1,
        hf_checkpoint="policy",
        seed=1,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_enable_weights_cpu_backup=False,
        _inference_preserve_rollout_weights=True,
    )
    kwargs, _ = module._compute_server_args(args, 0, "worker.example:1", 2, "worker.example", 3, base_gpu_id=0)
    assert kwargs["enable_weights_cpu_backup"] is True
    with pytest.raises(ValueError, match="CPU weight backup"):
        module._compute_server_args(
            args,
            0,
            "worker.example:1",
            2,
            "worker.example",
            3,
            base_gpu_id=0,
            sglang_overrides={"enable_weights_cpu_backup": False},
        )


def test_guarded_offload_aborts_and_flushes_before_releasing(sglang_engine_module, monkeypatch):
    from types import SimpleNamespace

    module = sglang_engine_module
    engine = module.SGLangEngine.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine._admission_server = SimpleNamespace(gate=AdmissionGate())
    engine.args = SimpleNamespace()
    engine._admission_server.gate.open()
    engine._resident_tags = {"weights", "kv_cache", "cuda_graph"}
    events = []
    monkeypatch.setattr(engine, "pause_generation", lambda **kwargs: events.append("pause"))
    monkeypatch.setattr(engine, "_make_request", lambda operation, *args, **kwargs: events.append(operation))
    monkeypatch.setattr(engine, "flush_cache", lambda **kwargs: events.append("flush"))

    engine.release_memory_occupation()

    assert events == ["pause", "abort_request", "flush", "release_memory_occupation"]
    assert engine._resident_tags == set()
    assert not engine._admission_server.gate.status()["ready"]
