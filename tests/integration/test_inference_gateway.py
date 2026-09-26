# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU Ray Serve wiring with a small HTTP engine, no training
dependencies."""

import socket
from types import SimpleNamespace

import httpx
import pytest


ray = pytest.importorskip("ray")
serve = pytest.importorskip("ray.serve")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from relax.core.inference import bind_inference_gateway, deploy_inference_gateway  # noqa: E402
from relax.engine.inference import SnapshotVersion  # noqa: E402
from relax.engine.inference_http import legacy_http  # noqa: E402


engine_app = FastAPI()


@serve.deployment(ray_actor_options={"num_cpus": 0})
@serve.ingress(engine_app)
class FakeEngine:
    @engine_app.post("/generate")
    @engine_app.post("/v1/chat/completions")
    async def generate(self, request: Request):
        payload = await request.json()
        if payload.get("stream"):

            async def chunks():
                yield b'data: {"text":"answer"}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return {"text": " answer ", "echo": payload, "extension": True}


@ray.remote(num_cpus=0)
class FakeManager:
    def __init__(self, url: str) -> None:
        self.url = url
        self.state = "ready"
        self.version = SnapshotVersion()

    def set_state(self, state: str) -> None:
        self.state = state

    def get_inference_snapshot(self) -> dict:
        return self.version.publish(
            {
                "default": {
                    "state": self.state,
                    "router_url": None,
                    "backend_model": "backend",
                    "engines": [
                        {
                            "engine_id": "default/0",
                            "base_url": self.url,
                            "state": self.state,
                            "direct_eligible": self.state == "ready",
                        }
                    ],
                    "engine_groups": [{"engines": [{"rank": 0, "status": "active", "url": self.url}]}],
                    "total_engines": 1,
                }
            }
        )


backend_app = FastAPI()


@serve.deployment(ray_actor_options={"num_cpus": 0})
@serve.ingress(backend_app)
class FakeBackend:
    def __init__(self, manager) -> None:
        self.manager = manager

    def run(self) -> str:
        return "original backend"

    def get_inference_bindings(self) -> list:
        return [{"manager": self.manager}]

    async def inference_legacy_http(self, method: str, path: str, query: str, body: bytes) -> dict:
        return await legacy_http(self.app, method, path, query, body)

    async def prepare_inference_request(self, key: str, messages: list, sampling_params: dict | None) -> dict:
        return {"input_ids": [1, 2], "sampling_params": sampling_params or {}}

    @backend_app.get("/get_step")
    def get_step(self) -> dict:
        return {"step": 7}

    @backend_app.post("/set_step")
    def set_step(self, step: int) -> dict:
        return {"step": step}

    @backend_app.get("/health")
    def health(self) -> dict:
        return {"status": "healthy", "service": "genrm"}


@pytest.fixture(scope="module")
def local_serve():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
    )
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        serve.run(FakeEngine.bind(), name="test_engine", route_prefix="/engine")
        yield f"http://127.0.0.1:{port}"
    finally:
        serve.shutdown()
        ray.shutdown()


def test_inference_gateway_three_roles_preserve_backend_handles_and_http(local_serve):
    gateways = {}
    managers = {}
    config = SimpleNamespace(enable_affinity=False)
    with httpx.Client(timeout=30) as client:
        for role in ("rollout", "genrm", "teacher"):
            manager = FakeManager.remote(local_serve + "/engine")
            managers[role] = manager
            backend = serve.run(FakeBackend.bind(manager), name=role, route_prefix=None)
            assert backend.run.remote().result() == "original backend"
            gateway = deploy_inference_gateway(role, config)
            gateways[role] = gateway
            bind_inference_gateway(gateway, backend.get_inference_bindings.remote().result(), backend)
            response = client.post(f"{local_serve}/{role}/generate", json={"input_ids": [1]})
            assert response.status_code == 200, response.text
            assert response.json()["extension"]
            chat = client.post(f"{local_serve}/{role}/chat/completions", json={"model": "default", "messages": []})
            assert chat.json()["echo"]["model"] == "backend"
            with client.stream("POST", f"{local_serve}/{role}/generate", json={"input_ids": [1], "stream": True}) as s:
                assert s.status_code == 200
                assert list(s.iter_lines()) == ['data: {"text":"answer"}', "", "data: [DONE]", ""]
            assert client.get(f"{local_serve}/{role}/get_step").json() == {"step": 7}
            redirect = client.get(f"{local_serve}/{role}/get_step/")
            assert redirect.headers["location"] == f"/{role}/get_step"
            assert client.post(f"{local_serve}/{role}/set_step", params={"step": "bad"}).status_code == 422
        legacy = client.post(local_serve + "/genrm/generate", json={"messages": [{"role": "user", "content": "hi"}]})
        assert legacy.json() == {"response": "answer"}
        assert client.get(local_serve + "/genrm/health").json() == {
            "status": "healthy",
            "service": "genrm",
            "gateway_status": "healthy",
        }
        discovery = client.get(local_serve + "/rollout/engines").json()
        assert discovery["models"]["default"]["engine_groups"][0]["engines"][0]["rank"] == 0
        assert len(discovery["models"]["default"]["engines"]) == 1
        ray.get(managers["teacher"].set_state.remote("sleeping"))
        assert client.get(local_serve + "/teacher/health").json()["gateway_status"] == "healthy"
        assert client.post(local_serve + "/teacher/generate", json={"input_ids": [1]}).status_code == 503
        # Rebinding backend sources does not replace the ingress application.
        gateways["rollout"].quiesce.remote().result()
        assert client.post(local_serve + "/rollout/generate", json={"input_ids": [1]}).status_code == 503
        replacement = FakeManager.remote(local_serve + "/engine")
        bind_inference_gateway(gateways["rollout"], [{"manager": replacement}])
        assert client.post(local_serve + "/rollout/generate", json={"input_ids": [1]}).status_code == 200
    config = serve.status()
    assert all(config.applications[f"inference_{role}"].status == "RUNNING" for role in gateways)
