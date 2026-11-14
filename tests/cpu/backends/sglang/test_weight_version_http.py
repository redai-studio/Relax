import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from starlette.exceptions import HTTPException

from relax.distributed.weight_sync.admission import InstallationFence
from relax.distributed.weight_sync.sglang_http import install_http_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["busy_once", "busy_until_deadline", "wrong_identity"])
async def test_post_probe_flush_retries_only_busy_workers(monkeypatch, tmp_path, failure):
    import json

    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers import io_struct, tokenizer_manager
    from sglang.srt.utils import auth, network

    class Tokenizer:
        def init_ipc_channels(self, port_args):
            pass

    app = FastAPI()
    app.add_exception_handler(HTTPException, http_server.validation_exception_handler)
    manager = SimpleNamespace(_relax_rpc_endpoint="test")
    monkeypatch.setattr(http_server, "app", app)
    monkeypatch.setattr(http_server, "get_global_state", lambda: SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(tokenizer_manager, "TokenizerManager", Tokenizer)
    monkeypatch.setattr(auth, "auth_level", lambda level: lambda function: function)
    socket = SimpleNamespace(setsockopt=lambda *args: None, poll=lambda *args: True)
    monkeypatch.setattr(network, "get_zmq_socket", lambda *args: socket)
    attempts = []

    def send(socket, message):
        payload = json.loads(message.parameters["payload"])
        attempts.append(payload)
        success = failure == "busy_once" and len(attempts) > 1
        receipt = {"rank": 0, "request_id": payload["request_id"], "success": success}
        if not success:
            receipt.update(
                error_code="WORKER_ERROR" if failure == "wrong_identity" else "WORKER_BUSY",
                error="stale worker" if failure == "wrong_identity" else "scheduler retiring probe",
            )
        (tmp_path / "rank-0.json").write_text(json.dumps(receipt))

    monkeypatch.setattr(io_struct, "sock_send", send)
    monkeypatch.setattr(io_struct, "sock_recv", lambda socket: io_struct.RpcReqOutput(success=True, message=""))
    install_http_extension()
    identity = {"epoch": 1, "installation_id": "one", "root": "root", "workers": {"0": "worker"}}
    worker = {key: value for key, value in identity.items() if key != "workers"} | {"instance": "worker"}
    payload = {
        "request_id": "flush",
        "command_id": "flush",
        "phase": "flush",
        "receipts": str(tmp_path),
        "workers": {"0": worker},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/relax_weight_version", json={"action": "claim", "identity": identity})
        response = await client.post(
            "/relax_weight_version",
            json={"action": "rpc", "identity": identity, "payload": json.dumps(payload), "timeout": 0.1},
        )
        if failure == "busy_once":
            assert response.status_code == 200 and len(attempts) == 2
            assert response.json()["generation_open"] is False
        elif failure == "wrong_identity":
            assert response.status_code == 409 and len(attempts) == 1
            assert "stale worker" in response.json()["detail"]["error"]
        else:
            assert response.status_code == 409 and len(attempts) > 1
            assert "expired" in response.json()["detail"]["error"]
        assert (await client.post("/generate", json={"input_ids": [0]})).status_code == 503


@pytest.mark.asyncio
async def test_probe_timeout_keeps_http_generation_closed_until_restart(monkeypatch):
    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers import tokenizer_manager
    from sglang.srt.utils import auth

    class Tokenizer:
        def init_ipc_channels(self, port_args):
            pass

    completed = []

    async def generate(request):
        try:
            await asyncio.Event().wait()
            yield {"output_ids": [1]}
        finally:
            completed.append("cancelled")

    manager = SimpleNamespace(generate_request=generate, _update_weight_version_if_provided=lambda version: None)
    app = FastAPI()
    app.add_exception_handler(HTTPException, http_server.validation_exception_handler)
    monkeypatch.setattr(http_server, "app", app)
    monkeypatch.setattr(http_server, "get_global_state", lambda: SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(tokenizer_manager, "TokenizerManager", Tokenizer)
    monkeypatch.setattr(auth, "auth_level", lambda level: lambda function: function)
    fence = InstallationFence()
    monkeypatch.setattr("relax.distributed.weight_sync.sglang_http.InstallationFence", lambda: fence)
    install_http_extension()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        identity = {"epoch": 1, "installation_id": "install", "root": "root", "workers": {"0": "instance"}}
        close = await client.post("/relax_weight_version", json={"action": "claim", "identity": identity})
        assert close.status_code == 200
        fence.phase = "VERIFIED"  # Isolate timeout handling after completed device verification.
        probe = await client.post(
            "/relax_weight_version",
            json={"action": "probe", "identity": identity, "version": "policy:1:root", "timeout": 0.01},
        )
        assert probe.status_code == 409
        assert probe.json()["detail"]["action"] == "restart_engine_and_recover_checkpoint"
        assert completed == ["cancelled"]
        for action in ("open", "rpc", "probe"):
            response = await client.post("/relax_weight_version", json={"action": action, "identity": identity})
            assert response.status_code == 409
            assert response.json()["detail"]["action"] == "restart_engine_and_recover_checkpoint"
        response = await client.post("/generate", json={"input_ids": [0]})
        assert response.status_code == 503
        close = await client.post("/relax_weight_version", json={"action": "close", "identity": identity})
        assert close.status_code == 200


@pytest.mark.asyncio
async def test_http_old_commands_and_unfenced_offload_are_rejected(monkeypatch):
    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers import tokenizer_manager
    from sglang.srt.utils import auth

    class Tokenizer:
        def init_ipc_channels(self, port_args):
            pass

    app = FastAPI()
    app.add_exception_handler(HTTPException, http_server.validation_exception_handler)
    mutations = []

    @app.post("/release_memory_occupation")
    async def offload():
        mutations.append("offload")
        return {"success": True}

    monkeypatch.setattr(http_server, "app", app)
    monkeypatch.setattr(tokenizer_manager, "TokenizerManager", Tokenizer)
    monkeypatch.setattr(auth, "auth_level", lambda level: lambda function: function)
    install_http_extension()
    old = {"epoch": 1, "installation_id": "old", "root": "one", "workers": {"0": "worker"}}
    new = {**old, "epoch": 2, "installation_id": "new", "root": "two"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for identity in (old, new):
            response = await client.post("/relax_weight_version", json={"action": "claim", "identity": identity})
            assert response.status_code == 200
        for action in ("claim", "close", "open", "probe", "commit"):
            response = await client.post("/relax_weight_version", json={"action": action, "identity": old})
            assert response.status_code == 409
        response = await client.post("/relax_weight_version", json={"action": "open", "identity": new})
        assert response.status_code == 409
        import json

        for headers in ({}, {"X-Relax-Installation": json.dumps(old)}):
            response = await client.post("/release_memory_occupation", headers=headers)
            assert response.status_code == 409
        assert mutations == []
        response = await client.post("/release_memory_occupation", headers={"X-Relax-Installation": json.dumps(new)})
        assert response.status_code == 200
        assert mutations == ["offload"]
