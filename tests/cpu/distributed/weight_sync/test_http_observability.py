"""Control-plane CPU tests using a minimal SGLang SDK boundary."""

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI

from relax.distributed.weight_sync.sglang_http import install_http_extension


def test_status_responds_while_mutation_is_locked_and_reports_failure(monkeypatch):
    # The extension and admission lock are real; no model or CUDA worker runs.
    names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.http_server",
        "sglang.srt.managers",
        "sglang.srt.managers.io_struct",
        "sglang.srt.managers.tokenizer_manager",
        "sglang.srt.utils",
        "sglang.srt.utils.auth",
        "sglang.srt.utils.network",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
    app = FastAPI()
    modules["sglang.srt.entrypoints.http_server"].app = app
    io = modules["sglang.srt.managers.io_struct"]
    for name in ("GenerateReqInput", "RpcReqInput", "RpcReqOutput", "sock_recv", "sock_send"):
        setattr(io, name, object)
    modules["sglang.srt.managers.tokenizer_manager"].TokenizerManager = type(
        "TokenizerManager", (), {"init_ipc_channels": lambda *args: None}
    )
    auth = modules["sglang.srt.utils.auth"]
    auth.AuthLevel = SimpleNamespace(ADMIN_OPTIONAL=0)
    auth.auth_level = lambda level: lambda function: function
    modules["sglang.srt.utils.network"].get_zmq_socket = object
    install_http_extension()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/relax_weight_version")
    controls = app.user_middleware[0].kwargs
    gate, control = controls["gate"], controls["control"]

    async def invoke(body):
        return await endpoint(SimpleNamespace(json=AsyncMock(return_value=body)))

    async def run():
        identity = {"epoch": 1, "workers": {"0": "worker"}, "installation_id": "install", "root": "root"}
        await invoke({"action": "claim", "identity": identity, "timeout": 1800})
        failed = await invoke({"action": "open", "identity": identity})
        assert failed.status_code == 409
        assert "Commit" in json.loads(failed.body)["detail"]["error"]
        async with control:
            status = await asyncio.wait_for(invoke({"action": "status"}), 0.1)
        assert not status["generation_open"]
        assert status["unavailable"]["status"] == "WAITING_ACTION"
        assert "Commit" in status["unavailable"]["reason"]
        await invoke({"action": "claim", "identity": {**identity, "epoch": 2}})
        assert gate.failure is None
        assert not gate.opened

    asyncio.run(run())
