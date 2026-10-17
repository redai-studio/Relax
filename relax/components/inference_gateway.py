# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from typing import Any

from fastapi import FastAPI, HTTPException, Request
from ray import serve
from starlette.responses import Response

from relax.inference.compat import RoleDiscovery
from relax.inference.gateway import InferenceGatewayHandler


app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 1}, max_ongoing_requests=1024)
@serve.ingress(app)
class InferenceGateway:
    def __init__(self, role: str, managers: dict[str, Any]) -> None:
        self._discovery = RoleDiscovery(role, managers)
        self._gateway = InferenceGatewayHandler(role, self._discovery.snapshot)

    @app.post("/generate")
    async def generate(self, request: Request) -> Response:
        return await self._gateway.handle_generate(request)

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat(self, request: Request) -> Response:
        return await self._gateway.handle_chat(request)

    @app.get("/v1/models")
    async def models(self) -> dict[str, Any]:
        return await self._gateway.models()

    @app.get("/engines")
    async def engines(self, schema: int = 2) -> dict[str, Any]:
        if schema != 2:
            raise HTTPException(status_code=400, detail="Unsupported discovery schema")
        return await self._gateway.discovery()

    @app.get("/health")
    async def health(self, schema: int = 2) -> dict[str, Any]:
        if schema != 2:
            raise HTTPException(status_code=400, detail="Unsupported health schema")
        return await self._gateway.health()

    async def __del__(self) -> None:
        if hasattr(self, "_gateway"):
            await self._gateway.aclose()
