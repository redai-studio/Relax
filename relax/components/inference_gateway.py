# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only inference ingress shared by all inference roles."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response
from ray import serve

from relax.engine.inference_http import GatewayRuntime


app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 0}, max_ongoing_requests=1024)
@serve.ingress(app)
class InferenceGateway:
    def __init__(self, role: str) -> None:
        self.runtime = GatewayRuntime(role)

    async def rebind_sources(self, sources: list[dict[str, Any]], backend: Any = None) -> None:
        self.runtime.rebind(sources, backend)

    async def quiesce(self) -> None:
        self.runtime.quiesce()

    async def __del__(self) -> None:
        await self.runtime.aclose()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def dispatch(self, request: Request, path: str) -> Response:
        return await self.runtime.handle(request, path)
