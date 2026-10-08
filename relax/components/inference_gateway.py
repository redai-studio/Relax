# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Standalone inference gateway deployment.

Rollout and GenRM already run a CPU-only Serve replica and hold their
``InferenceGateway`` there. A role whose managers are created outside Serve has
no replica to put one in -- today that is the managed OPD teacher, started by
the Controller -- so this deployment gives it the same endpoints.

The replica requests no GPU. It keeps answering discovery and health while the
role's engines are asleep, draining or being rebuilt.
"""

from typing import Any, Optional

import ray
from fastapi import FastAPI, HTTPException, Request
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.engine.inference.discovery import RoleSnapshot, role_snapshot_from_payloads
from relax.engine.inference.gateway import InferenceGateway
from relax.utils.env import Envs


app = FastAPI()

# Must stay module-level: it feeds the @serve.deployment decorator, which runs at import.
INFERENCE_GATEWAY_SERVE_MAX_ONGOING_REQUESTS = Envs.INFERENCE_GATEWAY_SERVE_MAX_ONGOING_REQUESTS


@serve.deployment(
    max_ongoing_requests=INFERENCE_GATEWAY_SERVE_MAX_ONGOING_REQUESTS,
    logging_config=LoggingConfig(log_level="WARNING", enable_access_log=False),
)
@serve.ingress(app)
class InferenceGatewayService:
    """Discovery and request forwarding for one inference role."""

    def __init__(self, role: str, managers: dict[str, Any], default_model: Optional[str] = None) -> None:
        """
        Args:
            role: Role this deployment serves, e.g. ``"teacher"``.
            managers: ``{model name: manager actor handle}``. Each manager must
                expose ``get_inference_snapshot``.
            default_model: Model used by requests that name neither a model nor
                a route key. Not needed when the role serves a single model.
        """
        self.role = role
        self._managers = dict(managers)
        self._default_model = default_model
        self._gateway = InferenceGateway(role, self._fetch_inference_snapshot, upstream_name=f"{role} engine")

    def _fetch_inference_snapshot(self) -> RoleSnapshot:
        payloads = ray.get([manager.get_inference_snapshot.remote() for manager in self._managers.values()])
        return role_snapshot_from_payloads(
            self.role, dict(zip(self._managers, payloads, strict=True)), default_model=self._default_model
        )

    @app.get("/engines")
    async def get_engines(self) -> dict:
        return await self._gateway.engines()

    @app.get("/health")
    async def health(self) -> dict:
        return await self._gateway.health()

    @app.get("/v1/models")
    async def list_models(self) -> dict:
        return await self._gateway.models()

    @app.post("/generate")
    async def generate(self, request: Request):
        """Forward a native engine ``/generate`` payload to a ready replica."""
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid request body: expected a JSON object")
        return await self._gateway.generate(payload)

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(self, request: Request):
        return await self._gateway.chat_completions(await request.body(), dict(request.headers))
