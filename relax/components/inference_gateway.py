# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU control plane shared by the Rollout, GenRM and Teacher HTTP facades."""

import asyncio
import json
from collections import defaultdict
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from relax.inference.routing import RoutingError, select_endpoint, select_model


class InferenceGateway:
    """Per-role proxy with no GPU placement or implicit model activation."""

    def __init__(
        self,
        role: str,
        managers: dict[str, Any],
        *,
        tokenizers: dict | None = None,
        instance_specs: dict | None = None,
        model_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.role = role
        self.managers = managers
        self.tokenizers = tokenizers or {}
        self.instance_specs = instance_specs or {}
        self.model_aliases = model_aliases or {}
        self._ordinals: dict[str, int] = defaultdict(int)
        self._client = httpx.AsyncClient(
            timeout=1800, limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048)
        )

    async def discovery(self) -> dict:
        try:
            snapshots = await asyncio.gather(*[m.get_inference_snapshot.remote() for m in self.managers.values()])
        except Exception as exc:
            raise HTTPException(503, "Inference discovery unavailable", headers={"Retry-After": "1"}) from exc
        models = {}
        for key, snapshot in zip(self.managers, snapshots, strict=True):
            entries = snapshot["models"]
            if self.role != "rollout" and len(entries) == 1:
                models[key] = next(iter(entries.values()))
            else:
                models.update(entries)
        routes = {name: name for name in models}
        for name, info in models.items():
            for alias in info.get("model_aliases", ()):
                if isinstance(alias, str) and alias and alias not in routes:
                    routes[alias] = name
        for name, aliases in self.model_aliases.items():
            if name not in models:
                continue
            for alias in aliases:
                if isinstance(alias, str) and alias and alias not in routes:
                    routes[alias] = name
        return {
            "role": self.role,
            "topology_revision": sum(s["topology_revision"] for s in snapshots),
            "phase": snapshots[0]["phase"] if snapshots else "starting",
            "models": models,
            "default_model": next(iter(models)) if len(models) == 1 else ("default" if "default" in models else None),
            "routes": routes,
        }

    async def models(self) -> dict:
        snapshot = await self.discovery()
        model_ids = list(snapshot["models"])
        for alias in snapshot["routes"]:
            if alias not in snapshot["models"]:
                model_ids.append(alias)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "created": 0, "owned_by": "relax"} for name in model_ids],
        }

    async def health(self) -> dict:
        snapshot = await self.discovery()
        states = {name: info["state"] for name, info in snapshot["models"].items()}
        unhealthy_states = {"failed", "dead", "unavailable"}
        return {
            "status": "unhealthy"
            if not states or any(state in unhealthy_states for state in states.values())
            else "healthy",
            "role": self.role,
            "models": states,
        }

    async def forward(self, request: Request, path: str) -> Response:
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "Expected a JSON request object") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected a JSON request object")
        snapshot = await self.discovery()
        try:
            name = select_model(snapshot, payload.get("model"), payload.get("route_key"))
            base_url = select_endpoint(snapshot, name, self._ordinals[name])
        except RoutingError as exc:
            raise HTTPException(
                exc.status_code, str(exc), headers={"Retry-After": "1"} if exc.status_code == 503 else None
            ) from exc
        self._ordinals[name] += 1
        payload.pop("route_key", None)
        # Internal model IDs need not be SGLang's checkpoint/served_model_name.
        payload.pop("model", None)
        legacy = self.role == "genrm" and path == "generate" and "messages" in payload
        if legacy:
            spec = self.instance_specs[name]
            sampling = spec.get("sampling_config", {})
            input_ids = await asyncio.to_thread(
                self.tokenizers[name].apply_chat_template,
                payload["messages"],
                tokenize=True,
                add_generation_prompt=True,
                **(sampling.get("chat_template_kwargs") or {}),
            )
            if not isinstance(input_ids, list):
                input_ids = input_ids["input_ids"] if "input_ids" in input_ids else list(input_ids)
            defaults = {
                "temperature": sampling.get("temperature", 0.2),
                "top_p": sampling.get("top_p", 1.0),
                "top_k": sampling.get("top_k", -1),
                "max_new_tokens": sampling.get("max_response_len", 1024),
            }
            defaults.update(payload.get("sampling_params") or {})
            payload = {"input_ids": input_ids, "sampling_params": defaults}
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() in {"authorization", "content-type", "x-request-id"}
        }
        try:
            upstream = await self._client.send(
                self._client.build_request("POST", f"{base_url}/{path}", json=payload, headers=headers),
                stream=bool(payload.get("stream")),
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Inference upstream unavailable: {type(exc).__name__}") from exc
        if not payload.get("stream") or upstream.status_code >= 400:
            try:
                content = await upstream.aread()
                if legacy and upstream.is_success:
                    content = json.dumps({"response": upstream.json().get("text", "").strip()}).encode()
                return Response(
                    content,
                    status_code=upstream.status_code,
                    headers={k: v for k, v in upstream.headers.items() if k in {"retry-after", "x-request-id"}},
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            finally:
                await upstream.aclose()

        async def chunks():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(chunks(), status_code=upstream.status_code, media_type="text/event-stream")

    async def aclose(self) -> None:
        await self._client.aclose()


def deploy_teacher_gateway(managers: dict[str, Any], args: Any) -> Any:
    """Deploy the same gateway on CPU, outside every inference GPU PG."""
    from ray import serve

    from relax.core.node_group_affinity import with_control_plane_affinity

    app = FastAPI()

    @serve.deployment(ray_actor_options=with_control_plane_affinity(args, {"num_cpus": 1, "num_gpus": 0}))
    @serve.ingress(app)
    class TeacherGateway:
        def __init__(self, handles: dict[str, Any]) -> None:
            self.gateway = InferenceGateway("teacher", handles)

        @app.get("/engines")
        async def engines(self) -> dict:
            return await self.gateway.discovery()

        @app.get("/health")
        async def health(self) -> dict:
            return await self.gateway.health()

        @app.get("/v1/models")
        async def models(self) -> dict:
            return await self.gateway.models()

        @app.post("/generate")
        async def generate(self, request: Request) -> Response:
            return await self.gateway.forward(request, "generate")

        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat(self, request: Request) -> Response:
            return await self.gateway.forward(request, "v1/chat/completions")

    return serve.run(TeacherGateway.bind(managers), name="teacher", route_prefix="/teacher")
