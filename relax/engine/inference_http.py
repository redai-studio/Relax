# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP gateway implementation, testable without Ray or model dependencies."""

import asyncio
import copy
import json
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from relax.engine.inference import InferenceError, InferenceRouter, SnapshotVersion, prepare_payload


def forward_headers(headers: Any) -> dict[str, str]:
    excluded = {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
    excluded.update(value.strip().lower() for value in headers.get("connection", "").split(","))
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


class UpstreamStreamingResponse(StreamingResponse):
    def __init__(self, upstream: httpx.Response) -> None:
        self.upstream = upstream
        super().__init__(
            upstream.aiter_raw(), status_code=upstream.status_code, headers=forward_headers(upstream.headers)
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(self.upstream.aclose())


async def legacy_http(app: FastAPI, method: str, path: str, query: str, body: bytes) -> dict[str, Any]:
    """Reuse bound FastAPI handlers and validation over a serializable RPC."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://legacy") as client:
        response = await client.request(
            method, path + (f"?{query}" if query else ""), content=body, headers={"content-type": "application/json"}
        )
    return {"status": response.status_code, "headers": forward_headers(response.headers), "body": response.content}


class GatewayRuntime:
    def __init__(self, role: str, *, client: httpx.AsyncClient | None = None) -> None:
        if role not in ("rollout", "genrm", "teacher"):
            raise ValueError(f"Unknown inference role: {role}")
        self.role = role
        self.backend: Any = None
        self.sources: list[dict[str, Any]] = []
        self._generation = 0
        self._active = False
        self._version = SnapshotVersion()
        self._router = InferenceRouter()
        self._cache: dict[int, dict[str, Any]] = {}
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=10),
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048, keepalive_expiry=600),
        )

    def rebind(self, sources: list[dict[str, Any]], backend: Any = None) -> None:
        self._generation += 1
        self.sources = sources
        self.backend = backend
        self._cache = {}
        self._active = True

    def quiesce(self) -> None:
        self._active = False
        self._generation += 1

    async def aclose(self) -> None:
        await self._client.aclose()

    async def snapshot(self) -> dict[str, Any]:
        generation = self._generation
        sources = tuple(self.sources)

        async def fetch(source: dict[str, Any]) -> dict[str, Any]:
            return await asyncio.wait_for(source["manager"].get_inference_snapshot.remote(), timeout=2)

        replies = await asyncio.gather(*(fetch(source) for source in sources), return_exceptions=True)
        if generation != self._generation:
            raise InferenceError(503, "Inference bindings changed; retry discovery")
        models: dict[str, Any] = {}
        versions, aliases, routes = [], {}, {}
        for index, (source, reply) in enumerate(zip(sources, replies, strict=True)):
            unavailable = isinstance(reply, BaseException)
            if unavailable:
                reply = self._cache.get(
                    index,
                    {
                        "models": {
                            source.get("model", "default"): {
                                "state": "unavailable",
                                "engines": [],
                                "router_url": None,
                            }
                        },
                        "epoch": None,
                        "revision": 0,
                    },
                )
            else:
                previous = self._cache.get(index)
                if previous and reply["epoch"] == previous["epoch"] and reply["revision"] < previous["revision"]:
                    reply = previous
                self._cache[index] = reply
            versions.append((reply["epoch"], reply["revision"]))
            for original_name, original in reply["models"].items():
                name = source.get("model") or original_name
                info = copy.deepcopy(original)
                if unavailable or not self._active:
                    info["state"] = "unavailable"
                    for engine in info["engines"]:
                        engine.update(state="unavailable", direct_eligible=False)
                for engine in info["engines"]:
                    engine["engine_id"] = name + "/" + engine["engine_id"].rsplit("/", 1)[-1]
                if name in models:
                    raise InferenceError(503, f"Duplicate model binding: {name}")
                models[name] = info
                routes[name] = name
                backend_model = info.get("backend_model")
                if backend_model:
                    aliases.setdefault(backend_model, []).append(name)
        # Ambiguous backend aliases require an explicit logical model name.
        resolved_aliases = {alias: names[0] for alias, names in aliases.items() if len(names) == 1}
        default = next(iter(models)) if len(models) == 1 else None
        if self.role == "rollout" and models:
            default = next(iter(models))  # Existing RolloutServer default is the first model.
        elif "__default__" in models:
            default = "__default__"
        result = self._version.publish(models, (generation, versions))
        return {
            "role": self.role,
            "topology_epoch": result["epoch"],
            "topology_revision": result["revision"],
            "phase": None,
            "models": result["models"],
            "routing": {"default_model": default, "route_keys": routes, "aliases": resolved_aliases},
            "total_engines": sum(info.get("total_engines", len(info["engines"])) for info in models.values()),
        }

    async def handle(self, request: Request, path: str) -> Response:
        try:
            return await self._handle(request, "/" + path.lstrip("/"))
        except InferenceError as exc:
            return JSONResponse(
                {"detail": str(exc)},
                status_code=exc.status_code,
                headers={"Retry-After": "1"} if exc.status_code == 503 else None,
            )
        except httpx.TimeoutException:
            return JSONResponse({"detail": "Inference upstream timed out"}, status_code=504)
        except httpx.RequestError:
            return JSONResponse({"detail": "Inference upstream unavailable"}, status_code=502)

    async def _handle(self, request: Request, path: str) -> Response:
        if path in ("/engines", "/health", "/v1/models"):
            if request.method != "GET":
                return Response(status_code=405)
            snapshot = await self.snapshot()
            if path == "/health":
                if (
                    self._active
                    and self.role == "genrm"
                    and self.backend is not None
                    and any(info["state"] in ("ready", "unavailable") for info in snapshot["models"].values())
                ):
                    try:
                        result = await asyncio.wait_for(
                            self.backend.inference_legacy_http.remote("GET", "/health", "", b""), timeout=10
                        )
                        health = json.loads(result["body"])
                    except Exception:
                        health = {"status": "unhealthy", "service": self.role}
                    return JSONResponse({**health, "gateway_status": "healthy"})
                instances = {
                    name: {"status": "healthy" if info["state"] == "ready" else "unhealthy"}
                    for name, info in snapshot["models"].items()
                }
                healthy = bool(instances) and all(info["status"] == "healthy" for info in instances.values())
                return JSONResponse(
                    {
                        "status": "healthy" if healthy else "unhealthy",
                        "service": self.role,
                        "gateway_status": "healthy",
                        "instances": instances,
                    }
                )
            if path == "/v1/models":
                return JSONResponse(
                    {
                        "object": "list",
                        "data": [
                            {"id": name, "object": "model", "created": 0, "owned_by": self.role}
                            for name in dict.fromkeys([*snapshot["models"], *snapshot["routing"]["aliases"]])
                        ],
                    }
                )
            model_name = request.query_params.get("model_name")
            if model_name is not None:
                name = self._router.select_model(snapshot, model_name, None)
                snapshot["models"] = {name: snapshot["models"][name]}
                snapshot["total_engines"] = snapshot["models"][name].get(
                    "total_engines", len(snapshot["models"][name]["engines"])
                )
            return JSONResponse(snapshot)
        if path not in ("/generate", "/chat/completions", "/v1/chat/completions"):
            if self.backend is None:
                return Response(status_code=404)
            result = await self.backend.inference_legacy_http.remote(
                request.method, path, request.url.query, await request.body()
            )
            location = result["headers"].get("location")
            if location and urlsplit(location).netloc == "legacy":
                redirect = urlsplit(location)
                result["headers"]["location"] = f"/{self.role}{redirect.path}" + (
                    f"?{redirect.query}" if redirect.query else ""
                )
            return Response(result["body"], status_code=result["status"], headers=result["headers"])
        if request.method != "POST":
            return Response(status_code=405)
        if not self._active:
            raise InferenceError(503, "Inference gateway is not ready")
        try:
            payload = await request.json()
        except (ValueError, UnicodeError):
            raise InferenceError(400, "Invalid JSON request") from None
        if not isinstance(payload, dict):
            raise InferenceError(400, "Expected a JSON object")
        generation = self._generation
        snapshot = await self.snapshot()
        target = self._router.select(snapshot, payload.get("model"), payload.get("route_key"))
        legacy = self.role == "genrm" and path == "/generate" and "messages" in payload
        if legacy:
            if not isinstance(payload["messages"], list) or any(not isinstance(m, dict) for m in payload["messages"]):
                raise InferenceError(422, "messages must be a list of objects")
            if payload.get("sampling_params") is not None and not isinstance(payload["sampling_params"], dict):
                raise InferenceError(422, "sampling_params must be an object")
            if any(key in payload for key in ("input_ids", "text")) or payload.get("stream"):
                raise InferenceError(400, "Legacy GenRM messages cannot be mixed with raw or streaming generation")
            payload = await self.backend.prepare_inference_request.remote(
                target.model, payload["messages"], payload.get("sampling_params")
            )
            # Tokenization can overlap an offload; re-check availability afterwards.
            snapshot = await self.snapshot()
            target = self._router.select(snapshot, target.model)
        if generation != self._generation or not self._active:
            raise InferenceError(503, "Inference bindings changed")
        chat = path != "/generate"
        path = "/v1/chat/completions" if chat else "/generate"
        payload = prepare_payload(payload, target, chat=chat)
        upstream_request = self._client.build_request(
            "POST", target.base_url + path, json=payload, headers=forward_headers(request.headers)
        )
        upstream = await self._client.send(upstream_request, stream=True)
        if payload.get("stream") and upstream.is_success:
            return UpstreamStreamingResponse(upstream)
        try:
            body = await upstream.aread()
            headers = forward_headers(upstream.headers)
            headers.pop("content-encoding", None)  # aread() decoded the body.
            if legacy and upstream.is_success:
                return JSONResponse({"response": json.loads(body).get("text", "").strip()})
            return Response(body, status_code=upstream.status_code, headers=headers)
        finally:
            await upstream.aclose()
