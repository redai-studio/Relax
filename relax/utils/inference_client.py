# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Async inference client for raw generation and OpenAI chat requests."""

from collections import defaultdict
from collections.abc import AsyncIterator

import httpx

from relax.inference.routing import select_endpoint, select_model


class InferenceClient:
    def __init__(self, service_url: str, *, direct: bool = False, timeout: float = 1800.0) -> None:
        self.service_url = service_url.rstrip("/")
        self.direct = direct
        self._client = httpx.AsyncClient(timeout=timeout)
        self._ordinals: dict[str, int] = defaultdict(int)

    async def discovery(self) -> dict:
        response = await self._client.get(f"{self.service_url}/engines")
        response.raise_for_status()
        return response.json()

    async def _route(self, payload: dict, model: str | None, route_key: str | None) -> tuple[str, dict]:
        snapshot = await self.discovery()
        name = select_model(
            snapshot,
            model if model is not None else payload.get("model"),
            route_key if route_key is not None else payload.get("route_key"),
        )
        base_url = select_endpoint(snapshot, name, self._ordinals[name])
        self._ordinals[name] += 1
        body = dict(payload)
        body.pop("route_key", None)
        if self.direct:
            body.pop("model", None)
        else:
            base_url = self.service_url
            body["model"] = name
        return base_url, body

    async def request(
        self, payload: dict, *, path: str = "generate", model: str | None = None, route_key: str | None = None
    ) -> dict:
        if payload.get("stream"):
            raise ValueError("Use InferenceClient.stream() for streaming requests")
        base_url, body = await self._route(payload, model, route_key)
        # Generation is not replayed after a transport failure: the backend
        # may already have accepted it. Next request reads fresh discovery.
        response = await self._client.post(f"{base_url}/{path.lstrip('/')}", json=body)
        response.raise_for_status()
        return response.json()

    async def stream(
        self, payload: dict, *, path: str = "generate", model: str | None = None, route_key: str | None = None
    ) -> AsyncIterator[bytes]:
        """Yield upstream SSE bytes; closing the iterator closes the
        connection."""
        base_url, body = await self._route({**payload, "stream": True}, model, route_key)
        async with self._client.stream("POST", f"{base_url}/{path.lstrip('/')}", json=body) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def generate(self, payload: dict, **routing) -> dict:
        return await self.request(payload, **routing)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "InferenceClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
