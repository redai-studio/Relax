# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Async raw generation/chat client with explicit gateway or direct routing."""

from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from relax.engine.inference import InferenceRouter, prepare_payload


class InferenceClient:
    def __init__(
        self,
        service_url: str,
        *,
        mode: Literal["gateway", "direct"] = "gateway",
        timeout: float = 1800,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if mode not in ("gateway", "direct"):
            raise ValueError(f"Unknown inference mode: {mode}")
        self.service_url = service_url.rstrip("/")
        self.mode = mode
        self._router = InferenceRouter()
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def _get(self, path: str) -> dict[str, Any]:
        response = await self._http.get(self.service_url + path)
        response.raise_for_status()
        return response.json()

    async def engines(self) -> dict[str, Any]:
        return await self._get("/engines")

    async def health(self) -> dict[str, Any]:
        return await self._get("/health")

    async def list_models(self) -> dict[str, Any]:
        return await self._get("/v1/models")

    async def _target(self, path: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.mode == "gateway":
            return self.service_url + path, payload
        snapshot = await self.engines()
        target = self._router.select(snapshot, payload.get("model"), payload.get("route_key"))
        return target.base_url + path, prepare_payload(payload, target, chat=path != "/generate")

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream"):
            raise ValueError("Use stream_generate or stream_chat_completions for streaming requests")
        url, body = await self._target(path, payload)
        response = await self._http.post(url, json=body)
        response.raise_for_status()
        return response.json()

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send raw SGLang generation; legacy messages use GenRMClient."""
        if "messages" in payload:
            raise ValueError("Use chat_completions for messages, or GenRMClient for legacy GenRM generation")
        return await self._post("/generate", payload)

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/chat/completions", payload)

    async def _stream(self, path: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        url, body = await self._target(path, {**payload, "stream": True})
        async with self._http.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    def stream_generate(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        if "messages" in payload:
            raise ValueError("Use stream_chat_completions for messages")
        return self._stream("/generate", payload)

    def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        return self._stream("/v1/chat/completions", payload)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "InferenceClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
