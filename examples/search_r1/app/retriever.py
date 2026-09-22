# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Async client for a local Search-R1 ``POST /retrieve`` contract."""

from __future__ import annotations

import httpx


class RetrieverClient:
    def __init__(
        self,
        *,
        endpoint: str,
        topk: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.topk = topk
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> RetrieverClient:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.client.aclose()

    async def retrieve(self, query: str) -> list[str]:
        response = await self.client.post(
            self.endpoint,
            json={"queries": [query], "topk": self.topk},
        )
        response.raise_for_status()
        return [document["contents"] for document in response.json()["result"][0]]


def format_information(documents: list[str], max_chars: int) -> str:
    references = []
    for index, document in enumerate(documents, start=1):
        lines = document.split("\n")
        title = lines[0]
        text = "\n".join(lines[1:])
        references.append(f"Doc {index}(Title: {title}) {text}")
    reference = "\n".join(references)
    return f"\n\n<information>{reference}</information>\n\n"[:max_chars]
