# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""E5 and GPU FAISS retrieval service.

Adapted from ``search_r1/search/retrieval_server.py`` in
https://github.com/PeterGriffinJin/Search-R1 at commit
``598e61bd1d36895726d28a8d06b3a15bed19f5d3`` (Apache-2.0).
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class E5Encoder:
    def __init__(self, model_path: str) -> None:
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    @torch.no_grad()
    def encode(self, queries: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            [f"query: {query}" for query in queries],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        hidden = output.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
        embeddings = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32, order="C")


class E5FlatRetriever:
    def __init__(self, *, index_path: str, corpus_path: str, model_path: str, topk: int) -> None:
        index = faiss.read_index(index_path)
        clone_options = faiss.GpuMultipleClonerOptions()
        clone_options.useFloat16 = True
        clone_options.shard = True
        self.index = faiss.index_cpu_to_all_gpus(index, co=clone_options)
        self.corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
        self.encoder = E5Encoder(model_path)
        self.topk = topk

    def search(
        self, queries: list[str], topk: int | None = None
    ) -> tuple[list[list[dict[str, Any]]], list[list[float]]]:
        topk = self.topk if topk is None else topk
        results: list[list[dict[str, Any]]] = []
        scores: list[list[float]] = []
        for start in range(0, len(queries), 512):
            embeddings = self.encoder.encode(queries[start : start + 512])
            batch_scores, batch_indices = self.index.search(embeddings, k=topk)
            index_rows = batch_indices.tolist()
            documents = [self.corpus[int(index)] for row in index_rows for index in row]
            results.extend(documents[offset : offset + topk] for offset in range(0, len(documents), topk))
            scores.extend(batch_scores.tolist())
        return results, scores


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


@dataclass
class _PendingRequest:
    queries: list[str]
    topk: int
    future: asyncio.Future[tuple[list[list[dict[str, Any]]], list[list[float]]]]


_STOP = object()


class RetrieverBatcher:
    def __init__(
        self,
        create_retriever: Callable[[], E5FlatRetriever],
        *,
        batch_wait_ms: float,
        max_batch_queries: int,
        max_pending_requests: int,
    ) -> None:
        self.create_retriever = create_retriever
        self.retriever: E5FlatRetriever | None = None
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.max_batch_queries = max_batch_queries
        self.queue: asyncio.Queue[_PendingRequest | object] = asyncio.Queue(maxsize=max_pending_requests)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-r1-retriever")
        self.worker: asyncio.Task[None] | None = None
        self.accepting = False
        self.deferred: _PendingRequest | object | None = None
        self.state_lock = asyncio.Lock()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self.retriever = await loop.run_in_executor(self.executor, self.create_retriever)
            self.worker = asyncio.create_task(self._run(), name="search-r1-retriever-batcher")
            self.accepting = True
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise

    async def close(self) -> None:
        async with self.state_lock:
            self.accepting = False
            await self.queue.put(_STOP)
        await self.worker
        await self.queue.join()
        self.worker = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._release_retriever)
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _release_retriever(self) -> None:
        self.retriever = None

    async def search(self, queries: list[str], topk: int) -> tuple[list[list[dict[str, Any]]], list[list[float]]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[list[list[dict[str, Any]]], list[list[float]]]] = loop.create_future()
        async with self.state_lock:
            if not self.accepting:
                raise RuntimeError("Search-R1 retriever batcher is not running.")
            await self.queue.put(_PendingRequest(queries=queries, topk=topk, future=future))
        return await future

    async def _collect_batch(self) -> tuple[list[_PendingRequest], bool]:
        first = self.deferred
        if first is None:
            first = await self.queue.get()
        else:
            self.deferred = None
        if first is _STOP:
            return [], True
        pending = [first]
        query_count = len(first.queries)
        deadline = asyncio.get_running_loop().time() + self.batch_wait_s
        while len(pending) < self.max_batch_queries:
            remaining_s = deadline - asyncio.get_running_loop().time()
            if remaining_s <= 0:
                break
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining_s)
                except asyncio.TimeoutError:
                    break
            if item is _STOP:
                return pending, True
            if query_count > 0 and query_count + len(item.queries) > self.max_batch_queries:
                self.deferred = item
                break
            pending.append(item)
            query_count += len(item.queries)
            if query_count >= self.max_batch_queries:
                break
        return pending, False

    def _execute_batch(
        self, pending: list[_PendingRequest]
    ) -> list[tuple[list[list[dict[str, Any]]], list[list[float]]]]:
        all_queries = [query for item in pending for query in item.queries]
        max_topk = max(item.topk for item in pending)
        all_results, all_scores = self.retriever.search(all_queries, max_topk)
        split_results = []
        offset = 0
        for item in pending:
            end = offset + len(item.queries)
            results = [documents[: item.topk] for documents in all_results[offset:end]]
            scores = [document_scores[: item.topk] for document_scores in all_scores[offset:end]]
            split_results.append((results, scores))
            offset = end
        return split_results

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            pending, stop = await self._collect_batch()
            active = [item for item in pending if not item.future.cancelled()]
            try:
                if active:
                    outputs = await loop.run_in_executor(self.executor, self._execute_batch, active)
                    for item, output in zip(active, outputs):
                        if not item.future.done():
                            item.future.set_result(output)
            except Exception as exc:
                for item in active:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _ in pending:
                    self.queue.task_done()
                if stop:
                    self.queue.task_done()
            if stop:
                return


@asynccontextmanager
async def lifespan(app: FastAPI):
    batcher = RetrieverBatcher(
        app.state.create_retriever,
        batch_wait_ms=app.state.batch_wait_ms,
        max_batch_queries=app.state.max_batch_queries,
        max_pending_requests=app.state.max_pending_requests,
    )
    await batcher.start()
    app.state.batcher = batcher
    try:
        yield
    finally:
        await batcher.close()


app = FastAPI(lifespan=lifespan)


@app.post("/retrieve")
async def retrieve(request: QueryRequest, raw_request: Request) -> dict[str, Any]:
    batcher: RetrieverBatcher = raw_request.app.state.batcher
    topk = batcher.retriever.topk if request.topk is None else request.topk
    results, scores = await batcher.search(request.queries, topk)
    if request.return_scores:
        rows = [
            [{"document": document, "score": score} for document, score in zip(documents, document_scores)]
            for documents, document_scores in zip(results, scores)
        ]
    else:
        rows = results
    return {"result": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Search-R1 E5 Flat retriever.")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--port", type=int, default=17389)
    parser.add_argument("--batch_wait_ms", type=float, default=5.0)
    parser.add_argument("--max_batch_queries", type=int, default=512)
    parser.add_argument("--max_pending_requests", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app.state.create_retriever = partial(
        E5FlatRetriever,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_path=args.retriever_model,
        topk=args.topk,
    )
    app.state.batch_wait_ms = args.batch_wait_ms
    app.state.max_batch_queries = args.max_batch_queries
    app.state.max_pending_requests = args.max_pending_requests
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
