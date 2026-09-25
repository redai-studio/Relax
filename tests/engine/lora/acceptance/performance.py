# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json
import math
import time
from uuid import uuid4

from . import support


async def traffic_window(
    ctx,
    bindings,
    publication=None,
    retry_of=None,
    *,
    managed=True,
    observe=True,
    warmup_seconds=0,
    on_ready=None,
    diagnostic=False,
):
    """Trace accepted requests, streamed business tokens and all
    completions."""
    support.configure_test(ctx, audit_mutations=True)
    mutation_count = sum(len(path.read_text().splitlines()) for path in ctx.root.glob("mutations-*.jsonl"))
    stop = asyncio.Event()
    progress = []
    requests = []
    engines = tuple(dict.fromkeys(bound["engine"]["engine_id"] for bound in bindings))
    first = [asyncio.Event() for _ in bindings]
    ids = ctx.tokenizer.encode(ctx.config.get("traffic_prompt", "Explain how rain forms."), add_special_tokens=True)

    async def worker(bound, warmed_event):
        identity, engine = bound["binding"], bound["engine"]
        warmed = False
        while not stop.is_set():
            rid = uuid4().hex
            record = {
                "rid": rid,
                "engine": engine["engine_id"],
                "version": identity["version_id"],
                "accepted": time.monotonic(),
                "completed": None,
                "tokens": 0,
                "first_token_at": None,
            }
            requests.append(record)
            payload = {
                "rid": rid,
                "input_ids": ids,
                "lora_path": bound["native_lora_id"] if managed else identity["lora_path"],
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": ctx.config.get("traffic_tokens", 128),
                    "ignore_eos": True,
                },
                "lora_binding": {
                    "cohort_id": identity["cohort_id"],
                    "engine_boot_id": engine["boot_id"],
                    "digest": identity["digest"],
                    "native_lora_id": bound["native_lora_id"],
                    "owner_epoch": ctx.native_owner,
                    "session_id": bound["session_id"],
                },
            }
            if not managed:
                payload.pop("lora_binding")
                payload["routed_dp_rank"] = bound["dp_rank"]
            try:
                terminal = False
                async with ctx.client.stream("POST", engine["endpoint"] + "/generate", json=payload) as response:
                    response.raise_for_status()
                    if diagnostic:
                        record["headers_at"] = time.monotonic()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        event = json.loads(line[5:])
                        meta = event.get("meta_info", {})
                        actual = meta.get("lora_adapter", {})
                        if managed and (
                            actual.get("native_lora_id") != bound["native_lora_id"]
                            or actual.get("engine_boot_id") != engine["boot_id"]
                            or actual.get("adapter_digest") != identity["digest"]
                        ):
                            raise AssertionError("streamed token adapter identity mismatch")
                        tokens = int(meta.get("completion_tokens", 0))
                        if tokens > record["tokens"]:
                            if record["first_token_at"] is None:
                                record["first_token_at"] = time.monotonic()
                            progress.append(
                                {
                                    "at": time.monotonic(),
                                    "engine": engine["engine_id"],
                                    "rid": rid,
                                    "tokens": tokens - record["tokens"],
                                }
                            )
                            record["tokens"] = tokens
                            if diagnostic:
                                record["last_token_at"] = progress[-1]["at"]
                            if warmed:
                                warmed_event.set()
                        if meta.get("finish_reason") is not None:
                            record["finish_reason"] = meta["finish_reason"]
                            if diagnostic:
                                record["terminal_at"] = time.monotonic()
                                record["server_timing"] = {
                                    key: meta[key]
                                    for key in (
                                        "queue_time",
                                        "forward_entry_time",
                                        "prefill_finished_time",
                                        "request_received_ts",
                                        "api_server_dispatch_finish_ts",
                                        "response_sent_to_client_ts",
                                        "request_finished_ts",
                                        "decode_throughput",
                                        "cached_tokens",
                                        "prompt_tokens",
                                        "e2e_latency",
                                    )
                                    if key in meta
                                }
                            terminal = True
                if not terminal:
                    raise AssertionError("stream ended without native terminal")
                if record["finish_reason"].get("type") != "length" or record["tokens"] != ctx.config.get(
                    "traffic_tokens", 128
                ):
                    raise AssertionError("traffic request ended early or was interrupted")
                record["completed"] = time.monotonic()
                # Measure only after warmup and the next decode has started.
                warmed = True
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                stop.set()
                raise

    resources = []

    async def observe_resources():
        while managed and observe and not stop.is_set():
            state = await support.request(ctx, "GET", "/lora/versions")
            observations = []
            for version in state["versions"].values():
                for receipt in version["ready"].values():
                    engine = receipt["engine"]
                    response = await ctx.client.post(
                        engine["endpoint"] + "/lora_version_status",
                        json={
                            "cohort_id": state["cohort_id"],
                            "engine_boot_id": engine["boot_id"],
                            "native_lora_id": receipt["native_lora_id"],
                            "digest": version["digest"],
                        },
                    )
                    response.raise_for_status()
                    observations.append(response.json())
            resources.append({"at": time.monotonic(), "manager": state, "native": observations})
            await asyncio.sleep(0.5)

    tasks = [asyncio.create_task(worker(bound, event)) for bound, event in zip(bindings, first, strict=True)]
    workers = asyncio.gather(*tasks)
    ready = asyncio.gather(*(event.wait() for event in first))
    observer = asyncio.create_task(observe_resources())
    publishing = None
    publication_timing = {}

    async def timed_publication():
        publication_timing["started"] = time.monotonic()
        try:
            publication_timing["result"] = await support.publish(ctx, publication, retry_of=retry_of)
        finally:
            publication_timing["completed"] = time.monotonic()
            publication_timing["seconds"] = publication_timing["completed"] - publication_timing["started"]

    try:
        await asyncio.wait_for(
            asyncio.wait((ready, workers), return_when=asyncio.FIRST_COMPLETED), ctx.config["timeout_seconds"]
        )
        if workers.done():
            await workers
            raise AssertionError("traffic stopped before the warmup completed")
        await ready
        await asyncio.sleep(warmup_seconds)
        if workers.done():
            await workers
        if on_ready is not None:
            await on_ready()
        start = time.monotonic()
        publishing = asyncio.create_task(timed_publication()) if publication else None
        await asyncio.sleep(ctx.config.get("traffic_window_seconds", 10))
        if publishing is not None:
            await publishing
        end = time.monotonic()
        stop.set()
        await asyncio.wait_for(workers, ctx.config.get("completion_followup_seconds", 30))
        await observer
    finally:
        stop.set()
        support.configure_test(ctx)
        observer.cancel()
        if publishing is not None and not publishing.done():
            publishing.cancel()
        await asyncio.gather(observer, *([publishing] if publishing is not None else []), return_exceptions=True)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        ready.cancel()
        await asyncio.gather(ready, workers, return_exceptions=True)
        # Even failed/censored requests survive in the report.
        ctx.report.setdefault("traffic_raw", []).append(
            {"requests": requests, "progress": progress, "resources": resources, "publication": publication_timing}
        )
    calls = sum(len(path.read_text().splitlines()) for path in ctx.root.glob("mutations-*.jsonl")) - mutation_count
    if calls:
        raise AssertionError("publication traffic invoked a forbidden native mutation")
    if publication:
        crossing = {
            engine: [
                item["rid"]
                for item in requests
                if item["engine"] == engine
                and item["first_token_at"] is not None
                and item["first_token_at"] <= publication_timing["started"]
                and item["completed"] is not None
                and item["completed"] >= publication_timing["completed"]
            ]
            for engine in engines
        }
        publication_timing["crossing_requests"] = crossing
        if not all(crossing.values()):
            raise AssertionError(
                "no old decode spans the full publication on every engine; traffic profile lacks coverage"
            )
    return {**summarize_window(requests, progress, engines, start, end, calls), "publication": publication_timing}


def summarize_window(requests, progress, engines, start, end, calls=0):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("performance window must have positive finite duration")
    accepted = [record for record in requests if start <= record["accepted"] <= end]
    if not accepted or any(record["completed"] is None or record.get("error") for record in accepted):
        raise AssertionError("traffic has failed, censored or missing accepted requests")
    latencies = sorted(record["completed"] - record["accepted"] for record in accepted)
    gaps = {}
    for engine in (None, *engines):
        times = [
            start,
            *sorted(
                event["at"]
                for event in progress
                if start <= event["at"] <= end and (engine is None or event["engine"] == engine)
            ),
            end,
        ]
        gaps[engine or "all"] = max(b - a for a, b in zip(times, times[1:]))
    return {
        "start": start,
        "end": end,
        "token_throughput": {
            engine or "all": sum(
                event["tokens"]
                for event in progress
                if start <= event["at"] < end and (engine is None or event["engine"] == engine)
            )
            / (end - start)
            for engine in (None, *engines)
        },
        "accepted": len(accepted),
        "failed": 0,
        "forbidden_mutation_calls": calls,
        "ttft": [item["first_token_at"] - item["accepted"] for item in accepted if item["first_token_at"] is not None],
        "p50": latencies[math.ceil(len(latencies) * 0.5) - 1],
        "p95": latencies[math.ceil(len(latencies) * 0.95) - 1],
        "p99": latencies[math.ceil(len(latencies) * 0.99) - 1],
        "max_progress_gaps": gaps,
        "throughput": sum(start <= item["completed"] <= end for item in requests if item["completed"] is not None)
        / (end - start),
    }


def compare_windows(actual, baseline, profile):
    if actual["failed"] or actual["forbidden_mutation_calls"]:
        raise AssertionError("performance window contains failures or forbidden mutations")
    if (
        actual["p95"] > baseline["p95"] * profile.get("max_p95_ratio", 1.5)
        or actual["throughput"] < baseline["throughput"] * profile.get("min_throughput_ratio", 0.8)
        or max(actual["max_progress_gaps"].values()) > profile.get("max_progress_gap", 2)
    ):
        raise AssertionError("performance exceeded the pre-registered profile")


async def ordinary_bindings(ctx, bindings):
    """Same request workload on pre-existing unmanaged LoRA engines.

    These are separate from cold numerical baselines. Hardware and all server
    flags must be pre-registered; this function never launches or changes them.
    """
    servers = ctx.config.get("performance_baselines")
    if not servers:
        return None
    engine_ids = sorted({bound["engine"]["engine_id"] for bound in bindings})
    if len(servers) != len(engine_ids):
        raise ValueError("ordinary performance baseline must match managed engine count")
    by_engine = dict(zip(engine_ids, servers, strict=True))
    ordinary, observations = [], []
    compared = (
        "tp_size",
        "pp_size",
        "dp_size",
        "dtype",
        "lora_backend",
        "max_running_requests",
        "max_total_tokens",
        "attention_backend",
        "enable_deterministic_inference",
        "disable_radix_cache",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
        "cuda_graph_max_bs_decode",
        "cuda_graph_bs_decode",
        "enable_lora_overlap_loading",
        "context_length",
    )
    if ctx.config.get("native_metrics"):
        compared += ("enable_metrics",)
    for original in bindings:
        server = by_engine[original["engine"]["engine_id"]]
        native_response, managed_response = await asyncio.gather(
            ctx.client.get(server["url"].rstrip("/") + "/server_info"),
            ctx.client.get(original["engine"]["endpoint"] + "/server_info"),
        )
        native_response.raise_for_status()
        managed_response.raise_for_status()
        native, managed_info = native_response.json(), managed_response.json()
        for info in (native, managed_info):
            if not info.get("internal_states") or not info.get("enable_lora") or info.get("disable_cuda_graph", True):
                raise AssertionError("performance baseline requires live graph-enabled LoRA engines")
        if ctx.config.get("native_metrics") and not (
            native.get("enable_metrics") and managed_info.get("enable_metrics")
        ):
            raise AssertionError("requested server timing requires metrics enabled on both arms")
        if any(key not in native or key not in managed_info or native[key] != managed_info[key] for key in compared):
            raise ValueError("ordinary and publication performance configurations differ")
        ordinary.append(
            {
                **original,
                "engine": {**original["engine"], "endpoint": server["url"].rstrip("/")},
                "binding": {**original["binding"], "lora_path": server["lora_path"]},
            }
        )
        observations.append({"ordinary": native, "publication": managed_info, "hardware": server["hardware"]})
    ctx.report["performance_server_configs"] = observations
    return ordinary


async def ordinary_baseline(ctx, bindings, *, concurrency=2, **traffic_options):
    ordinary = await ordinary_bindings(ctx, bindings)
    if ordinary is None:
        return None
    return await traffic_window(ctx, ordinary * concurrency, managed=False, **traffic_options)
