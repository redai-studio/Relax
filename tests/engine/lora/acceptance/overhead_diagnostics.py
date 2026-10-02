# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Intrusive diagnostics after formal throughput measurement, never pooled with
it."""

from __future__ import annotations

import asyncio
import collections
import gzip
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from . import performance


def quantiles(values: list[float]) -> dict:
    values = sorted(value for value in values if math.isfinite(value) and value >= 0)
    return {
        "count": len(values),
        **{
            f"p{percent}": values[math.ceil(len(values) * percent / 100) - 1] if values else None
            for percent in (50, 95, 99)
        },
    }


def request_breakdown(raw: dict, start: float, end: float) -> dict:
    """Client wall durations and server durations have different clock
    domains."""
    by_rid = collections.defaultdict(list)
    for event in raw["progress"]:
        by_rid[event["rid"]].append(event)
    rows = []
    for req in raw["requests"]:
        if not start <= req["accepted"] < end or req["completed"] is None:
            continue
        row = {
            "rid": req["rid"],
            "engine": req["engine"],
            "tokens": req["tokens"],
            "server_metadata": req.get("server_timing", {}),
        }
        for name, first, last in (
            ("client_ttft_s", "accepted", "first_token_at"),
            ("client_decode_s", "first_token_at", "last_token_at"),
            ("client_response_tail_s", "last_token_at", "completed"),
            ("client_http_headers_s", "accepted", "headers_at"),
        ):
            if req.get(first) is not None and req.get(last) is not None:
                row[name] = req[last] - req[first]
        events = by_rid[req["rid"]]
        row["chunk_gaps_seconds"] = quantiles([b["at"] - a["at"] for a, b in zip(events, events[1:])])
        if req["tokens"] > 1 and "client_decode_s" in row:
            row["client_mean_decode_s_per_token"] = row["client_decode_s"] / (req["tokens"] - 1)
        meta = row["server_metadata"]
        if "queue_time" in meta:
            row["server_queue_s"] = meta["queue_time"]
        for name, first, last in (
            ("server_api_dispatch_s", "request_received_ts", "api_server_dispatch_finish_ts"),
            ("server_prefill_span_s", "forward_entry_time", "prefill_finished_time"),
            ("server_after_prefill_s", "prefill_finished_time", "request_finished_ts"),
        ):
            if meta.get(first) and meta.get(last) and meta[last] >= meta[first]:
                row[name] = meta[last] - meta[first]
        rows.append(row)
    names = sorted({key for row in rows for key in row if key.endswith("_s") or key.endswith("_per_token")})
    return {
        "requests": rows,
        "per_engine": {
            engine: {
                key: quantiles([row[key] for row in rows if row["engine"] == engine and key in row]) for key in names
            }
            for engine in sorted({row["engine"] for row in rows})
        },
        "limitations": "Client chunk gaps include transport/coalescing, not exact GPU token intervals. "
        "Response tail is not native drain. Missing server timing requires SGLang enable_metrics; "
        "API dispatch includes preprocessing/IPC, not isolated binding validation. No cross-clock subtraction.",
    }


def summarize_trace(path: Path) -> dict:
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as source:
        trace = json.load(source)
    counts = collections.Counter()
    totals = collections.defaultdict(lambda: [0, 0.0])
    intervals = collections.defaultdict(list)
    for event in trace.get("traceEvents", []):
        cat = event.get("cat", "unknown")
        counts[cat] += 1
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)) or event["dur"] < 0:
            continue
        group = "gpu_kernel" if cat == "kernel" else "gpu_copy" if cat in ("gpu_memcpy", "gpu_memset") else cat
        key = (group, event.get("name", "unknown"))
        totals[key][0] += 1
        totals[key][1] += event["dur"]
        if group in ("gpu_kernel", "gpu_copy"):
            intervals[str(event.get("args", {}).get("device", event.get("pid")))].append(
                (event["ts"], event["ts"] + event["dur"])
            )
    devices = {}
    for device, spans in intervals.items():
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        span = merged[-1][1] - merged[0][0]
        busy = sum(end - start for start, end in merged)
        devices[device] = {
            "union_busy_ms": busy / 1000,
            "first_to_last_gpu_span_ms": span / 1000,
            "busy_fraction_within_span": busy / span if span else None,
            "max_internal_gap_ms": max((b[0] - a[1] for a, b in zip(merged, merged[1:])), default=0) / 1000,
        }
    groups = collections.defaultdict(list)
    for (group, name), (count, duration) in totals.items():
        groups[group].append(
            {
                "name": name,
                "count": count,
                "inclusive_total_ms": duration / 1000,
                "mean_event_ms": duration / count / 1000,
            }
        )
    return {
        "file": str(path),
        "status": "COMPLETE" if counts["kernel"] else "INCOMPLETE",
        "categories": dict(counts),
        "gpu_activity": devices,
        "top_events": {
            group: sorted(items, key=lambda item: item["inclusive_total_ms"], reverse=True)[:30]
            for group, items in groups.items()
        },
        "limitations": "CPU events are inclusive/nested; GPU streams overlap. Totals are NOT wall-time overhead. "
        "GPU gaps cover first-to-last observed GPU activity only. Graph replay can hide operator/source detail; "
        "missing LoRA names does not prove zero LoRA cost. Inspect retained full traces for causality.",
    }


def owned_engine_pids(endpoints: list[str]) -> dict:
    """Only inspect this isolated worker's descendants; never attach by
    name."""
    import psutil

    ports = {urlsplit(endpoint).port: endpoint for endpoint in endpoints}
    found = {}
    for process in psutil.Process().children(recursive=True):
        try:
            for connection in process.net_connections(kind="tcp"):
                if connection.status == psutil.CONN_LISTEN and connection.laddr.port in ports:
                    found[ports[connection.laddr.port]] = process.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def summarize_samples(path: Path) -> dict:
    data = json.loads(path.read_text())
    frames = data["shared"]["frames"]
    counts = collections.Counter()
    profiles = []
    for profile in data["profiles"]:
        if profile.get("type") != "sampled":
            continue
        local = collections.Counter()
        for stack in profile["samples"]:
            if stack:
                frame = frames[stack[-1]]
                local[(frame["name"], frame.get("file"), frame.get("line"))] += 1
        counts.update(local)
        profiles.append(
            {
                "name": profile.get("name"),
                "unit": profile.get("unit"),
                "samples": sum(local.values()),
                "top_leaf_frames": [
                    {"function": key[0], "file": key[1], "line": key[2], "samples": count}
                    for key, count in local.most_common(20)
                ],
            }
        )
    return {
        "profiles": profiles,
        "status": "COMPLETE" if counts else "INCOMPLETE",
        "top_leaf_frames": [
            {"function": key[0], "file": key[1], "line": key[2], "samples": count}
            for key, count in counts.most_common(40)
        ],
        "scope": "Sample counts, not exclusive CPU time. Full stacks/weights remain in speedscope JSON.",
    }


async def binding_timings(ctx) -> dict:
    rows = []
    for _ in range(12):
        payload = {"owner_epoch": ctx.native_owner, "session_id": "profile-" + uuid4().hex}
        # Unknown RPC outcome still belongs to us and must be closed during cleanup.
        ctx.native_sessions.append({"session_id": payload["session_id"]})
        row = {"session_id": payload["session_id"]}
        rows.append(row)
        try:
            previous = None
            for action in ("first_bind_s", "repeat_bind_s"):
                start = time.perf_counter()
                bound = await asyncio.wait_for(ctx.manager.lora_control.remote("bind", payload), 10)
                row[action] = time.perf_counter() - start
                if bound.get("error_code") or (previous is not None and previous != bound):
                    raise AssertionError(f"binding failed or changed on retry: {bound}")
                previous = bound
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                closed = await asyncio.wait_for(ctx.manager.lora_control.remote("close", payload), 10)
                if not closed.get("accepted"):
                    raise RuntimeError(closed)
                row["close"] = "ACCEPTED"
            except Exception as error:
                row["error"] = f"close failed: {error}"
        if row.get("error"):
            break
    return {
        "status": "INCOMPLETE" if any(row.get("error") for row in rows) else "COMPLETE",
        "raw": rows,
        **{key: quantiles([row[key] for row in rows if key in row]) for key in ("first_bind_s", "repeat_bind_s")},
        "scope": "Manager bind RPC round trip (Ray/loop dispatch/lock/selection), not isolated function CPU time; "
        "normal continuation reuses its binding and does not repeat this RPC. Close ACK is not drain.",
    }


async def collect(ctx, bindings) -> dict:
    root = Path(ctx.config["output"]) / "diagnostics"
    root.mkdir()
    report = {
        "status": "INCOMPLETE",
        "phases": [],
        "included_in_overhead_ci": False,
        "py_spy": shutil.which("py-spy"),
        "profile_v2": os.environ.get("SGLANG_PROFILE_V2", "false"),
    }
    config = ctx.config
    ctx.config = {**config, "traffic_window_seconds": 30, "completion_followup_seconds": 180}
    try:
        ordinary = await performance.ordinary_bindings(ctx, bindings)
        if not ordinary:
            raise ValueError("ordinary engines are required for diagnosis")
        report["binding"] = await binding_timings(ctx)
        for kind in ("cpu", "gpu"):
            for mode, targets in (("ordinary", ordinary), ("managed", bindings)):
                phase = {"kind": kind, "mode": mode, "status": "INCOMPLETE", "collectors": []}
                report["phases"].append(phase)
                folder = root / f"{mode}-{kind}"
                folder.mkdir()
                endpoints = sorted({target["engine"]["endpoint"] for target in targets})
                pids = owned_engine_pids(endpoints) if kind == "cpu" and report["py_spy"] else {}
                processes, attempted = [], []
                stop_tasks = []

                async def stop_gpu(endpoint, entry):
                    try:
                        response = await ctx.client.post(endpoint + "/stop_profile", timeout=120)
                        response.raise_for_status()
                        entry["stop"] = "ACK"
                    except Exception as error:
                        entry["stop_error"] = f"{type(error).__name__}: {error}"

                async def finish(endpoint, entry):
                    await asyncio.sleep(3)
                    await stop_gpu(endpoint, entry)

                async def start():
                    for index, endpoint in enumerate(endpoints):
                        entry = {"endpoint": endpoint}
                        phase["collectors"].append(entry)
                        try:
                            if kind == "cpu":
                                if not report["py_spy"] or endpoint not in pids:
                                    entry["error"] = (
                                        "py-spy unavailable or engine listener not found in owned descendants"
                                    )
                                    continue
                                path = folder / f"cpu-{index}.json"
                                log = folder / f"cpu-{index}.log"
                                with log.open("w") as sink:
                                    process = subprocess.Popen(
                                        [
                                            report["py_spy"],
                                            "record",
                                            "--pid",
                                            str(pids[endpoint]),
                                            "--subprocesses",
                                            "--duration",
                                            "20",
                                            "--rate",
                                            "99",
                                            "--format",
                                            "speedscope",
                                            "--output",
                                            str(path),
                                        ],
                                        stdout=sink,
                                        stderr=subprocess.STDOUT,
                                    )
                                entry.update(pid=pids[endpoint], file=str(path), log=str(log))
                                processes.append((process, entry))
                            else:
                                if report["profile_v2"].lower() not in ("false", "0", ""):
                                    entry["error"] = "manual profiling requires SGLANG_PROFILE_V2=false"
                                    continue
                                trace_dir = folder / str(index)
                                trace_dir.mkdir()
                                entry["directory"] = str(trace_dir)
                                attempted.append((endpoint, entry))  # A timed-out start can still have taken effect.
                                response = await ctx.client.post(
                                    endpoint + "/start_profile",
                                    json={
                                        "output_dir": str(trace_dir.resolve()),
                                        "activities": ["CPU", "GPU"],
                                        "with_stack": True,
                                        "record_shapes": False,
                                    },
                                    timeout=20,
                                )
                                response.raise_for_status()
                                entry["start"] = "ACK"
                        except Exception as error:
                            entry["error"] = f"{type(error).__name__}: {error}"
                        finally:
                            if kind == "gpu" and "directory" in entry:
                                stop_tasks.append(asyncio.create_task(finish(endpoint, entry)))

                try:
                    phase["metrics"] = await performance.traffic_window(
                        ctx,
                        targets * config["overhead_profile"]["concurrency"],
                        managed=mode == "managed",
                        observe=False,
                        warmup_seconds=2,
                        diagnostic=True,
                        on_ready=start,
                    )
                    phase["status"] = "COLLECTED"
                except Exception as error:
                    phase["error"] = f"{type(error).__name__}: {error}"
                finally:
                    if stop_tasks:
                        await asyncio.gather(*stop_tasks)
                    for process, entry in processes:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        entry["exit_code"] = process.returncode
                        if process.returncode:
                            entry["error"] = Path(entry["log"]).read_text(errors="replace")[-4000:]
                        elif Path(entry["file"]).is_file():
                            try:
                                entry["samples"] = summarize_samples(Path(entry["file"]))
                            except (OSError, ValueError, KeyError) as error:
                                entry["error"] = f"sample parse failed: {error}"
                    if ctx.report.get("traffic_raw"):
                        raw = ctx.report["traffic_raw"].pop()
                        (folder / "requests.json").write_text(json.dumps(raw) + "\n")
                        if "metrics" in phase:
                            phase["request_timing"] = request_breakdown(
                                raw, phase["metrics"]["start"], phase["metrics"]["end"]
                            )
                for _, entry in attempted:
                    entry["traces"] = []
                    for path in Path(entry["directory"]).glob("*.trace.json*"):
                        try:
                            entry["traces"].append(summarize_trace(path))
                        except Exception as error:
                            entry["error"] = f"trace parse failed: {error}"
                complete = phase["status"] == "COLLECTED" and len(phase["collectors"]) == len(endpoints)
                for entry in phase["collectors"]:
                    complete &= not entry.get("error") and (
                        entry.get("samples", {}).get("status") == "COMPLETE"
                        if kind == "cpu"
                        else entry.get("stop") == "ACK"
                        and bool(entry.get("traces"))
                        and all(trace["status"] == "COMPLETE" for trace in entry["traces"])
                    )
                phase["status"] = "COMPLETE" if complete else "INCOMPLETE"
                (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = (
            "COMPLETE"
            if all(p["status"] == "COMPLETE" for p in report["phases"]) and report["binding"]["status"] == "COMPLETE"
            else "INCOMPLETE"
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        ctx.config = config
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
