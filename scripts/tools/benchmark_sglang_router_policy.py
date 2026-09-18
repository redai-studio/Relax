#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Inspect SGLang routing with eight CPU-only mock workers.

By default each policy receives 64 prompt groups with 8 samples per group,
matching the 512-request shape of the Qwen3-VL training entry. Requests in a
group share the same token prefix, so the placement difference is observable
without loading a model. This is a routing diagnostic, not a GPU performance
benchmark.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import http.client
import importlib.metadata
import importlib.util
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_POLICIES = ("cache_aware", "round_robin")


def router_available() -> bool:
    try:
        return importlib.util.find_spec("sglang_router.launch_router") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _router_version() -> str | None:
    try:
        return importlib.metadata.version("sglang-router")
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _port_is_open(port: int, timeout: float = 0.2) -> bool:
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(port: int, method: str, path: str, payload: Any | None, timeout: float) -> tuple[int, Any]:
    body = _json_bytes(payload) if payload is not None else None
    headers = {"content-type": "application/json"} if body is not None else {}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if not raw:
            value = None
        else:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw.decode("utf-8", errors="replace")
        return response.status, value
    finally:
        connection.close()


class RequestRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]


class _MockHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 1024


def _make_handler(worker_index: int, recorder: RequestRecorder):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: int, value: Any) -> None:
            body = _json_bytes(value)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {
                "/health",
                "/health_generate",
                "/get_model_info",
                "/get_server_info",
                "/model_info",
                "/v1/models",
            }:
                self._reply(200, {"status": "ok", "worker_index": worker_index})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/generate":
                self._reply(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(size) or b"{}")
                rid = request["rid"]
                input_ids = request["input_ids"]
                if not isinstance(rid, str) or not isinstance(input_ids, list) or not input_ids:
                    raise ValueError("rid and non-empty input_ids are required")
                group_index = int(input_ids[0]) - 1000
            except Exception as exc:
                self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})
                return

            recorder.add({"rid": rid, "group_index": group_index, "worker_index": worker_index})
            self._reply(
                200,
                {
                    "rid": rid,
                    "text": "",
                    "output_ids": [worker_index],
                    "meta_info": {"finish_reason": {"type": "stop"}},
                    "_task24_mock_worker": worker_index,
                },
            )

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


class MockWorkerPool:
    """Own all mock worker threads and listening sockets."""

    def __init__(self, workers: int, recorder: RequestRecorder | None = None) -> None:
        if workers <= 0:
            raise ValueError("workers must be greater than zero")
        self.workers = workers
        self.recorder = recorder or RequestRecorder()
        self.servers: list[_MockHTTPServer] = []
        self.threads: list[threading.Thread] = []
        self.ports: list[int] = []
        self._closed = False

    def start(self) -> "MockWorkerPool":
        if self.servers or self._closed:
            raise RuntimeError("mock worker pool cannot be started twice")
        try:
            for worker_index in range(self.workers):
                server = _MockHTTPServer(("127.0.0.1", 0), _make_handler(worker_index, self.recorder))
                thread = threading.Thread(
                    target=server.serve_forever,
                    name=f"task24-mock-worker-{worker_index}",
                    daemon=True,
                )
                self.servers.append(server)
                self.threads.append(thread)
                self.ports.append(int(server.server_address[1]))
                thread.start()
        except Exception:
            self.stop()
            raise
        return self

    @property
    def urls(self) -> list[str]:
        return [f"http://127.0.0.1:{port}" for port in self.ports]

    def stop(self) -> None:
        if self._closed:
            return
        for server, thread in zip(self.servers, self.threads, strict=True):
            if thread.is_alive():
                server.shutdown()
        for server in self.servers:
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=5)
        self._closed = True
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("a mock worker thread did not stop")

    @property
    def stopped(self) -> bool:
        return (
            self._closed
            and all(not thread.is_alive() for thread in self.threads)
            and all(not _port_is_open(port) for port in self.ports)
        )

    def __enter__(self) -> "MockWorkerPool":
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()


class RouterProcess:
    """Own one real SGLang router and its process group."""

    def __init__(
        self,
        policy: str,
        worker_urls: Sequence[str],
        *,
        balance_abs_threshold: int,
        balance_rel_threshold: float,
    ) -> None:
        self.policy = policy
        self.worker_urls = list(worker_urls)
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self.port = _free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._closed = False

    def start(self) -> None:
        command = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            *self.worker_urls,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--policy",
            self.policy,
            "--balance-abs-threshold",
            str(self.balance_abs_threshold),
            "--balance-rel-threshold",
            str(self.balance_rel_threshold),
            "--retry-max-retries",
            "1",
            "--log-level",
            "warn",
            "--disable-health-check",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )

    def _log_tail(self) -> str:
        self.log.flush()
        self.log.seek(0)
        return self.log.read()[-4000:]

    def wait_ready(self, expected_workers: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self.process is not None
            if self.process.poll() is not None:
                raise RuntimeError(f"router exited with {self.process.returncode}: {self._log_tail()}")
            try:
                health_status, _ = _request_json(self.port, "GET", "/health", None, 0.5)
                workers_status, value = _request_json(self.port, "GET", "/workers", None, 0.5)
                registered = value.get("workers") if isinstance(value, dict) else None
                if health_status == 200 and workers_status == 200 and len(registered or []) == expected_workers:
                    time.sleep(0.1)
                    return
            except (ConnectionError, OSError, TimeoutError):
                pass
            time.sleep(0.05)
        raise RuntimeError(f"router was not ready within {timeout}s: {self._log_tail()}")

    def stop(self) -> None:
        if self._closed:
            return
        process = self.process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        elif process is not None:
            process.wait()
        self.log.close()
        self._closed = True

    @property
    def stopped(self) -> bool:
        return (
            self._closed
            and self.process is not None
            and self.process.poll() is not None
            and not _port_is_open(self.port)
        )


def _one_request(port: int, group_index: int, sample_index: int, timeout: float) -> dict[str, Any]:
    rid = f"g{group_index:03d}-s{sample_index:02d}"
    payload = {
        "rid": rid,
        "input_ids": [group_index + 1000, 42, 43, 44],
        "sampling_params": {"max_new_tokens": 1},
    }
    status, response = _request_json(port, "POST", "/generate", payload, timeout)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {response}")
    if not isinstance(response, dict) or response.get("rid") != rid:
        raise ValueError(f"invalid response: {response!r}")
    worker_index = response.get("_task24_mock_worker")
    if not isinstance(worker_index, int):
        raise ValueError("response did not identify a mock worker")
    return {"rid": rid, "group_index": group_index, "worker_index": worker_index}


def _send_workload(
    port: int,
    groups: int,
    samples_per_group: int,
    concurrency: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], float]:
    jobs = [(group, sample) for group in range(groups) for sample in range(samples_per_group)]
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as executor:
        futures = {
            executor.submit(_one_request, port, group, sample, timeout): (group, sample) for group, sample in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            group, sample = futures[future]
            rid = f"g{group:03d}-s{sample:02d}"
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append({"rid": rid, "error": f"{type(exc).__name__}: {exc}"})
    return completed, sorted(failures, key=lambda item: item["rid"]), time.monotonic() - started


def _summarize(
    policy: str,
    groups: int,
    samples_per_group: int,
    workers: int,
    completed: list[dict[str, Any]],
    failures: list[dict[str, str]],
    observations: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    expected = {f"g{group:03d}-s{sample:02d}": group for group in range(groups) for sample in range(samples_per_group)}
    by_rid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in observations:
        by_rid[str(record.get("rid"))].append(record)
    completed_by_rid = {record["rid"]: record for record in completed}

    missing = sorted(set(expected) - set(by_rid))
    unexpected = sorted(set(by_rid) - set(expected))
    duplicates = sorted(rid for rid, records in by_rid.items() if len(records) != 1)
    integrity_errors: list[str] = []
    worker_counts = [0] * workers
    group_counts: collections.Counter[int] = collections.Counter()
    group_workers: dict[int, set[int]] = collections.defaultdict(set)
    for rid, expected_group in expected.items():
        records = by_rid.get(rid, [])
        if len(records) != 1:
            continue
        record = records[0]
        worker = record.get("worker_index")
        group = record.get("group_index")
        if group != expected_group or not isinstance(worker, int) or not 0 <= worker < workers:
            integrity_errors.append(f"{rid}: group={group}, worker={worker}")
            continue
        response = completed_by_rid.get(rid)
        if response is None or response.get("worker_index") != worker:
            integrity_errors.append(f"{rid}: response/observation worker mismatch")
            continue
        worker_counts[worker] += 1
        group_counts[group] += 1
        group_workers[group].add(worker)

    spread = collections.Counter(len(group_workers[group]) for group in range(groups))
    complete_groups = sum(group_counts[group] == samples_per_group for group in range(groups))
    expected_requests = groups * samples_per_group
    complete = (
        not failures
        and len(completed) == expected_requests
        and len(completed_by_rid) == expected_requests
        and len(observations) == expected_requests
        and not missing
        and not unexpected
        and not duplicates
        and not integrity_errors
        and complete_groups == groups
    )
    mean = statistics.fmean(worker_counts)
    worker_cv = statistics.pstdev(worker_counts) / mean if mean else 0.0
    return {
        "policy": policy,
        "groups": groups,
        "samples_per_group": samples_per_group,
        "expected_requests": expected_requests,
        "completed_requests": len(completed),
        "failed_requests": len(failures),
        "complete_groups": complete_groups,
        "elapsed_s": elapsed_s,
        "request_count_per_worker": {str(index): count for index, count in enumerate(worker_counts)},
        "request_count_cv": worker_cv,
        "workers_per_group": {str(width): count for width, count in sorted(spread.items())},
        "missing_rids": missing[:20],
        "unexpected_rids": unexpected[:20],
        "duplicate_rids": duplicates[:20],
        "integrity_errors": integrity_errors[:20],
        "failures": failures[:20],
        "complete": complete,
    }


def run_policy(
    policy: str,
    *,
    groups: int = 64,
    samples_per_group: int = 8,
    workers: int = 8,
    concurrency: int = 512,
    balance_abs_threshold: int = 10,
    balance_rel_threshold: float = 1.2,
    startup_timeout: float = 20.0,
    request_timeout: float = 20.0,
) -> dict[str, Any]:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    values = (groups, samples_per_group, workers, concurrency, startup_timeout, request_timeout)
    if any(value <= 0 for value in values):
        raise ValueError("workload sizes, concurrency, and timeouts must be greater than zero")
    if not router_available():
        raise RuntimeError("sglang_router is not installed; run this tool in the Relax runtime image")

    recorder = RequestRecorder()
    pool = MockWorkerPool(workers, recorder)
    router: RouterProcess | None = None
    try:
        pool.start()
        router = RouterProcess(
            policy,
            pool.urls,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
        )
        router.start()
        router.wait_ready(workers, startup_timeout)
        completed, failures, elapsed_s = _send_workload(
            router.port, groups, samples_per_group, concurrency, request_timeout
        )
        result = _summarize(
            policy,
            groups,
            samples_per_group,
            workers,
            completed,
            failures,
            recorder.snapshot(),
            elapsed_s,
        )
    finally:
        try:
            if router is not None:
                router.stop()
        finally:
            pool.stop()

    cleanup = {
        "router_stopped": router is not None and router.stopped,
        "workers_stopped": pool.stopped,
        "router_port": router.port if router is not None else None,
        "worker_ports": list(pool.ports),
    }
    result["cleanup"] = cleanup
    result["complete"] = bool(result["complete"] and cleanup["router_stopped"] and cleanup["workers_stopped"])
    return result


def run_benchmark(
    policies: Sequence[str],
    *,
    groups: int = 64,
    samples_per_group: int = 8,
    workers: int = 8,
    concurrency: int = 512,
    balance_abs_threshold: int = 10,
    balance_rel_threshold: float = 1.2,
    startup_timeout: float = 20.0,
    request_timeout: float = 20.0,
) -> dict[str, Any]:
    if not policies:
        raise ValueError("at least one policy is required")
    runs = [
        run_policy(
            policy,
            groups=groups,
            samples_per_group=samples_per_group,
            workers=workers,
            concurrency=concurrency,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
        )
        for policy in policies
    ]
    return {
        "router_version": _router_version(),
        "policies": list(policies),
        "workers": workers,
        "groups": groups,
        "samples_per_group": samples_per_group,
        "requests_per_policy": groups * samples_per_group,
        "runs": runs,
        "complete": all(run["complete"] for run in runs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        choices=SUPPORTED_POLICIES,
        dest="policies",
        help="policy to run; repeat for both (default: both)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--balance-abs-threshold", type=int, default=10)
    parser.add_argument("--balance-rel-threshold", type=float, default=1.2)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = (
        args.workers,
        args.groups,
        args.samples_per_group,
        args.concurrency,
        args.startup_timeout,
        args.request_timeout,
    )
    if any(value <= 0 for value in positive):
        parser.error("workload sizes, concurrency, and timeouts must be greater than zero")
    try:
        result = run_benchmark(
            args.policies or SUPPORTED_POLICIES,
            groups=args.groups,
            samples_per_group=args.samples_per_group,
            workers=args.workers,
            concurrency=args.concurrency,
            balance_abs_threshold=args.balance_abs_threshold,
            balance_rel_threshold=args.balance_rel_threshold,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
        )
    except Exception as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
