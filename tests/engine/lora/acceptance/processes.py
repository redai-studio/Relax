# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local process ownership for opt-in GPU acceptance; never attach to existing
jobs."""

from __future__ import annotations

import csv
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import httpx

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def select_gpus(indices: list[str]) -> list[dict]:
    """Resolve physical nvidia-smi indices to UUIDs; never reuse another
    compute job's GPU."""
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError("select two distinct physical GPU indices")

    def query(fields: str) -> list[list[str]]:
        output = subprocess.check_output(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"], text=True, timeout=15
        )
        return [[value.strip() for value in row] for row in csv.reader(output.splitlines()) if row]

    inventory = {
        row[0]: dict(zip(("index", "uuid", "name", "memory_mb", "driver"), row, strict=True))
        for row in query("--query-gpu=index,uuid,name,memory.total,driver_version")
    }
    if any(index not in inventory for index in indices):
        raise ValueError(f"GPU indices {indices} not available; physical indices: {list(inventory)}")
    selected = [inventory[index] for index in indices]
    busy = {row[0]: row[1] for row in query("--query-compute-apps=gpu_uuid,pid")}
    for gpu in selected:
        if gpu["uuid"] in busy:
            raise RuntimeError(
                f"GPU {gpu['index']} has compute PID {busy[gpu['uuid']]}; "
                "stop your manual test engines first. No existing process was stopped."
            )
    return selected


def wait_ready(process: subprocess.Popen, client: httpx.Client, timeout: float, log: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"engine exited with {process.returncode}; see {log}")
        try:
            response = client.get("/health", timeout=2)
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"engine startup timed out; see {log}")


@contextmanager
def owned_process(command: list[str], env: dict, log: Path, records: list):
    token = uuid4().hex
    env = {**env, "RELAX_LORA_TEST_PROCESS_OWNER": token}
    record = {"command": command, "log": str(log), "cleanup": "PENDING", "owner": token}
    records.append(record)
    with log.open("x") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = process.pid
        try:
            yield process
        finally:
            for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if sig == signal.SIGKILL:
                        raise
            # Ray workers start separate process groups and can outlive a driver
            # interrupted during init. Reap only this invocation's inherited
            # random ownership token, never a global Ray/Python process pattern.
            import psutil

            descendants = []
            for child in psutil.process_iter():
                try:
                    if child.environ().get("RELAX_LORA_TEST_PROCESS_OWNER") == token:
                        descendants.append(child)
                        child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            _, alive = psutil.wait_procs(descendants, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=5)
            record["detached_processes_reaped"] = len(descendants)
            record["cleanup"] = "SIGNALLED_AND_LEADER_REAPED"


@contextmanager
def baseline(
    model: str,
    adapter: str,
    version: str,
    gpu: str,
    log: Path,
    cold: bool,
    records: list,
    *,
    deterministic: bool = False,
    metrics: bool = False,
):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    command += (
        "--dtype bfloat16 --tp-size 1 --enable-lora --lora-backend triton "
        "--max-lora-rank 8 --lora-target-modules q_proj v_proj "
        "--max-loras-per-batch 3 --max-loaded-loras 3 --mem-fraction-static 0.2 "
        "--context-length 4096 --max-running-requests 32 --cuda-graph-max-bs-decode 32 --max-total-tokens 8192"
    ).split()
    command += ["--lora-paths", f"{version}={adapter}"]
    if metrics:
        command.append("--enable-metrics")
    if cold:
        command.append("--disable-radix-cache")
    if deterministic:
        # Deterministic FlashInfer disables radix caching in v0.5.17.
        command += ["--enable-deterministic-inference", "--attention-backend", "triton"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    env.pop("RELAX_LORA_NATIVE_CONFIG", None)
    url = f"http://127.0.0.1:{port}"
    logger.info("Starting %s baseline on %s (%s)", version, gpu, log)
    with owned_process(command, env, log, records) as process:
        with httpx.Client(base_url=url, timeout=5, trust_env=False) as client:
            wait_ready(process, client, 600, log)
            response = client.get("/server_info", timeout=30)
            response.raise_for_status()
            profile = inference_profile(response.json(), deterministic=deterministic, cold=cold)
        yield {"url": url, "lora_path": version, "cache_enabled": not cold, "effective_profile": profile}


def inference_profile(info: dict, *, deterministic: bool, cold: bool) -> dict:
    """Reject silently disabled graphs/cache or an unapplied numerical
    profile."""
    expected = {
        "enable_deterministic_inference": deterministic,
        "disable_radix_cache": cold,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
    }
    if deterministic:
        expected["attention_backend"] = "triton"
    mismatch = {
        key: {"expected": value, "actual": info.get(key)} for key, value in expected.items() if info.get(key) != value
    }
    if mismatch:
        raise AssertionError(f"engine inference profile mismatch: {mismatch}")
    return {
        key: info.get(key) for key in (*expected, "attention_backend", "dtype", "lora_backend", "tp_size", "version")
    }
