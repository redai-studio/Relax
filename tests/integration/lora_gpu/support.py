# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Process and transport helpers for the opt-in LoRA GPU acceptance
experiment."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests

from relax.distributed.checkpoint_service.lora_publication import EngineReply, classify_engine_response


class Engines:
    def __init__(self, args: Any, output: Path) -> None:
        self.args = args
        self.output = output
        self.processes = []

    def launch(self, port: int, gpu: int, label: str) -> Any:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONPATH"] = str(Path(__file__).parent / "instrumentation") + os.pathsep + env.get("PYTHONPATH", "")
        env["LORA_GPU_EVENTS"] = str(self.output / f"{label}.events.jsonl")
        env["LORA_GPU_AOT"] = str(int(self.args.aot))
        env["NCCL_CUMEM_ENABLE"] = "0"
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--base-gpu-id",
            str(gpu),
            "--enable-lora",
            "--max-loras-per-batch",
            "3",
            "--max-loaded-loras",
            "3",
            "--max-lora-rank",
            "16",
            "--lora-target-modules",
            "all",
            "--mem-fraction-static",
            "0.12",
            "--context-length",
            "2048",
            "--disable-cuda-graph",
            "--log-level",
            "info",
        ]
        if self.args.deterministic:
            cmd += ["--enable-deterministic-inference", "--attention-backend", "triton", "--disable-overlap-schedule"]
        with (self.output / f"{label}.log").open("wb") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(proc)
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{label} exited: {proc.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    return proc
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"{label} startup timeout")

    def stop(self, proc: Any) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            # This session was created by launch(); never target another job.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
            raise
        finally:
            if proc.poll() is not None:
                self.processes.remove(proc)

    def close(self) -> None:
        failures = []
        for proc in list(self.processes):
            try:
                self.stop(proc)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(f"Owned engine shutdown failed: {failures}")


class Fanout:
    def __init__(self, ports: Any) -> None:
        self.ports = ports
        self.gate = threading.Event()
        self.gate.set()
        self.half_ready = threading.Event()
        self.hold_end = False
        self.reject_end = False
        self.calls = []

    def fire(self, endpoint: Any, payload: Any) -> Any:
        pending = {}
        for engine, port in self.ports.items():
            box = {}
            self.calls.append((engine, payload.get("op")))

            def send(engine: Any = engine, port: Any = port, box: Any = box) -> None:
                body = dict(payload)
                if self.reject_end and engine == "engine1" and body.get("op") == "end":
                    body["expected_checksums"] = {name: "0" * 64 for name in body["expected_checksums"]}
                if self.hold_end and payload.get("op") == "end" and engine == "engine1":
                    if not self.gate.wait(120):
                        box["reply"] = EngineReply(False, "gate timeout", ambiguous=True)
                        return
                try:
                    r = requests.post(f"http://127.0.0.1:{port}{endpoint}", json=body, timeout=180)
                    box["reply"] = classify_engine_response(r.status_code, r.json())
                    if self.hold_end and payload.get("op") == "end" and engine == "engine0":
                        if box["reply"].success:
                            self.half_ready.set()
                except Exception as exc:
                    box["reply"] = EngineReply(False, str(exc), ambiguous=True)

            thread = threading.Thread(target=send, daemon=True)
            thread.start()
            pending[engine] = (thread, box)
        return pending

    def collect(self, pending: Any) -> Any:
        replies = {}
        for engine, (thread, box) in pending.items():
            thread.join(185)
            replies[engine] = box.get("reply", EngineReply(False, "missing reply", ambiguous=True))
        return replies


def init_transport(ports: Any, source_gpu: int, group_name: str) -> Any:
    import torch

    from relax.utils.distributed_utils import init_process_group

    torch.cuda.set_device(source_gpu)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    responses = []

    def connect(engine_port: Any, rank: Any) -> None:
        responses.append(
            requests.post(
                f"http://127.0.0.1:{engine_port}/init_weights_update_group",
                json={
                    "master_address": "127.0.0.1",
                    "master_port": port,
                    "world_size": 3,
                    "rank_offset": rank,
                    "group_name": group_name,
                    "backend": "nccl",
                },
                timeout=180,
            )
        )

    threads = [
        threading.Thread(target=connect, args=(p, rank), daemon=True) for rank, p in enumerate(ports.values(), 1)
    ]
    for thread in threads:
        thread.start()
    group = init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=3,
        rank=0,
        group_name=group_name,
        timeout=timedelta(seconds=180),
    )
    for thread in threads:
        thread.join(185)
    assert len(responses) == 2 and all(r.status_code == 200 for r in responses)
    return group


def events(path: Path) -> Any:
    return [json.loads(line) for line in path.read_text().splitlines()]
