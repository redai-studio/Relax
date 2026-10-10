# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Start a TP8 Kimi K3 SGLang server in Ray and verify real generation.

Optionally wait for the export Ray job to succeed. The GPU task keeps the
server and its GPU reservation alive after validation; stop this Ray job to
stop serving. Logs, the launch command, and responses are saved to --log-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def _logger() -> Any:
    from relax.utils.logging_utils import get_logger

    return get_logger(__name__)


def _request(base_url: str, path: str, body: dict[str, Any] | None = None, timeout: int = 600) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def _verify(base_url: str, timeout: int) -> list[dict[str, Any]]:
    results = []
    for prompt in ("What is 2 + 3? Give a brief answer.", "请用一句话说明天空为什么通常是蓝色的。"):
        body = {
            "model": "kimi-k3-export",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0,
            "reasoning_effort": "low",
        }
        response = _request(base_url, "/v1/chat/completions", body, timeout)
        choice = response["choices"][0]
        message = choice["message"]
        if not (message.get("content") or "").strip():
            raise ValueError(f"Chat request did not produce a final answer: {response}")
        if choice.get("finish_reason") not in ("stop", "length") or response["usage"]["completion_tokens"] < 1:
            raise ValueError(f"Invalid chat generation result: {response}")
        results.append({"request": body, "response": response})
    body = {
        "text": "The capital of France is",
        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    response = _request(base_url, "/generate", body, timeout)
    meta = response["meta_info"]
    if not response.get("text", "").strip() or meta.get("completion_tokens", 0) < 1:
        raise ValueError(f"Empty raw generation: {response}")
    logprobs = meta.get("output_token_logprobs", [])
    if not logprobs or any(not math.isfinite(item[0]) for item in logprobs):
        raise ValueError(f"Missing or nonfinite output log probabilities: {response}")
    results.append({"request": body, "response": response})
    return results


def _serve(args: argparse.Namespace) -> None:
    import ray

    sys.path.insert(0, args.repo_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    host = ray.util.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", args.port))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--trust-remote-code",
        "--tp-size",
        str(args.tp_size),
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "4096",
        "--max-running-requests",
        "1",
        "--reasoning-parser",
        "kimi_k3",
        "--tool-call-parser",
        "kimi_k3",
        "--served-model-name",
        "kimi-k3-export",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    command.extend(args.server_args)
    base_url = f"http://{host}:{port}"
    local_url = f"http://127.0.0.1:{port}"
    (log_dir / "launch.json").write_text(json.dumps({"command": command, "base_url": base_url}, indent=2) + "\n")
    env = os.environ.copy()
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = log_dir / "server.log"
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            _logger().info("SGLang starting: endpoint=%s pid=%d log=%s", base_url, process.pid, log_path)
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"SGLang exited with {process.returncode}; inspect {log_path}")
                try:
                    with urllib.request.urlopen(local_url + "/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"SGLang startup timed out; inspect {log_path}")
                _logger().info("Waiting for SGLang readiness: endpoint=%s log=%s", base_url, log_path)
                time.sleep(30)
            _logger().info("SGLang ready; checking real chat and raw generation")
            results = _verify(local_url, args.request_timeout)
            report = {
                "status": "passed",
                "model_path": args.model_path,
                "base_url": base_url,
                "completed_at": datetime.now().isoformat(),
                "results": results,
            }
            (log_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            _logger().info("INFERENCE_VALIDATION_PASSED endpoint=%s report=%s", base_url, log_dir / "validation.json")
            if not args.exit_after_validation:
                _logger().info("Keeping SGLang and its GPU reservation alive until this Ray job is stopped")
                code = process.wait()
                if code:
                    raise RuntimeError(f"SGLang exited with {code}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--original-hf-model", action="store_true", help="Serve original HF weights without a Relax export report"
    )
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--port", type=int, default=0, help="0 chooses an available port")
    parser.add_argument("--startup-timeout", type=int, default=7200)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--job-address")
    parser.add_argument("--after-job-id")
    parser.add_argument("--exit-after-validation", action="store_true")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.repo_root = str(Path(__file__).resolve().parents[4])
    sys.path.insert(0, args.repo_root)
    args.model_path = str(Path(args.model_path).resolve())
    args.log_dir = str(Path(args.log_dir).resolve())
    if args.server_args[:1] == ["--"]:
        args.server_args.pop(0)
    if args.after_job_id:
        if not args.job_address:
            parser.error("--after-job-id requires --job-address")
        from convert_kimi_k3_torch_dist_to_hf import _wait_for_job

        _wait_for_job(args.job_address, args.after_job_id)
    if args.original_hf_model:
        model_dir = Path(args.model_path)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        if not shards or any(not (model_dir / shard).is_file() for shard in shards):
            raise ValueError("Original HF model has missing weight shards")
        for filename in ("config.json", "tokenizer_config.json"):
            json.loads((model_dir / filename).read_text())
    elif not (Path(args.model_path) / "relax_export_report.json").is_file():
        raise ValueError("Model directory has not passed the native K3 export validation")
    import ray

    ray.init(address=args.ray_address)
    worker = ray.remote(num_gpus=args.tp_size, num_cpus=16, max_retries=0)(_serve)
    try:
        ray.get(worker.remote(args))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
