# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sandbox worker: isolated official judging, then an in-sandbox grader."""

import argparse
import json
import os
import shlex
import shutil
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Iterator

import httpx

from examples.gagar.app.code_generate.reward import extract_code, judge_request, outcome_from_result
from examples.gagar.app.code_generate.scoring import grade_to_weights
from examples.gagar.app.config import load_task_config


class SandboxCleanupError(Exception):
    """Do not turn an unconfirmed container stop into a successful RM
    result."""


@contextmanager
def scratch_directory(parent: str | None) -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="gagar-code-", dir=parent))
    root.chmod(0o755)
    keep = False
    try:
        yield root
    except SandboxCleanupError:
        keep = True
        raise
    finally:
        if not keep:
            shutil.rmtree(root)


class GenRMProxy:
    """A group-local Unix socket that forwards only messages to the fixed
    GenRM."""

    def __init__(self, path: Path, service_url: str, *, timeout: float = 600) -> None:
        self.client = httpx.Client(timeout=timeout, trust_env=False)
        self.url = service_url.rstrip("/") + "/generate"
        self.calls = 0
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                self.connection.settimeout(5)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/generate" or not 0 < size <= 4 * 1024 * 1024 or proxy.calls >= 120:
                        self.send_error(400, "invalid or exhausted GenRM request")
                        return
                    payload = json.loads(self.rfile.read(size))
                    messages = payload["messages"]
                    if not isinstance(messages, list) or any(
                        m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str)
                        for m in messages
                    ):
                        raise ValueError("invalid messages")
                    proxy.calls += 1
                    response = proxy.client.post(proxy.url, json={"messages": messages})
                    response.raise_for_status()
                    body = json.dumps({"response": response.json()["response"]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (ValueError, KeyError, TypeError, AttributeError, httpx.HTTPError, OSError):
                    self.send_error(502, "GenRM request failed")

        self.server = socketserver.UnixStreamServer(str(path), Handler)
        path.chmod(0o666)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.client.close()
        self.thread.join()


def process_environment(tmp_dir: Path) -> dict[str, str]:
    # Do not inherit bind injection, proxies, credentials or APPTAINERENV_*.
    return {
        "PATH": os.defpath + ":/usr/local/bin",
        "HOME": str(Path.home()),  # Apptainer's host-side instance registry only.
        "TMPDIR": str(tmp_dir),
        "LANG": "C.UTF-8",
    }


@contextmanager
def instance(image: str, tmp_dir: Path, mounts: dict[Path, str], leases: Path) -> Iterator[Any]:
    from examples.gagar.app.code_generate.sandbox.apptainer import (
        ApptainerInstanceEnvironment,
        _new_instance_name,
        _stop_apptainer_instance,
    )

    name = _new_instance_name()
    leases.mkdir(exist_ok=True)
    lease = leases / name
    lease.write_text(str(tmp_dir))
    env = None
    args = [
        "--containall",
        "--cleanenv",
        "--no-home",
        "--no-mount",
        "hostfs,cwd,bind-paths,sys",
        "--net",
        "--network",
        "none",
        "--no-privs",
        "--drop-caps",
        "all",
    ]
    for source, destination in mounts.items():
        args.extend(["--bind", f"{source.resolve()}:{destination}:ro"])
    try:
        env = ApptainerInstanceEnvironment(
            tmp_dir=tmp_dir,
            instance_name=name,
            image=image,
            executable="apptainer",
            global_args=["--quiet"],
            exec_args=args,
            process_env=process_environment(tmp_dir),
            sandbox_build_retries=1,
        )
        yield env
    finally:
        try:
            if env is not None:
                env.cleanup()
            else:
                error = _stop_apptainer_instance("apptainer", ["--quiet"], name)
                if error:
                    raise RuntimeError(error)
            lease.unlink()
        except Exception as exc:
            raise SandboxCleanupError(f"could not stop {name}; preserve scratch at {tmp_dir}: {exc}") from exc


def cleanup_leases(leases: Path, owner: int) -> None:
    """A fresh host worker can retry cleanup after the grading worker exits."""
    from examples.gagar.app.code_generate.sandbox.apptainer import _instance_owner_pid, _stop_apptainer_instance

    for lease in leases.glob("mswe-*"):
        if _instance_owner_pid(lease.name) != owner:
            continue
        error = _stop_apptainer_instance("apptainer", ["--quiet"], lease.name)
        if error:
            raise SandboxCleanupError(f"cleanup pending at {lease}: {error}")
        root = Path(lease.read_text())
        lease.unlink()
        if root.exists():
            shutil.rmtree(root)


def cleanup_scratch(directory: Path, owner: int) -> None:
    record = directory / "scratch.json"
    if record.exists():
        scratch = json.loads(record.read_text())
        if scratch["owner"] == owner:
            root = Path(scratch["path"])
            if root.exists():
                shutil.rmtree(root)
            record.unlink()


def execute(env: Any, argv: list[str], timeout: int) -> str:
    """Only fixed harness commands cross this host/container boundary."""
    from examples.gagar.app.code_generate.sandbox.apptainer import _tracked_run

    # Limits apply to all descendants, including commands launched by the grader.
    command = (
        "export HOME=/workspace XDG_CONFIG_HOME=/workspace/.config; ulimit -f 65536; ulimit -t 120; ulimit -u 128; ulimit -v 4194304; exec "
        + shlex.join(argv)
    )
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        result = _tracked_run(
            [
                "apptainer",
                "--quiet",
                "exec",
                "--cleanenv",
                "--no-home",
                "--no-privs",
                "--drop-caps",
                "all",
                "--security",
                "uid:65534,gid:65534",
                "--pwd",
                "/workspace",
                str(env.sandbox_dir),
                "/bin/bash",
                "-c",
                command,
            ],
            env=env.apptainer_env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=errors,
            timeout=timeout,
            start_new_session=True,
        )
        errors.seek(0)
        if result.returncode:
            raise RuntimeError(
                f"sandbox command failed ({result.returncode}): {errors.read(8000).decode(errors='replace')}"
            )
        output.seek(0)
        return output.read(16 * 1024 * 1024).decode()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False))


def run_group(payload: dict, output: Path) -> None:
    """Runs in the mini-swe-agent venv, never in the training process."""
    from examples.gagar.app.code_generate.sandbox.apptainer import _drain_orphans, _install_child_subreaper_and_reaper

    _install_child_subreaper_and_reaper()
    config = payload.get("grader_config")
    if config is None:
        config = load_task_config("code_generate")
    settings = config["environment"]
    image = os.environ["GAGAR_SANDBOX_IMAGE"]
    code_root = Path(__file__).resolve().parents[3]
    scratch_root = os.environ.get("GAGAR_SANDBOX_TMPDIR")
    if scratch_root:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)
    original = []
    candidates = []
    leases = output.with_name("instances")
    try:
        # No writable host directories are mounted into either kind of instance.
        with scratch_directory(scratch_root) as root:
            write_json(output.with_name("scratch.json"), {"owner": os.getpid(), "path": str(root)})
            for index, response in enumerate(payload["responses"]):
                code = extract_code(response)
                reason, closed, _ = response.rpartition("</think>")
                candidate = {
                    "rollout_n": index,
                    "code": code or "",
                    "submission": response,
                    "reason": reason.removeprefix("<think>") if closed else "",
                    "reward": 0.0,
                }
                if code is not None:
                    judge_dir = root / f"judge-{index}"
                    judge_dir.mkdir()
                    write_json(judge_dir / "request.json", judge_request(payload["task_id"], payload["label"], code))
                    with instance(
                        image, root, {code_root: "/opt/relax/examples/gagar", judge_dir: "/input"}, leases
                    ) as env:
                        raw = execute(
                            env,
                            [
                                "/opt/sf/bin/python",
                                "-m",
                                "examples.gagar.app.code_generate.sandbox.sandbox_fusion",
                                "/input/request.json",
                            ],
                            settings["judge_timeout"],
                        )
                        result = json.loads(raw)
                        candidate["reward"] = outcome_from_result(payload["task_id"], result)
                        candidate["test_output"] = result
                else:
                    candidate["test_output"] = "No closed thinking block or no nonempty final code block."
                original.append(candidate["reward"])
                candidates.append(candidate)
            # Freeze official results outside the grader's writable filesystem.
            write_json(output.with_name("outcomes.json"), candidates)
            if not any(original):
                write_json(output, {"original": original, "grading": {"evaluations": []}, "fallback": False})
                return

            candidate_dir = root / "candidates"
            candidate_dir.mkdir()
            for candidate in candidates:
                directory = candidate_dir / str(candidate["rollout_n"])
                directory.mkdir()
                (directory / "solution.py").write_text(candidate["code"])
            grader_dir, socket_dir = root / "input", root / "genrm"
            grader_dir.mkdir()
            socket_dir.mkdir()
            manifest = {
                "task": payload["task"],
                "grader_config": config,
                "candidates": [
                    {key: candidate[key] for key in ("rollout_n", "code", "reward", "test_output")}
                    for candidate in candidates
                ],
                "genrm_url": "http://genrm",
                "genrm_socket": "/genrm/model.sock",
            }
            write_json(grader_dir / "manifest.json", manifest)
            grading = None
            error = None
            proxy = GenRMProxy(socket_dir / "model.sock", payload["genrm_url"], timeout=settings["proxy_timeout"])
            try:
                with instance(
                    image,
                    root,
                    {
                        code_root: "/opt/relax/examples/gagar",
                        grader_dir: "/input",
                        candidate_dir: "/candidates",
                        socket_dir: "/genrm",
                    },
                    leases,
                ) as env:
                    try:
                        execute(
                            env,
                            [
                                "/opt/grader/bin/python",
                                "-m",
                                "examples.gagar.core.grs",
                                "--task",
                                "code_generate",
                                "--input",
                                "/input/manifest.json",
                                "--output",
                                "/workspace/result.json",
                            ],
                            settings["grading_timeout"],
                        )
                        grading = json.loads(
                            execute(env, ["/bin/cat", "/workspace/result.json"], settings["trace_timeout"])
                        )
                        grade_to_weights(original, grading)
                    finally:
                        # Best effort trace retrieval must not hide a cleanup failure.
                        for name in ("trajectory.json", "events.jsonl", "summary.json"):
                            try:
                                trace = execute(env, ["/bin/cat", f"/workspace/{name}"], settings["trace_timeout"])
                                output.with_name(name).write_text(trace)
                            except (RuntimeError, subprocess.TimeoutExpired, OSError):
                                pass
            except (RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
            finally:
                proxy.close()
            write_json(
                output,
                {
                    "original": original,
                    "grading": grading if error is None else None,
                    "fallback": error is not None,
                    "error": error,
                    "genrm_calls": proxy.calls,
                },
            )
    finally:
        _drain_orphans()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cleanup-owner", type=int)
    args = parser.parse_args()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "true")

    if args.cleanup_owner is not None:
        cleanup_leases(args.output.with_name("instances"), args.cleanup_owner)
        cleanup_scratch(args.output.parent, args.cleanup_owner)
        return
    if args.input is None:
        parser.error("--input is required when grading")

    # Drop inherited supplementary host groups before Apptainer records them.
    if os.geteuid() == 0:
        os.setgroups([])

    def cancel(*_: Any) -> None:
        raise KeyboardInterrupt("sandbox worker cancelled")

    signal.signal(signal.SIGTERM, cancel)
    parent = os.getppid()

    def watch_parent() -> None:
        while os.getppid() == parent:
            time.sleep(5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch_parent, daemon=True).start()
    run_group(json.loads(args.input.read_text()), args.output)


if __name__ == "__main__":
    main()
