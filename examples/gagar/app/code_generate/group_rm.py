# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Code-generation group RM and supervision of its sandbox worker."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from examples.gagar.app.code_generate.sandbox.worker import SandboxCleanupError, cleanup_scratch, write_json
from examples.gagar.app.code_generate.scoring import grade_to_weights
from examples.gagar.app.config import load_task_config


async def finish_worker(proc: Any, python: str, directory: Path, log: Any) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    leases = directory / "instances"
    if not list(leases.glob("mswe-*")):
        cleanup_scratch(directory, proc.pid)
        return
    # Retry in a separate process: imports and Apptainer calls cannot block the
    # reward event loop. Keep leases on failure so cleanup remains retryable.
    for _ in range(2):
        cleanup = await asyncio.create_subprocess_exec(
            python,
            "-m",
            "examples.gagar.app.code_generate.sandbox.worker",
            "--output",
            str(directory / "result.json"),
            "--cleanup-owner",
            str(proc.pid),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        try:
            await asyncio.wait_for(cleanup.wait(), timeout=60)
        except asyncio.TimeoutError:
            cleanup.kill()
            await cleanup.wait()
        if cleanup.returncode == 0:
            cleanup_scratch(directory, proc.pid)
            return
    raise SandboxCleanupError(f"sandbox cleanup still pending; retry leases in {leases}")


async def reward_group(args: Any, samples: list[Any], **kwargs: Any) -> list[dict]:
    from examples.gagar.core.gar import reward_records
    from relax.utils.logging_utils import get_logger
    from relax.utils.utils import get_serve_url

    task_id, label = samples[0].metadata["task_id"], samples[0].label
    if any(s.metadata["task_id"] != task_id or s.label != label for s in samples):
        raise ValueError("one grading group must contain the same coding task")
    task = samples[0].metadata.get("task", samples[0].prompt)
    if not isinstance(task, str):
        task = json.dumps(task, ensure_ascii=False)
    config = load_task_config("code_generate")
    payload = {
        "task_id": task_id,
        "label": label,
        "task": task,
        "responses": [s.response for s in samples],
        "genrm_url": get_serve_url("genrm"),
        "grader_config": config,
    }
    audit_root = Path(os.environ["GAGAR_AUDIT_DIR"])
    audit_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="group-", dir=audit_root))
    write_json(directory / "input.json", payload)
    started = time.monotonic()
    python = str(Path(os.environ["MINI_SWE_AGENT_VENV"]) / "bin/python")
    with (directory / "worker.log").open("wb") as log:
        proc = await asyncio.create_subprocess_exec(
            python,
            "-m",
            "examples.gagar.app.runner",
            "--task",
            "code_generate",
            "--input",
            str(directory / "input.json"),
            "--output",
            str(directory / "result.json"),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        try:
            await asyncio.wait_for(
                proc.wait(),
                timeout=config["environment"]["judge_timeout"] * len(samples)
                + config["environment"]["grading_timeout"]
                + config["environment"]["worker_grace_seconds"],
            )
        finally:
            finishing = asyncio.create_task(finish_worker(proc, python, directory, log))
            try:
                await asyncio.shield(finishing)
            except asyncio.CancelledError:
                await finishing
                raise
    if proc.returncode:
        raise RuntimeError(f"code-generation sandbox judging failed; see {directory / 'worker.log'}")
    result = json.loads((directory / "result.json").read_text())
    original = result["original"]
    if result["fallback"]:
        get_logger(__name__).warning("Code grader fallback in %s: %s", directory, result.get("error"))
        weights = original.copy()
    else:
        weights = grade_to_weights(original, result["grading"])["weights"]
    for sample in samples:
        sample.metadata["grader_seconds"] = time.monotonic() - started
        sample.metadata["grader_audit_dir"] = str(directory)
    return reward_records(original, original, weights, fallback=result["fallback"])
