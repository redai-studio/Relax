# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local tools for the grader process that already lives inside Apptainer."""

import os
import signal
import subprocess
import tempfile
from pathlib import Path

from examples.gagar.app.code_generate.scoring import grade_to_weights
from examples.gagar.app.config import load_task_config
from examples.gagar.core.grs import GraderEnvironment


class SandboxBash:
    """Execute model commands inside the already isolated grader sandbox."""

    def __init__(self, manifest: dict, config: dict) -> None:
        # A configuration mistake must never turn this into host bash execution.
        if not os.environ.get("APPTAINER_CONTAINER") or not Path("/candidates").is_dir():
            raise RuntimeError("sandbox grader must run inside its Apptainer instance")
        self.timeout = manifest.get("command_timeout", config["command_timeout"])
        Path("/workspace").mkdir(exist_ok=True)

    def execute(self, action: dict) -> dict:
        if set(action) != {"name", "command"} or not isinstance(action["command"], str):
            raise ValueError("bash accepts only name and command; use /candidates/<rollout_n>/solution.py")
        # Neither host credentials nor the inference socket configuration
        # are put in the shell environment. Network isolation is enforced
        # by the outer container, not by checking command strings.
        env = {
            "PATH": "/opt/sf/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/workspace",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "PYTHONPATH": "/opt/sandboxfusion",
            "SANDBOX_CONFIG": "gagar",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(
                ["/bin/bash", "-c", action["command"]],
                cwd="/workspace",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also remove ordinary background jobs after a command exits.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            output.seek(0)
            text = output.read(12000).decode(errors="replace")
        return {"output": text, "returncode": -1 if timed_out else proc.returncode, "timed_out": timed_out}

    def close(self) -> None:
        # Container ownership and command-process cleanup stay with the worker.
        pass


def create_environment(manifest: dict, config: dict | None = None) -> GraderEnvironment:
    """Bind this task's tools and scoring to the shared grader environment."""
    if config is None:
        config = load_task_config("code_generate")["environment"]
    return GraderEnvironment(
        manifest,
        backend=SandboxBash(manifest, config),
        grade_to_weights=grade_to_weights,
        sections=set(config["evidence_sections"]),
        evidence_access=config["evidence_access"],
        environment_type=config["environment_type"],
    )
