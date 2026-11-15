# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Narrow Python endpoint for the pinned SandboxFusion image, inside Apptainer.

Reuse its lite runner; fix the image's 4 GiB limit and bridged network defaults
to match the recipe's stateless, unprivileged, 30-second/1-GiB execution
contract.
"""

import argparse
import asyncio
import os
import shlex
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not os.environ.get("APPTAINER_CONTAINER") or os.getuid() != 0:
        raise RuntimeError("Run through run-sandbox-apptainer.sh, with a root driver inside Apptainer")

    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from sandbox.configs.run_config import RunConfig
    from sandbox.runners import base
    from sandbox.runners.isolation import execute_command
    from sandbox.server.sandbox_api import RunCodeRequest, run_code

    config = RunConfig.get_instance_sync()
    config.sandbox.isolation = "lite"
    config.sandbox.cleanup_process = False
    config.sandbox.restore_bash = False
    original_cgroup, original_network = base.tmp_cgroup, base.tmp_netns
    original_command = base.run_command_bare

    # These resources are private to concurrent requests. Upstream pools reuse
    # idle cgroups/netns; payload processes cannot create children or threads.
    base.tmp_cgroup = lambda **_: original_cgroup(mem_limit="1G", cpu_limit=1)
    base.tmp_netns = lambda *_: original_network(no_bridge=True)

    async def restricted_command(
        command: list[str], timeout: float, stdin: str | None, cwd: str, extra_env: dict, use_exec: bool
    ) -> Any:
        if not use_exec or "chroot" not in command:
            raise RuntimeError("Expected the pinned SandboxFusion lite runner")
        # The trusted parent creates the script; the payload only needs to read
        # it. No model/data/repository directories are mounted into this image.
        Path(cwd).chmod(0o755)
        launcher = Path(cwd) / ".retool_python.py"
        launcher.write_bytes(Path("/opt/retool_python.py").read_bytes())
        for path in Path(cwd).iterdir():
            path.chmod(0o644)
        at = command.index("chroot")
        # Mount proc where the payload will see it after chroot, inside the new
        # PID namespace rather than exposing the service's process namespace.
        command[command.index("--mount-proc")] = f"--mount-proc={command[at + 1]}/proc"
        # Apptainer gives /tmp a separate mount. Overlay lowerdir=/ does not
        # include submounts, so expose only this request's fresh script directory.
        script_dir = Path(command[at + 1]) / cwd.lstrip("/")
        script_dir.mkdir(parents=True, exist_ok=True)
        await execute_command(["mount", "--bind", cwd, str(script_dir)])
        command = [
            *command[:at],
            "setpriv",
            "--no-new-privs",
            "chroot",
            "--userspec=65534:65534",
            command[at + 1],
            "prlimit",
            "--as=1073741824:1073741824",
            "--data=1073741824:1073741824",
            "--nproc=0:0",
            "--cpu=30:31",
            "--nofile=64:64",
            "--fsize=262144:262144",
            *command[at + 2 :],
        ]
        # exec replaces the shell without requiring a child under RLIMIT_NPROC.
        command[-1] = command[-1].replace(" && python ", f" && exec python {shlex.quote(str(launcher))} ", 1)
        try:
            await execute_command(["mount", "-o", "remount,bind,ro", str(script_dir)])
            return await original_command(command, timeout, stdin, cwd, extra_env, use_exec)
        finally:
            await execute_command(["umount", str(script_dir)])

    base.run_command_bare = restricted_command
    app = FastAPI()
    slots = asyncio.Semaphore(32)

    @app.get("/health")
    async def health() -> dict:
        return dict(backend="apptainer-sandboxfusion-lite", concurrency=32)

    @app.post("/run_code")
    async def execute(request: Request) -> Any:
        body = await request.json()
        if (
            body.get("language") != "python"
            or body.get("run_timeout") != 30
            or body.get("memory_limit_MB") != 1024
            or body.get("files")
            or body.get("fetch_files")
            or body.get("stdin")
            or not isinstance(body.get("code"), str)
            or len(body["code"].encode()) > 65536
        ):
            raise HTTPException(400, "Require stateless Python, 30 seconds, 1 GiB, and no file transfer")
        async with slots:
            return await run_code(RunCodeRequest(**body))

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
