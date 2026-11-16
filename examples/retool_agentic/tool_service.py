# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Choose the local isolated tool backend without changing the rollout
protocol."""

import ctypes
import errno
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from examples.retool_agentic.landlock_service import local_service
from examples.retool_agentic.sandbox import SandboxClient
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
SANDBOX_IMAGE = "docker://vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609"


@contextmanager
def apptainer_service() -> Iterator[str]:
    if shutil.which("apptainer") is None:
        raise RuntimeError("This kernel lacks Landlock; install Apptainer or set SANDBOX_FUSION_URL")
    image = Path(
        os.environ.get("SANDBOX_IMAGE")
        or str(Path(os.environ["SAVE_DIR"]) / ".retool-sandbox/sandboxfusion-server-20250609.sif")
    )
    if not image.exists():
        if os.environ.get("SANDBOX_IMAGE"):
            raise FileNotFoundError(image)
        image.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["apptainer", "pull", str(image), SANDBOX_IMAGE], check=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    script = Path(__file__).parent / "scripts/run-sandbox-apptainer.sh"
    process = subprocess.Popen(
        ["bash", str(script)], env=dict(os.environ, SANDBOX_IMAGE=str(image), SANDBOX_PORT=str(port))
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError(f"Apptainer sandbox exited with code {process.returncode}")
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    health = json.load(response)
                if health.get("backend") != "apptainer-sandboxfusion-lite":
                    raise RuntimeError("Unexpected sandbox backend")
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        else:
            raise TimeoutError("Apptainer sandbox did not become ready")
        url = f"http://127.0.0.1:{port}/run_code"
        client = SandboxClient(url)
        probe = client._request(
            "import os, resource; print(os.getuid(), os.getgid(), "
            "resource.getrlimit(resource.RLIMIT_AS)[0], resource.getrlimit(resource.RLIMIT_NPROC)[0])"
        )
        if probe["status"] != "ok" or probe["stdout"].strip() != "65534 65534 1073741824 0":
            raise RuntimeError(f"Apptainer sandbox resource/uid validation failed: {probe}")
        numeric = client._request(
            "from scipy.optimize import linprog; "
            "r = linprog([-1, -2, -3], A_ub=[[2, 1, 1]], b_ub=[10], bounds=[(0, 5)] * 3, method='highs'); "
            "print(r.success, r.fun)"
        )
        if numeric["status"] != "ok" or numeric["stdout"].strip() != "True -25.0":
            raise RuntimeError(f"Apptainer sandbox single-thread HiGHS validation failed: {numeric}")
        with tempfile.NamedTemporaryFile(prefix="retool-host-canary-", dir="/tmp") as canary:
            os.chmod(canary.name, 0o644)
            forbidden = [
                f"open({canary.name!r}).read()",
                "import os; os.fork()",
                f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=1)",
            ]
            for code in forbidden:
                if client._request(code)["status"] != "code_error":
                    raise RuntimeError("Apptainer sandbox isolation validation failed")
        logger.info("Started job-owned Apptainer sandbox at %s", url)
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextmanager
def tool_service() -> Iterator[str]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(444, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1))
    if abi < 0 and ctypes.get_errno() == errno.ENOSYS:
        logger.info("Kernel does not implement Landlock; using the existing Apptainer sandbox backend")
        context = apptainer_service()
    else:
        context = local_service()
    with context as url:
        yield url
