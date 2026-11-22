# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Own the local tool service for the lifetime of a single-node ReTool Ray
job."""

import os
import signal
import subprocess
import sys
from contextlib import nullcontext

from examples.retool_agentic.tool_service import tool_service


def main() -> None:
    child = None

    def stopped(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, stopped) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        external_url = os.environ.get("SANDBOX_FUSION_URL")
        # Keep explicitly supplied external services available after this job.
        with nullcontext(external_url) if external_url else tool_service() as url:
            env = dict(os.environ, SANDBOX_FUSION_URL=url)
            names = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
            env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
                dict.fromkeys([*filter(None, names), "SANDBOX_FUSION_URL", "DATA_DIR"])
            )
            # train.py uses os._exit on both success and failure. Keep this
            # parent alive so its finally block can close the tool service.
            try:
                child = subprocess.Popen([sys.executable, "-m", "relax.entrypoints.train", *sys.argv[1:]], env=env)
                code = child.wait()
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            raise SystemExit(code if code >= 0 else 128 - code)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
