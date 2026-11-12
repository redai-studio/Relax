# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SandboxFusion client.

Execute each new snippet once, without retries.
"""

import asyncio
import json
import urllib.request
from urllib.parse import urlsplit

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def parse_result(value: dict) -> dict[str, str]:
    if value.get("status") not in {"Success", "Failed"}:
        raise RuntimeError("Sandbox infrastructure failure")
    result = value.get("run_result") or value.get("compile_result") or {}
    status = result.get("status")
    if status not in {"Finished", "Error", "TimeLimitExceeded"}:
        raise RuntimeError("Sandbox returned no recognized execution result")
    # SandboxFusion's Finished means the process exited, including Python
    # exceptions. The Landlock-compatible service omits return_code.
    if status == "Finished":
        return_code = result.get("return_code", 0)
        if type(return_code) is not int:
            raise RuntimeError(f"Invalid sandbox process exit code: {return_code}")
        if return_code:
            status = "Error"
            # A completed isolated payload can crash under its resource limits.
            # This is tool feedback, while transport/service failures still raise.
            if return_code < 0 and not result.get("stderr"):
                result = dict(result, stderr=f"Python program terminated by signal {-return_code}")
    return {
        "status": {"Finished": "ok", "Error": "code_error", "TimeLimitExceeded": "timeout"}[status],
        "stdout": result.get("stdout") or "",
        "stderr": result.get("stderr") or "",
    }


class SandboxClient:
    def __init__(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Configure an explicit SandboxFusion HTTP(S) URL without credentials")
        self.url = url

    def _request(self, code: str, *, timeout: float | None = 70) -> dict[str, str]:
        payload = dict(
            language="python",
            code=code,
            compile_timeout=30,
            run_timeout=30,
            memory_limit_MB=1024,
            stdin=None,
            files={},
            fetch_files=[],
        )
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise RuntimeError("Sandbox response exceeds 4 MiB")
        value = json.loads(body)
        try:
            return parse_result(value)
        except RuntimeError:
            logger.exception("Sandbox execution failed: code=%r response=%r", code, value)
            raise

    async def execute(self, code: str) -> dict[str, str]:
        # Queueing precedes the server's execution budget. The managed agent's
        # active-time budget bounds stuck calls; startup probes keep a timeout.
        return await asyncio.to_thread(self._request, code, timeout=None)

    async def close(self) -> None:
        pass
