# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld environment adapter.

Thin async facade over the vendored `desktop_env.DesktopEnv`:
- Owns exactly one Docker VM (via `provider_name="docker"`) for the session.
- Runs blocking VM RPC calls in a worker thread so the asyncio event loop
  stays responsive.
- Exposes primitives the session driver needs: `reset`, `exec_pyautogui`,
  `exec_mcp`, `evaluate`, `close`.

For P0/P1 the VM lifecycle is session-internal (one container per session).
P2+ will swap this for `vm_pool_client.py` that acquires a pre-warmed VM
from an external gateway.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image

# Make the vendored trees importable without polluting site-packages.
_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"
for _p in (_VENDOR_ROOT, _VENDOR_ROOT / "desktop_env", _VENDOR_ROOT / "mm_agents",
           _VENDOR_ROOT / "mcp_tools", _VENDOR_ROOT / "agents"):
    p_str = str(_p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from desktop_env.desktop_env import DesktopEnv  # noqa: E402

logger = logging.getLogger("osworld_agentic.env")


@dataclass
class StepObs:
    """Post-step observation surfaced back to the agent driver."""
    screenshot_bytes: bytes
    cur_app: Optional[str] = None
    app_info: Optional[str] = None
    tool_list: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    exec_ok: bool = True
    exec_result: dict[str, Any] = field(default_factory=dict)


class OSWorldEnv:
    """One VM, one task, one session."""

    def __init__(
        self,
        *,
        task_config: dict[str, Any],
        provider_name: str = "docker",
        snapshot_name: str = "init_state",
        screen_width: int = 1920,
        screen_height: int = 1080,
        headless: bool = True,
        require_a11y_tree: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.task_config = task_config
        self.provider_name = provider_name
        self.snapshot_name = snapshot_name
        self.screen_size = (screen_width, screen_height)
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.cache_dir = cache_dir or os.environ.get(
            "OSWORLD_CACHE_DIR", "/tmp/osworld_cache"
        )
        self._env: Optional[DesktopEnv] = None
        self._lock = asyncio.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────

    async def reset(self) -> StepObs:
        """Boot the VM (first call) and reset it to the task's initial state."""
        async with self._lock:
            return await asyncio.to_thread(self._reset_sync)

    def _reset_sync(self) -> StepObs:
        if self._env is None:
            logger.info("Instantiating DesktopEnv (provider=%s)", self.provider_name)
            self._env = DesktopEnv(
                provider_name=self.provider_name,
                snapshot_name=self.snapshot_name,
                action_space="pyautogui",
                cache_dir=self.cache_dir,
                screen_size=self.screen_size,
                headless=self.headless,
                require_a11y_tree=self.require_a11y_tree,
                os_type="Ubuntu",
            )
        obs = self._env.reset(task_config=self.task_config)
        return self._pack_obs(obs, done=False, exec_ok=True, exec_result={})

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception:
            logger.exception("DesktopEnv.close raised")
        self._env = None

    # ── actions ──────────────────────────────────────────────────────────

    async def exec_pyautogui(self, code: str) -> StepObs:
        """Run a pyautogui code snippet (or WAIT/DONE/FAIL) in the VM."""
        async with self._lock:
            return await asyncio.to_thread(self._exec_pyautogui_sync, code)

    def _exec_pyautogui_sync(self, code: str) -> StepObs:
        assert self._env is not None, "reset() must be called before exec_*"
        obs, _reward, done, info = self._env.step(code)
        return self._pack_obs(obs, done=done, exec_ok=True, exec_result={"info": info})

    async def exec_mcp(self, tool_name: str, params: dict[str, Any]) -> StepObs:
        async with self._lock:
            return await asyncio.to_thread(self._exec_mcp_sync, tool_name, params)

    def _exec_mcp_sync(self, tool_name: str, params: dict[str, Any]) -> StepObs:
        assert self._env is not None, "reset() must be called before exec_*"
        exec_ok = True
        raw_result = ""
        try:
            raw_result = self._env.call_mcp_tool(tool_name, params)
            # Heuristic: `is_error=True` in the raw CallToolResult string means the
            # MCP tool itself reported an error even though the RPC succeeded.
            if isinstance(raw_result, str) and "is_error=True" in raw_result:
                exec_ok = False
        except Exception as exc:  # noqa: BLE001
            exec_ok = False
            raw_result = f"exception: {exc}"
            logger.warning("call_mcp_tool(%s) raised: %s", tool_name, exc)

        exec_result = {
            "ok": exec_ok,
            "result": raw_result if exec_ok else "",
            "error": "" if exec_ok else raw_result,
        }
        # MCP tools don't advance the VM step counter; re-fetch a fresh obs so
        # the model sees any visual side effects (or their absence).
        obs = self._env._get_obs()  # noqa: SLF001 (verbatim upstream API)
        return self._pack_obs(obs, done=False, exec_ok=exec_ok, exec_result=exec_result)

    async def evaluate(self) -> float:
        async with self._lock:
            return await asyncio.to_thread(self._evaluate_sync)

    def _evaluate_sync(self) -> float:
        assert self._env is not None, "reset() must be called before evaluate()"
        try:
            return float(self._env.evaluate())
        except Exception:
            logger.exception("evaluate() raised; returning 0.0")
            return 0.0

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pack_obs(
        obs: dict[str, Any],
        *,
        done: bool,
        exec_ok: bool,
        exec_result: dict[str, Any],
    ) -> StepObs:
        screenshot = obs.get("screenshot")
        if not isinstance(screenshot, (bytes, bytearray)):
            # Some code paths hand back a base64 string; normalise to bytes.
            if isinstance(screenshot, str):
                screenshot = base64.b64decode(screenshot)
            else:
                # Last resort: an empty PNG so the agent can still parse an image.
                buf = BytesIO()
                Image.new("RGB", (16, 16), (0, 0, 0)).save(buf, format="PNG")
                screenshot = buf.getvalue()
        return StepObs(
            screenshot_bytes=bytes(screenshot),
            cur_app=obs.get("cur_app"),
            app_info=obs.get("app_info"),
            tool_list=obs.get("tool_list") or [],
            done=done,
            exec_ok=exec_ok,
            exec_result=exec_result,
        )


def parse_mcp_action(action: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Parse ``MCP:<tool>:<json>`` produced by HybridAgentLocal into (name, params).

    Returns ``None`` if the string is not an MCP action.
    """
    if not isinstance(action, str) or not action.startswith("MCP:"):
        return None
    body = action[len("MCP:"):]
    tool, _, arg_json = body.partition(":")
    try:
        params = json.loads(arg_json) if arg_json else {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse MCP arg JSON: %s", arg_json[:200])
        params = {}
    return tool, params
