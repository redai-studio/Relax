# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld environment adapter.

Thin async facade over the nex-OSWorld `desktop_env.DesktopEnv`:
- Owns exactly one nex sandbox (via `provider_name="nex"`) for the session.
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
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image


# Make the vendored trees importable without polluting site-packages.
_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"
for _p in (_VENDOR_ROOT, _VENDOR_ROOT / "agents"):
    p_str = str(_p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# desktop_env is NOT vendored: it comes from the nex-enabled OSWorld checkout
# (providers/nex + MCP-aware DesktopEnv). Appended so vendor mm_agents still wins.
_OSWORLD_REPO = os.environ.get("OSWORLD_REPO", "")
if not _OSWORLD_REPO:
    raise RuntimeError("OSWORLD_REPO must point at the OSWorld-main-nex checkout")
if _OSWORLD_REPO not in sys.path:
    sys.path.append(_OSWORLD_REPO)

from desktop_env.desktop_env import DesktopEnv  # noqa: E402


logger = logging.getLogger("osworld_agentic.env")

_RESET_ATTEMPTS = 3
_RESET_BACKOFF_S = 5.0


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
        provider_name: str = "nex",
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
        self.cache_dir = cache_dir or os.environ.get("OSWORLD_CACHE_DIR", "/tmp/osworld_cache")
        self._env: Optional[DesktopEnv] = None
        self._lock = asyncio.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────

    async def reset(self) -> StepObs:
        """Boot the VM (first call) and reset it to the task's initial
        state."""
        if getattr(self, "_env_slot_fd", None) is None:
            if _ENV_GROUP_LIMIT > 0:
                self._env_slot_fd = await _acquire_env_group_slot_async(_task_key(self.task_config))
            else:
                self._env_slot_fd = await _acquire_env_slot_async()
        async with self._lock:
            return await asyncio.to_thread(self._reset_sync)

    def _reset_sync(self) -> StepObs:
        # OSWorld setup regularly hits transient guest failures (HTTP 500 on the
        # file upload, sandbox-creation lock timeouts). Retry on a fresh sandbox:
        # a session that dies in here never reports back to Relax and deadlocks
        # the whole rollout batch. See PITFALLS.md section 12.
        last_error: Exception | None = None
        for attempt in range(_RESET_ATTEMPTS):
            try:
                return self._reset_once()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "env reset attempt %d/%d failed: %s",
                    attempt + 1,
                    _RESET_ATTEMPTS,
                    exc,
                )
                self._close_sync()
                time.sleep(_RESET_BACKOFF_S)
        raise RuntimeError(f"env reset failed after {_RESET_ATTEMPTS} attempts: {last_error}")

    def _reset_once(self) -> StepObs:
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
                enable_proxy=True,
            )
        obs = self._env.reset(task_config=self.task_config)
        return self._pack_obs(obs, done=False, exec_ok=True, exec_result={})

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        try:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    logger.exception("DesktopEnv.close raised")
                self._env = None
        finally:
            _release_env_slot(getattr(self, "_env_slot_fd", None))
            self._env_slot_fd = None

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
    """Parse ``MCP:<tool>:<json>`` produced by HybridAgentLocal into (name,
    params).

    Returns ``None`` if the string is not an MCP action.
    """
    if not isinstance(action, str) or not action.startswith("MCP:"):
        return None
    body = action[len("MCP:") :]
    tool, _, arg_json = body.partition(":")
    try:
        params = json.loads(arg_json) if arg_json else {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse MCP arg JSON: %s", arg_json[:200])
        params = {}
    return tool, params


# ---------------------------------------------------------------------------
# [env_gate] Cross-process concurrency gate for nex sandboxes.
# N slot files under /dev/shm; holding an flock on one file = holding a slot.
# The kernel drops a flock automatically when its process dies, so a killed
# run can never leak slots. Tune with OSWORLD_MAX_CONCURRENT_ENVS (0 = off).
# Queued sessions wait on the event loop (asyncio.sleep), not in the
# to_thread pool, so active sessions are never starved of threads.
# ---------------------------------------------------------------------------
import fcntl as _fcntl
import os as _os
import random as _random


_ENV_SLOT_LIMIT = int(_os.environ.get("OSWORLD_MAX_CONCURRENT_ENVS", "0"))
_ENV_SLOT_DIR = _os.environ.get("OSWORLD_ENV_SLOT_DIR", "/dev/shm/osworld_env_slots")


async def _acquire_env_slot_async():
    if _ENV_SLOT_LIMIT <= 0:
        return None
    _os.makedirs(_ENV_SLOT_DIR, exist_ok=True)
    waited = 0.0
    last_log = 0.0
    while True:
        for i in range(_ENV_SLOT_LIMIT):
            fd = _os.open(_os.path.join(_ENV_SLOT_DIR, "slot_%04d" % i), _os.O_CREAT | _os.O_RDWR, 0o666)
            try:
                _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except OSError:
                _os.close(fd)
                continue
            if waited > 0:
                logger.info("[env_gate] got slot %d after %.0fs in queue (limit=%d)", i, waited, _ENV_SLOT_LIMIT)
            return fd
        step = 5.0 + 5.0 * _random.random()
        await asyncio.sleep(step)
        waited += step
        if waited - last_log >= 120:
            last_log = waited
            logger.info("[env_gate] all %d env slots busy, session queued %.0fs", _ENV_SLOT_LIMIT, waited)


def _release_env_slot(fd):
    if fd is None:
        return
    try:
        _os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# [env_gate] Group-aware variant. Relax leases a group to the runtime only
# after ALL of its sessions have issued their first LLM request
# (SessionGroup.first_request_barrier_crossed). A per-session gate therefore
# deadlocks: slot holders wait for siblings that are queued behind the gate.
# Here sessions of the same task share ONE group slot (LOCK_SH on the task's
# lock file) and at most OSWORLD_MAX_CONCURRENT_GROUPS distinct tasks may hold
# a slot at once. A global LOCK_EX mutex makes count-and-acquire atomic across
# the shard processes. Kernel drops every flock when a process dies.
# ---------------------------------------------------------------------------
import hashlib as _hashlib
import json as _json

_ENV_GROUP_LIMIT = int(_os.environ.get("OSWORLD_MAX_CONCURRENT_GROUPS", "0"))


def _task_key(task_config):
    tid = ""
    try:
        tid = str((task_config or {}).get("id") or "")
    except Exception:
        tid = ""
    if not tid:
        tid = _hashlib.md5(_json.dumps(task_config, sort_keys=True, default=str).encode()).hexdigest()
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tid)[:80]


def _flock_is_held(path):
    fd = _os.open(path, _os.O_CREAT | _os.O_RDWR, 0o666)
    try:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            return True
        _fcntl.flock(fd, _fcntl.LOCK_UN)
        return False
    finally:
        _os.close(fd)


async def _acquire_env_group_slot_async(task_key):
    _os.makedirs(_ENV_SLOT_DIR, exist_ok=True)
    mutex_path = _os.path.join(_ENV_SLOT_DIR, "_groups.mutex")
    my_name = "task_%s.lock" % task_key
    my_path = _os.path.join(_ENV_SLOT_DIR, my_name)
    waited = 0.0
    last_log = 0.0
    while True:
        mfd = _os.open(mutex_path, _os.O_CREAT | _os.O_RDWR, 0o666)
        try:
            _fcntl.flock(mfd, _fcntl.LOCK_EX)
            if _flock_is_held(my_path):
                fd = _os.open(my_path, _os.O_CREAT | _os.O_RDWR, 0o666)
                _fcntl.flock(fd, _fcntl.LOCK_SH)
                return fd
            busy = 0
            for name in _os.listdir(_ENV_SLOT_DIR):
                if name.startswith("task_") and name != my_name and _flock_is_held(_os.path.join(_ENV_SLOT_DIR, name)):
                    busy += 1
            if busy < _ENV_GROUP_LIMIT:
                fd = _os.open(my_path, _os.O_CREAT | _os.O_RDWR, 0o666)
                _fcntl.flock(fd, _fcntl.LOCK_SH)
                logger.info("[env_gate] group slot %d/%d -> task %s (queued %.0fs)", busy + 1, _ENV_GROUP_LIMIT, task_key[:12], waited)
                return fd
        finally:
            _os.close(mfd)
        step = 8.0 + 7.0 * _random.random()
        await asyncio.sleep(step)
        waited += step
        if waited - last_log >= 300:
            last_log = waited
            logger.info("[env_gate] all %d group slots busy, task %s queued %.0fs", _ENV_GROUP_LIMIT, task_key[:12], waited)
