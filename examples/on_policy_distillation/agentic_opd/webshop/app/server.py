# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cluster-shared WebShop environment server.

Why a server at all
-------------------
WebShop's ``SimServer`` is a several-GB in-memory asset (product catalog + Lucene
index) that must be loaded to run *any* session. ALFWorld's env is tiny, so the
ALFWorld recipe spins up one env per session process; doing that for WebShop would
cost ``N × several GB``. WebShop's ``WebAgentTextEnv(server=...)`` instead lets a
single ``SimServer`` be shared by many cheap ``SimBrowser`` sessions. One server
process serves all nodes in the training run, and every per-session agent
process talks to it over HTTP.

Session isolation
-----------------
``SimServer.user_sessions`` is keyed by a session id and stores the cart / current
page for that session. WebShop's own ``env.reset(session=X)`` overloads ``X`` as
BOTH the cart key AND (when it's an int) the goal selector. Under GRPO the same
``goal_idx`` is rolled out ``group_size`` times concurrently, so keying the cart by
``goal_idx`` would make those concurrent carts collide. We therefore decouple the
two using the lower-level ``browser.get(session_id=..., session_int=...)`` API:

* cart key   = ``instance_id`` (== ``RELAX_SESSION_ID``, globally unique)
* goal        = ``session_int`` = ``goal_idx``

``user_sessions`` grows unboundedly in upstream WebShop (never deletes), so
``/close`` MUST drop the entry or a long run leaks memory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - webshop.server - %(message)s")
logger = logging.getLogger("webshop.server")

_CONFIG_PATH = Path(__file__).with_name("config.yaml")


# --------------------------------------------------------------------------- #
# WebShop import (not a pip package; lives under $WEBSHOP_HOME).
# --------------------------------------------------------------------------- #
def _import_webshop():
    webshop_home = os.environ.get("WEBSHOP_HOME")
    if webshop_home and webshop_home not in sys.path:
        sys.path.insert(0, webshop_home)
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: WPS433

    return WebAgentTextEnv


class WebshopServerState:
    """Owns the one shared ``SimServer`` and the live per-instance envs."""

    def __init__(self, env_cfg: dict[str, Any], step_concurrency: int) -> None:
        self._env_cfg = env_cfg
        self._web_agent_text_env = _import_webshop()

        # Build the heavy shared SimServer once by constructing a throwaway env
        # (server=None path) and stealing its ``.server``. We avoid depending on
        # SimServer's positional signature this way.
        boot_kwargs = self._build_env_kwargs()
        logger.info("Loading WebShop catalog (this can take a while)...")
        boot_env = self._web_agent_text_env(**boot_kwargs)
        self.sim_server = boot_env.server
        # __init__ ran a throwaway random reset() -> stray user_sessions entry.
        self.sim_server.user_sessions.pop(boot_env.session, None)
        self.num_goals = len(self.sim_server.goals)
        logger.info("WebShop ready: %d goals loaded", self.num_goals)

        self._instances: dict[str, Any] = {}
        self._instances_lock = threading.Lock()
        self._step_sem = threading.Semaphore(max(1, int(step_concurrency)))
        self.ready = True

    def _build_env_kwargs(self, *, with_server: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "observation_mode": self._env_cfg.get("observation_mode", "text"),
            "num_products": self._env_cfg.get("num_products"),
            "human_goals": bool(self._env_cfg.get("human_goals", False)),
            "seed": int(self._env_cfg.get("seed", 42)),
            # Env returns only the current page; multi-turn history lives agent-side.
            "num_prev_obs": 0,
            "num_prev_actions": 0,
        }
        # Only override the catalog paths when explicitly set; otherwise WebShop's
        # DEFAULT_FILE_PATH / DEFAULT_ATTR_PATH (the 1000-item subset) are used.
        if self._env_cfg.get("file_path"):
            kwargs["file_path"] = self._env_cfg["file_path"]
        if self._env_cfg.get("attr_path"):
            kwargs["attr_path"] = self._env_cfg["attr_path"]
        if with_server:
            kwargs["server"] = self.sim_server
        return kwargs

    # ------------------------------------------------------------------ #
    def reset(self, instance_id: str, goal_idx: int) -> dict[str, Any]:
        env = self._web_agent_text_env(**self._build_env_kwargs(with_server=True))
        # The constructor's throwaway reset() left a stray random session key.
        self.sim_server.user_sessions.pop(env.session, None)

        # Decouple cart key (unique instance_id) from goal selection (goal_idx);
        # replicates WebAgentTextEnv.reset() but without overloading `session`.
        env.session = instance_id
        env.browser.get(f"{env.base_url}/{instance_id}", session_id=instance_id, session_int=int(goal_idx))
        env.text_to_clickable = None
        env.instruction_text = env.get_instruction_text()
        obs = env.observation
        available = self._available_actions(env)

        with self._instances_lock:
            # If this instance_id already existed (retry / crash), drop the old one.
            old = self._instances.pop(instance_id, None)
            if old is not None:
                self.sim_server.user_sessions.pop(instance_id, None)
            self._instances[instance_id] = env
        return {"instance_id": instance_id, "obs": obs, "available_actions": available}

    def step(self, instance_id: str, action: str) -> dict[str, Any]:
        with self._instances_lock:
            env = self._instances.get(instance_id)
        if env is None:
            raise KeyError(f"unknown instance_id: {instance_id}")

        # Cap concurrent WebShop CPU work (search is the hot path); different
        # instances step safely in parallel (keyed user_sessions writes + GIL).
        with self._step_sem:
            raw_obs, task_score, env_done, _info = env.step(action)
        available = self._available_actions(env)
        won = bool(env_done and float(task_score) == 1.0)
        return {
            "obs": raw_obs,
            "reward": float(task_score),
            "done": bool(env_done),
            "won": won,
            "task_score": float(task_score),
            "available_actions": available,
        }

    def close(self, instance_id: str) -> dict[str, Any]:
        with self._instances_lock:
            env = self._instances.pop(instance_id, None)
        # Drop the SimServer session entry (upstream never deletes -> leak) and
        # let the cheap SimBrowser be garbage-collected.
        self.sim_server.user_sessions.pop(instance_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001 - close is best-effort
                pass
        return {"ok": True}

    @staticmethod
    def _available_actions(env: Any) -> list[str]:
        from app.prompt import flatten_actions

        return flatten_actions(env.get_available_actions())


# --------------------------------------------------------------------------- #
# HTTP layer (stdlib; no FastAPI so we don't fight the webshop conda's pinned
# pydantic 1.x). ThreadingHTTPServer serves each request in its own thread.
# --------------------------------------------------------------------------- #
STATE: WebshopServerState | None = None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence per-request access logs
        pass

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") == "/health":
            ready = STATE is not None and STATE.ready
            self._send(200 if ready else 503, {"ready": bool(ready)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if STATE is None or not STATE.ready:
            self._send(503, {"error": "server not ready"})
            return
        route = self.path.rstrip("/")
        try:
            body = self._read_json()
            if route == "/reset":
                self._send(200, STATE.reset(str(body["instance_id"]), int(body["goal_idx"])))
            elif route == "/step":
                self._send(200, STATE.step(str(body["instance_id"]), str(body["action"])))
            elif route == "/close":
                self._send(200, STATE.close(str(body["instance_id"])))
            else:
                self._send(404, {"error": "not found"})
        except KeyError as exc:
            self._send(404, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface env errors to the agent
            logger.exception("request %s failed", route)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


class _Server(ThreadingHTTPServer):
    # stdlib default is 5: with a few hundred concurrent agent sessions all
    # opening a fresh connection around the same time (e.g. a rollout batch
    # starting together), that backlog overflows, the kernel drops the extra
    # SYNs, and clients see a 30s httpx.ConnectTimeout instead of a fast
    # connection. Size generously above the max expected concurrent sessions.
    request_queue_size = 1024


def _load_config() -> dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    global STATE
    cfg = _load_config()
    server_cfg = cfg.get("server", {})
    host = os.environ.get("WEBSHOP_HOST", server_cfg.get("host", "127.0.0.1"))
    port = int(os.environ.get("WEBSHOP_PORT", server_cfg.get("port", 36001)))
    step_concurrency = int(server_cfg.get("step_concurrency", 64))

    # Bind the port before loading the multi-GB catalog so duplicate launches
    # fail fast instead of wasting minutes loading a second copy. Serving runs
    # in a background thread so /health answers 503 (ready:false) during load,
    # then flips to ready:true once STATE is set.
    httpd = _Server((host, port), _Handler)
    httpd.daemon_threads = True
    server_thread = threading.Thread(target=httpd.serve_forever, name="webshop-http", daemon=True)
    server_thread.start()
    logger.info("Bound WebShop env on http://%s:%d; loading catalog...", host, port)
    try:
        STATE = WebshopServerState(cfg.get("env", {}), step_concurrency)
        logger.info("Serving WebShop env on http://%s:%d", host, port)
        server_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
