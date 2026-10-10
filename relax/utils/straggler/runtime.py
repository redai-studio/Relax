# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Attach the timer shim and publish one window of straggler metrics."""

from __future__ import annotations

import gc

import torch.distributed as dist

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import JudgeState, judge, metrics_from, rows_from_payload
from relax.utils.straggler.stages import STAGES
from relax.utils.straggler.timer_shim import get_nonblocking_timers


logger = get_logger(__name__)

_state = JudgeState()
_rollouts = 0
_gloo = None
_gc_ms = 0.0
_gc_hook = None
_window = {name: 0.0 for name in STAGES}
_window_tokens: float | None = None


def attach_timers(config, args, *, role: str = "actor") -> None:
    """Install non-blocking timers on the actor, or leave ``config.timers`` as
    ``None``."""
    cfg = StragglerConfig.from_args(args)
    if role != "actor" or not cfg.enabled:
        config.timers = None
        return
    timers = get_nonblocking_timers(cfg.enable_module_stages, cfg.max_pending_events)
    config.timers = timers
    _ensure_gc_hook()


def on_rollout_end(args, rollout_id: int, tokens: float | None = None) -> None:
    """Drain completed events and, every ``interval`` rollouts, judge the
    window.

    Called from every rank that runs ``train``. Failures are logged and
    swallowed.
    """
    cfg = StragglerConfig.from_args(args)
    if not cfg.enabled:
        return
    try:
        _on_rollout_end(cfg, rollout_id, tokens)
    except Exception as exc:
        logger.warning("straggler window skipped: %s", exc)


def _on_rollout_end(cfg: StragglerConfig, rollout_id: int, tokens: float | None) -> None:
    global _rollouts, _gc_ms, _window_tokens
    timers = get_nonblocking_timers(cfg.enable_module_stages, cfg.max_pending_events)
    for name, value in timers.drain().items():
        _window[name] = _window.get(name, 0.0) + float(value)
    if tokens is not None:
        _window_tokens = (0.0 if _window_tokens is None else _window_tokens) + float(tokens)
    _rollouts += 1
    if _rollouts % max(cfg.interval, 1) != 0:
        return
    payload = {
        "rank": _rank(),
        "pp": _pp_rank(),
        "tokens": _window_tokens,
        "gc_ms": _gc_ms,
        **_window,
    }
    _gc_ms = 0.0
    for name in STAGES:
        _window[name] = 0.0
    _window_tokens = None
    gathered = _gather(payload)
    if not gathered or _rank() != 0:
        return
    rows = [rows_from_payload(item) for item in gathered if isinstance(item, dict)]
    alerts, _ = judge(rows, cfg, _state)
    _log(rollout_id, metrics_from(rows, alerts, timers.dropped_events), alerts)


def _gather(payload: dict):
    if not dist.is_available() or not dist.is_initialized():
        return [payload]
    group = _gloo_group()
    if group is None:
        return [payload]
    world = dist.get_world_size()
    output = [None] * world
    dist.all_gather_object(output, _plain(payload), group=group)
    return output


def _plain(payload: dict) -> dict:
    plain = {"rank": int(payload["rank"]), "pp": int(payload["pp"]), "gc_ms": float(payload["gc_ms"])}
    tokens = payload.get("tokens")
    plain["tokens"] = None if tokens is None else float(tokens)
    for name in STAGES:
        plain[name] = float(payload.get(name, 0.0))
    return plain


def _gloo_group():
    global _gloo
    if _gloo is not None:
        return _gloo
    try:
        _gloo = dist.new_group(backend="gloo")
    except Exception as exc:
        logger.warning("straggler gloo group unavailable: %s", exc)
        return None
    return _gloo


def _log(rollout_id: int, metrics: dict, alerts) -> None:
    line = ", ".join(f"rank {item.rank} {item.reason}" for item in alerts) or "none"
    logger.info("straggler rollout %s: %s", rollout_id, line)
    try:
        from megatron.training.global_vars import get_args

        from relax.utils import tracking_utils

        args = get_args()
        payload = dict(metrics)
        payload["rollout/step"] = rollout_id
        tracking_utils.log(args, payload, step_key="rollout/step")
    except Exception as exc:
        logger.warning("straggler metrics were not exported: %s", exc)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _pp_rank() -> int:
    try:
        from megatron.core import mpu

        return int(mpu.get_pipeline_model_parallel_rank())
    except Exception:
        return 0


def _ensure_gc_hook() -> None:
    global _gc_hook
    if _gc_hook is not None:
        return

    def _hook(phase, info):
        global _gc_ms
        del info
        if phase == "start":
            _hook.t0 = __import__("time").perf_counter()
        elif phase == "stop" and getattr(_hook, "t0", None) is not None:
            _gc_ms += (__import__("time").perf_counter() - _hook.t0) * 1000.0

    _gc_hook = _hook
    if hasattr(gc, "callbacks"):
        gc.callbacks.append(_hook)


def reset_runtime_for_tests() -> None:
    global _state, _rollouts, _gloo, _gc_ms, _gc_hook, _window, _window_tokens
    _state = JudgeState()
    _rollouts = 0
    _gloo = None
    _gc_ms = 0.0
    _window = {name: 0.0 for name in STAGES}
    _window_tokens = None
    if _gc_hook is not None and hasattr(gc, "callbacks"):
        try:
            gc.callbacks.remove(_gc_hook)
        except ValueError:
            pass
    _gc_hook = None
