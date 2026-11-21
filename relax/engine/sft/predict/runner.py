# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT periodic predict driver.

Two halves, both extracted from existing inline code:

* ``run_sft_predict(actor, rollout_id)`` — the Megatron-actor-side cycle:
  sleep → update_weights (NCCL) → HTTP GET /predict → wake_up, with perf
  timings logged to the tracking backend. Previously lived inside
  ``MegatronTrainRayActor.train_actor``.
* ``handle_predict(rollout, train_step)`` — the Rollout-side handler that
  drives ``rollout_manager.run_predict`` and ensures a post-predict full
  offload so the next cycle's onload paths see a clean slate. Previously
  lived inside ``components/rollout.py::predict``.

The two halves are bound together by a single HTTP call so they're packaged
in the same module.
"""

import time
from typing import Any

import requests
import torch.distributed as dist

from relax.engine.sft.runtime import is_sft_mode
from relax.utils import tracking_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.utils import get_serve_url


logger = get_logger(__name__)


def run_sft_predict(actor, rollout_id: int, *, published_version: dict | None = None) -> None:
    """SFT predict cycle driven from the Megatron actor.

    Use the weights published by this training step when available. Otherwise
    update the rollout engines before requesting prediction. Colocated actors
    release GPU memory around the prediction pass.
    """
    from relax.backends.megatron.initialize import is_megatron_main_rank

    args = actor.args
    scoring = bool(getattr(args, "custom_rm_path", None) or getattr(args, "rm_type", None))
    predict_error = None
    # Cache the topology-derived role while Megatron process groups are live.
    # sleep() and update_weights() intentionally destroy the reloadable NCCL
    # groups until wake_up(), so querying mpu ranks during predict would access
    # a ReloadableProcessGroup whose inner group is None.
    is_main_rank = is_megatron_main_rank()
    dist.barrier(group=get_gloo_group())
    _t_predict_start = time.monotonic()
    _t_sleep = _t_update = _t_http = _t_wake = 0.0
    if args.offload_train:
        _t = time.monotonic()
        actor.sleep()
        _t_sleep = time.monotonic() - _t
    _t = time.monotonic()
    if published_version is None:
        published_version = actor.update_weights()
    _t_update = time.monotonic() - _t
    dist.barrier(group=get_gloo_group())
    needs_release = args.offload_train and getattr(args, "offload_rollout", False)
    release_unconfirmed = is_main_rank and needs_release
    require_prediction = scoring or published_version is not None
    if is_main_rank:
        _t = time.monotonic()
        try:
            # Bound the time other training ranks wait for the prediction pass.
            params = {"train_step": rollout_id}
            if published_version is not None:
                params.update(weight_version=published_version["version"], weight_root=published_version["root"])
            response = requests.get(
                f"{get_serve_url('rollout')}/predict",
                params=params,
                timeout=1800,
            )
            response.raise_for_status()
            if require_prediction or needs_release:
                result = response.json()
                release_unconfirmed = needs_release and result.get("resources_released") is not True
                if require_prediction and result.get("status") != "ok":
                    raise RuntimeError(result.get("message", "SFT prediction failed"))
                if release_unconfirmed:
                    raise RuntimeError(result.get("message", "Rollout did not confirm GPU release after prediction"))
        except Exception as e:
            logger.warning(f"SFT predict at rollout_id {rollout_id} failed: {e}")
            predict_error = str(e)
        _t_http = time.monotonic() - _t
    if require_prediction or needs_release:
        outcomes = [None] * dist.get_world_size(group=get_gloo_group())
        dist.all_gather_object(outcomes, (predict_error, release_unconfirmed), group=get_gloo_group())
        predict_error = next((error for error, _ in outcomes if error is not None), None)
        release_unconfirmed = any(held for _, held in outcomes)
    dist.barrier(group=get_gloo_group())
    if args.offload_train and not release_unconfirmed:
        _t = time.monotonic()
        actor.wake_up()
        _t_wake = time.monotonic() - _t
    if release_unconfirmed:
        raise RuntimeError(f"SFT prediction failed at step {rollout_id}; training remains offloaded: {predict_error}")
    if require_prediction and predict_error is not None:
        raise RuntimeError(f"SFT prediction failed at step {rollout_id}: {predict_error}")
    if is_main_rank:
        step = compute_rollout_step(args, rollout_id)
        metrics = {
            "perf/sft_predict_time": time.monotonic() - _t_predict_start,
            "perf/sft_predict_sleep_time": _t_sleep,
            "perf/sft_predict_update_weights_time": _t_update,
            "perf/sft_predict_http_time": _t_http,
            "perf/sft_predict_wake_up_time": _t_wake,
            "rollout/step": step,
        }
        tracking_utils.log(args, metrics, step_key="rollout/step")
        tracking_utils.flush_metrics(args, step)
        logger.info(f"SFT predict @ rollout_id={rollout_id}: {metrics}")


async def handle_predict(
    rollout, train_step: int, weight_version: int | None = None, weight_root: str | None = None
) -> dict[str, Any]:
    """Rollout-side handler for the /predict HTTP endpoint.

    Awaits ``rollout_manager.run_predict`` (renders + generates + writes
    predictions JSONL) then unconditionally re-offloads on SFT so the next
    predict cycle's ``onload_weights`` / ``onload_kv`` resume cleanly.

    Returns the JSON payload the FastAPI handler should send back.
    """
    rollout._logger.info(f"Received request to predict train_step {train_step}")
    errors = []
    released = False
    try:
        if weight_version is None and weight_root is None:
            await rollout.rollout_manager.run_predict.remote(train_step)
        else:
            await rollout.rollout_manager.run_predict.remote(train_step, weight_version, weight_root)
    except Exception as exc:
        message = f"Predict failed for train_step {train_step}: {type(exc).__name__}: {exc}"
        rollout._logger.exception(message)
        errors.append(message)
    finally:
        if is_sft_mode(rollout.config) and rollout.config.offload_rollout:
            from .loop import managed_predict_judge

            released = True
            try:
                judge = managed_predict_judge(rollout.config)
                if judge is not None:
                    await judge.offload.remote()
            except Exception as exc:
                errors.append(f"SFT judge offload failed: {exc}")
                released = False
            try:
                await rollout.rollout_manager.offload.remote()
            except Exception as exc:
                errors.append(f"Post-predict offload failed at step {train_step}: {exc}")
                released = False
    if errors:
        message = "; ".join(errors)
        rollout._logger.warning(message)
        rollout.healthy.report_error.remote("rollout", message)
        return {"status": "error", "message": message, "resources_released": released}
    return {"status": "ok", "rollout_id": train_step, "resources_released": released}
