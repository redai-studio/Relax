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

from relax.utils import tracking_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.utils import get_serve_url


logger = get_logger(__name__)


def run_sft_predict(actor, rollout_id: int) -> None:
    """SFT predict cycle driven from the Megatron actor.

    Megatron sleep → update_weights (NCCL) → /predict (SGLang onload_kv +
    generate + finally full offload) → wake_up. Timings are logged so we can
    attribute predict-step latency.
    """
    from relax.backends.megatron.initialize import is_megatron_main_rank

    args = actor.args
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
    actor.update_weights()
    _t_update = time.monotonic() - _t
    dist.barrier(group=get_gloo_group())
    if is_main_rank:
        _t = time.monotonic()
        try:
            # Without a timeout, a hung rollout service blocks this rank
            # forever and the Gloo barrier on line 65 traps every other rank
            # until the much-larger NCCL/Gloo watchdog fires — all GPUs idle.
            # 1800s is generous enough for a full eval-set generate; on
            # timeout the except below logs and we proceed (predict skipped).
            response = requests.get(
                f"{get_serve_url('rollout')}/predict",
                params={"train_step": rollout_id},
                timeout=1800,
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"SFT predict at rollout_id {rollout_id} failed: {e}")
        _t_http = time.monotonic() - _t
    dist.barrier(group=get_gloo_group())
    if args.offload_train:
        _t = time.monotonic()
        actor.wake_up()
        _t_wake = time.monotonic() - _t
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


async def handle_predict(rollout, train_step: int) -> dict[str, Any]:
    """Rollout-side handler for the /predict HTTP endpoint.

    Awaits ``rollout_manager.run_predict`` (renders + generates + writes
    predictions JSONL) then unconditionally re-offloads on SFT so the next
    predict cycle's ``onload_weights`` / ``onload_kv`` resume cleanly.

    Returns the JSON payload the FastAPI handler should send back.
    """
    rollout._logger.info(f"Received request to predict train_step {train_step}")
    try:
        await rollout.rollout_manager.run_predict.remote(train_step)
        return {"status": "ok", "rollout_id": train_step}
    except Exception as e:
        error_msg = f"Predict failed for train_step {train_step}: {type(e).__name__}: {str(e)}"
        rollout._logger.exception(error_msg)
        rollout.healthy.report_error.remote("rollout", error_msg)
        return {"status": "error", "message": error_msg}
    finally:
        # Mirror RL's per-cycle reset: full no-tag offload after each
        # predict puts all 3 tags back in `offload_tags`, so the next
        # predict cycle's `update_weights → onload_weights` and
        # `onload_kv` both resume cleanly. SFT's per-step `update_weights`
        # is gated on `_should_run_sft_predict`, so this offload only
        # ever fires on predict steps. Must run even on exception —
        # leaking KV/weights would starve the next training step's GPU
        # memory. RL: rollout manages its own offload in `_async_run`,
        # skip here.
        if getattr(rollout.config, "loss_type", None) == "sft" and rollout.config.offload_rollout:
            try:
                await rollout.rollout_manager.offload.remote()
            except Exception as exc:
                rollout._logger.warning(f"Post-predict offload failed at step {train_step}: {exc}")
