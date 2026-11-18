# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Coordinate inference peers once per model version, outside training
ranks."""

import time
from argparse import Namespace
from typing import Any, Callable

import requests

from relax.distributed.settlement import wait_all
from relax.engine.sft.runtime import is_offline_mode
from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_serve_url


logger = get_logger(__name__)


def check_services_health(args: Namespace, *, managed: bool = False) -> tuple[bool, bool]:
    """Pause Rollout and select existing weight receivers for this version."""
    rollout_only = actor_fwd_only = False
    actor_fwd_absent = getattr(args, "true_on_policy_mode", False)
    # Check rollout service
    try:
        if managed:
            # Version publication owns rollout readiness. Do not enter
            # the legacy pause/recovery handshake.
            rollout_serve_url = None
        else:
            rollout_serve_url = get_serve_url("rollout")
        retry_deadline = None
        while rollout_serve_url is not None:
            request_timeout = args.rollout_http_timeout
            if retry_deadline is not None:
                request_timeout = retry_deadline - time.monotonic()
                if request_timeout <= 0:
                    raise requests.exceptions.Timeout("elastic scale-in fence did not clear")
            response = requests.get(
                f"{rollout_serve_url}/can_do_update_weight_for_async",
                timeout=request_timeout,
            )
            if getattr(response, "status_code", 200) == 503:
                if retry_deadline is None:
                    retry_deadline = time.monotonic() + max(float(args.rollout_http_timeout), 1.0)
                logger.warning("Elastic scale-in is draining; retrying before weight update.")
                time.sleep(min(1.0, max(0.0, retry_deadline - time.monotonic())))
                continue
            response.raise_for_status()
            # A successful non-503 response means the scale-in fence
            # has cleared. Do not let an earlier draining deadline
            # bound the normal readiness polling below.
            retry_deadline = None
            res = response.json()
            if res:
                if getattr(args, "use_fault_tolerance", False) and not getattr(args, "inference_recovery", False):
                    response = requests.get(f"{rollout_serve_url}/recover_rollout_engines")
                    response.raise_for_status()
                break
            else:
                time.sleep(1)
    except Exception as e:
        logger.warning(
            f"Error checking rollout service: {e}, maybe caused by rollout server failure. "
            "Will continue without rollout update for this step."
        )
        actor_fwd_only = True

    # Check actor_fwd service. Skip the probe when actor_fwd is
    # intentionally absent — either true_on_policy_mode (log_probs
    # recomputed inline by train forward) or hybrid mode (actor
    # handles forward via _switch_model, no separate service).
    if actor_fwd_absent or getattr(args, "hybrid", False):
        rollout_only = True
    else:
        try:
            actor_fwd_serve_url = get_serve_url("actor_fwd")
            response = requests.get(f"{actor_fwd_serve_url}/get_step", timeout=args.rollout_http_timeout)
            response.raise_for_status()
        except Exception as e:
            logger.warning(
                f"Error checking actor_fwd service: {e}, maybe caused by actor_fwd server failure. "
                "Will continue without actor_fwd update for this step."
            )
            rollout_only = True

    return rollout_only, actor_fwd_only


def wait_for_previous_eval(max_wait_seconds: int = 1800) -> None:
    """Wait for the existing Rollout evaluation before swapping its weights."""
    try:
        rollout_serve_url = get_serve_url("rollout")
        start_time = time.monotonic()
        while True:
            response = requests.get(f"{rollout_serve_url}/is_eval_done", timeout=10)
            response.raise_for_status()
            if response.json().get("done", True):
                break
            elapsed = time.monotonic() - start_time
            if elapsed >= max_wait_seconds:
                logger.warning(
                    f"Timed out waiting for previous evaluation after {elapsed:.1f}s "
                    f"(max_wait_seconds={max_wait_seconds}), proceeding with weight update."
                )
                break
            logger.info("Waiting for previous evaluation to finish before updating weights...")
            time.sleep(10)
    except Exception as e:
        logger.warning(f"Error checking eval status: {e}, proceeding with weight update.")


def _end_rollout_update(config: Namespace, step: int, *, managed: bool) -> None:
    if managed:
        return
    try:
        response = requests.get(f"{get_serve_url('rollout')}/end_update_weight", timeout=config.rollout_http_timeout)
        response.raise_for_status()
    except Exception as exc:
        logger.warning(f"Error ending update weight for rollout_id {step}: {exc}")


def _evaluate_rollout(config: Namespace, step: int, *, asynchronous: bool, managed: bool) -> None:
    try:
        # Generation can take much longer than a health probe.
        response = requests.get(f"{get_serve_url('rollout')}/evaluate", params={"train_step": step})
        response.raise_for_status()
    except Exception as exc:
        logger.warning(f"Error during actor post-train evaluation for rollout_id {step}: {exc}")
    finally:
        if asynchronous:
            _end_rollout_update(config, step, managed=managed)


def complete_inference_step(
    config: Namespace,
    train_group: Any,
    step: int,
    *,
    has_rollout: bool,
    managed: bool,
    settle: Callable[[list], list] = wait_all,
) -> None:
    """Coordinate the same post-training stages with or without recovery."""
    if is_offline_mode(config):
        settle(train_group.async_inference_call("finish_training_metrics", step))
        settle(train_group.async_inference_call("evaluate_after_training", step))
        return
    if not has_rollout or config.debug_train_only or config.debug_rollout_only:
        settle(train_group.async_inference_call("finish_training_metrics", step))
        return
    asynchronous = config.fully_async or config.hybrid
    if asynchronous:
        wait_for_previous_eval()
        started = time.monotonic()
        rollout_only, actor_fwd_only = check_services_health(config, managed=managed)
        settle(train_group.async_inference_call("record_inference_wait", time.monotonic() - started))
        if config.hybrid and actor_fwd_only:
            _end_rollout_update(config, step, managed=managed)
            settle(train_group.async_inference_call("finish_training_metrics", step))
            return
        if config.hybrid:
            refs = train_group.async_inference_call("update_weights")
        else:
            refs = train_group.async_inference_call(
                "update_weights_fully_async", step, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only
            )
    else:
        refs = train_group.async_inference_call("update_weights")
    settle(refs)
    if not asynchronous:
        settle(train_group.async_inference_call("finish_training_metrics", step))
    _evaluate_rollout(config, step, asynchronous=asynchronous, managed=managed)
    if step + 1 == config.num_rollout:
        wait_for_previous_eval()
    if asynchronous:
        settle(train_group.async_inference_call("finish_training_metrics", step))
