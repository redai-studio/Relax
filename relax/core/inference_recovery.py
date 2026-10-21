# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded recovery for explicit inference errors at complete-batch
boundaries."""

import time
from argparse import Namespace
from typing import Any

import ray

from relax.utils.async_utils import run


class UnsafeInferenceError(RuntimeError):
    """An unknown old operation cannot participate in local recovery."""


def wait_all(refs: list, timeout: float = 600) -> list:
    """An exception from one rank does not prove its peers have stopped."""
    if not refs:
        return []
    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
    if pending:
        raise UnsafeInferenceError(f"{len(pending)} inference operations still running; require global restart")
    results, errors = [], []
    for ref in refs:
        try:
            results.append(ray.get(ref))
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]
    return results


@ray.remote(num_cpus=0)
class InferenceRecoveryState:
    """One serialized admission gate shared by the existing components."""

    def __init__(self, has_actor_fwd: bool, max_attempts: int = 3) -> None:
        self.has_actor_fwd = has_actor_fwd
        self.max_attempts = max_attempts
        self.paused = False
        self.failures: dict[str, str] = {}
        self.active: dict[str, int] = {}
        self.sync = False
        self.unsafe = False
        self.fwd_ready = False
        self.generation = 0
        self.step = 0
        self.attempts = 0

    def snapshot(self) -> dict:
        return dict(
            paused=self.paused,
            failures=dict(self.failures),
            active=dict(self.active),
            sync=self.sync,
            unsafe=self.unsafe,
            fwd_ready=self.fwd_ready,
            generation=self.generation,
            step=self.step,
        )

    def enter(self, role: str, step: int, generation: int) -> bool:
        if self.paused or generation != self.generation:
            return False
        if role in self.active:
            raise RuntimeError(f"Concurrent {role} operation")
        self.active[role] = step
        return True

    def leave(self, role: str) -> None:
        self.active.pop(role, None)

    def report_error(self, role: str, error: str, unsafe: bool = False) -> None:
        self.failures[role] = error
        self.unsafe |= self.sync or unsafe
        self.paused = True

    def request_recovery(self, role: str) -> int:
        self.paused = True
        self.attempts += 1
        if self.unsafe or self.sync or set(self.failures) != {role}:
            raise RuntimeError("Concurrent failure or weight synchronization requires global restart")
        if self.attempts > self.max_attempts:
            raise RuntimeError(f"Local recovery budget ({self.max_attempts}) exhausted; require global restart")
        return self.attempts

    def set_fwd_ready(self, ready: bool) -> None:
        self.fwd_ready = ready

    def begin_sync(self, maintenance: bool = False) -> bool:
        if self.sync or (self.paused and not maintenance):
            return False
        if not maintenance and self.has_actor_fwd and not self.fwd_ready:
            return False
        self.sync = True
        self.fwd_ready = False
        return True

    def end_sync(self) -> None:
        self.sync = False

    def resume(self, step: int) -> int:
        if self.active or self.sync or self.unsafe:
            raise RuntimeError("Old operations have not stopped")
        self.generation += 1
        self.step = step
        self.failures.clear()
        self.fwd_ready = False
        self.paused = False
        return self.generation


class RecoveryGate:
    def __init__(self, config: Namespace, role: str) -> None:
        self.config = config
        self.role = str(role)
        self.state = getattr(config, "inference_recovery_state", None)
        self.generation = 0

    def admit(self, step: int, client: Any = None, fields: list | tuple = (), stop_event: Any = None) -> int | None:
        if self.state is None:
            return step
        while stop_event is None or not stop_event.is_set():
            status = ray.get(self.state.snapshot.remote())
            if status["generation"] != self.generation:
                self.generation = status["generation"]
                step = status["step"]
            if not status["paused"]:
                ready = not fields or self._batch_ready(client, step, list(dict.fromkeys(fields)))
                if ready and ray.get(self.state.enter.remote(self.role, step, self.generation)):
                    return step
            time.sleep(0.05)
        return None

    def _batch_ready(self, client: Any, step: int, fields: list[str]) -> bool:
        partition = f"train_{step}"
        if not run(client.async_check_production_completed(partition)):
            return False
        ids, status = run(client.async_get_production_status(data_fields=fields, partition_id=partition))
        size = self.config.rollout_batch_size * self.config.n_samples_per_prompt
        return (
            ids is not None
            and status is not None
            and len(ids) == size
            and tuple(status.shape) == (size, len(fields))
            and bool((status == 1).all())
        )

    def leave(self) -> None:
        if self.state is not None:
            ray.get(self.state.leave.remote(self.role))

    def fail(self, exc: Exception) -> None:
        if self.state is not None:
            ray.get(
                self.state.report_error.remote(
                    self.role,
                    f"{type(exc).__name__}: {exc}",
                    unsafe=isinstance(exc, (UnsafeInferenceError, ray.exceptions.RayActorError)),
                )
            )


def configure_recovery(config: Namespace) -> Any:
    """Enable supported full-async/hybrid recovery by default; colocate is opt-
    in."""
    import os

    from relax.core.node_group_affinity import with_control_plane_affinity

    config.inference_recovery = False
    config.inference_recovery_state = None
    choice = os.environ.get("RELAX_ENABLE_INFERENCE_RECOVERY")
    if choice is not None:
        choice = choice.strip().lower()
    if choice not in (None, "1", "true"):
        return None
    hybrid = bool(getattr(config, "hybrid", False))
    fully_async = bool(getattr(config, "fully_async", False))
    if choice is None and not (fully_async or hybrid):
        return None

    conflicts = [
        f"{name}={getattr(config, name, None)!r} (requires {required!r})"
        for name, required in (
            ("train_backend", "megatron"),
            ("advantage_estimator", "grpo"),
            ("kl_coef", 0),
            ("multimodal_keys", None),
            ("rollout_function_path", "relax.engine.rollout.sglang_rollout.generate_rollout"),
        )
        if getattr(config, name, None) != required
    ]
    conflicts.extend(
        f"{name}={getattr(config, name)!r} (requires disabled or empty)"
        for name in (
            "use_kl_loss",
            "use_agentic_rollout",
            "use_opd",
            "use_critic",
            "rollout_external",
            "debug_train_only",
            "debug_rollout_only",
            "use_dynamic_global_batch_size",
            "use_fault_tolerance",
            "use_slime_router",
            "autoscaler_config",
            "sglang_config",
            "lora_adapter_mode",
            "custom_generate_function_path",
            "dynamic_sampling_filter_path",
            "_genrm_instances_resolved",
            "sglang_router_ip",
        )
        if getattr(config, name, False)
    )
    if fully_async and getattr(config, "use_routing_replay", False):
        conflicts.append(f"use_routing_replay={config.use_routing_replay!r} (requires disabled when fully_async=True)")
    if hybrid:
        conflicts.extend(
            f"{name}={getattr(config, name)!r} (requires disabled when hybrid=True)"
            for name in ("offload_train", "offload_rollout")
            if getattr(config, name, False)
        )
    elif getattr(config, "colocate", False):
        conflicts.extend(
            f"{name}={getattr(config, name, False)!r} (requires enabled when colocate=True)"
            for name in ("offload_train", "offload_rollout", "enable_weights_backuper")
            if not getattr(config, name, False)
        )
    if conflicts:
        if choice in ("1", "true"):
            raise ValueError("Unsupported inference recovery configuration: " + "; ".join(conflicts))
        return None
    try:
        max_attempts = int(os.environ.get("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", "3"))
    except ValueError as exc:
        raise ValueError("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS must be a non-negative integer") from exc
    if max_attempts < 0:
        raise ValueError("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS must be a non-negative integer")
    config.inference_recovery_state = InferenceRecoveryState.options(**with_control_plane_affinity(config)).remote(
        fully_async and not hybrid and not getattr(config, "true_on_policy_mode", False), max_attempts=max_attempts
    )
    config.inference_recovery = True
    return config.inference_recovery_state
