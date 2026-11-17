# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Component admission at complete training-batch boundaries."""

import time
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import ray

from relax.distributed.settlement import UnsafeInferenceError, wait_all
from relax.utils.async_utils import run


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

    def wait(self, refs: list) -> list:
        """Settle all ranks for local recovery; otherwise propagate
        failures."""
        if self.state is None:
            return ray.get(refs)
        return wait_all(refs)

    def synchronize(self, operation: Callable[[], None]) -> bool:
        """Reserve the inference tail after all training ranks have settled.

        A failed or unknown tail retains the reservation, forcing global
        restart. Never clear it in a finally block.
        """
        if self.state is not None:
            while not ray.get(self.state.begin_sync.remote(), timeout=60):
                if ray.get(self.state.snapshot.remote(), timeout=60)["paused"]:
                    return False
                time.sleep(0.1)
        operation()
        if self.state is not None:
            ray.get(self.state.end_sync.remote(), timeout=60)
        return True

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


def inference_recovery_enabled(config: Namespace) -> bool:
    """Select component recovery for supported fault-tolerant training."""
    if not getattr(config, "use_fault_tolerance", False):
        return False
    hybrid = bool(getattr(config, "hybrid", False))
    fully_async = bool(getattr(config, "fully_async", False))

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
    return not conflicts
