# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from relax.engine.rewards.dapo_genrm import async_compute_score_genrm
from relax.engine.rollout.base_types import call_rollout_fn
from relax.engine.rollout.deferred_scoring import DeferredBatch
from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.utils.async_utils import run
from relax.utils.logging_utils import get_logger
from relax.utils.utils import build_rollout_custom_meta, convert_samples_to_train_data


logger = get_logger(__name__)


def deferred_roles(args: Any) -> frozenset[str]:
    return frozenset(getattr(args, "inference_defer_roles", ()) or ())


def validate_deferred_workload(args: Any) -> None:
    roles = deferred_roles(args)
    if not roles:
        return
    if roles - {"teacher", "genrm"}:
        raise ValueError("Only Teacher and GenRM may be deferred")
    if not getattr(args, "colocate", False) or getattr(args, "fully_async", False):
        raise ValueError("Deferred scoring requires synchronous colocate")
    for name in (
        "partial_rollout",
        "use_dynamic_global_batch_size",
        "use_dynamic_batch_size",
        "debug_rollout_only",
        "debug_train_only",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Deferred closed batches do not support {name} yet")
    for name in (
        "custom_reward_post_process_path",
        "custom_convert_samples_to_train_data_path",
        "agentic_custom_advantage_path",
        "dynamic_sampling_filter_path",
        "rollout_sample_filter_path",
    ):
        if getattr(args, name, None):
            raise ValueError(f"Deferred scoring requires an explicit stage adapter for {name}")
    if getattr(args, "loss_type", None) == "sft":
        raise ValueError("Deferred scoring requires an RL workload")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("Deferred scoring currently requires the Megatron phase adapter")
    if getattr(args, "eval_interval", None) is not None and getattr(args, "eval_prompt_data", None):
        raise ValueError("Deferred model evaluation requires a separate scoring phase adapter")
    if "teacher" in roles and not (getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"):
        raise ValueError("Deferred Teacher requires SGLang OPD")
    if "genrm" in roles:
        models = getattr(args, "_genrm_instances_resolved", {})
        if len(models) != 1 or getattr(args, "rm_type", None) != "dapo-genrm" or getattr(args, "custom_rm_path", None):
            raise ValueError("Deferred GenRM currently requires one model with the dapo-genrm reward adapter")
    if getattr(args, "_genrm_instances_resolved", {}) and "genrm" not in roles:
        raise ValueError("Coordinated defer currently requires every managed scorer to be deferred")
    if getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang" and "teacher" not in roles:
        raise ValueError("Coordinated defer currently requires Teacher to participate in the scoring schedule")
    if getattr(args, "sglang_config", None) is not None or getattr(args, "prefill_num_servers", None):
        raise ValueError("Deferred weight restore currently requires one regular Rollout model group")
    if getattr(args, "opd_token_selection", None) in {"teacher_topk", "union"} and getattr(args, "opd_kl_coef", 0):
        if getattr(args, "over_sampling_batch_size", None) != getattr(args, "rollout_batch_size", None):
            raise ValueError(
                "Deferred student scoring requires equal oversampling and rollout batch sizes to avoid old-policy surplus"
            )
    default_path = (
        "relax.agentic.rollout.generate_rollout"
        if getattr(args, "use_agentic_rollout", False)
        else "relax.engine.rollout.sglang_rollout.generate_rollout"
    )
    if getattr(args, "rollout_function_path", default_path) != default_path:
        raise ValueError("Custom rollout needs an explicit deferred transfer adapter")


@dataclass
class DeferredTransferCollector:
    rollout_id: int
    groups: list[list[Any]] = field(default_factory=list)
    max_samples: int = 65536
    count: int = 0
    finalize_callback: Callable[[list[Any]], Any] | None = None
    finalized: bool = False

    def add(self, groups: list, rollout_id: int) -> None:
        if rollout_id != self.rollout_id:
            raise ValueError("Deferred transfer cannot cross rollout partitions")

        def collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                if value and not isinstance(value[0], (list, tuple)):
                    self.groups.append(list(value))
                    self.count += len(value)
                else:
                    for item in value:
                        collect(item)
            else:
                self.groups.append([value])
                self.count += 1

        collect(groups)
        if self.count > self.max_samples:
            raise ValueError("Deferred transfer staging exceeded sample capacity")

    def register_finalize(self, callback: Callable[[list[Any]], Any]) -> None:
        if self.finalize_callback is not None and self.finalize_callback is not callback:
            raise RuntimeError("Deferred rollout finalizer has already been registered")
        self.finalize_callback = callback

    async def finalize(self, samples: list[Any]) -> None:
        if self.finalized or self.finalize_callback is None:
            return
        result = self.finalize_callback(samples)
        if inspect.isawaitable(result):
            await result
        self.finalized = True


_collector: contextvars.ContextVar[DeferredTransferCollector | None] = contextvars.ContextVar(
    "inference_deferred_transfer", default=None
)


@contextmanager
def capture_transfers(collector: DeferredTransferCollector) -> Iterator[None]:
    token = _collector.set(collector)
    try:
        yield
    finally:
        _collector.reset(token)


def capture_deferred_transfer(args: Any, groups: list, rollout_id: int) -> bool:
    if not deferred_roles(args):
        return False
    collector = _collector.get()
    if collector is None:
        raise RuntimeError("Deferred producer tried to publish outside its managed batch")
    collector.add(groups, rollout_id)
    return True


def register_deferred_rollout_finalize(args: Any, rollout_id: int, callback: Callable[[list[Any]], Any]) -> bool:
    if not deferred_roles(args):
        return False
    collector = _collector.get()
    if collector is None or collector.rollout_id != rollout_id:
        raise RuntimeError("Deferred rollout tried to register finalization outside its managed batch")
    collector.register_finalize(callback)
    return True


async def wait_inference_commit(args: Any, rollout_id: int) -> None:
    if not deferred_roles(args):
        return
    coordinator = getattr(args, "_inference_coordinator", None)
    if coordinator is None:
        raise RuntimeError("Deferred training has no commit coordinator")
    await coordinator.wait_committed.remote(rollout_id, timeout=float(getattr(args, "rollout_http_timeout", 1800.0)))


def run_deferred_rollout(manager: Any, rollout_id: int) -> Any:

    args = manager.args
    coordinator = args._inference_coordinator
    collector = DeferredTransferCollector(rollout_id)

    async def begin():
        await coordinator.begin_batch.remote(rollout_id)

    run(begin())
    try:
        with capture_transfers(collector):
            output = call_rollout_fn(
                manager.generate_rollout,
                args,
                rollout_id,
                manager.data_source,
                manager.data_system_client,
                evaluation=False,
            )
        if not collector.groups:
            raise RuntimeError("Deferred rollout produced no captured training groups")
        collector.groups.sort(key=lambda group: group[0].index)
        run(_score_and_publish(manager, collector))
        return output
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        released = True
        try:
            manager._offload_local()
        except BaseException as release_exc:
            released = False
            message += f"; rollout release unconfirmed: {type(release_exc).__name__}: {release_exc}"
            logger.error(f"Deferred rollout {rollout_id} could not release rollout GPUs: {release_exc}")

        async def fail():
            try:
                if released:
                    await coordinator.acknowledge_rollout_offloaded.remote(rollout_id)
            finally:
                await coordinator.fail_batch.remote(rollout_id, message)

        run(fail())
        raise


async def _score_and_publish(manager: Any, collector: DeferredTransferCollector) -> None:

    args, rollout_id = manager.args, collector.rollout_id
    coordinator = args._inference_coordinator
    roles = deferred_roles(args)
    opd = None
    if "teacher" in roles:
        opd = OpdManager(args)
    batch = DeferredBatch(
        getattr(args, "_inference_run_id", "current-run"), timeout=float(getattr(args, "rollout_http_timeout", 1800))
    )

    async def offload_rollout():
        await asyncio.to_thread(manager._offload_local)
        await coordinator.acknowledge_rollout_offloaded.remote(rollout_id)

    async def activate(role):
        await coordinator.activate.remote(role, f"{rollout_id}:{role}:activate")

    async def deactivate(role):
        if role == "rollout":
            await offload_rollout()
        else:
            await coordinator.deactivate.remote(role, f"{rollout_id}:{role}:deactivate")

    async def reward(groups):

        for group in groups:
            results = await asyncio.gather(
                *(async_compute_score_genrm(args, sample) for sample in group), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            for sample, result in zip(group, results, strict=True):
                if result.get("format_error") == "judge_transient_error":
                    raise RuntimeError("Deferred GenRM request failed")
                key = getattr(args, "reward_key", None)
                sample.reward = {key: result[key]} if key else result["score"]

    async def restore_student(version):
        await coordinator.begin_student_scoring.remote(rollout_id)
        await asyncio.to_thread(manager._onload_local)
        await asyncio.to_thread(manager.mark_inference_weights_ready)
        snapshot = manager.get_inference_snapshot()
        model = next(iter(snapshot["models"].values()))
        if model["state"] != "READY" or model["weight_version"] != version:
            raise RuntimeError("Deferred student restored a different or unavailable policy version")

    async def student(samples):
        from relax.engine.rollout.sglang_rollout import _encode_multimodal_inputs

        await opd.student_prefill(samples, _encode_multimodal_inputs)

    async def publish(completed):
        rollout_batch = convert_samples_to_train_data(args, completed.samples)
        partition_id = f"train_{rollout_id}"
        await manager.data_system_client.async_put(
            data=rollout_batch,
            partition_id=partition_id,
            custom_meta=build_rollout_custom_meta(rollout_batch),
            is_last=True,
        )
        try:
            await collector.finalize(completed.samples)
        except BaseException:
            try:
                await manager.data_system_client.async_clear_partition(partition_id=partition_id)
            except Exception as clear_exc:
                logger.error(f"Failed to clear uncommitted deferred partition {partition_id}: {clear_exc}")
            raise
        await coordinator.commit_batch.remote(rollout_id)

    await batch.complete(
        rollout_id,
        collector.groups,
        offload_rollout=offload_rollout,
        activate_role=activate,
        deactivate_role=deactivate,
        reward_stage=reward if "genrm" in roles else None,
        prepare_teacher=opd.prepare if opd else None,
        teacher_stage=opd.teacher_prefill if opd else None,
        student_stage=student if opd and opd.topk_worker and opd.topk_worker.spec.student_at_teacher else None,
        assemble_validate=opd.assemble_validate if opd else None,
        restore_student=restore_student,
        commit=publish,
    )
