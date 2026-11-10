# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit GPU fault experiment; never installed by an ordinary recipe."""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep
from typing import Any


class InjectedRecoveryFailure(RuntimeError):
    pass


def pause_schedulers(engines: Any) -> None:
    """Pause actual scheduler execution, rather than only aborting requests."""
    import ray

    # SGLang's empty pause request defaults to mode=abort. That pauses the
    # tokenizer but deliberately never sends a pause command to schedulers.
    ray.get(
        [engine._make_request.remote("pause_generation", {"mode": "in_place"}, timeout=30) for engine in engines],
        timeout=40,
    )


def scheduler_state() -> dict:
    """Inspect the real scheduler frame executing this test-only request."""
    frame = inspect.currentframe()
    try:
        while frame is not None:
            candidate = frame.f_locals.get("self")
            if hasattr(candidate, "_engine_paused") and hasattr(candidate, "tp_worker"):
                return dict(paused=bool(candidate._engine_paused))
            frame = frame.f_back
    finally:
        del frame
    raise RuntimeError("Recovery test is not executing inside a real scheduler")


def catch_injected_fault(original: Any) -> Any:
    def update(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except InjectedRecoveryFailure as exc:
            return False, f"InjectedRecoveryFailure: {exc}"

    return update


def worker_control(model: Any, control: dict) -> dict:
    import torch

    from relax.distributed.weight_sync.store import atomic_write

    receiver = model._relax_sharded_memory_receiver
    directory = Path(control["directory"]) / "failure-probe"
    member = control["member"]
    if control["command"] == "arm_failure":
        from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater

        if getattr(model, "_relax_recovery_probe", None) is not None:
            raise RuntimeError("Fault probe already armed")
        if not getattr(WeightUpdater, "_relax_fault_catch", False):
            WeightUpdater._relax_original_tensor_update = WeightUpdater.update_weights_from_tensor
            WeightUpdater.update_weights_from_tensor = catch_injected_fault(WeightUpdater.update_weights_from_tensor)
            WeightUpdater._relax_fault_catch = True
        state = dict(triggered=False, forward_attempts=0, injected=control["inject"], applied_buckets=0)
        original_bucket, original_group, original_forward = receiver._apply_bucket, receiver.apply_group, model.forward

        def forward(*args, **kwargs):
            state["forward_attempts"] += 1
            raise RuntimeError("Forward is forbidden during the recovery experiment")

        def bucket(*args, **kwargs):
            before, target = None, None
            if state["injected"] and not state["triggered"]:
                from relax.backends.sglang.sharded_memory_loader import _PREFIX

                metadata = args[0]
                name, start, _ = json.loads(metadata["names"][0][0][len(_PREFIX) :])
                shape = metadata["names"][0][1]
                target = model.get_parameter(name).narrow(0, start, shape[0]).view(torch.uint8).reshape(-1)[:32]
                before = target.clone()
            original_bucket(*args, **kwargs)
            state["applied_buckets"] += 1
            if state["injected"] and not state["triggered"]:
                torch.cuda.synchronize()
                state["triggered"] = True
                state["changed_probe_bytes"] = int(torch.count_nonzero(before != target).item())
                if not state["changed_probe_bytes"]:
                    raise AssertionError("Fault did not follow an actual GPU value change")
                deadline = monotonic() + 300

                def peers_installed():
                    for other in control["other_members"]:
                        path = directory / f"{other}-installed.json"
                        if not path.exists() or json.loads(path.read_text())["group_update"] != state["group_update"]:
                            return False
                    return True

                while not peers_installed():
                    if monotonic() >= deadline:
                        raise TimeoutError("Fault peers did not complete their real FULL installation")
                    sleep(0.1)
                atomic_write(directory / f"{member}-partial.json", json.dumps(state).encode())
                raise InjectedRecoveryFailure("Expected failure after a real FULL bucket GPU write")

        def apply_group(group, receive):
            state.update(group_update=group["group_update"], attempted_version=group["version"])
            result = original_group(group, receive)
            atomic_write(directory / f"{member}-installed.json", json.dumps(state).encode())
            return result

        receiver._apply_bucket, receiver.apply_group, model.forward = bucket, apply_group, forward
        model._relax_recovery_probe = (state, original_bucket, original_group, original_forward)
    state, original_bucket, original_group, original_forward = model._relax_recovery_probe
    result = dict(state, **scheduler_state(), receiver_poisoned=receiver.poisoned, receiver_version=receiver.version)
    if control["command"] == "disarm_failure":
        from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater

        receiver._apply_bucket, receiver.apply_group, model.forward = original_bucket, original_group, original_forward
        model._relax_recovery_probe = None
        WeightUpdater.update_weights_from_tensor = WeightUpdater._relax_original_tensor_update
        del WeightUpdater._relax_original_tensor_update
        del WeightUpdater._relax_fault_catch
    return result


class _DeferredCall:
    def __init__(self, actor: Any, name: str, calls: list) -> None:
        self.actor, self.name, self.calls = actor, name, calls

    def remote(self, *args, **kwargs):
        import ray

        self.calls.append(self.name)
        return ray.put(dict(success=True, deferred=True))


class _DeferredEngine:
    def __init__(self, actor: Any) -> None:
        self.actor, self._actor_id, self.calls = actor, actor._actor_id, []
        self.continue_generation = _DeferredCall(actor, "continue", self.calls)
        self.disk_weight_generation_probe = _DeferredCall(actor, "probe", self.calls)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "actor"), name)


class _DeferredManager:
    def __init__(self, engines: tuple, lock: Any) -> None:
        self.engines, self.lock = engines, lock

    @property
    def get_rollout_engines_and_lock(self):
        return self

    def remote(self):
        import ray

        return ray.put((self.engines, self.lock, 0, [2] * 4, [0, 2, 4, 6]))


def install(args: Any) -> None:
    import os

    import ray
    import torch
    import torch.distributed as dist

    from relax.backends.megatron.weight_update.update_weight_from_memory import UpdateWeightFromMemory
    from relax.backends.megatron.weight_update.update_weight_from_sharded_memory import UpdateWeightFromShardedMemory
    from relax.distributed.weight_sync.store import atomic_write
    from relax.utils.distributed_utils import get_gloo_group
    from task_delta_weight_sync import measure_sharded

    cls = UpdateWeightFromShardedMemory
    if type(args.num_rollout) is not int or not 1 <= args.num_rollout <= 20:
        raise ValueError("Recovery probe requires a bounded 1..20-update experiment")
    raw_update = cls.update_weights
    measure_sharded.install(args)
    paired_update = cls.update_weights
    rank = dist.get_rank(get_gloo_group())
    directory = Path(os.environ["TENSORBOARD_DIR"]).resolve().parents[1] / "audit"

    def experiment(self, manager):
        result = paired_update(self, manager)
        if self._ab_step != args.num_rollout:
            return result
        engines, lock = self._ab_engines, self._ab_lock
        target = engines[0]._actor_id.hex()
        other_members = [f"{engine._actor_id.hex()}:tp:{tp}" for engine in engines[1:] for tp in range(2)]
        update_id = f"optimizer-{self._ab_step:06d}"
        fresh = None
        fresh_closed = False

        def close_fresh():
            errors = []
            for resource in [*fresh.sources.values(), *fresh.encoders.values()]:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            if errors:
                raise RuntimeError(f"Recovery cleanup failed: {errors}")

        def root_action(action):
            error, value = None, None
            if rank == 0:
                try:
                    value = action()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            self._gate(error)
            return value

        def local_action(action):
            error, value = None, None
            try:
                value = action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._gate(error)
            return value

        def audit(command, phase):
            controls = [
                dict(
                    command=command,
                    phase=phase,
                    update_id=update_id,
                    paused=True,
                    directory=str(directory),
                    node_budget_bytes=384 << 30,
                    inject=engine._actor_id.hex() == target,
                    other_members=other_members,
                )
                for engine in engines
            ]
            ray.get(
                [
                    engine.run_memory_audit.remote(control, 600)
                    for engine, control in zip(engines, controls, strict=True)
                ],
                timeout=610,
            )

        def inspect_reports(phase, *, compare=False):
            reports = []
            for engine in engines:
                for tp in range(2):
                    member = f"{engine._actor_id.hex()}:tp:{tp}"
                    report = json.loads((directory / member / f"{update_id}-{phase}.json").read_text())
                    if compare:
                        inventory = json.loads((directory / member / "inventory.json").read_text())
                        assert (
                            report["compared_bytes"] == inventory["snapshot_bytes"] and report["mismatched_bytes"] == 0
                        )
                    else:
                        assert report["paused"] and report["forward_attempts"] == 0
                        if engine._actor_id.hex() == target:
                            assert (
                                report["triggered"] and report["applied_buckets"] > 0 and report["receiver_poisoned"]
                            )
                            assert report["changed_probe_bytes"] > 0 and report["receiver_version"] == versions[0]
                        else:
                            assert not report["triggered"] and not report["receiver_poisoned"]
                            assert report["receiver_version"] == versions[0] + 1
                            installed = json.loads(
                                (directory / "failure-probe" / f"{member}-installed.json").read_text()
                            )
                            assert installed["group_update"] == report["group_update"]
                    reports.append(report)
            return reports

        try:
            from task_delta_weight_sync.native_call import guarded_native_call
            from task_delta_weight_sync.prepared_nccl import run as run_prepared

            guarded_native_call(
                lambda: run_prepared(self, directory) if rank == 0 else None,
                process_group=self._ab_native_control,
                timeout_s=600,
                failure_path=directory / f"prepared-native-failure-rank-{rank}.json",
                quarantine=(lambda: UpdateWeightFromMemory._close_generation(self, manager)) if rank == 0 else None,
            )
            root_action(lambda: pause_schedulers(engines))
            root_action(lambda: audit("arm_failure", "failure-armed"))
            root_action(lambda: audit("complement", "failure-complement"))
            versions = (self.version, {key: encoder.version for key, encoder in self.encoders.items()})
            error, started = None, monotonic()
            try:
                raw_update(self, manager, full_only=True)
            except RuntimeError as exc:
                error = str(exc)

            def check_failed():
                assert error is not None and "InjectedRecoveryFailure" in error and self.failed
                assert versions == (self.version, {key: e.version for key, e in self.encoders.items()})
                try:
                    raw_update(self, manager)
                except RuntimeError as exc:
                    assert "Discard the failed" in str(exc)
                else:
                    raise AssertionError("Old updater accepted a retry")

            local_action(check_failed)
            root_action(lambda: audit("failure_state", "failure-quarantined"))
            quarantined = root_action(lambda: inspect_reports("failure-quarantined"))

            def close_failed_session():
                value = ray.get(engines[0].end_weight_update.remote(), timeout=30)
                if value is not None and value.get("success") is not True:
                    raise RuntimeError(f"Cannot close failed session: {value}")

            root_action(close_failed_session)
            root_action(lambda: audit("disarm_failure", "failure-disarmed"))

            def build_fresh():
                for encoder in self.encoders.values():
                    encoder.close()
                assert self.budget.used_bytes == 0
                torch.cuda.empty_cache()
                value = cls(
                    args,
                    self.iterator.model,
                    self.weights_getter,
                    model_name=self.iterator.model_name,
                    quantization_config=None,
                )
                value.config = replace(self.config, stream=self.config.stream + "-recovered")
                return value

            fresh = local_action(build_fresh)
            deferred = tuple(_DeferredEngine(engine) for engine in engines)
            fresh_result = raw_update(fresh, _DeferredManager(deferred, lock), full_only=True)
            root_action(lambda: audit("compare", "fresh-full-recovery"))
            compared = root_action(lambda: inspect_reports("fresh-full-recovery", compare=True))

            def check_schemas():
                from task_delta_weight_sync.sync_metrics import validate_stream_receipts

                return validate_stream_receipts(fresh_result, compared)

            fresh_schema_check = root_action(check_schemas)
            local_action(close_fresh)
            fresh_closed = True

            def resume():
                assert all(engine.calls == ["continue", "probe"] for engine in fresh.engines)
                ray.get([engine.continue_generation.remote(timeout=30) for engine in engines], timeout=40)
                ray.get([engine.disk_weight_generation_probe.remote(30) for engine in engines], timeout=40)
                atomic_write(
                    directory / "recovery-verified.json",
                    json.dumps(
                        dict(
                            status="VERIFIED",
                            optimizer_update=self._ab_step,
                            seconds=monotonic() - started,
                            failed_stream=self.config.stream,
                            recovered_stream=fresh.config.stream,
                            old_retry_rejected=True,
                            quarantined=quarantined,
                            compared=compared,
                            fresh_result=fresh_result,
                            fresh_schema_check=fresh_schema_check,
                            fresh_result_scope="FULL installation with test-deferred resume/probe; correctness evidence only",
                            failure=error,
                            limitation="Forced FULL after complement; test-only deferred resume; excluded from performance",
                        ),
                        indent=2,
                    ).encode(),
                )

            root_action(resume)
            return result
        except BaseException:
            if fresh is not None:
                fresh.failed = True
            if rank == 0:
                self.engines = engines
                UpdateWeightFromMemory._close_generation(self, manager)
            raise
        finally:
            if fresh is not None and not fresh_closed:
                original_error = sys.exc_info()[1]
                try:
                    close_fresh()
                except Exception as exc:
                    if original_error is not None and hasattr(original_error, "add_note"):
                        original_error.add_note(str(exc))
                    try:
                        atomic_write(
                            directory / f"recovery-cleanup-rank-{rank}.json", json.dumps(dict(error=str(exc))).encode()
                        )
                    except OSError:
                        pass

    cls.update_weights = experiment
