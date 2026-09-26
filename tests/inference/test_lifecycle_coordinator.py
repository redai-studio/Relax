# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
from typing import Any

import pytest

from relax.distributed.ray.inference_lifecycle import (
    InferenceBatchError,
    InferenceLifecycleError,
    LifecycleCoordinatorState,
)


class _Method:
    def __init__(self, manager: "_Manager", name: str) -> None:
        self.manager = manager
        self.name = name

    async def remote(self) -> None:
        self.manager.calls.append(self.name)
        self.manager.started[self.name].set()
        gate = self.manager.gates.get(self.name)
        if gate is not None:
            await gate.wait()
        error = self.manager.errors.get(self.name)
        if error is not None:
            raise error
        self.manager.completed.append(self.name)


class _Manager:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.completed: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.errors: dict[str, BaseException] = {}
        self.started = {"onload": asyncio.Event(), "offload": asyncio.Event()}
        self.onload = _Method(self, "onload")
        self.offload = _Method(self, "offload")


@pytest.mark.asyncio
async def test_rollout_and_scoring_leases_are_exclusive_until_offload_acknowledges() -> None:
    coordinator = LifecycleCoordinatorState()
    genrm, teacher = _Manager(), _Manager()
    await coordinator.register("genrm", [genrm])
    await coordinator.register("teacher", [teacher])
    await coordinator.begin_batch(0)

    with pytest.raises(InferenceLifecycleError, match="leases held by rollout"):
        await coordinator.activate("teacher", "teacher-on")
    assert not teacher.calls
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("genrm", "genrm-on")
    with pytest.raises(InferenceLifecycleError, match="leases held by genrm"):
        await coordinator.activate("teacher", "teacher-on")
    with pytest.raises(InferenceLifecycleError, match="before inference offload"):
        await coordinator.commit_batch(0)
    await coordinator.deactivate("genrm", "genrm-off")
    await coordinator.activate("teacher", "teacher-on")
    await coordinator.deactivate("teacher", "teacher-off")
    await coordinator.commit_batch(0)
    await coordinator.wait_committed(0, timeout=1.0)
    assert genrm.calls == teacher.calls == ["onload", "offload"]


@pytest.mark.asyncio
async def test_concurrent_duplicate_operation_joins_one_manager_rpc() -> None:
    coordinator = LifecycleCoordinatorState()
    teacher = _Manager()
    teacher.gates["onload"] = asyncio.Event()
    await coordinator.register("teacher", [teacher])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)

    first = asyncio.create_task(coordinator.activate("teacher", "on"))
    second = asyncio.create_task(coordinator.activate("teacher", "on"))
    await teacher.started["onload"].wait()
    assert teacher.calls == ["onload"]
    assert (await coordinator.get_state())["roles"]["teacher"] == "ONLOADING"
    with pytest.raises(InferenceLifecycleError, match="ONLOADING"):
        await coordinator.deactivate("teacher", "off")
    teacher.gates["onload"].set()
    await asyncio.gather(first, second)
    await coordinator.activate("teacher", "on")
    assert teacher.calls == ["onload"]


@pytest.mark.asyncio
async def test_operation_id_fingerprint_rejects_different_action_role_and_batch() -> None:
    coordinator = LifecycleCoordinatorState()
    await coordinator.register("teacher", [_Manager()])
    await coordinator.register("genrm", [_Manager()])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("teacher", "unique")
    with pytest.raises(InferenceLifecycleError, match="different fingerprint"):
        await coordinator.deactivate("teacher", "unique")
    with pytest.raises(InferenceLifecycleError, match="different fingerprint"):
        await coordinator.activate("genrm", "unique")
    await coordinator.deactivate("teacher", "off")
    await coordinator.commit_batch(0)
    await coordinator.begin_batch(1)
    await coordinator.acknowledge_rollout_offloaded(1)
    with pytest.raises(InferenceLifecycleError, match="different fingerprint"):
        await coordinator.activate("teacher", "unique")


@pytest.mark.asyncio
async def test_commit_waits_for_every_manager_release_and_wakes_all_consumers() -> None:
    coordinator = LifecycleCoordinatorState()
    first, second = _Manager(), _Manager()
    second.gates["offload"] = asyncio.Event()
    await coordinator.register("teacher", [first, second])
    consumers = [asyncio.create_task(coordinator.wait_committed(0, timeout=1.0)) for _ in range(3)]
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("teacher", "on")
    offload = asyncio.create_task(coordinator.deactivate("teacher", "off"))
    await second.started["offload"].wait()
    with pytest.raises(InferenceLifecycleError, match="before inference offload"):
        await coordinator.commit_batch(0)
    assert not any(consumer.done() for consumer in consumers)
    second.gates["offload"].set()
    await offload
    assert not any(consumer.done() for consumer in consumers)
    await coordinator.commit_batch(0)
    await asyncio.gather(*consumers)
    await coordinator.commit_batch(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["onload", "offload"])
async def test_manager_failure_retains_lease_and_fails_current_and_future_batch_waiters(method: str) -> None:
    coordinator = LifecycleCoordinatorState()
    teacher = _Manager()
    await coordinator.register("teacher", [teacher])
    await coordinator.register("genrm", [_Manager()])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    if method == "offload":
        await coordinator.activate("teacher", "on")
    teacher.errors[method] = ConnectionError("manager response lost")
    waiters = [asyncio.create_task(coordinator.wait_committed(index, timeout=1.0)) for index in (0, 1)]
    operation = coordinator.activate if method == "onload" else coordinator.deactivate
    for _ in range(2):
        with pytest.raises(InferenceLifecycleError, match="lease retained"):
            await operation("teacher", "failure")
    assert teacher.calls.count(method) == 1
    assert (await coordinator.get_state())["roles"]["teacher"] == "BLOCKED"
    for waiter in waiters:
        with pytest.raises(InferenceBatchError, match="manager response lost"):
            await waiter
    with pytest.raises(InferenceBatchError):
        await coordinator.activate("genrm", "genrm-on")
    with pytest.raises(InferenceBatchError):
        await coordinator.commit_batch(0)
    with pytest.raises(InferenceBatchError):
        await coordinator.begin_batch(1)


@pytest.mark.asyncio
async def test_timeout_never_releases_lease_even_after_late_rpc_success() -> None:
    coordinator = LifecycleCoordinatorState(operation_timeout=0.02)
    teacher = _Manager()
    teacher.gates["offload"] = asyncio.Event()
    await coordinator.register("teacher", [teacher])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("teacher", "on")
    with pytest.raises(InferenceLifecycleError, match="lease retained"):
        await coordinator.deactivate("teacher", "off")
    teacher.gates["offload"].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert teacher.completed == ["onload", "offload"]
    assert (await coordinator.get_state())["roles"]["teacher"] == "BLOCKED"
    with pytest.raises(InferenceBatchError):
        await coordinator.wait_committed(0, timeout=1.0)


@pytest.mark.asyncio
async def test_cancelled_caller_can_rejoin_shielded_operation_without_duplicate_onload() -> None:
    coordinator = LifecycleCoordinatorState()
    teacher = _Manager()
    teacher.gates["onload"] = asyncio.Event()
    await coordinator.register("teacher", [teacher])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    caller = asyncio.create_task(coordinator.activate("teacher", "on"))
    await teacher.started["onload"].wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert (await coordinator.get_state())["roles"]["teacher"] == "ONLOADING"
    teacher.gates["onload"].set()
    await coordinator.activate("teacher", "on")
    assert teacher.calls == ["onload"]
    assert (await coordinator.get_state())["roles"]["teacher"] == "ACTIVE"


@pytest.mark.asyncio
async def test_batch_failure_does_not_release_active_scorer_and_allows_confirmed_cleanup() -> None:
    coordinator = LifecycleCoordinatorState()
    teacher = _Manager()
    await coordinator.register("teacher", [teacher])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("teacher", "on")
    await coordinator.fail_batch(0, "Teacher result is missing")
    await coordinator.fail_batch(0, "Cleanup follows original error")
    assert (await coordinator.get_state())["roles"]["teacher"] == "ACTIVE"
    await coordinator.deactivate("teacher", "off")
    assert (await coordinator.get_state())["roles"]["teacher"] == "SLEEPING"
    with pytest.raises(InferenceBatchError, match="Teacher result is missing"):
        await coordinator.wait_committed(0, timeout=1.0)
    with pytest.raises(InferenceBatchError):
        await coordinator.commit_batch(0)


@pytest.mark.asyncio
async def test_wait_timeout_does_not_change_batch_or_lease() -> None:
    coordinator = LifecycleCoordinatorState()
    await coordinator.begin_batch(0)
    with pytest.raises(asyncio.TimeoutError):
        await coordinator.wait_committed(0, timeout=0.01)
    assert (await coordinator.get_state()) == {
        "rollout_id": 0,
        "batch_state": "PENDING",
        "error": None,
        "roles": {"rollout": "ACTIVE"},
    }
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.commit_batch(0)
    await coordinator.wait_committed(0, timeout=1.0)


@pytest.mark.asyncio
async def test_only_one_batch_is_pending_and_old_commit_remains_available() -> None:
    coordinator = LifecycleCoordinatorState()
    await coordinator.begin_batch(0)
    await coordinator.begin_batch(0)
    with pytest.raises(InferenceLifecycleError, match="Only one pending"):
        await coordinator.begin_batch(1)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.commit_batch(0)
    await coordinator.begin_batch(1)
    await coordinator.wait_committed(0, timeout=1.0)
    with pytest.raises(InferenceLifecycleError, match="not the current"):
        await coordinator.acknowledge_rollout_offloaded(0)
    assert (await coordinator.get_state())["roles"]["rollout"] == "ACTIVE"


@pytest.mark.asyncio
async def test_committed_batches_and_operations_are_evicted_with_bounded_history(monkeypatch) -> None:
    from relax.distributed.ray import inference_lifecycle as module

    monkeypatch.setattr(module, "_COMMITTED_HISTORY_LIMIT", 3)
    monkeypatch.setattr(module, "_OPERATION_HISTORY_LIMIT", 4)
    coordinator = LifecycleCoordinatorState()
    await coordinator.register("teacher", [_Manager()])
    for rollout_id in range(10):
        await coordinator.begin_batch(rollout_id)
        await coordinator.acknowledge_rollout_offloaded(rollout_id)
        await coordinator.activate("teacher", f"{rollout_id}:on")
        await coordinator.deactivate("teacher", f"{rollout_id}:off")
        await coordinator.commit_batch(rollout_id)

    await coordinator.begin_batch(10)

    assert list(coordinator._batches) == [10]
    assert coordinator._operations == {}
    assert len(coordinator._committed) == 3
    assert len(coordinator._retired_operations) == 4
    await coordinator.wait_committed(9, timeout=1.0)
    await coordinator.commit_batch(9)
    await coordinator.begin_batch(9)
    assert (await coordinator.get_state())["rollout_id"] == 10
    await coordinator.acknowledge_rollout_offloaded(10)
    with pytest.raises(InferenceLifecycleError, match="different fingerprint"):
        await coordinator.activate("teacher", "9:on")


@pytest.mark.asyncio
async def test_repeated_generation_ack_does_not_release_student_scoring_lease() -> None:
    coordinator = LifecycleCoordinatorState()
    rollout = _Manager()
    await coordinator.register("rollout", [rollout])
    await coordinator.begin_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("rollout", "student-on")
    await coordinator.acknowledge_rollout_offloaded(0)
    with pytest.raises(InferenceLifecycleError, match="before inference offload"):
        await coordinator.commit_batch(0)
    await coordinator.deactivate("rollout", "student-off")
    await coordinator.commit_batch(0)
    assert rollout.calls == ["onload", "offload"]


@pytest.mark.asyncio
async def test_local_student_scoring_requires_prior_release_and_another_offload_ack() -> None:
    coordinator = LifecycleCoordinatorState()
    teacher = _Manager()
    await coordinator.register("teacher", [teacher])
    await coordinator.begin_batch(0)
    with pytest.raises(InferenceLifecycleError, match="every inference role"):
        await coordinator.begin_student_scoring(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.activate("teacher", "teacher-on")
    with pytest.raises(InferenceLifecycleError, match="every inference role"):
        await coordinator.begin_student_scoring(0)
    await coordinator.deactivate("teacher", "teacher-off")
    await coordinator.begin_student_scoring(0)
    await coordinator.begin_student_scoring(0)
    with pytest.raises(InferenceLifecycleError, match="leases held by rollout"):
        await coordinator.activate("teacher", "teacher-on-again")
    with pytest.raises(InferenceLifecycleError, match="before inference offload"):
        await coordinator.commit_batch(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.acknowledge_rollout_offloaded(0)
    await coordinator.begin_student_scoring(0)
    assert (await coordinator.get_state())["roles"]["rollout"] == "SLEEPING"
    await coordinator.commit_batch(0)


@pytest.mark.asyncio
async def test_registration_is_idempotent_and_topology_is_frozen_before_first_batch() -> None:
    coordinator = LifecycleCoordinatorState()
    teacher = _Manager()
    await coordinator.register("teacher", [teacher])
    await coordinator.register("teacher", [teacher])
    with pytest.raises(InferenceLifecycleError, match="already been registered"):
        await coordinator.register("teacher", [_Manager()])
    with pytest.raises(ValueError, match="at least one"):
        await coordinator.register("genrm", [])
    with pytest.raises(ValueError, match="Unknown inference role"):
        await coordinator.register("actor", [_Manager()])
    await coordinator.begin_batch(0)
    with pytest.raises(InferenceLifecycleError, match="before the first batch"):
        await coordinator.register("genrm", [_Manager()])


@pytest.mark.asyncio
@pytest.mark.parametrize("rollout_id", [True, -1, "", " 1", None, []])
async def test_invalid_batch_identity_is_rejected(rollout_id: Any) -> None:
    coordinator = LifecycleCoordinatorState()
    with pytest.raises(ValueError, match="rollout_id"):
        await coordinator.begin_batch(rollout_id)


def test_actor_wrapper_reserves_cpu_without_gpu() -> None:
    from relax.distributed.ray.inference_lifecycle import InferenceLifecycleCoordinator

    assert InferenceLifecycleCoordinator._default_options["num_cpus"] == 1
    assert InferenceLifecycleCoordinator._default_options["num_gpus"] == 0
