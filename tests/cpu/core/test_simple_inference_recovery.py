# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _source():
    path = Path(__file__).parents[3] / "relax/core/inference_recovery.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    namespace = {
        "Namespace": SimpleNamespace,
        "Any": object,
        "time": SimpleNamespace(sleep=lambda _: None),
        "ray": SimpleNamespace(),
        "run": asyncio.run,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _remote(method):
    return SimpleNamespace(remote=method)


def test_budget_shared_between_both_inference_roles_and_fourth_requires_global():
    state = _source()["InferenceRecoveryState"](True)
    for index, role in enumerate(["rollout", "actor_fwd", "rollout"], 1):
        state.report_error(role, "explicit error")
        assert state.request_recovery(role) == index
        assert state.resume(index) == index
    state.report_error("actor_fwd", "explicit error")
    with pytest.raises(RuntimeError, match="exhausted"):
        state.request_recovery("actor_fwd")


@pytest.mark.parametrize("max_attempts", [0, 1, 5])
def test_configured_budget_is_shared_and_not_reset_after_success(max_attempts):
    state = _source()["InferenceRecoveryState"](True, max_attempts=max_attempts)
    for index in range(1, max_attempts + 1):
        role = "rollout" if index % 2 else "actor_fwd"
        state.report_error(role, "explicit error")
        assert state.request_recovery(role) == index
        state.resume(index)
    state.report_error("rollout", "explicit error")
    with pytest.raises(RuntimeError, match=rf"budget \({max_attempts}\) exhausted"):
        state.request_recovery("rollout")


def test_failed_producer_closes_admission_before_any_reset():
    state = _source()["InferenceRecoveryState"](False)
    assert state.enter("actor", 4, 0)
    state.report_error("rollout", "ValueError")
    assert not state.enter("advantages", 4, 0)
    with pytest.raises(RuntimeError, match="not stopped"):
        state.resume(4)
    state.leave("actor")
    state.request_recovery("rollout")
    state.resume(5)
    assert not state.enter("actor", 4, 0)
    assert state.enter("actor", 5, 1)


def test_sync_ready_is_consumed_and_failure_during_sync_cannot_recover_locally():
    state = _source()["InferenceRecoveryState"](True)
    assert not state.begin_sync()
    state.set_fwd_ready(True)
    assert state.begin_sync()
    assert not state.snapshot()["fwd_ready"]
    state.report_error("rollout", "bad response")
    state.end_sync()
    with pytest.raises(RuntimeError, match="Concurrent"):
        state.request_recovery("rollout")


def test_second_failed_role_does_not_get_another_local_recovery():
    state = _source()["InferenceRecoveryState"](True)
    state.report_error("actor_fwd", "forward error")
    state.report_error("actor", "training error")
    with pytest.raises(RuntimeError, match="Concurrent"):
        state.request_recovery("actor_fwd")


def test_healthy_receiver_can_be_released_while_all_new_work_is_paused():
    state = _source()["InferenceRecoveryState"](True)
    state.report_error("rollout", "ValueError")
    state.set_fwd_ready(True)
    assert not state.begin_sync()
    assert state.begin_sync(maintenance=True)
    assert not state.enter("actor", 3, 0)
    state.end_sync()
    state.request_recovery("rollout")
    state.resume(3)


def test_wait_all_observes_other_ranks_after_first_rank_error():
    module = _source()
    observed = []

    def get(ref):
        observed.append(ref)
        if ref == "bad":
            raise ValueError("rank failed")
        return ref

    module["ray"].wait = lambda refs, **kw: (refs, [])
    module["ray"].get = get
    with pytest.raises(ValueError, match="rank failed"):
        module["wait_all"](["bad", "good"])
    assert observed == ["bad", "good"]


def test_wait_all_unknown_rank_is_unsafe_without_cancelling_it():
    module = _source()
    module["ray"].wait = lambda refs, **kw: ([refs[0]], refs[1:])
    module["ray"].get = Mock()
    with pytest.raises(module["UnsafeInferenceError"]):
        module["wait_all"](["bad", "still running"])
    module["ray"].get.assert_not_called()


@pytest.mark.parametrize(
    "rows,columns,ready,expected", [(4, 2, True, True), (3, 2, True, False), (4, 1, True, False), (4, 2, False, False)]
)
def test_admission_requires_all_expected_rows_and_fields(rows, columns, ready, expected):
    module = _source()
    gate = module["RecoveryGate"](SimpleNamespace(rollout_batch_size=2, n_samples_per_prompt=2), "actor")

    class Status:
        shape = rows, columns

        def __eq__(self, value):
            return self

        def all(self):
            return ready

    async def completed(partition):
        return True

    async def status(**kwargs):
        return list(range(rows)), Status()

    client = SimpleNamespace(async_check_production_completed=completed, async_get_production_status=status)
    assert gate._batch_ready(client, 1, ["tokens", "advantages"]) is expected


def test_pause_between_readiness_and_enter_prevents_dispatch():
    module = _source()
    state = module["InferenceRecoveryState"](False)
    stop = threading.Event()
    proxy = SimpleNamespace(snapshot=_remote(state.snapshot))

    def enter(role, step, generation):
        state.report_error("rollout", "raced with admission")
        stop.set()
        return state.enter(role, step, generation)

    proxy.enter = _remote(enter)
    gate = module["RecoveryGate"](SimpleNamespace(inference_recovery_state=proxy), "actor")
    module["ray"].get = lambda value: value
    assert gate.admit(2, stop_event=stop) is None
    assert state.active == {}


def test_generation_change_uses_original_actors_next_step():
    module = _source()
    state = module["InferenceRecoveryState"](False)
    state.resume(9)
    proxy = SimpleNamespace(snapshot=_remote(state.snapshot), enter=_remote(state.enter))
    gate = module["RecoveryGate"](SimpleNamespace(inference_recovery_state=proxy), "actor_fwd")
    module["ray"].get = lambda value: value
    assert gate.admit(12) == 9
    assert state.active == {"actor_fwd": 9}


def test_disabled_gate_has_no_state_or_tq_side_effects():
    module = _source()
    gate = module["RecoveryGate"](SimpleNamespace(), "actor")
    assert gate.admit(3, client=object(), fields=["tokens"]) == 3
    gate.leave()
    gate.fail(ValueError("legacy behavior"))


@pytest.mark.parametrize("flag,hybrid", [("", False), ("0", False), ("", True), ("0", True)])
def test_explicit_disable_skips_eligibility_checks(monkeypatch, flag, hybrid):
    import sys

    module = _source()
    monkeypatch.setenv("RELAX_EXPERIMENTAL_ROLLOUT_RECOVERY", flag)
    affinity = Mock()
    monkeypatch.setitem(
        sys.modules, "relax.core.node_group_affinity", SimpleNamespace(with_control_plane_affinity=affinity)
    )
    config = SimpleNamespace(hybrid=hybrid, inference_recovery=True, inference_recovery_state=object())
    assert module["configure_recovery"](config) is None
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None
    affinity.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("write_fails", [False, True])
async def test_generation_error_waits_for_storage_ack_without_cancelling_put(write_fails):
    path = Path(__file__).parents[3] / "relax/engine/rollout/sglang_rollout.py"
    tree = ast.parse(path.read_text())
    wrapper = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_rollout_async"
    )
    state = SimpleNamespace(pendings=set(), protected_pendings=set())
    acknowledged = []

    async def write():
        await asyncio.sleep(0.01)
        acknowledged.append(True)
        if write_fails:
            raise RuntimeError("Storage ACK missing")

    async def generate(*args):
        state.recovery_transfers.append(asyncio.create_task(write()))
        raise ValueError("invalid generated response")

    namespace = {
        "asyncio": asyncio,
        "Namespace": SimpleNamespace,
        "Callable": object,
        "Any": object,
        "Sample": object,
        "RolloutFnTrainOutput": object,
        "GenerateState": lambda args: state,
        "_generate_rollout_async": generate,
    }
    import typing

    namespace["Callable"] = typing.Callable
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(path), "exec"), namespace)
    # Install only the narrow exception type used by the real wrapper.
    import sys
    from unittest.mock import patch

    unsafe = _source()["UnsafeInferenceError"]
    with patch.dict(sys.modules, {"relax.core.inference_recovery": SimpleNamespace(UnsafeInferenceError=unsafe)}):
        with pytest.raises(unsafe if write_fails else ValueError):
            await namespace["generate_rollout_async"](SimpleNamespace(inference_recovery=True), 3, None, None)
    assert acknowledged == [True]
    assert not state.recovery_transfers[0].cancelled()
