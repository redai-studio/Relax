# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent communication-group policy and per-step offload contracts."""

import ast
import inspect
import json
import os
import textwrap
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron.train_process_groups import configure_train_process_group_retention


def _agreement_worker(rank: int, rendezvous: str, output: str, mode: str) -> None:
    requested = mode != "disabled" and (mode != "mismatch" or rank == 0)
    os.environ["RELAX_COLOCATE_KEEP_TRAIN_PROCESS_GROUPS"] = str(int(requested))
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        args = SimpleNamespace(colocate=True, offload_train=True)
        error, selected = None, None
        try:
            selected = configure_train_process_group_retention(
                args, SimpleNamespace(uses_tms=mode == "tms"), dist.group.WORLD
            )
        except ValueError as exc:
            error = str(exc)
        Path(output, f"{rank}.json").write_text(json.dumps({"selected": selected, "error": error}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["disabled", "enabled", "mismatch", "tms"])
def test_train_group_policy_agrees_before_offload(tmp_path: Path, mode: str) -> None:
    context = mp.spawn(_agreement_worker, args=(str(tmp_path / "gloo"), str(tmp_path), mode), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 60
        while not context.join(timeout=5):
            if time.monotonic() >= deadline:
                raise TimeoutError("training-group policy agreement did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    records = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    assert records[0] == records[1]
    if mode in ("disabled", "enabled"):
        assert records[0] == {"selected": mode == "enabled", "error": None}
    else:
        assert records[0]["selected"] is None
        assert ("disagree" if mode == "mismatch" else "--selective-offload") in records[0]["error"]


@pytest.fixture
def actor_module():
    pytest.importorskip("megatron.core")
    from relax.backends.megatron import actor

    return actor


@pytest.mark.parametrize("keep", [False, True])
def test_per_step_sleep_keeps_groups_but_explicit_sleep_cleans(monkeypatch, actor_module, keep):
    module = actor_module
    actor = object.__new__(module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(offload_train=True, use_critic=False)
    actor.role = "actor"
    actor._keep_train_process_groups = keep
    actor._flush_resident_sft_parameters = Mock()
    actor._train_state_offloader = SimpleNamespace(offload=Mock())
    destroy = Mock()
    monkeypatch.setattr(module, "clear_memory", lambda **kwargs: None)
    monkeypatch.setattr(module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "destroy_process_groups", destroy)
    module.MegatronTrainRayActor.sleep.__wrapped__(actor, keep_process_groups=keep)
    assert destroy.call_count == (0 if keep else 1)
    actor._train_state_offloader.offload.assert_called_once_with()
    destroy.reset_mock()
    module.MegatronTrainRayActor.sleep.__wrapped__(actor)
    destroy.assert_called_once_with()


@pytest.mark.parametrize("keep,reconnect", [(False, False), (True, False), (True, True)])
def test_sync_completion_obeys_group_policy_and_preserves_reconnect_cleanup(
    monkeypatch, actor_module, keep, reconnect
):
    module = actor_module
    actor = object.__new__(module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=True,
        offload_rollout=False,
        use_fault_tolerance=False,
        use_critic=reconnect,
        colocate=not reconnect,
        distributed_timeout_minutes=5,
        ci_test=False,
        keep_old_actor=False,
    )
    actor._keep_train_process_groups = keep
    actor._flush_resident_sft_parameters = Mock()
    actor._train_state_offloader = SimpleNamespace(disable_during_update=nullcontext)
    actor.weight_updater = SimpleNamespace(update_weights=Mock(), connect_rollout_engines=Mock())
    actor.rollout_manager = SimpleNamespace(
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: ([], None, 0, [], [])),
        clear_num_new_engines=SimpleNamespace(remote=lambda: None),
    )
    actor.sleep, actor.wake_up = Mock(), Mock()
    destroy = Mock()
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "reload_process_groups", Mock())
    monkeypatch.setattr(module, "destroy_process_groups", destroy)
    monkeypatch.setattr(module, "print_memory", lambda *args, **kwargs: None)
    module.MegatronTrainRayActor.update_weights.__wrapped__(actor)
    actor.weight_updater.update_weights.assert_called_once_with()
    assert destroy.call_count == (0 if keep or reconnect else 1)
    assert actor.sleep.call_count == int(reconnect)
    assert actor.wake_up.call_count == int(reconnect)


@pytest.mark.parametrize("pending", [False, True])
def test_train_transition_offloads_once_or_defers_until_sync(actor_module, pending):
    # Execute the actual training epilogue without mocking the condition under test.
    method = inspect.unwrap(actor_module.MegatronTrainRayActor.train_actor)
    body = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0].body
    transition = next(
        node for node in body if isinstance(node, ast.If) and ast.unparse(node.test) == "self._per_step_rollout"
    )
    code = compile(ast.Module(body=[transition], type_ignores=[]), inspect.getsourcefile(method), "exec")
    events = []
    actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=True),
        _per_step_rollout=True,
        _actor_weights_pending_sync=pending,
        _keep_train_process_groups=True,
        sleep=lambda **kwargs: events.append(("sleep", kwargs)),
        update_weights=lambda: events.append(("sync", None)),
    )
    exec(code, {"self": actor, "has_rollout": True, "colocate_transition_started": None})
    expected = [] if pending else [("sleep", {"keep_process_groups": True})]
    assert events == expected + [("sync", None)]
