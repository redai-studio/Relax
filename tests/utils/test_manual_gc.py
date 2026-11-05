# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import gc
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from relax.utils import memory_utils
from relax.utils.memory_utils import collect_manual_gc, manual_gc_eval, manual_gc_training


@pytest.fixture(autouse=True)
def restore_gc_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(memory_utils, "_manual_gc_enabled", None)
    monkeypatch.setattr(memory_utils, "_manual_gc_step", 0)
    enabled = gc.isenabled()
    try:
        yield
    finally:
        gc.enable() if enabled else gc.disable()


@pytest.fixture
def args() -> Namespace:
    return Namespace(manual_gc=True, manual_gc_interval=6, manual_gc_eval=True, num_rollout=3)


@pytest.fixture
def collect(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mocked = Mock(return_value=0)
    monkeypatch.setattr(gc, "collect", mocked)
    return mocked


def test_disabled_manual_gc_does_not_change_collection_or_state(args: Namespace, collect: Mock) -> None:
    args.manual_gc = False
    enabled = gc.isenabled()

    @manual_gc_training
    def train(actor: SimpleNamespace, rollout_id: int) -> str:
        collect_manual_gc(actor.args)
        with manual_gc_eval(actor.args):
            collect_manual_gc(actor.args, final=True)
        return "early return"

    assert train(SimpleNamespace(args=args), 2) == "early return"
    collect.assert_not_called()
    assert gc.isenabled() == enabled


@pytest.mark.parametrize("interval,expected", [(6, [0, 6, 12, 14]), (0, [14])])
@pytest.mark.parametrize("initially_enabled", [True, False])
def test_collection_counts_actual_steps_across_variable_rollouts(
    args: Namespace, collect: Mock, interval: int, expected: list[int], initially_enabled: bool
) -> None:
    gc.enable() if initially_enabled else gc.disable()
    args.manual_gc_interval = interval
    original_args = vars(args).copy()
    updates = (2, 5, 7)
    completed = []
    boundaries = []
    collect.side_effect = lambda: boundaries.append(len(completed)) or 0

    @manual_gc_training
    def train(actor: SimpleNamespace, rollout_id: int) -> int:
        assert not gc.isenabled()
        for _ in range(updates[rollout_id]):
            collect_manual_gc(actor.args)
            assert vars(actor.args) == original_args
            completed.append(rollout_id)
        if rollout_id == 2:
            collect_manual_gc(actor.args, final=True)
        return len(completed)

    actor = SimpleNamespace(args=args)
    # A new lifecycle in the same process must start counting from zero again.
    for _ in range(2):
        completed.clear()
        boundaries.clear()
        assert train(actor, 0) == 2
        assert not gc.isenabled()
        assert train(actor, 1) == 7
        assert not gc.isenabled()
        assert train(actor, 2) == 14
        assert boundaries == expected
        assert gc.isenabled() == initially_enabled
        assert memory_utils._manual_gc_enabled is None
        assert memory_utils._manual_gc_step == 0
        assert vars(args) == original_args


@pytest.mark.parametrize("phase", ["fetch", "train", "eval"])
@pytest.mark.parametrize("initially_enabled", [True, False])
def test_training_failure_restores_original_gc_state(
    args: Namespace, collect: Mock, phase: str, initially_enabled: bool
) -> None:
    gc.enable() if initially_enabled else gc.disable()
    error = RuntimeError(f"{phase} failed")

    @manual_gc_training
    def train(actor: SimpleNamespace, rollout_id: int) -> None:
        assert not gc.isenabled()
        if phase == "fetch":
            raise error
        collect_manual_gc(actor.args)
        if phase == "train":
            raise error
        with manual_gc_eval(actor.args):
            raise error

    with pytest.raises(RuntimeError) as caught:
        train(SimpleNamespace(args=args), 0)
    assert caught.value is error
    assert gc.isenabled() == initially_enabled
    assert memory_utils._manual_gc_enabled is None
    assert memory_utils._manual_gc_step == 0


@pytest.mark.parametrize("manual,eval_gc", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_eval_collection_obeys_flags_and_runs_cleanup_after_failure(
    args: Namespace, collect: Mock, manual: bool, eval_gc: bool, fail: bool
) -> None:
    args.manual_gc, args.manual_gc_eval = manual, eval_gc
    enabled = gc.isenabled()

    def evaluate() -> None:
        with manual_gc_eval(args):
            assert collect.call_args_list == ([call()] if manual and eval_gc else [])
            if fail:
                raise RuntimeError("eval failed")

    if fail:
        with pytest.raises(RuntimeError, match="eval failed"):
            evaluate()
    else:
        evaluate()
    assert collect.call_args_list == ([call(), call(0)] if manual and eval_gc else [])
    assert gc.isenabled() == enabled


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_negative_interval_fails_before_touching_gc_state(
    args: Namespace, collect: Mock, initially_enabled: bool
) -> None:
    gc.enable() if initially_enabled else gc.disable()
    args.manual_gc_interval = -1
    original = Mock()
    train = manual_gc_training(original)
    with pytest.raises(ValueError, match="manual_gc_interval"):
        train(SimpleNamespace(args=args), 0)
    original.assert_not_called()
    collect.assert_not_called()
    assert gc.isenabled() == initially_enabled


def test_train_one_step_collects_after_clearing_gradients(args: Namespace, collect: Mock) -> None:
    # Execute the production boundary, stopping before model computation needs CUDA.
    source = Path(__file__).resolve().parents[2] / "relax/backends/megatron/model.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_step"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(function)
    args.custom_megatron_before_train_step_hook_path = None
    namespace = {
        "get_args": lambda: args,
        "capture_hooks": Mock(),
        "collect_manual_gc": collect_manual_gc,
        "PackingMetrics": Mock(side_effect=RuntimeError("stop before model computation")),
    }
    exec(compile(module, str(source), "exec"), namespace)
    events = []
    chunk = SimpleNamespace(zero_grad_buffer=lambda: events.append("model gradients"))
    optimizer = SimpleNamespace(zero_grad=lambda: events.append("optimizer gradients"))
    collect.side_effect = lambda: events.append("gc")
    for step in range(7):
        events.clear()
        with pytest.raises(RuntimeError, match="stop before model computation"):
            namespace["train_one_step"](args, step // 2, step % 2, [], [chunk], optimizer, None, 1, 1)
        assert events == ["model gradients", "optimizer gradients"] + (["gc"] if step in (0, 6) else [])
