# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise checkpoint step traversal without importing the Megatron
runtime."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest
import torch


class _HybridOptimizer:
    def __init__(self, *steps: int) -> None:
        self.state = {index: {"step": torch.tensor(float(step))} for index, step in enumerate(steps)}
        self.param_groups = [{"step": 0}]
        self.sub_optimizers = [SimpleNamespace(param_groups=[{"step": 0}])]


class _ChainedOptimizer:
    def __init__(self, *children: object) -> None:
        self.chained_optimizers = list(children)

    @property
    def optimizer(self) -> object:
        raise AssertionError("ChainedOptimizer.optimizer must not be accessed when traversing children")


@pytest.fixture
def sync_steps(monkeypatch: pytest.MonkeyPatch) -> Callable[[object], None]:
    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/checkpoint.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_sync_hybrid_optimizer_checkpoint_steps"
    )
    module = ModuleType("megatron.core.optimizer.cpu_offloading")
    module.HybridDeviceOptimizer = _HybridOptimizer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def test_checkpoint_optimizer_steps_preserves_each_chained_child(sync_steps: Callable[[object], None]) -> None:
    first, second = _HybridOptimizer(7, 7), _HybridOptimizer(11)
    chain = _ChainedOptimizer(SimpleNamespace(optimizer=first), _ChainedOptimizer(second), _ChainedOptimizer())
    # Wrappers can share child references; cyclic containers must also terminate.
    chain.chained_optimizers.extend([first, chain])

    sync_steps(chain)

    for optimizer, expected in ((first, 7), (second, 11)):
        assert optimizer.param_groups == [{"step": expected}]
        assert optimizer.sub_optimizers[0].param_groups == [{"step": expected}]


@pytest.mark.parametrize("steps, expected", [((), 0), ((5,), 5)])
def test_checkpoint_optimizer_steps_handles_plain_wrapper(
    sync_steps: Callable[[object], None], steps: tuple[int, ...], expected: int
) -> None:
    optimizer = _HybridOptimizer(*steps)

    sync_steps(SimpleNamespace(optimizer=optimizer))

    assert optimizer.param_groups == [{"step": expected}]
    assert optimizer.sub_optimizers[0].param_groups == [{"step": expected}]


def test_checkpoint_optimizer_steps_rejects_inconsistent_child(sync_steps: Callable[[object], None]) -> None:
    optimizer = _HybridOptimizer(7, 8)

    with pytest.raises(RuntimeError, match="inconsistent checkpoint steps"):
        sync_steps(_ChainedOptimizer(optimizer))

    assert optimizer.param_groups == [{"step": 0}]
