# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Caller budgets retain their scope across nested operations and failures."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.distributed.weight_sync import deadline


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_returns_before_entering_operation(value):
    entered = False
    with pytest.raises(ValueError):
        with deadline.time_budget(value):
            entered = True
    assert not entered


def test_nested_budget_preserves_parent_and_restores_after_exception(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(deadline.time, "monotonic", lambda: clock[0])
    with deadline.time_budget(10):
        clock[0] = 102
        with deadline.time_budget(20):
            assert deadline.operation_remaining(99) == 8
        with pytest.raises(RuntimeError):
            with deadline.time_budget(2):
                assert deadline.operation_remaining(99) == 2
                raise RuntimeError("operation failed")
        assert deadline.operation_remaining(99) == 8
        clock[0] = 110
        with pytest.raises(TimeoutError, match="resume the same target"):
            deadline.checkpoint("storage commit")
    assert deadline.operation_remaining(3) == 3


def test_independent_operation_budget_and_thread_scope(monkeypatch):
    monkeypatch.setattr(deadline.time, "monotonic", lambda: 100.0)
    with deadline.time_budget(2):
        with deadline.time_budget(10, inherit=False):
            assert deadline.operation_remaining(99) == 10
        assert deadline.operation_remaining(99) == 2
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(deadline.operation_remaining, 20).result() == 20
