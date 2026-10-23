# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Torch 2.13 collective delegation; real smoke tests use an isolated CPU Gloo
group."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import pytest


torch = pytest.importorskip("torch")
dist = torch.distributed

from relax.utils import reloadable_process_group as runtime  # noqa: E402


_NEW_METHODS = (
    "reduce_scatter_single",
    "reduce_scatter_single_coalesced",
    "all_gather_single",
    "all_gather_single_coalesced",
    "all_to_all_single",
)


class _InnerGroup:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.result = object()

    def __getattr__(self, name: str) -> Any:
        if name not in _NEW_METHODS:
            raise AttributeError(name)

        def collective(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return self.result

        return collective


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(runtime.ReloadableProcessGroup, "GROUPS", {})
    with monkeypatch.context() as construction:
        construction.setattr(dist, "get_rank", lambda group: 0)
        construction.setattr(dist, "get_world_size", lambda group: 1)
        return runtime.ReloadableProcessGroup(_InnerGroup(), [0])


@pytest.mark.parametrize("method", _NEW_METHODS)
def test_new_collective_forwards_arguments_result_and_memory_guard(
    wrapper: Any, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    entered = []

    @contextmanager
    def memory_guard() -> Any:
        entered.append("enter")
        yield
        entered.append("exit")

    monkeypatch.setattr(runtime, "_wrap_low_level_call", memory_guard)
    output, input, options = object(), object(), object()

    result = getattr(wrapper, method)(output, input, opts=options)

    assert result is wrapper.group.result
    assert wrapper.group.calls == [(method, (output, input), {"opts": options})]
    assert entered == ["enter", "exit"]
    assert method in runtime.ReloadableProcessGroup.__dict__, "a C++ method cannot rely on __getattr__ delegation"


@pytest.mark.parametrize("method", _NEW_METHODS)
def test_new_collective_uses_reloaded_inner_group_not_old_bound_method(
    wrapper: Any, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    old_group = wrapper.group
    captured_call = getattr(wrapper, method)
    first_output, second_output = object(), object()
    assert captured_call(first_output) is old_group.result
    replacement = _InnerGroup()
    created = []

    def new_group(**kwargs: Any) -> Any:
        created.append(kwargs)
        return replacement

    monkeypatch.setitem(runtime.old_new_group_dict, os.getpid(), new_group)
    monkeypatch.setattr(runtime.device_utils, "get_dist_backend", lambda: "gloo")
    wrapper.group = None

    runtime.ReloadableProcessGroup.reload_process_groups(timeout_minutes=1, max_retries=1, retry_delay=0)
    result = captured_call(second_output)

    assert result is replacement.result
    assert wrapper.group is replacement
    assert old_group.calls == [(method, (first_output,), {})]
    assert replacement.calls == [(method, (second_output,), {})]
    assert created == [{"ranks": [0], "backend": "gloo", "timeout": timedelta(minutes=1)}]


@pytest.mark.parametrize("method", _NEW_METHODS)
def test_new_collective_fails_before_dispatch_when_inner_group_is_destroyed(wrapper: Any, method: str) -> None:
    old_group = wrapper.group
    wrapper.group = None

    with pytest.raises(RuntimeError, match="inner PG is None"):
        getattr(wrapper, method)(object(), object())

    assert old_group.calls == []


@pytest.mark.parametrize(
    "api_name",
    [
        "reduce_scatter_single",
        "all_gather_single",
        "all_to_all_single",
        "reduce_scatter_tensor",
        "all_gather_into_tensor",
    ],
)
@pytest.mark.parametrize("async_op", [False, True])
def test_public_collective_and_deprecated_alias_use_real_gloo_wrapper_without_cuda(
    monkeypatch: pytest.MonkeyPatch, api_name: str, async_op: bool
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires a CPU Gloo backend")
    if not hasattr(dist, api_name) or not hasattr(dist.ProcessGroup, "reduce_scatter_single"):
        pytest.skip("requires the target Torch 2.13 collective APIs")
    if dist.is_initialized():
        pytest.skip("requires its own isolated default group; must not destroy another test's process group")
    if torch.cuda.is_initialized():
        pytest.skip("this CPU-only smoke requires a fresh process without CUDA initialization")
    monkeypatch.setattr(runtime.ReloadableProcessGroup, "GROUPS", {})
    monkeypatch.setattr(runtime, "_wrap_low_level_call_enabled", False)
    subgroup = None
    wrapper = None
    try:
        dist.init_process_group(
            backend="gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=10),
        )
        wrapper = runtime.ReloadableProcessGroup(dist.group.WORLD, [0])
        subgroup = dist.new_group(ranks=[0], backend="gloo", timeout=timedelta(seconds=10))
        # Exercise both the original and replacement real groups with the same wrapper.
        for step, group in enumerate((dist.group.WORLD, subgroup)):
            wrapper.group = group
            input = torch.arange(4, dtype=torch.float32) + step * 10
            output = torch.empty_like(input)
            try:
                work = getattr(dist, api_name)(output, input, group=wrapper, async_op=async_op)
                if async_op:
                    assert work is not None
                    work.wait()
            except RuntimeError as error:
                if api_name == "all_to_all_single" and "does not support" in str(error).lower():
                    pytest.skip(f"installed Gloo does not support all-to-all: {error}")
                raise
            torch.testing.assert_close(output, input)
        assert not torch.cuda.is_initialized()
    finally:
        if wrapper is not None:
            wrapper.group = None
        if subgroup is not None:
            dist.destroy_process_group(subgroup)
        if dist.is_initialized():
            dist.destroy_process_group()
