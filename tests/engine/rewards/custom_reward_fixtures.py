# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import fcntl
import functools
import os
import time
from typing import Any


_sync_load_attribute_count = 0
_module_reload_count = globals().get("_module_reload_count", 0) + 1


def sync_pid_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, int]:
    del kwargs
    if getattr(args, "reward_delay", 0):
        time.sleep(args.reward_delay)
    return {"pid": os.getpid(), "index": sample.index}


async def async_pid_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, int]:
    del args, kwargs
    return {"pid": os.getpid(), "index": sample.index}


async def async_batch_pid_reward(args: Any, samples: list[Any], **kwargs: Any) -> list[dict[str, int]]:
    del args, kwargs
    pid = os.getpid()
    return [{"pid": pid, "index": sample.index, "batch_size": len(samples)} for sample in samples]


async def async_reload_count_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, int]:
    del args, kwargs
    return {"pid": os.getpid(), "index": sample.index, "reload_count": _module_reload_count}


class AsyncPidReward:
    async def __call__(self, args: Any, sample: Any, **kwargs: Any) -> dict[str, int]:
        del args, kwargs
        return {"pid": os.getpid(), "index": sample.index}


async_callable_pid_reward = AsyncPidReward()


def _wrapped_async_pid_reward(function):
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any):
        return function(*args, **kwargs)

    return wrapper


wrapped_async_pid_reward = _wrapped_async_pid_reward(async_pid_reward)


def sync_conditional_reward(args: Any, sample: Any, **kwargs: Any) -> int:
    del kwargs
    if args.fail_reward:
        raise ValueError(f"bad sample index={sample.index}")
    return sample.index


def _update_concurrency_counter(counter_path: str, delta: int) -> None:
    with open(counter_path, "r+", encoding="utf-8") as counter_file:
        fcntl.flock(counter_file.fileno(), fcntl.LOCK_EX)
        current, maximum = (int(value) for value in counter_file.read().split(","))
        counter_file.seek(0)
        counter_file.truncate()
        current += delta
        counter_file.write(f"{current},{max(maximum, current)}")
        counter_file.flush()
        fcntl.flock(counter_file.fileno(), fcntl.LOCK_UN)


def sync_concurrency_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    del sample, kwargs
    _update_concurrency_counter(args.counter_path, 1)
    try:
        time.sleep(args.reward_delay)
    finally:
        _update_concurrency_counter(args.counter_path, -1)
    return 1.0


def sync_batch_conditional_reward(args: Any, samples: list[Any], **kwargs: Any) -> list[dict[str, int]]:
    del kwargs
    if args.fail_reward:
        raise ValueError("batch reward failed")
    pid = os.getpid()
    return [{"pid": pid, "index": sample.index, "batch_size": len(samples)} for sample in samples]


def sync_batch_concurrency_reward(args: Any, samples: list[Any], **kwargs: Any) -> list[float]:
    del kwargs
    _update_concurrency_counter(args.counter_path, 1)
    try:
        time.sleep(args.reward_delay)
    finally:
        _update_concurrency_counter(args.counter_path, -1)
    return [1.0] * len(samples)


def sync_reload_count_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, int]:
    del args, kwargs
    return {"pid": os.getpid(), "index": sample.index, "reload_count": _module_reload_count}


def _sync_load_count_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, int]:
    del kwargs
    if getattr(args, "reward_delay", 0):
        time.sleep(args.reward_delay)
    return {"pid": os.getpid(), "index": sample.index, "load_count": _sync_load_attribute_count}


def __getattr__(name: str) -> Any:
    # load_function resolves this dynamic attribute once in each process. The
    # reward result exposes the count so the test can distinguish loading from
    # repeated reward invocations.
    global _sync_load_attribute_count
    if name == "sync_load_count_reward":
        _sync_load_attribute_count += 1
        return _sync_load_count_reward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
