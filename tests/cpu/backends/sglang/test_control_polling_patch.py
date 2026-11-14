# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import asyncio
import functools
import sys
from types import FunctionType, SimpleNamespace

import pytest

from relax.backends.sglang.control_polling_patch import patch_control_poll_interval
from relax.utils.env import Envs


async def _health_generate(request):
    max_new_tokens = 1
    for _ in range(request):
        await asyncio.sleep(1)
    return max_new_tokens


async def _pause_generation(self, obj):
    if obj.mode == "abort":
        while True:
            self.aborts.append(True)
            if not self.locked.pop(0):
                break
            await asyncio.sleep(1.0)
    else:
        self.dispatched.append(obj.mode)


async def _unsupported(self, obj):
    await asyncio.sleep(0.5)


def _install(monkeypatch, pause=_pause_generation):
    health = FunctionType(_health_generate.__code__, globals(), _health_generate.__name__)
    pause = FunctionType(pause.__code__, globals(), pause.__name__)

    @functools.wraps(health)
    async def wrapped(request):
        return await health(request)

    routes = [SimpleNamespace(path=path, endpoint=wrapped) for path in ("/health", "/health_generate")]
    module = SimpleNamespace(health_generate=wrapped, app=SimpleNamespace(routes=routes))
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", SimpleNamespace(http_server=module))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.tokenizer_manager",
        SimpleNamespace(TokenizerManager=SimpleNamespace(pause_generation=pause)),
    )
    return module, health, pause


def test_poll_patch_preserves_routes_abort_and_other_one_constants(monkeypatch):
    monkeypatch.delenv("SGLANG_CONTROL_POLL_INTERVAL", raising=False)
    assert Envs.SGLANG_CONTROL_POLL_INTERVAL == 0.01
    module, health, pause = _install(monkeypatch)
    original = (health.__code__, pause.__code__)
    patch_control_poll_interval(1.0)
    assert (health.__code__, pause.__code__) == original
    route = module.app.routes[0].endpoint
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    patch_control_poll_interval(Envs.SGLANG_CONTROL_POLL_INTERVAL)
    patch_control_poll_interval(0.01)
    assert module.app.routes[0].endpoint is route
    assert asyncio.run(route(2)) == 1
    state = SimpleNamespace(aborts=[], locked=[True, True, False], dispatched=[])
    asyncio.run(pause(state, SimpleNamespace(mode="abort")))
    assert len(state.aborts) == 3 and not state.locked
    asyncio.run(pause(state, SimpleNamespace(mode="retract")))
    assert state.dispatched == ["retract"]
    assert delays == [0.01] * 4
    assert asyncio.sleep is sleep
    assert health._relax_control_poll_interval == 0.01
    with pytest.raises(RuntimeError, match="different interval"):
        patch_control_poll_interval(0.02)


def test_unsupported_second_function_does_not_partially_patch_first(monkeypatch):
    _module, health, pause = _install(monkeypatch, _unsupported)
    original = (health.__code__, pause.__code__)
    patch_control_poll_interval(0.01)
    assert (health.__code__, pause.__code__) == original
    assert not hasattr(health, "_relax_control_poll_interval")
    _module, health, pause = _install(monkeypatch)
    original = (health.__code__, pause.__code__)
    patch_control_poll_interval(0.01, tokenizer_worker_num=2)
    assert (health.__code__, pause.__code__) == original
    # A decorator cycle is also an incompatible introspection target.
    health.__wrapped__ = health
    patch_control_poll_interval(0.01)
    assert (health.__code__, pause.__code__) == original


@pytest.mark.parametrize("interval", [0, -1, 2, float("nan")])
def test_invalid_poll_interval_rejected(interval):
    with pytest.raises(ValueError):
        patch_control_poll_interval(interval)
