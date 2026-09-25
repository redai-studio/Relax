# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU doubles for native request ownership; live hooks import real
ReqState."""

import asyncio
from types import SimpleNamespace

import pytest


def request_state():
    return SimpleNamespace(
        identity=None,
        acquired=False,
        acquire_complete=False,
        used_gpu=False,
        dispatched=False,
        abort_requested=False,
        drained=False,
        delivery="WAITING",
        acquire_task=None,
        release_task=None,
        error=None,
        completion=asyncio.Event(),
        consumer_detached=False,
    )


@pytest.fixture
def native_abort_type(monkeypatch):
    # Protocol-only tests do not install the CUDA-dependent SGLang package.
    import sys

    module = SimpleNamespace(AbortReq=lambda rid: SimpleNamespace(rid=rid, abort_all=False))
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", module)


def load_tokenizer_controls(root, native, rpc, monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType

    class Dependency(ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            value = type(name, (), {})
            setattr(self, name, value)
            return value

    for name in (
        "fastapi",
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.managers",
        "sglang.srt.managers.io_struct",
        "sglang.srt.managers.load_snapshot",
        "sglang.srt.server_args",
        "sglang.srt.utils",
        "sglang.srt.utils.msgspec_utils",
        "sglang.utils",
    ):
        stub = Dependency(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "sglang.srt.lora.version_control", native)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.communicator", rpc)
    sys.modules["sglang.srt.managers.io_struct"].AbortReq = lambda rid: SimpleNamespace(rid=rid, abort_all=False)
    sys.modules["sglang.srt.utils"].get_bool_env_var = lambda *a: False
    sys.modules["sglang.srt.server_args"].LoRARef = lambda **kw: SimpleNamespace(**kw)
    spec = importlib.util.spec_from_file_location(
        "native_tokenizer_controls", root / "python/sglang/srt/managers/tokenizer_control_mixin.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.TokenizerControlMixin
