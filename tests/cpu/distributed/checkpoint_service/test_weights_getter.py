# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for the DCS ``weights_getter`` override (hybrid-mode TensorBackuper
push source).

Covers:

- The live-model fallback when no ``weights_getter`` is set.
- CUDA snapshot staging tests live under tests/distributed/checkpoint_service/.
- ``CheckpointEngineClient`` threading ``weights_getter`` through to the backend.
- ``MegatronTrainRayActor.update_weights_fully_async`` with ``rollout_only=True``
  skipping actor_fwd process-group init (hybrid mode has no actor_fwd service).
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


def _bare_backend(*, weights_getter, device, use_bridge=False) -> DeviceDirectBackend:
    """Build a DeviceDirectBackend without running its heavy ``__init__``.

    ``_named_params_and_buffers`` only reads the handful of attributes set
    here, so we bypass the constructor (which pulls in httpx, megatron dep
    loading, and thread pools) and wire them by hand.
    """
    backend = DeviceDirectBackend.__new__(DeviceDirectBackend)
    backend._weights_getter = weights_getter
    backend.device = device
    backend.args = Namespace()
    backend.model = [torch.nn.Linear(2, 2)]
    backend._use_bridge = use_bridge
    backend._megatron = SimpleNamespace()  # live-path tests replace this
    return backend


def test_weights_getter_none_falls_back_to_live_model():
    recorded = {}

    def fake_named_params_and_buffers(args, model, include_persistent_buffers=False):
        recorded["args"] = args
        recorded["model"] = model
        recorded["include_persistent_buffers"] = include_persistent_buffers
        yield "module.w", torch.randn(2)

    for use_bridge in (True, False):
        recorded.clear()
        backend = _bare_backend(weights_getter=None, device=torch.device("cpu"), use_bridge=use_bridge)
        backend._megatron = SimpleNamespace(named_params_and_buffers=fake_named_params_and_buffers)

        [(name, _)] = list(backend._named_params_and_buffers())

        assert name == "module.w"
        assert recorded["include_persistent_buffers"] is use_bridge
        assert recorded["args"] is backend.args
        assert recorded["model"] is backend.model


def test_bridge_snapshot_names_are_initialized_before_first_sync(monkeypatch):
    from relax.distributed.checkpoint_service.backends import device_direct
    from relax.distributed.checkpoint_service.config import BackendType

    model = torch.nn.Linear(2, 2, bias=False)
    snapshot = model.weight.detach().clone()

    def named_params(_args, _model, *, convert_to_global_name, include_persistent_buffers):
        assert include_persistent_buffers
        yield "layers.3.weight" if convert_to_global_name else "module.weight", model.weight

    monkeypatch.setattr(
        device_direct,
        "_load_megatron_dependencies",
        lambda: SimpleNamespace(named_params_and_buffers=named_params, get_dcs_bucket_splitter=lambda *a, **kw: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "relax.backends.megatron.weight_update.bridge_converter",
        SimpleNamespace(BridgeConverter=lambda **kw: object()),
    )
    monkeypatch.setattr(device_direct.device_module, "set_device", lambda _: None)
    monkeypatch.setattr(device_direct.httpx, "Client", MagicMock())
    monkeypatch.setattr(device_direct, "is_lora_enabled", lambda _: False)
    backend = DeviceDirectBackend(
        Namespace(megatron_to_hf_mode="bridge"),
        BackendType.GLOO,
        None,
        [model],
        model_name="qwen3",
        quantization_config=None,
        weights_getter=lambda: {"module.weight": snapshot},
    )
    try:
        [(name, tensor)] = list(backend._named_params_and_buffers())
        assert name == "layers.3.weight"
        assert tensor is snapshot
    finally:
        backend.http_client.close()
        backend._thread_pool.shutdown()


def test_background_backend_binds_model_device_before_delta_exporter(monkeypatch):
    """The async DCS constructor must not inherit a training-thread device.

    Execute the real background loop and constructor with a thread-local device
    boundary. Real CUDA/NCCL placement remains a hardware gate.
    """
    from relax.backends.megatron.weight_update import delta_export
    from relax.distributed.checkpoint_service.backends import device_direct
    from relax.distributed.checkpoint_service.config import BackendType
    from relax.utils.async_utils import run

    state = threading.local()
    state.device = "main-thread-only"
    main_thread = threading.get_ident()
    model = torch.nn.Linear(2, 2, bias=False)
    observed = []
    monkeypatch.setattr(
        device_direct,
        "device_module",
        SimpleNamespace(
            current_device=lambda: getattr(state, "device", "unbound-background-device"),
            set_device=lambda device: setattr(state, "device", device),
        ),
    )
    monkeypatch.setattr(
        device_direct,
        "_load_megatron_dependencies",
        lambda: SimpleNamespace(get_dcs_bucket_splitter=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(device_direct.httpx, "Client", MagicMock())
    monkeypatch.setattr(device_direct, "is_lora_enabled", lambda args: False)

    def create_exporter(args, models, **kwargs):
        device = next(models[0].parameters()).device
        if device_direct.device_module.current_device() != device:
            raise RuntimeError("Delta exporter entered before binding the background thread to its model device")
        observed.append((threading.get_ident(), device))
        return object()

    monkeypatch.setattr(delta_export, "create_delta_exporter", create_exporter)

    async def construct():
        backend = DeviceDirectBackend(
            Namespace(megatron_to_hf_mode="raw"),
            BackendType.GLOO,
            SimpleNamespace(role_name="actor"),
            [model],
            model_name="qwen3",
            quantization_config=None,
        )
        try:
            return backend.device
        finally:
            backend.http_client.close()
            backend._thread_pool.shutdown()

    assert run(construct()) == model.weight.device
    assert len(observed) == 1
    assert observed[0][1] == model.weight.device
    assert observed[0][0] != main_thread
    assert state.device == "main-thread-only"


def test_engine_client_threads_weights_getter(monkeypatch):
    from relax.distributed.checkpoint_service.client import engine as engine_mod

    def weights_getter():
        return {"w": torch.randn(2)}

    captured = {}

    class _CaptureBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(engine_mod, "DeviceDirectBackend", _CaptureBackend)
    client = engine_mod.CheckpointEngineClient(
        args=Namespace(),
        coordinator_url="http://127.0.0.1:9",
        role="actor",
        rank=0,
        weights_getter=weights_getter,
    )

    asyncio.run(client._init_backend(lock=None))

    assert isinstance(client._backend, _CaptureBackend)
    assert captured["weights_getter"] is weights_getter


def _import_actor_module():
    """Import ``relax.backends.megatron.actor`` in this environment.

    transformers>=5 ships a built-in ``qwen3_asr`` config, so megatron-bridge's
    vendored ``Auto*.register("qwen3_asr", ...)`` calls raise ``ValueError`` —
    which relax/backends/megatron/__init__.py's ``except ImportError`` guards
    don't cover. Swallow duplicate-registration ValueErrors for the duration of
    the import (the registrations are irrelevant to what this test exercises).
    """
    from transformers import AutoConfig, AutoModel, AutoProcessor

    originals = {}
    for auto_cls in (AutoConfig, AutoModel, AutoProcessor):
        original = auto_cls.register
        originals[auto_cls] = original

        def _tolerant_register(cls, *args, __original=original, **kwargs):
            try:
                return __original(*args, **kwargs)
            except ValueError:
                return None

        auto_cls.register = classmethod(_tolerant_register)
    try:
        import relax.backends.megatron.actor as actor_mod
    finally:
        for auto_cls, original in originals.items():
            auto_cls.register = original
    return actor_mod


def test_rollout_only_skips_actor_fwd_group_init(monkeypatch):
    pytest.importorskip("megatron.core")
    actor_mod = _import_actor_module()

    actor = actor_mod.MegatronTrainRayActor.__new__(actor_mod.MegatronTrainRayActor)
    actor._weight_sync_lock = None
    client = MagicMock()
    actor.checkpoint_engine_client = client

    def fake_run(awaitable):
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        return None

    monkeypatch.setattr(actor_mod, "run", fake_run)
    monkeypatch.setattr(actor_mod, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_mod, "get_gloo_group", lambda: None)
    monkeypatch.setattr(dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    actor.update_weights_fully_async(0, rollout_only=True)

    # rollout_only push (hybrid has no actor_fwd service): no actor_fwd/ref group
    # init, and the rollout push goes through with actor_fwd_only=False.
    client.init_process_groups_for_actor_fwd_ref.assert_not_called()
    client.update_weights_for_rollout.assert_called_once_with(True, False)
