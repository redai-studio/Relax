# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tests for the DCS ``weights_getter`` override (hybrid-mode TensorBackuper
push source).

Covers:

- ``DeviceDirectBackend._named_params_and_buffers``: snapshot-path H2D coercion that
  preserves Megatron TP attributes (the highest-risk regression — dropping them makes
  DCS's ``all_gather_param`` silently gather/broadcast the wrong shards), device-
  resident snapshot passthrough identity, ``name_filter`` applied before the H2D copy,
  and the live-model fallback when no ``weights_getter`` is set.
- ``CheckpointEngineClient`` threading ``weights_getter`` through to the backend.
- ``MegatronTrainRayActor.update_weights_fully_async`` with ``rollout_only=True``
  skipping actor_fwd process-group init (hybrid mode has no actor_fwd service).
"""

from __future__ import annotations

import asyncio
import inspect
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


def _mark_tp(tensor: torch.Tensor, *, parallel_mode="column", partition_dim=0, partition_stride=1):
    """Attach Megatron TP attributes the way Megatron marks a column-parallel
    param."""
    tensor.tensor_model_parallel = True
    tensor.partition_dim = partition_dim
    tensor.partition_stride = partition_stride
    tensor.parallel_mode = parallel_mode
    return tensor


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H2D coercion requires CUDA")
def test_weights_getter_host_snapshot_h2d_copies_tp_attrs():
    cpu_tensor = _mark_tp(torch.randn(4, 3))
    backend = _bare_backend(
        weights_getter=lambda: {"module.layer.weight": cpu_tensor},
        device=torch.device("cuda:0"),
    )

    [(name, tensor)] = list(backend._named_params_and_buffers())

    assert name == "module.layer.weight"
    assert tensor.device.type == "cuda"
    assert torch.equal(tensor.cpu(), cpu_tensor)
    # TP attributes must survive the .to(device) reallocation.
    assert tensor.tensor_model_parallel is True
    assert tensor.partition_dim == 0
    assert tensor.partition_stride == 1
    assert tensor.parallel_mode == "column"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device passthrough requires CUDA")
def test_weights_getter_device_snapshot_passthrough():
    gpu_tensor = _mark_tp(torch.randn(2, 2, device="cuda"))
    backend = _bare_backend(
        weights_getter=lambda: {"w": gpu_tensor},
        device=torch.device("cuda:0"),
    )

    [(name, tensor)] = list(backend._named_params_and_buffers())

    assert name == "w"
    # Already device-resident: yielded verbatim, no reallocation.
    assert tensor is gpu_tensor


class _ExplodingH2DTensor(torch.Tensor):
    """Raises on any ``.to()`` copy — stands in for a filtered-out CPU snapshot
    tensor."""

    def to(self, *args, **kwargs):
        raise AssertionError("filtered-out tensor must never be moved to device")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="name_filter/H2D ordering requires CUDA")
def test_weights_getter_name_filter_applied_before_h2d():
    keep1 = torch.randn(4)
    keep2 = torch.randn(2, 2)
    expert = torch.randn(8).as_subclass(_ExplodingH2DTensor)
    snapshot = {
        "layer0.mlp.weight": keep1,
        "layer1.mlp.weight": keep2,
        "layer0.mlp.experts.w1": expert,
    }
    backend = _bare_backend(weights_getter=lambda: snapshot, device=torch.device("cuda:0"))

    pairs = list(backend._named_params_and_buffers(name_filter=lambda n: ".experts." not in n))

    # Only the non-expert tensors are yielded; the expert tensor was filtered out
    # before the H2D branch (its .to() would have raised otherwise).
    assert [name for name, _ in pairs] == ["layer0.mlp.weight", "layer1.mlp.weight"]
    for _, tensor in pairs:
        assert tensor.device.type == "cuda"


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
