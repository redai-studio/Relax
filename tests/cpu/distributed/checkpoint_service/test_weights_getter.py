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
import weakref
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


@pytest.mark.parametrize("rollout_only", [False, True])
def test_ep1_preserves_etp_reconstruction_and_releases_converted_source(monkeypatch, rollout_only):
    def no_ep_collective(*args, **kwargs):
        pytest.fail("EP=1 must not copy already reconstructed expert weights")

    monkeypatch.setattr(dist, "all_gather", no_ep_collective)
    monkeypatch.setattr(dist, "all_gather_object", no_ep_collective)
    gathered, raw, sent = [], [], []

    def gather(args, name, param, **options):
        assert options == {"copy_single_rank": True}
        full = torch.cat([param, param + 10])
        gathered.append(weakref.ref(full))
        return full

    def send(weights, pbar=None):
        assert not bucket and gathered[0]() is None
        sent.extend((name, value.clone()) for name, value in weights)

    backend = DeviceDirectBackend.__new__(DeviceDirectBackend)
    backend.args = Namespace(update_weight_buffer_size=1024, use_shard_p2p_weight_sync=True)
    backend._megatron = SimpleNamespace(
        all_gather_param=gather,
        mpu=SimpleNamespace(get_expert_model_parallel_world_size=lambda: 1),
    )
    backend._lora_enabled = backend._lora_merge_mode = backend._lora_adapter_mode = False
    backend._is_pp_src_rank = backend._use_bridge = True
    backend._bridge_converter = SimpleNamespace(convert=lambda name, param: [("hf." + name, param + 100)])
    backend._update_bucket_weights_from_distributed = send
    backend._update_bucket_weights_from_distributed_for_actor_fwd_ref = lambda weights: raw.extend(
        (name, value.clone()) for name, value in weights
    )
    bucket = []
    local = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    backend._update_expert_weight_from_distributed("expert", local, bucket, 0, rollout_only)
    backend._update_expert_bucket_weights_from_distributed(bucket, rollout_only)
    expected = torch.cat([local, local + 10])
    assert sent[0][0] == "hf.expert"
    torch.testing.assert_close(sent[0][1], expected + 100)
    assert len(raw) == int(not rollout_only)
    if raw:
        torch.testing.assert_close(raw[0][1], expected)


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


@pytest.mark.parametrize("world_size", [1, 2, None])
def test_actor_fwd_buckets_skip_only_when_topology_has_no_receivers(monkeypatch, world_size):
    from relax.distributed.checkpoint_service.backends import device_direct

    backend = _bare_backend(weights_getter=None, device=torch.device("cpu"))
    backend.role_info = SimpleNamespace(role_name="actor")
    backend.backend_type = "gloo"
    backend._model_update_groups_for_actor_fwd_ref = None
    backend._megatron.mpu = SimpleNamespace(
        get_pipeline_model_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda *, with_context_parallel: 0,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    backend.http_client = MagicMock()
    backend.coordinator_url = "http://127.0.0.1:9"
    group = object()
    init_group = MagicMock(return_value=group)
    monkeypatch.setattr(device_direct, "init_process_group", init_group)
    topology = {"pp_groups": {"update_actor_pp_0": "tcp://127.0.0.1:12345"}}
    if world_size is not None:
        topology["world_size"] = world_size
    backend.init_process_groups_for_actor_fwd_ref(topology)

    broadcast = MagicMock()
    monkeypatch.setattr(dist, "broadcast", broadcast)
    weight = torch.arange(4)
    backend._update_bucket_weights_from_distributed_for_actor_fwd_ref([("w", weight)])
    assert init_group.call_count == backend.http_client.post.call_count == broadcast.call_count == int(world_size != 1)
    if world_size != 1:
        assert broadcast.call_args.args[0].data_ptr() == weight.data_ptr()
        assert broadcast.call_args.args[1] == 0
        assert broadcast.call_args.kwargs == {"group": group, "async_op": True}
        broadcast.return_value.wait.assert_called_once_with()


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
