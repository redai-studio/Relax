# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration across real HF encoding, engine controls, TCP and loading.

Only Ray scheduling and HTTP service delivery are replaced. The tiny CPU model
does not establish real Megatron conversion or SGLang GPU model support.
"""

import json
from datetime import timedelta
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import requests
import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer
from torch import nn

from relax.backends.sglang.hf_memory_loader import HF_MEMORY_LOADER, load_hf_memory_weights
from relax.backends.sglang.memory_delta_loader import MEMORY_LOADER, load_memory_weights
from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.distributed.weight_sync.config import MemorySyncConfig
from relax.distributed.weight_sync.memory_delta import MemoryDeltaEncoder
from relax.distributed.weight_sync.streaming_delta import SnapshotBudget, StreamingDeltaEncoder
from relax.utils import distributed_utils


class _Remote:
    def __init__(self, function) -> None:
        self.function = function

    def remote(self, *args, **kwargs):
        return partial(self.function, *args, **kwargs)


def _get(refs, **_kwargs):
    return [_get(ref) for ref in refs] if isinstance(refs, list) else refs()


class _TinyModel(nn.Module):
    def __init__(self, precise_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.low = nn.Parameter(torch.zeros(64, dtype=torch.bfloat16), requires_grad=False)
        self.precise = nn.Parameter(torch.zeros(64, dtype=precise_dtype), requires_grad=False)
        self.omit_precise = False

    def load_weights(self, weights) -> None:
        for name, value in weights:
            if name == "precise" and self.omit_precise:
                continue
            getattr(self, name).copy_(value)


def _receiver(model):
    return getattr(model, "_relax_hf_memory_receiver", None) or model._relax_memory_receiver


# Actual scheduler graph-memory report schema, in GB, including draft phases.
_ZERO_GRAPH_USAGE = dict.fromkeys(
    ("prefill", "decode", "target_verify", "draft_prefill", "draft_decode", "draft_extend"), 0.0
)


class _EngineBoundary:
    """Fake Ray handle and HTTP server, using the actual engine install
    method."""

    def __init__(self, index: int, workers: int, updater, model_path: str) -> None:
        self._actor_id = SimpleNamespace(hex=lambda: f"engine-{index}")
        self.models = [_TinyModel(torch.float32 if updater.canonical_hf else torch.bfloat16) for _ in range(workers)]
        self.paused = False
        self.resumes = 0
        self.acks_on_failure: set[str] = set()
        self.install_observations: list[tuple[int, int, str]] = []
        self.encodings: list[tuple[str, ...]] = []
        self.updater = updater
        self.weight_requests = 0
        self.capability_queries = 0
        self.change_before_install: dict = {}
        self.server_info = dict(
            model_path=model_path,
            tp_size=workers,
            pp_size=1,
            dp_size=1,
            ep_size=1,
            moe_dp_size=1,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=False,
            cuda_graph_config={"decode": {"backend": "disabled"}, "prefill": {"backend": "disabled"}},
            quantization=None,
            enable_lora=False,
            speculative_algorithm=None,
            custom_weight_loader=[HF_MEMORY_LOADER if updater.canonical_hf else MEMORY_LOADER],
        )
        self.runtime_state = dict(
            self.server_info,
            memory_usage={"graph": dict(_ZERO_GRAPH_USAGE)},
        )
        self.engine = SimpleNamespace(
            rank=index,
            node_rank=0,
            worker_type="regular",
            server_host=f"engine-{index}",
            server_port=8000,
            args=SimpleNamespace(rollout_num_gpus_per_engine=workers),
            num_gpus_per_engine=workers,
            _make_request=self._http,
        )
        self.engine.get_hf_memory_capability = partial(SGLangEngine.get_hf_memory_capability, self.engine)
        self.get_hf_memory_capability = _Remote(self.engine.get_hf_memory_capability)
        self.engine.get_dense_memory_capability = partial(SGLangEngine.get_dense_memory_capability, self.engine)
        self.get_dense_memory_capability = _Remote(self.engine.get_dense_memory_capability)
        self.pause_generation = _Remote(self._pause)
        self.continue_generation = _Remote(self._resume)
        self.memory_weight_generation_probe = _Remote(self._probe)
        self.install_hf_memory_update = _Remote(partial(SGLangEngine.install_hf_memory_update, self.engine))
        self.install_memory_update = _Remote(partial(SGLangEngine.install_memory_update, self.engine))

    def _pause(self, **_kwargs) -> None:
        self.paused = True

    def _resume(self, **_kwargs) -> None:
        # The coordinator may resume only after each worker is installed and
        # the actual sender encoder has crossed its commit gate.
        assert all(_receiver(model).version == self.updater.encoder.version for model in self.models)
        self.paused = False
        self.resumes += 1

    def _probe(self, _timeout) -> dict:
        assert not self.paused
        return {"success": True}

    def _http(self, endpoint: str, payload: dict, **_kwargs) -> dict:
        if endpoint == "update_weights_from_tensor":
            self.weight_requests += 1
            assert self.paused
            expected_loader = HF_MEMORY_LOADER if self.updater.canonical_hf else MEMORY_LOADER
            assert payload["load_format"] == expected_loader
            assert payload["flush_cache"] is True
            serialized = payload["serialized_named_tensors"]
            assert len(serialized) == len(self.models)
            self.encodings.append(tuple(bucket.encoding for bucket in self.updater.source._buckets))
            base = self.updater.encoder.version
            for model, control in zip(self.models, serialized, strict=True):
                # Exercise SGLang's real safe deserializer and the production
                # custom loader; no fixture constructs receiver manifests.
                try:
                    loader = load_hf_memory_weights if self.updater.canonical_hf else load_memory_weights
                    loader(model, MultiprocessingSerializer.deserialize(control))
                except Exception:
                    self.acks_on_failure = set(self.updater.source._acks)
                    raise
                receiver = _receiver(model)
                assert self.updater.encoder.version == base
                assert receiver.version == base + 1
                if self.updater.canonical_hf:
                    assert receiver.plan_fingerprint == self.updater.receiver_plan.fingerprint
                self.install_observations.append((base, receiver.version, receiver.stream))
        elif endpoint == "update_weight_version":
            self.engine.weight_version = payload["new_version"]
        else:
            assert endpoint in {"begin_weight_update", "end_weight_update"}
        return {"success": True}

    def _server_info(self) -> dict:
        self.capability_queries += 1
        if self.capability_queries == 2:
            self.runtime_state.update(self.change_before_install)
        return self.server_info | {"internal_states": [self.runtime_state]}


class _Lock:
    def __init__(self) -> None:
        self.held = False
        self.acquire = _Remote(self._acquire)
        self.release = _Remote(self._release)

    def _acquire(self) -> bool:
        assert not self.held
        self.held = True
        return True

    def _release(self) -> None:
        assert self.held
        self.held = False


def _updater_class():
    # Import the exact production file without the unrelated GPU-only
    # Megatron package initializer, as in test_memory_updater_failure.py.
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_update/update_weight_from_memory.py"
    spec = spec_from_file_location("hf_memory_integration_updater", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UpdateWeightFromMemory


@pytest.fixture
def cpu_boundary(tmp_path, monkeypatch):
    assert not dist.is_initialized()
    assert distributed_utils.GLOO_GROUP is None
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1, timeout=timedelta(seconds=10)
    )
    group = distributed_utils.init_gloo_group(distributed_timeout_minutes=1)
    monkeypatch.setattr(ray, "get", _get)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "cpu-test-node"))
    updaters = []
    http_engines = {}
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    def get_server_info(url, **_kwargs):
        engine = http_engines[url]
        return SimpleNamespace(raise_for_status=lambda: None, json=engine._server_info)

    monkeypatch.setattr(requests, "get", get_server_info)

    def build(worker_counts: tuple[int, ...], *, layout: str = "canonical_hf"):
        values = dict(
            low=torch.ones(64, dtype=torch.bfloat16),
            precise=torch.zeros(64, dtype=torch.float32 if layout == "canonical_hf" else torch.bfloat16),
        )
        updater = object.__new__(_updater_class())
        updater.config = MemorySyncConfig(
            stream="cpu-integration",
            layout=layout,
            bucket_bytes=128,
            gpu_budget_bytes=1 << 20,
            host_budget_bytes=1 << 20,
            receiver_gpu_budget_bytes=1 << 20,
            sync_timeout_s=10,
        )
        updater.weights_getter = lambda: values
        # The fixture supplies already-converted HF tensors. Conversion is
        # outside this CPU boundary; capture/encode/install remain production.
        updater.iterator = SimpleNamespace(get_hf_weight_chunks=lambda weights: [list(weights.items())])
        updater.canonical_hf = layout == "canonical_hf"
        updater.encoder = (
            StreamingDeltaEncoder(SnapshotBudget(updater.config.gpu_budget_bytes), bucket_bytes=128)
            if updater.canonical_hf
            else MemoryDeltaEncoder(snapshot_budget_bytes=updater.config.gpu_budget_bytes, bucket_bytes=128)
        )
        updater.receiver_plan = updater.source = updater.members = None
        updater.engines = ()
        updater.failed = False
        engines = [
            _EngineBoundary(index, count, updater, str(model_path)) for index, count in enumerate(worker_counts)
        ]
        for engine in engines:
            http_engines[f"http://{engine.engine.server_host}:{engine.engine.server_port}/server_info"] = engine
        lock = _Lock()
        manager = SimpleNamespace(worker_counts=list(worker_counts))
        manager.get_rollout_engines_and_lock = _Remote(
            lambda: (
                engines,
                lock,
                0,
                manager.worker_counts,
                [sum(manager.worker_counts[:index]) for index in range(len(engines))],
            )
        )
        updaters.append(updater)
        return updater, values, engines, manager, lock

    try:
        yield build
    finally:
        for updater in updaters:
            if updater.source is not None:
                updater.source.close()
            if updater.canonical_hf:
                updater.encoder.close()
            else:
                updater.encoder.abort()
        dist.destroy_process_group(group)
        distributed_utils.GLOO_GROUP = None
        dist.destroy_process_group()


def _assert_installed(engines, values, version: int) -> None:
    for engine in engines:
        for model in engine.models:
            assert _receiver(model).version == version
            for name, tensor in model.named_parameters():
                assert torch.equal(tensor.view(torch.uint8), values[name].view(torch.uint8))


def test_real_engine_tp3_all_workers_commit_delta_then_reject_topology_change(cpu_boundary) -> None:
    updater, values, engines, manager, lock = cpu_boundary((3,))
    updater.update_weights(manager)
    assert updater.encoder.version == 0 and updater.source_ranks == (0,)
    assert updater.members == {f"engine-0:tp:{tp}" for tp in range(3)}
    _assert_installed(engines, values, 0)
    assert all(encoding == "full" for encoding in engines[0].encodings[0])

    values["low"].view(torch.int16)[0] = -32768
    values["low"].view(torch.int16)[1] = 0x7F81
    values["precise"].view(torch.int32)[0] = -(2**31)
    values["precise"].view(torch.int32)[1] = 0x7F800001
    updater.update_weights(manager)
    assert updater.encoder.version == 1 and not updater.failed and not lock.held
    _assert_installed(engines, values, 1)
    assert engines[0].encodings[1] == ("sparse", "sparse")
    assert engines[0].install_observations == [(-1, 0, "cpu-integration")] * 3 + [(0, 1, "cpu-integration")] * 3
    assert not engines[0].paused and engines[0].resumes == 2

    manager.worker_counts[0] = 4
    with pytest.raises(RuntimeError, match="worker topology changed"):
        updater.update_weights(manager)
    assert updater.failed and updater.encoder.version == 1
    assert engines[0].paused and engines[0].resumes == 2 and not lock.held
    _assert_installed(engines, values, 1)


def test_worker_loader_failure_quarantines_both_engines_without_sender_commit(cpu_boundary) -> None:
    updater, values, engines, manager, lock = cpu_boundary((1, 1))
    updater.update_weights(manager)
    _assert_installed(engines, values, 0)
    old_precise = engines[1].models[0].precise.detach().clone()
    values["low"][0] = 7
    values["precise"][0] = 11
    engines[1].models[0].omit_precise = True
    with pytest.raises(RuntimeError, match="no protected parameter writes"):
        updater.update_weights(manager)
    assert updater.failed and updater.encoder.version == 0 and updater.source is None
    assert all(engine.paused and engine.resumes == 1 for engine in engines)
    assert not lock.held
    assert engines[1].acks_on_failure == {"engine-0:tp:0"}
    # One worker really installed the next version, but the group remains
    # unavailable and the sender's baseline has not advanced.
    _assert_installed(engines[:1], values, 1)
    failed_model = engines[1].models[0]
    assert failed_model._relax_hf_memory_receiver.poisoned
    assert failed_model._relax_hf_memory_receiver.version == 0
    assert torch.equal(failed_model.precise.view(torch.uint8), old_precise.view(torch.uint8))
    with pytest.raises(RuntimeError, match="Failed memory stream must be discarded"):
        updater.update_weights(manager)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tp_size": 2}, "runtime TP worker count"),
        ({"ep_size": 2}, "runtime PP/DP/EP/MoE-DP=1"),
        ({"speculative_algorithm": "EAGLE"}, "unsupported quantization, adapter, draft or routing"),
        ({"disable_cuda_graph": False}, "runtime CUDA graphs disabled"),
        ({"memory_usage": {"graph": _ZERO_GRAPH_USAGE | {"prefill": 0.01}}}, "runtime CUDA graphs disabled"),
        ({"memory_usage": {"graph": _ZERO_GRAPH_USAGE | {"draft_extend": 0.01}}}, "runtime CUDA graphs disabled"),
        ({"memory_usage": {"graph": 0.01}}, "runtime CUDA graphs disabled"),
        (
            {"memory_usage": {"graph": _ZERO_GRAPH_USAGE | {"decode": {"allocated": 0}}}},
            "graph allocation report is invalid",
        ),
        (
            {"memory_usage": {"graph": _ZERO_GRAPH_USAGE | {"decode": float("nan")}}},
            "graph allocation report is invalid",
        ),
        ({"memory_usage": {"graph": -1}}, "graph allocation report is invalid"),
        ({"memory_usage": {"graph": {"decode": 0.0, "prefill": 0.0}}}, "graph allocation report is invalid"),
        (
            {"cuda_graph_config": {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}}},
            "resolved CUDA graph backends disabled",
        ),
    ],
)
def test_runtime_override_rejected_before_publication_or_weight_write(cpu_boundary, override, message) -> None:
    updater, _, engines, manager, lock = cpu_boundary((3,))
    # The HTTP process still reports the supported base config. Scheduler-side
    # resolved overrides must take precedence in the real capability validator.
    engines[0].runtime_state.update(override)
    with pytest.raises(RuntimeError, match=message):
        updater.update_weights(manager)
    assert updater.failed and updater.encoder.version == -1 and updater.source is None
    assert engines[0].weight_requests == 0 and engines[0].resumes == 0 and engines[0].paused
    assert not lock.held
    assert all(not hasattr(model, "_relax_hf_memory_receiver") for model in engines[0].models)


def test_runtime_contract_drift_rejected_again_before_installation(cpu_boundary) -> None:
    updater, _, engines, manager, lock = cpu_boundary((3,))
    # Both capabilities are individually admissible. The fixed session must
    # still reject a changed runtime contract between planning and installation.
    engines[0].change_before_install = {"dtype": "bfloat16"}
    with pytest.raises(RuntimeError, match="runtime capability changed before installation"):
        updater.update_weights(manager)
    assert engines[0].capability_queries == 2 and engines[0].weight_requests == 0
    assert updater.failed and updater.encoder.version == -1 and updater.source is None
    assert engines[0].paused and engines[0].resumes == 0 and not lock.held


def test_legacy_dense_full_delta_and_install_capability_recheck(cpu_boundary) -> None:
    updater, values, engines, manager, lock = cpu_boundary((1,), layout="dense")
    updater.update_weights(manager)
    _assert_installed(engines, values, 0)
    values["low"][0] = 3
    updater.update_weights(manager)
    _assert_installed(engines, values, 1)
    assert engines[0].encodings[-1] == ("sparse", "sparse")
    # Drift only at the next installation check, after planning succeeded.
    engines[0].capability_queries = 0
    engines[0].change_before_install = {"dtype": "bfloat16"}
    with pytest.raises(RuntimeError, match="Dense memory runtime capability changed before installation"):
        updater.update_weights(manager)
    assert updater.failed and updater.encoder.version == 1 and updater.source is None
    assert engines[0].weight_requests == 2 and engines[0].resumes == 2 and engines[0].paused
    assert not lock.held


@pytest.mark.parametrize(
    "allocated,override", [(2, {}), (1, {"tp_size": 2}), (1, {"cuda_graph_backend_decode": "full"})]
)
def test_legacy_dense_runtime_mismatch_rejected_before_publication(cpu_boundary, allocated, override) -> None:
    updater, _, engines, manager, lock = cpu_boundary((allocated,), layout="dense")
    engines[0].runtime_state.update(override)
    with pytest.raises(RuntimeError):
        updater.update_weights(manager)
    assert updater.failed and updater.encoder.version == -1 and updater.source is None
    assert engines[0].weight_requests == 0 and engines[0].resumes == 0 and engines[0].paused
    assert not lock.held
