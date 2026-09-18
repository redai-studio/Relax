# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SGLang native-diffusion engine tests.

The live diffusion server requires the pinned SGLang image and GPUs, so these
tests cover the pure Python data-plane contracts plus the static SGLang patch
contract Relax checks before launching the server.
"""

from __future__ import annotations

import base64
import dataclasses
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang import diffusion_engine as de
from relax.backends.sglang.diffusion_engine import (
    SGLangNativeGenerationEngine,
    _deserialize,
    _local_maybe_deserialize,
    _verify_diffusion_patch_contract,
)


def _enc(t: torch.Tensor) -> dict:
    """Encode a tensor in SGLang's base64 envelope."""
    return {
        "__tensor__": True,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "data": base64.b64encode(t.contiguous().numpy().tobytes()).decode("ascii"),
    }


def _build_static_patch_modules(
    *,
    omit_sampling_field: str | None = None,
    include_dance: bool = True,
    preserve_driver_sigmas: bool = True,
) -> dict[str, dict[str, object]]:
    field_kwargs = {
        "seed": (int | None, None),
        "initial_noise_group_ids": (list[str] | None, None),
        "initial_noise_latent_shape": (list[int] | None, None),
        "initial_noise_seed": (int | None, None),
        "denoise_seeds": (list[str] | None, None),
        "sigmas": (list[float] | None, None),
        "timesteps": (list[float] | None, None),
    }
    if omit_sampling_field is not None:
        field_kwargs.pop(omit_sampling_field)

    SamplingParams = dataclasses.make_dataclass(
        "SamplingParams",
        [(name, type_, dataclasses.field(default=default)) for name, (type_, default) in field_kwargs.items()],
    )

    class SchedulerRLMixin:
        def flow_sde_sampling(self):
            if include_dance:
                return "dance"
            return "sde"

    class FlowMatchEulerDiscreteScheduler:
        if preserve_driver_sigmas:

            def set_timesteps(self):
                """Driver-supplied sigmas are already in replay sigma space."""
                return None

        else:

            def set_timesteps(self):
                return None

    def prepare_request(server_args, sampling_params):
        del server_args
        value = getattr(sampling_params, "initial_noise_group_ids", None)
        return torch.as_tensor(value or [], dtype=torch.float32)

    return {
        "sglang.multimodal_gen.configs.pipeline_configs.qwen_image": {
            "_QWEN_MAX_SEQUENCE_LENGTH": 512,
        },
        "sglang.multimodal_gen.configs.post_training.rl_rollout": {
            "_VALID_ROLLOUT_SDE_TYPES": ("ode", "sde", "cps", "dance"),
        },
        "sglang.multimodal_gen.configs.sample.sampling_params": {
            "SamplingParams": SamplingParams,
        },
        "sglang.multimodal_gen.runtime.entrypoints.utils": {
            "prepare_request": prepare_request,
        },
        "sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete": {
            "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
        },
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising": {
            "_make_step_generators": lambda *args, **kwargs: [],
        },
        "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation": {
            "_driver_xt_recipe": lambda batch: None,
        },
        "sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin": {
            "SchedulerRLMixin": SchedulerRLMixin,
        },
    }


@contextmanager
def fake_sglang_static_patch(**kwargs):
    """Swap the sglang namespace for a minimal static-patch module tree."""
    saved = {name: mod for name, mod in sys.modules.items() if name == "sglang" or name.startswith("sglang.")}
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]

    modules: dict[str, types.ModuleType] = {}

    def _ensure(dotted: str) -> types.ModuleType:
        if dotted in modules:
            return modules[dotted]
        module = types.ModuleType(dotted)
        module.__path__ = []
        modules[dotted] = module
        sys.modules[dotted] = module
        if "." in dotted:
            parent, _, leaf = dotted.rpartition(".")
            setattr(_ensure(parent), leaf, module)
        return module

    try:
        for dotted, attrs in _build_static_patch_modules(**kwargs).items():
            module = _ensure(dotted)
            for key, value in attrs.items():
                setattr(module, key, value)
        yield modules
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Static SGLang patch contract
# ---------------------------------------------------------------------------


def test_verify_diffusion_patch_contract_accepts_static_patch():
    with fake_sglang_static_patch():
        _verify_diffusion_patch_contract()


def test_verify_diffusion_patch_contract_rejects_missing_sampling_field():
    with fake_sglang_static_patch(omit_sampling_field="denoise_seeds"):
        with pytest.raises(RuntimeError, match="SamplingParams missing fields"):
            _verify_diffusion_patch_contract()


def test_verify_diffusion_patch_contract_rejects_stock_sigma_shifting():
    with fake_sglang_static_patch(preserve_driver_sigmas=False):
        with pytest.raises(RuntimeError, match="driver sigmas"):
            _verify_diffusion_patch_contract()


# ---------------------------------------------------------------------------
# Tensor wire format
# ---------------------------------------------------------------------------


def test_local_deserialize_roundtrip_nested():
    t = torch.randn(2, 3, 4)
    obj = {"a": _enc(t), "b": [_enc(t), {"c": _enc(t)}], "d": 5, "e": "x"}
    out = _local_maybe_deserialize(obj)
    assert torch.allclose(out["a"], t)
    assert torch.allclose(out["b"][0], t)
    assert torch.allclose(out["b"][1]["c"], t)
    assert out["d"] == 5 and out["e"] == "x"


@pytest.fixture
def reset_deserializer_cache():
    """``_deserialize`` resolves SGLang's decoder once per interpreter."""
    saved = (de._SGLANG_DESERIALIZER, de._SGLANG_DESERIALIZER_RESOLVED)
    de._SGLANG_DESERIALIZER, de._SGLANG_DESERIALIZER_RESOLVED = None, False
    try:
        yield
    finally:
        de._SGLANG_DESERIALIZER, de._SGLANG_DESERIALIZER_RESOLVED = saved


def test_deserialize_does_not_mask_a_wire_format_divergence(reset_deserializer_cache):
    """Decode failures must surface instead of silently using the fallback."""
    module = types.ModuleType("sglang.multimodal_gen.runtime.entrypoints.post_training.utils")

    def _maybe_deserialize(obj):
        raise ValueError("unknown dtype in envelope")

    module._maybe_deserialize = _maybe_deserialize
    saved = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        with pytest.raises(ValueError, match="unknown dtype"):
            _deserialize({"__tensor__": True})
    finally:
        if saved is None:
            del sys.modules[module.__name__]
        else:
            sys.modules[module.__name__] = saved


def test_deserialize_falls_back_when_sglang_is_absent(reset_deserializer_cache):
    t = torch.randn(3)
    assert torch.allclose(_deserialize(_enc(t)), t)


# ---------------------------------------------------------------------------
# Engine: response mapping
# ---------------------------------------------------------------------------


def test_map_response_extracts_trajectory_and_echoes_sde_indices():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    latents = torch.randn(4, 16, 8, 8)
    timesteps = torch.linspace(1000, 0, 3)
    logp = torch.randn(3)
    resp = {
        "request_id": "r1",
        "seed": 42,
        "dit_trajectory": {"latents": _enc(latents), "timesteps": _enc(timesteps)},
        "rollout_log_probs": {"log_probs": _enc(logp)},
        "denoising_env": {"pos_cond_kwargs": {"prompt_embeds": _enc(torch.randn(5, 32))}},
    }
    request = {"rollout_sde_step_indices": [0, 2], "_task": "t2i", "height": 384, "width": 512}
    out = eng._map_response(resp, request)
    assert torch.allclose(out["trajectory_latents"], latents)
    assert torch.allclose(out["timesteps"], timesteps)
    assert out["sde_indices"] == [0, 2]
    assert out["task"] == "t2i"
    assert torch.allclose(out["rollout_log_probs"], logp)
    assert "prompt_embeds" in out["denoising_env"]["pos_cond_kwargs"]


def test_map_response_rejects_invalid_log_prob_wrapper():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    with pytest.raises(ValueError, match="missing required 'log_probs'"):
        eng._map_response({"rollout_log_probs": {"unexpected": [1.0]}}, {})


def test_map_response_echoes_the_requested_geometry():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    out = eng._map_response({"request_id": "r1"}, {"_task": "t2i", "height": 384, "width": 512})
    assert out["height"] == 384 and out["width"] == 512


def test_map_responses_requires_exactly_one_candidate():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    request = {"_task": "t2i"}
    assert eng._map_responses([{"request_id": "x"}], request)["request_id"] == "x"
    with pytest.raises(RuntimeError, match="empty"):
        eng._map_responses([], request)
    with pytest.raises(RuntimeError, match="returned 2 responses"):
        eng._map_responses([{"request_id": "x"}, {"request_id": "y"}], request)


def test_post_decodes_sglang_msgpack_response(monkeypatch):
    import msgspec
    import requests

    payload = [{"request_id": "r1", "generated_output": {"__tensor__": True, "data": b"tensor"}}]

    class Response:
        status_code = 200
        headers = {"content-type": "application/msgpack"}
        content = msgspec.msgpack.encode(payload)
        text = ""

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: Response())
    engine = SGLangNativeGenerationEngine(args=None, rank=0)
    engine._url = "http://127.0.0.1:1234"
    assert engine._post("/rollout/generate", {}) == payload


def test_engine_method_surface():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    for method_name in (
        "generate_batch",
        "update_weights_from_tensor",
        "set_lora_from_tensors",
        "get_weights_checksum",
        "commit_weight_version",
        "get_base_gpu_id",
        "health_generate",
    ):
        assert callable(getattr(eng, method_name)), method_name
    for gone in (
        "init_weights_update_group",
        "update_weights_from_distributed",
        "update_weights_from_disk",
        "generate_grouped",
        "unload_lora_adapter",
    ):
        assert not hasattr(eng, gone), gone


def test_commit_weight_version_tracks_active():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    assert eng.get_weight_version() == 0
    digest = "a" * 64
    eng.commit_weight_version(7, digest)
    assert eng.get_weight_version() == 7
    assert eng._map_response({}, {})["weight_manifest_sha256"] == digest


def test_commit_weight_version_rejects_invalid_manifest_digest():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    with pytest.raises(ValueError, match="weight_manifest_sha256"):
        eng.commit_weight_version(7, "")


# ---------------------------------------------------------------------------
# Engine: health + offload contracts
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.text = "boom"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@contextmanager
def _fake_requests(get_result):
    module = types.ModuleType("requests")

    def get(url, timeout=None):
        del url, timeout
        if isinstance(get_result, Exception):
            raise get_result
        return get_result

    module.get = get
    saved = sys.modules.get("requests")
    sys.modules["requests"] = module
    try:
        yield module
    finally:
        if saved is None:
            del sys.modules["requests"]
        else:
            sys.modules["requests"] = saved


def test_health_generate_returns_true_on_200():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    eng._url = "http://127.0.0.1:21000"
    with _fake_requests(_FakeResponse(200)):
        assert eng.health_generate() is True


def test_health_generate_raises_on_error_status():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    eng._url = "http://127.0.0.1:21000"
    with _fake_requests(_FakeResponse(500)):
        with pytest.raises(RuntimeError):
            eng.health_generate()


def test_health_generate_propagates_a_transport_failure():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    eng._url = "http://127.0.0.1:21000"
    with _fake_requests(TimeoutError("hung server")):
        with pytest.raises(TimeoutError):
            eng.health_generate()


def test_offload_and_onload_are_fatal_on_failure():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    eng._post = lambda path, body: {"success": False, "message": "no such endpoint"}
    with pytest.raises(RuntimeError, match="release_memory_occupation failed"):
        eng.release_memory_occupation()
    with pytest.raises(RuntimeError, match="resume_memory_occupation failed"):
        eng.resume_memory_occupation()

    def _boom(path, body):
        raise RuntimeError("404 Client Error")

    eng._post = _boom
    with pytest.raises(RuntimeError, match="404"):
        eng.release_memory_occupation()


def test_offload_passes_through_a_successful_reply():
    eng = SGLangNativeGenerationEngine(args=None, rank=0)
    eng._post = lambda path, body: {"success": True}
    assert eng.release_memory_occupation() == {"success": True}
    assert eng.resume_memory_occupation() == {"success": True}


# ---------------------------------------------------------------------------
# Engine: server args
# ---------------------------------------------------------------------------


def test_cuda_visible_devices_reorders_block_first_keeps_all_visible():
    assert SGLangNativeGenerationEngine._cuda_visible_devices(None, 1, 8) is None
    assert SGLangNativeGenerationEngine._cuda_visible_devices(3, 1, 8) == "3,0,1,2,4,5,6,7"
    assert SGLangNativeGenerationEngine._cuda_visible_devices(0, 1, 8) == "0,1,2,3,4,5,6,7"
    assert SGLangNativeGenerationEngine._cuda_visible_devices(4, 2, 8) == "4,5,0,1,2,3,6,7"


def test_cuda_visible_devices_rejects_a_block_off_the_end_of_the_node():
    with pytest.raises(RuntimeError, match="num-gpus-per-node"):
        SGLangNativeGenerationEngine._cuda_visible_devices(2, 4, 4)


def test_launch_server_requires_a_node_width(monkeypatch):
    args = SimpleNamespace(model_path="/models/Qwen-Image", model_revision=None, num_gpus_per_node=0)
    eng = SGLangNativeGenerationEngine(args=args, rank=0, base_gpu_id=0, num_gpus_per_engine=1)
    monkeypatch.setattr(eng, "_get_current_node_ip_and_free_port", lambda start_port: ("127.0.0.1", 21000))
    with pytest.raises(RuntimeError, match="num_gpus_per_node"):
        eng._launch_server("127.0.0.1", 21000, {})


def test_compute_server_args_maps_model_ports_and_overrides():
    args = SimpleNamespace(
        model_path="/models/Qwen-Image/",
        model_revision=None,
        rollout_num_gpus_per_engine=1,
    )
    eng = SGLangNativeGenerationEngine(
        args=args, rank=0, base_gpu_id=2, sglang_overrides={"enable_torch_compile": True}
    )
    sa = eng._compute_server_args(
        "127.0.0.1",
        30010,
        num_gpus=1,
        engine_args={"nccl_port": 40010, "dist_init_addr": "127.0.0.1:50010"},
    )
    assert sa["model_path"] == "/models/Qwen-Image"
    assert sa["host"] == "127.0.0.1" and sa["port"] == 30010
    assert sa["num_gpus"] == 1
    assert sa["master_port"] == 40010
    assert sa["scheduler_port"] == 50010
    assert sa["trust_remote_code"] is True
    assert sa["enable_torch_compile"] is True
    assert "revision" not in sa
    assert "ulysses_degree" not in sa


def test_compute_server_args_rejects_an_invalid_scheduler_rendezvous_address():
    args = SimpleNamespace(model_path="/models/Qwen-Image", model_revision=None, rollout_num_gpus_per_engine=1)
    eng = SGLangNativeGenerationEngine(args=args, rank=0)
    with pytest.raises(ValueError, match="invalid diffusion engine dist_init_addr"):
        eng._compute_server_args("127.0.0.1", 30010, num_gpus=1, engine_args={"dist_init_addr": "invalid"})


def test_compute_server_args_multi_gpu_forces_ulysses_not_cfg_parallel():
    args = SimpleNamespace(model_path="/models/Qwen-Image", model_revision=None, rollout_num_gpus_per_engine=2)
    eng = SGLangNativeGenerationEngine(args=args, rank=0, base_gpu_id=4, num_gpus_per_engine=2)
    sa = eng._compute_server_args("127.0.0.1", 30010, num_gpus=2, engine_args={})
    assert sa["num_gpus"] == 2
    assert sa["ulysses_degree"] == 2


def test_compute_server_args_override_can_replace_ulysses_degree():
    args = SimpleNamespace(model_path="/models/Qwen-Image", model_revision=None, rollout_num_gpus_per_engine=4)
    eng = SGLangNativeGenerationEngine(
        args=args, rank=0, base_gpu_id=0, sglang_overrides={"ulysses_degree": 1, "sp_degree": 4}
    )
    sa = eng._compute_server_args("127.0.0.1", 30010, num_gpus=4, engine_args={})
    assert sa["ulysses_degree"] == 1 and sa["sp_degree"] == 4


def test_compute_server_args_none_override_does_not_clobber_model_path():
    args = SimpleNamespace(model_path="/models/Qwen-Image", hf_checkpoint=None, model_revision=None)
    eng = SGLangNativeGenerationEngine(args=args, rank=0, base_gpu_id=0, sglang_overrides={"model_path": None})
    sa = eng._compute_server_args("127.0.0.1", 30010, num_gpus=1, engine_args={})
    assert sa["model_path"] == "/models/Qwen-Image"
    assert "master_port" not in sa


def test_resolve_model_path_precedence_and_error():
    args = SimpleNamespace(model_path=None, sglang_hf_checkpoint=None, hf_checkpoint="/models/hf")
    assert SGLangNativeGenerationEngine(args=args, rank=0)._resolve_model_path() == "/models/hf"

    missing = SimpleNamespace(model_path=None, sglang_hf_checkpoint=None, hf_checkpoint=None)
    eng = SGLangNativeGenerationEngine(args=missing, rank=0)
    with pytest.raises(RuntimeError, match="no model path"):
        eng._resolve_model_path()
