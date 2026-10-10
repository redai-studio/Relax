# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the shipped reload hooks without importing SGLang's GPU kernels."""

from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
PATCH = Path(__file__).resolve().parents[3] / "docker/patch/sglang/v0.5.17.patch"


def _method(backend: str):
    patch = PATCH.read_text().rsplit("diff --git a/python/sglang/srt/layers/quantization/mxfp4.py", 1)[1]
    source = "\n".join(line[1:] for line in patch.splitlines() if line.startswith(("+", " ")))
    source = source[source.index("    @torch.no_grad()") :]
    source = source.split("    def _process_mxfp4_weights_after_loading")[0]
    namespace = {
        "torch": torch,
        "Parameter": torch.nn.Parameter,
        "_UE8M0_ONE": 127,
        "_flashinfer_mxfp4_permute_indices_cache": {},
        "_flashinfer_mxfp4_permute_indices_device_cache": {},
        "get_moe_a2a_backend": lambda: SimpleNamespace(is_megamoe=lambda: backend == "megamoe"),
    }
    exec("class ReloadHooks:\n" + source, namespace)
    method = namespace["ReloadHooks"]()
    method.use_deep_gemm = backend in ("deep_gemm", "megamoe")
    method.use_triton_kernels = backend == "triton"
    method.use_flashinfer = backend == "sm90"

    def pack(layer):
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            if method.use_deep_gemm:
                # DeepGEMM mutates Parameter.data, changing dtype/shape/stride.
                weight.data = weight.data.view(torch.int8)
                scale.data = scale.sum(dim=-1, keepdim=True, dtype=torch.int32).transpose(1, 2)
                if backend == "megamoe":
                    weight.data = weight.transpose(1, 2).contiguous()
            elif backend == "triton":
                wrap = lambda tensor: SimpleNamespace(storage=SimpleNamespace(data=tensor))
                setattr(method, f"{prefix}_weight_triton_tensor", wrap(weight.float() * 2))
                setattr(method, f"{prefix}_precision_config", SimpleNamespace(weight_scale=wrap(scale.clone())))
                delattr(layer, f"{prefix}_weight")
            else:
                # SM90/Marlin replace Parameters; AITER also mutates bytes.
                weight.add_(1)
                layer.register_parameter(
                    f"{prefix}_weight", torch.nn.Parameter(weight.float() * 2, requires_grad=False)
                )
        layer.register_parameter(
            "swiglu_alpha", torch.nn.Parameter(torch.ones(2, device=scale.device), requires_grad=False)
        )
        layer.workspace = torch.zeros(4, dtype=torch.int32, device=scale.device)

    method._process_mxfp4_weights_after_loading = pack
    return method


def _layer(device: str = "cpu"):
    layer = torch.nn.Module()
    for prefix in ("w13", "w2"):
        for suffix, shape, value, dtype in (
            ("weight", (2, 4, 16), 1, torch.uint8),
            ("weight_scale", (2, 4, 4), 127, torch.uint8),
            ("weight_bias", (2, 4), 0, torch.bfloat16),
        ):
            param = torch.nn.Parameter(torch.full(shape, value, dtype=dtype, device=device), requires_grad=False)
            param.weight_loader = "loader sentinel"
            layer.register_parameter(f"{prefix}_{suffix}", param)
    return layer


def _load(method, layer, value: int) -> None:
    method.restore_weights_before_loading(layer)
    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight")
        scale = getattr(layer, f"{prefix}_weight_scale")
        assert weight.dtype == scale.dtype == torch.uint8
        assert weight.shape == (2, 4, 16)
        assert scale.shape == (2, 4, 4)
        assert weight.weight_loader == scale.weight_loader == "loader sentinel"
        weight.fill_(value)
        scale.fill_(128 + value)
    # A duplicate restore must not erase weights already loaded by a bucket.
    method.restore_weights_before_loading(layer)
    assert torch.all(layer.w13_weight == value)


@pytest.mark.parametrize("backend", ["sm90", "deep_gemm", "megamoe", "triton"])
def test_mxfp4_reload_preserves_runtime_storage_and_updates_values(backend: str) -> None:
    method, layer = _method(backend), _layer()
    method.process_weights_after_loading(layer)
    runtime = dict(layer.named_parameters())
    pointers = {name: param.data_ptr() for name, param in runtime.items()}
    workspace = layer.workspace
    triton = getattr(method, "_mxfp4_runtime_triton", None)
    for prefix in ("w13", "w2"):
        loader_weight = layer._mxfp4_pristine_params[f"{prefix}_weight"]
        target = (
            getattr(method, f"{prefix}_weight_triton_tensor").storage.data
            if backend == "triton"
            else runtime[f"{prefix}_weight"]
        )
        assert loader_weight.data_ptr() == target.data_ptr()
        assert loader_weight is not target

    for value in (3, 5):
        _load(method, layer, value)
        method.process_weights_after_loading(layer)
        # A duplicate postprocess must not quantize/reorder runtime data again.
        method.process_weights_after_loading(layer)
        assert {name: param.data_ptr() for name, param in layer.named_parameters()} == pointers
        assert all(getattr(layer, name) is param for name, param in runtime.items())
        assert layer.workspace is workspace
        assert torch.all(layer.w13_weight_bias == 0)
        if backend == "triton":
            assert method.w13_weight_triton_tensor is triton[0][0]
            assert method.w13_precision_config is triton[0][1]
            assert torch.all(triton[0][0].storage.data == value * 2)
            assert torch.all(triton[0][1].weight_scale.storage.data == 128 + value)
        else:
            expected = value if method.use_deep_gemm else (value + 1) * 2
            assert torch.all(layer.w13_weight == expected)
        if method.use_deep_gemm:
            assert layer.w13_weight_scale.dtype == torch.int32
            assert torch.all(layer.w13_weight_scale == (128 + value) * 4)
        if backend == "megamoe":
            assert layer.mega_l1_weights[0].data_ptr() == layer.w13_weight.data_ptr()
            assert layer.mega_l1_weights[1].data_ptr() == layer.w13_weight_scale.data_ptr()
            assert layer.mega_l2_weights[0].data_ptr() == layer.w2_weight.data_ptr()


def test_mxfp4_reload_resets_discarded_padding_and_bias() -> None:
    method, layer = _method("sm90"), _layer()
    layer.w13_weight_bias.fill_(2)
    method.process_weights_after_loading(layer)
    # Emulate sleep discarding the serialized buffers, including unpushed bias.
    for param in layer._mxfp4_pristine_params.values():
        param.fill_(77)
    method.restore_weights_before_loading(layer)
    assert torch.all(layer.w13_weight == 0)
    assert torch.all(layer.w13_weight_scale == 127)
    assert torch.all(layer.w13_weight_bias == 2)
    assert torch.all(layer.w2_weight_bias == 0)


def test_mxfp4_reload_invalidates_discarded_flashinfer_permutations() -> None:
    method, layer = _method("sm90"), _layer()
    method.process_weights_after_loading(layer)
    namespace = method.restore_weights_before_loading.__wrapped__.__globals__
    caches = [
        namespace["_flashinfer_mxfp4_permute_indices_cache"],
        namespace["_flashinfer_mxfp4_permute_indices_device_cache"],
    ]
    for cache in caches:
        cache["discarded"] = torch.zeros(8, dtype=torch.int64)
    method.restore_weights_before_loading(layer)
    assert all(not cache for cache in caches)
    # A repeated restore inside the same upload must not invalidate fresh indices.
    for cache in caches:
        cache["fresh"] = torch.arange(8)
    method.restore_weights_before_loading(layer)
    assert all("fresh" in cache for cache in caches)


def test_mxfp4_loading_storage_reuse_checks_offset_and_capacity() -> None:
    method = _method("sm90")
    layer = torch.nn.Module()
    loader = torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False)
    layer._mxfp4_pristine_params = {"w13_weight": loader}
    allocation = torch.zeros(8, dtype=torch.int32)
    target = allocation[2:4]
    layer._mxfp4_runtime_params = {"w13_weight": target}
    method._reuse_mxfp4_loading_storage(layer)
    assert loader.data_ptr() == target.data_ptr()
    assert loader.dtype == torch.uint8
    assert loader.shape == (4,)
    # A broadcast view's logical numel does not describe its storage capacity.
    small = torch.zeros(1, dtype=torch.uint8).expand(4)
    layer._mxfp4_runtime_params["w13_weight"] = small
    previous_ptr = loader.data_ptr()
    method._reuse_mxfp4_loading_storage(layer)
    assert small.untyped_storage().nbytes() == 1
    assert loader.data_ptr() == previous_ptr


def test_mxfp4_reload_rejects_changed_runtime_layout() -> None:
    method, layer = _method("sm90"), _layer()
    method.process_weights_after_loading(layer)
    pack = method._process_mxfp4_weights_after_loading

    def changed_pack(layer):
        pack(layer)
        layer.w13_weight.data = layer.w13_weight.data[:, :2].contiguous()

    method._process_mxfp4_weights_after_loading = changed_pack
    _load(method, layer, 3)
    with pytest.raises(RuntimeError, match="runtime layout changed"):
        method.process_weights_after_loading(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a CUDA GPU")
@pytest.mark.parametrize("backend", ["sm90", "deep_gemm", "megamoe", "triton"])
def test_mxfp4_reload_cuda_graph_observes_updated_weights(backend: str) -> None:
    method, layer = _method(backend), _layer("cuda")
    method.process_weights_after_loading(layer)

    def forward():
        if backend == "triton":
            weight = method.w13_weight_triton_tensor.storage.data
            scale = method.w13_precision_config.weight_scale.storage.data
        elif backend == "megamoe":
            weight, scale = layer.mega_l1_weights
        else:
            weight, scale = layer.w13_weight, layer.w13_weight_scale
        return weight.float().sum() + scale.float().sum() + layer.swiglu_alpha.sum() + layer.workspace.sum()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = forward()
    previous = result.clone()
    for value in (3, 5):
        _load(method, layer, value)
        method.process_weights_after_loading(layer)
        graph.replay()
        torch.testing.assert_close(result, forward())
        assert not torch.equal(result, previous)
        previous = result.clone()
