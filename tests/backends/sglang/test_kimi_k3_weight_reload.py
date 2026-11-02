# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression checks for K3 hooks in the installed SGLang source."""

import ast
from types import SimpleNamespace

import pytest

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def kimi_source():
    # CI rebuilds the image when docker/patch changes; exercise that artifact.
    return ast.parse(sglang_source_path("srt/models/kimi_k3.py").read_text())


def _function(tree: ast.AST, name: str, namespace: dict):
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<sglang-reload-hook>", "exec"), namespace)
    return namespace[name]


@pytest.fixture
def hooks(kimi_source):
    ns = {
        "torch": torch,
        "nn": torch.nn,
        "ReplicatedLinear": object,
        "RMSNorm": object,
        "_is_npu": False,
        "_is_hip": False,
    }
    _function(ast.parse(sglang_source_path("srt/layers/attn_residual.py").read_text()), "get_cw", ns)
    if any(isinstance(node, ast.FunctionDef) and node.name == "_get_k3_dense_weight" for node in kimi_source.body):
        _function(kimi_source, "_get_k3_dense_weight", ns)
    for name in ("PPMissingLayer", "KimiK3MoE", "KimiK3DeltaAttention"):
        ns[name] = type(name, (), {})
    return kimi_source, ns


@torch.no_grad()
@pytest.mark.parametrize("padding", [1, 8])
def test_k3_merged_reload_keeps_captured_storage(hooks, padding):
    tree, ns = hooks
    merge = _function(tree, "_merge_weights_as_views", ns)
    mods = [torch.nn.Linear(4, n, bias=False) for n in (3, 2)]
    merged, _ = merge(mods, pad_rows_to=padding)
    captured = merged
    pointers = [mod.weight.data_ptr() for mod in mods]
    for value in (2.0, 5.0):
        for mod in mods:
            mod.weight.fill_(value)
        merged, _ = merge(mods, pad_rows_to=padding, merged=merged)
        assert merged is captured
        assert [mod.weight.data_ptr() for mod in mods] == pointers
        torch.testing.assert_close(captured[:5], torch.full_like(captured[:5], value))
        assert torch.count_nonzero(captured[5:]) == 0


@torch.no_grad()
def test_k3_post_load_refreshes_mla_and_attn_residual_in_place(hooks):
    tree, ns = hooks
    post_load = _function(tree, "post_load_weights", ns)
    attn = SimpleNamespace(kv_b_proj=torch.nn.Linear(4, 12, bias=False), qk_nope_head_dim=2, v_head_dim=1)
    proj = torch.nn.Linear(4, 1, bias=False)
    norm = SimpleNamespace(weight=torch.ones(4))
    layer = SimpleNamespace(
        self_attn=attn,
        mlp=object(),
        use_attn_residuals=True,
        self_attention_res_proj=proj,
        self_attention_res_norm=norm,
        mlp_res_proj=proj,
        mlp_res_norm=norm,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(full_attention_layer_ids=[0]), model=SimpleNamespace(layers=[layer])
    )
    post_load(model)
    captured = [attn.w_kc, attn.w_vc, *proj._attn_res_cw_cache.values()]
    pointers = [tensor.data_ptr() for tensor in captured]
    strides = [tensor.stride() for tensor in captured]
    for value in (2.0, 5.0):
        attn.kv_b_proj.weight.fill_(value)
        proj.weight.fill_(value)
        norm.weight.fill_(3.0)
        post_load(model)
        updated = [attn.w_kc, attn.w_vc, *proj._attn_res_cw_cache.values()]
        assert [tensor.data_ptr() for tensor in updated] == pointers
        assert [tensor.stride() for tensor in updated] == strides
        for tensor in captured[:2]:
            torch.testing.assert_close(tensor, torch.full_like(tensor, value))
        for tensor in captured[2:]:
            torch.testing.assert_close(tensor, torch.full_like(tensor, value * 3))


@torch.no_grad()
def test_k3_fused_decode_reload_refreshes_captured_inputs(hooks):
    tree, ns = hooks
    ns.update(_is_hip=False, rank0_log=lambda message: None)
    prepare = _function(tree, "_prepare_fused_decode", ns)
    attn = SimpleNamespace(
        conv_weights=torch.ones(3 * 1536, 4),
        bias=torch.ones(3 * 1536),
        A_log=torch.ones(1, 1, 12, 1),
        dt_bias=torch.ones(1536),
    )
    model = SimpleNamespace(attn=attn, o_norm=SimpleNamespace(weight=torch.ones(128), eps=1e-5))
    prepare(model)
    captured = attn._k3_fused_decode_args
    pointers = [tensor.data_ptr() for tensor in captured[:-1]]
    for value in (2.0, 5.0):
        for tensor in (attn.conv_weights, attn.bias, attn.A_log, model.o_norm.weight):
            tensor.fill_(value)
        prepare(model)
        assert attn._k3_fused_decode_args is captured
        assert [tensor.data_ptr() for tensor in captured[:-1]] == pointers
        for tensor in captured[:-1]:
            torch.testing.assert_close(tensor, torch.full_like(tensor, value))


@torch.no_grad()
@pytest.mark.parametrize("hip", [False])
def test_k3_gate_weight_keeps_checkpoint_name_after_fusion(hooks, hip):
    import re
    from collections.abc import Iterable
    from types import MethodType

    tree, ns = hooks
    ns.update(
        _is_hip=hip,
        Iterable=Iterable,
        get_layer_id=lambda _: None,
        _EXPERT_WEIGHT_NAME=re.compile(r"experts\.\d+\.\w+\."),
        maybe_remap_kv_scale_name=lambda name, _: name,
        _maybe_map_fp8_pb_scale_name=lambda name, _: name,
        default_weight_loader=lambda param, value, **_: param.data.copy_(value),
    )
    merge = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_merge_bfa_weights"
    )
    if not hip and not any(isinstance(node, ast.Attribute) and node.attr == "_bfa_f_b_w" for node in ast.walk(merge)):
        pytest.skip("This SGLang version does not fuse the CUDA K3 gate projection")
    _function(tree, "_merge_weights_as_views", ns)
    attn = torch.nn.Module()
    attn.f_a_proj = torch.nn.Linear(4, 3, bias=False)
    attn.b_proj = torch.nn.Linear(4, 1, bias=False)
    attn.f_b_proj = torch.nn.Linear(3, 4, bias=False)
    attn.use_full_rank_gate, attn._bfa_uses_block_fp8 = True, False
    attn._merge_kda_inproj_weights_hip = lambda: True
    attn._merge_bfa_weights = MethodType(_function(tree, "_merge_bfa_weights", ns), attn)
    model = torch.nn.Module()
    model.self_attn = attn
    model.config = SimpleNamespace(linear_attn_config={"use_full_rank_gate": True}, is_moe=False, num_hidden_layers=1)
    model.post_load_weights = attn._merge_bfa_weights
    model.load_weights = MethodType(_function(tree, "load_weights", ns), model)
    attn._merge_bfa_weights()
    captured = attn._bfa_f_b_w
    pointer = captured.data_ptr()
    for value in (2.0, 5.0):
        loaded = model.load_weights([("self_attn.f_b_proj.weight", torch.full_like(attn.f_b_proj.weight, value))])
        assert "self_attn.f_b_proj.weight" in loaded
        assert captured.data_ptr() == pointer == attn._bfa_f_b_w.data_ptr()
        torch.testing.assert_close(captured, torch.full_like(captured, value))
    assert "self_attn.f_b_proj.weight" in dict(model.named_parameters())
    assert "self_attn._bfa_f_b_w" not in dict(model.named_parameters())


def test_k3_weight_update_defers_post_load_until_successful_end(hooks):
    from collections.abc import Iterable
    from types import MethodType

    tree, ns = hooks
    ns["Iterable"] = Iterable
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(linear_attn_config=None, is_moe=False, num_hidden_layers=1),
        named_parameters=lambda: [],
        post_load_weights=lambda: calls.append("refresh"),
    )
    for name in ("begin_weight_update", "end_weight_update", "load_weights"):
        setattr(model, name, MethodType(_function(tree, name, ns), model))
    model.load_weights([])
    assert calls == ["refresh"]
    for _ in range(2):
        model.begin_weight_update()
        for _ in range(3):
            model.load_weights([])
        before = len(calls)
        model.end_weight_update()
        assert len(calls) == before + 1
        assert not model._relax_defer_post_load
    model.begin_weight_update()

    def fail():
        raise RuntimeError("post-load failed")

    model.post_load_weights = fail
    with pytest.raises(RuntimeError, match="post-load failed"):
        model.end_weight_update()
    assert model._relax_defer_post_load


def test_k3_post_load_warms_kda_once_per_kernel_signature(hooks, monkeypatch):
    import sys

    tree, ns = hooks
    warmed, refreshed = [], []
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.fla.kda",
        SimpleNamespace(precompile_k3_recompute_w_u_kernel=lambda **kwargs: warmed.append(kwargs) or True),
    )
    ns["rank0_log"] = lambda message: None
    attn = ns["KimiK3DeltaAttention"]()
    attn.local_num_heads = 12
    attn.o_proj = SimpleNamespace(weight=torch.ones(1, dtype=torch.bfloat16))
    attn.dt_bias = torch.ones(1)
    attn._merge_bfa_weights = lambda: refreshed.append("merge")
    attn._prepare_fused_decode = lambda: refreshed.append("decode")
    model = SimpleNamespace(
        config=SimpleNamespace(full_attention_layer_ids=[]),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn, mlp=object(), use_attn_residuals=False)]),
    )
    post_load = _function(tree, "post_load_weights", ns)
    for _ in range(3):
        post_load(model)
    assert len(warmed) == 1
    assert len(refreshed) == 6
    attn.o_proj.weight = torch.ones(1, dtype=torch.float32)
    post_load(model)
    assert len(warmed) == 2


def test_k3_runner_finishes_deferred_refresh_after_repack(monkeypatch):
    import sys
    from contextlib import nullcontext

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_loader.loader",
        SimpleNamespace(device_loading_context=lambda module, device: nullcontext()),
    )
    calls = []
    quant = SimpleNamespace(
        restore_weights_before_loading=lambda module: calls.append("restore"),
        process_weights_after_loading=lambda module: calls.append("repack"),
    )
    model = SimpleNamespace(
        begin_weight_update=lambda: calls.append("begin"),
        end_weight_update=lambda: calls.append("end"),
        named_modules=lambda: [("expert", SimpleNamespace(quant_method=quant))],
    )
    runner = SimpleNamespace(device="cpu", model=model)
    post_process = _function(
        ast.parse(sglang_source_path("srt/model_executor/model_runner.py").read_text()),
        "post_process_weights",
        {"torch": torch},
    )
    assert post_process(runner, restore_weights_before_load=True)[0]
    assert calls == ["begin", "restore"]
    assert post_process(runner, post_process_quantization=True)[0]
    assert calls == ["begin", "restore", "repack", "end"]

    def fail(module):
        raise RuntimeError("repack failed")

    quant.process_weights_after_loading = fail
    with pytest.raises(RuntimeError, match="repack failed"):
        post_process(runner, post_process_quantization=True)
    assert calls.count("end") == 1
