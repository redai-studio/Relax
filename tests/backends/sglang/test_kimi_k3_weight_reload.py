# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression checks for K3 hooks in the repository's SGLang patch."""

import ast
import importlib.util
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
PATCH = Path(__file__).resolve().parents[3] / "docker/patch/sglang/v0.5.17.patch"


def _source_path(relative: str) -> Path:
    root = os.environ.get("SGLANG_TEST_SOURCE_ROOT")
    source_override = root is not None
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None:
            pytest.skip("Requires SGLang source or SGLANG_TEST_SOURCE_ROOT")
        root = str(Path(spec.origin).parent)
    path = Path(root) / relative
    if not path.exists():
        if not source_override:
            pytest.skip("Requires a SGLang version with Kimi K3")
        pytest.fail(f"Missing SGLang source: {path}")
    return path


def _patch_section(relative: str) -> str:
    header = f"diff --git a/python/sglang/{relative} b/python/sglang/{relative}\n"
    return header + PATCH.read_text().split(header, 1)[1].split("\ndiff --git ", 1)[0]


@pytest.fixture(scope="module")
def kimi_source(tmp_path_factory):
    # CI may still use the previous image. Apply the checked-in patch to an
    # isolated copy instead of accidentally exercising its installed old hooks.
    root = tmp_path_factory.mktemp("sglang-k3-reload")
    relative = "srt/models/kimi_k3.py"
    destination = root / "python/sglang" / relative
    destination.parent.mkdir(parents=True)
    destination.write_text(_source_path(relative).read_text())
    patch = _patch_section(relative)
    check = subprocess.run(["git", "apply", "--check", "-"], input=patch, text=True, cwd=root, capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=root, check=True)
    else:
        # An already-patched image is supported, but a mismatched source must
        # fail rather than silently dropping the regression checks.
        subprocess.run(["git", "apply", "--reverse", "--check", "-"], input=patch, text=True, cwd=root, check=True)
    return ast.parse(destination.read_text())


def _added_patch_function(relative: str, name: str, namespace: dict):
    # ModelRunner.post_process_weights is a complete added method in this patch;
    # read it directly so an older installed implementation cannot win.
    lines = _patch_section(relative).splitlines()
    start = lines.index(f"+    def {name}(")
    added = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        added.append(line[1:])
    return _function(ast.parse(textwrap.dedent("\n".join(added))), name, namespace)


def _function(tree: ast.AST, name: str, namespace: dict):
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<sglang-reload-hook>", "exec"), namespace)
    return namespace[name]


@pytest.fixture
def hooks(kimi_source):
    ns = {"torch": torch, "ReplicatedLinear": object, "RMSNorm": object}
    _function(ast.parse(_source_path("srt/layers/attn_residual.py").read_text()), "get_cw", ns)
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
    post_process = _added_patch_function(
        "srt/model_executor/model_runner.py", "post_process_weights", {"torch": torch}
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
