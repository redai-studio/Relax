# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate runtime patches against real pinned sources, without CUDA
imports."""

import ast
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "blackwell_validation", ROOT / "docker/scripts/validate_blackwell_runtime.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


@pytest.fixture
def fla_tree(tmp_path):
    source = os.environ.get("BLACKWELL_FLA_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_FLA_SOURCE_DIR to the unpacked fla-core/flash-linear-attention 0.5.2 wheel root")
    shutil.copytree(Path(source) / "fla", tmp_path / "fla")
    # Ensure the new convolution patches compose with main's existing KDA precision patch.
    subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "-i",
            str(ROOT / "docker/patch/fla/0.5.2-kda-backward-precision.patch"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def kernel_bodies(path):
    return {
        n.name: [ast.dump(stmt) for stmt in n.body]
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


def install_conv_entries(fla_tree):
    for name in ("0.5.2-qwen38-output-config.patch", "0.5.2-conv-static-entries.patch"):
        command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(ROOT / "docker/patch/fla" / name)]
        result = subprocess.run(command, cwd=fla_tree, capture_output=True, text=True, check=True)
        assert "offset" not in result.stdout and "fuzz" not in result.stdout
        assert subprocess.run(command + ["--dry-run"], cwd=fla_tree, capture_output=True).returncode != 0


def test_fla_entries_preserve_kernel_arithmetic(fla_tree):
    package = fla_tree / "fla"
    kernels = package / "modules/conv/triton/kernels.py"
    before = kernel_bodies(kernels)
    install_conv_entries(fla_tree)
    hashes = validation.check_fla_conv(package)
    assert set(hashes) == {"kernels.py", "kernels_pow2.py", "kernels_none.py", "ops.py", "conv_policy.py"}
    assert kernel_bodies(kernels) == before


@pytest.mark.parametrize("filename", ["kernels.py", "kernels_pow2.py", "kernels_none.py"])
def test_fla_validation_rejects_kernel_arithmetic_changes(fla_tree, filename):
    install_conv_entries(fla_tree)
    package = fla_tree / "fla"
    kernels = package / "modules/conv/triton" / filename
    kernels.write_text(
        kernels.read_text().replace("i_d, i_t, i_b = tl.program_id(0)", "i_d, i_t, i_b = NB + tl.program_id(0)", 1)
    )
    with pytest.raises(ValueError, match="arithmetic"):
        validation.check_fla_conv(package)


@pytest.mark.parametrize("mode", ["original", "pow2", "none", "auto"])
@pytest.mark.parametrize("arch", ["blackwell", "hopper", ""])
def test_fla_policy_selects_entries_once_per_process(fla_tree, monkeypatch, mode, arch):
    install_conv_entries(fla_tree)
    monkeypatch.setenv("RELAX_FLA_CONV_NB_MODE", mode)
    monkeypatch.setenv("RELAX_GPU_ARCH", arch)
    monkeypatch.delenv("RELAX_QWEN38_FLA_CONV_CACHE_REUSE", raising=False)
    scope = {}
    source = fla_tree / "fla/modules/conv/triton/conv_policy.py"
    exec(compile(source.read_text(), str(source), "exec"), scope)
    expected = ("none" if arch == "blackwell" else "original") if mode == "auto" else mode
    assert scope["CONV_NB_MODE"] == expected
    for batch, length in [(1, 1), (2, 1025), (1, 4097), (2, 65537)]:
        nb = (batch * length + 1023) // 1024
        wanted = {} if expected == "none" else {"NB": (1 << (nb - 1).bit_length()) if expected == "pow2" else nb}
        assert scope["conv_nb_kwargs"](batch, length) == wanted
    monkeypatch.setenv("RELAX_FLA_CONV_NB_MODE", "typo")
    assert scope["CONV_NB_MODE"] == expected
    assert scope["conv_nb_kwargs"](batch, length) == wanted
    with pytest.raises(ValueError, match="Unknown FLA"):
        exec(compile(source.read_text(), str(source), "exec"), {})


@pytest.mark.parametrize("mode", ["original", "pow2", "none"])
def test_fla_forward_backward_bind_same_static_entry(fla_tree, monkeypatch, mode):
    from types import SimpleNamespace

    install_conv_entries(fla_tree)
    tree = ast.parse((fla_tree / "fla/modules/conv/triton/ops.py").read_text())
    branch = next(node for node in tree.body if isinstance(node, ast.If) and "CONV_NB_MODE" in ast.unparse(node.test))
    entries = {
        name: SimpleNamespace(causal_conv1d_fwd_kernel=object(), causal_conv1d_bwd_kernel=object())
        for name in ("kernels", "kernels_none", "kernels_pow2")
    }
    for variant in ("none", "pow2"):
        for direction in ("fwd", "bwd"):
            name = f"causal_conv1d_{direction}_kernel"
            setattr(entries[f"kernels_{variant}"], f"{name}_{variant}", getattr(entries[f"kernels_{variant}"], name))
    scope = dict(vars(entries["kernels"]), CONV_NB_MODE=mode)
    scope["__builtins__"] = {"__import__": lambda name, *_args: entries[name]}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "entry_selection", "exec"), scope)
    selected = entries["kernels" if mode == "original" else f"kernels_{mode}"]
    assert scope["causal_conv1d_fwd_kernel"] is selected.causal_conv1d_fwd_kernel
    assert scope["causal_conv1d_bwd_kernel"] is selected.causal_conv1d_bwd_kernel


def test_fla_original_cache_override_is_frozen(fla_tree, monkeypatch):
    install_conv_entries(fla_tree)
    monkeypatch.setenv("RELAX_FLA_CONV_NB_MODE", "original")
    monkeypatch.setenv("RELAX_QWEN38_FLA_CONV_CACHE_REUSE", "1")
    scope = {}
    exec((fla_tree / "fla/modules/conv/triton/conv_policy.py").read_text(), scope)
    assert scope["conv_nb_kwargs"](2, 65537) == {"NB": 1}
    monkeypatch.setenv("RELAX_QWEN38_FLA_CONV_CACHE_REUSE", "0")
    assert scope["conv_nb_kwargs"](2, 65537) == {"NB": 1}


def test_fla_validation_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        validation.check_fla_conv(tmp_path, "typo")


def test_pinned_mcore_exposes_required_runtime_contracts():
    source = os.environ.get("BLACKWELL_MEGATRON_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_MEGATRON_SOURCE_DIR to the assembled and patched Megatron source root")
    validation.check_mcore(Path(source) / "megatron")


def test_pinned_hybridep_retains_both_dispatch_apis(tmp_path):
    source = os.environ.get("BLACKWELL_HYBRIDEP_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_HYBRIDEP_SOURCE_DIR to the pinned HybridEP source root")
    package = Path(source) / "deep_ep"
    validation.check_hybridep(package)
    for name in ("__init__.py", "hybrid_ep_buffer.py"):
        shutil.copy(package / name, tmp_path / name)
    p = tmp_path / "__init__.py"
    p.write_text(p.read_text().replace("from .buffer import Buffer", ""))
    with pytest.raises(ValueError, match="ordinary DeepEP"):
        validation.check_hybridep(tmp_path)


def load_methods(path, class_name, methods, globals_dict):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_dict)
    return globals_dict


@pytest.mark.parametrize("dense", [False, True])
def test_main_hybridep_padding_growth_and_combine_gradients(monkeypatch, dense):
    from types import SimpleNamespace

    import torch

    source = os.environ.get("BLACKWELL_MEGATRON_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_MEGATRON_SOURCE_DIR to the patched source root")
    path = Path(source) / "megatron/core/transformer/moe/token_dispatcher.py"
    ns = load_methods(
        path,
        "_HybridEPManager",
        {"setup_metadata", "combine"},
        {
            "torch": torch,
            "HYBRIDEP_TOKEN_ALIGNMENT": 128,
            "HAVE_HYBRIDEP_DENSE_ROUTING": dense,
            "_HYBRIDEP_INT16_EXPERT_LIMIT": 32767,
            "hybrid_ep_combine": lambda **kwargs: kwargs["x"],
        },
    )
    group = object()
    manager = SimpleNamespace(
        group=group,
        num_experts=4,
        router_topk=2,
        drop_and_pad=False,
        moe_expert_rank_capacity_factor=None,
        num_permuted_tokens=None,
        pad_multiple=None,
        config=SimpleNamespace(
            moe_hybridep_pad_variable_tokens=True,
            context_parallel_size=4,
            moe_hybridep_routing_map_mode="bool",
            moe_permute_fusion_into_hybridep=False,
        ),
    )
    for local, group_max in [(0, 65), (17, 200), (2049, 4097), (23, 64)]:

        def all_reduce(tensor, *, op, group):
            assert group is manager.group
            assert op == torch.distributed.ReduceOp.MAX
            tensor.fill_(group_max)

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        routes = torch.zeros((local, 2), dtype=torch.int64) if dense else torch.ones((local, 4), dtype=torch.bool)
        probs = torch.full((local, 4), 0.25)
        ns["setup_metadata"](manager, routes, probs)
        padded = (group_max + 127) // 128 * 128
        assert manager._padded_num_tokens == padded
        assert manager.token_probs.shape == (padded, 4)
        assert not manager.token_probs[local:].any()
        if dense:
            assert (manager.topk_idx[local:] == -1).all()
        else:
            assert not manager.routing_map[local:].any()
        manager.handle = object()
        hidden = torch.randn(padded, 8, requires_grad=True)
        combined = ns["combine"](manager, hidden)
        assert combined.shape == (local, 8)
        combined.sum().backward()
        assert (hidden.grad[:local] == 1).all()
        assert not hidden.grad[local:].any()
        assert manager.handle is None


def test_pinned_hybridep_requests_buffer_growth():
    from types import SimpleNamespace

    source = os.environ.get("BLACKWELL_HYBRIDEP_SOURCE_DIR")
    if not source:
        pytest.skip("Set BLACKWELL_HYBRIDEP_SOURCE_DIR to the pinned source root")

    class Config(SimpleNamespace):
        def is_valid(self):
            return True

    updates = []
    ns = load_methods(
        Path(source) / "deep_ep/hybrid_ep_buffer.py",
        "HybridEPBuffer",
        {"update_template_config"},
        {
            "os": os,
            "hybrid_ep_cpp": SimpleNamespace(HybridEpConfigInstance=Config, UINT8=8, UINT16=16),
        },
    )
    buffer = SimpleNamespace(
        config=Config(hidden_dim=4096, max_num_of_tokens_per_rank=512, num_of_experts_per_rank=2, num_of_nodes=1),
        num_of_hybrid_ep_ranks_per_nvlink_domain=8,
        num_of_nodes=1,
        use_fp8=False,
        num_sms_preprocessing_api=108,
        num_blocks_permute_api=8,
        num_sms_dispatch_api=16,
        num_sms_combine_api=16,
        runtime=SimpleNamespace(update_buffer=updates.append),
    )
    for tokens in (128, 4096, 256):
        ns["update_template_config"](buffer, num_of_tokens_per_rank=tokens)
    assert [config.max_num_of_tokens_per_rank for config in updates] == [512, 4096, 4096]
