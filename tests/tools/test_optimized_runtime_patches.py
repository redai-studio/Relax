# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Replay dependency deltas on pinned sources, without importing GPU code."""

import ast
import copy
import functools
import hashlib
import inspect
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_manifest_patch_hashes_use_files_copied_into_runtime_stage(tmp_path):
    """Builder-only files must not satisfy the runtime manifest's inputs."""
    dockerfile = (ROOT / "docker/Dockerfile.cu13").read_text().replace("\\\n", "")
    stage = None
    copied = {}
    for line in dockerfile.splitlines():
        if not line.startswith(("FROM ", "COPY ")):
            continue
        words = shlex.split(line)
        if words[0] == "FROM":
            stage = words[-1]
        if words[0] != "COPY" or stage not in {"relax_build", "relax_ep", "relax_blackwell"}:
            continue
        if words[1].startswith("--"):
            continue
        for source in words[1:-1]:
            if not source.endswith(".patch") or "$" in source:
                continue
            destination = words[-1]
            if destination.endswith("/"):
                destination += Path(source).name
            target = tmp_path / destination.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, target)
            copied[destination] = ROOT / source

    tree = ast.parse((ROOT / "docker/scripts/record_qwen35_runtime.py").read_text())
    report = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "report" for t in n.targets)
    )
    expression = next(value for key, value in zip(report.keys, report.values) if key.value == "patch_sha256")
    result = eval(
        compile(ast.Expression(expression), "<manifest-patch-hashes>", "eval"),
        {"hashlib": hashlib, "Path": lambda path: tmp_path / path.lstrip("/")},
    )
    paths = ast.literal_eval(expression.generators[0].iter.func.value)
    assert set(result) == {"deepep", "fa4", "flashqla", "megatron", "model_operators", "checkpoint"}
    for name, path in paths.items():
        assert result[name] == hashlib.sha256(copied[path].read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "patch",
    [
        "megatron/20260915-4a4de8657.patch",
        "megatron/20261016-qwen38-models.patch",
        "megatron/20261020-qwen35-optimized.patch",
        "megatron/20261020-model-operators.patch",
        "megatron/20261028-dp-reshardable-checkpoint.patch",
        "transformer_engine/2.18.0-cp-padding-no-host-sync.patch",
    ],
)
def test_dependency_patch_hunk_counts_are_valid(patch):
    # Parse the complete patch even when pinned source trees are unavailable.
    subprocess.run(
        ["git", "apply", "--numstat", str(ROOT / "docker/patch" / patch)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "variable,patch",
    [
        ("OPTIMIZED_MCORE_SOURCE", "megatron/20261020-qwen35-optimized.patch"),
        ("MODEL_OPERATORS_SOURCE", "megatron/20261020-model-operators.patch"),
        ("OPTIMIZED_DEEPEP_SOURCE", "deepep/2.1.0-relax-namespace.patch"),
        ("OPTIMIZED_FA4_SOURCE", "flash-attention/56e48e3-fa4-isolated.patch"),
        ("OPTIMIZED_FLASHQLA_SOURCE", "flash-qla/0.1.3-kkt-no-tma.patch"),
    ],
)
def test_optimized_patch_replays_without_fuzz(variable, patch, tmp_path):
    source = os.environ.get(variable)
    if not source:
        pytest.skip(f"Set {variable} to the pinned source (MCore includes the existing three patches)")
    patchfile = ROOT / "docker/patch" / patch
    # Copy only touched files, so tests do not duplicate entire native source trees.
    for line in patchfile.read_text().splitlines():
        if line.startswith("--- a/"):
            name = line[6:]
            dest = tmp_path / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source) / name, dest)
        elif line.startswith("+++ b/"):
            (tmp_path / line[6:]).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patchfile)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "offset" not in result.stdout and "fuzz" not in result.stdout
    for path in tmp_path.rglob("*.py"):
        compile(path.read_text(), str(path), "exec")
        if variable == "MODEL_OPERATORS_SOURCE":
            _check_model_helper_signatures(path)
    if variable == "OPTIMIZED_MCORE_SOURCE":
        mtp = (tmp_path / "megatron/core/transformer/multi_token_prediction.py").read_text()
        tree = ast.parse(mtp)
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_relax_selected_mtp_head"
        )
        assert ast.unparse(call.args[1]) == "layer_loss_mask"
        assert "cumulative_mtp_input_mask" in mtp
        _exercise_fused_expert_contract((tmp_path / "megatron/core/transformer/moe/experts.py").read_text())
        _exercise_lazy_deepep_contract((tmp_path / "megatron/core/transformer/moe/fused_a2a.py").read_text())
        backward = tmp_path / "megatron/core/fusions/linear_cross_entropy/blackwell/bwd_partial_dlogits.py"
        output_tile = next(
            node
            for node in ast.walk(ast.parse(backward.read_text()))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.local_tile"
            and node.args
            and ast.unparse(node.args[0]) == "mDlogits_partial"
        )
        # Widen before multiplying by the row stride; a 64-bit final pointer is insufficient.
        assert ast.unparse(output_tile.args[2]) == "(cutlass.Int64(pidm), cutlass.Int64(pidn))"
    if variable == "OPTIMIZED_DEEPEP_SOURCE":
        setup = (tmp_path / "setup.py").read_text()
        assert "name='relax-deep-ep-v2'" in setup
        assert "'_relax_deep_ep_v2._C'" in setup
        for path in (tmp_path / "deep_ep").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("deep_ep.")
                elif isinstance(node, ast.Import):
                    assert all(not a.name.startswith("deep_ep.") for a in node.names)
    if variable == "OPTIMIZED_FLASHQLA_SOURCE":
        source = tmp_path / "flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py"
        tree = ast.parse(source.read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_relax_verify_kkt_store")
        scope = {"lru_cache": functools.lru_cache}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"), scope)

        class Kernel:
            def __init__(self, code):
                self.code = code
                self.reads = 0

            def get_kernel_source(self):
                self.reads += 1
                return self.code

        check = scope["_relax_verify_kkt_store"]
        safe = Kernel("ordinary_store")
        check(safe)
        check(safe)
        assert safe.reads == 1
        with pytest.raises(RuntimeError, match="TMA store"):
            check(Kernel("tl::tma_store(...)"))


def _check_model_helper_signatures(path):
    """Bind actual patched call sites against helpers without importing
    CUDA."""
    for function in ast.walk(ast.parse(path.read_text())):
        if not isinstance(function, ast.FunctionDef):
            continue
        imports = {
            alias.asname or alias.name: (node.module, alias.name)
            for node in ast.walk(function)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("relax.models.")
            for alias in node.names
        }
        for call in ast.walk(function):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id not in imports or any(isinstance(arg, ast.Starred) for arg in call.args):
                continue
            if any(keyword.arg is None for keyword in call.keywords):
                continue
            module, name = imports[call.func.id]
            source = ROOT / (module.replace(".", "/") + ".py")
            if not source.exists():
                source = source.with_suffix("") / "__init__.py"
            if not source.exists():
                continue
            helper = next(
                (
                    node
                    for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                ),
                None,
            )
            if helper is None:
                continue
            args = copy.deepcopy(helper.args)
            for argument in ast.walk(args):
                if isinstance(argument, ast.arg):
                    argument.annotation = None
            args.defaults = [ast.Constant(None) for _ in args.defaults]
            args.kw_defaults = [None if value is None else ast.Constant(None) for value in args.kw_defaults]
            stub = ast.FunctionDef(name="helper", args=args, body=[ast.Pass()], decorator_list=[])
            namespace = {}
            exec(
                compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(source), "exec"),
                namespace,
            )
            try:
                inspect.signature(namespace["helper"]).bind(
                    *[None for _ in call.args], **{keyword.arg: None for keyword in call.keywords}
                )
            except TypeError as error:
                pytest.fail(f"{path}:{call.lineno} -> {module}.{name}: {error}")


@pytest.mark.parametrize("marker", ["", '; platform_system == "Linux" and platform_machine == "x86_64"'])
def test_runtime_check_accepts_only_documented_nccl_override(marker):
    validator = _load_dependency_validator()
    line = (
        f"torch 2.11.0+cu130 has requirement nvidia-nccl-cu13==2.29.7{marker}, but you have nvidia-nccl-cu13 2.30.7.\n"
    )
    validator(subprocess.CompletedProcess([], 1, line, ""))
    validator(subprocess.CompletedProcess([], 0, "No broken requirements found.\n", ""))


@pytest.mark.parametrize(
    "output",
    [
        "deep-ep 2.1.0 requires pynvml, which is not installed.",
        "torch 2.11.0 has requirement nvidia-nccl-cu13==2.29.7, but you have nvidia-nccl-cu13 2.30.6.",
        "other 1.0 has requirement nvidia-nccl-cu13==2.29.7, but you have nvidia-nccl-cu13 2.30.7.",
        "torch 2.11.0 has requirement numpy==1.0, but you have numpy 2.0.",
        "",
    ],
)
def test_runtime_check_rejects_missing_or_unrelated_conflicts(output):
    with pytest.raises(RuntimeError):
        _load_dependency_validator()(subprocess.CompletedProcess([], 1, output, ""))


def _load_dependency_validator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "delivery_manifest", ROOT / "docker/scripts/record_qwen35_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_pip_check


@pytest.mark.parametrize("installed", [None, "0.1.10", "0.1.11"])
def test_copied_jit_source_dependencies_are_checked(tmp_path, monkeypatch, installed):
    scope = _load_dependency_validator().__globals__
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["apache-tvm-ffi>=0.1.11,<0.2"]\n')

    def version(name):
        if installed is None:
            raise scope["md"].PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(scope["md"], "version", version)
    check = scope["validate_source_dependencies"]
    if installed == "0.1.11":
        check(pyproject)
    else:
        with pytest.raises(RuntimeError, match="apache-tvm-ffi"):
            check(pyproject)


def _exercise_lazy_deepep_contract(source):
    """Legacy imports must not initialize private NCCL/JIT state."""
    import builtins
    from types import SimpleNamespace

    imported = []

    class FakeBuffer:
        @staticmethod
        def get_buffer_size_hint(group, **kwargs):
            return 64

        def __init__(self, group, **kwargs):
            self.group = group
            self.__dict__.update(kwargs)

    def import_backend(name, *args, **kwargs):
        if name == "importlib.util":
            return SimpleNamespace(find_spec=lambda name: object())
        if name == "_relax_deep_ep_v2":
            imported.append(name)
            return SimpleNamespace(ElasticBuffer=FakeBuffer)
        return builtins.__import__(name, *args, **kwargs)

    nodes = [
        node
        for node in ast.parse(source).body
        if (isinstance(node, ast.ImportFrom) and node.module == "importlib.util")
        or (isinstance(node, ast.If) and "find_spec" in ast.unparse(node.test))
        or (isinstance(node, ast.FunctionDef) and node.name == "get_elastic_buffer")
    ]
    scope = {"__builtins__": {**vars(builtins), "__import__": import_backend}, "_elastic_buffer": None}
    exec("from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])), scope)
    assert scope["HAVE_DEEP_EP_V2"] and imported == []
    group = object()
    buffer = scope["get_elastic_buffer"](group, 256, 2048, 8)
    assert buffer is scope["get_elastic_buffer"](group, 256, 2048, 8)
    assert imported == ["_relax_deep_ep_v2"]


def _exercise_fused_expert_contract(source):
    """Execute the patched method without CUDA/TE to check dispatch and
    padding."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "TEGroupedMLP")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_fused_forward")
    scope = {
        "torch": torch,
        "nullcontext": nullcontext,
        "skip_routed_expert_padding": lambda config: False,
    }

    class Offload:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self.value

        def __exit__(self, *args):
            return False

        def group_offload(self, value, **kwargs):
            return value

    scope["off_interface"] = lambda enabled, value, name: Offload(value)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "experts.py", "exec"), scope)
    config = SimpleNamespace(
        fp8=False,
        fp4=False,
        moe_paged_stash=False,
        delay_offload_until_cuda_graph=False,
        _relax_qwen35_optimized=True,
    )
    counts = torch.tensor([1, 0, 2], dtype=torch.int32)
    hidden = torch.randn(3, 4, requires_grad=True)
    probs = torch.ones(3)
    calls = []

    def fused(value, fc1_counts, probabilities, fc2_counts):
        assert fc1_counts.dtype == torch.int64
        assert fc1_counts is fc2_counts
        assert fc1_counts.tolist() == [1, 0, 2]
        assert value is hidden and probabilities is probs
        calls.append(True)
        return value * 2

    module = SimpleNamespace(config=config, _use_grouped_tensor=False, _fused_ops=(fused,))
    result = scope["_fused_forward"](module, hidden, counts, probs)
    result.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2))
    assert calls == [True]
    config._relax_qwen35_optimized = False
    with pytest.raises(RuntimeError, match="moe_use_grouped_tensor"):
        scope["_fused_forward"](module, hidden, counts, probs)


def test_rebuilt_fa4_matches_delivered_implementation(tmp_path):
    import json

    source = os.environ.get("OPTIMIZED_FA4_SOURCE")
    if not source:
        pytest.skip("Set OPTIMIZED_FA4_SOURCE to the pinned FA4 source")
    shutil.copytree(Path(source) / "flash_attn/cute", tmp_path / "flash_attn/cute")
    subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "-i",
            str(ROOT / "docker/patch/flash-attention/56e48e3-fa4-isolated.patch"),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("manifest", ROOT / "docker/scripts/record_qwen35_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pins = json.loads((ROOT / "docker/patch/flash-attention/fa4-source-pins.json").read_text())
    assert module.python_ast_hashes(tmp_path / "flash_attn/cute") == pins["fa4-isolated-python-ast"]


def test_conv_backward_passes_strides_to_current_fla_kernel():
    from types import SimpleNamespace

    import torch

    source = ROOT / "relax/models/qwen3_5/ops/causal_conv_backward.py"
    fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*, stride_dy_n, stride_dy_t, stride_dy_d, **kwargs):
                assert (stride_dy_n, stride_dy_t, stride_dy_d) == kwargs["dy"].stride()
                assert tuple(kwargs[f"stride_dx_{axis}"] for axis in "ntd") == kwargs["dx"].stride()
                kwargs["dx"].copy_(kwargs["dy"])
                kwargs["dw"].zero_()
                launches.append(True)

            return launch

    scope = {
        "torch": torch,
        "triton": SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda n, d: (n + d - 1) // d
        ),
        "causal_conv1d_bwd_kernel": Kernel(),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    x = torch.randn(1, 7, 10)[..., :4]
    dy = torch.randn(1, 7, 8)[..., ::2]
    dx = torch.empty(1, 7, 12)[..., :4]
    result = scope["causal_conv1d_bwd"](x, dy, None, weight=torch.randn(4, 3), dx_out=dx)
    assert result[0].data_ptr() == dx.data_ptr()
    torch.testing.assert_close(result[0], dy)
    assert launches == [True]
