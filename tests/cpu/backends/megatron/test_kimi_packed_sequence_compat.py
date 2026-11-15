# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check Kimi's packed embedding call against the pinned Bridge helper."""

import ast
from importlib import metadata, util
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version


torch = pytest.importorskip("torch")
try:
    _sglang_version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(Version(_sglang_version) < Version("0.5.21"), reason="CUDA 13 SGLang 0.5.21 image")


def _bridge_root() -> Path:
    spec = util.find_spec("megatron")
    roots = spec.submodule_search_locations if spec is not None else []
    for location in roots or []:
        path = Path(location) / "bridge/models"
        if (path / "kimi_vl/modeling_kimi_k25_vl.py").is_file():
            return path
    pytest.skip("Megatron Bridge with Kimi support is required")


@pytest.mark.parametrize("tp,cp,rank", [(2, 2, 1)])
@pytest.mark.parametrize("mask_dtype", [torch.bool])
def test_kimi_packed_call_preserves_sequence_first_cp_layout(tp: int, cp: int, rank: int, mask_dtype) -> None:
    root = _bridge_root()
    utility = root / "qwen_vl/modelling_qwen3_vl/utils.py"
    function = next(
        node
        for node in ast.parse(utility.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "preprocess_packed_seqs"
    )
    namespace = {
        "torch": torch,
        "PackedSeqParams": SimpleNamespace,
        "mpu": SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: tp,
            get_context_parallel_world_size=lambda: cp,
            get_context_parallel_rank=lambda: rank,
        ),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(utility), "exec"), namespace)
    helper = namespace[function.name]
    captured = {}

    def preprocess(*args, **kwargs):
        result = helper(*args, **kwargs)
        captured["metadata"] = result[1]
        return result

    model_path = root / "kimi_vl/modeling_kimi_k25_vl.py"
    tree = ast.parse(model_path.read_text())
    expression = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "inputs_embeds" for target in node.targets)
        and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "preprocess_packed_seqs"
            for call in ast.walk(node.value)
        )
    )
    inputs = torch.arange(1, 37, dtype=torch.float32).reshape(2, 6, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=mask_dtype)
    output = eval(
        compile(ast.Expression(expression), str(model_path), "eval"),
        {
            "inputs_embeds": inputs,
            "saved_attention_mask": mask,
            "packed_seq_params": SimpleNamespace(),
            "preprocess_packed_seqs": preprocess,
        },
    )
    align = tp * cp * 2 if cp > 1 else tp
    expected = []
    padded_lengths = []
    for row, length in zip(inputs, [5, 2], strict=True):
        padded = ((length + align - 1) // align) * align
        padded_lengths.append(padded)
        full = torch.cat((row[:length], torch.zeros(padded - length, 3)))
        if cp > 1:
            chunks = full.chunk(2 * cp)
            full = torch.cat((chunks[rank], chunks[2 * cp - rank - 1]))
        expected.append(full)
    expected = torch.cat(expected).unsqueeze(1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.is_contiguous()
    assert output.shape[0] % tp == 0
    params = captured["metadata"]
    assert params.qkv_format == "thd"
    assert params.cu_seqlens_q.tolist() == [0, padded_lengths[0], sum(padded_lengths)]
    assert params.cu_seqlens_kv.tolist() == params.cu_seqlens_q.tolist()


@pytest.mark.parametrize("cp,has_labels,has_mask", [(2, True, True), (1, False, False)])
def test_kimi_forward_returns_mask_only_when_present(cp: int, has_labels: bool, has_mask: bool) -> None:
    model_path = _bridge_root() / "kimi_vl/modeling_kimi_k25_vl.py"
    tree = ast.parse(model_path.read_text())
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "forward"
        and any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "outputs" for target in statement.targets)
            for statement in node.body
        )
    )
    start = next(
        index
        for index, statement in enumerate(forward.body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "outputs" for target in statement.targets)
    )
    names = [
        "self",
        "cp_size",
        "inputs_embeds",
        "position_ids",
        "attention_mask",
        "labels",
        "loss_mask",
        "runtime_gather_output",
        "packed_seq_params",
    ]
    function = ast.FunctionDef(
        name="forward_tail",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in names],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=forward.body[start:],
        decorator_list=[],
        type_params=[],
    )
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(model_path), "exec"),
        namespace,
    )
    output = torch.randn(2, 3, 4, requires_grad=True)
    labels = torch.ones(2, 3, dtype=torch.long) if has_labels else None
    mask = torch.ones(2, 3) if has_mask else None
    captured = {}

    def language_forward(**kwargs):
        captured.update(kwargs)
        return output

    result = namespace["forward_tail"](
        SimpleNamespace(language_model=SimpleNamespace(forward=language_forward)),
        cp,
        torch.zeros(3, 2, 4),
        None,
        None,
        labels,
        mask,
        False,
        None,
    )
    if cp > 1 and has_mask:
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] is output and result[1] is mask
        logits_or_loss = result[0]
    else:
        assert result is output
        logits_or_loss = result
    assert captured["labels"] is labels and captured["loss_mask"] is mask
    logits_or_loss.sum().backward()
    torch.testing.assert_close(output.grad, torch.ones_like(output))
