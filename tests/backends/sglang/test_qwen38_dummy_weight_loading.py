# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute SGLang's actual dummy/fused-projection lifecycle without a full
model."""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_function(path: Path, class_name: str | None, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    if class_name is not None:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def dummy_lifecycle():
    source = os.environ.get("SGLANG_SOURCE_DIR")
    if not source:
        pytest.skip("Set SGLANG_SOURCE_DIR to the patched SGLang checkout")
    root = Path(source) / "python/sglang/srt"

    class UnquantizedLinearMethod:
        pass

    class GDN(torch.nn.Module):
        def __init__(self, device: str):
            super().__init__()
            self.in_proj_qkvz = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16, device=device)
            self.in_proj_ba = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16, device=device)
            for projection in (self.in_proj_qkvz, self.in_proj_ba):
                projection.quant_method = UnquantizedLinearMethod()
            self._fused_in_proj_weight = None
            self._fused_in_proj_qkvz_width = 0

    class PLE(torch.nn.Module):
        def __init__(self, device: str):
            super().__init__()
            self.ngram_embedding = torch.nn.Embedding(4, 4, device=device, dtype=torch.bfloat16)
            self.ngram_embedding.register_buffer("weight_scale", torch.ones(1, device=device, dtype=torch.bfloat16))

    class HC(torch.nn.Module):
        def finalize_mix_weights(self):
            self.finalized = True

    class Model(torch.nn.Module):
        def __init__(self, device: str):
            super().__init__()
            self.gdn = GDN(device)
            self.ple = PLE(device)
            self.hc = HC()

    config = SimpleNamespace(load_format="dummy")
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "_is_cuda": True,
        "UnquantizedLinearMethod": UnquantizedLinearMethod,
        "get_lora": lambda: SimpleNamespace(enable_lora=False, lora_paths=None),
        "get_model": lambda: config,
        "Qwen3_5GatedDeltaNet": GDN,
        "Qwen4ExpNGramEmbedding": PLE,
        "GatedResidual": HC,
    }
    GDN.finalize_fused_in_proj = _load_function(
        root / "models/qwen3_5.py", "Qwen3_5GatedDeltaNet", "finalize_fused_in_proj", namespace
    )
    for method in ("post_load_weights", "_finalize_fused_in_proj"):
        implementation = _load_function(
            root / "models/qwen4_exp.py", "Qwen4ExpForConditionalGeneration", method, namespace
        )
        setattr(Model, method, implementation)
    initialize = _load_function(root / "model_loader/weight_utils.py", None, "initialize_dummy_weights", namespace)
    post_load = _load_function(root / "model_loader/loader.py", None, "_post_load_weights", namespace)
    return SimpleNamespace(Model=Model, config=config, initialize=initialize, post_load=post_load, root=root)


def test_qwen38_dummy_restores_scale_and_preserves_projection_storage(dummy_lifecycle):
    model = dummy_lifecycle.Model("cpu")
    dummy_lifecycle.initialize(model)
    assert not torch.equal(model.ple.ngram_embedding.weight_scale, torch.ones(1))
    dummy_lifecycle.post_load(model)
    assert model.hc.finalized
    assert model.ple.ngram_embedding.weight_scale.item() == 1
    qkvz = model.gdn.in_proj_qkvz.weight
    captured_storage = qkvz.detach()
    old_pointer = qkvz.data_ptr()
    with torch.no_grad():
        qkvz.fill_(2)
    model._finalize_fused_in_proj()
    assert qkvz.data_ptr() == old_pointer
    assert torch.equal(captured_storage, torch.full_like(captured_storage, 2))
    assert torch.equal(model.gdn._fused_in_proj_weight[:8], captured_storage)


@pytest.mark.parametrize("load_format", ["auto", "safetensors", "sharded_state", "remote_instance"])
def test_qwen38_real_checkpoint_scale_survives_post_load(dummy_lifecycle, load_format):
    model = dummy_lifecycle.Model("cpu")
    dummy_lifecycle.config.load_format = load_format
    model.ple.ngram_embedding.weight_scale.fill_(0.25)
    dummy_lifecycle.post_load(model)
    assert model.ple.ngram_embedding.weight_scale.item() == 0.25


def test_qwen38_reload_finalization_preserves_received_fp8_scale(dummy_lifecycle):
    model = dummy_lifecycle.Model("cpu")
    dummy_lifecycle.initialize(model)
    dummy_lifecycle.post_load(model)
    model.ple.ngram_embedding.weight_scale.fill_(0.25)
    model._finalize_fused_in_proj()
    assert model.ple.ngram_embedding.weight_scale.item() == 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_qwen38_dummy_graph_reads_reloaded_projection(dummy_lifecycle):
    model = dummy_lifecycle.Model("cuda")
    dummy_lifecycle.initialize(model)
    dummy_lifecycle.post_load(model)
    gdn = model.gdn
    x = torch.ones((2, 4), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.nn.functional.linear(x, gdn._fused_in_proj_weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.nn.functional.linear(x, gdn._fused_in_proj_weight)
    pointer = gdn._fused_in_proj_weight.data_ptr()
    with torch.no_grad():
        gdn.in_proj_qkvz.weight.fill_(2)
        gdn.in_proj_ba.weight.fill_(3)
    model._finalize_fused_in_proj()
    graph.replay()
    torch.cuda.synchronize()
    assert pointer == gdn._fused_in_proj_weight.data_ptr()
    assert torch.equal(output[:, :8], torch.full_like(output[:, :8], 8))
    assert torch.equal(output[:, 8:], torch.full_like(output[:, 8:], 12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_qwen38_late_projection_fusion_reproduces_stale_graph(dummy_lifecycle):
    model = dummy_lifecycle.Model("cuda")
    dummy_lifecycle.initialize(model)
    gdn = model.gdn
    x = torch.ones((2, 4), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gdn.in_proj_qkvz(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    # Reproduce the original dummy loader: capture before load_weights first
    # establishes fused storage. Hold the old allocation as a graph would.
    old_storage = gdn.in_proj_qkvz.weight.detach()
    with torch.cuda.graph(graph):
        output = gdn.in_proj_qkvz(x)
    expected_dummy = torch.nn.functional.linear(x, old_storage)
    model._finalize_fused_in_proj()
    with torch.no_grad():
        gdn.in_proj_qkvz.weight.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert old_storage.data_ptr() != gdn.in_proj_qkvz.weight.data_ptr()
    assert torch.equal(output, expected_dummy)
    assert not torch.equal(output, gdn.in_proj_qkvz(x))


def test_qwen38_mtp_post_load_initializes_hc_and_reload_preserves_storage(dummy_lifecycle, monkeypatch):
    # Execute the installed hook and HC lifecycle methods. Only the CUDA
    # permutation kernel is replaced with a CPU allocation of equivalent dtype.
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.elementwise.hc_mix",
        SimpleNamespace(permute_pad_up_weight=lambda weight, count: weight.detach().clone()),
    )

    class HC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_mix_weight_up = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
            self.hc_count = 2
            self._jit_mix_ok = True
            self._match_training = False
            self._mix_up_weight_padded = None

    namespace = {"torch": torch, "GatedResidual": HC}
    for name in ("finalize_mix_weights", "_load_mix_up_weight"):
        setattr(
            HC,
            name,
            _load_function(dummy_lifecycle.root / "layers/hyperconnection.py", "GatedResidual", name, namespace),
        )

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hc = HC()

    Draft.post_load_weights = _load_function(
        dummy_lifecycle.root / "models/qwen4_exp_mtp.py",
        "Qwen4ExpForCausalLMMTP",
        "post_load_weights",
        namespace,
    )
    model = Draft()
    dummy_lifecycle.initialize(model)
    assert model.hc._mix_up_weight_padded is None
    dummy_lifecycle.post_load(model)
    cached = model.hc._mix_up_weight_padded
    assert cached is not None
    pointer = cached.data_ptr()
    assert torch.equal(cached, model.hc.input_mix_weight_up.weight)
    with torch.no_grad():
        model.hc._load_mix_up_weight(
            model.hc.input_mix_weight_up.weight, torch.full_like(model.hc.input_mix_weight_up.weight, 2)
        )
        dummy_lifecycle.post_load(model)
    assert model.hc._mix_up_weight_padded.data_ptr() == pointer
    assert torch.equal(cached, torch.full_like(cached, 2))


@pytest.mark.parametrize("restore,process", [(True, False), (False, True), (True, True)])
def test_eagle_v2_post_process_delegates_only_to_draft(dummy_lifecycle, restore, process):
    calls = []
    namespace = {
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda **kwargs: calls.append("barrier"))),
        "logger": SimpleNamespace(error=lambda message: calls.append(("error", message))),
        "PostProcessWeightsReqOutput": SimpleNamespace,
    }
    delegate = _load_function(
        dummy_lifecycle.root / "speculative/eagle_worker_v2.py", "EAGLEWorkerV2", "post_process_weights", namespace
    )
    scheduler = _load_function(
        dummy_lifecycle.root / "managers/scheduler_components/weight_updater.py",
        "SchedulerWeightUpdaterManager",
        "post_process_weights",
        namespace,
    )

    def draft_hook(**kwargs):
        calls.append(("draft", kwargs))
        return True, "draft processed"

    def target_hook(request):
        calls.append(("target", request))
        return True, "target processed"

    worker = SimpleNamespace(
        _draft_worker=SimpleNamespace(draft_runner=SimpleNamespace(post_process_weights=draft_hook))
    )
    worker.post_process_weights = lambda request: delegate(worker, request)
    manager = SimpleNamespace(
        tp_worker=SimpleNamespace(post_process_weights=target_hook), draft_worker=worker, tp_cpu_group=object()
    )
    request = SimpleNamespace(restore_weights_before_load=restore, post_process_quantization=process)
    result = scheduler(manager, request)
    assert result.success and result.message == "draft processed"
    assert calls == [
        ("target", request),
        ("draft", {"restore_weights_before_load": restore, "post_process_quantization": process}),
        "barrier",
    ]


def test_eagle_v2_post_process_non_draft_pp_rank_is_noop(dummy_lifecycle):
    delegate = _load_function(
        dummy_lifecycle.root / "speculative/eagle_worker_v2.py", "EAGLEWorkerV2", "post_process_weights", {}
    )
    assert delegate(SimpleNamespace(_draft_worker=None), None) == (True, "Success")


def test_eagle_v2_post_process_preserves_draft_failure(dummy_lifecycle):
    delegate = _load_function(
        dummy_lifecycle.root / "speculative/eagle_worker_v2.py", "EAGLEWorkerV2", "post_process_weights", {}
    )
    worker = SimpleNamespace(
        _draft_worker=SimpleNamespace(
            draft_runner=SimpleNamespace(post_process_weights=lambda **kwargs: (False, "repacking failed"))
        )
    )
    request = SimpleNamespace(restore_weights_before_load=False, post_process_quantization=True)
    assert delegate(worker, request) == (False, "repacking failed")
