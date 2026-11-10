# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""K3 EP16 local adapter precision: 56 experts, ETP1, rank16, alpha32.

Run GPU cases with RELAX_TEST_K3_OPS_GPU=1 pytest -q this_file.py. The local
tests deliberately do not claim to validate distributed dispatch.
"""

import copy
import importlib.util
import os
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[2]
GPU = os.environ.get("RELAX_TEST_K3_OPS_GPU") == "1" and torch.cuda.is_available()


def load_split():
    spec = importlib.util.spec_from_file_location("k3_expert_split", ROOT / "relax/models/kimi_k3/expert_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_adapter(input_is_parallel, device="cuda", dtype=torch.bfloat16):
    from megatron.bridge.peft.utils import GroupedExpertLinearAdapter
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection

    config = ModelParallelConfig(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        pipeline_dtype=dtype,
        context_parallel_size=4,
        expert_model_parallel_size=16,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
    )
    return GroupedExpertLinearAdapter(
        3072 if input_is_parallel else 3584,
        3584 if input_is_parallel else 6144,
        16,
        num_local_experts=56,
        base_linear_name="linear_fc2" if input_is_parallel else "linear_fc1",
        activation="identity",
        input_is_parallel=input_is_parallel,
        dropout=0.0,
        alpha=32,
        model_parallel_config=config,
        params_device=torch.device(device),
        params_dtype=dtype,
        pg_collection=ProcessGroupCollection(),
    )


def optimize(adapter):
    module = load_split()
    adapter.forward = MethodType(module.forward, adapter)
    adapter._forward_per_expert = MethodType(module._forward_per_expert, adapter)
    return adapter


def splits_for(kind):
    if kind == "empty":
        return [0] * 56
    if kind == "skew":
        return [0] * 27 + [113] + [0] * 28
    return [0 if i % 7 == 0 or i == 55 else (i * 13) % 31 + 1 for i in range(56)]


@pytest.mark.skipif(not GPU, reason="Set RELAX_TEST_K3_OPS_GPU=1 on an idle CUDA GPU")
@pytest.mark.parametrize("input_is_parallel", [False, True])
@pytest.mark.parametrize("backend", ["grouped", "per_expert"])
@pytest.mark.parametrize(
    "case,recompute,zero_b",
    [
        ("ragged", False, False),
        ("ragged", True, False),
        ("skew", True, False),
        ("empty", False, False),
        ("ragged", True, True),
    ],
)
def test_expert_split_gradients_and_optimizer(input_is_parallel, backend, case, recompute, zero_b):
    torch.manual_seed(1234)
    original = make_adapter(input_is_parallel)
    if not zero_b:
        with torch.no_grad():
            original.linear_in.weight.normal_(0, 0.02)
            original.linear_out.weight.normal_(0, 0.02)
    if backend == "per_expert":
        original._can_use_grouped_mm = lambda x: False
    else:
        assert original._can_use_grouped_mm(
            torch.empty(8, original.linear_in.weight.shape[-1], device="cuda", dtype=torch.bfloat16)
        )
    candidate = optimize(copy.deepcopy(original))
    assert original.state_dict().keys() == candidate.state_dict().keys()
    # Parameter metadata must survive unchanged: expert weights are not dense-DP reduced.
    for a, b in zip(original.parameters(), candidate.parameters(), strict=True):
        assert a.allreduce is False
        for name in ("allreduce", "tensor_model_parallel", "partition_dim", "partition_stride"):
            # deepcopy does not preserve arbitrary Parameter attrs; compare actual installation separately.
            if hasattr(b, name):
                assert getattr(a, name) == getattr(b, name)
    sizes = splits_for(case)
    width = original.linear_in.weight.shape[-1]
    out_width = original.linear_out.weight.shape[1]
    main_grads = [[torch.zeros_like(p) for p in model.parameters()] for model in (original, candidate)]
    prehooks = [0, 0]
    for index, model in enumerate((original, candidate)):

        def hook(module, args, index=index):
            prehooks[index] += 1

        model.linear_in.register_forward_pre_hook(hook)
        model.linear_out.register_forward_pre_hook(hook)
    # Multiple microbatches and nonzero B expose dx/A errors hidden at LoRA initialization.
    for microbatch in range(3):
        x0 = torch.randn(sum(sizes), width, device="cuda", dtype=torch.bfloat16) * 0.2
        dy = torch.randn(sum(sizes), out_width, device="cuda", dtype=torch.bfloat16) * 0.1
        outputs, dxs, grads = [], [], []
        for index, model in enumerate((original, candidate)):
            model.zero_grad(set_to_none=True)
            x = x0.detach().clone().requires_grad_()

            def fn(value):
                # A second branch exercises autograd accumulation into the same activation.
                return model(value, sizes) + value.sum(-1, keepdim=True) * 0.001

            y = checkpoint(fn, x, use_reentrant=True) if recompute else fn(x)
            y.backward(dy)
            outputs.append(y.detach())
            dxs.append(x.grad)
            grads.append([p.grad for p in model.parameters()])
            for acc, grad in zip(main_grads[index], grads[-1], strict=True):
                assert grad is not None
                acc.add_(grad)
        for tensor in outputs + dxs + grads[0] + grads[1]:
            assert torch.isfinite(tensor).all()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(dxs[0], dxs[1], rtol=0, atol=0)
        for a, b in zip(grads[0], grads[1], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert prehooks[0] == prehooks[1] and prehooks[0] >= 6
    for a, b in zip(*main_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # CPU FP32 master AdamW receives the accumulated BF16 gradients, matching offload's precision boundary.
    masters = [[torch.nn.Parameter(p.detach().float().cpu()) for p in m.parameters()] for m in (original, candidate)]
    opts = [torch.optim.AdamW(ps, lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1) for ps in masters]
    for ps, accs, opt in zip(masters, main_grads, opts, strict=True):
        for p, g in zip(ps, accs, strict=True):
            p.grad = g.float().cpu()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
    for a, b in zip(*masters, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(opts[0].state[a][key], opts[1].state[b][key], rtol=0, atol=0)


@pytest.mark.skipif(not GPU, reason="Set RELAX_TEST_K3_OPS_GPU=1 on an idle CUDA GPU")
@pytest.mark.parametrize("backend,expected_split_backward", [("grouped", 0), ("per_expert", 1)])
def test_expert_split_removes_full_input_slice_backward(backend, expected_split_backward):
    original = make_adapter(True)
    if backend == "per_expert":
        original._can_use_grouped_mm = lambda x: False
    with torch.no_grad():
        original.linear_out.weight.normal_(0, 0.02)
    candidate = optimize(copy.deepcopy(original))
    counts = []
    for model in (original, candidate):
        x = torch.randn(56 * 16, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = model(x, [16] * 56)
        dy = torch.ones_like(y)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            y.backward(dy)
        counts.append({e.key: e.count for e in profile.key_averages()})
    assert counts[0].get("aten::slice_backward", 0) == 56
    assert counts[1].get("aten::slice_backward", 0) == 0
    assert counts[1].get("SplitWithSizesBackward0", 0) == expected_split_backward


def test_training_ops_expert_split_installation(monkeypatch):
    import sys
    from types import ModuleType

    from megatron.bridge.peft.utils import GroupedExpertLinearAdapter

    package = ModuleType("_k3_split_install")
    package.__path__ = [str(ROOT / "relax/models/kimi_k3")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.training_ops", ROOT / "relax/models/kimi_k3/training_ops.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_forward = GroupedExpertLinearAdapter.forward
    original_per_expert = GroupedExpertLinearAdapter._forward_per_expert
    monkeypatch.setattr(GroupedExpertLinearAdapter, "forward", original_forward)
    monkeypatch.setattr(GroupedExpertLinearAdapter, "_forward_per_expert", original_per_expert)
    monkeypatch.setenv("RELAX_KIMI_K3_FUSED_SITU", "0")
    monkeypatch.setenv("RELAX_KIMI_K3_FUSED_ATTN_RES", "0")
    monkeypatch.setenv("RELAX_KIMI_K3_EXPERT_SPLIT", "0")
    module.configure_training_ops()
    assert GroupedExpertLinearAdapter.forward is original_forward
    # Install on an existing instance: the class identity, weights and checkpoint metadata survive.
    adapter = make_adapter(False, device="cpu", dtype=torch.float32)
    metadata = [(id(p), p.allreduce, p.partition_dim) for p in adapter.parameters()]
    keys = list(adapter.state_dict())
    monkeypatch.setenv("RELAX_KIMI_K3_EXPERT_SPLIT", "1")
    module.configure_training_ops()
    installed = GroupedExpertLinearAdapter.forward
    assert installed is not original_forward and installed._relax_expert_split
    module.configure_training_ops()
    assert GroupedExpertLinearAdapter.forward is installed
    assert metadata == [(id(p), p.allreduce, p.partition_dim) for p in adapter.parameters()]
    assert keys == list(adapter.state_dict())


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_64gpu_recipe_expert_split_worker_environment(enabled, tmp_path):
    import json
    import shlex
    import subprocess

    env = dict(os.environ, DRY_RUN="1", MODEL_DIR=str(tmp_path / "model"), DATA_DIR=str(tmp_path / "data"))
    env.pop("RELAX_KIMI_K3_EXPERT_SPLIT", None)
    if enabled is not None:
        env["RELAX_KIMI_K3_EXPERT_SPLIT"] = enabled
    command = subprocess.check_output(
        ["bash", str(ROOT / "examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh")],
        env=env,
        text=True,
    )
    args = shlex.split(command)
    worker_env = json.loads(args[args.index("--train-env-vars") + 1])
    assert worker_env["RELAX_KIMI_K3_EXPERT_SPLIT"] == (enabled or "1")
    for flag, value in (
        ("--tensor-model-parallel-size", "4"),
        ("--pipeline-model-parallel-size", "4"),
        ("--expert-tensor-parallel-size", "1"),
    ):
        assert args[args.index(flag) + 1] == value
