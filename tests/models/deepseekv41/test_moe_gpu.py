# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real V4.1 MoE formula, EP and DeepEP checks on two explicitly selected GPUs.

Run with RELAX_TEST_V41_GPU=1 and CUDA_VISIBLE_DEVICES containing two idle
GPUs. The fixture reduces dimensions while retaining V4.1's frozen sqrtsoftplus
router, expert correction bias, grouped SwiGLU experts and clamped shared
expert.
"""

import datetime
import gc
import json
import os
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    os.environ.get("RELAX_TEST_V41_GPU") != "1" or torch.cuda.device_count() < 2,
    reason="Requires RELAX_TEST_V41_GPU=1 and two explicitly selected idle GPUs.",
)

SEED = 419
EXPERTS = 8
TOPK = 2
HIDDEN = 256
INTERMEDIATE = 128
TOKENS = 64
CLAMP = 1.0
SCALE = 1.5


def _canonical(name: str, offset: int) -> str:
    if name.startswith("experts."):
        prefix, index = name.rsplit("weight", 1)
        return f"{prefix}weight{int(index) + offset}"
    return name


def _make_model(ep_size: int, dispatcher: str, recompute: bool, skew: bool) -> torch.nn.Module:
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.models.deepseekv41.model import _freeze_routers

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        ffn_hidden_size=INTERMEDIATE,
        moe_ffn_hidden_size=INTERMEDIATE,
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_grouped_gemm=True,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_router_score_function="sqrtsoftplus",
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_load_balancing_type="none",
        moe_router_topk_scaling_factor=SCALE,
        moe_shared_expert_intermediate_size=INTERMEDIATE,
        moe_shared_expert_overlap=False,
        moe_token_dispatcher_type=dispatcher,
        moe_flex_dispatcher_backend="deepep",
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        activation_func_clamp_value=CLAMP,
        activation_func_clamp_shared_expert=True,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["moe"] if recompute else [],
    )
    model = get_moe_module_spec(num_experts=EXPERTS, moe_grouped_gemm=True)(config=config, layer_number=1).cuda()
    offset = model.local_expert_indices[0]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter_seed = SEED + zlib.crc32(_canonical(name, offset).encode())
            generator = torch.Generator(device="cuda").manual_seed(parameter_seed)
            parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.1)
        model.router.expert_bias.zero_()
        if skew:
            model.router.expert_bias[:TOPK] = 100
    _freeze_routers(SimpleNamespace(decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=model)])))
    return model


def _oracle(value: torch.Tensor, weights: dict[str, torch.Tensor], bias: torch.Tensor) -> torch.Tensor:
    """Independent PyTorch definition with the same BF16 rounding and FP32
    router."""
    flat = value.reshape(-1, HIDDEN)
    scores = F.softplus(F.linear(flat.float(), weights["router.weight"].float())).sqrt()
    indices = torch.topk(scores + bias, TOPK, dim=-1).indices
    selected = scores.gather(1, indices)
    probabilities = selected / selected.sum(-1, keepdim=True) * SCALE
    output = torch.zeros_like(flat)

    def mlp(fc1: torch.Tensor, fc2: torch.Tensor, probability: torch.Tensor | None = None) -> torch.Tensor:
        gate, linear = F.linear(flat, fc1).chunk(2, dim=-1)
        activation = F.silu(gate.clamp(max=CLAMP)) * linear.clamp(-CLAMP, CLAMP)
        if probability is not None:
            activation = (activation * probability).to(flat.dtype)
        return F.linear(activation, fc2)

    for expert in range(EXPERTS):
        probability = ((indices == expert) * probabilities).sum(-1, keepdim=True)
        output = output + mlp(
            weights[f"experts.linear_fc1.weight{expert}"], weights[f"experts.linear_fc2.weight{expert}"], probability
        )
    output = output + mlp(weights["shared_experts.linear_fc1.weight"], weights["shared_experts.linear_fc2.weight"])
    return output.view_as(value)


def _compare(actual: torch.Tensor, expected: torch.Tensor, name: str, records: list[dict], tolerance: float) -> None:
    actual, expected = actual.double(), expected.double()
    difference = actual - expected
    error, reference = difference.norm().item(), expected.norm().item()
    records.append(
        {"name": name, "relative_l2": error / max(reference, 1e-30), "max_abs": difference.abs().max().item()}
    )
    assert torch.isfinite(actual).all(), name
    assert error <= tolerance * reference + 2e-5, f"{name}: rel_l2={error / max(reference, 1e-30):.6g}"


def _backward(
    model: torch.nn.Module, value: torch.Tensor, gradient: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output, bias = model(value)
    assert bias is None
    output.backward(gradient)
    assert model.router.weight.grad is None
    assert model.router.expert_bias.grad is None
    gradients = {
        _canonical(name, model.local_expert_indices[0]): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return output.detach(), gradients


def _check_frozen_router(model: torch.nn.Module, source: torch.Tensor, records: list[dict]) -> list[int]:
    value = source.detach().clone().requires_grad_()
    actual, routing_map = model.router(value)
    reference_input = source.double().requires_grad_()
    scores = F.softplus(F.linear(reference_input.reshape(-1, HIDDEN), model.router.weight.double())).sqrt()
    indices = torch.topk(scores + model.router.expert_bias.double(), TOPK, dim=-1).indices
    selected = scores.gather(1, indices)
    probabilities = selected / selected.sum(-1, keepdim=True) * SCALE
    expected = torch.zeros_like(scores).scatter(1, indices, probabilities)
    cotangent = torch.linspace(-1, 1, EXPERTS, device="cuda")
    (actual * cotangent).sum().backward()
    (expected * cotangent.double()).sum().backward()
    assert value.grad.norm() > 0
    assert model.router.weight.grad is model.router.expert_bias.grad is None
    _compare(actual, expected, "router_fp64/probabilities", records, 0.01)
    _compare(value.grad, reference_input.grad, "router_fp64/input_grad", records, 0.01)
    return routing_map.sum(0).tolist()


def _worker(rank: int, rendezvous: str, output_path: str, skew: bool, recompute: bool) -> None:
    from deep_ep import Buffer
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe import fused_a2a

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=120)
    )
    records = []
    buffer = None
    try:
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        source = torch.randn(TOKENS, 1, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
        gradient = torch.randn(source.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.01
        parallel_state.initialize_model_parallel(expert_model_parallel_size=1)
        model_parallel_cuda_manual_seed(SEED)
        model = _make_model(1, "alltoall", False, skew)
        counts = _check_frozen_router(model, source, records)
        value = source.detach().clone().requires_grad_()
        expected, expected_gradients = _backward(model, value, gradient)
        expected_input_gradient = value.grad.detach().clone()
        weights = {
            name: parameter.detach().clone().requires_grad_(parameter.requires_grad)
            for name, parameter in model.named_parameters()
        }
        oracle_input = source.detach().clone().requires_grad_()
        oracle_output = _oracle(oracle_input, weights, model.router.expert_bias)
        oracle_output.backward(gradient)
        _compare(expected, oracle_output, "formula/output", records, 0.02)
        _compare(expected_input_gradient, oracle_input.grad, "formula/input_grad", records, 0.02)
        for name, actual in expected_gradients.items():
            _compare(actual, weights[name].grad, f"formula/{name}", records, 0.02)
        assert sum(counts) == TOKENS * TOPK
        if skew:
            assert counts[:TOPK] == [TOKENS] * TOPK and counts[TOPK:] == [0] * (EXPERTS - TOPK)
        del model, value, oracle_input, oracle_output, weights
        gc.collect()
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(expert_model_parallel_size=2)
        model_parallel_cuda_manual_seed(SEED)
        local = slice(rank * (TOKENS // 2), (rank + 1) * (TOKENS // 2))
        alltoall_output = alltoall_input_grad = alltoall_gradients = None
        for dispatcher in ("alltoall", "flex"):
            model = _make_model(2, dispatcher, recompute, skew)
            if dispatcher == "flex":
                group = model.tp_ep_group
                nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(HIDDEN * 2, group.size())
                    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size()))
                )
                buffer = Buffer(group, nvl_bytes, 0, explicitly_destroy=True)
                fused_a2a._buffer = buffer
            value = source[local].detach().clone().requires_grad_()
            actual, gradients = _backward(model, value, gradient[local])
            for name, actual_gradient in gradients.items():
                if not name.startswith("experts."):
                    dist.all_reduce(actual_gradient, group=model.ep_group)
                _compare(actual_gradient, expected_gradients[name], f"{dispatcher}/{name}", records, 0.02)
            _compare(actual, expected[local], f"{dispatcher}/output", records, 0.02)
            _compare(value.grad, expected_input_gradient[local], f"{dispatcher}/input_grad", records, 0.02)
            if dispatcher == "alltoall":
                alltoall_output = actual
                alltoall_input_grad = value.grad.detach().clone()
                alltoall_gradients = gradients
            else:
                _compare(actual, alltoall_output, "deepep_vs_alltoall/output", records, 0.01)
                _compare(value.grad, alltoall_input_grad, "deepep_vs_alltoall/input_grad", records, 0.01)
                for name, actual_gradient in gradients.items():
                    _compare(actual_gradient, alltoall_gradients[name], f"deepep_vs_alltoall/{name}", records, 0.01)
            del model, value
            torch.cuda.synchronize()
        Path(f"{output_path}.rank{rank}.json").write_text(
            json.dumps(
                {
                    "seed": SEED,
                    "dtype": "bfloat16",
                    "hidden": HIDDEN,
                    "intermediate": INTERMEDIATE,
                    "tokens": TOKENS,
                    "experts": EXPERTS,
                    "topk": TOPK,
                    "clamp": CLAMP,
                    "scaling": SCALE,
                    "skew": skew,
                    "selective_moe_recompute": recompute,
                    "router_frozen": True,
                    "tokens_per_expert": counts,
                    "checks": records,
                },
                indent=2,
            )
        )
    finally:
        torch.cuda.synchronize()
        if buffer is not None:
            fused_a2a._buffer = None
            buffer.destroy()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "skew,recompute",
    [(False, False), (True, False), (True, True)],
    ids=["random-routing", "empty-experts", "empty-experts-moe-recompute"],
)
def test_v41_moe_ep_and_deepep_match_formula(tmp_path: Path, skew: bool, recompute: bool) -> None:
    output_dir = Path(os.environ.get("RELAX_TEST_V41_OUTPUT", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"moe-skew{int(skew)}-recompute{int(recompute)}"
    torch.multiprocessing.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(output), skew, recompute),
        nprocs=2,
        join=True,
    )
