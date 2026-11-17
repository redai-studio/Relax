# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU regression for Qwen3.8's non-aligned GDN shards and MoE FP8 boundary."""

from argparse import Namespace

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an FP8-capable CUDA GPU")


@pytest.fixture
def runtime(tmp_path):
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    if not te.is_mxfp8_available():
        pytest.skip("MXFP8 requires a supported Blackwell device")
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2560,
        num_attention_heads=20,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
    )
    yield config
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    FP8GlobalStateManager.reset()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _args(primary_fp8=False) -> Namespace:
    return Namespace(
        fp8="e4m3",
        fp8_recipe="mxfp8",
        bf16=True,
        fp8_param_gather=primary_fp8,
        use_distributed_optimizer=True,
        reuse_grad_buf_for_mxfp8_param_ag=primary_fp8,
    )


def test_qwen38_fp8_gdn_tp4_shard_preserves_forward_and_backward(runtime):
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.fp8_utils import get_fp8_context

    from relax.models.qwen3_8.precision import configure_fp8

    model = torch.nn.Module()
    model.in_proj = TELinear(
        2560,
        4120,  # Real Qwen3.8 GDN local output width with TP4.
        parallel_mode="column",
        config=runtime,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=torch.distributed.group.WORLD,
    )
    original = dict(model.named_parameters())
    plan = configure_fp8(model, _args())
    assert plan == {"in_proj": "bf16"}
    assert not model.in_proj.will_execute_quantized(True)
    assert all(parameter is original[name] for name, parameter in model.named_parameters())
    x = torch.randn(32, 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference, _ = model.in_proj(x)
    grad = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, (x, model.in_proj.weight), grad)
    with get_fp8_context(runtime):
        actual, _ = model.in_proj(x)
    received = torch.autograd.grad(actual, (x, model.in_proj.weight), grad)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for got, want in zip(received, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("splits", [(32, 64), (0, 64)])
@pytest.mark.parametrize("primary_fp8", [False, True])
def test_qwen38_fp8_expert_gradients_with_empty_and_unequal_splits(runtime, splits, primary_fp8):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    from megatron.core.fp8_utils import get_fp8_context

    from relax.models.qwen3_8.precision import configure_fp8

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    layer = TEGroupedLinear(
        2,
        2560,
        1280,
        parallel_mode="column",
        config=runtime,
        init_method=lambda weight: torch.nn.init.normal_(weight, std=0.02),
        bias=False,
        skip_bias_add=False,
        is_expert=True,
    )
    model.mlp.experts.linear_fc1 = layer
    plan = configure_fp8(model, _args(primary_fp8))
    assert plan == {"mlp.experts.linear_fc1": "mxfp8"}
    assert layer.will_execute_quantized(True)
    x = torch.randn(sum(splits), 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = tuple(layer.parameters())
    reference, _ = layer(x, list(splits))
    grad = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, (x, *weights), grad)
    layer.is_first_microbatch = True
    with get_fp8_context(runtime):
        actual, _ = layer(x, list(splits))
    received = torch.autograd.grad(actual, (x, *weights), grad)
    for got, want in zip((actual, *received), (reference, *expected), strict=True):
        assert torch.isfinite(got).all()
        relative_error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-6)
        assert relative_error < 0.15
    assert all(weight.dtype == torch.bfloat16 and weight.requires_grad for weight in weights)


def test_qwen38_fp8_primary_keeps_module_boundary_and_parallel_metadata(runtime):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.fp8_utils import get_fp8_context
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    from relax.models.qwen3_8.precision import configure_fp8

    model = torch.nn.Module()
    for branch in ("decoder", "mtp", "vision_model"):
        block = torch.nn.Module()
        block.shared_experts = torch.nn.Module()
        block.shared_experts.linear_fc1 = TEColumnParallelLinear(
            256,
            128,
            config=runtime,
            init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=torch.distributed.group.WORLD,
        )
        model.add_module(branch, block)
    original = dict(model.named_parameters())
    target = model.decoder.shared_experts.linear_fc1
    target.weight.allreduce = False
    target.weight.partition_dim = 0
    configure_fp8(model, _args(True))
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            assert isinstance(parameter, MXFP8Tensor)
            assert parameter._rowwise_data.dtype == torch.uint8
            assert parameter._columnwise_data.dtype == torch.uint8
            assert parameter is not original[name]
            assert parameter.allreduce is False
            assert parameter.partition_dim == 0
        else:
            assert parameter is original[name]
            assert not isinstance(parameter, MXFP8Tensor)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with get_fp8_context(runtime):
        outputs = [block.shared_experts.linear_fc1(x)[0] for block in model.children()]
    sum(y.float().square().mean() for y in outputs).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("branch", ["decoder", "mtp"])
def test_qwen38_fp8_grouped_checkpoint_roundtrip(runtime, tmp_path, branch):
    from megatron.core import dist_checkpointing
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
    from megatron.core.fp8_utils import get_fp8_context

    from relax.models.qwen3_8.precision import configure_fp8

    def build():
        model = torch.nn.Module()
        block = torch.nn.Module()
        model.add_module(branch, block)
        block.experts = torch.nn.Module()
        layer = TEColumnParallelGroupedLinear(
            2,
            256,
            128,
            config=runtime,
            init_method=lambda weight: torch.nn.init.normal_(weight, std=0.02),
            bias=False,
            skip_bias_add=False,
            is_expert=True,
        )
        block.experts.linear_fc1 = layer
        configure_fp8(model, _args())
        return layer

    layer = build()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with get_fp8_context(runtime):
        reference, _ = layer(x, [32, 32])
    reference.float().square().mean().backward()
    assert layer.fp8 == (branch == "decoder")
    metadata = {"dp_cp_group": torch.distributed.group.WORLD}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dist_checkpointing.save(layer.sharded_state_dict(metadata=metadata), str(checkpoint))

    restored = build()
    restored.load_state_dict(
        dist_checkpointing.load(restored.sharded_state_dict(metadata=metadata), str(checkpoint)), strict=True
    )
    for got, want in zip(restored.parameters(), layer.parameters(), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    with get_fp8_context(runtime):
        actual, _ = restored(x, [32, 32])
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in restored.parameters()
    )
