# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real MCore head checks; runnable with pytest or two-rank torchrun."""

import io
import os
import tempfile

import pytest
import torch
import torch.distributed as dist


pytest.importorskip("megatron.core")
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

from relax.backends.megatron.fp32_head import configure_fp32_lm_head


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 GEMM")


@pytest.fixture(scope="module", autouse=True)
def groups():
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
    else:
        with tempfile.NamedTemporaryFile(delete=False) as file:
            path = file.name
        dist.init_process_group("nccl", init_method=f"file://{path}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(123)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def make_head(sequence_parallel=False, fusion=False, tied=False):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=dist.get_world_size(),
        gradient_accumulation_fusion=fusion,
    )
    head = ColumnParallelLinear(
        32,
        64,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_weight_param_allocation=tied,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    model = torch.nn.Module()
    model.output_layer = head
    return model, head


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_fp32_head_preserves_logits_and_parallel_gradients(sequence_parallel, frozen):
    if sequence_parallel and dist.get_world_size() == 1:
        pytest.skip("sequence parallel validation requires two local GPUs")
    torch.manual_seed(123)
    model, head = make_head(sequence_parallel=sequence_parallel)
    x = torch.randn(8, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = head.weight.detach() if frozen else head.weight
    before, _ = head(x, weight=weight)
    upstream = torch.randn_like(before, dtype=torch.float32)
    before.backward(upstream.to(before.dtype))
    expected_dx = x.grad.clone()
    expected_dw = None if frozen else head.weight.grad.clone()
    x.grad = None
    head.weight.grad = None
    parameter = head.weight
    configure_fp32_lm_head(model)
    configure_fp32_lm_head(model)
    # Explicit class.forward is also used by the chunked MTP implementation.
    output, _ = type(head).forward(head, x, weight=weight)
    assert head.weight is parameter
    assert output.dtype == torch.float32
    assert output._base is None
    assert (output != output.bfloat16().float()).any()
    output.backward(upstream)
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
    if not frozen:
        torch.testing.assert_close(head.weight.grad, expected_dw, rtol=0, atol=0)
    assert not torch.equal(output.detach(), before.float())
    if not sequence_parallel:
        reference = x.detach().double().reshape(-1, 32) @ weight.detach().double().T
        error = (output.detach().reshape_as(reference).double() - reference).abs().mean()
        old_error = (before.detach().reshape_as(reference).double() - reference).abs().mean()
        assert error < old_error


@pytest.mark.parametrize("fusion", [False, True])
def test_fp32_head_accumulates_shared_chunks_and_restores(fusion):
    torch.manual_seed(19)
    model, head = make_head(fusion=fusion, tied=True)
    # A tied embedding weight is supplied explicitly on every call.
    weight = torch.nn.Parameter(torch.randn(64 // dist.get_world_size(), 32, device="cuda", dtype=torch.bfloat16))
    if fusion:
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
    xs = [torch.randn(rows, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True) for rows in (8, 16, 8)]
    upstream = [torch.randn(len(x), 1, weight.shape[0], device="cuda", dtype=torch.float32) for x in xs]
    for x, grad in zip(xs, upstream):
        old, _ = head(x, weight=weight)
        old.backward(grad.bfloat16())
    reference = (weight.main_grad if fusion else weight.grad).clone()
    expected_dx = [x.grad.clone() for x in xs]
    weight.grad = None
    if fusion:
        weight.main_grad.zero_()
        weight.grad_added_to_main_grad = False
    for x in xs:
        x.grad = None
    configure_fp32_lm_head(model)
    losses = []
    for x, grad in zip(xs, upstream):
        output, _ = head(x, weight=weight)
        losses.append((output * grad).sum())
    sum(losses).backward()
    torch.testing.assert_close(weight.main_grad if fusion else weight.grad, reference, rtol=0.02, atol=0.125)
    for x, expected in zip(xs, expected_dx):
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    if fusion:
        assert weight.grad_added_to_main_grad
    # Parameter-only state dicts retain their keys and values.
    state = model.state_dict()
    model.load_state_dict(state, strict=True)


def test_fp32_head_allows_native_cross_entropy_inplace():
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    model, head = make_head()
    configure_fp32_lm_head(model)
    x = torch.randn(8, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output, _ = head(x)
    labels = torch.arange(8, device="cuda").reshape(8, 1)
    loss = vocab_parallel_cross_entropy(output, labels, tp_group=head.tp_group).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_fp32_head_restores_optimizer_and_next_shared_head_update():
    model, head = make_head()
    configure_fp32_lm_head(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1, 32, device="cuda", dtype=torch.bfloat16)

    def step(module, opt):
        opt.zero_grad()
        first, _ = module.output_layer(x)
        second, _ = type(module.output_layer).forward(module.output_layer, x * 0.5)
        (first.square().mean() + second.square().mean()).backward()
        opt.step()
        return first.detach(), second.detach()

    step(model, optimizer)
    data = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, data)
    expected = step(model, optimizer)
    restored, _ = make_head()
    configure_fp32_lm_head(restored)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    data.seek(0)
    state = torch.load(data, weights_only=False)
    restored.load_state_dict(state["model"], strict=True)
    restored_optimizer.load_state_dict(state["optimizer"])
    actual = step(restored, restored_optimizer)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(restored.output_layer.weight, head.weight, rtol=0, atol=0)
