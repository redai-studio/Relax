# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check GPU imports, optionally running small operators on the current GPU."""

import itertools
import os
import subprocess
import sys
from argparse import ArgumentParser
from importlib import import_module
from importlib import metadata as md
from pathlib import Path
from unittest.mock import patch


def check_te_attention() -> None:
    """Check the training FlashAttention path for Qwen3.5 vision and text."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformer_engine.pytorch import DotProductAttention

    torch.manual_seed(42)
    lengths = (65, 127)
    total = sum(lengths)
    cu_seqlens = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    # Keep NVTE_FLASH_ATTN_V4 from the image; select FlashAttention as training does.
    with patch.dict(os.environ, NVTE_FLASH_ATTN="1", NVTE_FUSED_ATTN="0", NVTE_UNFUSED_ATTN="0"):
        for head_dim, heads, kv_heads, causal in ((72, 16, 16, False), (256, 8, 1, True)):
            q, k, v = [
                (torch.randn(total, n, head_dim, device="cuda", dtype=torch.bfloat16) * 0.25).requires_grad_()
                for n in (heads, kv_heads, kv_heads)
            ]
            reference_qkv = [x.detach().float().requires_grad_() for x in (q, k, v)]
            reference_parts = []
            with sdpa_kernel(SDPBackend.MATH):
                for start, length in ((0, lengths[0]), (lengths[0], lengths[1])):
                    rq, rk, rv = [
                        x[start : start + length].repeat_interleave(heads // x.shape[1], dim=1).transpose(0, 1)
                        for x in reference_qkv
                    ]
                    reference_parts.append(
                        F.scaled_dot_product_attention(rq, rk, rv, is_causal=causal).transpose(0, 1)
                    )
            reference = torch.cat(reference_parts)
            attention = DotProductAttention(
                heads,
                head_dim,
                num_gqa_groups=kv_heads,
                attention_dropout=0,
                qkv_format="thd",
                attn_mask_type="padding_causal" if causal else "padding",
                layer_number=1,
            )
            output = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(lengths),
                max_seqlen_kv=max(lengths),
                pad_between_seqs=False,
            ).view(total, heads, head_dim)
            grad = torch.randn_like(output) * 0.25
            output.backward(grad)
            reference.backward(grad.float())
            errors = []
            for actual, expected in zip(
                (output, q.grad, k.grad, v.grad),
                (reference, *(x.grad for x in reference_qkv)),
                strict=True,
            ):
                error = (
                    torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
                ).item()
                assert error < 0.02, (head_dim, error)
                errors.append(error)
            sys.stdout.write(
                f"TE FlashAttention head_dim={head_dim} forward/backward OK: max relative L2={max(errors):.4%}\n"
            )


def check_operators() -> None:
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    from transformer_engine.pytorch import Linear

    is_hopper = torch.cuda.get_device_capability() == (9, 0)
    # Check both ordinary heads and the large-head FA2 kernels used by MLA/VLMs.
    for head_dim in (64, 256):
        qkv = torch.randn(3, 2, 16, 4, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference_qkv = qkv.detach().float().requires_grad_()
        reference = F.scaled_dot_product_attention(
            *(x.transpose(1, 2) for x in reference_qkv.unbind()), is_causal=True
        ).transpose(1, 2)
        output = flash_attn_func(*qkv.unbind(), causal=True)
        torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.02)
        output.float().square().sum().backward()
        reference.square().sum().backward()
        torch.testing.assert_close(qkv.grad.float(), reference_qkv.grad, rtol=0.08, atol=0.08)

        if is_hopper:
            fa3_output = import_module("flash_attn_interface").flash_attn_func(*qkv.unbind(), causal=True)
            torch.testing.assert_close(fa3_output.float(), reference.detach(), rtol=0.03, atol=0.02)
            fa3_output.float().square().sum().backward()
            assert torch.isfinite(qkv.grad).all()

    linear = Linear(64, 64, params_dtype=torch.bfloat16, device="cuda")
    inputs = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = linear(inputs)
    reference = F.linear(inputs.float(), linear.weight.float(), linear.bias.float())
    torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.02)
    output.float().square().sum().backward()
    assert torch.isfinite(inputs.grad).all() and torch.isfinite(linear.weight.grad).all()

    quantize = import_module("fake_int4_quant_cuda").fake_int4_quant_cuda
    values = (torch.arange(128, device="cuda", dtype=torch.float32).reshape(4, 32) / 16 - 4).bfloat16()
    blocks = values.float()
    for symmetric in (True, False):
        if symmetric:
            scales = (blocks.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-5)
            zero = 0
        else:
            minimum = blocks.amin(dim=1, keepdim=True)
            scales = ((blocks.amax(dim=1, keepdim=True) - minimum) / 15).clamp_min(1e-5)
            zero = (-minimum / scales).round().clamp(0, 15)
        expected = ((blocks / scales).round() + zero).bfloat16()
        output, actual_scales, actual_zero = quantize(values, [1, 32], symmetric)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, scales.bfloat16(), rtol=0, atol=0)
        if not symmetric:
            torch.testing.assert_close(actual_zero, zero.bfloat16(), rtol=0, atol=0)

    weights = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    normalized = import_module("sgl_kernel").rmsnorm(inputs.detach(), weights, 1e-6)
    reference = inputs.detach().float() * torch.rsqrt(inputs.detach().float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized.float(), reference, rtol=0.03, atol=0.02)
    torch.cuda.synchronize()
    sys.stdout.write("GPU operators OK: FA2 forward/backward, TE Linear, INT4, SGLang RMSNorm\n")
    if is_hopper:
        sys.stdout.write("Hopper FA3 forward/backward OK\n")


def check_deepep_coexistence() -> None:
    # Fresh interpreters exercise both registration orders, not just cached modules.
    code = """
import importlib, sys, torch
assert torch.cuda.is_available()
for name in sys.argv[1:]:
    importlib.import_module(name)
modules = ['deep_ep._C', '_relax_hybrid_ep.deep_ep_cpp', '_relax_deep_ep_v2._C']
events = [importlib.import_module(name).EventHandle for name in modules]
assert len(set(events)) == 3
for event, module in zip(events, modules):
    assert event.__module__ == module
    event().current_stream_wait()
torch.cuda.synchronize()
"""
    for order in itertools.permutations(("deep_ep", "_relax_hybrid_ep", "_relax_deep_ep_v2")):
        subprocess.run([sys.executable, "-c", code, *order], check=True, timeout=180)


def main() -> None:
    import torch

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--operators", action="store_true", help="also check small forward/backward CUDA operators")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU/driver unavailable"
    import_module("transformer_engine.pytorch")
    assert torch.backends.cudnn.version() is not None, "cuDNN unavailable"
    import_module("sgl_kernel")
    if os.environ.get("RELAX_GPU_ARCH") in {"b300", "hopper"} or os.environ.get("RELAX_HYBRID_EP_COMMIT"):
        deep_ep = import_module("_relax_hybrid_ep")
        import_module("_relax_hybrid_ep.deep_ep_cpp")
        import_module("_relax_hybrid_ep.hybrid_ep_cpp")
        assert hasattr(deep_ep, "Buffer") and hasattr(deep_ep, "HybridEPBuffer")
    if os.environ.get("RELAX_GPU_ARCH") == "b300":
        from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

        assert callable(parallel_cross_entropy)
        if Path("/etc/relax-qwen35-runtime.json").is_file():
            check_deepep_coexistence()
            optimized_ep = import_module("_relax_deep_ep_v2")
            assert hasattr(optimized_ep, "ElasticBuffer")
            fa4 = import_module("_relax_fa4_pr2968.cute.interface")
            assert callable(fa4.flash_attn_varlen_func)
            flashqla = import_module("flash_qla")
            assert callable(flashqla.chunk_gated_delta_rule)
            expected_nccl = Path(
                md.distribution("nvidia-nccl-cu13").locate_file("nvidia/nccl/lib/libnccl.so.2")
            ).resolve()
            mapped_nccl = {
                Path(line.split()[-1]).resolve()
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libnccl.so" in line
            }
            assert mapped_nccl == {expected_nccl}, f"Unexpected NCCL libraries: {mapped_nccl}"
    sys.stdout.write(f"GPU imports OK: {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}\n")
    if args.operators:
        check_te_attention()
        check_operators()


if __name__ == "__main__":
    main()
