# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from relax.backends.fsdp_turbo.performance_ops.qsa_context_parallel import checkpoint_attention_chunk
from relax.backends.fsdp_turbo.qwen4_attention import attention_checkpoint_contexts


_calls = []


@torch.library.custom_op("relax_qsa_checkpoint_test::npu_fusion_attention", mutates_args=())
def attention(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # CPU stand-in only for the external NPU op, with the same saved-output contract.
    _calls.append(1)
    return value * mask


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[1], output)


def backward(ctx, grad):
    mask, output = ctx.saved_tensors
    return grad * mask + output * 0, None


attention.register_autograd(backward, setup_context=setup_context)


@pytest.mark.parametrize("skip,expected_calls", [(False, 6), (True, 3)])
def test_nested_checkpoint_preserves_attention_cache_and_releases_masks(skip, expected_calls):
    _calls.clear()
    masks, live_counts = [], []
    value = torch.arange(1.0, 13.0, requires_grad=True)

    def outer(x):
        outputs = []
        for start in range(0, 12, 4):

            def attend(part):
                mask = torch.ones_like(part) * 2
                masks.append(weakref.ref(mask))
                live_counts.append(sum(ref() is not None for ref in masks))
                return attention(part, mask)

            outputs.append(checkpoint_attention_chunk(attend, x[start : start + 4], skip_attention_recompute=skip))
        return torch.cat(outputs)

    result = checkpoint(outer, value, use_reentrant=False, context_fn=attention_checkpoint_contexts)
    result.sum().backward()
    torch.testing.assert_close(result, torch.arange(1.0, 13.0) * 2)
    torch.testing.assert_close(value.grad, torch.full_like(value, 2))
    assert len(_calls) == expected_calls
    assert max(live_counts) == 1
