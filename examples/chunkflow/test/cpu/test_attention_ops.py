# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU native-executor fixtures and runtime lookup regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.chunkflow.adapters import DenseAttentionCall
from examples.chunkflow.runtime import ChunkflowRuntime, require_runtime


class SDPAttention(torch.nn.Module):
    """Small native PyTorch module used to test connection-layer behavior."""

    def __init__(self, gain: bool = False):
        super().__init__()
        if gain:
            self.gain = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, query, key, value, *, q_lengths, kv_lengths, scale):
        outputs = []
        q_start = k_start = 0
        repeat = query.shape[1] // key.shape[1]
        for q_length, kv_length in zip(q_lengths, kv_lengths):
            q = query[q_start : q_start + q_length]
            k = key[k_start : k_start + kv_length].repeat_interleave(repeat, dim=1)
            v = value[k_start : k_start + kv_length].repeat_interleave(repeat, dim=1)
            q_pos = torch.arange(q_length, device=q.device) + kv_length - q_length
            k_pos = torch.arange(kv_length, device=q.device)
            mask = k_pos[None, :] <= q_pos[:, None]
            output = F.scaled_dot_product_attention(
                q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2), attn_mask=mask, scale=scale
            ).permute(1, 0, 2)
            outputs.append(output * self.gain if hasattr(self, "gain") else output)
            q_start += q_length
            k_start += kv_length
        return torch.cat(outputs)


def native_sdpa(original, call: DenseAttentionCall):
    key, value = call.joined_kv()
    return original(
        call.query,
        key,
        value,
        q_lengths=call.q_lengths,
        kv_lengths=call.kv_lengths,
        scale=call.softmax_scale,
    )


def test_schedule_reuses_runtime_installed_below_model_wrappers():
    class Wrapper:
        def __init__(self, module):
            self.module = module

    bare = Wrapper(None)
    runtime = ChunkflowRuntime(chunk_size=16)
    bare.chunkflow_runtime = runtime
    wrapped = Wrapper(Wrapper(bare))
    assert require_runtime(wrapped) is runtime
    assert not hasattr(wrapped, "chunkflow_runtime")
