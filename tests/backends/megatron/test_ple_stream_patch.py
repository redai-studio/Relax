# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@pytest.fixture(autouse=True)
def _require_qwen38_mcore():
    hc = pytest.importorskip("megatron.core.transformer.hyper_connection")
    pytest.importorskip("megatron.core.transformer.per_layer_embedding")
    if not hasattr(hc, "gated_residual_group_rmsnorm"):
        pytest.skip("Requires the image-built Qwen38 MCore runtime")


def _reference_ple(ple, hidden, key_raw, value, positions):
    from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm

    sequence_length, batch_size, _ = hidden.shape
    streams, channels = ple.n, ple.hidden_size
    key = gated_residual_group_rmsnorm(key_raw, ple.norm_key.weight, streams, ple.norm_eps)
    query = gated_residual_group_rmsnorm(hidden, ple.norm_query.weight, streams, ple.norm_eps)
    gate = (
        key.float().view(sequence_length, batch_size, streams, channels)
        * query.float().view(sequence_length, batch_size, streams, channels)
    ).sum(dim=-1, keepdim=True) / math.sqrt(channels)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = (torch.sigmoid(gate) * value.float().unsqueeze(-2)).to(hidden.dtype)
    gated_value = gated_value.view(sequence_length, batch_size, streams * channels)
    normed = gated_residual_group_rmsnorm(gated_value, ple.norm_conv.weight, streams, ple.norm_eps)
    padded = F.pad(normed, (0, 0, 0, 0, ple.halo, 0))
    weight = ple.conv1d.weight.squeeze(1).to(hidden.dtype)
    conv = torch.zeros_like(normed, dtype=torch.float32)
    for tap in range(ple.conv_kernel_size):
        offset = (ple.conv_kernel_size - 1 - tap) * ple.conv_dilation
        source = padded[ple.halo - offset : ple.halo - offset + sequence_length]
        valid = (positions >= offset).unsqueeze(-1)
        conv = conv + torch.where(valid, source.float(), 0.0) * weight[:, tap].float()
    return gated_value + F.silu(conv.to(hidden.dtype).float()).to(hidden.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kernel,dilation", [(3, 2), (4, 3)])
def test_ple_stream_patch_matches_reference_forward_and_hidden_gradient(dtype, kernel, dilation):
    from relax.models.qwen3_8.ops.ple_stream import _stream_forward

    torch.manual_seed(7)
    streams, channels, sequence_length, batch_size = 4, 8, 19, 2
    ple = nn.Module()
    ple.n = streams
    ple.hidden_size = channels
    ple.norm_eps = 1e-6
    ple.conv_kernel_size = kernel
    ple.conv_dilation = dilation
    ple.halo = (kernel - 1) * dilation
    ple.norm_key = nn.Module()
    ple.norm_query = nn.Module()
    ple.norm_conv = nn.Module()
    for name in ("norm_key", "norm_query", "norm_conv"):
        getattr(ple, name).weight = nn.Parameter(torch.randn(streams * channels).to(dtype), requires_grad=False)
    ple.conv1d = nn.Module()
    ple.conv1d.weight = nn.Parameter(
        torch.randn(streams * channels, 1, ple.conv_kernel_size).to(dtype), requires_grad=False
    )
    hidden = torch.randn(sequence_length, batch_size, streams * channels, dtype=dtype)
    key_raw = torch.randn_like(hidden)
    value = torch.randn(sequence_length, batch_size, channels, dtype=dtype)
    positions = torch.arange(sequence_length).view(-1, 1).expand(-1, batch_size).clone()
    positions[7:, 1] -= 7

    reference_hidden = hidden.detach().requires_grad_(True)
    reference = _reference_ple(ple, reference_hidden, key_raw, value, positions)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    streamed_hidden = hidden.detach().requires_grad_(True)
    pieces = [
        checkpoint(
            lambda h, k, v, p, stream=stream: _stream_forward(ple, h, k, v, p, stream),
            streamed_hidden[..., stream * channels : (stream + 1) * channels],
            key_raw[..., stream * channels : (stream + 1) * channels],
            value,
            positions,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        for stream in range(streams)
    ]
    streamed = torch.cat(pieces, dim=-1)
    streamed.backward(upstream)
    tolerance = 1e-6 if dtype == torch.float32 else 6e-3
    torch.testing.assert_close(streamed, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(streamed_hidden.grad, reference_hidden.grad, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("outer_reentrant", [False, True])
def test_ple_stream_patch_full_forward_inside_outer_checkpoint(monkeypatch, outer_reentrant):
    from megatron.core.ssm import mamba_context_parallel

    from relax.models.qwen3_8.ops import ple_stream as ple_stream_patch

    torch.manual_seed(11)
    streams, channels, sequence_length, batch_size = 4, 8, 13, 1
    embedding_dim = 12
    ple = nn.Module()
    ple.n = streams
    ple.hidden_size = channels
    ple.norm_eps = 1e-6
    ple.conv_kernel_size = 4
    ple.conv_dilation = 3
    ple.halo = 9
    ple.config = type("Config", (), {"sequence_parallel": False})()
    ple.tp_group = type("TPGroup", (), {"size": lambda self: 1})()
    ple._ngram_ids = torch.zeros(1, sequence_length, 1, dtype=torch.long)
    ple._cp_packed_seq_params = object()
    ple.norm_key = nn.Module()
    ple.norm_query = nn.Module()
    ple.norm_conv = nn.Module()
    for name in ("norm_key", "norm_query", "norm_conv"):
        getattr(ple, name).weight = nn.Parameter(torch.randn(streams * channels), requires_grad=False)
    ple.conv1d = nn.Module()
    ple.conv1d.weight = nn.Parameter(torch.randn(streams * channels, 1, ple.conv_kernel_size), requires_grad=False)
    ple.key_proj = nn.Linear(embedding_dim, streams * channels, bias=False)
    ple.value_proj = nn.Linear(embedding_dim, channels, bias=False)
    ple.key_proj.requires_grad_(False)
    ple.value_proj.requires_grad_(False)
    embeddings = torch.randn(sequence_length, batch_size, embedding_dim)
    ple.ple_embedding = nn.Module()
    ple.ple_embedding.forward = lambda _ids: embeddings
    positions = torch.arange(sequence_length).view(-1, 1)
    ple._local_positions = lambda _length: positions
    monkeypatch.setattr(ple_stream_patch.parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mamba_context_parallel, "reconstruct_tensor_cp", lambda tensor, *_args, **_kwargs: tensor)
    monkeypatch.setattr(mamba_context_parallel, "split_tensor_cp", lambda tensor, *_args, **_kwargs: tensor)

    hidden = torch.randn(sequence_length, batch_size, streams * channels)
    reference_hidden = hidden.detach().requires_grad_(True)
    reference = _reference_ple(ple, reference_hidden, ple.key_proj(embeddings), ple.value_proj(embeddings), positions)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    streamed_hidden = hidden.detach().requires_grad_(True)
    streamed = checkpoint(
        lambda value: ple_stream_patch._streamed_forward(ple, value),
        streamed_hidden,
        use_reentrant=outer_reentrant,
    )
    streamed.backward(upstream)
    torch.testing.assert_close(streamed, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(streamed_hidden.grad, reference_hidden.grad, rtol=1e-6, atol=1e-6)
