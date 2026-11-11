# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


_BOUNDARIES = (0, 8, 20, 28, 32)
_REAL_LENGTHS = (5, 10, 7)


def _local_indices(size: int, rank: int) -> torch.Tensor:
    """Independent document-wise zigzag layout, including the padding-only
    tail."""
    indices = []
    for start, end in zip(_BOUNDARIES, _BOUNDARIES[1:]):
        if size == 1:
            indices.extend(range(start, end))
        else:
            width = (end - start) // (2 * size)
            for chunk in (rank, 2 * size - rank - 1):
                indices.extend(range(start + chunk * width, start + (chunk + 1) * width))
    return torch.tensor(indices, device="cuda")


def _unpadded_reference(inputs: list[torch.Tensor]) -> torch.Tensor:
    """Separate ordinary causal attention calls contain only real document
    tokens."""
    q, k, v = inputs
    output = q.new_zeros((q.shape[0], q.shape[1] * q.shape[2]))
    for start, length in zip(_BOUNDARIES, _REAL_LENGTHS):
        end = start + length
        result = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1)[None],
            k[start:end].repeat_interleave(8, 1).transpose(0, 1)[None],
            v[start:end].repeat_interleave(8, 1).transpose(0, 1)[None],
            is_causal=True,
        )[0].transpose(0, 1)
        output[start:end] = result.flatten(1)
    return output


def _check_packed_padding(size: int, rank: int, group: object | None, explicit_override: bool) -> None:
    from relax.models.keye_vl2.attention import KeyeCoreAttention

    torch.manual_seed(385)
    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=size,
        hidden_size=2048,
        layernorm_epsilon=1e-6,
        rotary_base=1e7,
        keye_indexer_loss_coeff=0.0,
        keye_attention_chunk_size=3,
    )
    attention = KeyeCoreAttention(config, 1, pg_collection=SimpleNamespace(cp=group)).cuda().bfloat16()
    attention.sa_indexer.requires_grad_(False)
    full = [torch.randn(32, heads, 128, device="cuda", dtype=torch.bfloat16) for heads in (32, 4, 4)]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in full]
    indices = _local_indices(size, rank)
    local = [tensor.index_select(0, indices).clone().requires_grad_() for tensor in full]
    hidden = torch.randn(32, 1, 2048, device="cuda", dtype=torch.bfloat16).index_select(0, indices)
    valid = torch.zeros(32, device="cuda", dtype=torch.bool)
    for start, length in zip(_BOUNDARIES, _REAL_LENGTHS):
        valid[start : start + length] = True
    packed = SimpleNamespace(
        cu_seqlens_q_cpu=list(_BOUNDARIES),
        # This global count alone incorrectly masks the last document's final five tokens.
        num_valid_tokens=sum(_REAL_LENGTHS),
        valid_seqlens_q_cpu=(0, 0, 0) if explicit_override else _REAL_LENGTHS,
    )
    if explicit_override:
        packed.keye_valid_mask = valid
    actual = attention(*local, hidden_states=hidden, packed_seq_params=packed)
    expected = _unpadded_reference(reference_inputs)
    local_valid = valid.index_select(0, indices)
    torch.testing.assert_close(
        actual[local_valid], expected.index_select(0, indices)[local_valid], atol=0.02, rtol=0.02
    )
    # The extra physical document is entirely padding and has no attention support.
    padding_tail = indices >= _BOUNDARIES[-2]
    torch.testing.assert_close(actual[padding_tail], torch.zeros_like(actual[padding_tail]), atol=0, rtol=0)
    upstream = torch.randn_like(expected) / 16
    upstream[~valid] = 0
    upstream[_BOUNDARIES[-2] :] = 1  # Even nonzero dY on the padding-only tail must produce no gradient.
    (actual * upstream.index_select(0, indices)).float().sum().backward()
    (expected * upstream).float().sum().backward()
    for tensor, reference in zip(local, reference_inputs):
        target = reference.grad.index_select(0, indices)
        torch.testing.assert_close(tensor.grad, target, atol=0.002, rtol=0.03)
        torch.testing.assert_close(
            tensor.grad[~local_valid], torch.zeros_like(tensor.grad[~local_valid]), atol=0, rtol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Packed Keye attention needs CUDA")
@pytest.mark.parametrize("explicit_override", [False, True])
def test_keye_packed_real_tokens_match_unpadded_documents(explicit_override: bool) -> None:
    _check_packed_padding(1, 0, None, explicit_override)


def _cp_worker(rank: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=init_method)
    try:
        _check_packed_padding(2, rank, dist.group.WORLD, False)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Packed Keye CP2 regression needs two CUDA devices")
def test_keye_cp2_padded_documents_match_unpadded_forward_and_gradients(tmp_path) -> None:
    mp.spawn(_cp_worker, args=(f"file://{tmp_path / 'packed-cp-init'}",), nprocs=2, join=True)
