# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


pytest.importorskip("megatron.core", exc_type=ImportError)

from relax.backends.megatron import cp_utils
from relax.utils.sft_utils import align_loss_mask_for_sft


def _cp_kwargs(monkeypatch, cp_rank: int, dynamic_cp: bool) -> dict[str, int]:
    if dynamic_cp:
        return {"dynamic_cp_size": 2, "dynamic_cp_rank": cp_rank}
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_rank", lambda: cp_rank)
    return {}


@pytest.mark.parametrize(
    ("qkv_format", "max_seq_lens", "padded_total_lengths"),
    [("thd", None, None), ("thd", None, [12]), ("bshd", [8], None)],
)
@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_sft_cp_reducer_uses_predictor_mask_and_keeps_span_end(
    monkeypatch,
    qkv_format: str,
    max_seq_lens: list[int] | None,
    padded_total_lengths: list[int] | None,
    dynamic_cp: bool,
) -> None:
    total_length = 8
    raw_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    aligned_mask = align_loss_mask_for_sft(raw_mask)
    # The allgather-redistributed SFT vector is target-coordinate: index 0 is
    # the dummy pre-sequence slot, and indices 1..7 are log-probs for targets
    # 1..7. A wrong target-coordinate mask selects target 2 and drops target 7.
    full_log_probs = torch.zeros(total_length)
    full_log_probs[2] = 100.0
    full_log_probs[7] = 1.0

    reduced_sum = torch.tensor(0.0)
    reduced_mean = torch.tensor(0.0)
    token_count = torch.tensor(0)
    for cp_rank in range(2):
        cp_kwargs = _cp_kwargs(monkeypatch, cp_rank, dynamic_cp)
        local_log_probs = cp_utils.slice_log_prob_with_cp(
            full_log_probs,
            total_length,
            total_length,
            qkv_format=qkv_format,
            max_token_len=max_seq_lens[0] if max_seq_lens is not None else None,
            padded_total_length=padded_total_lengths[0] if padded_total_lengths is not None else None,
            **cp_kwargs,
        )
        sum_reducer = cp_utils.get_sum_of_sample_mean(
            [total_length],
            [total_length],
            [aligned_mask],
            calculate_per_token_loss=True,
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
            **cp_kwargs,
        )
        mean_reducer = cp_utils.get_sum_of_sample_mean(
            [total_length],
            [total_length],
            [aligned_mask],
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
            **cp_kwargs,
        )
        reduced_sum += sum_reducer(local_log_probs)
        reduced_mean += mean_reducer(local_log_probs)
        token_count += cp_utils.get_cp_local_num_tokens(
            [total_length],
            [total_length],
            [aligned_mask],
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
            **cp_kwargs,
        )

    assert reduced_sum.item() == 1.0
    assert reduced_mean.item() == pytest.approx(1.0 / 5.0)
    assert token_count.item() == 5


@pytest.mark.parametrize(("qkv_format", "max_seq_lens"), [("thd", None), ("bshd", [8])])
@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_rl_cp_reducer_keeps_target_coordinate_mask(
    monkeypatch,
    qkv_format: str,
    max_seq_lens: list[int] | None,
    dynamic_cp: bool,
) -> None:
    total_length = 8
    response_length = 4
    response_mask = torch.tensor([1, 0, 1, 1])
    # Response-vector entries correspond to global target positions 4..7.
    full_log_probs = torch.tensor([4.0, 5.0, 6.0, 7.0])

    reduced_sum = torch.tensor(0.0)
    token_count = torch.tensor(0)
    for cp_rank in range(2):
        cp_kwargs = _cp_kwargs(monkeypatch, cp_rank, dynamic_cp)
        local_log_probs = cp_utils.slice_log_prob_with_cp(
            full_log_probs,
            total_length,
            response_length,
            qkv_format=qkv_format,
            max_token_len=max_seq_lens[0] if max_seq_lens is not None else None,
            **cp_kwargs,
        )
        reducer = cp_utils.get_sum_of_sample_mean(
            [total_length],
            [response_length],
            [response_mask],
            calculate_per_token_loss=True,
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            **cp_kwargs,
        )
        reduced_sum += reducer(local_log_probs)
        token_count += cp_utils.get_cp_local_num_tokens(
            [total_length],
            [response_length],
            [response_mask],
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            **cp_kwargs,
        )

    assert reduced_sum.item() == 4.0 + 6.0 + 7.0
    assert token_count.item() == 3
