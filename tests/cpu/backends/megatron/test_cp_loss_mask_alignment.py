# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron import cp_utils
from relax.utils.sft_utils import align_loss_mask_for_sft


@pytest.mark.parametrize("layout", ["text", "vl_text", "media", "unsplit_text"])
def test_cp_output_gather_matches_nullable_multimodal_layout(monkeypatch, layout):
    from relax.backends.megatron import actor

    monkeypatch.setattr(actor.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(actor.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    padded_lengths = []

    def gather(value, total_length, response_length, *, padded_total_length, **kwargs):
        padded_lengths.append(padded_total_length)
        return value

    monkeypatch.setattr(actor, "all_gather_with_cp", gather)
    instance = SimpleNamespace(
        args=SimpleNamespace(
            qkv_format="thd", is_vl_model=layout == "vl_text", uses_unsplit_forward=layout == "unsplit_text"
        )
    )
    rollout_data = {
        "total_lengths": [5, 9],
        "response_lengths": [2, 2],
        "multimodal_train_inputs": [None, {"pixel_values": torch.ones(4, 8)} if layout == "media" else None],
    }
    output = {"log_probs": [torch.zeros(2), torch.zeros(2)]}
    actual = actor.MegatronTrainRayActor._gather_cp_output_for_transfer_queue(instance, output, rollout_data)
    assert padded_lengths == ([None, None] if layout == "text" else [8, 16])
    assert actual["log_probs"] == output["log_probs"]


def _cp_kwargs(monkeypatch, cp_rank: int, dynamic_cp: bool) -> dict[str, int]:
    if dynamic_cp:
        return {"dynamic_cp_size": 2, "dynamic_cp_rank": cp_rank}
    monkeypatch.setattr(
        cp_utils,
        "mpu",
        SimpleNamespace(get_context_parallel_world_size=lambda: 2, get_context_parallel_rank=lambda: cp_rank),
    )
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
