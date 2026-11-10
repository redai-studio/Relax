# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.backends.megatron.selective_logits import prepare_selective_logits


def test_main_selection_preserves_weights_and_empty_rows():
    masks = [torch.tensor([0.0, 0.5, 0.0, 1.0]), torch.zeros(2)]
    batch = {"loss_masks": masks, "full_loss_masks": torch.cat(masks)[None]}
    prepare_selective_logits(batch, Namespace(enable_mtp_training=False))
    assert [n for n, _ in batch["selective_logit_rows"]] == [4, 2]
    assert batch["selective_logit_rows"][0][1].tolist() == [1, 3]
    assert batch["selective_logit_rows"][1][1].numel() == 0
    assert masks[0][1] == 0.5


def test_mtp_plan_respects_document_and_physical_padding():
    pytest.importorskip("megatron.core")
    from megatron.core.packed_seq_params import PackedSeqParams

    from relax.backends.megatron.data import move_tensors_to_device

    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 10], dtype=torch.int32),
    )
    batch = {
        "loss_masks": [torch.tensor([0.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 1.0, 1.0])],
        "full_loss_masks": torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
        "packed_seq_params": packed,
    }
    prepare_selective_logits(batch, Namespace(enable_mtp_training=True, mtp_num_layers=2))
    assert [x.tolist() for x in packed.relax_sft_selective_mtp_rows] == [[0, 1, 5, 6], [0, 4, 5]]
    mapped = move_tensors_to_device(batch, torch.device("cpu"))["packed_seq_params"]
    assert mapped is not packed
    assert mapped.relax_sft_selective_full_rows == 10
    assert [x.tolist() for x in mapped.relax_sft_selective_mtp_rows] == [[0, 1, 5, 6], [0, 4, 5]]


@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("case", ["packed", "response", "dummy", "repeated_dummy", "vl_text_only"])
def test_ordinary_pp4_iterator_preserves_selection_plans(monkeypatch, allgather_cp, case):
    """Exercise ingestion -> ordinary DataIterator -> get_batch without async
    prepack.

    Device transfer is mocked to CPU; rebuild the normal GPU-batch plan from
    its retained CPU masks and compare with the full packed-mask reference. The
    microbatch deliberately reorders samples and has odd document lengths/tail
    padding, which exposes off-by-one and global-roll mistakes.
    """
    from copy import copy

    from megatron.core.transformer.multi_token_prediction import roll_tensor

    from relax.backends.megatron import data
    from relax.backends.megatron.selective_logits import prepare_selective_logits_from_cpu_masks
    from relax.utils.data import stream_dataloader

    monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", "1")
    args = Namespace(
        loss_type="sft",
        qkv_format="thd",
        sft_chunked_logits=True,
        enable_mtp_training=True,
        mtp_num_layers=2,
        allgather_cp=allgather_cp,
        pipeline_model_parallel_size=4,
        is_vl_model=case == "vl_text_only",
        use_opd=False,
    )
    monkeypatch.setattr(data, "get_args", lambda: args)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(data.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(stream_dataloader.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    response_lengths = [3, 5] if case != "response" else [2, 3]
    masks = [[0, 1, 1], [1, 0, 1, 1, 1]] if case != "response" else [[1, 1], [0, 1, 1]]
    if case == "dummy":
        masks = [[0] * len(mask) for mask in masks]
    rollout = {
        "tokens": [[1, 2, 3], [4, 5, 6, 7, 8]],
        "loss_masks": masks,
        "total_lengths": [3, 5],
        "response_lengths": response_lengths,
    }
    stream_dataloader.post_process_rollout_data(args, rollout)
    assert "selective_loss_masks_cpu" in rollout
    assert all(mask.device.type == "cpu" for mask in rollout["selective_loss_masks_cpu"])
    iterator = data.DataIterator(
        rollout,
        micro_batch_indices=[[1, 0]],
        dummy_micro_batch_offsets={0} if case in ("dummy", "repeated_dummy") else None,
    )
    # get_batch must request the optional CPU field itself, as current callers do
    # not include the feature's metadata key in their standard field list.
    seen = []
    get_next = iterator.get_next

    def capture_next(keys):
        selected = get_next(keys)
        seen.extend(selected["selective_loss_masks_cpu"])
        return selected

    monkeypatch.setattr(iterator, "get_next", capture_next)
    batch = data.get_batch(
        iterator,
        ["tokens", "loss_masks", "total_lengths", "response_lengths"],
        pad_multiplier=8,
        allgather_cp=allgather_cp,
        is_vl_model=args.is_vl_model,
    )
    assert "selective_loss_masks_cpu" not in batch  # Do not copy original CPU masks to GPU.
    assert batch["total_lengths"] == [5, 3]
    assert torch.equal(seen[0], rollout["selective_loss_masks_cpu"][1])
    assert batch["tokens"].shape == (1, 16)
    assert "unsplit_tokens" not in batch
    assert "vlm_packed_seq_params" not in batch
    assert batch.get("__is_dummy__", False) == (case in ("dummy", "repeated_dummy"))
    expected_main = [(mask.numel(), mask.nonzero().flatten()) for mask in batch["loss_masks"]]
    packed = batch["packed_seq_params"]
    expected_mtp = []
    full_mask = batch["full_loss_masks"]
    for _ in range(2):
        [full_mask] = roll_tensor([full_mask], cp_group=None, packed_seq_params=packed)
        expected_mtp.append(full_mask[0].nonzero().flatten())

    # CPU get_batch provided one reference plan. Normal PP4 training uses this
    # separate helper because its tokens/full_loss_masks already reside on CUDA.
    rebuilt = dict(batch, selective_loss_masks_cpu=seen)
    rebuilt["packed_seq_params"] = copy(packed)
    rebuilt.pop("selective_logit_rows")
    del rebuilt["packed_seq_params"].relax_sft_selective_mtp_rows
    prepare_selective_logits_from_cpu_masks(rebuilt, args)
    for planned_batch in (batch, rebuilt):
        for (actual_length, actual_rows), (length, rows) in zip(
            planned_batch["selective_logit_rows"], expected_main, strict=True
        ):
            assert actual_length == length
            torch.testing.assert_close(actual_rows, rows)
        metadata = planned_batch["packed_seq_params"]
        assert metadata.relax_sft_selective_full_rows == 16
        for actual, expected in zip(metadata.relax_sft_selective_mtp_rows, expected_mtp, strict=True):
            torch.testing.assert_close(actual, expected)
        if case != "dummy":
            assert 0 < sum(rows.numel() for _, rows in planned_batch["selective_logit_rows"]) < 8
            assert 0 < metadata.relax_sft_selective_mtp_rows[0].numel() < 16
        else:
            assert all(rows.numel() == 0 for _, rows in planned_batch["selective_logit_rows"])
            assert all(rows.numel() == 0 for rows in metadata.relax_sft_selective_mtp_rows)
