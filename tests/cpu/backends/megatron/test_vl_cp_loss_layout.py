# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bridge's per-document zigzag output must retain next-token loss
alignment."""

from argparse import Namespace
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron import cp_utils, data
    from relax.backends.megatron import loss as loss_module
    from relax.engine.sft.eval.ppl import compute_sft_eval_step
    from relax.utils.sft_utils import align_loss_mask_for_sft
except (ImportError, AttributeError) as exc:
    pytest.skip(f"Requires the pinned Megatron runtime: {exc}", allow_module_level=True)


def _local_hidden(hidden, lengths, padded, cp_size, rank):
    if cp_size == 1:
        return hidden.unsqueeze(0)
    result = []
    for values, physical in zip(hidden.split(lengths), padded, strict=True):
        values = F.pad(values, (0, 0, 0, physical - len(values)))
        chunks = values.chunk(2 * cp_size)
        result.extend((chunks[rank], chunks[2 * cp_size - rank - 1]))
    return torch.cat(result).unsqueeze(0)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("chunked,recompute", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("dynamic", [False, True])
def test_vl_cp_sft_loss_gradients_and_eval_match_unpartitioned(monkeypatch, cp_size, chunked, recompute, dynamic):
    """Include padding, empty supervision, a one-token document and span
    ends."""
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(loss_module, "validate_opd_topk_gather", lambda *_: None)

    def fail_redistribution(*_, **__):
        raise AssertionError("Bridge output is already zigzag; no contiguous redistribution is valid")

    monkeypatch.setattr(loss_module, "_allgather_cp_redistribute", fail_redistribution)

    def log_probs(logits, tokens, *_, **__):
        return -F.cross_entropy(logits, tokens, reduction="none").unsqueeze(-1), None

    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", log_probs)
    torch.manual_seed(17)
    lengths = [7, 10, 1]
    padded = [((length + 4 * cp_size - 1) // (4 * cp_size)) * (4 * cp_size) for length in lengths]
    tokens = [torch.randint(0, 19, (length,)) for length in lengths]
    masks = [torch.tensor([0, 0, 1, 1, 0, 1, 1]), torch.zeros(10, dtype=torch.long), torch.zeros(1)]
    aligned = [align_loss_mask_for_sft(mask) for mask in masks]
    hidden = torch.randn(sum(lengths), 5, requires_grad=True)
    weight = torch.randn(19, 5, requires_grad=True)
    ref = sum(
        (F.cross_entropy((values[:-1] @ weight.T), labels[1:], reduction="none") * mask[1:]).sum()
        for values, labels, mask in zip(hidden.split(lengths), tokens, masks, strict=True)
    )
    expected_grads = torch.autograd.grad(ref, (hidden, weight), retain_graph=True)
    args = Namespace(
        qkv_format="thd",
        loss_type="sft",
        allgather_cp=True,
        rollout_temperature=1.0,
        sft_chunked_logits=chunked,
        recompute_loss_function=recompute,
        sft_logits_chunk_size=3,
        log_probs_chunk_size=3,
    )
    actual = hidden.sum() * 0
    eval_loss, eval_count = 0.0, 0
    for rank in range(cp_size):
        monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda r=rank: r)
        metadata = dict(
            total_lengths=lengths,
            response_lengths=lengths,
            loss_masks=aligned,
            padded_total_lengths=padded if cp_size > 1 else None,
            dynamic_cp_size=cp_size if dynamic else None,
            dynamic_cp_rank=rank if dynamic else None,
        )
        batch = dict(unconcat_tokens=tokens, cp_output_is_zigzag=cp_size > 1, **metadata)
        reducer = cp_utils.get_sum_of_sample_mean(**metadata, calculate_per_token_loss=True)
        local_hidden = _local_hidden(hidden, lengths, padded, cp_size, rank)
        local_logits = local_hidden @ weight.T
        if chunked:
            value, _ = loss_module.sft_loss_function_chunked(
                args, batch, local_hidden, reducer, lm_head_forward=lambda h: (h @ weight.T, None)
            )
        else:
            value, _ = loss_module.sft_loss_function(args, batch, local_logits, reducer)
        actual = actual + value
        _, result = compute_sft_eval_step(local_logits, args=args, **batch)
        eval_loss += result["sum_neg_log_prob"][0].item()
        eval_count += result["num_tokens"][0].item()
    if recompute:
        actual.backward()
        actual_grads = (hidden.grad, weight.grad)
    else:
        actual_grads = torch.autograd.grad(actual, (hidden, weight))
    torch.testing.assert_close(actual, ref)
    for got, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, expected)
    assert eval_loss == pytest.approx(ref.item(), rel=1e-6)
    assert eval_count == sum(mask[1:].sum().item() for mask in masks)


def test_contiguous_allgather_layout_is_preserved_without_bridge_metadata(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: 1)
    args = Namespace(
        qkv_format="thd",
        loss_type="sft",
        allgather_cp=True,
        rollout_temperature=1.0,
        sft_chunked_logits=False,
    )
    chunks = list(
        loss_module.get_responses(
            torch.arange(8, 16, dtype=torch.float32).reshape(1, 8, 1),
            args=args,
            unconcat_tokens=[torch.arange(8), torch.arange(8, 16)],
            total_lengths=[8, 8],
            response_lengths=[4, 4],
        )
    )
    assert chunks[0][0].numel() == 0
    assert chunks[1][0].flatten().tolist() == [11, 12, 13, 14]
    assert chunks[1][1].tolist() == [12, 13, 14, 15]


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_glm_packed_batch_matches_unsplit_next_token_loss(monkeypatch, cp_size):
    args = Namespace(
        qkv_format="thd",
        loss_type="sft",
        allgather_cp=False,
        rollout_temperature=1.0,
        glm53_packed_cp=True,
        sft_chunked_logits=False,
    )
    monkeypatch.setattr(data, "get_args", lambda: args)
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: cp_size)
    documents = [torch.arange(1, 6), torch.arange(11, 28)]
    lengths = [doc.numel() for doc in documents]
    masks = [torch.tensor([0, 0, 1, 0, 1]), torch.zeros(17, dtype=torch.int)]
    expected = [(3, 2), (5, 4)]
    supervised = []
    for rank in range(cp_size):
        monkeypatch.setattr(data.mpu, "get_context_parallel_rank", lambda r=rank: r)
        batch = dict(
            tokens=documents,
            total_lengths=lengths,
            response_lengths=lengths,
            loss_masks=masks,
            multimodal_train_inputs=[{"pixel_values": torch.ones(2, 3)}, None],
        )
        packed = data.get_batch(
            iter([(batch, None)]), list(batch), pad_multiplier=8, is_vl_model=True, pack_device=torch.device("cpu")
        )
        local_ids = packed["tokens"]
        if cp_size > 1:
            padded = packed["padded_total_lengths"]
            padded_documents = [F.pad(doc, (0, size - len(doc))) for doc, size in zip(documents, padded, strict=True)]
            assert packed["unsplit_tokens"].tolist() == [torch.cat(padded_documents).tolist()]
            for key in ("packed_seq_params", "vlm_packed_seq_params"):
                assert packed[key].cu_seqlens_q.tolist() == [0, padded[0], sum(padded)]
            torch.testing.assert_close(
                local_ids.squeeze(0),
                _local_hidden(torch.cat(documents).unsqueeze(-1), lengths, padded, cp_size, rank).flatten(),
            )
        else:
            assert "unsplit_tokens" not in packed
        assert "unsplit_attention_mask" not in packed
        assert packed["multimodal_train_inputs"]["pixel_values"].shape == (2, 3)
        metadata = dict(
            total_lengths=lengths,
            response_lengths=lengths,
            padded_total_lengths=packed.get("padded_total_lengths"),
        )
        local_masks = cp_utils.get_cp_local_loss_masks(loss_masks=packed["loss_masks"], **metadata)
        responses = loss_module.get_responses(
            local_ids.float().unsqueeze(-1),
            args=args,
            unconcat_tokens=packed["unconcat_tokens"],
            cp_output_is_zigzag=packed.get("cp_output_is_zigzag", False),
            **metadata,
        )
        for (predictors, labels), mask in zip(responses, local_masks, strict=True):
            supervised.extend(
                zip(labels[mask.bool()].tolist(), predictors[mask.bool()].flatten().tolist(), strict=True)
            )
        assert packed["full_loss_masks"].sum() == sum(mask.sum() for mask in local_masks)
    assert sorted(supervised) == expected


def test_glm_cp_vision_matches_full_insertion_and_gradients():
    try:
        from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashModel
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Requires the pinned GLM Bridge runtime: {exc}")

    class Vision(torch.nn.Module):
        dtype = torch.float32

        def forward(self, pixels, grid_thw):
            return SimpleNamespace(pooler_output=pixels * 2)

    model = GLM53FlashModel.__new__(GLM53FlashModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(image_token_id=28, video_start_token_id=29, video_end_token_id=30)
    model.get_placeholder_mask = MethodType(Glm5NextModel.get_placeholder_mask, model)
    model.visual = Vision()
    model.get_image_features = lambda pixels, grid: SimpleNamespace(pooler_output=(pixels * 2,))
    model.get_video_features = lambda pixels, grid: SimpleNamespace(pooler_output=(pixels * 3,))

    # GLM uses the same placeholder ID in both modalities. The image straddles
    # two zigzag shards and the video is entirely on rank 1.
    tokens = torch.tensor([[1, 2, 3, 28, 28, 28, 29, 28, 28, 30, 4, 5, 6, 0, 0, 0]])
    embeds = torch.randn(1, 16, 5, requires_grad=True)
    images = torch.randn(3, 5, requires_grad=True)
    video = torch.randn(2, 5, requires_grad=True)
    inputs = dict(pixel_values=images, pixel_values_videos=video, image_grid_thw=None, video_grid_thw=None)
    reference = model._insert_vision_features(tokens, embeds, **inputs)
    variables = (embeds, images, video)
    expected_grads = torch.autograd.grad(reference.square().sum(), variables)
    local_loss = 0
    for indices in ([0, 1, 2, 3, 12, 13, 14, 15], [4, 5, 6, 7, 8, 9, 10, 11]):
        index = torch.tensor(indices)
        local = model._insert_cp_vision_features(tokens, embeds.index_select(1, index), index, **inputs)
        torch.testing.assert_close(local, reference.index_select(1, index))
        local_loss += local.square().sum()
    for grad, expected in zip(torch.autograd.grad(local_loss, variables), expected_grads, strict=True):
        torch.testing.assert_close(grad, expected)
