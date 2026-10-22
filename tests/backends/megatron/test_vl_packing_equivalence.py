"""Numerical reference for the SFT packer's document and loss boundaries."""

from argparse import Namespace
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from relax.utils.sft_utils import align_loss_mask_for_sft
from tests.backends.megatron.test_data_vpp import _load_data_module


def test_vl_packing_matches_independent_samples_logits_and_gradients(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "PackedSeqParams", SimpleNamespace)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=0.0))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module, "slice_with_cp", lambda values, *_args, **_kwargs: values)

    tokens = [torch.tensor([1, 2, 3, 4, 5, 6, 7]), torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 16])]
    raw_masks = [torch.tensor([0, 0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])]
    media = [
        {"pixel_values": torch.tensor([[1.0, 2.0]]), "image_grid_thw": torch.tensor([[1, 1, 1]])},
        {"pixel_values": torch.tensor([[3.0, 4.0]]), "image_grid_thw": torch.tensor([[1, 1, 1]])},
    ]
    batch = {
        "tokens": tokens,
        "loss_masks": raw_masks,
        "total_lengths": [len(row) for row in tokens],
        "response_lengths": [len(row) for row in tokens],
        "multimodal_train_inputs": media,
    }
    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, is_vl_model=True)
    lengths = [len(row) for row in tokens]
    real_length = sum(lengths)
    boundaries = packed["packed_seq_params"].cu_seqlens_q[:3].tolist()
    assert boundaries == [0, lengths[0], real_length]
    assert packed["multimodal_num_items"]["pixel_values"] == [1, 1]
    torch.testing.assert_close(packed["tokens"][0, :real_length], torch.cat(tokens))
    torch.testing.assert_close(
        packed["full_loss_masks"][0, :real_length],
        torch.cat([align_loss_mask_for_sft(mask) for mask in raw_masks]),
    )

    torch.manual_seed(9)
    embedding = torch.nn.Embedding(32, 8, dtype=torch.float64)
    image_projection = torch.nn.Linear(2, 8, bias=False, dtype=torch.float64)
    qkv = torch.nn.Linear(8, 24, bias=False, dtype=torch.float64)
    output = torch.nn.Linear(8, 32, bias=False, dtype=torch.float64)
    parameters = tuple(p for module in (embedding, image_projection, qkv, output) for p in module.parameters())

    def forward(ids, image_values, document_ids):
        hidden = embedding(ids).clone()
        # One visual feature at position 1 of each document, as in the VL merge.
        image_positions = [int((document_ids == document).nonzero()[1]) for document in document_ids.unique()]
        hidden[image_positions] = hidden[image_positions] + image_projection(image_values)
        q, k, v = qkv(hidden).chunk(3, dim=-1)
        positions = torch.arange(len(ids))
        permitted = (document_ids[:, None] == document_ids[None, :]) & (positions[:, None] >= positions[None, :])
        scores = q @ k.T / q.shape[-1] ** 0.5
        attention = torch.softmax(scores.masked_fill(~permitted, float("-inf")), dim=-1)
        return output(attention @ v)

    packed_ids = packed["tokens"][0, :real_length]
    packed_images = packed["multimodal_train_inputs"]["pixel_values"].to(torch.float64)
    packed_documents = torch.bucketize(torch.arange(real_length), torch.tensor(boundaries[1:-1]), right=True)
    packed_logits = forward(packed_ids, packed_images, packed_documents)
    packed_mask = packed["full_loss_masks"][0, :real_length].to(torch.float64)
    packed_targets = torch.cat([F.pad(row[1:], (0, 1)) for row in tokens])
    packed_loss = (F.cross_entropy(packed_logits, packed_targets, reduction="none") * packed_mask).sum()
    packed_grads = torch.autograd.grad(packed_loss, parameters)

    separate_logits = []
    separate_loss = 0
    for row, mask, image in zip(tokens, raw_masks, media):
        logits = forward(row, image["pixel_values"].to(torch.float64), torch.zeros(len(row), dtype=torch.long))
        aligned_mask = align_loss_mask_for_sft(mask).to(torch.float64)
        separate_logits.append(logits)
        separate_loss = (
            separate_loss + (F.cross_entropy(logits, F.pad(row[1:], (0, 1)), reduction="none") * aligned_mask).sum()
        )
    separate_grads = torch.autograd.grad(separate_loss, parameters)

    torch.testing.assert_close(packed_logits, torch.cat(separate_logits), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(packed_loss, separate_loss, atol=1e-12, rtol=1e-12)
    for packed_grad, separate_grad in zip(packed_grads, separate_grads):
        torch.testing.assert_close(packed_grad, separate_grad, atol=1e-12, rtol=1e-12)
