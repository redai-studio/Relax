# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Image-reference transport and flattened tool-call contracts."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from relax.engine.sft.dataset.streaming import (
    ProcessedSample,
    SFTStreamingDataset,
    pack_samples_for_tq,
)
from relax.utils.utils import dict_to_tensordict
from tests.engine.sft.dataset.test_streaming import _FakeTokenizer, _write_jsonl


def test_pack_samples_for_tq_ships_image_refs_only_when_present():
    from relax.utils.data.image_refs import SFT_IMAGE_REFS_FIELD

    descriptor = {
        "version": 1,
        "kind": "kimi_k3_sft_image_v1",
        "image_refs": ["/data/a.jpg"],
        "pixel_shape": [4, 3],
        "image_grid_thw": [[1, 2, 4]],
    }
    text_sample = ProcessedSample(
        tokens=torch.tensor([1, 2]),
        loss_mask=torch.tensor([0, 1]),
        total_length=2,
        multimodal_train_inputs=None,
        source_idx=0,
    )
    image_sample = ProcessedSample(
        tokens=torch.tensor([1, 2, 3]),
        loss_mask=torch.tensor([0, 1, 1]),
        total_length=3,
        multimodal_train_inputs={"image_grid_thw": torch.tensor([[1, 2, 4]])},
        source_idx=1,
        image_refs_descriptor=descriptor,
    )

    text_batch = pack_samples_for_tq([text_sample])
    assert SFT_IMAGE_REFS_FIELD not in text_batch

    batch = pack_samples_for_tq([text_sample, image_sample])
    # Text-only rows stay None; the image row lost its pixels but keeps the grid.
    assert batch[SFT_IMAGE_REFS_FIELD] == [None, descriptor]
    assert batch["multimodal_train_inputs"][0] is None
    assert list(batch["multimodal_train_inputs"][1]) == ["image_grid_thw"]

    payload = dict_to_tensordict(batch, batch_size=2)
    assert SFT_IMAGE_REFS_FIELD in payload.keys()


def _write_image_rows(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>\nPath-based image."},
                    {"role": "assistant", "content": "A"},
                ],
                "images": ["/data/from-disk.jpg"],
            },
            {
                "messages": [
                    {"role": "user", "content": "<image>\nInline payload image."},
                    {"role": "assistant", "content": "B"},
                ],
                "images": [{"base64": "aGVsbG8="}],
            },
        ],
    )


def _make_image_ref_dataset(path: Path) -> SFTStreamingDataset:
    return SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=MagicMock(),
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
        image_preprocess_on_rank=True,
    )


def test_image_refs_automatically_keep_pixel_payload_for_inline_rows(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    _write_image_rows(path)
    ds = _make_image_ref_dataset(path)
    from relax.utils.data.image_refs import SFT_IMAGE_REFS_FIELD

    short = (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 1]))
    mm_inputs = {"pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    with (
        # Force the K3 encoding branch so row 0 is eligible for the descriptor.
        patch("relax.engine.sft.dataset.streaming.is_kimi_k3_tokenizer", return_value=True),
        patch("relax.engine.sft.dataset.streaming.render_with_loss_mask", return_value=short),
        patch(
            "relax.engine.sft.dataset.streaming.preprocess_multimodal",
            side_effect=[([1, 2, 3], dict(mm_inputs)), ([1, 2, 3], dict(mm_inputs))],
        ),
    ):
        samples = ds.get_batch_in_order(0, 2)
    batch = pack_samples_for_tq(samples)

    # Row 0 (file path) is reference-mode; row 1 (inline payload) fell back.
    descriptor = batch[SFT_IMAGE_REFS_FIELD][0]
    assert descriptor["image_refs"] == ["/data/from-disk.jpg"]
    assert batch[SFT_IMAGE_REFS_FIELD][1] is None
    assert "pixel_values" not in samples[0].multimodal_train_inputs
    assert "pixel_values" in samples[1].multimodal_train_inputs
    # Tokens/masks/grid are untouched for both rows.
    assert samples[0].multimodal_train_inputs["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert samples[1].tokens.tolist() == [1, 2, 3]
    inline_only_batch = pack_samples_for_tq([samples[1]], force_image_refs_field=True)
    assert inline_only_batch[SFT_IMAGE_REFS_FIELD] == [None]
    assert "pixel_values" in inline_only_batch["multimodal_train_inputs"][0]
    ds.stop()


def test_image_refs_stamps_hf_kind_for_generic_processor_path(tmp_path: Path):
    """Generic HF processor samples (non-K3) rebuild through the hf kind."""
    from relax.utils.data.image_refs import (
        HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND,
        SFT_HF_IMAGE_REBUILD_SUPPORTED,
        SFT_IMAGE_REFS_FIELD,
    )

    path = tmp_path / "rows.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>\nPath image."},
                    {"role": "assistant", "content": "A"},
                ],
                "images": ["/data/from-disk.jpg"],
            }
        ],
    )
    ds = _make_image_ref_dataset(path)

    short = (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 1]))
    with (
        patch("relax.engine.sft.dataset.streaming.render_with_loss_mask", return_value=short),
        patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered"),
        patch(
            "relax.engine.sft.dataset.streaming.preprocess_multimodal",
            return_value=(
                [1, 2, 3],
                {
                    "pixel_values": torch.zeros(4, 3),
                    "image_grid_thw": torch.tensor([[1, 2, 4]]),
                    SFT_HF_IMAGE_REBUILD_SUPPORTED: True,
                },
            ),
        ),
    ):
        samples = ds.get_batch_in_order(0, 1)
    batch = pack_samples_for_tq(samples)

    descriptor = batch[SFT_IMAGE_REFS_FIELD][0]
    assert descriptor["kind"] == HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND
    assert descriptor["image_refs"] == ["/data/from-disk.jpg"]
    assert "pixel_values" not in samples[0].multimodal_train_inputs
    ds.stop()
