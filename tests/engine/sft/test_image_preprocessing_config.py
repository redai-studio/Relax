# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.engine.sft.image_preprocessing_config import configure_image_preprocessing


_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "overrides,is_sft,expected",
    [
        ({}, True, True),
        ({"sft_train_data_prefetch": False}, True, False),
        ({"per_rank_fetch": False}, True, False),
        ({"multimodal_keys": None}, True, False),
        ({"multimodal_keys": {"video": "videos"}}, True, False),
        ({"sft_async_prepack": True}, True, False),
        ({"max_staleness": 0}, True, False),
        ({}, False, False),
    ],
)
def test_image_preprocessing_is_derived_without_changing_other_options(overrides, is_sft, expected):
    values = dict(
        sft_train_data_prefetch=True,
        per_rank_fetch=True,
        multimodal_keys={"image": "images"},
        sft_async_prepack=False,
        max_staleness=1,
    )
    values.update(overrides)
    args = SimpleNamespace(**values)
    configure_image_preprocessing(args, is_sft)
    assert args.sft_image_preprocess_on_rank is expected
    assert vars(args) == {**values, "sft_image_preprocess_on_rank": expected}


@pytest.mark.parametrize(
    "inputs",
    [
        {"pixel_values": object()},
        {"pixel_values_videos": object()},
        {"pixel_values": None, "image_grid_thw": object()},
        {"pixel_values": object(), "image_grid_thw": None},
    ],
)
def test_unsupported_image_payload_keeps_original_pixels(inputs):
    source = _ROOT / "relax/engine/sft/dataset/image_transport.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], cls], type_ignores=[])
    namespace = {"SFT_HF_IMAGE_REBUILD_SUPPORTED": "_sft_hf_image_rebuild_supported"}
    exec(compile(module, str(source), "exec"), namespace)
    transport = namespace["ImageReferenceTransport"](True)
    rendered = SimpleNamespace(image_fallback=False, image_refs=["/data/a.jpg"])
    retained, descriptor = transport.split(inputs, rendered, "test")
    assert retained is inputs
    assert descriptor is None
