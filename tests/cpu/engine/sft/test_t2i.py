# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Text-to-image supervision uses target images without synthetic edit
inputs."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.backends.fsdp.arguments import validate_generative_config
from relax.engine.sft.bootstrap import resolve_sft_num_rollout
from relax.engine.sft.image import ImageEditDataset, ImageSFTDataset


def _args(**overrides):
    values = dict(
        loss_type="sft",
        generation_task="t2i",
        train_backend="fsdp",
        model_path="/worker/model",
        prompt_data="/worker/data.json",
        t2i_sft_config={"data_root": "/worker", "dataset_size": 19},
        global_batch_size=8,
        rollout_batch_size=8,
        fsdp_trainable_mode="lora",
        lora_rank=8,
        num_rollout=None,
        num_epoch=2,
        seed=42,
    )
    return Namespace(**{**values, **overrides})


def test_t2i_validation_and_sizing_require_no_model_or_data_mount():
    args = _args()
    validate_generative_config(args)
    resolve_sft_num_rollout(args)
    assert args.num_rollout == 6
    assert args.num_rollout_per_epoch == 3
    assert args.model_adapter_path == "relax.models.qwen_image.sft_adapter.QwenImageSFTAdapter"
    assert args.fsdp_master_dtype == "fp32"
    assert args.t2i_sft_config["height"] == args.t2i_sft_config["width"] == 512
    assert args.t2i_sft_config["text_encoder_placement"] == "sharded"
    assert args.t2i_sft_config["cache_vae_posterior"] is False
    assert "conditioning_images" not in args.t2i_sft_config
    assert not hasattr(args, "i2i_sft_config")


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_size", True),
        ("dataset_size", 0),
        ("height", 0),
        ("height", 513),
        ("height", "512"),
        ("width", 15),
        ("width", True),
        ("max_sequence_length", 513),
        ("max_sequence_length", 0),
        ("text_encoder_placement", "automatic"),
        ("text_encoder_placement", None),
        ("text_encoder_placement", True),
        ("text_encoder_placement", ["replicated"]),
        ("caption_dropout", -0.1),
        ("caption_dropout", 1.1),
        ("caption_dropout", True),
        ("caption_dropout", float("nan")),
        ("caption_dropout", float("inf")),
        ("random_crop", "true"),
        ("gradient_checkpointing", 1),
        ("cache_vae_posterior", 1),
        ("cache_vae_posterior", "true"),
        ("cache_vae_posterior", None),
        ("deterministic_attention", 0),
        ("image_prefetch_depth", 1),
        ("image_prefetch_depth", False),
        ("image_prefetch_workers", 0),
        ("data_root", ""),
        ("conditioning_images", 1),
        ("fsdp_branch_groups", True),
    ],
)
def test_t2i_rejects_invalid_or_unimplemented_preprocessing(field, value):
    args = _args()
    args.t2i_sft_config[field] = value
    with pytest.raises(ValueError, match=field):
        validate_generative_config(args)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"fsdp_trainable_mode": "full", "lora_rank": 0}, "supports only"),
        ({"lora_rank": 0}, "positive --lora-rank"),
        ({"i2i_sft_config": {}}, "does not use --i2i-sft-config"),
        ({"use_dynamic_batch_size": True}, "use-dynamic-batch-size"),
        ({"global_batch_size": 7}, "equal positive"),
        ({"fsdp_master_dtype": "bf16"}, "FP32 optimizer masters"),
        ({"fsdp_activation_checkpointing": True}, "fsdp-activation-checkpointing"),
    ],
)
def test_t2i_rejects_unsupported_training_contract(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_generative_config(_args(**overrides))


@pytest.mark.parametrize("dropout", [0, 0.5, 1])
def test_t2i_accepts_explicit_preprocessing_controls(dropout):
    args = _args()
    args.t2i_sft_config.update(height=256, width=768, caption_dropout=dropout, random_crop=True)
    validate_generative_config(args)
    assert args.t2i_sft_config["caption_dropout"] == dropout
    assert args.t2i_sft_config["random_crop"] is True


@pytest.mark.parametrize("placement", ["sharded", "replicated"])
def test_t2i_accepts_explicit_text_encoder_placement(placement):
    args = _args()
    args.t2i_sft_config["text_encoder_placement"] = placement
    validate_generative_config(args)
    assert args.t2i_sft_config["text_encoder_placement"] == placement


@pytest.mark.parametrize("enabled", [False, True])
def test_t2i_accepts_explicit_posterior_bundle_policy(enabled):
    args = _args()
    args.t2i_sft_config["cache_vae_posterior"] = enabled
    validate_generative_config(args)
    assert args.t2i_sft_config["cache_vae_posterior"] is enabled


def test_t2i_metadata_contains_only_caption_and_target(tmp_path):
    row = {"image": "target.png", "prompt": "a red boat"}
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row]))
    args = _args(prompt_data=str(path), t2i_sft_config={"data_root": str(tmp_path), "dataset_size": 1})
    validate_generative_config(args)

    class Adapter:
        def prepare_sample(self, value, root, config, *, index):
            assert root == tmp_path
            assert config is args.t2i_sft_config
            assert index == 0
            return value

    dataset = ImageSFTDataset(args, Adapter())
    assert dataset[0] == row
    with pytest.raises(ValueError, match="requires generation_task=i2i"):
        ImageEditDataset(args, Adapter())


@pytest.mark.parametrize(
    "row,match",
    [
        ({"prompt": "a boat"}, "Missing target image"),
        ({"image": "target.png", "prompt": None}, "Invalid prompt"),
        ({"image": "", "prompt": "a boat"}, "Missing target image"),
        ({"image": "target.png", "prompt": "a boat", "edit_image": []}, "must not contain edit_image"),
        ({"image": "target.png", "prompt": "a boat", "edit_image": ["fake.png"]}, "must not contain edit_image"),
    ],
)
def test_t2i_metadata_rejects_missing_targets_or_edit_inputs(tmp_path, row, match):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row]))
    args = _args(prompt_data=str(path), t2i_sft_config={"data_root": str(tmp_path), "dataset_size": 1})
    with pytest.raises(ValueError, match=match):
        ImageSFTDataset(args, SimpleNamespace(prepare_sample=lambda *_args: None))
