# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Image data ordering must survive restart without consuming training RNG."""

import json
from argparse import ArgumentParser

import pytest
import torch
from torch.utils.data import DistributedSampler

from relax.backends.fsdp.arguments import add_generative_arguments, validate_i2i_sft_config
from relax.engine.sft.bootstrap import resolve_sft_num_rollout
from relax.engine.sft.image import ImageBatchSampler, ImageEditDataset
from relax.utils.opd.opd_utils import add_opd_arguments
from tests.cpu.relax_parser import args_for_flags, parser_for_flags


@pytest.mark.parametrize("size", [3, 16, 19, 64])
def test_sample_order_matches_eight_rank_reference(size):
    planner = ImageBatchSampler(size, 8, 42)
    initial_rng = torch.get_rng_state().clone()
    for epoch in range(2):
        for rank in range(8):
            reference = DistributedSampler(range(size), num_replicas=8, rank=rank, seed=42, drop_last=False)
            reference.set_epoch(epoch)
            actual = [
                planner.batch(epoch * planner.steps_per_epoch + step)[rank] for step in range(planner.steps_per_epoch)
            ]
            assert actual == list(reference)
    assert torch.equal(initial_rng, torch.get_rng_state())
    restored = ImageBatchSampler(size, 8, 42)
    assert restored.batch(planner.steps_per_epoch + 1) == planner.batch(planner.steps_per_epoch + 1)


def _args(argv=(), **overrides):
    values = dict(
        loss_type="sft",
        generation_task="i2i",
        train_backend="fsdp",
        model_path="/worker/model",
        prompt_data="/worker/data.json",
        i2i_sft_config={"data_root": "/worker", "dataset_size": 19},
        global_batch_size=8,
        rollout_batch_size=8,
        num_rollout=None,
        num_epoch=2,
        seed=42,
        micro_batch_size=1,
        debug_train_only=False,
        debug_rollout_only=False,
    )
    defaults = args_for_flags(
        "--task-type",
        "--fully-async",
        "--hybrid",
        "--offload-train",
        "--use-dynamic-batch-size",
        "--balance-data",
        "--sft-async-prepack",
        "--sft-chunked-logits",
        "--sft-predict-interval",
        "--eval-prompt-data",
        "--eval-size",
        "--custom-dataset-class-path",
        "--use-routing-replay",
        "--use-rollout-routing-replay",
        "--enable-mtp-training",
        "--lora-rank",
        "--lora-merge-mode",
        "--lora-adapter-mode",
        "--num-critic-only-steps",
        "--sft-train-data-prefetch",
        "--n-samples-per-prompt",
    )
    parser = add_opd_arguments(add_generative_arguments(ArgumentParser()))
    args = parser.parse_args(argv, namespace=defaults)
    vars(args).update({**values, **overrides})
    return args


def test_head_validation_and_sizing_need_no_worker_mount():
    args = _args()
    validate_i2i_sft_config(args)
    resolve_sft_num_rollout(args)
    assert args.num_rollout == 6
    assert args.num_rollout_per_epoch == 3
    assert args.i2i_sft_config["max_pixels"] == 1024**2


def test_image_sft_rejects_sequence_classification():
    parsed = parser_for_flags("--task-type").parse_args(["--task-type", "seq_cls"])
    with pytest.raises(ValueError, match="Image SFT does not support text classification"):
        validate_i2i_sft_config(_args(task_type=parsed.task_type))


@pytest.mark.parametrize("tiers", [[True], [False], [1024, True], [1.5], ["1024"], [0], [-1], [], True])
def test_fixed_pixels_tiers_reject_invalid_values(tiers):
    args = _args(i2i_sft_config={"data_root": "/worker", "dataset_size": 19, "fixed_pixels_tiers": tiers})
    with pytest.raises(ValueError, match="fixed_pixels_tiers"):
        validate_i2i_sft_config(args)


@pytest.mark.parametrize("tiers", [None, [1], [1024], [1440**2, 2048**2]])
def test_fixed_pixels_tiers_accept_positive_integers(tiers):
    args = _args(i2i_sft_config={"data_root": "/worker", "dataset_size": 19, "fixed_pixels_tiers": tiers})
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["fixed_pixels_tiers"] == tiers


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_tier_balancing_requires_boolean(value):
    args = _args(i2i_sft_config={"data_root": "/worker", "dataset_size": 19, "dp_tier_micro_balance": value})
    with pytest.raises(ValueError, match="dp_tier_micro_balance"):
        validate_i2i_sft_config(args)


def test_tier_balancing_validates_and_preserves_checkpoint_semantics():
    args = _args()
    validate_i2i_sft_config(args)
    assert "dp_tier_micro_balance" not in args.i2i_sft_config
    args.i2i_sft_config["dp_tier_micro_balance"] = False
    validate_i2i_sft_config(args)
    assert "dp_tier_micro_balance" not in args.i2i_sft_config
    args.i2i_sft_config["dp_tier_micro_balance"] = True
    with pytest.raises(ValueError, match="at least two"):
        validate_i2i_sft_config(args)
    args.i2i_sft_config["fixed_pixels_tiers"] = [4, 9]
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["dp_tier_micro_balance"] is True
    assert args.i2i_sft_config["seed"] == 42


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_rank_rotation_requires_boolean(value):
    args = _args(i2i_sft_config={"data_root": "/worker", "dataset_size": 19, "dp_tier_rank_rotation": value})
    with pytest.raises(ValueError, match="dp_tier_rank_rotation"):
        validate_i2i_sft_config(args)


def test_rank_rotation_validates_and_preserves_disabled_checkpoint_contract():
    args = _args()
    args.i2i_sft_config["dp_tier_rank_rotation"] = False
    validate_i2i_sft_config(args)
    assert "dp_tier_rank_rotation" not in args.i2i_sft_config
    args.i2i_sft_config["dp_tier_rank_rotation"] = True
    with pytest.raises(ValueError, match="requires dp_tier_micro_balance"):
        validate_i2i_sft_config(args)
    args.i2i_sft_config.update(dp_tier_micro_balance=True, fixed_pixels_tiers=[4, 9])
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["dp_tier_rank_rotation"] is True


@pytest.mark.parametrize(
    "option,value",
    [
        ("fully_async", True),
        ("sft_async_prepack", True),
        ("tensor_model_parallel_size", 2),
        ("lora_rank", 32),
        ("global_batch_size", 7),
        ("eval_size", 8),
    ],
)
def test_reject_unsupported_training_contract(option, value):
    with pytest.raises(ValueError):
        validate_i2i_sft_config(_args(**{option: value}))


def test_image_sft_rejects_enabled_activation_checkpointing():
    with pytest.raises(ValueError, match="Image SFT does not support --fsdp-activation-checkpointing"):
        validate_i2i_sft_config(_args(["--fsdp-activation-checkpointing"]))


@pytest.mark.parametrize("argv", [[], ["--fsdp-master-dtype", "fp32"]])
def test_image_sft_derives_fp32_master_dtype(argv):
    args = _args(argv)
    validate_i2i_sft_config(args)
    assert args.fsdp_master_dtype == "fp32"


def test_image_sft_rejects_explicit_bf16_master_dtype():
    with pytest.raises(ValueError, match="Image SFT requires FP32 optimizer masters"):
        validate_i2i_sft_config(_args(["--fsdp-master-dtype", "bf16"]))


@pytest.mark.parametrize("style", ["linear", "cosine"])
def test_image_sft_rejects_nonconstant_fsdp_schedule(style):
    with pytest.raises(ValueError, match="constant"):
        validate_i2i_sft_config(_args(["--fsdp-lr-scheduler", style]))


@pytest.mark.parametrize("argv", [["--fsdp-param-dtype", "fp32"], ["--fsdp-reduce-dtype", "bf16"]])
def test_image_sft_rejects_unsupported_compute_precision(argv):
    with pytest.raises(ValueError, match="Image SFT requires BF16 compute and FP32 reduction"):
        validate_i2i_sft_config(_args(argv))


def test_image_sft_rejects_enabled_cpu_offload():
    with pytest.raises(ValueError, match="Image SFT does not support --fsdp-cpu-offload"):
        validate_i2i_sft_config(_args(["--fsdp-cpu-offload"]))


def test_image_sft_rejects_enabled_debug_fingerprint():
    with pytest.raises(ValueError, match="Image SFT does not support --fsdp-debug-fingerprint"):
        validate_i2i_sft_config(_args(["--fsdp-debug-fingerprint"]))


def test_image_sft_lora_mode_requires_positive_rank():
    args = _args(["--fsdp-trainable-mode", "lora"], lora_rank=0)
    with pytest.raises(ValueError, match="LoRA mode requires positive --lora-rank"):
        validate_i2i_sft_config(args)
    validate_i2i_sft_config(_args(["--fsdp-trainable-mode", "lora"], lora_rank=8))


def test_metadata_preserves_condition_order_and_checks_size(tmp_path):
    path = tmp_path / "data.json"
    row = {"image": "target.png", "edit_image": ["source.png", "glyph.png"], "prompt": "replace"}
    path.write_text(json.dumps([row]))
    args = _args(
        prompt_data=str(path), i2i_sft_config={"data_root": str(tmp_path), "dataset_size": 1, "conditioning_images": 2}
    )
    validate_i2i_sft_config(args)

    class Adapter:
        def prepare_sample(self, value, root, config):
            assert root == tmp_path
            return value

    dataset = ImageEditDataset(args, Adapter())
    assert dataset[0] == row
    args.i2i_sft_config["dataset_size"] = 2
    with pytest.raises(ValueError, match="dataset_size"):
        ImageEditDataset(args, Adapter())


@pytest.mark.parametrize("explicit_seed", [None, 7])
def test_multitier_seed_defaults_to_training_seed(explicit_seed):
    args = _args(
        seed=19,
        i2i_sft_config={
            "data_root": "/worker",
            "dataset_size": 19,
            "fixed_pixels_tiers": [1440**2, 2048**2],
            "seed": explicit_seed,
        },
    )
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["seed"] == (19 if explicit_seed is None else explicit_seed)


def test_legacy_max_pixels_keeps_seed_contract():
    args = _args()
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["seed"] is None
    assert "image_prefetch_depth" not in args.i2i_sft_config


@pytest.mark.parametrize("depth,workers", [(0, 2), (4, 2)])
def test_prefetch_config_accepts_explicit_controls(depth, workers):
    args = _args(
        i2i_sft_config={
            "data_root": "/worker",
            "dataset_size": 19,
            "image_prefetch_depth": depth,
            "image_prefetch_workers": workers,
        }
    )
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["image_prefetch_depth"] == depth


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_prefetch_depth", -1),
        ("image_prefetch_depth", True),
        ("image_prefetch_workers", 0),
        ("image_prefetch_workers", "2"),
    ],
)
def test_prefetch_config_rejects_invalid_controls(field, value):
    args = _args(i2i_sft_config={"data_root": "/worker", "dataset_size": 19, field: value})
    with pytest.raises(ValueError, match=field):
        validate_i2i_sft_config(args)


@pytest.mark.parametrize("accept_kwargs", [False, True])
def test_indexed_adapter_receives_row_id(tmp_path, accept_kwargs):
    row = {"image": "target.png", "edit_image": ["source.png"], "prompt": "edit"}
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row, row]))
    args = _args(
        prompt_data=str(path),
        i2i_sft_config={
            "data_root": str(tmp_path),
            "dataset_size": 2,
            "conditioning_images": 1,
        },
    )

    class Indexed:
        def prepare_sample(self, row, root, config, *, index):
            return index

    class Kwargs:
        def prepare_sample(self, row, root, config, **kwargs):
            return kwargs["index"]

    dataset = ImageEditDataset(args, Kwargs() if accept_kwargs else Indexed())
    assert dataset[1] == 1


def test_indexed_adapter_type_error_is_not_retried(tmp_path):
    row = {"image": "target.png", "edit_image": ["source.png"], "prompt": "edit"}
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row]))
    args = _args(
        prompt_data=str(path),
        i2i_sft_config={
            "data_root": str(tmp_path),
            "dataset_size": 1,
            "conditioning_images": 1,
        },
    )

    class Adapter:
        calls = 0

        def prepare_sample(self, row, root, config, index=0):
            self.calls += 1
            raise TypeError("sample processing failed")

    adapter = Adapter()
    dataset = ImageEditDataset(args, adapter)
    with pytest.raises(TypeError, match="sample processing failed"):
        dataset[0]
    assert adapter.calls == 1
