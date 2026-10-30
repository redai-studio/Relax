# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.utils.training import microbatch_cost
from relax.utils.training.qwen35_cost import estimate_sample_costs


def tiny_config(*, moe=False, vision=False):
    text = SimpleNamespace(
        model_type="qwen3_5_moe_text" if moe else "qwen3_5_text",
        hidden_size=8,
        vocab_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
    )
    if moe:
        text.num_experts, text.num_experts_per_tok = 4, 2
        text.moe_intermediate_size, text.shared_expert_intermediate_size = 4, 4
    if not vision:
        return text
    return SimpleNamespace(
        model_type="qwen3_5_moe" if moe else "qwen3_5",
        text_config=text,
        vision_config=SimpleNamespace(
            hidden_size=4,
            intermediate_size=8,
            depth=1,
            num_heads=2,
            spatial_merge_size=2,
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            out_hidden_size=8,
            deepstack_visual_indexes=[],
        ),
    )


def test_dense_and_moe_costs_use_actual_gated_projections_and_active_experts():
    batch = {"total_lengths": [2, 4]}
    # Independent hand count for the two tiny layers: 1144 projection/head
    # weights, plus causal attention and three GDN contractions.
    assert estimate_sample_costs(tiny_config(), batch, Namespace()) == [14304, 28992]
    assert estimate_sample_costs(tiny_config(moe=True), batch, Namespace()) == [15264, 30912]


def test_image_video_frame_boundaries_merger_and_independent_freezing():
    config = tiny_config(vision=True)
    baseline = estimate_sample_costs(config, {"total_lengths": [4]}, Namespace())[0]
    image = {"total_lengths": [4], "multimodal_train_inputs": [{"image_grid_thw": [[1, 2, 2]]}]}
    video = {"total_lengths": [4], "multimodal_train_inputs": [{"video_grid_thw": [[2, 2, 2]]}]}
    original = copy.deepcopy(image)
    assert estimate_sample_costs(config, image, Namespace())[0] - baseline == 7680
    assert estimate_sample_costs(config, video, Namespace())[0] - baseline == 2 * 7680
    assert estimate_sample_costs(config, image, Namespace(freeze_vision_model=True))[0] - baseline == 4352
    assert estimate_sample_costs(config, image, Namespace(freeze_vision_projection=True))[0] - baseline == 6912
    args = Namespace(freeze_vision_model=True, freeze_vision_projection=True)
    assert estimate_sample_costs(config, image, args)[0] - baseline == 2816
    assert image == original
    config.vision_config.deepstack_visual_indexes = [0]
    assert estimate_sample_costs(config, image, Namespace())[0] - baseline == 7680 + 2304


def test_frozen_language_still_backpropagates_to_trainable_media():
    config = tiny_config(vision=True)
    batch = {"total_lengths": [4], "multimodal_train_inputs": [{"image_grid_thw": [[1, 2, 2]]}]}
    full = estimate_sample_costs(config, batch, Namespace())[0]
    frozen = estimate_sample_costs(config, batch, Namespace(freeze_language_model=True))[0]
    # Frozen language retains dX, removing only the language dW work.
    assert 0 < full - frozen < 28992 / 3


def test_mtp_executes_each_depth_and_detached_head_keeps_input_backward():
    config = tiny_config()
    batch = {"total_lengths": [4]}
    baseline = estimate_sample_costs(config, batch, Namespace())[0]
    args = Namespace(enable_mtp_training=True, mtp_num_layers=2)
    detached = estimate_sample_costs(config, batch, args)[0]
    args.mtp_detach_paths = ()
    attached = estimate_sample_costs(config, batch, args)[0]
    assert attached - detached == 2 * 2 * 4 * 8 * 16
    assert detached > baseline
    args.mtp_num_layers = 1
    assert estimate_sample_costs(config, batch, args)[0] - baseline == (attached - baseline) / 2
    assert estimate_sample_costs(config, batch, Namespace(mtp_num_layers=3))[0] == baseline


@pytest.mark.parametrize("failure", ["types", "rows", "missing-grid", "cuda-grid", "lora", "mtp", "length"])
def test_unsupported_or_incomplete_cost_inputs_fail_explicitly(failure):
    config = tiny_config(vision=True)
    batch = {"total_lengths": [4]}
    args = Namespace()
    if failure == "types":
        config.text_config.layer_types = ["unknown"]
    elif failure == "rows":
        batch["multimodal_train_inputs"] = []
    elif failure == "missing-grid":
        batch["multimodal_train_inputs"] = [{"pixel_values": object()}]
    elif failure == "cuda-grid":
        batch["multimodal_train_inputs"] = [{"image_grid_thw": SimpleNamespace(device=SimpleNamespace(type="cuda"))}]
    elif failure == "lora":
        args.lora_rank = 8
    elif failure == "mtp":
        args.enable_mtp_training, args.mtp_num_layers = True, 2
        config.text_config.layer_types[-1] = "linear_attention"
    else:
        batch["total_lengths"] = [0]
    with pytest.raises(ValueError):
        estimate_sample_costs(config, batch, args)


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"])
def test_real_hf_config_dispatch_for_each_family(model_type, tmp_path):
    import transformers

    cls = getattr(
        transformers,
        {
            "qwen3_5": "Qwen3_5Config",
            "qwen3_5_text": "Qwen3_5TextConfig",
            "qwen3_5_moe": "Qwen3_5MoeConfig",
            "qwen3_5_moe_text": "Qwen3_5MoeTextConfig",
        }[model_type],
    )
    config = cls()
    config.save_pretrained(tmp_path)
    args = Namespace(microbatch_balance="flops", use_dynamic_batch_size=True, hf_checkpoint=str(tmp_path))
    costs = microbatch_cost.sample_costs(args, {"total_lengths": [64, 128]})
    assert 0 < costs[0] < costs[1]


def test_custom_estimator_registration_and_default_no_config_load(monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(microbatch_cost, "_ESTIMATORS", {})
    monkeypatch.setattr(misc, "get_hf_config", lambda _: SimpleNamespace(model_type="test_family"))
    args = Namespace(use_dynamic_batch_size=True, hf_checkpoint="not-a-real-model")
    batch = {"total_lengths": [2, 3]}
    assert microbatch_cost.sample_costs(args, batch) is None
    args.microbatch_balance = "flops"
    with pytest.raises(ValueError, match="register_sample_cost_estimator"):
        microbatch_cost.sample_costs(args, batch)
    microbatch_cost.register_sample_cost_estimator("test_family", lambda config, data, options: [4, 9])
    assert microbatch_cost.sample_costs(args, batch) == [4, 9]
    with pytest.raises(ValueError, match="already registered"):
        microbatch_cost.register_sample_cost_estimator("test_family", lambda *a: [])


@pytest.mark.parametrize("values", [[1], [0, 1], [-1, 1], [float("inf"), 1], [float("nan"), 1]])
def test_estimator_output_validation(values, monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(microbatch_cost, "_ESTIMATORS", {"test_family": lambda *a: values})
    monkeypatch.setattr(misc, "get_hf_config", lambda _: SimpleNamespace(model_type="test_family"))
    with pytest.raises(ValueError):
        microbatch_cost.sample_costs(
            Namespace(microbatch_balance="flops", use_dynamic_batch_size=True, hf_checkpoint="unused"),
            {"total_lengths": [2, 3]},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_dynamic_batch_size": False},
        {"sft_packing_strategy": "binpack"},
        {"sft_wholepack": True},
        {"fully_async": True},
        {"hybrid": True},
        {"loss_type": "dpo"},
        {"loss_type": "rm"},
        {"train_backend": "fsdp"},
    ],
)
def test_bypassed_schedulers_reject_flops(overrides):
    with pytest.raises(ValueError):
        microbatch_cost.validate_microbatch_balance(
            Namespace(**{"microbatch_balance": "flops", "use_dynamic_batch_size": True, **overrides})
        )
