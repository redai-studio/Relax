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


@pytest.mark.parametrize("objective", ["microbatch_balance", "dp_balance"])
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
        {"moe_aux_loss_coeff": 0.01},
        {"moe_aux_loss_coeff": [0.0, 0.01]},
        {"moe_expert_capacity_factor": 1.25},
        {"moe_router_load_balancing_type": "sinkhorn"},
        {"moe_router_load_balancing_type": ["none", "sinkhorn"]},
        {"moe_router_load_balancing_type": "quantile_balancing"},
    ],
)
def test_bypassed_schedulers_reject_flops(objective, overrides):
    validator = (
        microbatch_cost.validate_dp_balance
        if objective == "dp_balance"
        else microbatch_cost.validate_microbatch_balance
    )
    with pytest.raises(ValueError):
        validator(Namespace(**{objective: "flops", "use_dynamic_batch_size": True, "loss_type": "sft", **overrides}))


def test_default_objectives_preserve_configuration_and_do_not_load_model(monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(misc, "get_hf_config", lambda *_: pytest.fail("Disabled FLOPs must not load model config"))
    args = Namespace()
    batch, metadata = {"total_lengths": [2]}, [{"total_lengths": 2}]
    microbatch_cost.configure_flops_balance(args)
    assert vars(args) == {}
    assert microbatch_cost.sample_costs(args, batch) is None
    assert microbatch_cost.sample_costs(args, batch, for_dp=True) is None
    assert microbatch_cost.add_flops_cost_metadata(args, batch, metadata) is metadata
    assert microbatch_cost.add_flops_cost_metadata(None, batch, metadata) is metadata


@pytest.mark.parametrize("microbatch", ["tokens", "flops"])
@pytest.mark.parametrize("dp", ["tokens", "flops"])
@pytest.mark.parametrize("loss_type", ["sft", "grpo"])
def test_independent_objectives_gate_costs_and_metadata(monkeypatch, microbatch, dp, loss_type):
    from relax.utils import misc

    calls = []

    def config(*_):
        calls.append(1)
        return tiny_config()

    monkeypatch.setattr(misc, "get_hf_config", config)
    args = Namespace(
        microbatch_balance=microbatch,
        dp_balance=dp,
        balance_data=True,
        use_dynamic_batch_size=True,
        loss_type=loss_type,
        hf_checkpoint="unused",
    )
    original = vars(args).copy()
    microbatch_cost.configure_flops_balance(args)
    assert vars(args) == original
    batch, metadata = {"total_lengths": [2, 4]}, [{"total_lengths": 2}, {"total_lengths": 4}]
    published = microbatch_cost.add_flops_cost_metadata(args, batch, metadata)
    if dp == "flops":
        assert [row["flops_cost"] for row in published] == [14304, 28992]
        batch["flops_cost"] = [row["flops_cost"] for row in published]
    else:
        assert published is metadata
        batch["flops_cost"] = [-1, -1]  # Without DP FLOPs, ignore any unrelated cache.
    costs = microbatch_cost.sample_costs(args, batch)
    assert costs == ([14304, 28992] if microbatch == "flops" else None)
    assert len(calls) == int(microbatch == "flops" or dp == "flops")


@pytest.mark.parametrize(
    "cached", [[90.5, 20.25], None, [], [1], [None, 1], [0, 1], [-1, 1], [float("nan"), 1], [float("inf"), 1]]
)
def test_cached_costs_are_validated_and_reusable_without_estimation(monkeypatch, cached):
    from relax.utils import misc

    monkeypatch.setattr(misc, "get_hf_config", lambda *_: pytest.fail("Cache reuse must not load model config"))
    args = Namespace(microbatch_balance="flops", dp_balance="flops", loss_type="sft", use_dynamic_batch_size=True)
    batch = {"total_lengths": [2, 4]}
    if cached is not None:
        batch["flops_cost"] = cached
    if cached != [90.5, 20.25]:
        with pytest.raises(ValueError):
            microbatch_cost.sample_costs(args, batch)
        return
    for _ in range(2):  # Repacking must not consume or mutate cached costs.
        costs = microbatch_cost.sample_costs(args, batch)
        assert costs == cached and costs is not cached
    assert batch == {"total_lengths": [2, 4], "flops_cost": [90.5, 20.25]}


def test_cached_costs_reject_tensors_without_converting_them(monkeypatch):
    import torch

    def fail_conversion(*_):
        pytest.fail("Cache validation must not convert tensor scalars")

    monkeypatch.setattr(torch.Tensor, "__float__", fail_conversion)
    args = Namespace(microbatch_balance="flops", dp_balance="flops", loss_type="sft", use_dynamic_batch_size=True)
    with pytest.raises(ValueError, match="CPU numeric scalars"):
        microbatch_cost.sample_costs(args, {"total_lengths": [2], "flops_cost": [torch.tensor(1.0)]})


def test_queue_payload_preserves_cost_precision():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    pytest.importorskip("tensordict")

    from relax.utils.utils import build_rollout_custom_meta, dict_to_tensordict

    costs = [1_000_000_000_000.125, 1_000_000_000_000.25]
    batch = dict_to_tensordict({"total_lengths": [2, 4], "sample_indices": [8, 2], "flops_cost": costs}, batch_size=2)
    assert batch["flops_cost"].dtype == torch.float64
    assert [row["flops_cost"] for row in build_rollout_custom_meta(batch)] == costs


def test_dp_only_requires_legacy_RL_enablement_and_never_enables_microbatches():
    args = Namespace(dp_balance="flops", microbatch_balance="tokens", use_dynamic_batch_size=True)
    with pytest.raises(ValueError, match="requires --balance-data"):
        microbatch_cost.configure_flops_balance(args)
    assert args.microbatch_balance == "tokens" and not hasattr(args, "balance_data")
    args.balance_data = True
    microbatch_cost.configure_flops_balance(args)
    assert args.microbatch_balance == "tokens"


@pytest.mark.parametrize("option", ["microbatch_balance", "dp_balance"])
def test_invalid_independent_objective_is_rejected(option):
    with pytest.raises(ValueError, match="must be tokens or flops"):
        microbatch_cost.configure_flops_balance(Namespace(**{option: "invalid"}))


@pytest.mark.parametrize("setting", [{"n_samples_per_prompt": 2}, {"sft_image_preprocess_on_rank": True}])
def test_dp_metadata_restrictions_do_not_disable_microbatch_only_mode(setting):
    args = Namespace(
        microbatch_balance="flops",
        dp_balance="tokens",
        loss_type="sft",
        use_dynamic_batch_size=True,
        **setting,
    )
    microbatch_cost.configure_flops_balance(args)
    args.dp_balance = "flops"
    with pytest.raises(ValueError):
        microbatch_cost.configure_flops_balance(args)


def test_producer_cost_metadata_uses_the_same_estimator_and_does_not_mutate(monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(misc, "get_hf_config", lambda _: tiny_config())
    args = Namespace(
        dp_balance="flops",
        microbatch_balance="tokens",
        loss_type="sft",
        use_dynamic_batch_size=True,
        hf_checkpoint="unused",
    )
    batch = {"total_lengths": [2, 4], "flops_cost": [1, 1]}
    metadata = [{"sample_index": 7, "total_lengths": 2}, {"sample_index": 7, "total_lengths": 4}]
    original = copy.deepcopy(metadata)
    result = microbatch_cost.add_flops_cost_metadata(args, batch, metadata)
    assert [row["flops_cost"] for row in result] == [14304, 28992]
    assert [row["sample_index"] for row in result] == [7, 7]
    assert metadata == original
    with pytest.raises(ValueError):
        microbatch_cost.add_flops_cost_metadata(args, batch, metadata[:1])
