# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent arithmetic and legacy regression for the P2 work interface."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.training.flops_counter import _estimate_qwen3_5_absolute_flops, _estimate_qwen3_5_flops
from relax.utils.training.flops_estimator import RelaxFlopsEstimator, SampleMeta


CONFIGS = Path(__file__).parent / "configs"


def tiny_config(*, moe: bool = False, layers: tuple[str, ...] = ("full_attention", "linear_attention")) -> dict:
    text = dict(
        hidden_size=8,
        vocab_size=16,
        num_hidden_layers=len(layers),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        layer_types=list(layers),
    )
    if moe:
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=6, shared_expert_intermediate_size=3)
    else:
        text["intermediate_size"] = 12
    return {"model_type": "qwen3_5_moe" if moe else "qwen3_5", "text_config": text}


def namespace(value: object) -> object:
    return (
        SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value
    )


def vision_config() -> dict:
    return dict(
        num_heads=2,
        depth=1,
        hidden_size=4,
        intermediate_size=6,
        out_hidden_size=8,
        spatial_merge_size=1,
        in_channels=3,
        temporal_patch_size=1,
        patch_size=2,
    )


@pytest.mark.parametrize("lengths", [(1,), (3,), (1, 3), (512, 1024)])
@pytest.mark.parametrize("moe,linear", [(False, 8136), (True, 9480)])
def test_training_numerator_matches_independent_hand_arithmetic(
    lengths: tuple[int, ...], moe: bool, linear: int
) -> None:
    # H8/V16: hybrid attention weights494, dense MLP576 (MoE800), embed/head256.
    # 6*(494+576+256)=7956 plus GDN recurrence180 gives8136; causal term48*sum(L²).
    expected = linear * sum(lengths) + 48 * sum(length * length for length in lengths)
    assert (
        RelaxFlopsEstimator(tiny_config(moe=moe), mtp_training_enabled=False).estimate_sample(SampleMeta(lengths))
        == expected
    )


@pytest.mark.parametrize(
    "layers,expected",
    [(("full_attention", "full_attention"), 25056), (("linear_attention", "linear_attention"), 24624)],
)
def test_full_attention_and_gdn_have_independent_limits(layers: tuple[str, ...], expected: int) -> None:
    assert (
        RelaxFlopsEstimator(tiny_config(layers=layers), mtp_training_enabled=False).estimate_sample(SampleMeta((3,)))
        == expected
    )


def test_segments_and_microbatch_additivity_do_not_merge_attention_boundaries() -> None:
    estimator = RelaxFlopsEstimator(tiny_config(), mtp_training_enabled=False)
    a, b = SampleMeta((1,)), SampleMeta((3,))
    assert estimator.estimate_microbatch([a, b]) == estimator.estimate_sample(SampleMeta((1, 3))) == 33024
    assert estimator.estimate_sample(SampleMeta((4,))) - 33024 == 288
    assert estimator.estimate_microbatch([]) == 0


def test_vision_patch_sequences_and_deepstack_have_independent_oracle() -> None:
    config = tiny_config()
    config["vision_config"] = vision_config()
    meta = SampleMeta((2,), (4,))
    # ViT parameters: patch48 + MLP48 + attention64 + merger48 =208.
    # 6*208*4 + 12*(head_dim2)*(heads2)*4² =5760, added to text16464.
    assert RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(meta) == 22224
    config["vision_config"]["deepstack_visual_indexes"] = [0]
    assert RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(meta) == 22224 + 6 * 48 * 4
    estimator = RelaxFlopsEstimator(config, mtp_training_enabled=False)
    assert estimator.estimate_microbatch([meta, meta]) == 2 * estimator.estimate_sample(meta)


@pytest.mark.parametrize("model", ["Qwen3.5-27B", "Qwen3.5-35B-A3B"])
@pytest.mark.parametrize("delta_time", [0.25, 1.0, 3.7])
@pytest.mark.parametrize("images", [(), (256, 512)])
def test_pinned_real_model_configs_match_old_counter_scope(
    model: str, delta_time: float, images: tuple[int, ...]
) -> None:
    config = json.loads((CONFIGS / (model + ".json.txt")).read_text())
    meta = SampleMeta((512, 1024, 2048), images)
    absolute = RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(meta)
    rate = _estimate_qwen3_5_flops(namespace(config), 3584, meta.sequence_lengths, delta_time, images_seqlens=images)
    assert absolute == pytest.approx(rate * delta_time * 1e12, rel=1e-14)


def test_pinned_config_artifacts_have_exact_upstream_hashes() -> None:
    for source in json.loads((CONFIGS / "sources.json").read_text()):
        path = CONFIGS / (source["model"].split("/")[-1] + ".json.txt")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["artifact_sha256"]
        assert path.read_bytes().endswith(b"\n")
        assert hashlib.sha256(path.read_bytes()[:-1]).hexdigest() == source["upstream_sha256"]
        assert source["revision"] in source["config_url"]


@pytest.mark.parametrize("images", [None, [], [4]])
def test_legacy_wrapper_keeps_visual_kwargs_and_normalization(images: list[int] | None) -> None:
    config = tiny_config()
    config["vision_config"] = vision_config()
    expected = 8184 + (5760 if images else 0)
    assert _estimate_qwen3_5_flops(namespace(config), 1, [1], 2.0, images_seqlens=images) == expected / 2 / 1e12
    config["vision_config"] = None
    assert _estimate_qwen3_5_flops(namespace(config), 1, [1], 2.0, images_seqlens=images) == 8184 / 2 / 1e12
    with pytest.raises(ZeroDivisionError):
        _estimate_qwen3_5_flops(namespace(config), 1, [1], 0)


def test_config_and_metadata_snapshots_ignore_later_caller_mutations() -> None:
    config = tiny_config()
    estimator = RelaxFlopsEstimator(config, mtp_training_enabled=False)
    lengths = [1, 3]
    meta = SampleMeta(lengths)
    config["text_config"]["hidden_size"] = 1000
    config["text_config"]["layer_types"].clear()
    lengths.append(1000)
    assert estimator.estimate_sample(meta) == 33024
    assert estimator.compute_scope == "training"


def test_explicit_layer_interval_is_equivalent_without_guessing_defaults() -> None:
    config = tiny_config(layers=("linear_attention", "full_attention"))
    expected = RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(SampleMeta((3,)))
    del config["text_config"]["layer_types"]
    config["text_config"]["full_attention_interval"] = 2
    assert RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(SampleMeta((3,))) == expected
    del config["text_config"]["full_attention_interval"]
    with pytest.raises(ValueError, match="full_attention_interval"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


@pytest.mark.parametrize("scope", ["forward", "backward", "unknown", None])
def test_unvalidated_compute_scopes_are_rejected(scope: str) -> None:
    with pytest.raises(ValueError, match="training"):
        RelaxFlopsEstimator(tiny_config(), compute_scope=scope, mtp_training_enabled=False)


@pytest.mark.parametrize("length", [0, -1, 1.5, True, float("nan"), float("inf"), "10"])
def test_invalid_lengths_are_rejected(length: object) -> None:
    with pytest.raises(ValueError):
        SampleMeta((length,))


@pytest.mark.parametrize("lengths", [(), "12", None])
def test_empty_and_nonsequence_sample_metadata_is_rejected(lengths: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SampleMeta(lengths)


def test_extreme_cost_cannot_silently_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        RelaxFlopsEstimator(tiny_config(), mtp_training_enabled=False).estimate_sample(SampleMeta((10**400,)))


@pytest.mark.parametrize(
    "field",
    ["hidden_size", "vocab_size", "head_dim", "num_hidden_layers", "linear_num_value_heads", "intermediate_size"],
)
@pytest.mark.parametrize("value", [None, 0, -1, True, float("nan"), 1.2])
def test_invalid_config_scalars_are_rejected(field: str, value: object) -> None:
    config = tiny_config()
    config["text_config"][field] = value
    with pytest.raises(ValueError, match=field):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


@pytest.mark.parametrize("field", ["head_dim", "hidden_size", "linear_conv_kernel_dim", "intermediate_size"])
def test_missing_config_is_not_filled_with_guessed_dimensions(field: str) -> None:
    config = tiny_config()
    del config["text_config"][field]
    with pytest.raises(ValueError, match=field):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


@pytest.mark.parametrize("layers", [[], ["full_attention"], ["linear_attention", "unknown"], "full_attention"])
def test_invalid_layer_structure_is_rejected(layers: object) -> None:
    config = tiny_config()
    config["text_config"]["layer_types"] = layers
    with pytest.raises(ValueError, match="layer_types"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


def test_unsupported_model_placement_and_moe_fields_are_rejected() -> None:
    config = tiny_config()
    for changes in ({"model_type": "qwen2"}, {"tie_word_embeddings": True}):
        with pytest.raises(ValueError):
            RelaxFlopsEstimator({**config, **changes}, mtp_training_enabled=False)
    for field, value in (("mlp_only_layers", [0]), ("num_experts", 4), ("num_key_value_heads", 3)):
        broken = copy.deepcopy(config)
        broken["text_config"][field] = value
        with pytest.raises(ValueError):
            RelaxFlopsEstimator(broken, mtp_training_enabled=False)
    config = tiny_config(moe=True)
    config["text_config"]["num_experts_per_tok"] = 5
    with pytest.raises(ValueError, match="exceed"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)
    del config["text_config"]["num_experts_per_tok"]
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


def test_image_metadata_requires_valid_vision_configuration() -> None:
    with pytest.raises(ValueError, match="vision_config"):
        RelaxFlopsEstimator(tiny_config(), mtp_training_enabled=False).estimate_sample(SampleMeta((1,), (4,)))
    config = tiny_config()
    config["vision_config"] = vision_config()
    config["vision_config"]["deepstack_visual_indexes"] = [0, 0]
    with pytest.raises(ValueError, match="Duplicate"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)
    config["vision_config"]["deepstack_visual_indexes"] = [1]
    with pytest.raises(ValueError, match="Invalid"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


def test_import_and_estimation_do_not_query_gpu_or_import_training_services(monkeypatch: pytest.MonkeyPatch) -> None:
    from relax.utils.training import flops_counter

    monkeypatch.setattr(flops_counter, "get_device_peak_flops", lambda **kwargs: pytest.fail("Device query"))
    assert RelaxFlopsEstimator(tiny_config(), mtp_training_enabled=False).estimate_sample(SampleMeta((1,))) == 8184
    code = (
        "import sys; import relax.utils.training.flops_estimator; "
        "assert not {'torch', 'ray', 'megatron', 'transformers'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("field", ["mtp_num_hidden_layers", "num_nextn_predict_layers"])
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("moe", [False, True])
def test_mtp_structure_metadata_is_not_the_training_switch(field: str, count: int, moe: bool) -> None:
    config = tiny_config(moe=moe)
    config["text_config"][field] = count
    assert RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(SampleMeta((3,))) == (
        (9480 if moe else 8136) * 3 + 48 * 9
    )


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("moe", [False, True])
def test_mtp_enabled_training_never_silently_falls_back_to_main_model(count: int, moe: bool) -> None:
    config = tiny_config(moe=moe)
    config["text_config"]["mtp_num_hidden_layers"] = count
    with pytest.raises(ValueError, match="MTP-enabled training FLOPs are not supported"):
        RelaxFlopsEstimator(config, mtp_training_enabled=True)


@pytest.mark.parametrize("value", [None, "false", 0, 1, 0.0, [], {}])
def test_mtp_mode_is_a_strict_explicit_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="explicit bool"):
        RelaxFlopsEstimator(tiny_config(), mtp_training_enabled=value)


def test_missing_mtp_runtime_mode_is_rejected() -> None:
    with pytest.raises(TypeError, match="mtp_training_enabled"):
        RelaxFlopsEstimator(tiny_config())


@pytest.mark.parametrize(
    "model,short,long,ratio",
    [
        ("Qwen3.5-27B", 166443136057344, 1366179704733696, 8.20808678),
        ("Qwen3.5-35B-A3B", 21725655859200, 188236336988160, 8.66424186),
    ],
)
def test_official_mtp_metadata_configs_preserve_non_mtp_main_model_oracle(
    model: str, short: int, long: int, ratio: float
) -> None:
    config = json.loads((CONFIGS / f"{model}.json.txt").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    estimator = RelaxFlopsEstimator(config, mtp_training_enabled=False)
    assert estimator.estimate_sample(SampleMeta((1024,))) == short
    assert estimator.estimate_sample(SampleMeta((8192,))) == long
    assert long / short == pytest.approx(ratio, abs=5e-9)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("level", ["text", "flat_text", "root_duplicate"])
@pytest.mark.parametrize(
    "field,value", [("attn_output_gate", True), ("attention_bias", False), ("hidden_act", "silu"), ("mlp_bias", False)]
)
def test_supported_architecture_flags_and_verified_defaults(moe: bool, level: str, field: str, value: object) -> None:
    config = tiny_config(moe=moe)
    baseline = RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(SampleMeta((3,)))
    if level == "flat_text":
        config = {"model_type": config["model_type"] + "_text", **config["text_config"]}
    target = config["text_config"] if level == "text" else config
    target[field] = value
    assert RelaxFlopsEstimator(config, mtp_training_enabled=False).estimate_sample(SampleMeta((3,))) == baseline


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("level", ["text", "flat_text", "root_duplicate"])
@pytest.mark.parametrize(
    "field,value",
    [("attn_output_gate", value) for value in (False, 1, None, "true")]
    + [("attention_bias", value) for value in (True, 0, None, "false")]
    + [("hidden_act", value) for value in ("gelu", "relu", None, True)]
    + [("mlp_bias", value) for value in (True, 0, None, "false")],
)
def test_architecture_variants_and_ambiguous_root_flags_fail_closed(
    moe: bool, level: str, field: str, value: object
) -> None:
    config = tiny_config(moe=moe)
    if level == "flat_text":
        config = {"model_type": config["model_type"] + "_text", **config["text_config"]}
    target = config["text_config"] if level == "text" else config
    target[field] = value
    with pytest.raises(ValueError, match=field):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


def test_vision_validation_is_eager_and_separate_from_text_activation() -> None:
    config = tiny_config()
    config["vision_config"] = {"hidden_size": 4}
    with pytest.raises(ValueError, match="num_heads"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)
    config["vision_config"] = {**vision_config(), "hidden_act": "gelu_pytorch_tanh"}
    estimator = RelaxFlopsEstimator(config, mtp_training_enabled=False)
    assert estimator.estimate_sample(SampleMeta((2,), (4,))) == 22224
    assert estimator.estimate_sample(SampleMeta((1,))) == 8184


@pytest.mark.parametrize("activation", ["silu", "relu", None, True])
def test_unsupported_vision_activation_is_rejected_before_text_only_estimation(activation: object) -> None:
    config = tiny_config()
    config["vision_config"] = {**vision_config(), "hidden_act": activation}
    with pytest.raises(ValueError, match="vision hidden_act"):
        RelaxFlopsEstimator(config, mtp_training_enabled=False)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("images", [None, [], [4]])
def test_absolute_helper_returns_int_without_changing_numerator_or_empty_images(moe: bool, images: object) -> None:
    config = tiny_config(moe=moe)
    config["vision_config"] = vision_config()
    value = _estimate_qwen3_5_absolute_flops(namespace(config), 3, [3], images_seqlens=images)
    expected = (9480 if moe else 8136) * 3 + 48 * 9 + (5760 if images else 0)
    assert type(value) is int
    assert value == expected
    assert _estimate_qwen3_5_flops(namespace(config), 3, [3], 2.5, images_seqlens=images) == expected / 2.5 / 1e12
