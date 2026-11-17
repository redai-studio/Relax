# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import sys
import types

import pytest

from relax.utils.training.flops_counter import (
    _DEVICE_FLOPS,
    FlopsCounter,
    SequenceFlopsEstimator,
    get_device_peak_flops,
)


# ---------------------------------------------------------------------------
# Helper: lightweight config object (mirrors verl test pattern)
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# GPU peak FLOPS tests
# Device-name inputs are synthetic fixtures, not measured hardware.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_name, expected_raw_flops",
    [
        ("NVIDIA H100 80GB HBM3", 989e12),  # gitleaks:allow
        ("NVIDIA H800", 989e12),
        ("NVIDIA A100-SXM4-80GB", 312e12),
        ("NVIDIA A800-SXM4-80GB", 312e12),
        ("NVIDIA L40S", 362.05e12),
        ("NVIDIA H20", 148e12),
        ("Ascend910B3", 354e12),
    ],
)
def test_get_device_peak_flops_known_gpus(device_name, expected_raw_flops):
    result = get_device_peak_flops(unit="T", device_name=device_name)
    expected_tflops = expected_raw_flops / 1e12
    assert math.isclose(result, expected_tflops, rel_tol=1e-6), (
        f"Expected {expected_tflops} TFLOPS for {device_name}, got {result}"
    )


def test_get_device_peak_flops_unknown_gpu():
    result = get_device_peak_flops(unit="T", device_name="SomeUnknownGPU-XYZ")
    assert result == float("inf")


def test_get_device_peak_flops_defaults_to_cpu_when_device_properties_unavailable(monkeypatch):
    device_module = types.ModuleType("relax.utils.device")

    def get_device_properties():
        raise AttributeError("module 'torch.cpu' has no attribute 'get_device_properties'")

    device_module.device_module = types.SimpleNamespace(get_device_properties=get_device_properties)
    monkeypatch.setitem(sys.modules, "relax.utils.device", device_module)

    result = get_device_peak_flops(unit="T")

    assert math.isclose(result, _DEVICE_FLOPS["CPU"] / 1e12, rel_tol=1e-6)


def test_get_device_peak_flops_unit_conversion():
    tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100")  # gitleaks:allow
    gflops = get_device_peak_flops(unit="G", device_name="NVIDIA H100")  # gitleaks:allow
    assert math.isclose(tflops * 1000, gflops, rel_tol=1e-6)


def test_device_flops_table_has_common_gpus():
    common_keys = ["H100", "H800", "A100", "A800", "H20"]  # gitleaks:allow
    for key in common_keys:
        assert key in _DEVICE_FLOPS, f"Missing common GPU {key} in _DEVICE_FLOPS"


# ---------------------------------------------------------------------------
# FlopsCounter model FLOPS tests (expected values from verl test suite)
# ---------------------------------------------------------------------------
FLOPS_TEST_CONFIGS = {
    "qwen3_dense": {
        "config": {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [180997438046208 / 1e12, 648394032807936 / 1e12],
    },
    "qwen3_moe": {
        "config": {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "head_dim": 128,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [78593069678592 / 1e12, 306570470621184 / 1e12],
    },
    "deepseek_v3": {
        "config": {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "num_attention_heads": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [848766538088448 / 1e12, 3145850406567936 / 1e12],
    },
}


FLOPS_TEST_VL_CONFIGS = {
    "qwen3_vl": {
        # Exact config of Qwen/Qwen3-VL-4B-Instruct. Note head_dim=128 is set
        # explicitly and num_attention_heads * head_dim (32 * 128 = 4096) !=
        # hidden_size (2560), so head_dim must be read from the config.
        "config": {
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 2560,
                "vocab_size": 151936,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "num_attention_heads": 32,
                "head_dim": 128,
                "intermediate_size": 9728,
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "out_hidden_size": 2560,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
                "deepstack_visual_indexes": [5, 11, 17],
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [110225906466816 / 1e12, 415025911037952 / 1e12],
    },
    "qwen3_5": {
        "config": {
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "layer_types": ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(32)],
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 4096,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [206090394402816 / 1e12, 724521757704192 / 1e12],
    },
    "qwen3_5_moe": {
        "config": {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "layer_types": ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(40)],
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 2048,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [88082762170368 / 1e12, 321470349705216 / 1e12],
    },
}

FLOPS_TEST_OMNI_CONFIGS = {
    "qwen3_omni_moe": {
        "config": {
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 2048,
                    "vocab_size": 3584,
                    "num_hidden_layers": 28,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 28,
                    "head_dim": 128,
                    "moe_intermediate_size": 768,
                    "num_experts_per_tok": 8,
                    "num_experts": 128,
                },
                "vision_config": {
                    "num_heads": 16,
                    "depth": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "out_hidden_size": 3584,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "patch_size": 16,
                    "deepstack_visual_indexes": [8, 16, 24],
                },
                "audio_config": {
                    "d_model": 1280,
                    "num_hidden_layers": 32,
                    "encoder_attention_heads": 20,
                    "encoder_ffn_dim": 5120,
                    "num_mel_bins": 128,
                    "output_dim": 3584,
                    "n_window": 100,
                },
            },
        },
        "batch_seqlens": [512, 1024, 2048],
        "images_seqlens": [512, 1024],
        "audio_seqlens": [500, 1000],
        "expected_tflops": 43060677083136 / 1e12,
    },
}


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_CONFIGS.keys()))
def test_flops_counter_model_estimation(config_name):
    test_data = FLOPS_TEST_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"], test_data["expected_tflops_list"], strict=True
    ):
        # delta_time=1 so returned value = raw TFLOPS
        estimated_tflops, _ = counter.estimate(batch_seqlens, delta_time=1.0)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.2f} TFLOPS, got {estimated_tflops:.2f}"
        )


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_VL_CONFIGS.keys()))
def test_flops_counter_vl_estimation(config_name):
    test_data = FLOPS_TEST_VL_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, images_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"],
        test_data["images_seqlens_list"],
        test_data["expected_tflops_list"],
        strict=True,
    ):
        estimated_tflops, _ = counter.estimate(batch_seqlens, delta_time=1.0, images_seqlens=images_seqlens)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.2f} TFLOPS, got {estimated_tflops:.2f}"
        )


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_OMNI_CONFIGS.keys()))
def test_flops_counter_omni_estimation(config_name):
    test_data = FLOPS_TEST_OMNI_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    estimated_tflops, _ = counter.estimate(
        test_data["batch_seqlens"],
        delta_time=1.0,
        images_seqlens=test_data["images_seqlens"],
        audio_seqlens=test_data["audio_seqlens"],
    )
    assert math.isclose(estimated_tflops, test_data["expected_tflops"], rel_tol=1e-6), (
        f"{config_name}: expected {test_data['expected_tflops']:.2f} TFLOPS, got {estimated_tflops:.2f}"
    )


def test_mfu_calculation():
    peak_tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100 80GB HBM3")  # gitleaks:allow
    achieved_tflops = 200.0
    mfu = achieved_tflops / peak_tflops
    assert 0 < mfu < 1
    assert math.isclose(mfu, 200.0 / 989.0, rel_tol=1e-6)


def test_unknown_model_type_fallback():
    config = Config(
        {
            "model_type": "unknown_model_xyz",
            "hidden_size": 4096,
            "vocab_size": 151936,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
    )
    counter = FlopsCounter(config)
    estimated, _ = counter.estimate([1024, 2048], delta_time=1.0)
    assert estimated > 0


def test_unknown_model_type_missing_fields_returns_zero():
    config = Config({"model_type": "unknown_model_xyz"})
    counter = FlopsCounter(config)
    estimated, _ = counter.estimate([1024, 2048], delta_time=1.0)
    assert estimated == 0


@pytest.fixture(params=["qwen3_5", "qwen3_5_moe"])
def workload_config(request):
    return Config(FLOPS_TEST_VL_CONFIGS[request.param]["config"])


def test_sequence_workload_matches_fixed_text_flops(workload_config):
    expected = {
        "qwen3_5": [55409909956608, 224113540988928],
        "qwen3_5_moe": [21725655859200, 89994999889920],
    }[workload_config.model_type]
    estimator = SequenceFlopsEstimator.from_config(workload_config)
    assert estimator.estimate_per_sample([1024, 4096]) == expected
    assert estimator.estimate([1024, 4096]) == sum(expected)
    assert all(type(work) is int for work in estimator.estimate_per_sample([1024, 4096]))


def test_sequence_workload_matches_legacy_text_formula(workload_config):
    estimator = SequenceFlopsEstimator.from_config(workload_config)
    lengths = [1, 512, 1024, 2048, 32768]
    legacy_rate, _ = FlopsCounter(workload_config).estimate(lengths, delta_time=2.5)
    assert math.isclose(estimator.estimate(lengths), legacy_rate * 2.5 * 1e12, rel_tol=1e-12)


def test_sequence_workload_accepts_flat_text_config(workload_config):
    flat = workload_config.text_config
    flat.model_type = workload_config.model_type + "_text"
    assert SequenceFlopsEstimator.from_config(flat) == SequenceFlopsEstimator.from_config(workload_config)


def test_sequence_workload_empty_batch(workload_config):
    estimator = SequenceFlopsEstimator.from_config(workload_config)
    assert estimator.estimate_per_sample([]) == []
    assert estimator.estimate([]) == 0


def test_sequence_workload_does_not_query_device_or_reinspect_config(monkeypatch, workload_config):
    import relax.utils.training.flops_counter as module

    def unexpected(*args, **kwargs):
        raise AssertionError("Pure workload estimation must not query hardware or inspect signatures")

    monkeypatch.setattr(module, "get_device_peak_flops", unexpected)
    monkeypatch.setattr(module.inspect, "signature", unexpected)
    estimator = SequenceFlopsEstimator.from_config(workload_config)
    before = estimator.estimate([1024, 4096])
    workload_config.text_config.hidden_size = 1
    monkeypatch.setattr(module, "_compute_qwen3_5_text_flops_coefficients", unexpected)
    assert estimator.estimate([1024, 4096]) == before


@pytest.mark.parametrize("length", [0, -1])
def test_sequence_workload_rejects_nonpositive_lengths(length, workload_config):
    with pytest.raises(ValueError, match="positive"):
        SequenceFlopsEstimator.from_config(workload_config).estimate_per_sample([1024, length])


@pytest.mark.parametrize("length", [True, 1.5, float("nan"), float("inf"), "1024", None])
def test_sequence_workload_rejects_noninteger_lengths(length, workload_config):
    with pytest.raises(TypeError, match="integer"):
        SequenceFlopsEstimator.from_config(workload_config).estimate_per_sample([length])


def test_sequence_workload_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        SequenceFlopsEstimator.from_config(Config({"model_type": "unknown_model_xyz"}))


def test_qwen3_explicit_head_width_matches_dense_reporting():
    config = Config(
        dict(
            model_type="qwen3",
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=6,
            num_hidden_layers=2,
            vocab_size=16,
            intermediate_size=4,
        )
    )
    estimator = SequenceFlopsEstimator.from_config(config)
    assert estimator == SequenceFlopsEstimator(6144, 144)
    lengths = [3, 17, 128]
    assert estimator.estimate(lengths) == pytest.approx(FlopsCounter(config).estimate(lengths, 1)[0] * 1e12)
    config.num_key_value_heads = 3
    with pytest.raises(ValueError, match="head geometry"):
        SequenceFlopsEstimator.from_config(config)


def _olmoe_config():
    return Config(
        dict(
            model_type="olmoe",
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=2,
            vocab_size=16,
            intermediate_size=4,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )


def test_olmoe_counts_active_experts_and_matches_reported_flops():
    config = _olmoe_config()
    estimator = SequenceFlopsEstimator.from_config(config)
    assert estimator == SequenceFlopsEstimator(6 * 1088, 96)
    assert estimator.estimate_per_sample([3, 5]) == [20448, 35040]
    counter = FlopsCounter(config)
    achieved, _ = counter.estimate([3, 5], 2.0)
    assert math.isclose(achieved * 2.0 * 1e12, 55488, rel_tol=1e-12)
    config.num_experts = 8
    more_experts = SequenceFlopsEstimator.from_config(config)
    assert more_experts.linear_flops - estimator.linear_flops == 6 * 2 * 8 * 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_experts_per_tok", 5),
        ("num_experts_per_tok", 0),
        ("intermediate_size", -1),
        ("hidden_size", 7),
        ("num_key_value_heads", 3),
    ],
)
def test_olmoe_rejects_invalid_geometry(field, value):
    config = _olmoe_config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        SequenceFlopsEstimator.from_config(config)


@pytest.mark.parametrize("field,value", [("hidden_size", 0), ("num_hidden_layers", -1), ("head_dim", 1.5)])
def test_sequence_workload_rejects_invalid_model_dimensions(field, value, workload_config):
    setattr(workload_config.text_config, field, value)
    with pytest.raises(ValueError, match=field):
        SequenceFlopsEstimator.from_config(workload_config)


def test_sequence_workload_rejects_missing_config_field(workload_config):
    del workload_config.text_config.linear_key_head_dim
    with pytest.raises(ValueError, match="linear_key_head_dim"):
        SequenceFlopsEstimator.from_config(workload_config)


@pytest.mark.parametrize("layer_types", [[], ["full_attention"], ["unknown"] * 32])
def test_sequence_workload_rejects_invalid_layer_types(layer_types, workload_config):
    workload_config.text_config.layer_types = layer_types
    with pytest.raises(ValueError, match="layer_types"):
        SequenceFlopsEstimator.from_config(workload_config)


def test_sequence_workload_interval_matches_explicit_layers(workload_config):
    expected = SequenceFlopsEstimator.from_config(workload_config)
    del workload_config.text_config.layer_types
    workload_config.text_config.full_attention_interval = 4
    assert SequenceFlopsEstimator.from_config(workload_config) == expected
    workload_config.text_config.full_attention_interval = 0
    with pytest.raises(ValueError, match="full_attention_interval"):
        SequenceFlopsEstimator.from_config(workload_config)


def test_sequence_workload_all_gdn_has_no_quadratic_term(workload_config):
    config = workload_config.text_config
    config.layer_types = ["linear_attention"] * config.num_hidden_layers
    estimator = SequenceFlopsEstimator.from_config(workload_config)
    assert estimator.quadratic_flops == 0
    assert estimator.estimate([4096]) == 4 * estimator.estimate([1024])


def test_sequence_workload_rejects_invalid_moe_topk():
    config = Config(FLOPS_TEST_VL_CONFIGS["qwen3_5_moe"]["config"])
    config.text_config.num_experts_per_tok = config.text_config.num_experts + 1
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        SequenceFlopsEstimator.from_config(config)
