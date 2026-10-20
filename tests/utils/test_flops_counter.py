# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import sys
import types

import pytest

from relax.utils.training.flops_counter import _DEVICE_FLOPS, FlopsCounter, get_device_peak_flops


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
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_name, expected_raw_flops",
    [
        ("NVIDIA H100 80GB HBM3", 989e12),
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
    tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100")
    gflops = get_device_peak_flops(unit="G", device_name="NVIDIA H100")
    assert math.isclose(tflops * 1000, gflops, rel_tol=1e-6)


def test_device_flops_table_has_common_gpus():
    common_keys = ["H100", "H800", "A100", "A800", "H20"]
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
    peak_tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100 80GB HBM3")
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
