# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""K3 arithmetic regressions, using the pinned HF architecture and CPU
autograd."""

import copy
from types import SimpleNamespace

import pytest
import torch

from relax.utils.training.flops_counter import FlopsCounter
from relax.utils.training.kimi_k3_flops import _vision_flops


def k3_config():
    # moonshotai/Kimi-K3 revision f831ab66814297da540d832a5235f8e904f29d06.
    return SimpleNamespace(
        model_type="kimi_k3",
        text_config=SimpleNamespace(
            model_type="kimi_linear",
            hidden_size=7168,
            num_hidden_layers=93,
            num_attention_heads=96,
            head_dim=74,
            vocab_size=163840,
            intermediate_size=33792,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            num_experts=896,
            num_experts_per_token=16,
            num_shared_experts=2,
            moe_intermediate_size=3072,
            routed_expert_hidden_size=3584,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            linear_attn_config=dict(
                kda_layers=[i for i in range(1, 93) if i % 4],
                full_attn_layers=[*range(4, 93, 4), 93],
                num_heads=96,
                head_dim=128,
                short_conv_kernel_size=4,
            ),
        ),
        vision_config=SimpleNamespace(
            vt_hidden_size=1024,
            qkv_hidden_size=1536,
            vt_intermediate_size=4096,
            vt_num_hidden_layers=27,
            patch_size=14,
            merge_kernel_size=[2, 2],
            text_hidden_size=7168,
        ),
    )


@pytest.fixture(autouse=True)
def cpu_peak(monkeypatch):
    monkeypatch.setattr("relax.utils.training.flops_counter.get_device_peak_flops", lambda **_: 2250.0)


def flops(config=None, lengths=(128,), **kwargs):
    return FlopsCounter(config or k3_config()).estimate(lengths, 1.0, **kwargs)[0] * 1e12


@pytest.mark.parametrize("lengths", [[], [9, 5], [8192], [131072], [262144]])
@pytest.mark.parametrize("forward_only", [False, True])
def test_k3_full_config_matches_independently_counted_gemm_shapes(lengths, forward_only):
    # Counted from Bridge's concrete weight shapes, NOT the dense fallback:
    # active projection/conv weights = 104,184,266,752 (one vocab output);
    # KDA's three state contractions = 651,165,696 FLOPs/token;
    # 24 causal MLA layers = 737,280 FLOPs per squared sequence length.
    expected_forward = 209_019_699_200 * sum(lengths) + 737_280 * sum(s * s for s in lengths)
    assert flops(lengths=lengths, forward_only=forward_only) == pytest.approx(
        expected_forward * (1 if forward_only else 3)
    )


def test_k3_moe_active_experts_shared_experts_and_latent_width_are_counted():
    base = k3_config()
    top1 = copy.deepcopy(base)
    top1.text_config.num_experts_per_token = 1
    assert flops(base) - flops(top1) == pytest.approx(6 * 128 * 92 * 3 * 3584 * 3072 * 15)
    shared0 = copy.deepcopy(base)
    shared0.text_config.num_shared_experts = 0
    assert flops(base) - flops(shared0) == pytest.approx(6 * 128 * 92 * 3 * 7168 * 3072 * 2)
    wider = copy.deepcopy(base)
    wider.text_config.routed_expert_hidden_size = 4096
    assert flops(wider) - flops(base) == pytest.approx(6 * 128 * 92 * 512 * (2 * 7168 + 3 * 3072 * 16))


def test_k3_all_kda_is_linear_and_mla_uses_per_sample_squares():
    config = k3_config()
    config.text_config.linear_attn_config["kda_layers"] = list(range(1, 94))
    assert flops(config, [2048]) == pytest.approx(2 * flops(config, [1024]))
    config.text_config.linear_attn_config["kda_layers"] = []
    assert flops(config, [2048]) - flops(config, [1024, 1024]) == pytest.approx(
        3 * 93 * 96 * 320 * (2048**2 - 2 * 1024**2)
    )


def test_k3_reduced_layer_config_does_not_use_fallback():
    config = k3_config()
    config.text_config.num_hidden_layers = 5
    config.text_config.num_experts = 128
    config.text_config.linear_attn_config["kda_layers"] = [1, 2, 3]
    reduced = flops(config)
    assert 0 < reduced < flops()
    config.text_config.linear_attn_config["kda_layers"] = [1, 2, 3, 90]
    assert flops(config) == reduced  # stale out-of-range placement isn't a real layer


@pytest.mark.parametrize("freeze_encoder", [False, True])
@pytest.mark.parametrize("freeze_projector", [False, True])
def test_k3_images_use_independent_encoder_and_projector_gradients(freeze_encoder, freeze_projector):
    config = k3_config()
    opts = dict(freeze_vision_model=freeze_encoder, freeze_vision_projection=freeze_projector)
    # Official shapes for a 4x6 grid: 24 patches, 6 spatially merged tokens.
    patch = 2 * 24 * 588 * 1024
    blocks = 27 * (8 * 24 * 1024 * 1536 + 4 * 24 * 1024 * 4096 + 4 * 24**2 * 1536)
    p1, p2 = 2 * 6 * 4096**2, 2 * 6 * 4096 * 7168
    factors = {(True, True): (1, 1), (True, False): (2, 3), (False, True): (2, 2), (False, False): (3, 3)}
    f1, f2 = factors[freeze_encoder, freeze_projector]
    expected = patch * (1 if freeze_encoder else 2) + blocks * (1 if freeze_encoder else 3) + f1 * p1 + f2 * p2
    added = flops(config, image_grid_thw=[(1, 4, 6)], **opts) - flops(config, **opts)
    assert added == pytest.approx(expected)
    assert flops(config, image_grid_thw=[(1, 4, 6)] * 8, **opts) - flops(config, **opts) == pytest.approx(8 * expected)
    forward = flops(config, image_grid_thw=[(1, 4, 6)], forward_only=True, **opts)
    assert forward == pytest.approx(flops(config, forward_only=True) + patch + blocks + p1 + p2)


def test_k3_temporal_grid_attends_before_pooling_and_legacy_still_image_matches():
    config = k3_config()
    base = flops(config, forward_only=True)
    image = flops(config, image_grid_thw=[(1, 4, 6)], forward_only=True) - base
    video = flops(config, image_grid_thw=[(3, 4, 6)], forward_only=True) - base
    # Triple N scales linear encoder terms 3x, attention 9x, projector stays 1x.
    attention = 27 * 4 * 24**2 * 1536
    projector = 2 * 6 * (4096**2 + 4096 * 7168)
    assert video - 3 * image == pytest.approx(6 * attention - 2 * projector)
    assert flops(config, images_seqlens=[24]) == flops(config, image_grid_thw=[(1, 4, 6)])


@pytest.mark.parametrize("freeze_encoder", [False, True])
@pytest.mark.parametrize("freeze_projector", [False, True])
def test_projector_formula_matches_cpu_autograd_mm_counts(freeze_encoder, freeze_projector):
    from torch.utils.flop_counter import FlopCounterMode

    # Real two-linear topology of PatchMergerMLPV2, scaled down for CPU.
    vision = SimpleNamespace(
        vt_hidden_size=2,
        qkv_hidden_size=2,
        vt_intermediate_size=8,
        vt_num_hidden_layers=0,
        patch_size=1,
        merge_kernel_size=[2, 2],
        text_hidden_size=3,
    )
    pixels = torch.randn(4, 3)
    patch = torch.nn.Linear(3, 2, bias=False).requires_grad_(not freeze_encoder)
    projector = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=False), torch.nn.GELU(), torch.nn.Linear(8, 3, bias=False)
    ).requires_grad_(not freeze_projector)
    with FlopCounterMode(display=False) as counter:
        output = projector(patch(pixels).reshape(1, 8))
        if output.requires_grad:
            output.sum().backward()
    estimated = _vision_flops(
        vision,
        [(1, 2, 2)],
        forward_only=False,
        freeze_vision_model=freeze_encoder,
        freeze_vision_projection=freeze_projector,
    )
    assert estimated == counter.get_total_flops()


def test_kda_recurrent_contractions_match_cpu_matmul_counts():
    from torch.utils.flop_counter import FlopCounterMode

    heads, dim, length = 2, 4, 3
    state = torch.zeros(heads, dim, dim)
    q, k, v = torch.randn(3, length, heads, dim)
    with FlopCounterMode(display=False) as counter:
        for t in range(length):
            error = v[t] - (k[t].unsqueeze(-2) @ state).squeeze(-2)
            state = state + k[t].unsqueeze(-1) @ error.unsqueeze(-2)
            output = (q[t].unsqueeze(-2) @ state).squeeze(-2)
    assert output.shape == (heads, dim)
    assert counter.get_total_flops() == 6 * length * heads * dim * dim


def test_k3_frozen_language_still_propagates_input_gradients_to_vision():
    config = k3_config()
    frozen = dict(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=True)
    assert flops(config, **frozen) == flops(config, forward_only=True)
    assert flops(config, image_grid_thw=[(1, 4, 6)], **frozen) == flops(
        config, image_grid_thw=[(1, 4, 6)], forward_only=True
    )
    frozen["freeze_vision_projection"] = False
    assert flops(config, image_grid_thw=[(1, 4, 6)], **frozen) > flops(
        config, image_grid_thw=[(1, 4, 6)], forward_only=True
    )


@pytest.mark.parametrize("layout", ["list", "dict", "numpy"])
def test_k3_original_temporal_grid_survives_metadata_extraction(layout):
    import numpy as np

    from relax.utils.utils import _extract_image_grids, _extract_images_seqlens

    grid = [[3, 4, 6], [1, 8, 2]]
    values = np.array(grid) if layout == "numpy" else torch.tensor(grid)
    metadata = {"image_grid_thw": values}
    if layout == "list":
        metadata = [None, {}, metadata]
    assert _extract_image_grids(metadata) == [(3, 4, 6), (1, 8, 2)]
    assert _extract_images_seqlens(metadata) == [24, 24, 24, 16]


def _logged_metrics(monkeypatch, counter, **args):
    from relax.utils.training import train_metric_utils

    timer = SimpleNamespace(
        seq_lens=[8192],
        images_seqlens=[24, 24, 24],
        image_grid_thw=[(3, 4, 6)],
        response_lens=[100],
        log_dict=lambda: {"actor_train": 2.0, "log_probs": 1.0, "ref_log_probs": 1.0},
        reset=lambda: None,
    )
    logs = {}
    monkeypatch.setattr(train_metric_utils, "Timer", lambda: timer)
    monkeypatch.setattr(train_metric_utils.tracking_utils, "log", lambda _, values, **kw: logs.update(values))
    train_metric_utils.log_perf_data_raw(
        0, SimpleNamespace(wandb_always_use_train_step=False, **args), True, counter, world_size=128
    )
    return logs


def test_k3_metric_pipeline_reports_per_gpu_rates_and_separate_forward_work(monkeypatch):
    config = k3_config()
    logs = _logged_metrics(monkeypatch, FlopsCounter(config), freeze_vision_model=True, freeze_vision_projection=True)
    train = flops(config, [8192], image_grid_thw=[(3, 4, 6)], freeze_vision_model=True, freeze_vision_projection=True)
    forward = flops(config, [8192], image_grid_thw=[(3, 4, 6)], forward_only=True)
    assert logs["perf/actor_train_tflops"] == pytest.approx(train / 1e12 / 128 / 2)
    assert logs["perf/log_probs_tflops"] == pytest.approx(forward / 1e12 / 128)
    assert logs["perf/ref_log_probs_tflops"] == logs["perf/log_probs_tflops"]
    assert logs["perf/mfu/actor_train"] == pytest.approx(logs["perf/actor_train_tflops"] / 2250.0)
    assert forward > train / 3  # Frozen vision still executes in the forward pass.


@pytest.mark.parametrize(
    "options",
    [
        dict(lora_rank=8),
        dict(freeze_params_name_list=["linear_fc1"]),
        dict(only_train_params_name_list=["linear_fc2"]),
    ],
)
def test_k3_unsupported_trainability_omits_mfu_but_keeps_throughput(monkeypatch, options):
    from unittest.mock import Mock

    warning = Mock()
    monkeypatch.setattr("relax.utils.training.flops_counter.logger.warning", warning)
    counter = FlopsCounter(k3_config())
    for _ in range(2):
        logs = _logged_metrics(monkeypatch, counter, **options)
        assert "perf/actor_train_tflops" not in logs
        assert "perf/log_probs_tflops" not in logs
        assert not any(key.startswith("perf/mfu/") for key in logs)
        assert logs["perf/actor_train_tok_per_s"] == 4096
    warning.assert_called_once()


def test_legacy_models_keep_train_and_forward_metric_values(monkeypatch):
    from tests.utils.test_flops_counter import FLOPS_TEST_CONFIGS, Config

    config = Config(FLOPS_TEST_CONFIGS["qwen3_dense"]["config"])
    counter = FlopsCounter(config)
    train, _ = counter.estimate([8192], 1.0)
    logs = _logged_metrics(monkeypatch, counter, freeze_vision_model=True, lora_rank=8)
    assert logs["perf/actor_train_tflops"] == pytest.approx(train / 128 / 2)
    assert logs["perf/log_probs_tflops"] == pytest.approx(train / 3 / 128)
