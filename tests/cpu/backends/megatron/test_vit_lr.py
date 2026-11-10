# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.training.arguments")

from megatron.core.optimizer import OptimizerConfig  # noqa: E402
from megatron.core.optimizer.optimizer import param_group_identifier_keys  # noqa: E402

from relax.backends.megatron.model import (  # noqa: E402
    _build_optimizer_config_overrides,
    _is_vit_parameter_name,
    _validate_vit_lr_trainable_params,
)
from relax.utils.arguments import get_slime_extra_args_provider  # noqa: E402


def test_vit_lr_argument_is_optional_and_parses_value():
    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).vit_lr is None
    assert parser.parse_args(["--vit-lr", "1e-5"]).vit_lr == pytest.approx(1e-5)


@pytest.mark.parametrize(
    "name",
    [
        "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
        "module.visual.blocks.0.mlp.linear_fc1.adapter.linear_out.weight",
        "module.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
        "module.image_encoder.blocks.0.attn.qkv.weight",
        "module.vision_model.decoder.layers.0.self_attention.projection.weight",
    ],
)
def test_vit_lr_matches_vision_encoder_parameters(name):
    assert _is_vit_parameter_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "module.module.vision_model.merger.linear_fc1.adapter.linear_in.weight",
        "module.module.vision_model.decoder.deepstack_merger_list.0.linear_fc1.adapter.linear_in.weight",
        "module.visual.multi_modal_projector.linear.weight",
        "module.vision_tower.projector.linear.weight",
        "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
    ],
)
def test_vit_lr_excludes_projection_and_language_parameters(name):
    assert not _is_vit_parameter_name(name)


def test_vit_lr_scales_min_lr_and_preserves_standard_overrides():
    config = OptimizerConfig(lr=4.5e-5, min_lr=1e-6)
    overrides = _build_optimizer_config_overrides(SimpleNamespace(vit_lr=1e-5), config)

    matches = [override for key, override in overrides.items() if key.with_name_predicate]

    assert any(override.get("wd_mult") == 0.0 for override in overrides.values())
    assert {
        "lr_mult": pytest.approx(1e-5 / 4.5e-5),
        "max_lr": 1e-5,
        "min_lr": pytest.approx(1e-6 * (1e-5 / 4.5e-5)),
    } in matches
    assert "lr_mult" in param_group_identifier_keys


def test_vit_lr_is_opt_in():
    config = OptimizerConfig(lr=4.5e-5, min_lr=1e-6)
    overrides = _build_optimizer_config_overrides(SimpleNamespace(vit_lr=None), config)

    assert not any(key.with_name_predicate and "max_lr" in override for key, override in overrides.items())


def test_vit_lr_rejects_nonzero_warmup_init():
    config = OptimizerConfig(lr=4.5e-5, min_lr=1e-6)

    with pytest.raises(ValueError, match="requires --lr-warmup-init 0"):
        _build_optimizer_config_overrides(SimpleNamespace(vit_lr=1e-5, lr_warmup_init=1e-7), config)


def test_vit_lr_requires_a_trainable_vision_parameter():
    language_only = torch.nn.Module()
    language_only.decoder = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="did not match any trainable vision-encoder parameters"):
        _validate_vit_lr_trainable_params(SimpleNamespace(vit_lr=1e-5), [language_only])

    multimodal = torch.nn.Module()
    multimodal.vision_model = torch.nn.Linear(2, 2)
    _validate_vit_lr_trainable_params(SimpleNamespace(vit_lr=1e-5), [multimodal])
