# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.cpu.utils.test_arguments_opd_teacher_colocate import (
    _opd_args,
)
from tests.cpu.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def _install_megatron_mtp_backend(monkeypatch, *, supports_repeated_layer: bool) -> None:
    mtp_mod = ModuleType("megatron.core.transformer.multi_token_prediction")
    cfg_mod = ModuleType("megatron.core.transformer.transformer_config")

    if supports_repeated_layer:

        class _TransformerConfig:
            mtp_use_repeated_layer = False

        class _MultiTokenPredictionBlock:
            def _build_layers(self):
                return self.config.mtp_use_repeated_layer

    else:

        class _TransformerConfig:
            pass

        class _MultiTokenPredictionBlock:
            def _build_layers(self, pg_collection):
                return pg_collection

    mtp_mod.MultiTokenPredictionBlock = _MultiTokenPredictionBlock
    cfg_mod.TransformerConfig = _TransformerConfig

    megatron_pkg = sys.modules.get("megatron") or ModuleType("megatron")
    megatron_core_pkg = sys.modules.get("megatron.core") or ModuleType("megatron.core")
    transformer_pkg = sys.modules.get("megatron.core.transformer") or ModuleType("megatron.core.transformer")

    monkeypatch.setitem(sys.modules, "megatron", megatron_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.multi_token_prediction", mtp_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_config", cfg_mod)
    monkeypatch.setattr(megatron_pkg, "core", megatron_core_pkg, raising=False)
    monkeypatch.setattr(megatron_core_pkg, "transformer", transformer_pkg, raising=False)
    monkeypatch.setattr(transformer_pkg, "multi_token_prediction", mtp_mod, raising=False)
    monkeypatch.setattr(transformer_pkg, "transformer_config", cfg_mod, raising=False)


def test_mtp_detach_paths_defaults_to_all_paths(arguments_module):
    parser = argparse.ArgumentParser()
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    arguments_module.get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).mtp_detach_paths == arguments_module._MTP_DETACH_PATHS

    args = parser.parse_args(["--mtp-detach-paths", "lm-head", "embedding"])
    arguments_module._normalize_mtp_detach_paths(args)
    assert args.mtp_detach_paths == ("embedding", "lm-head")

    args = parser.parse_args(["--mtp-detach-paths", "none"])
    arguments_module._normalize_mtp_detach_paths(args)
    assert args.mtp_detach_paths == ()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["embedding"], ("embedding",)),
        (["lm-head", "embedding"], ("embedding", "lm-head")),
        (["backbone", "backbone"], ("backbone",)),
        (["none"], ()),
    ],
)
def test_normalize_mtp_detach_paths(arguments_module, values, expected):
    args = SimpleNamespace(mtp_detach_paths=values)

    arguments_module._normalize_mtp_detach_paths(args)

    assert args.mtp_detach_paths == expected


def test_normalize_mtp_detach_paths_rejects_none_with_another_path(arguments_module):
    with pytest.raises(ValueError, match="none cannot be combined"):
        arguments_module._normalize_mtp_detach_paths(SimpleNamespace(mtp_detach_paths=["none", "embedding"]))


@pytest.mark.parametrize("flag", ["--mtp-detach-main-model", "--no-mtp-detach-main-model"])
def test_removed_mtp_detach_flags_fail_fast_even_when_unknown_args_are_ignored(arguments_module, flag):
    with pytest.raises(ValueError, match="has been removed"):
        arguments_module._reject_removed_mtp_detach_flags([flag])


def _args(arguments_module, **overrides) -> SimpleNamespace:
    defaults = {
        "mtp_only_training": True,
        "loss_type": "sft",
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "lora_rank": 0,
        "sft_chunked_logits": False,
        "overlap_moe_expert_parallel_comm": False,
        "fully_async": False,
        "hybrid": False,
        "mtp_num_layers": None,
        "mtp_use_repeated_layer": False,
        "mtp_loss_scaling_factor": 0.2,
        "enable_mtp_training": False,
        "mtp_detach_paths": arguments_module._MTP_DETACH_PATHS,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_normalize_mtp_only_training_enables_mtp_and_freezes_non_mtp_params(arguments_module):
    args = _args(arguments_module)

    arguments_module._normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is True
    assert args.mtp_detach_paths == arguments_module._MTP_DETACH_PATHS
    assert args.mtp_num_layers == 1
    assert args.only_train_params_name_list == [arguments_module._MTP_ONLY_PARAM_PATTERN]


@pytest.mark.parametrize("depths", [1, 2, 3])
def test_normalize_mtp_only_training_allows_repeated_prediction_depths(arguments_module, monkeypatch, depths):
    _install_megatron_mtp_backend(monkeypatch, supports_repeated_layer=True)
    args = _args(arguments_module, mtp_num_layers=depths, mtp_use_repeated_layer=True)

    arguments_module._normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is True
    assert args.mtp_detach_paths == arguments_module._MTP_DETACH_PATHS
    assert args.mtp_num_layers == depths
    assert args.mtp_use_repeated_layer is True


def test_normalize_mtp_only_training_rejects_repeated_depths_when_backend_lacks_support(arguments_module, monkeypatch):
    _install_megatron_mtp_backend(monkeypatch, supports_repeated_layer=False)

    single_depth_args = _args(arguments_module, mtp_num_layers=1)
    arguments_module._normalize_mtp_only_training_args(single_depth_args)
    assert single_depth_args.mtp_num_layers == 1

    with pytest.raises(ValueError, match="backend support for --mtp-use-repeated-layer"):
        arguments_module._normalize_mtp_only_training_args(
            _args(arguments_module, mtp_num_layers=2, mtp_use_repeated_layer=True)
        )


def _sft_mtp_validate_args(arguments_module, **overrides) -> SimpleNamespace:
    args = _opd_args()
    args.loss_type = "sft"
    args.use_opd = False
    args.prompt_data = "/tmp/train.jsonl"
    args.use_dynamic_batch_size = True
    args.max_tokens_per_gpu = 1024
    args.sft_oversize_strategy = "keep"
    args.mtp_only_training = True
    args.mtp_num_layers = 1
    args.mtp_use_repeated_layer = False
    args.mtp_loss_scaling_factor = 0.2
    args.mtp_detach_paths = arguments_module._MTP_DETACH_PATHS
    args.lora_rank = 0
    args.sft_chunked_logits = False
    args.overlap_moe_expert_parallel_comm = False
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_slime_validate_args_checks_mtp_repeated_depths_after_yaml_override(arguments_module, monkeypatch, tmp_path):
    promote_to_k2_yaml = tmp_path / "promote_to_k2.yaml"
    promote_to_k2_yaml.write_text("mtp_num_layers: 2\nmtp_use_repeated_layer: true\n")

    _install_megatron_mtp_backend(monkeypatch, supports_repeated_layer=False)
    with pytest.raises(ValueError, match="backend support for --mtp-use-repeated-layer"):
        arguments_module.slime_validate_args(
            _sft_mtp_validate_args(
                arguments_module,
                mtp_num_layers=1,
                mtp_use_repeated_layer=False,
                custom_config_path=str(promote_to_k2_yaml),
            )
        )

    _install_megatron_mtp_backend(monkeypatch, supports_repeated_layer=True)
    promoted_args = _sft_mtp_validate_args(
        arguments_module,
        mtp_num_layers=1,
        mtp_use_repeated_layer=False,
        custom_config_path=str(promote_to_k2_yaml),
    )
    arguments_module.slime_validate_args(promoted_args)
    assert promoted_args.mtp_num_layers == 2
    assert promoted_args.mtp_use_repeated_layer is True
    assert promoted_args.enable_mtp_training is True

    disable_repeated_yaml = tmp_path / "disable_repeated.yaml"
    disable_repeated_yaml.write_text("mtp_use_repeated_layer: false\n")
    with pytest.raises(ValueError, match="--mtp-use-repeated-layer"):
        arguments_module.slime_validate_args(
            _sft_mtp_validate_args(
                arguments_module,
                mtp_num_layers=2,
                mtp_use_repeated_layer=True,
                custom_config_path=str(disable_repeated_yaml),
            )
        )


@pytest.mark.parametrize(
    ("override", "expected_flag"),
    [
        ({"loss_type": "policy_loss"}, "--loss-type sft"),
        ({"only_train_params_name_list": ["decoder"]}, "--only-train-params-name-list"),
        ({"freeze_params_name_list": ["output_layer"]}, "--freeze-params-name-list"),
        ({"lora_rank": 8}, "--lora-rank"),
        ({"sft_chunked_logits": True}, "--sft-chunked-logits"),
        ({"overlap_moe_expert_parallel_comm": True}, "--overlap-moe-expert-parallel-comm"),
        ({"fully_async": True}, "--fully-async"),
        ({"hybrid": True}, "--hybrid"),
        ({"mtp_detach_paths": ()}, "--mtp-detach-paths"),
        ({"mtp_num_layers": 0}, "at least one"),
        ({"mtp_num_layers": 2}, "--mtp-use-repeated-layer"),
        ({"mtp_loss_scaling_factor": 0.0}, "greater than 0"),
    ],
)
def test_normalize_mtp_only_training_rejects_unsafe_combinations(arguments_module, override, expected_flag):
    with pytest.raises(ValueError, match=expected_flag):
        arguments_module._normalize_mtp_only_training_args(_args(arguments_module, **override))


def test_normalize_mtp_only_training_is_noop_when_disabled(arguments_module):
    args = _args(arguments_module, mtp_only_training=False)

    arguments_module._normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is False
    assert args.mtp_detach_paths == arguments_module._MTP_DETACH_PATHS
    assert args.mtp_num_layers is None
    assert args.only_train_params_name_list is None
