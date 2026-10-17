# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture
from tests.utils.test_arguments_rloo import _rloo_args


arguments_module = _arguments_module_fixture


def _sc_args(**overrides) -> SimpleNamespace:
    values = dict(
        advantage_estimator="score_centering",
        loss_type="policy_loss",
        use_opd=False,
        # The existing OPD coefficient defaults to 1 even when OPD is disabled.
        opd_kl_coef=1.0,
        rollout_temperature=1.0,
        rollout_top_logprobs_num=128,
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
    )
    values.update(overrides)
    return _rloo_args(**values)


def test_sc_parser_keeps_recording_k_separate_from_sampling_k(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    args = parser.parse_args(["--advantage-estimator", "score_centering", "--rollout-top-logprobs-num", "32"])
    assert args.rollout_top_logprobs_num == 32
    assert args.rollout_top_k == -1
    assert args.score_centering_is == "none"


@pytest.mark.parametrize("fully_async,hybrid", [(False, False), (True, False), (True, True)])
def test_sc_preserves_execution_modes_and_diagnostic_only_mismatch(arguments_module, fully_async, hybrid):
    args = _sc_args(
        fully_async=fully_async,
        hybrid=hybrid,
        partial_rollout=True,
        max_staleness=2,
        context_parallel_size=2,
        dynamic_context_parallel=True,
        get_mismatch_metrics=True,
        custom_tis_function_path=None,
        resource={"actor": [0], "rollout": [0], "actor_fwd": [0]},
    )
    arguments_module.slime_validate_args(args)
    assert args.use_critic is False


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"use_opd": True}, "OPD cannot be enabled together"),
        ({"use_opd": True, "opd_kl_coef": 0.0, "opd_loss_coef": 1.0}, "OPD cannot be enabled together"),
        ({"opd_fkl_entropy_gate": True}, "OPD cannot be enabled together"),
        ({"use_tis": True}, "--use-tis"),
        ({"custom_tis_function_path": "custom.tis"}, "--custom-tis-function-path"),
        ({"use_opsm": True}, "--use-opsm"),
        ({"use_unbiased_kl": True}, "--use-unbiased-kl"),
        ({"custom_pg_loss_reducer_function_path": "custom.reducer"}, "--custom-pg-loss-reducer"),
        ({"rollout_temperature": 0.0}, "positive --rollout-temperature"),
        ({"rollout_top_logprobs_num": 0}, "positive integer"),
        ({"score_centering_tis_clip": float("inf")}, "finite and positive"),
        ({"score_centering_mis_low": 2.0, "score_centering_mis_high": 1.0}, "0 < low <= high"),
        ({"normalize_advantages": True}, "normalize-advantages"),
        ({"rewards_normalization": False}, "rewards normalization"),
        ({"calculate_per_token_loss": False}, "calculate-per-token-loss"),
        ({"kl_coef": 0.1}, "does not support nonzero --kl-coef"),
        ({"multimodal_keys": {"image": "images"}}, "text-only"),
    ],
)
def test_sc_rejects_incompatible_effective_config(arguments_module, overrides, match):
    with pytest.raises(ValueError, match=match):
        arguments_module.slime_validate_args(_sc_args(**overrides))


def test_sc_yaml_cannot_enable_opd_after_validation(arguments_module, tmp_path):
    args = _sc_args()
    arguments_module.slime_validate_args(args)
    config = tmp_path / "override.yaml"
    config.write_text("use_opd: true\nopd_loss_coef: 1.0\nopd_kl_coef: 0.0\n")
    args.custom_config_path = str(config)
    with pytest.raises(ValueError, match="OPD cannot be enabled together"):
        arguments_module.apply_custom_config_overrides(args)
