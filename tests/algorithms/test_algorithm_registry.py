# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Registry contracts and the algorithm declarations migrated from main."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from relax.algorithms import get_algorithm, list_algorithm_names
from relax.algorithms.spec import ALGORITHM_SPECS, AlgorithmSpec


# Transcribed from main 5cec8ca1569801d835a56ac86af19babd83caa82, not generated
# from AlgorithmSpec. Numerical adapter parity is tested in the stage tests.
MAIN_ROUTES = {
    "grpo": ("group_mean_std", "grpo_broadcast", "ppo_clip"),
    "gspo": ("group_mean_std", "grpo_broadcast", "ppo_clip"),
    "sapo": ("group_mean_std", "grpo_broadcast", "sapo"),
    "cispo": ("group_mean_std", "grpo_broadcast", "cispo"),
    "m2po": ("none", "grpo_broadcast", "m2po"),
    "rloo": ("group_leave_one_out", "grpo_broadcast", "rloo"),
    "ppo": ("none", "gae", "ppo_clip"),
    "reinforce_plus_plus": ("none", "reinforce_plus_plus", "ppo_clip"),
    "reinforce_plus_plus_baseline": ("group_mean", "reinforce_plus_plus_baseline", "ppo_clip"),
}
MAIN_CAPABILITIES = {
    "needs_full_log_probs": {"gspo"},
    "needs_critic": {"ppo"},
    "requires_complete_reward_groups": {"grpo", "gspo", "sapo", "cispo", "rloo", "reinforce_plus_plus_baseline"},
    "requires_normalize_advantages": {"reinforce_plus_plus", "reinforce_plus_plus_baseline"},
    "forbids_normalize_advantages": {"rloo"},
    "requires_rewards_normalization": {"rloo", "reinforce_plus_plus_baseline"},
    "forbids_reward_side_kl": {"rloo", "reinforce_plus_plus_baseline"},
    "requires_global_token_loss": {"rloo"},
    "requires_on_policy_updates": {"rloo"},
}


def test_main_algorithms_remain_registered_and_specs_are_consistent():
    assert set(MAIN_ROUTES) <= set(list_algorithm_names())
    assert "sft" not in ALGORITHM_SPECS
    for name, spec in ALGORITHM_SPECS.items():
        assert spec.name == name
        assert not (spec.requires_normalize_advantages and spec.forbids_normalize_advantages)


@pytest.mark.parametrize("name", MAIN_ROUTES)
def test_existing_algorithm_routes_and_capabilities_match_main(name):
    spec = get_algorithm(name)
    assert (spec.reward_normalizer, spec.advantage_fn, spec.policy_loss_fn) == MAIN_ROUTES[name]
    for field, enabled_algorithms in MAIN_CAPABILITIES.items():
        assert getattr(spec, field) is (name in enabled_algorithms), field
    assert spec.kl_level == ("sequence" if name == "gspo" else "token")
    assert spec.advantage_normalization == (
        "token_global" if name in {"reinforce_plus_plus", "reinforce_plus_plus_baseline"} else "whiten"
    )
    assert spec.min_group_size == (2 if name in {"rloo", "reinforce_plus_plus_baseline"} else 1)


def test_spec_is_frozen():
    with pytest.raises(FrozenInstanceError):
        get_algorithm("grpo").name = "mutated"


def test_get_algorithm_unknown_name_raises_with_available_names():
    with pytest.raises(KeyError, match="Unknown advantage estimator.*grpo"):
        get_algorithm("does_not_exist")


def test_reward_normalization_does_not_implicitly_require_complete_groups():
    spec = AlgorithmSpec(
        name="future_batch_algorithm",
        reward_normalizer="future_batch_normalizer",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
    )
    assert not spec.requires_complete_reward_groups
    assert not spec.is_group_normalized


def test_m2po_cp_limit_is_independent_of_reward_grouping():
    spec = get_algorithm("m2po")
    assert not spec.supports_context_parallel
    assert spec.reward_normalizer == "none"
    assert not spec.requires_complete_reward_groups
    assert spec.policy_scalar_metric_names == (
        "ppo_kl_m2_before",
        "ppo_kl_m2_after",
        "m2po_eps_low",
        "m2po_eps_high",
    )


@pytest.mark.parametrize("metric_names", [("",), ("duplicate", "duplicate")])
def test_invalid_policy_scalar_metric_names_are_rejected(metric_names):
    with pytest.raises(ValueError, match="policy scalar metric name"):
        AlgorithmSpec("probe", "none", "grpo_broadcast", "ppo_clip", policy_scalar_metric_names=metric_names)


@pytest.mark.parametrize(
    ("field", "bad"),
    [("kl_level", "Sequence"), ("kl_level", "seq"), ("advantage_normalization", "token-global")],
)
def test_unsupported_enum_values_are_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        AlgorithmSpec("probe", "none", "grpo_broadcast", "ppo_clip", **{field: bad})


def test_every_spec_identifier_resolves_to_a_registered_implementation():
    from relax.algorithms.advantages import ADVANTAGE_FNS
    from relax.algorithms.policy import POLICY_LOSS_FNS
    from relax.algorithms.rewards import REWARD_NORMALIZERS

    for spec in ALGORITHM_SPECS.values():
        assert spec.reward_normalizer in REWARD_NORMALIZERS, spec.name
        assert spec.advantage_fn in ADVANTAGE_FNS, spec.name
        assert spec.policy_loss_fn in POLICY_LOSS_FNS, spec.name


def test_registry_import_does_not_require_training_dependencies():
    code = """
import sys

class BlockTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'ray', 'megatron', 'transfer_queue', 'tensordict'}:
            raise AssertionError(f'Registry imported {fullname}')

sys.meta_path.insert(0, BlockTrainingImports())
from relax.algorithms import get_algorithm
assert get_algorithm('ppo').needs_critic
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)
