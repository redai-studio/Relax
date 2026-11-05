# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Role registration and critic capabilities without importing Ray services."""

import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from relax.algorithms import ALGORITHM_SPECS
from tests.cpu.relax_parser import args_for_flags, parser_for_flags


@pytest.fixture()
def load_registry(monkeypatch):
    """Execute the complete registry with inert component classes, on CPU."""
    components = ModuleType("relax.components")
    components.__path__ = []
    monkeypatch.setitem(sys.modules, components.__name__, components)
    classes = {}
    for module_name, class_name in {
        "actor": "Actor",
        "actor_fwd": "ActorFwd",
        "advantages": "Advantages",
        "critic": "Critic",
        "rollout": "Rollout",
        "sft": "SFT",
    }.items():
        module = ModuleType(f"relax.components.{module_name}")
        classes[class_name] = type(class_name, (), {})
        setattr(module, class_name, classes[class_name])
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def load():
        path = Path(__file__).resolve().parents[3] / "relax/core/registry.py"
        spec = importlib.util.spec_from_file_location("_algorithm_test_registry", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, classes

    return load


def _args_for(name, **overrides):
    return args_for_flags(
        "--use-rollout-indexer-replay",
        "--use-rollout-routing-replay",
        "--kl-coef",
        "--use-kl-loss",
        "--multimodal-keys",
        "--fully-async",
        "--hybrid",
        "--use-rollout-logprobs",
        "--true-on-policy-mode",
        **{
            "advantage_estimator": name,
            "use_opd": False,
            "loss_type": None,
            "debug_rollout_only": False,
            "debug_train_only": False,
            **overrides,
        },
    )


def _register_critic_algorithm(monkeypatch):
    name = "new_critic_algorithm"
    monkeypatch.setitem(ALGORITHM_SPECS, name, replace(ALGORITHM_SPECS["ppo"], name=name))
    return name


def test_registered_algorithms_have_the_expected_role_classes(load_registry):
    registry, classes = load_registry()
    base = {
        "actor": classes["Actor"],
        "rollout": classes["Rollout"],
        "advantages": classes["Advantages"],
        "reference": classes["ActorFwd"],
        "actor_fwd": classes["ActorFwd"],
    }
    for name, spec in ALGORITHM_SPECS.items():
        expected = {**base, "critic": classes["Critic"]} if spec.needs_critic else base
        assert registry.ALGOS[name] == expected, name
    assert registry.ALGOS["sft"] == {"sft": classes["SFT"], "actor": classes["Actor"]}


@pytest.mark.parametrize("name", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reinforce_plus_plus_uses_sync_colocate_roles(load_registry, name):
    registry, _ = load_registry()
    assert registry.process_role(_args_for(name, loss_type="policy_loss")) is registry.ROLES_COLOCATE


@pytest.mark.parametrize("needs_critic", [False, True])
def test_new_algorithm_is_registered_without_editing_role_tables(monkeypatch, load_registry, needs_critic):
    name = "new_algorithm"
    source = "ppo" if needs_critic else "grpo"
    monkeypatch.setitem(ALGORITHM_SPECS, name, replace(ALGORITHM_SPECS[source], name=name))
    registry, _ = load_registry()

    assert registry.ALGOS[name] == registry.ALGOS[source]
    for fully_async in (False, True):
        assert registry.process_role(_args_for(name, fully_async=fully_async)) is registry.process_role(
            _args_for(source, fully_async=fully_async)
        )


def test_each_algorithm_gets_an_independent_role_dict(load_registry):
    registry, _ = load_registry()
    assert len({id(roles) for roles in registry.ALGOS.values()}) == len(registry.ALGOS)


@pytest.mark.parametrize("consumer", ["actor", "critic", "advantages"])
def test_a_second_critic_algorithm_gets_the_critic_rollout_fields(monkeypatch, consumer):
    from relax.utils.training.data_fields import build_data_fields

    name = _register_critic_algorithm(monkeypatch)
    fields = build_data_fields(_args_for(name), consumer=consumer)
    assert fields == build_data_fields(_args_for("ppo"), consumer=consumer)
    assert ("values" in fields) is (consumer != "critic")


@pytest.mark.parametrize(
    ("flags", "overrides", "actor_ref", "advantages_ref"),
    [
        ([], {}, False, False),
        (["--kl-coef", "0"], {"use_kl_loss": False}, False, False),
        (["--kl-coef", "0.25"], {}, True, True),
        (["--use-kl-loss"], {}, True, False),
    ],
)
def test_ppo_reference_fields_follow_declared_kl_options(
    flags: list[str], overrides: dict[str, bool], actor_ref: bool, advantages_ref: bool
) -> None:
    from relax.utils.training.data_fields import build_data_fields

    parsed = vars(parser_for_flags("--kl-coef", "--use-kl-loss").parse_args(flags))
    args = _args_for("ppo", **(parsed | overrides))
    actor = build_data_fields(args, consumer="actor")
    advantages = build_data_fields(args, consumer="advantages")
    critic = build_data_fields(args, consumer="critic")

    assert actor.count("ref_log_probs") == int(actor_ref)
    assert "log_probs" not in actor
    assert ("ref_log_probs" in advantages) is advantages_ref
    assert ("log_probs" in advantages) is advantages_ref
    assert "ref_log_probs" not in critic


def _args_with_opd(*flags: str) -> Namespace:
    from relax.utils.opd.opd_utils import add_opd_arguments

    parsed = add_opd_arguments(ArgumentParser()).parse_args(flags)
    return _args_for("ppo", **vars(parsed))


@pytest.mark.parametrize("consumer", ["actor", "advantages", "critic"])
@pytest.mark.parametrize("teacher", [None, "megatron", "sglang"], ids=["disabled", "megatron", "sglang"])
def test_transfer_fields_follow_declared_opd_mode(consumer: str, teacher: str | None) -> None:
    from relax.utils.training.data_fields import build_data_fields

    args = _args_with_opd("--use-opd", "--opd-type", teacher) if teacher is not None else _args_with_opd()

    fields = build_data_fields(args, consumer=consumer, require_declared_opd=True)

    assert fields == build_data_fields(args, consumer=consumer)
    assert fields.count("teacher_log_probs") == int(teacher == "sglang" and consumer != "critic")
    assert fields.count("rollout_log_probs") == 1


@pytest.mark.parametrize("field", ["use_opd", "opd_type"])
def test_parsed_transfer_fields_require_enabled_opd_declarations(field: str) -> None:
    from relax.utils.training.data_fields import build_data_fields

    args = _args_with_opd("--use-opd", "--opd-type", "sglang")
    delattr(args, field)

    assert "teacher_log_probs" not in build_data_fields(args)
    with pytest.raises(AttributeError, match=field):
        build_data_fields(args, require_declared_opd=True)


def test_transfer_fields_keep_explicit_null_opd_options_disabled() -> None:
    from relax.utils.training.data_fields import build_data_fields

    args = _args_with_opd()
    declared_disabled = build_data_fields(args)
    args.use_opd = None
    del args.opd_type
    assert build_data_fields(args, require_declared_opd=True) == declared_disabled

    args.use_opd = True
    args.opd_type = None
    assert build_data_fields(args, require_declared_opd=True) == declared_disabled


def test_a_second_critic_algorithm_requires_a_critic_resource(monkeypatch):
    from relax.utils.training.ppo_utils import validate_ppo_config

    name = _register_critic_algorithm(monkeypatch)
    with pytest.raises(ValueError, match="requires a 'critic' entry"):
        validate_ppo_config(_args_for(name, resource={"actor": "a"}))
    validate_ppo_config(_args_for("grpo", resource={"actor": "a"}))


def test_ppo_config_uses_declared_defaults_and_rejects_explicit_staleness():
    from relax.utils.training.ppo_utils import validate_ppo_config

    parser = parser_for_flags(
        "--resource", "--fully-async", "--hybrid", "--colocate", "--max-staleness", "--critic-load"
    )
    args = parser.parse_args(
        ["--colocate", "--resource", '{"actor": [1, 1], "critic": [1, 1]}'],
        namespace=Namespace(advantage_estimator="ppo", load=None),
    )
    validate_ppo_config(args)
    args.max_staleness = 1
    with pytest.raises(ValueError, match="Synchronous colocate PPO requires --max-staleness 0"):
        validate_ppo_config(args)
