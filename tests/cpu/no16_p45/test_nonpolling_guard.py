# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reject fixed TQ nonpolling before estimator construction or state
publication."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import yaml

from relax.utils.data import global_flops_pack_protocol as protocol
from tests.cpu.no16.harness import EvidenceLog
from tests.cpu.no16.source import ROOT, load

from .support import partition, pull, setup


ERROR = "Pack-first requires polling_mode=True.*nonpolling branch.*cached assignments"


def apply_yaml(args, path, polling):
    # Execute the actual --custom-config-path YAML reader/merge, not bool("False").
    path.write_text(f"polling_mode: {str(polling).lower()}\n", encoding="utf-8")
    args.custom_config_path = str(path)
    args.polling_mode = not polling
    args.use_critic = False
    module = load(
        ROOT / "relax/utils/arguments.py",
        names={"apply_custom_config_overrides"},
        injected=dict(
            yaml=yaml,
            logger=EvidenceLog(),
            is_offline_mode=lambda a: a.loss_type == "sft",
            # Algorithm-specific validation is outside this configuration regression.
            validate_kl_coefficients=lambda a: None,
            _validate_reinforce_plus_plus_args=lambda *a, **kw: None,
            validate_algorithm_args=lambda a: None,
            validate_reward_side_kl=lambda *a, **kw: None,
            validate_update_schedule=lambda a: None,
            derive_global_batch_size=lambda *a, **kw: None,
            validate_batch_shape=lambda a: None,
            _normalize_sync_ppo_kl_args=lambda a: None,
        ),
    )
    module.apply_custom_config_overrides(args)
    assert args.polling_mode is polling


def controller(args, refs, effects):
    def make_base(cls, **options):
        base = cls(**options)
        effects["bases"].append((base, copy.deepcopy(base.save_checkpoint())))
        return base

    def resolve_backend():
        effects["backend"].append(True)
        return {}

    def initialize(conf):
        effects["tq"].append(conf)

    module = load(
        ROOT / "relax/core/controller.py",
        class_name="Controller",
        names={"_initialize_data_system"},
        omitted_imports=("relax.engine.sft.runtime", "relax.utils.data.wholepack", "relax.utils.data.flops_sampler"),
        injected=dict(
            resolve_sft_algo_key=lambda a: "sft" if a.loss_type == "sft" else "rl",
            compute_dp_size=lambda a: 2,
            is_i2i_sft=lambda a: False,
            with_control_plane_affinity=lambda a: {},
            IdentityWindowSampler=lambda **kw: make_base(refs["rl"], **kw),
            SeqlenBalancedSampler=lambda **kw: make_base(refs["sft"], **kw),
            enabled=lambda a: False,
            configure_sampler=refs["factory"].configure_sampler,
            OmegaConf=SimpleNamespace(create=lambda conf, **kw: conf),
            tq=SimpleNamespace(init=initialize),
            reap_unusable_tq_controller=lambda: effects["reap"].append(True),
            logger=EvidenceLog(),
        ),
    )
    instance = SimpleNamespace(config=args, _resolve_tq_backend=resolve_backend)
    return lambda: module._initialize_data_system(instance)


def effects_log():
    return {name: [] for name in ("bases", "backend", "tq", "reap", "context", "collective")}


def forbid_effects(monkeypatch, refs, effects):
    def context(*a, **kw):
        effects["context"].append(True)
        raise AssertionError("Estimator/runtime_context reached before rejecting nonpolling")

    def collective(*a, **kw):
        effects["collective"].append(True)
        raise AssertionError("Nonpolling configuration must fail before collectives")

    monkeypatch.setattr(refs["factory"], "runtime_context", context)
    for name in ("agree_cpu", "checked_fetch", "checked_drained", "agree_invocation", "checked_schedule"):
        monkeypatch.setattr(protocol, name, collective)


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_yaml_false_actual_controller_rejects_before_side_effects(kind, tmp_path, monkeypatch):
    args, _, _, _, refs = setup(tmp_path, kind)
    apply_yaml(args, tmp_path / "nonpolling.yaml", False)
    effects = effects_log()
    forbid_effects(monkeypatch, refs, effects)
    with pytest.raises(ValueError, match=ERROR):
        controller(args, refs, effects)()
    assert len(effects["bases"]) == 1
    for base, checkpoint in effects["bases"]:
        assert base.save_checkpoint() == checkpoint
    assert all(not effects[name] for name in ("backend", "tq", "reap", "context", "collective"))
    assert not hasattr(args, "tq_config")


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_direct_factory_false_preserves_checkpoint_metadata_and_no_collective(kind, tmp_path, monkeypatch):
    args, _, contract, sampler, refs = setup(tmp_path, kind)
    task = "sft_train" if kind == "sft" else "train_actor"
    part = partition([1, 1, 1, 2])
    pull(sampler, part, contract, 0, task=task)
    before = copy.deepcopy((sampler.save_checkpoint(), refs["original"].save_checkpoint(), part.metadata))
    args.polling_mode = False
    # Even a missing model path cannot mask the more specific early configuration rejection.
    args.hf_checkpoint = str(tmp_path / "missing-model")
    effects = effects_log()
    forbid_effects(monkeypatch, refs, effects)
    with pytest.raises(ValueError, match=ERROR):
        refs["factory"].configure_sampler(args, refs["original"], 2)
    assert (sampler.save_checkpoint(), refs["original"].save_checkpoint(), part.metadata) == before
    assert all(not values for values in effects.values())


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_direct_wrapper_constructor_false_rejects_without_base_mutation(kind, tmp_path):
    args, context, _, sampler, refs = setup(tmp_path, kind)
    args.polling_mode = False
    before = copy.deepcopy(refs["original"].save_checkpoint())
    with pytest.raises(ValueError, match=ERROR):
        type(sampler)(args, context, 2, refs["original"])
    assert refs["original"].save_checkpoint() == before


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_yaml_true_pack_first_sft_and_balanced_rl_still_publish(kind, tmp_path):
    args, _, contract, _, refs = setup(tmp_path, kind)
    apply_yaml(args, tmp_path / "polling.yaml", True)
    effects = effects_log()
    controller(args, refs, effects)()
    assert len(effects["tq"]) == 1
    assert args.tq_config["controller"]["polling_mode"] is True
    sampler = args.tq_config["controller"]["sampler"]
    task = "sft_train" if kind == "sft" else "train_actor"
    part = partition([1, 1, 1, 2])
    groups = [pull(sampler, part, contract, rank, task=task)[0] for rank in range(2)]
    assert sorted(row for group in groups for row in group) == sorted(part.global_indexes)
    assert sorted(map(len, groups)) == [1, 3]
    assert ("p", task, 0) in sampler.plans


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
@pytest.mark.parametrize("polling", [False, True])
def test_default_dp_first_four_modes_keep_polling_choice(kind, mode, polling, tmp_path):
    args, _, _, _, refs = setup(tmp_path, kind)
    args.flops_scheduling = mode
    del args.flops_scheduling_order  # Exercise the production default, not a test-only override.
    apply_yaml(args, tmp_path / "legacy.yaml", polling)
    effects = effects_log()
    controller(args, refs, effects)()
    assert len(effects["tq"]) == 1
    assert args.tq_config["controller"]["polling_mode"] is polling
    selected = args.tq_config["controller"]["sampler"]
    assert type(selected).__name__ != "GlobalPackSampler"
    assert len(effects["bases"]) == 1
    original, before = effects["bases"][0]
    assert original.save_checkpoint() == before
    if mode in ("off", "microbatch"):
        assert selected is original
    elif kind == "sft":
        assert type(selected).__name__ == "FlopsSftSampler"
    else:
        assert selected.placement == "balanced" and selected.row_placement is not None
