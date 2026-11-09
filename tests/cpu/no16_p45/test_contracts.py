# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Mode contracts and failures must reject before publishing state."""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from relax.utils.data.global_flops_pack_protocol import STEP_PLANS, checked_schedule, seal_header
from tests.cpu.no16.harness import EvidenceLog
from tests.cpu.no16.source import ROOT, load

from .support import partition, pull, restore, setup


def test_actual_parser_order_default_and_exact_choices():
    tree = ast.parse((ROOT / "relax/utils/arguments.py").read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value in ("--flops-scheduling", "--flops-scheduling-order")
    ]
    assert len(calls) == 2
    parser = argparse.ArgumentParser()
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[ast.Expr(n) for n in calls], type_ignores=[])),
            "<actual-parser>",
            "exec",
        ),
        {"parser": parser},
    )
    assert vars(parser.parse_args([])) == dict(flops_scheduling="off", flops_scheduling_order="dp-first")
    for order in ("dp-first", "pack-first"):
        assert parser.parse_args(["--flops-scheduling-order", order]).flops_scheduling_order == order
    with pytest.raises(SystemExit):
        parser.parse_args(["--flops-scheduling-order", "automatic"])


@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
@pytest.mark.parametrize("order", ["dp-first", "pack-first"])
@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_actual_factory_order_matrix_has_no_invalid_state_publication(mode, order, kind, tmp_path):
    args, _, _, _, refs = setup(tmp_path, kind)
    args.flops_scheduling, args.flops_scheduling_order = mode, order
    before = copy.deepcopy(refs["original"].save_checkpoint())
    if order == "pack-first" and mode != "both":
        with pytest.raises(ValueError, match="pack-first requires"):
            refs["factory"].configure_sampler(args, refs["original"], 2)
    else:
        selected = refs["factory"].configure_sampler(args, refs["original"], 2)
        if order == "dp-first" and mode in ("off", "microbatch"):
            assert selected is refs["original"]
        if order == "pack-first":
            assert type(selected).__name__ == "GlobalPackSampler"
    assert refs["original"].save_checkpoint() == before


@pytest.mark.parametrize(
    "flag",
    [
        "enable_mtp_training",
        "fully_async",
        "use_critic",
        "dynamic_context_parallel",
        "sft_train_data_prefetch",
        "hybrid",
    ],
)
def test_existing_and_new_scope_checks_reject_before_sampler_publication(flag, tmp_path):
    args, _, _, _, refs = setup(tmp_path)
    setattr(args, flag, True)
    before = copy.deepcopy(refs["original"].save_checkpoint())
    with pytest.raises(ValueError):
        refs["factory"].configure_sampler(args, refs["original"], 2)
    assert refs["original"].save_checkpoint() == before


def test_pack_first_never_silently_changes_rl_sequential_admission(tmp_path):
    args, _, _, _, refs = setup(tmp_path, "rl")
    args.balance_data = False
    original = refs["rl"](dp_size=2, placement="sequential")
    before = copy.deepcopy(original.save_checkpoint())
    with pytest.raises(ValueError, match="balance_data=True"):
        refs["factory"].configure_sampler(args, original, 2)
    assert original.save_checkpoint() == before


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("damage", ["poisoned", "rank", "consumed", "base_field", "plan_K", "plan_rows", "generation"])
def test_checkpoint_rejection_is_atomic_including_all_rank_and_consumed_caches(kind, damage, tmp_path):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    task = "sft_train" if kind == "sft" else "train"
    pull(sampler, partition([1, 1, 1, 2]), contract, 0, task=task)
    before = sampler.save_checkpoint()
    bad = copy.deepcopy(before)
    key = ("p", task, 0)
    if damage == "poisoned":
        bad.pop("poisoned")
    elif damage == "rank":
        bad["base"]["_states"]["p"][task].pop(1)
    elif damage == "consumed":
        sampled, _ = bad["base"]["_states"]["p"][task][1][0]
        bad["base"]["_states"]["p"][task][1][0] = (sampled, [999])
    elif damage == "base_field":
        bad["base"].pop(next(name for name in bad["base"] if name != "_states"))
    elif damage == "generation":
        bad["generations"]["p"] = 7
    else:
        plan = bad["plans"][key]
        bad["plans"][key] = replace(plan, **({"K": 7} if damage == "plan_K" else {"rows": (999,)}))
    with pytest.raises(ValueError, match="Invalid pack-first checkpoint"):
        sampler.load_checkpoint(bad)
    assert sampler.save_checkpoint() == before


def test_execution_geometry_failure_is_part_of_shared_error_boundary(tmp_path):
    _, ctx, contract, sampler, _ = setup(tmp_path)
    part = partition([1, 1, 1, 2])
    rows, meta = pull(sampler, part, contract, 0)
    payload = restore(part, rows, meta, ctx, contract, 0)
    calls = []

    def agree(values):
        calls.append(values)
        return values

    with pytest.raises(ValueError, match="execution geometry"):
        checked_schedule(payload, contract, ctx.estimator, 0, agree, contract_provider=lambda: replace(contract, pp=2))
    assert calls == [[1, 0, 0]]


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_opaque_producer_metadata_does_not_break_diagnostics(kind, tmp_path):
    _, ctx, contract, sampler, _ = setup(tmp_path, kind)
    part = partition([1, 1, 1, 2])
    for value in part.metadata.values():
        value["opaque"] = b"producer-pickle-data"
    task = "sft_train" if kind == "sft" else "train"
    rows, meta = pull(sampler, part, contract, 0, task=task)
    restore(part, rows, meta, ctx, contract, 0, task=task)
    assert all(value["opaque"] == b"producer-pickle-data" for value in part.metadata.values())


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_persistent_partial_write_poison_blocks_replay_until_clear(kind, tmp_path, monkeypatch):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    part = partition([1, 1, 1, 2])
    task = "sft_train" if kind == "sft" else "train"
    original = part.set_custom_meta

    def fail(updates):
        row = next(iter(updates))
        original({row: updates[row]})
        raise RuntimeError("permanent metadata write failure")

    monkeypatch.setattr(part, "set_custom_meta", fail)
    with pytest.raises(RuntimeError, match="permanent"):
        pull(sampler, part, contract, 0, task=task)
    with pytest.raises(RuntimeError, match="requires partition clear"):
        pull(sampler, part, contract, 1, task=task)
    with pytest.raises(RuntimeError, match="requires partition clear"):
        sampler.has_cached_result("p", task, {"batch_index": 0})


def test_resealed_capacity_and_dummy_corruption_are_rejected(tmp_path):
    from relax.utils.data.global_flops_pack_protocol import local_schedule

    _, ctx, contract, sampler, _ = setup(tmp_path)
    part = partition([1, 1, 1, 2])
    rows, meta = pull(sampler, part, contract, 0)
    payload = restore(part, rows, meta, ctx, contract, 0)
    header = payload[STEP_PLANS][0]["header"]
    header["slots"][0]["physical_tokens"] = 999
    payload[STEP_PLANS][0]["header"] = seal_header(header)
    with pytest.raises(ValueError, match="accounting"):
        local_schedule(payload, contract, ctx.estimator, 0)


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_sft_orphan_cache_and_rl_poisoned_state_corruption_reject(kind, tmp_path):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    task = "sft_train" if kind == "sft" else "train"
    pull(sampler, partition([1, 1, 1, 2]), contract, 0, task=task)
    before = sampler.save_checkpoint()
    bad = copy.deepcopy(before)
    if kind == "sft":
        bad["base"]["_balanced_cache"][("p", task, 1)] = bad["base"]["_balanced_cache"][("p", task, 0)]
    else:
        bad["poisoned"] = []
    with pytest.raises(ValueError):
        sampler.load_checkpoint(bad)
    assert sampler.save_checkpoint() == before


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
def test_actual_controller_pack_order_routes_or_rejects_before_tq_init(kind, mode, tmp_path):
    config, _, _, _, refs = setup(tmp_path, kind)
    config.flops_scheduling = mode
    config.algo = kind
    config.dp = 2
    config.polling_mode = True
    initialized = []
    module = load(
        ROOT / "relax/core/controller.py",
        class_name="Controller",
        names={"_initialize_data_system"},
        omitted_imports=("relax.engine.sft.runtime", "relax.utils.data.wholepack", "relax.utils.data.flops_sampler"),
        injected=dict(
            resolve_sft_algo_key=lambda a: a.algo,
            compute_dp_size=lambda a: a.dp,
            is_i2i_sft=lambda a: False,
            with_control_plane_affinity=lambda a: {},
            IdentityWindowSampler=refs["rl"],
            SeqlenBalancedSampler=refs["sft"],
            enabled=lambda a: False,
            configure_sampler=refs["factory"].configure_sampler,
            OmegaConf=SimpleNamespace(create=lambda c, **kw: c),
            tq=SimpleNamespace(init=lambda conf: initialized.append(conf)),
            reap_unusable_tq_controller=lambda: None,
            logger=EvidenceLog(),
        ),
    )
    controller = SimpleNamespace(config=config, _resolve_tq_backend=lambda: {})
    if mode != "both":
        with pytest.raises(ValueError, match="pack-first requires"):
            module._initialize_data_system(controller)
        assert not initialized and not hasattr(config, "tq_config")
    else:
        module._initialize_data_system(controller)
        assert type(config.tq_config["controller"]["sampler"]).__name__ == "GlobalPackSampler"
        assert len(initialized) == 1


@pytest.mark.parametrize("drained", [False, True])
def test_queue_status_check_uses_shared_control_boundary(drained):
    from relax.utils.data.global_flops_pack_protocol import checked_drained

    assert checked_drained(lambda: drained, lambda values: values) == drained

    def failure():
        raise RuntimeError("injected status RPC failure")

    calls = []

    def agree(values):
        calls.append(values)
        return values

    with pytest.raises(RuntimeError, match="Pack-first queue status"):
        checked_drained(failure, agree)
    assert calls == [[1, 1]]


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_cached_plan_rejects_changed_logical_quota_without_state_mutation(kind, tmp_path):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    part = partition([1, 1, 1, 2])
    task = "sft_train" if kind == "sft" else "train"
    pull(sampler, part, contract, 0, task=task)
    before = sampler.save_checkpoint()
    with pytest.raises(ValueError, match="logical batch quota changed"):
        pull(sampler, part, contract, 1, task=task, quota=3)
    assert sampler.save_checkpoint() == before


def test_checkpoint_geometry_signature_rejects_even_empty_incompatible_state(tmp_path):
    args, ctx, _, sampler, refs = setup(tmp_path)
    before = sampler.save_checkpoint()
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "microbatch_group_size_per_vp_stage",
        "max_tokens_per_gpu",
    ):
        changed = copy.deepcopy(args)
        setattr(changed, name, 7)
        fresh = refs["factory"].configure_sampler(changed, refs["original"], 2)
        assert fresh.signature != sampler.signature
        empty = fresh.save_checkpoint()
        with pytest.raises(ValueError):
            fresh.load_checkpoint(before)
        assert fresh.save_checkpoint() == empty
