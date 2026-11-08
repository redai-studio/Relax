# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from no16.harness import Partition, evaluate, production_partitioner
from no16.source import HERE, ROOT, load, samplers
from no16_p2.test_estimator import tiny_config

from relax.utils.data.flops_scheduling import (
    RowPlacement,
    assign_dp_rows,
    build_estimator,
    plan_microbatches,
    runtime_context,
)

from .support import iterator


def args(mode="both", **changes):
    values = dict(flops_scheduling=mode, enable_mtp_training=False, use_dynamic_batch_size=True, loss_type="sft")
    return SimpleNamespace(**{**values, **changes})


def context(mode="both"):
    return runtime_context(args(mode), tiny_config())


def unused(*unused_args):
    pytest.fail("Unexpected fallback")


def test_off_does_not_read_model_path_or_runtime_flags():
    assert runtime_context(SimpleNamespace(flops_scheduling="off")) is None
    assert build_estimator(SimpleNamespace(), object()) is None


@pytest.mark.parametrize(
    "flag",
    [
        "enable_lora",
        "freeze_params_name_list",
        "only_train_params_name_list",
        "freeze_vision_model",
        "sft_async_prepack",
        "fully_async",
        "multimodal_keys",
        "sft_wholepack",
        "use_critic",
    ],
)
def test_enabled_rejects_unverified_runtime_modes(flag):
    with pytest.raises(ValueError, match=flag):
        runtime_context(args(**{flag: True}), tiny_config())


@pytest.mark.parametrize("mode", ["microbatch", "dp", "both"])
def test_enabled_requires_actual_mtp_switch_and_rejects_mtp(mode):
    config = args(mode)
    config.enable_mtp_training = True
    with pytest.raises(ValueError, match="MTP-enabled"):
        runtime_context(config, tiny_config())
    del config.enable_mtp_training
    with pytest.raises(ValueError, match="enable_mtp_training"):
        runtime_context(config, tiny_config())


def test_weighted_membership_has_independent_small_oracle():
    plan = plan_microbatches([4] * 4, [4] * 4, [1, 1, 1, 100], 2, 12, unused)
    assert sorted(sorted(group) for group in plan.indices) == [[0, 1, 2], [3]]
    assert sorted(plan.flops) == [3, 100]
    assert plan.fallback_reason is None


def test_real_lora_and_classification_runtime_flags_are_rejected():
    for options in [dict(lora_rank=8, lora_merge_mode=True), dict(task_type="seq_cls")]:
        with pytest.raises(ValueError):
            runtime_context(args(**options), tiny_config())


def test_weighted_capacity_failure_calls_legacy_with_physical_lengths():
    calls = []

    def legacy(lengths, count, cap):
        calls.append((lengths, count, cap))
        return [[0, 1], [2, 3]], set()

    plan = plan_microbatches([5] * 4, [6] * 4, [100, 1, 1, 1], 2, 12, legacy)
    assert calls == [([6] * 4, 2, 12)]
    assert plan.fallback_reason == "weighted_kk_token_capacity"
    assert plan.logical_tokens == (10, 10) and plan.physical_tokens == (12, 12)
    assert plan.indices == ((0, 1), (2, 3))


def test_fallback_does_not_silently_increase_impossible_k():
    production = production_partitioner()
    with pytest.raises(RuntimeError):
        plan_microbatches([6] * 3, [6] * 3, [100, 1, 1], 2, 10, production._get_seqlen_partitions_with_dummy_padding)


def test_dummy_and_oversized_singleton_are_explicit():
    plan = plan_microbatches([2, 20], [2, 20], [1, 100], 4, 10, unused)
    assert len(plan.indices) == 4 and len(plan.dummy_offsets) == 2
    assert sorted(i for j, g in enumerate(plan.indices) if j not in plan.dummy_offsets for i in g) == [0, 1]
    assert plan.oversized_singletons
    assert all(
        plan.indices[i] in [g for j, g in enumerate(plan.indices) if j not in plan.dummy_offsets]
        for i in plan.dummy_offsets
    )


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf"), "4", 10**400])
def test_invalid_cost_cannot_enter_kk(bad):
    with pytest.raises(ValueError):
        plan_microbatches([1], [1], [bad], 1, 1, unused)


@pytest.mark.parametrize(
    "logical,physical,k,capacity",
    [([], [], 1, 1), ([2], [1], 1, 2), ([0], [1], 1, 1), ([1], [1], 0, 1), ([1], [1], 1, 0)],
)
def test_invalid_capacity_metadata_is_rejected(logical, physical, k, capacity):
    with pytest.raises(ValueError):
        plan_microbatches(logical, physical, [1] * len(logical), k, capacity, unused)


def test_dp_placement_is_not_token_capacity_or_admission():
    rows = [100, 107, 114, 121, 128, 135]
    # Equal-count KK combines (100,5) with reversed (4,3), then reversed
    # (2,1): its heavy side is 100+3+1. KK is not an exact optimizer.
    assert assign_dp_rows(rows, [1, 2, 3, 4, 5, 100], 2) == [[100, 114, 135], [107, 121, 128]]
    assert assign_dp_rows(rows, [1] * 6, 1) == [rows]
    with pytest.raises(ValueError, match="unique"):
        assign_dp_rows([100, 100], [1, 2], 2)


@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
def test_actual_iterator_four_modes_keep_tokens_counts_and_offsets(mode):
    active = None if mode == "off" else context(mode)
    batches, k, reductions, its = iterator([6, 6, 1, 1], active, vpp=2, group_size=2, counts=[2, 2])
    assert k == [2, 2]
    for stage in range(2):
        local = batches[stage * 4 : (stage + 1) * 4]
        assert sorted(i for b in local if not b.get("__is_dummy__", False) for i in b["ids"]) == [0, 1, 2, 3]
    assert len(reductions) == {"off": 1, "microbatch": 2, "dp": 2, "both": 2}[mode]
    assert its[0].dummy_micro_batch_offsets == set()


def test_enabled_multi_mini_dummy_offsets_are_cumulative():
    batches, k, _, its = iterator([3, 4], context(), vpp=2, group_size=2, counts=[1, 1])
    assert k == [2, 2] and its[0].dummy_micro_batch_offsets == {1, 3}
    assert [b["ids"] for b in batches[:4]] == [[0], [0], [1], [1]]


def test_actual_iterator_keeps_cp_physical_proxy_and_peer_error_agreement():
    batches, k, _, _ = iterator([5, 5], context(), cp=2, physical=True)
    assert sum(not b.get("__is_dummy__", False) for b in batches) == sum(k)

    def peer_error(values):
        values[:] = 1

    with pytest.raises(ValueError, match="peer error"):
        iterator([3, 4], context(), agree=peer_error)
    with pytest.raises(ValueError, match="metadata validation"):
        iterator([0, 4], context())


@pytest.mark.parametrize("mode", ["dp", "both"])
def test_real_rl_sampler_cache_resume_and_signature(mode):
    cls = samplers()["rl"]
    place = RowPlacement(context(mode))
    samples = [dict(id=str(i), row=100 + 7 * i, rollout_id=i // 2, length=i + 1) for i in range(8)]
    partition = Partition(samples)
    sampler = cls(dp_size=2, placement="balanced", row_placement=place)
    ready = sorted(partition.global_indexes)
    kw = dict(partition_id="p", task_name="train", partition=partition, batch_index=0)
    first = sampler.sample(ready, 2, dp_rank=0, **kw)
    restored = cls(dp_size=2, placement="balanced", row_placement=place)
    restored.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    second = restored.sample([], 2, dp_rank=1, **kw)
    assert sorted(first[0] + second[0]) == ready
    assert restored.sample([], 2, dp_rank=0, **kw) == first
    with pytest.raises(ValueError, match="placement"):
        cls(dp_size=2, placement="balanced").load_checkpoint(sampler.save_checkpoint())
    with pytest.raises(ValueError, match="geometry"):
        cls(dp_size=4, placement="balanced", row_placement=place).load_checkpoint(sampler.save_checkpoint())


def test_actual_sft_prepublication_hook_and_cache_resume():
    refs = samplers()
    adapter = load(
        ROOT / "relax/utils/data/flops_sampler.py",
        omitted_imports=("transfer_queue",),
        injected={"BaseSampler": refs["base"], "SeqlenBalancedSampler": refs["sft"]},
    )
    sampler = adapter.FlopsSftSampler(2, RowPlacement(context("dp")))
    samples = [dict(id=str(i), row=100 + 7 * i, rollout_id=i, length=i + 1) for i in range(8)]
    partition = Partition(samples)
    kw = dict(partition_id="p", task_name="train", partition=partition, batch_index=0)
    first = sampler.sample(sorted(partition.global_indexes), 4, dp_rank=0, **kw)
    restored = adapter.FlopsSftSampler(2, RowPlacement(context("dp")))
    restored.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    second = restored.sample([], 4, dp_rank=1, **kw)
    assert sorted(first[0] + second[0]) == sorted(partition.global_indexes)
    assert restored.sample([], 4, dp_rank=0, **kw) == first


@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
@pytest.mark.parametrize("objective", ["rl", "sft"])
def test_actual_controller_mode_routing(mode, objective, tmp_path):
    from no16.harness import EvidenceLog

    refs = samplers()
    adapter = load(
        ROOT / "relax/utils/data/flops_sampler.py",
        omitted_imports=("transfer_queue", "relax.utils.data.identity_window_sampler"),
        injected={
            "BaseSampler": refs["base"],
            "SeqlenBalancedSampler": refs["sft"],
            "IdentityWindowSampler": refs["rl"],
        },
    )
    (tmp_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    config = args(
        mode,
        loss_type="sft" if objective == "sft" else "policy_loss",
        algo=objective,
        dp=2,
        hf_checkpoint=str(tmp_path),
        n_samples_per_prompt=1,
        polling_mode=True,
        balance_data=False,
    )

    original_states = []
    queue_initializations = []

    def configure(current, original, dp):
        assert mode != "off", "off must not enter the new factory"
        original_states.append((original, copy.deepcopy(original.save_checkpoint())))
        return adapter.configure_sampler(current, original, dp)

    def initialize_queue(conf):
        queue_initializations.append(conf)
        return conf

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
            configure_sampler=configure,
            OmegaConf=SimpleNamespace(create=lambda c, **kw: c),
            tq=SimpleNamespace(init=initialize_queue),
            reap_unusable_tq_controller=lambda: None,
            logger=EvidenceLog(),
        ),
    )
    controller = SimpleNamespace(config=config, _resolve_tq_backend=lambda: {})
    if objective == "rl" and mode in ("dp", "both"):
        with pytest.raises(ValueError, match="RL FLOPs-aware DP requires balance_data=True"):
            module._initialize_data_system(controller)
        assert queue_initializations == []
        assert not hasattr(config, "tq_config")
        assert len(original_states) == 1
        original, state = original_states[0]
        assert original.placement == "sequential"
        assert original.save_checkpoint() == state
        return
    module._initialize_data_system(controller)
    selected = config.tq_config["controller"]["sampler"]
    if objective == "sft":
        assert isinstance(selected, adapter.FlopsSftSampler if mode in ("dp", "both") else refs["sft"])
    else:
        assert selected.placement == "sequential"
        assert selected.row_placement is None


def test_real_parser_argument_has_off_default_and_exactly_four_modes():
    import argparse
    import ast

    tree = ast.parse((ROOT / "relax/utils/arguments.py").read_text(encoding="utf-8"))
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "--flops-scheduling"
    )
    parser = argparse.ArgumentParser()
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[ast.Expr(call)], type_ignores=[])), "<actual-argument>", "exec"
        ),
        {"parser": parser},
    )
    assert parser.parse_args([]).flops_scheduling == "off"
    for mode in ("off", "microbatch", "dp", "both"):
        assert parser.parse_args(["--flops-scheduling", mode]).flops_scheduling == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--flops-scheduling", "approx"])


def test_actor_delegate_keeps_legacy_call_and_passes_private_context_only():
    calls = []

    def target(*values, **options):
        calls.append((values, options))
        return "iterator"

    module = load(
        ROOT / "relax/backends/megatron/actor.py",
        class_name="MegatronTrainRayActor",
        names={"_get_data_iterator"},
        injected={"get_data_iterator": target},
    )
    actor = SimpleNamespace(args=object(), model=object())
    payload = {"total_lengths": [1]}
    assert module._get_data_iterator(actor, payload) == "iterator"
    assert calls[-1] == ((actor.args, actor.model, payload), {})
    actor._flops_context = context()
    module._get_data_iterator(actor, payload)
    assert calls[-1][1]["flops_context"] is actor._flops_context
    assert not hasattr(actor.args, "_flops_context")


@pytest.mark.parametrize(
    "case", json.loads((HERE / "fixtures/schedules.json").read_text())["cases"], ids=lambda c: c["name"]
)
def test_current_off_schedule_matches_immutable_p1_golden(case):
    expected = json.loads((HERE / "golden/relax.json").read_text())
    assert evaluate(case, "relax") == expected[case["name"]]
