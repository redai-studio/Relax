# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy

import pytest

from relax.utils.data.global_flops_pack_protocol import PLAN_METADATA, STEP_PLANS, local_schedule

from .support import backend, concat, partition, pull, restore, setup


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_both_paths_preserve_complete_mbs_cache_and_explicit_execution(kind, tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, kind, capacity=3)
    task = "sft_train" if kind == "sft" else "train"
    part = partition([1, 1, 1, 2])
    observed = []
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    for rank in [1, 0]:
        rows, meta = pull(sampler, part, contract, rank, task=task)
        payload = restore(part, rows, meta, ctx, contract, rank, task=task)
        batches, K, _ = backend(args, ctx, contract, rank, payload)
        expected = [[100, 107, 114]] if rank == 0 else [[121]]
        assert [b["ids"] for b in batches] == expected and K == [1]
        assert all(not b.get("__is_dummy__", False) for b in batches)
        for batch in batches:
            assert batch["tokens"] == [[row] * part.metadata[row]["total_lengths"] for row in batch["ids"]]
            assert batch["loss_masks"] == [[1] * part.metadata[row]["total_lengths"] for row in batch["ids"]]
        observed.extend(rows)
        assert pull(sampler, part, contract, rank, task=task, ready=[])[0] == rows
        assert backend(args, ctx, contract, rank, payload)[1] == K  # no destructive pop
    assert sorted(observed) == sorted(part.global_indexes)
    saved = sampler.save_checkpoint()
    sampler.load_checkpoint(copy.deepcopy(saved))
    assert sampler.save_checkpoint() == saved


def test_rl_multi_row_identities_multi_mini_offsets_and_optimizer_boundaries(tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, "rl", capacity=6, vpp=2, mb_group=2)
    part = partition([4, 4, 4, 1, 1, 1, 1, 1, 3, 2, 2, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7])
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    batches = {0: [], 1: []}
    rows_by_mini = []
    for mini in range(2):
        consumed = []
        for rank in [mini % 2, 1 - mini % 2]:
            rows, meta = pull(sampler, part, contract, rank, task="train", mini=mini)
            batches[rank].append(restore(part, rows, meta, ctx, contract, rank, task="train", mini=mini))
            consumed.extend(rows)
        rows_by_mini.append(set(consumed))
    assert len(rows_by_mini[0]) == 8 and len(rows_by_mini[1]) == 5 and rows_by_mini[0].isdisjoint(rows_by_mini[1])
    K_by_rank = []
    for rank in range(2):
        merged = concat(batches[rank])
        assert merged["rollout_mini_global_sample_counts"] == [4, 4]  # identities, not 8/5 physical rows
        actual, K, _ = backend(args, ctx, contract, rank, merged)
        K_by_rank.append(K)
        assert len(K) == 2 and all(k % 2 == 0 for k in K)
        offset = 0
        for mini, k in enumerate(K):
            real = [row for b in actual[offset : offset + k] if not b.get("__is_dummy__", False) for row in b["ids"]]
            assert sorted(real) == sorted(batches[rank][mini]["ids"])
            assert set(real) <= rows_by_mini[mini]
            offset += k
    assert K_by_rank[0] == K_by_rank[1]


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_checkpoint_task_isolation_reset_replay_and_partition_generation(kind, tmp_path):
    _, ctx, contract, sampler, _ = setup(tmp_path, kind)
    task = "sft_train" if kind == "sft" else "train"
    part = partition([1, 1, 1, 2])
    original, _ = pull(sampler, part, contract, 0, task=task)
    # A forward consumer has an independent admission/cache and does not overwrite training metadata.
    before = copy.deepcopy(part.metadata)
    sampler.sample(
        sorted(part.global_indexes),
        2,
        task_name="ref_log_probs",
        partition_id="p",
        partition=part,
        dp_rank=0,
        batch_index=0,
    )
    assert part.metadata == before
    saved = sampler.save_checkpoint()
    _, _, _, restored, _ = setup(tmp_path, kind)
    restored.load_checkpoint(saved)
    assert pull(restored, part, contract, 0, task=task, ready=[])[0] == original
    # reset_consumption is TQ-owned: replay with ready rows again must not re-admit/replan.
    assert pull(restored, part, contract, 0, task=task)[0] == original
    digest = restored.plans[("p", task, 0)].plan_digest
    restored.clear_cache("p")
    pull(restored, part, contract, 0, task=task)
    assert restored.plans[("p", task, 0)].identity[1] == 1
    assert restored.plans[("p", task, 0)].plan_digest != digest
    stale = partition([1, 1, 1, 2])  # recycled same row IDs without the committed metadata
    with pytest.raises(ValueError, match="Stale/missing"):
        pull(restored, stale, contract, 0, task=task)


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_partial_metadata_failure_rolls_back_admission_cache_and_plan(kind, tmp_path, monkeypatch):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    task = "sft_train" if kind == "sft" else "train"
    part = partition([1, 1, 1, 2])
    before = sampler.save_checkpoint()
    metadata = copy.deepcopy(part.metadata)
    original = part.set_custom_meta
    calls = []

    def fail_once(updates):
        calls.append(1)
        if len(calls) == 1:
            row = next(iter(updates))
            original({row: updates[row]})
            raise RuntimeError("injected partial metadata write")
        original(updates)

    monkeypatch.setattr(part, "set_custom_meta", fail_once)
    with pytest.raises(RuntimeError, match="partial metadata"):
        pull(sampler, part, contract, 0, task=task)
    assert sampler.save_checkpoint() == before
    for row, old in metadata.items():
        assert {key: val for key, val in part.metadata[row].items() if key != PLAN_METADATA} == old
        assert part.metadata[row].get(PLAN_METADATA, {}) == {}
    pull(sampler, part, contract, 1, task=task)  # bounded transient failure can retry the same cohort


def test_rl_cache_publication_failure_restores_reserved_served_state(tmp_path, monkeypatch):
    _, _, contract, sampler, _ = setup(tmp_path, "rl")
    part = partition([1, 1, 1, 2])
    before = sampler.save_checkpoint()
    original = sampler.base._cache_result

    def fail_second(partition_id, task_name, rank, mini, result):
        if rank == 1:
            raise RuntimeError("injected rank cache failure")
        return original(partition_id, task_name, rank, mini, result)

    monkeypatch.setattr(sampler.base, "_cache_result", fail_second)
    with pytest.raises(RuntimeError, match="rank cache failure"):
        pull(sampler, part, contract, 0, task="train")
    assert sampler.save_checkpoint() == before


@pytest.mark.parametrize("field", ["schema", "order", "signature", "plans"])
def test_checkpoint_incompatible_or_missing_plan_is_rejected_without_mutation(field, tmp_path):
    _, _, contract, sampler, _ = setup(tmp_path, "rl")
    part = partition([1, 1, 1, 2])
    pull(sampler, part, contract, 0, task="train")
    before = sampler.save_checkpoint()
    bad = copy.deepcopy(before)
    bad[field] = {} if field == "plans" else "wrong"
    with pytest.raises(ValueError):
        sampler.load_checkpoint(bad)
    assert sampler.save_checkpoint() == before


@pytest.mark.parametrize("damage", ["missing", "digest", "row", "length", "K", "mini", "logical_count"])
def test_plan_damage_is_detected_by_actual_local_validator(damage, tmp_path):
    _, ctx, contract, sampler, _ = setup(tmp_path)
    part = partition([1, 1, 1, 2])
    rows, meta = pull(sampler, part, contract, 0)
    payload = restore(part, rows, meta, ctx, contract, 0)
    if damage == "missing":
        payload.pop(STEP_PLANS)
    elif damage == "row":
        payload[STEP_PLANS][0]["payload_rows"][0] = 999
    elif damage == "length":
        payload["total_lengths"][0] += 1
    elif damage == "mini":
        payload["rollout_mini_local_sample_counts"].append(1)
    elif damage == "logical_count":
        payload["rollout_mini_global_sample_counts"][0] = 99
    else:
        payload[STEP_PLANS][0]["header"]["K" if damage == "K" else "global_plan_digest"] = 99
    with pytest.raises((ValueError, KeyError, TypeError)):
        local_schedule(payload, contract, ctx.estimator, 0)


def test_sft_global_plan_discards_provisional_token_dp_ownership(tmp_path, monkeypatch):
    _, _, contract, first, _ = setup(tmp_path)
    part = partition([1, 1, 1, 2])
    pull(first, part, contract, 0)
    expected = first.plans[("p", "sft_train", 0)].plan_digest
    _, _, _, second, _ = setup(tmp_path)
    namespace = second.base.sample.__globals__
    original = namespace["get_seqlen_balanced_partitions"]

    def permuted(*values, **options):
        return [list(reversed(group)) for group in reversed(original(*values, **options))]

    monkeypatch.setitem(namespace, "get_seqlen_balanced_partitions", permuted)
    pull(second, partition([1, 1, 1, 2]), contract, 1)
    assert second.plans[("p", "sft_train", 0)].plan_digest == expected


def test_rl_global_placement_keeps_original_balanced_admission_reserved_and_served(tmp_path):
    _, _, contract, sampler, refs = setup(tmp_path, "rl", capacity=6, vpp=2, mb_group=2)
    part = partition([4, 4, 4, 1, 1, 1, 1, 1, 3, 2, 2, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7])
    original = refs["rl"](dp_size=2, placement="balanced")
    for mini in range(2):
        for rank in (1, 0):
            pull(sampler, part, contract, rank, task="train", mini=mini)
            original.sample(
                sorted(part.global_indexes),
                2,
                task_name="train",
                partition_id="p",
                partition=part,
                dp_rank=rank,
                batch_index=mini,
            )
        assert sampler.base._partition_task_states == original._partition_task_states
        assert sampler.base._assigned_global == original._assigned_global
