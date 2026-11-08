# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Production function-chain tests; queue service and GPU materialization are
seams."""

from __future__ import annotations

import copy

import pytest

from relax.utils.data.global_flops_pack_protocol import STEP_PLANS

from .support import SourceContractClient, backend, concat, fetch_through_actor, partition, pull, restore, setup


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_complete_source_chain_from_global_cohort_through_actor_and_iterator(kind, tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, kind)
    task = "sft_train" if kind == "sft" else "train"
    part = partition([1, 1, 1, 2])
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    seen = []
    for rank in (1, 0):
        payload, meta = fetch_through_actor(args, ctx, contract, sampler, part, rank, task=task)
        expected = [[100, 107, 114]] if rank == 0 else [[121]]
        before = copy.deepcopy(payload)
        actual, K, _ = backend(args, ctx, contract, rank, payload)
        assert [mb["ids"] for mb in actual] == expected and K == [1]
        assert payload == before
        assert sorted(payload["ids"]) == sorted(meta.global_indexes)
        seen.extend(payload["ids"])
        replay, _ = fetch_through_actor(args, ctx, contract, sampler, part, rank, task=task)
        assert replay == before
    assert sorted(seen) == sorted(part.global_indexes)


def test_rl_source_chain_keeps_two_optimizer_boundaries_and_logical_identity_counts(tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, "rl", capacity=6, vpp=2, mb_group=2)
    part = partition([4, 4, 4, 1, 1, 1, 1, 1, 3, 2, 2, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7])
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    payloads = {0: [], 1: []}
    for mini in range(2):
        for rank in (mini % 2, 1 - mini % 2):
            payload, _ = fetch_through_actor(args, ctx, contract, sampler, part, rank, mini=mini, task="train")
            payloads[rank].append(payload)
    for rank, minis in payloads.items():
        merged = concat(minis)
        assert merged["rollout_mini_global_sample_counts"] == [4, 4]
        actual, K, _ = backend(args, ctx, contract, rank, merged)
        # One iteration per VPP stage: check each stage has the same two optimizer boundaries.
        for stage in range(contract.vpp):
            offset = stage * sum(K)
            for mini, count in enumerate(K):
                real = [
                    row for mb in actual[offset : offset + count] if not mb.get("__is_dummy__") for row in mb["ids"]
                ]
                assert sorted(real) == sorted(minis[mini]["ids"])
                offset += count
        assert [p["header"]["identity"][3] for p in merged[STEP_PLANS]] == [0, 1]


def test_local_repack_actor_variant_is_detected_by_frozen_mb_membership(tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, capacity=4, global_batch_size=6)
    part = partition([2] * 6)
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    payloads = []
    # Six SFT rows require a per-rank logical quota of three.
    for rank in range(2):
        rows, meta = pull(sampler, part, contract, rank, quota=3)
        payloads.append(restore(part, rows, meta, ctx, contract, rank))
    rank = next(i for i, p in enumerate(payloads) if len(p["ids"]) == 4)
    payload = payloads[rank]
    good, _, _ = backend(args, ctx, contract, rank, payload)
    frozen = [slot["rows"] for slot in payload[STEP_PLANS][0]["header"]["slots"] if not slot["dummy"]]
    assert [mb["ids"] for mb in good if not mb.get("__is_dummy__")] == frozen
    # Fault injection: Actor treats training as legacy and actually calls the old rank-local packing path.
    bad, _, _ = backend(args, ctx, contract, rank, payload, training=False)
    with pytest.raises(AssertionError):
        assert [mb["ids"] for mb in bad if not mb.get("__is_dummy__")] == frozen


@pytest.mark.parametrize("damage", ["metadata", "fetch", "payload"])
def test_actor_fetch_rejects_faults_before_gpu_postprocess(damage, tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path)
    part = partition([1, 1, 1, 2])
    client = SourceContractClient(sampler, part)
    original = client.get_data

    def corrupt(meta):
        payload = original(meta)
        if damage == "metadata":
            meta._metadata[0].clear()
        elif damage == "payload":
            payload["total_lengths"][0] += 1
        else:
            raise RuntimeError("injected TQ data failure")
        return payload

    monkeypatch.setattr(client, "get_data", corrupt)
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    with pytest.raises((ValueError, RuntimeError), match="Pack-first"):
        fetch_through_actor(args, ctx, contract, sampler, part, 0, client=client)


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_actor_fetch_honors_existing_timeout_after_sixty_seconds(kind, tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, kind)
    part = partition([1, 1, 1, 2])
    client = SourceContractClient(sampler, part)
    original = client.get_meta
    clock = [0.0]
    calls = []

    def delayed(**options):
        clock[0] += 30
        calls.append(options)
        if len(calls) < 3:
            from .support import Meta

            return Meta(part, [])
        return original(**options)

    monkeypatch.setattr(client, "get_meta", delayed)
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    task = "sft_train" if kind == "sft" else "train"
    payload, _ = fetch_through_actor(args, ctx, contract, sampler, part, 0, task=task, client=client)
    assert STEP_PLANS in payload and clock[0] == 90.0 and len(calls) == 3


@pytest.mark.parametrize("task", ["train", "train_actor", "actor_train"])
def test_rl_training_task_aliases_have_explicit_scoped_plan(task, tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, "rl")
    part = partition([1, 1, 1, 2])
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    payload, _ = fetch_through_actor(args, ctx, contract, sampler, part, 0, task=task)
    assert payload[STEP_PLANS][0]["header"]["identity"][2] == task
    actual, K, _ = backend(args, ctx, contract, 0, payload)
    assert [b["ids"] for b in actual] == [[100, 107, 114]] and K == [1]


def test_training_task_scope_includes_actual_runtime_names():
    from types import SimpleNamespace

    from no16.source import ROOT, load

    from relax.utils.data.global_flops_pack_planner import TRAIN_TASKS

    runtime = load(ROOT / "relax/engine/sft/runtime.py", names={"sft_task_name", "is_offline_mode"})
    for loss_type in ("sft", "policy_loss"):
        for component in ("actor", "backend"):
            assert runtime.sft_task_name(SimpleNamespace(loss_type=loss_type), component=component) in TRAIN_TASKS
