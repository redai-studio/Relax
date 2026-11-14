# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Post-review failures: no held-batch loss, no reuse of an untrusted shared
base."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from relax.utils.data.global_flops_pack_protocol import PackFirstExhausted, checked_fetch
from tests.cpu.no16.source import ROOT, load

from .support import Meta, SourceContractClient, backend, fetch_through_actor, partition, pull, setup


def test_empty_finished_producer_stops_without_waiting_for_default_timeout():
    traces = []
    fetches = []
    status = []

    def fetch():
        fetches.append(1)
        return [None, None]

    def done():
        status.append(1)
        return True

    def agree(values):
        traces.append(values)
        return values

    with pytest.raises(PackFirstExhausted):
        checked_fetch(fetch, True, agree, is_drained=done, timeout_seconds=1800)
    assert len(fetches) == len(status) == 1
    assert [len(row) for row in traces] == [5, 2]


@pytest.mark.parametrize("role", ["query", "tp_peer", "pp_peer", "cp_peer"])
def test_actual_actor_raw_status_has_no_inner_collective_before_error_agreement(role):
    from relax.utils.data.global_flops_pack_protocol import checked_drained

    topology = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: int(role == "tp_peer"),
        get_pipeline_model_parallel_rank=lambda: int(role == "pp_peer"),
        get_context_parallel_rank=lambda: int(role == "cp_peer"),
    )

    def forbidden(*a, **kw):
        pytest.fail("A queue error must not strand peers in the legacy MP collective")

    module = load(
        ROOT / "relax/backends/megatron/actor.py",
        class_name="MegatronTrainRayActor",
        names={"_pack_first_local_consumed"},
        injected=dict(mpu=topology, run=lambda value: value, _agree_drained=forbidden, sft_partition_id=forbidden),
    )
    for fail in (False, True):
        calls = []
        agreement = []

        def rpc(task, pid):
            calls.append((task, pid))
            if fail:
                raise RuntimeError("injected real Actor query RPC error")
            return False

        actor = SimpleNamespace(data_system_client=SimpleNamespace(async_check_consumption_status=rpc))

        def agree(values):
            agreement.append(values)
            return values

        query = lambda: module._pack_first_local_consumed(actor, "sft_train", 7, partition_id="actual_override")
        if role == "query" and fail:
            with pytest.raises(RuntimeError, match="Pack-first queue status failed"):
                checked_drained(query, agree)
            assert agreement == [[1, 1]]
        else:
            assert checked_drained(query, agree) == (role != "query")
        assert calls == ([("sft_train", "actual_override")] if role == "query" else [])


def test_empty_live_producer_recovers_ready_and_consumes_the_batch_once():
    payload = {"ready": True}
    calls = []
    status = []
    traces = []

    def fetch():
        calls.append(1)
        return [None, None] if len(calls) < 3 else [payload, "meta"]

    def done():
        status.append(1)
        return False

    def agree(values):
        traces.append(values)
        return values

    result = checked_fetch(fetch, True, agree, is_drained=done, poll_seconds=0)
    assert result[0] is payload and len(calls) == 3 and len(status) == 2
    assert [len(row) for row in traces] == [5, 2, 5, 2, 5]


@pytest.mark.parametrize("locally_ready", [False, True])
def test_partial_ready_preserves_assignment_and_never_checks_consumed(locally_ready):
    payload = {"frozen_assignment": [100, 107]}
    calls = []
    rounds = []

    def fetch():
        calls.append(1)
        return [payload, "meta"] if locally_ready or len(calls) >= 3 else [None, None]

    def forbidden():
        pytest.fail("A held peer batch must not be discarded by checking consumed")

    def agree(values):
        rounds.append(values)
        return [0, 1, int(len(rounds) < 3), 0, 1]

    result = checked_fetch(fetch, True, agree, is_drained=forbidden, poll_seconds=0)
    assert result[0] is payload
    assert len(calls) == (1 if locally_ready else 3)


def test_nonfetching_partner_joins_shared_empty_termination():
    shapes = []

    def fetch():
        pytest.fail("Nonfetching TP partner must not call queue RPC")

    def agree(values):
        shapes.append(len(values))
        return [0, 0, 1, 0, 1] if len(values) == 5 else values

    with pytest.raises(PackFirstExhausted):
        checked_fetch(fetch, False, agree, is_drained=lambda: True)
    assert shapes == [5, 2]


@pytest.mark.parametrize("failure", ["fetch", "status", "timeout"])
def test_empty_fetch_failures_are_bounded_and_collectively_reported(failure, monkeypatch):
    shapes = []

    def agree(values):
        shapes.append(len(values))
        return values

    def fetch():
        if failure == "fetch":
            raise RuntimeError("injected queue RPC error")
        return [None, None]

    def done():
        if failure == "status":
            raise RuntimeError("injected status RPC error")
        return False

    if failure == "timeout":
        times = iter([0.0, 2.0])
        monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.time.monotonic", lambda: next(times))
    with pytest.raises((RuntimeError, TimeoutError)):
        checked_fetch(fetch, True, agree, is_drained=done, timeout_seconds=1, poll_seconds=0)
    assert shapes == ([5] if failure == "fetch" else [5, 2])


@pytest.mark.parametrize("kind", ["sft", "rl"])
def test_actual_actor_fetch_maps_agreed_end_to_no_batch(kind, tmp_path, monkeypatch):
    args, ctx, contract, sampler, _ = setup(tmp_path, kind)
    part = partition([1, 1, 1, 2])
    client = SourceContractClient(sampler, part)
    monkeypatch.setattr(client, "get_meta", lambda **kw: Meta(part, []))
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    status = []

    def done():
        status.append(1)
        return True

    task = "sft_train" if kind == "sft" else "train"
    before = sampler.save_checkpoint()
    assert fetch_through_actor(args, ctx, contract, sampler, part, 0, task=task, client=client, is_drained=done) == (
        None,
        None,
    )
    assert len(status) == 1 and sampler.save_checkpoint() == before and sampler.plans == {}


def assert_shared_base_blocked(sampler, part, contract, task):
    operations = [
        lambda: pull(sampler, part, contract, 1, mini=1, task=task),
        lambda: pull(sampler, part, contract, 1, task="train_actor"),
        lambda: sampler.sample([], 2, task_name="ref_log_probs", partition_id="other"),
        lambda: sampler.has_cached_result("p", task, {"batch_index": 0}),
        sampler.save_checkpoint,
        lambda: sampler.clear_cache("p"),
    ]
    sampler.poisoned.clear()  # Clearing the narrower metadata marker cannot recover the shared base.
    for operation in operations:
        with pytest.raises(RuntimeError, match="shared base state is untrusted"):
            operation()


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("rollback", ["raises", "silently_skips"])
def test_failed_sample_rollback_disables_every_task_mini_and_recovers_only_from_trusted_state(
    kind, rollback, tmp_path, monkeypatch
):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    part = partition([1, 1, 1, 2])
    task = "sft_train" if kind == "sft" else "train"
    trusted = sampler.save_checkpoint()
    original_load = sampler.base.load_checkpoint
    original_set = part.set_custom_meta

    def fail_publication(updates):
        row = next(iter(updates))
        original_set({row: updates[row]})
        raise RuntimeError("injected partial metadata publication")

    def fail_restore(state):
        if rollback == "raises":
            raise RuntimeError("injected shared base rollback error")

    monkeypatch.setattr(part, "set_custom_meta", fail_publication)
    monkeypatch.setattr(sampler.base, "load_checkpoint", fail_restore)
    with pytest.raises(RuntimeError, match="partial metadata publication"):
        pull(sampler, part, contract, 0, task=task)
    assert sampler.base._states and sampler._unrecoverable is not None
    assert_shared_base_blocked(sampler, part, contract, task)
    monkeypatch.setattr(sampler.base, "load_checkpoint", original_load)
    monkeypatch.setattr(part, "set_custom_meta", original_set)
    sampler.load_checkpoint(trusted)
    assert sampler._unrecoverable is None and sampler.save_checkpoint() == trusted
    assert pull(sampler, part, contract, 0, task=task)[0]


@pytest.mark.parametrize("kind", ["sft", "rl"])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_live_checkpoint_load_failure_and_second_rollback_have_one_health_policy(
    kind, rollback_fails, tmp_path, monkeypatch
):
    _, _, contract, sampler, _ = setup(tmp_path, kind)
    empty = sampler.save_checkpoint()
    part = partition([1, 1, 1, 2])
    task = "sft_train" if kind == "sft" else "train"
    pull(sampler, part, contract, 0, task=task)
    trusted = sampler.save_checkpoint()
    base = sampler.base
    original = type(base).load_checkpoint
    calls = []

    def failure(self, state):
        if self is base:
            calls.append(1)
            if len(calls) == 1 or rollback_fails:
                self._states = {"partially_loaded": {}}
                raise RuntimeError("injected live base load failure")
        return original(self, state)

    monkeypatch.setattr(base, "load_checkpoint", MethodType(failure, base))
    with pytest.raises(RuntimeError, match="live base load failure"):
        sampler.load_checkpoint(empty)
    assert len(calls) == 2
    if rollback_fails:
        assert_shared_base_blocked(sampler, part, contract, task)
    else:
        assert sampler._unrecoverable is None and sampler.save_checkpoint() == trusted
        assert pull(sampler, part, contract, 1, task=task)[0]
    monkeypatch.setattr(base, "load_checkpoint", MethodType(original, base))
    sampler.load_checkpoint(trusted)
    assert sampler._unrecoverable is None and sampler.save_checkpoint() == trusted


def test_sft_true_runtime_is_offline_single_mini_with_multiple_microbatches():
    runtime = load(ROOT / "relax/engine/sft/runtime.py", names={"is_offline_mode"})
    assert runtime.is_offline_mode(SimpleNamespace(loss_type="sft"))
    # The actual Actor selection is covered by the fixed production branch, not a fabricated online SFT route.
    import ast

    tree = ast.parse((ROOT / "relax/backends/megatron/actor.py").read_text(encoding="utf-8"))
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and getattr(n.test.func, "id", None) == "is_offline_mode"
        and any(
            isinstance(x, ast.Assign) and any(getattr(t, "id", None) == "num_rollout_minis" for t in x.targets)
            for x in n.body
        )
    )
    namespace = dict(
        is_offline_mode=runtime.is_offline_mode,
        self=SimpleNamespace(args=SimpleNamespace(loss_type="sft", global_batch_size=12)),
        dp_size=2,
        mpu=SimpleNamespace(get_data_parallel_world_size=lambda **kw: 2),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "<actual-SFT-mini-branch>", "exec"
        ),
        namespace,
    )
    assert namespace["num_rollout_minis"] == 1 and namespace["batch_size"] == 6


@pytest.mark.parametrize("steps", [2, 3])
def test_sft_multi_optimizer_mini_configuration_rejects_before_state_or_metadata(steps, tmp_path):
    args, _, _, _, refs = setup(tmp_path)
    args.num_steps_per_rollout = steps
    before = copy.deepcopy(refs["original"].save_checkpoint())
    with pytest.raises(ValueError, match="one optimizer mini"):
        refs["factory"].configure_sampler(args, refs["original"], 2)
    assert refs["original"].save_checkpoint() == before


def test_sft_second_mini_request_rejects_before_base_state_or_metadata(tmp_path):
    _, _, contract, sampler, _ = setup(tmp_path)
    part = partition([6, 6, 6, 1])
    before = sampler.save_checkpoint()
    metadata = copy.deepcopy(part.metadata)
    with pytest.raises(ValueError, match="one optimizer mini"):
        pull(sampler, part, contract, 0, mini=1)
    assert sampler.save_checkpoint() == before and part.metadata == metadata


def test_valid_sft_single_optimizer_mini_keeps_unequal_real_mb_counts_and_pickle_replay(tmp_path, monkeypatch):
    import pickle

    args, ctx, contract, sampler, _ = setup(tmp_path, capacity=10)
    part = partition([6, 6, 6, 1])
    seen = []
    real_counts = []
    monkeypatch.setattr("relax.utils.data.global_flops_pack_protocol.agree_cpu", lambda values, group: values)
    for rank in (1, 0):
        payload, _ = fetch_through_actor(args, ctx, contract, sampler, part, rank)
        actual, K, _ = backend(args, ctx, contract, rank, payload)
        assert K == [2] and len(K) == 1  # One optimizer mini, not two just because K is two.
        real = [mb for mb in actual if not mb.get("__is_dummy__")]
        real_counts.append(len(real))
        seen.extend(row for mb in real for row in mb["ids"])
        header = payload["_no16_global_pack_steps_v1"][0]["header"]
        assert [mb["ids"] for mb in actual] == [slot["rows"] for slot in header["slots"]]
        assert payload["rollout_mini_global_sample_counts"] == [4]
    assert sorted(real_counts) == [1, 2] and sorted(seen) == sorted(part.global_indexes)
    trusted = pickle.loads(pickle.dumps(sampler.save_checkpoint()))
    sampler.load_checkpoint(trusted)
    assert sampler.save_checkpoint() == trusted
    for rank in (0, 1):
        assert pull(sampler, part, contract, rank, ready=[])[0]


@pytest.mark.parametrize("damage", ["cohort", "rows", "lengths", "weights", "owner", "identity"])
def test_fixed_checkpoint_tampering_still_rejects_without_state_change(damage, tmp_path):
    _, _, contract, sampler, _ = setup(tmp_path, "rl")
    part = partition([1, 1, 1, 2])
    pull(sampler, part, contract, 0, task="train")
    before = sampler.save_checkpoint()
    bad = copy.deepcopy(before)
    key = ("p", "train", 0)
    plan = bad["plans"][key]
    changes = {
        "cohort": {"cohort_signature": "f" * 64},
        "rows": {"rows": tuple(reversed(plan.rows))},
        "lengths": {"lengths": (99, *plan.lengths[1:])},
        "weights": {"weights": (999.0, *plan.weights[1:])},
        "owner": {"rank_mbs": tuple(reversed(plan.rank_mbs))},
        "identity": {"identity": ("p", 1, "train", 0)},
    }
    bad["plans"][key] = replace(plan, **changes[damage])
    with pytest.raises(ValueError):
        sampler.load_checkpoint(bad)
    assert sampler.save_checkpoint() == before
