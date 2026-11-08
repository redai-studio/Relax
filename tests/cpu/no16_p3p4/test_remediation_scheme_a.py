# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Scheme A: preserve admission, exercise the actual factory and iterator."""

from __future__ import annotations

import copy
import json

import pytest
from no16.harness import Partition
from no16.source import ROOT, load, samplers
from no16_p2.test_estimator import tiny_config

from relax.utils.data.flops_scheduling import row_weights, runtime_context

from .support import iterator
from .test_planning import args


def adapter_module():
    refs = samplers()
    module = load(
        ROOT / "relax/utils/data/flops_sampler.py",
        omitted_imports=("transfer_queue", "relax.utils.data.identity_window_sampler"),
        injected={
            "BaseSampler": refs["base"],
            "SeqlenBalancedSampler": refs["sft"],
            "IdentityWindowSampler": refs["rl"],
        },
    )
    return module, refs


def model_path(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
@pytest.mark.parametrize("balance_data", [False, True])
def test_rl_factory_balance_mode_matrix(mode, balance_data, tmp_path):
    module, refs = adapter_module()
    original = refs["rl"](dp_size=2, placement="balanced" if balance_data else "sequential")
    config = args(mode, loss_type="policy_loss", balance_data=balance_data, hf_checkpoint=model_path(tmp_path))
    state = copy.deepcopy(original.save_checkpoint())
    options = copy.deepcopy(vars(config))
    if not balance_data and mode in ("dp", "both"):
        with pytest.raises(ValueError, match="balance_data=True.*--balance-data.*Sequential"):
            module.configure_sampler(config, original, 2)
    else:
        selected = module.configure_sampler(config, original, 2)
        if mode in ("off", "microbatch"):
            assert selected is original
        else:
            assert isinstance(selected, refs["rl"])
            assert selected.placement == "balanced" and selected.row_placement is not None
            samples = [dict(id=str(i), row=100 + 7 * i, rollout_id=i // 2, length=i + 1) for i in range(8)]
            partition = Partition(samples)
            kw = dict(partition_id="p", task_name="train", partition=partition, batch_index=0)
            first = selected.sample(sorted(partition.global_indexes), 2, dp_rank=0, **kw)
            second = selected.sample([], 2, dp_rank=1, **kw)
            assert sorted(first[0] + second[0]) == sorted(partition.global_indexes)
            assert selected.sample([], 2, dp_rank=0, **kw) == first
    assert original.save_checkpoint() == state
    assert vars(config) == options


@pytest.mark.parametrize("mode", ["dp", "both"])
def test_rl_rejection_preserves_warm_window_cache_before_estimator_or_constructor(mode, tmp_path, monkeypatch):
    module, refs = adapter_module()
    original = refs["rl"](dp_size=2, placement="sequential")
    samples = [dict(id=str(i), row=100 + 7 * i, rollout_id=i // 2, length=i + 1) for i in range(8)]
    partition = Partition(samples)
    first = original.sample(
        sorted(partition.global_indexes),
        2,
        partition_id="p",
        task_name="train",
        partition=partition,
        batch_index=0,
        dp_rank=0,
    )
    assert first[0] and original._partition_task_states
    state = copy.deepcopy(vars(original))
    config = args(mode, loss_type="policy_loss", balance_data=False, hf_checkpoint=str(tmp_path / "missing-model"))
    options = copy.deepcopy(vars(config))
    calls = []

    def forbidden(*values, **options):
        calls.append((values, options))
        pytest.fail("Sequential RL must be rejected before estimator or new sampler construction")

    monkeypatch.setattr(module, "runtime_context", forbidden)
    monkeypatch.setattr(module, "IdentityWindowSampler", forbidden)
    with pytest.raises(ValueError, match="RL FLOPs-aware DP requires balance_data=True"):
        module.configure_sampler(config, original, 2)
    assert calls == []
    assert vars(original) == state
    assert vars(config) == options


def test_actual_iterator_has_fixed_p2_weighted_positive_example():
    lengths = [1, 1, 1, 2]
    active = runtime_context(args("microbatch"), tiny_config())
    # Independent P2 tiny hybrid arithmetic: 8136*L + 48*L^2.
    assert row_weights(active.estimator, lengths) == [8184, 8184, 8184, 16464]
    off, old_k, _, _ = iterator(lengths, capacity=3)
    weighted, new_k, _, _ = iterator(lengths, active, capacity=3)
    assert old_k == new_k == [2]
    assert [batch["ids"] for batch in off] == [[0, 3], [1, 2]]
    assert [batch["ids"] for batch in weighted] == [[0, 1, 2], [3]]
    for batches in (off, weighted):
        assert sorted(i for batch in batches for i in batch["ids"]) == list(range(4))
        assert all(sum(lengths[i] for i in batch["ids"]) <= 3 for batch in batches)
        assert all(not batch.get("__is_dummy__", False) for batch in batches)


@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
@pytest.mark.parametrize("n", [1, 2])
def test_sft_prompt_group_boundary_uses_production_factory_and_sampler(mode, n, tmp_path):
    module, refs = adapter_module()
    original = refs["sft"](dp_size=2, n_samples_per_prompt=n)
    config = args(
        mode,
        loss_type="sft",
        balance_data=False,
        n_samples_per_prompt=n,
        hf_checkpoint=model_path(tmp_path),
    )
    state = copy.deepcopy(original.save_checkpoint())
    if n > 1 and mode in ("dp", "both"):
        with pytest.raises(ValueError, match="n_samples_per_prompt=1"):
            module.configure_sampler(config, original, 2)
        assert original.save_checkpoint() == state
        return
    selected = module.configure_sampler(config, original, 2)
    if mode in ("off", "microbatch"):
        assert selected is original
    else:
        assert isinstance(selected, module.FlopsSftSampler)
        assert original.save_checkpoint() == state
    samples = [dict(id=str(i), row=100 + i, rollout_id=i, length=i + 1) for i in range(8)]
    partition = Partition(samples)
    kw = dict(partition_id="p", task_name="train", partition=partition, batch_index=0)
    rank_rows = [selected.sample(sorted(partition.global_indexes), 4, dp_rank=0, **kw)[0]]
    rank_rows.append(selected.sample([], 4, dp_rank=1, **kw)[0])
    assert sorted(i for rows in rank_rows for i in rows) == sorted(partition.global_indexes)
    assert selected.sample([], 4, dp_rank=0, **kw)[0] == rank_rows[0]
    if n == 2:
        for start in range(100, 108, 2):
            assert any({start, start + 1} <= set(rows) for rows in rank_rows)


@pytest.mark.parametrize("mode", ["off", "microbatch", "dp", "both"])
def test_balanced_backend_retains_existing_collective_order(mode):
    active = None if mode == "off" else runtime_context(args(mode), tiny_config())
    _, k, reductions, _ = iterator([3, 4], active, dp=2, balance_data=True)
    assert k == [1]
    expected = [[1], [1]]  # Existing balance_data step MAX, then existing K MAX.
    if mode != "off":
        expected.insert(0, [0, 1, -1, 4, -4])  # Existing enabled metadata agreement.
    assert reductions == expected
