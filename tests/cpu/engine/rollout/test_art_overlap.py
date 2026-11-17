# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART pipeline reward/update equivalence and unpublished reference
ownership."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from relax.engine.rewards import generative as gr
from relax.engine.rollout import native_generation as ng
from relax.engine.rollout.group_pipeline import GroupPipeline
from tests.cpu.engine.rollout.test_art_routing import _acquisition


@pytest.mark.parametrize("invalid", [(), (0,), (7,), (15,), tuple(range(16))])
def test_serial_and_prefetched_art_rewards_and_rows_are_identical(monkeypatch, invalid):
    consume = gr._consume_prefetched_rewards
    args, _rows, _caches, samples = _acquisition(monkeypatch)
    monkeypatch.setattr(gr, "_consume_prefetched_rewards", consume)
    args.reward_runtime = "cpu"
    calls = []

    def score(requests):
        indices = [r["index"] for r in requests]
        calls.extend(indices)
        return {
            "ocr": [1.0 if i >= 8 else float(i) for i in indices],
            "lpips": [0.25] * len(indices),
            "__valid_mask__": [i not in invalid for i in indices],
        }

    monkeypatch.setattr(gr, "_get_manager", lambda _args: SimpleNamespace(score=score))
    monkeypatch.setattr(gr, "_build_reward_request", lambda _args, sample: {"index": sample.index})
    monkeypatch.setattr(gr, "_release_memory_images", lambda _requests: None)
    serial, overlapped = deepcopy(samples), deepcopy(samples)
    serial_rows = ng.convert_samples_to_train_data(args, serial)
    assert sorted(calls) == list(range(16))
    calls.clear()
    pipeline = GroupPipeline(lambda group: group, lambda group: gr.prefetch_group_rewards(args, group), 2)
    try:
        for group in (overlapped[:8], overlapped[8:]):
            pipeline.reserve()
            pipeline.submit(group)
        pipeline.finish()
    finally:
        pipeline.close()
    overlap_rows = ng.convert_samples_to_train_data(args, overlapped)
    assert sorted(calls) == list(range(16))  # No duplicate scoring at the final barrier.
    assert serial_rows.keys() == overlap_rows.keys()
    for key in serial_rows.keys():
        torch.testing.assert_close(serial_rows[key], overlap_rows[key], rtol=0, atol=0)
    assert all(not hasattr(sample, "_prefetched_generative_reward") for sample in overlapped)


def test_failed_unpublished_round_releases_only_new_owned_refs_before_dropping_them(monkeypatch):
    import ray._private.internal_api as api

    old, current, next_round = object(), object(), object()
    cache = {(3, 0): old, (4, 0): current, (5, 0): next_round}
    monkeypatch.setattr(ng, "_TRAJECTORY_REF_CACHE", cache)
    freed = []

    def free(refs, *, local_only):
        assert cache[(4, 0)] is current  # Do not lose ownership before successful free.
        assert local_only is False
        freed.extend(refs)

    monkeypatch.setattr(api, "free", free)
    ng._discard_unpublished_art_refs(4, {(3, 0)})
    assert freed == [current] and cache == {(3, 0): old, (5, 0): next_round}


@pytest.mark.parametrize("release_error", [False, True])
def test_art_reward_failure_joins_workers_and_rolls_back_before_any_tq_put(monkeypatch, release_error):
    from threading import enumerate as threads

    import ray._private.internal_api as api

    old, cache, freed = object(), {}, []
    cache[(2, 0)] = old
    groups = [[ng.Sample(index=g * 2 + i, group_index=g, prompt="p") for i in range(2)] for g in range(2)]
    args = SimpleNamespace(
        sampling_config={
            "algorithm": "art",
            "wave_groups": 1,
            "overlap_group_processing": True,
            "overlap_pending_groups": 2,
            "art": {"train_step_indices": [0], "max_cache_bytes": 10000, "max_replay_bytes": 10000},
        },
        reward_runtime="cpu",
        reward_scorer_path="relax.engine.rewards.textpecker_reward.TextPeckerRewardScorer",
        custom_reward_post_process_path="relax.engine.rewards.generative.post_process",
        rollout_batch_size=2,
    )
    adapter = SimpleNamespace(build_rollout_request=lambda *_a, **_kw: {"height": 1, "width": 1})
    monkeypatch.setattr(ng, "_TRAJECTORY_REF_CACHE", cache)
    monkeypatch.setattr(ng, "_load_adapter", lambda _args: adapter)
    monkeypatch.setattr(ng, "_get_engines", lambda *_args: [object(), object()])
    monkeypatch.setattr(ng, "_get_samples", lambda *_args: groups)
    monkeypatch.setattr(ng, "prune_stale_artifacts", lambda *_args: None)
    monkeypatch.setattr(ng, "_dispatch_groups", lambda *_args, **_kwargs: [[{}, {}]])

    def prepare(_args, _sampling, _adapter, rollout, _fp, group, _responses, **_kwargs):
        cache[(rollout, group[0].group_index)] = object()
        return group

    def score(_args, group):
        for sample in group:
            sample._prefetched_generative_reward = (4, {"ocr": 1.0}, True)
        raise OSError("reward failed")

    def free(refs, *, local_only):
        assert not any(t.name.startswith(("rollout-prepare", "rollout-reward")) for t in threads())
        assert local_only is False and old not in refs
        if release_error:
            raise OSError("free failed")
        freed.extend(refs)

    monkeypatch.setattr(ng, "_prepare_generated_group", prepare)
    monkeypatch.setattr(gr, "prefetch_group_rewards", score)
    monkeypatch.setattr(api, "free", free)
    monkeypatch.setattr(
        "relax.utils.utils.transfer_batch_to_data_system", lambda *_a, **_k: pytest.fail("Early TQ put")
    )
    with pytest.raises(OSError, match="reward failed"):
        ng.generate_rollout(args, 4, object(), object())
    assert cache[(2, 0)] is old
    assert all(not hasattr(sample, "_prefetched_generative_reward") for group in groups for sample in group)
    if release_error:
        assert any(key[0] == 4 for key in cache)  # Retain refs if the global free failed.
    else:
        assert freed and cache == {(2, 0): old}
