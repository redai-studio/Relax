# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.data.identity_window_sampler import IdentityWindowSampler


class _Partition:
    def __init__(self, lengths: dict[int, int]) -> None:
        self.global_indexes = set(lengths)
        self._custom_meta = {
            index: {"sample_index": index, "total_lengths": length} for index, length in lengths.items()
        }

    def get_custom_meta(self, global_indexes: list[int]) -> dict[int, dict]:
        return {index: self._custom_meta[index] for index in global_indexes}


def test_identity_window_sampler_backfills_lagging_dp_dummy_round() -> None:
    """A DP that revisits an old empty round must not be blocked by window
    closure."""
    sampler = IdentityWindowSampler(dp_size=2, placement="streaming")
    partition = _Partition({0: 20, 1: 1, 2: 20, 3: 1, 4: 20, 5: 1, 6: 20, 7: 1})
    common = {
        "batch_size": 0,
        "task_name": "actor_train",
        "partition_id": "train_0",
        "token_budget": 10,
        "dp_size": 2,
        "partition": partition,
        "rollout_mini_index": 0,
        "window_quota": 8,
    }

    first_ready = [0, 1, 2, 3]
    assert sampler.sample(first_ready, dp_rank=0, batch_index=0, production_done=False, **common)[0] == [0]
    assert sampler.sample(first_ready, dp_rank=1, batch_index=0, production_done=False, **common)[0] == [1, 3]
    assert sampler.sample(first_ready, dp_rank=0, batch_index=1, production_done=False, **common)[0] == [2]
    assert sampler.sample(first_ready, dp_rank=1, batch_index=1, production_done=False, **common)[0] == []
    assert not sampler.is_dummy_round("train_0", "actor_train", 1, 1)

    all_ready = list(range(8))
    assert sampler.sample(all_ready, dp_rank=0, batch_index=2, production_done=False, **common)[0] == [4]
    assert sampler.sample(all_ready, dp_rank=0, batch_index=3, production_done=False, **common)[0] == [6]

    # DP1 is still waiting on index 1 when DP0 closes the window at index 3.
    # It must receive a dummy for the old hole before consuming its already
    # cached real data and trailing dummy from indexes 2 and 3.
    assert sampler.sample(all_ready, dp_rank=1, batch_index=1, production_done=False, **common)[0] == []
    assert sampler.is_dummy_round("train_0", "actor_train", 1, 1)
    assert sampler.sample(all_ready, dp_rank=1, batch_index=2, production_done=False, **common)[0] == [5, 7]
    assert sampler.sample(all_ready, dp_rank=1, batch_index=3, production_done=False, **common)[0] == []
    assert sampler.is_dummy_round("train_0", "actor_train", 1, 3)

    window = sampler._partition_task_states[("train_0", "actor_train")].windows[0]
    assert window.terminal_batch_by_dp == {0: 3, 1: 3}
    assert window.finalized_dps == {0, 1}
    assert sampler.is_window_drained("train_0", "actor_train", 0, 8, False, False)
