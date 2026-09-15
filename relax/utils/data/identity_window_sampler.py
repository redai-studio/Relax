# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal


try:
    from transfer_queue import StreamingTokenBudgetSampler
except ImportError as e:
    raise ImportError(
        "transfer_queue is out of date (missing StreamingTokenBudgetSampler). Upgrade with:\n"
        '    pip install "transferqueue @ git+https://github.com/redai-studio/'
        'TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" --no-deps\n'
        "or use the latest image."
    ) from e

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


@dataclass
class _WindowState:
    window_id: int
    logical_quota: int | None
    admitted_identities: list[int] = field(default_factory=list)
    rows_by_identity: dict[int, list[int]] = field(default_factory=dict)
    unplanned_rows_by_identity: dict[int, set[int]] = field(default_factory=dict)
    unserved_rows_by_identity: dict[int, set[int]] = field(default_factory=dict)
    completed_identities: set[int] = field(default_factory=set)
    served_cache_keys: set[tuple[int, int]] = field(default_factory=set)
    admission_closed: bool = False
    terminal_batch_by_dp: dict[int, int] = field(default_factory=dict)
    finalized_dps: set[int] = field(default_factory=set)

    @property
    def admitted_count(self) -> int:
        return len(self.admitted_identities)

    def can_close(self, dp_size: int) -> bool:
        return (
            self.admission_closed
            and all(not rows for rows in self.unplanned_rows_by_identity.values())
            and all(not rows for rows in self.unserved_rows_by_identity.values())
            and self.finalized_dps == set(range(dp_size))
        )


@dataclass
class _PartitionTaskState:
    windows: dict[int, _WindowState] = field(default_factory=dict)
    identity_to_window: dict[int, int] = field(default_factory=dict)
    row_to_identity: dict[int, int] = field(default_factory=dict)
    reserved_indexes: set[int] = field(default_factory=set)


class _BalancedRowPlacement:
    @staticmethod
    def place(rows: list[int], lengths: dict[int, int], dp_size: int) -> list[list[int]]:
        positions = get_seqlen_balanced_partitions(
            [lengths[index] for index in rows],
            dp_size,
            equal_size=len(rows) % dp_size == 0,
        )
        return [[rows[position] for position in partition] for partition in positions]


class _StreamingRowPlacement:
    @staticmethod
    def place(
        rows: list[int],
        lengths: dict[int, int],
        buckets: dict[int, list[int]],
        dp_size: int,
        priority_dps: list[int],
    ) -> None:
        row_order = {index: position for position, index in enumerate(rows)}
        if len(rows) >= dp_size:
            assignments = _BalancedRowPlacement.place(rows, lengths, dp_size)
        else:
            assignments = [[] for _ in range(dp_size)]
            available_dps = priority_dps + [rank for rank in range(dp_size) if rank not in priority_dps]
            for index, dp_rank in zip(rows, available_dps[: len(rows)], strict=True):
                assignments[dp_rank].append(index)

        for dp_rank, assigned_rows in enumerate(assignments):
            assigned_rows.sort(key=row_order.__getitem__)
            buckets[dp_rank].extend(assigned_rows)


class IdentityWindowSampler(StreamingTokenBudgetSampler):
    """Identity-aware RL sampler with physical-row placement across DPs."""

    def __init__(
        self,
        dp_size: int,
        placement: Literal["sequential", "balanced", "streaming"],
        balance_unit_multiplier: int = 1,
    ) -> None:
        super().__init__(n_samples_per_prompt=1, balance_unit_multiplier=balance_unit_multiplier)
        self.dp_size = dp_size
        self.placement = placement
        self._partition_task_states: dict[tuple[str, str], _PartitionTaskState] = {}
        self._cache_window_by_key: dict[tuple[str, str, int, int], int] = {}
        self._window_buckets: dict[tuple[str, str, int], dict[int, list[int]]] = {}
        self._window_resolved_lengths: dict[tuple[str, str, int], dict[int, int]] = {}

    def _cached_result(
        self,
        partition_id: str,
        task_name: str,
        dp_rank: int,
        batch_index: int,
    ) -> tuple[list[int], list[int]] | None:
        return self._states.get(partition_id, {}).get(task_name, {}).get(dp_rank, {}).get(batch_index)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        if kwargs.get("token_budget") is None:
            return self._sample_by_count(
                ready_indexes,
                batch_size,
                task_name,
                partition_id,
                **kwargs,
            )

        result = super().sample(
            ready_indexes,
            batch_size,
            *args,
            task_name=task_name,
            partition_id=partition_id,
            **kwargs,
        )
        dp_rank = kwargs.get("dp_rank")
        batch_index = kwargs.get("batch_index")
        if dp_rank is not None and batch_index is not None:
            self._mark_cache_served(partition_id, task_name, dp_rank, batch_index, result[0])
        return result

    def _sample_by_count(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str,
        partition_id: str,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        dp_rank = kwargs.get("dp_rank")
        batch_index = kwargs.get("batch_index")
        partition = kwargs["partition"]

        if dp_rank is None or batch_index is None:
            rows_by_identity, identity_order, _ = self._ready_identities(
                ready_indexes,
                partition,
                _PartitionTaskState(),
            )
            if len(identity_order) < batch_size:
                return [], []
            sampled = [index for identity in identity_order[:batch_size] for index in rows_by_identity[identity]]
            return sampled, sampled.copy()

        cached = self._cached_result(partition_id, task_name, dp_rank, batch_index)
        if cached is not None:
            self._mark_cache_served(partition_id, task_name, dp_rank, batch_index, cached[0])
            return cached

        task_state = self._task_state(partition_id, task_name)
        target_identity_count = batch_size * self.dp_size
        rows_by_identity, identity_order, custom_meta = self._ready_identities(
            ready_indexes,
            partition,
            task_state,
        )

        if self.placement != "balanced":
            if any(
                window_id < batch_index and not window.admission_closed
                for window_id, window in task_state.windows.items()
            ):
                return [], []
            if len(identity_order) < batch_size:
                return [], []

            window = self._window(task_state, batch_index, target_identity_count)
            selected_identities = identity_order[:batch_size]
            self._admit_identities(task_state, window, selected_identities, rows_by_identity)
            if window.admitted_count >= target_identity_count:
                window.admission_closed = True

            selected_rows = [index for identity in selected_identities for index in rows_by_identity[identity]]
            result = (selected_rows, selected_rows.copy())
            self._cache_result(partition_id, task_name, dp_rank, batch_index, result)
            self._cache_window_by_key[(partition_id, task_name, dp_rank, batch_index)] = window.window_id
            self._mark_rows_planned(window, selected_rows)
            window.terminal_batch_by_dp[dp_rank] = batch_index
            self._mark_cache_served(partition_id, task_name, dp_rank, batch_index, selected_rows)
            return result

        if len(identity_order) < target_identity_count:
            return [], []

        window = self._window(task_state, batch_index, target_identity_count)
        selected_identities = identity_order[:target_identity_count]
        self._admit_identities(task_state, window, selected_identities, rows_by_identity)
        window.admission_closed = True

        selected_rows = [index for identity in selected_identities for index in rows_by_identity[identity]]
        lengths = {index: int(custom_meta[index]["total_lengths"]) for index in selected_rows}
        rank_rows = _BalancedRowPlacement.place(selected_rows, lengths, self.dp_size)

        for rank, rows in enumerate(rank_rows):
            self._cache_result(partition_id, task_name, rank, batch_index, (rows, rows.copy()))
            self._cache_window_by_key[(partition_id, task_name, rank, batch_index)] = window.window_id
            self._mark_rows_planned(window, rows)
            window.terminal_batch_by_dp[rank] = batch_index

        result = self._cached_result(partition_id, task_name, dp_rank, batch_index)
        assert result is not None
        self._mark_cache_served(partition_id, task_name, dp_rank, batch_index, result[0])
        return result

    def _prepare_batch_index(
        self,
        partition_id: str,
        task_name: str,
        batch_index: int,
        dp_size: int,
        token_budget: int,
        allow_underfill: bool,
        ready_indexes: list[int],
        partition,
        production_done: bool = False,
        window_id: int | None = None,
        window_quota: int | None = None,
    ) -> None:
        del allow_underfill
        if all(self._cached_result(partition_id, task_name, rank, batch_index) is not None for rank in range(dp_size)):
            return

        resolved_window_id = 0 if window_id is None else window_id
        task_state = self._task_state(partition_id, task_name)
        window = self._window(task_state, resolved_window_id, window_quota)
        # A lagging DP may still need a dummy for an earlier uncached round.
        # Only indexes beyond the established terminal are guaranteed to be done.
        if len(window.terminal_batch_by_dp) == dp_size and batch_index > max(window.terminal_batch_by_dp.values()):
            return
        bucket_key = (partition_id, task_name, resolved_window_id)
        buckets = self._window_buckets.setdefault(bucket_key, {})
        resolved_lengths = self._window_resolved_lengths.setdefault(bucket_key, {})
        for rank in range(dp_size):
            buckets.setdefault(rank, [])

        if not window.admission_closed:
            rows_by_identity, identity_order, custom_meta = self._ready_identities(
                ready_indexes,
                partition,
                task_state,
            )
            identity_offset = 0

            def bucket_tokens(rank: int) -> int:
                return sum(resolved_lengths[index] for index in buckets[rank])

            while not window.admission_closed and not all(
                self._cached_result(partition_id, task_name, rank, batch_index) is not None
                or bucket_tokens(rank) >= token_budget
                for rank in range(dp_size)
            ):
                remaining_quota = None if window_quota is None else max(window_quota - window.admitted_count, 0)
                if remaining_quota == 0:
                    window.admission_closed = True
                    break

                max_admissions = dp_size * self.balance_unit_multiplier
                if remaining_quota is not None:
                    max_admissions = min(max_admissions, remaining_quota)
                selected_identities = identity_order[identity_offset : identity_offset + max_admissions]
                if not selected_identities:
                    if production_done:
                        window.admission_closed = True
                    break
                selected_rows = [index for identity in selected_identities for index in rows_by_identity[identity]]
                ranks_without_rows = sum(
                    self._cached_result(partition_id, task_name, rank, batch_index) is None and not buckets[rank]
                    for rank in range(dp_size)
                )
                closes_window = remaining_quota is not None and len(selected_identities) >= remaining_quota
                if len(selected_rows) < ranks_without_rows and not production_done and not closes_window:
                    break

                self._admit_identities(task_state, window, selected_identities, rows_by_identity)
                identity_offset += len(selected_identities)
                new_lengths = {index: int(custom_meta[index]["total_lengths"]) for index in selected_rows}
                resolved_lengths.update(new_lengths)
                _StreamingRowPlacement.place(
                    selected_rows,
                    new_lengths,
                    buckets,
                    dp_size,
                    [
                        rank
                        for rank in range(dp_size)
                        if self._cached_result(partition_id, task_name, rank, batch_index) is None
                        and not buckets[rank]
                    ],
                )
                self._mark_rows_planned(window, selected_rows)

                if window_quota is not None and window.admitted_count >= window_quota:
                    window.admission_closed = True
                elif production_done and identity_offset == len(identity_order):
                    window.admission_closed = True

        if (
            production_done
            and window.admission_closed
            and window_quota is not None
            and window.admitted_count < window_quota
        ):
            raise RuntimeError(
                "Identity window underproduced its fixed logical quota: "
                f"partition_id={partition_id!r}, task_name={task_name!r}, window_id={window.window_id}, "
                f"admitted={window.admitted_count}, quota={window_quota}."
            )

        for rank in range(dp_size):
            if self._cached_result(partition_id, task_name, rank, batch_index) is not None:
                continue
            bucket = buckets[rank]
            if not bucket:
                continue
            row_count = self._select_up_to_budget(bucket, resolved_lengths, token_budget)
            row_count = max(row_count, 1)
            rows = bucket[:row_count]
            del bucket[:row_count]
            result = (rows, rows.copy())
            self._cache_result(partition_id, task_name, rank, batch_index, result)
            self._cache_window_by_key[(partition_id, task_name, rank, batch_index)] = window.window_id

        has_real_slice = any(
            (result := self._cached_result(partition_id, task_name, rank, batch_index)) is not None and bool(result[0])
            for rank in range(dp_size)
        )
        if window.admission_closed and has_real_slice:
            dummy_rounds = self._dummy_rounds.setdefault((partition_id, task_name), set())
            for rank in range(dp_size):
                if self._cached_result(partition_id, task_name, rank, batch_index) is None:
                    dummy_rounds.add((rank, batch_index))
                    self._cache_result(partition_id, task_name, rank, batch_index, ([], []))
                    self._cache_window_by_key[(partition_id, task_name, rank, batch_index)] = window.window_id

        if window.admission_closed and all(not bucket for bucket in buckets.values()):
            for rank in range(dp_size):
                # Replaying an older hole must not move the terminal backwards.
                window.terminal_batch_by_dp.setdefault(rank, batch_index)
                if (rank, batch_index) in window.served_cache_keys:
                    window.finalized_dps.add(rank)

    def _task_state(self, partition_id: str, task_name: str) -> _PartitionTaskState:
        key = (partition_id, task_name)
        state = self._partition_task_states.get(key)
        if state is None:
            state = _PartitionTaskState()
            self._partition_task_states[key] = state
            self._assigned_global[key] = state.reserved_indexes
        return state

    @staticmethod
    def _window(
        task_state: _PartitionTaskState,
        window_id: int,
        logical_quota: int | None,
    ) -> _WindowState:
        window = task_state.windows.get(window_id)
        if window is None:
            window = _WindowState(window_id=window_id, logical_quota=logical_quota)
            task_state.windows[window_id] = window
        return window

    @staticmethod
    def _ready_identities(
        ready_indexes: list[int],
        partition,
        task_state: _PartitionTaskState,
    ) -> tuple[dict[int, list[int]], list[int], dict[int, dict[str, Any]]]:
        available = sorted(index for index in ready_indexes if index not in task_state.reserved_indexes)
        custom_meta = partition.get_custom_meta(available)
        all_custom_meta = partition.get_custom_meta(sorted(partition.global_indexes))
        identity_row_counts = Counter(int(meta["sample_index"]) for meta in all_custom_meta.values())
        rows_by_identity: dict[int, list[int]] = {}
        identity_order: list[int] = []
        for index in available:
            identity = int(custom_meta[index]["sample_index"])
            if identity not in rows_by_identity:
                identity_order.append(identity)
                rows_by_identity[identity] = []
            rows_by_identity[identity].append(index)
        identity_order = [
            identity for identity in identity_order if len(rows_by_identity[identity]) == identity_row_counts[identity]
        ]
        rows_by_identity = {identity: rows_by_identity[identity] for identity in identity_order}
        return rows_by_identity, identity_order, custom_meta

    @staticmethod
    def _admit_identities(
        task_state: _PartitionTaskState,
        window: _WindowState,
        identities: list[int],
        rows_by_identity: dict[int, list[int]],
    ) -> None:
        for identity in identities:
            if identity in task_state.identity_to_window:
                continue
            rows = list(rows_by_identity[identity])
            task_state.identity_to_window[identity] = window.window_id
            window.admitted_identities.append(identity)
            window.rows_by_identity[identity] = rows
            window.unplanned_rows_by_identity[identity] = set(rows)
            window.unserved_rows_by_identity[identity] = set(rows)
            for index in rows:
                task_state.row_to_identity[index] = identity
            task_state.reserved_indexes.update(rows)

    @staticmethod
    def _mark_rows_planned(window: _WindowState, rows: list[int]) -> None:
        remaining = set(rows)
        for identity in window.admitted_identities:
            identity_rows = window.unplanned_rows_by_identity[identity]
            identity_rows.difference_update(remaining)

    def _mark_cache_served(
        self,
        partition_id: str,
        task_name: str,
        dp_rank: int,
        batch_index: int,
        rows: list[int],
    ) -> None:
        cache_key = (partition_id, task_name, dp_rank, batch_index)
        window_id = self._cache_window_by_key.get(cache_key)
        if window_id is None:
            return
        task_state = self._partition_task_states[(partition_id, task_name)]
        window = task_state.windows[window_id]
        served_key = (dp_rank, batch_index)
        if served_key in window.served_cache_keys:
            return

        for index in rows:
            identity = task_state.row_to_identity[index]
            unserved = window.unserved_rows_by_identity[identity]
            unserved.discard(index)
            if not unserved:
                window.completed_identities.add(identity)
        window.served_cache_keys.add(served_key)
        self._dispatched.setdefault((partition_id, task_name), {})[window_id] = len(window.completed_identities)
        if window.terminal_batch_by_dp.get(dp_rank) == batch_index:
            window.finalized_dps.add(dp_rank)

    def is_window_drained(
        self,
        partition_id: str,
        task_name: str,
        window_id: int,
        window_quota: int | None,
        production_done: bool,
        partition_drained: bool,
    ) -> bool:
        del window_quota, production_done, partition_drained
        task_state = self._partition_task_states.get((partition_id, task_name))
        if task_state is None or window_id not in task_state.windows:
            return False
        window = task_state.windows[window_id]
        if window.admission_closed and window.admitted_count == 0:
            return True
        return window.can_close(self.dp_size)

    def save_checkpoint(self) -> dict:
        state = super().save_checkpoint()
        state.update(
            {
                "partition_task_states": self._partition_task_states,
                "cache_window_by_key": self._cache_window_by_key,
                "window_buckets": self._window_buckets,
                "window_resolved_lengths": self._window_resolved_lengths,
                "assigned_global": self._assigned_global,
                "resolved_lengths": self._resolved_lengths,
                "dispatched": self._dispatched,
                "dummy_rounds": self._dummy_rounds,
            }
        )
        return state

    def load_checkpoint(self, state: dict) -> None:
        super().load_checkpoint(state)
        self._partition_task_states = state["partition_task_states"]
        self._cache_window_by_key = state["cache_window_by_key"]
        self._window_buckets = state["window_buckets"]
        self._window_resolved_lengths = state["window_resolved_lengths"]
        self._assigned_global = state["assigned_global"]
        self._resolved_lengths = state["resolved_lengths"]
        self._dispatched = state["dispatched"]
        self._dummy_rounds = state["dummy_rounds"]
        for key, task_state in self._partition_task_states.items():
            self._assigned_global[key] = task_state.reserved_indexes

    def clear_cache(self, partition_id: str) -> None:
        super().clear_cache(partition_id)
        for mapping in (
            self._partition_task_states,
            self._window_buckets,
            self._window_resolved_lengths,
        ):
            for key in [key for key in mapping if key[0] == partition_id]:
                del mapping[key]
        for key in [key for key in self._cache_window_by_key if key[0] == partition_id]:
            del self._cache_window_by_key[key]
