# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any

from transfer_queue.sampler.grpo_group_n_sampler import GRPOGroupNSampler
from transfer_queue.sampler.seqlen_balanced_sampler import get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


class StreamingTokenBudgetSampler(GRPOGroupNSampler):
    """Streaming sampler that returns samples up to a token budget per call.

    Designed for fully-async + dynamic-batch consumers that pull data from
    TransferQueue as a stream rather than waiting for the entire rollout.

    Semantics
    ---------
    Each call to :meth:`sample` is parameterised (via ``sampling_config`` /
    kwargs) by ``dp_rank`` and ``token_budget``. The sampler returns a
    contiguous slice of GRPO-complete prompt groups assigned to this
    ``dp_rank`` whose accumulated ``total_lengths`` reaches ``token_budget``.

    Internally the sampler maintains per-DP buckets of "assigned but not yet
    consumed" indexes. When a bucket cannot satisfy the budget, the sampler
    pulls the next ``balance_unit`` (= ``dp_size * n_samples_per_prompt * k``)
    GRPO-complete groups out of the currently-ready pool, runs a token-balanced
    partition across ``dp_size`` DP buckets, and tries again. Inside one
    balance unit, token totals are balanced across DPs; across units totals
    may differ (acceptable carry-over).

    When the ready pool cannot supply another full balance unit AND
    ``allow_underfill=True``, the sampler returns whatever remains in this
    DP's bucket (possibly underfill, possibly empty).

    Required ``custom_meta``: each sample must have
    ``{"total_lengths": <int>}`` populated by the producer at insert time.

    Required ``sampling_config`` keys:
        - ``dp_rank``: int, DP rank of the caller.
        - ``dp_size``: int, total DP world size.
        - ``token_budget``: int, target accumulated token count for this fetch.
        - ``allow_underfill``: bool, default True. If False, return ``([], [])``
          when budget cannot be reached (controller will retry).

    The ``partition`` kwarg (``DataPartitionStatus``) must be passed by the
    controller; the sampler uses ``partition.get_custom_meta`` to read
    ``total_lengths``.

    The ``batch_size`` argument is ignored (the budget governs slice size).
    """

    def __init__(
        self,
        n_samples_per_prompt: int = 1,
        balance_unit_multiplier: int = 1,
    ):
        """Create a streaming token-budget sampler.

        Args:
            n_samples_per_prompt: GRPO group size. Must be > 0.
            balance_unit_multiplier: A balance unit pulls
                ``balance_unit_multiplier * dp_size * n_samples_per_prompt``
                samples from the ready pool at a time. Larger values give
                better token balance but require waiting for more samples to
                be ready before any DP can progress. Default 1 = minimum unit.
        """
        super().__init__(n_samples_per_prompt=n_samples_per_prompt)
        if balance_unit_multiplier <= 0:
            raise ValueError(f"balance_unit_multiplier must be positive, got {balance_unit_multiplier}")
        self.balance_unit_multiplier = balance_unit_multiplier

        # Per (partition_id, task_name) state.
        # _buckets[(pid, tn)][dp_rank] -> list[int] of indexes assigned to this DP
        #                                 but not yet returned to the caller.
        self._buckets: dict[tuple[str, str], dict[int, list[int]]] = {}
        # _assigned_global[(pid, tn)] -> set[int] of indexes currently in any bucket
        # (used to filter ready_indexes coming from the controller, since they
        # are not yet marked consumed until the caller actually fetches them).
        self._assigned_global: dict[tuple[str, str], set[int]] = {}
        # _resolved_lengths[(pid, tn)] -> dict[idx, int] of total_lengths used
        # for token-budget accounting.  Populated in _run_balance_round from
        # custom_meta (with fallback to round-average when missing).  Reused
        # by _select_up_to_budget so a single sample sees a stable length
        # across calls even if custom_meta later changes.
        self._resolved_lengths: dict[tuple[str, str], dict[int, int]] = {}
        # _dispatched[(pid, tn)][window_id] -> int count of samples popped-and-
        # returned (globally, summed across DPs) for that rollout-mini window.
        # Governs the per-window global dispatch cap and is_window_drained, so a
        # window stops once mini_global_samples have been handed out across all
        # DPs, regardless of how unevenly the token-budget split them.
        self._dispatched: dict[tuple[str, str], dict[int, int]] = {}
        # _dummy_rounds[(pid, tn)] -> set of (dp_rank, batch_index) that are
        # "dummy rounds": at end-of-stream this dp's bucket was empty for this
        # batch_index but some OTHER dp still got a real slice that round. The
        # consumer emits a zero-grad dummy micro-batch for these so every dp runs
        # the SAME number of micro-batches (required for MoE EP all-to-all, which
        # matches collectives by call order across DP ranks).
        self._dummy_rounds: dict[tuple[str, str], set[tuple[int, int]]] = {}

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Return up to ``token_budget`` worth of samples assigned to ``dp_rank``.

        See class docstring for the streaming semantics.

        Fallback behaviour: when called WITHOUT ``token_budget`` (e.g. by
        :func:`compute_ref_log_prob` / :func:`compute_actor_log_prob` which
        use the legacy sample-count fetch), delegate to the inherited
        :class:`GRPOGroupNSampler.sample` so those consumers keep working
        unchanged on the same controller.
        """
        token_budget = kwargs.get("token_budget", None)
        if token_budget is None:
            # Strip kwargs that GRPO doesn't expect (extras supplied for the
            # streaming path by the controller / relax client).
            grpo_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("token_budget", "dp_size", "allow_underfill", "partition")
            }
            return super().sample(
                ready_indexes,
                batch_size,
                task_name=task_name,
                partition_id=partition_id,
                **grpo_kwargs,
            )

        dp_rank = kwargs.get("dp_rank", None)
        dp_size = kwargs.get("dp_size", None)
        allow_underfill = kwargs.get("allow_underfill", True)
        partition = kwargs.get("partition", None)
        batch_index = kwargs.get("batch_index", None)
        # production_done: the controller tells us when EVERY pre-allocated
        # sample of this partition has been produced (no more data is coming).
        # At that point the tail-flush may dump all remaining ready samples.
        production_done = bool(kwargs.get("production_done", False))
        # Per-window global dispatch cap (multi-mini streaming): window_id keys
        # the per-window dispatched counter; window_quota is the global sample
        # count (mini_global_samples) after which this window stops dispatching.
        # Both absent (None) on the legacy / fwd path → no cap, no counting.
        window_id = kwargs.get("rollout_mini_index", None)
        window_quota = kwargs.get("window_quota", None)

        if dp_rank is None or dp_size is None:
            raise ValueError(
                "StreamingTokenBudgetSampler requires dp_rank and dp_size in sampling_config "
                "when token_budget is provided"
            )

        # PP-stage cache: when multiple PP stages request the same
        # (partition_id, task_name, dp_rank, batch_index), return the
        # cached result from the first call so all stages see identical data.
        # A per-dp slice, once cached, is FROZEN — _prepare_batch_index below is
        # idempotent per (dp, batch_index), so a later re-prepare (to serve a dp
        # that had no data on the first round) never re-pops or overwrites a dp
        # already served.  This is what keeps every PP stage of a dp on the same
        # micro-batch; the previous code dropped the dp=0 guard on a cache miss,
        # which evicted dp=0's real slice and let a re-prepare hand different PP
        # stages different data.
        if batch_index is not None:
            cached = self._states.get(partition_id, {}).get(task_name, {}).get(dp_rank, {}).get(batch_index, None)
            if cached is not None:
                logger.debug(
                    "[stream-sampler] cache HIT: task=%s pid=%s dp=%s batch_idx=%s",
                    task_name,
                    partition_id,
                    dp_rank,
                    batch_index,
                )
                return cached

        if partition is None:
            raise ValueError("StreamingTokenBudgetSampler requires partition kwarg from the controller")

        # batch_index is the alignment key.  When it is present (always true on
        # the streaming train path), the FIRST request for a given batch_index
        # (from any dp_rank / PP stage) atomically prepares the micro-batch
        # slices for ALL dp_ranks against a single ``available_ready`` snapshot
        # and caches every dp's result.  Every subsequent request — other PP
        # stages of this dp, or other dp_ranks — hits the cache and gets the
        # identical, pre-determined data.  This keeps all dp_ranks in lockstep
        # by batch_index and removes the order-dependent state mutation that
        # broke PP>1 (orphaned-in-``assigned`` samples → under-consume/deadlock).
        if batch_index is not None:
            self._prepare_batch_index(
                partition_id,
                task_name,
                batch_index,
                dp_size,
                token_budget,
                allow_underfill,
                ready_indexes,
                partition,
                production_done,
                window_id,
                window_quota,
            )
            return self._states.get(partition_id, {}).get(task_name, {}).get(dp_rank, {}).get(batch_index, ([], []))

        # Fallback: no batch_index (should not happen on the streaming path) —
        # serve this single dp_rank immediately from shared state.
        return self._extract_one_dp(
            partition_id,
            task_name,
            dp_rank,
            dp_size,
            token_budget,
            allow_underfill,
            ready_indexes,
            partition,
            batch_index,
        )

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
        """Atomically prepare and cache one micro-batch slice for every dp_rank
        at ``batch_index``.

        Empty slices are NOT cached: if no data can be served for this
        batch_index yet (rollout still producing), we leave the cache untouched
        so the next poll re-evaluates against freshly produced samples.  Once a
        round yields real data, all participating dp_ranks' (non-empty) slices
        are cached together against a single ``available_ready`` snapshot, so
        every PP stage / dp request for this batch_index becomes a pure cache
        read with identical, pre-determined data.

        Early-return only when EVERY dp already has a cached slice for this
        batch_index.  A per-dp slice, once cached, is frozen: re-prepares (to
        serve a dp whose data arrived late) skip already-served dps, so all PP
        stages of a dp read identical data.
        """

        def _dp_served(dp_i: int) -> bool:
            # A (dp, batch_index) already holding a cached slice (real OR dummy)
            # is FROZEN: never re-pop or overwrite it, so every PP stage of that
            # dp reads identical data across re-prepares.
            return self._states.get(partition_id, {}).get(task_name, {}).get(dp_i, {}).get(batch_index) is not None

        # Skip work only when EVERY dp has already been served this batch_index.
        # A partial round (some dp still empty, e.g. its data arrived later) must
        # re-run to serve the stragglers — but idempotently, leaving the dps
        # already frozen above untouched.
        if all(_dp_served(dp_i) for dp_i in range(dp_size)):
            return

        key = (partition_id, task_name)
        buckets = self._buckets.setdefault(key, {})
        assigned = self._assigned_global.setdefault(key, set())
        resolved_lengths = self._resolved_lengths.setdefault(key, {})

        # Single shared snapshot for the whole batch_index across all dp_ranks.
        available_ready = [i for i in ready_indexes if i not in assigned]
        balance_unit = dp_size * self.n_samples_per_prompt * self.balance_unit_multiplier
        self._token_budget_for_fallback = token_budget

        def _bucket_tokens(dp_i: int) -> int:
            return sum(resolved_lengths.get(i, 0) for i in buckets.get(dp_i, []))

        is_eos = production_done

        logger.debug(
            "[stream-sampler] prepare batch_idx=%s task=%s pid=%s dp_size=%d budget=%d ready=%d avail=%d eos=%s",
            batch_index,
            task_name,
            partition_id,
            dp_size,
            token_budget,
            len(ready_indexes),
            len(available_ready),
            is_eos,
        )

        # ── Phase 1: balance rounds ──────────────────────────────────────
        # Pull balance rounds (each token-balances a chunk of complete GRPO
        # groups across ALL dp buckets) until every dp bucket holds a
        # token-budget worth or the ready pool runs dry.
        #
        # Round size: ideally a full ``balance_unit`` for best token balance,
        # but we must NOT require a full balance_unit to be ready before making
        # progress.  With long responses the producer trickles ~1 group at a
        # time and the slow per-mb consumer keeps ``ready`` pinned below
        # balance_unit (observed: stuck at ready=8 < balance_unit=16 forever).
        # So the minimum round is ONE complete group (``n_samples_per_prompt``),
        # split per-sample across dps.  This drains the trickle; token balance
        # is slightly worse on small rounds, which the dummy-pad tolerates.
        group = self.n_samples_per_prompt
        max_rounds = len(available_ready) // max(group, 1) + 2
        rounds = 0
        while len(available_ready) >= group and rounds < max_rounds:
            if all(_bucket_tokens(dp_i) >= token_budget for dp_i in range(dp_size)):
                break
            # Round size = as many full groups as are ready, capped at balance_unit.
            round_size = min(balance_unit, (len(available_ready) // group) * group)
            prev_avail = len(available_ready)
            if not self._run_balance_round(
                available_ready, round_size, dp_size, partition, buckets, assigned, resolved_lengths
            ):
                break
            available_ready = [i for i in available_ready if i not in assigned]
            rounds += 1
            if len(available_ready) >= prev_avail:
                break  # no progress — avoid spinning

        # ── Phase 2: unconditional end-of-stream tail flush ──────────────
        # Once the producer is DONE for this partition (is_eos) and a sub-
        # balance_unit remainder is still sitting in the ready pool, distribute
        # EVERY remaining ready sample across the dp buckets — bypassing the
        # GRPO-group and balance_unit constraints entirely.  This guarantees no
        # produced sample is ever orphaned (which would keep all_consumed False
        # forever → livelock).  GRPO group integrity is not needed here:
        # advantages are precomputed per-sample upstream.  Cross-dp imbalance
        # from this uneven flush is corrected by the iterator's dummy-pad.
        if is_eos and available_ready:
            self._flush_tail(available_ready, dp_size, partition, buckets, assigned, resolved_lengths)
            available_ready = [i for i in available_ready if i not in assigned]

        # ── Phase 3: slice one budget-sized mb per dp ────────────────────────
        # Pop only a token-budget slice (never the whole bucket — that builds an
        # oversized mb and OOMs on long sequences); residue drains over successive
        # batch_index rounds as the consumer advances each dp on non-empty fetches.
        # At EOS, cache an explicit empty result for an already-empty dp so the
        # dp=0 ``already`` guard stays set and batch_index isn't recomputed/frozen
        # (which would strand other dps' residue).
        #
        # Per-window global cap: never dispatch more than the window's remaining
        # quota (mini_global_samples minus what this window already handed out).
        # The last mb of a window is clamped to the exact remainder; anything
        # left in the buckets stays there (still in ``assigned``) and carries
        # over to the NEXT window's batch_indexes — buckets are keyed by
        # (partition, task), not by window.  Streaming and uneven per-dp counts
        # Per-window global cap: bound total dispatch to window_quota. Distribute
        # the (final-round) remainder FAIRLY across the dps that have data this
        # round — never skip later dps in loop order (that stranded dp2/dp3 and
        # deadlocked). Residue beyond the quota stays in buckets for the next
        # window (buckets are keyed by (partition, task), not by window).
        if window_quota is not None and window_id is not None:
            dispatched_so_far = self._dispatched.setdefault(key, {}).get(window_id, 0)
            remaining_quota: int | None = max(window_quota - dispatched_so_far, 0)
        else:
            remaining_quota = None

        dps_with_data = [dp_i for dp_i in range(dp_size) if buckets.setdefault(dp_i, [])]
        per_dp_cap: dict[int, int] | None = None
        if remaining_quota is not None and dps_with_data:
            base, extra = divmod(remaining_quota, len(dps_with_data))
            per_dp_cap = {dp_i: base + (1 if idx < extra else 0) for idx, dp_i in enumerate(dps_with_data)}

        popped_this_call = 0
        got_real = [False] * dp_size
        for dp_i in range(dp_size):
            if _dp_served(dp_i):
                continue  # idempotent: a dp already served this batch_index is frozen
            bucket = buckets.setdefault(dp_i, [])
            if not bucket:
                continue
            sel_count = self._select_up_to_budget(bucket, resolved_lengths, token_budget)
            sel_count = max(sel_count, 1)  # always make progress
            if per_dp_cap is not None:
                cap = per_dp_cap.get(dp_i, 0)
                if cap <= 0:
                    continue  # quota exhausted for this dp this round; residue carries over
                sel_count = min(sel_count, cap)
            result = self._pop_and_return(bucket, sel_count, assigned)
            if result[0]:
                self._cache_result(partition_id, task_name, dp_i, batch_index, result)
                popped_this_call += len(result[0])
                got_real[dp_i] = True

        # Count only what was freshly popped this call (once per batch_index via
        # the ``already`` guard; the EOS re-prepare only pops new residue), so
        # PP-stage replays never double-count.
        if window_id is not None and popped_this_call:
            window_counts = self._dispatched.setdefault(key, {})
            window_counts[window_id] = window_counts.get(window_id, 0) + popped_this_call

        # A window ends either because the producer is done (is_eos) OR because
        # this window's global quota is now fully dispatched. In the fully-async
        # case the producer runs ahead, so a window normally terminates by QUOTA
        # while production_done is still False. The consumer's END test
        # (is_window_drained) fires on quota alone, so the DUMMY padding below MUST
        # fire under the same condition — otherwise a dp whose bucket emptied one
        # round early ENDs unpadded and its micro-batch count falls short of a
        # busier dp (MoE EP all-to-all mismatch → cross-DP deadlock).
        window_now_drained = (
            window_id is not None
            and window_quota is not None
            and self._dispatched.get(key, {}).get(window_id, 0) >= window_quota
        )

        # ── Per-round DUMMY marking (end-of-stream / quota-drained) ──────────
        # This round produced a real slice for someone, but this dp's bucket was
        # empty → mark it a DUMMY round for that dp. The consumer emits a dummy
        # micro-batch and advances, so every dp ends at the SAME batch_index (the
        # first all-empty round = END) with equal micro-batch counts. Cache an
        # explicit empty so the dp=0 ``already`` guard keeps this batch_index from
        # being recomputed. (Before EOS/quota an empty bucket means "wait", not dummy.)
        if (is_eos or window_now_drained) and popped_this_call > 0:
            dummy_set = self._dummy_rounds.setdefault(key, set())
            for dp_i in range(dp_size):
                # Only a dp with NO cached slice this batch_index (genuinely empty
                # this round) becomes a dummy; never overwrite an already-served
                # (real) slice with an empty one.
                if not got_real[dp_i] and not _dp_served(dp_i):
                    dummy_set.add((dp_i, batch_index))
                    self._cache_result(partition_id, task_name, dp_i, batch_index, ([], []))

        logger.debug(
            "[stream-sampler] batch_idx=%s prepared: per-dp cached sizes=%s remaining_avail=%d eos=%s",
            batch_index,
            [
                len(self._states.get(partition_id, {}).get(task_name, {}).get(dp_i, {}).get(batch_index, ([],))[0])
                for dp_i in range(dp_size)
            ],
            len(available_ready),
            is_eos,
        )

    def _extract_one_dp(
        self,
        partition_id: str,
        task_name: str,
        dp_rank: int,
        dp_size: int,
        token_budget: int,
        allow_underfill: bool,
        ready_indexes: list[int],
        partition,
        batch_index: int | None,
    ) -> tuple[list[int], list[int]]:
        """Legacy single-dp extraction (fallback when batch_index is None)."""
        key = (partition_id, task_name)
        buckets = self._buckets.setdefault(key, {})
        bucket = buckets.setdefault(dp_rank, [])
        assigned = self._assigned_global.setdefault(key, set())
        resolved_lengths = self._resolved_lengths.setdefault(key, {})

        available_ready = [i for i in ready_indexes if i not in assigned]
        balance_unit = dp_size * self.n_samples_per_prompt * self.balance_unit_multiplier

        while True:
            if bucket:
                sel_count = self._select_up_to_budget(bucket, resolved_lengths, token_budget)
                if sel_count > 0:
                    cur_tokens = sum(resolved_lengths.get(i, 0) for i in bucket[:sel_count])
                    if cur_tokens >= token_budget:
                        result = self._pop_and_return(bucket, sel_count, assigned)
                        self._cache_result(partition_id, task_name, dp_rank, batch_index, result)
                        return result
                    if sel_count < len(bucket):
                        result = self._pop_and_return(bucket, sel_count, assigned)
                        self._cache_result(partition_id, task_name, dp_rank, batch_index, result)
                        return result

            if len(available_ready) >= balance_unit:
                self._token_budget_for_fallback = token_budget
                assigned_this_round = self._run_balance_round(
                    available_ready,
                    balance_unit,
                    dp_size,
                    partition,
                    buckets,
                    assigned,
                    resolved_lengths,
                )
                if assigned_this_round:
                    available_ready = [i for i in available_ready if i not in assigned]
                    bucket = buckets[dp_rank]
                    continue

            if allow_underfill and bucket:
                result = self._pop_and_return(bucket, len(bucket), assigned)
                self._cache_result(partition_id, task_name, dp_rank, batch_index, result)
                return result
            return [], []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_result(
        self,
        partition_id: str,
        task_name: str,
        dp_rank: int,
        batch_index: int | None,
        result: tuple[list[int], list[int]],
    ) -> None:
        """Store a sampling result so other PP stages can retrieve it."""
        if batch_index is None:
            return
        self._states.setdefault(partition_id, {}).setdefault(task_name, {}).setdefault(dp_rank, {})[batch_index] = (
            result
        )

    def _select_up_to_budget(self, bucket: list[int], resolved_lengths: dict[int, int], token_budget: int) -> int:
        """Return the largest prefix count k that packs into the budget.

        k is grown while ``sum(lengths[bucket[:k]]) <= token_budget``; the first
        sample that would push the running total OVER the budget stops the slice
        (so the returned slice never overshoots, avoiding oversized micro-batches).
        A single leading sample larger than the whole budget is still included
        (k>=1) so the stream always makes progress. Returns ``len(bucket)`` when
        the entire bucket fits. Reads per-sample lengths from ``resolved_lengths``
        (populated by :meth:`_run_balance_round` with custom_meta + fallback)."""
        if not bucket:
            return 0
        accum = 0
        for k, idx in enumerate(bucket):
            tl = resolved_lengths.get(idx, 0)
            # Always include at least one sample even if it alone exceeds budget
            # (otherwise we'd never make progress on oversized samples).
            if accum > 0 and accum + tl > token_budget:
                return k
            accum += tl
            if accum >= token_budget:
                return k + 1
        return len(bucket)

    def _pop_and_return(self, bucket: list[int], n: int, assigned: set) -> tuple[list[int], list[int]]:
        """Pop n items from the front of bucket, mark them as consumed
        (remove from assigned set), and return as (sampled, consumed)."""
        sampled = bucket[:n]
        del bucket[:n]
        assigned.difference_update(sampled)
        return sampled, sampled.copy()

    def _run_balance_round(
        self,
        available_ready: list[int],
        balance_unit: int,
        dp_size: int,
        partition,
        buckets: dict[int, list[int]],
        assigned: set,
        resolved_lengths: dict[int, int],
    ) -> bool:
        """Pull balance_unit GRPO-complete groups from available_ready, balance
        token totals across dp_size DP buckets, and append to per-DP buckets.

        Returns True if at least one balance round was completed."""
        # Use parent GRPO logic to find balance_unit complete groups.
        # We bypass the cache by using a unique task_name/partition_id for
        # this internal call (so it does not interfere with the consumer-facing
        # state cache of GRPOGroupNSampler).
        grpo_sampled, _ = super().sample(
            sorted(available_ready),
            balance_unit,
            task_name="__streaming_internal__",
            partition_id="__streaming_internal__",
        )
        if not grpo_sampled:
            return False

        # Read per-sample total_lengths.  If some samples in the round lack
        # total_lengths in custom_meta (producer race where rollout pushed
        # samples but set_custom_meta hasn't landed yet, or a producer that
        # skipped set_custom_meta entirely), fall back to the average of the
        # present samples so the round can still proceed.  This sacrifices
        # exact token-budget accuracy but avoids infinite-defer deadlock.
        custom_meta = partition.get_custom_meta(grpo_sampled)
        missing = [i for i in grpo_sampled if "total_lengths" not in custom_meta.get(i, {})]
        if missing:
            # Be conservative: assume missing samples are as large as the full
            # token budget (passed via kwargs).  Using avg or 0 would risk
            # packing many real-but-unknown long samples into one mb → OOM
            # (observed with avg fallback when an entire GRPO group lacks
            # custom_meta).  Over-estimating means each missing sample tends
            # to occupy a whole mb by itself, which is safe but inefficient.
            fallback = self._token_budget_for_fallback
            logger.warning(
                "[stream-sampler] %d/%d samples missing total_lengths in this "
                "round; using fallback=%d (token_budget) for safety "
                "(picked=%s missing=%s)",
                len(missing),
                len(grpo_sampled),
                fallback,
                grpo_sampled[:8],
                missing[:4],
            )
            sample_lengths = [custom_meta.get(i, {}).get("total_lengths", fallback) for i in grpo_sampled]
        else:
            sample_lengths = [custom_meta[i]["total_lengths"] for i in grpo_sampled]

        # Record the resolved lengths so _select_up_to_budget can reuse them
        # later without re-querying custom_meta (which may still race).
        for idx, tl in zip(grpo_sampled, sample_lengths, strict=False):
            resolved_lengths[idx] = tl

        # Per-sample balance across DPs.  GRPO group integrity is NOT required
        # at the training DP split: group-relative advantages are computed
        # upstream by the Advantages service and stored per-sample, so the
        # actor's loss (grpo/gspo/sapo) only reads per-sample advantages — it
        # never re-normalizes across a group.  We therefore balance individual
        # samples (not whole groups) across DPs, which keeps each DP's sample
        # count and token total close and minimizes the dummy-mb padding the
        # streaming schedule needs for cross-DP micro-batch alignment.
        #
        # ``equal_size=True`` forces equal sample COUNT per DP (balance_unit is
        # a multiple of dp_size), so each DP gets the same number of samples
        # with balanced token sums — making per-DP micro-batch counts equal in
        # the common case (token packing may still differ by one mb at the
        # margins, which the schedule's dummy-pad barrier handles).
        balanced = get_seqlen_balanced_partitions(sample_lengths, dp_size, equal_size=True)
        for dp_i, sample_idx_list in enumerate(balanced):
            dp_samples = [grpo_sampled[j] for j in sample_idx_list]
            buckets.setdefault(dp_i, []).extend(dp_samples)
            assigned.update(dp_samples)

        return True

    def _flush_tail(
        self,
        available_ready: list[int],
        dp_size: int,
        partition,
        buckets: dict[int, list[int]],
        assigned: set,
        resolved_lengths: dict[int, int],
    ) -> None:
        """End-of-stream flush: distribute ALL remaining ready samples across
        DP buckets, bypassing GRPO-group and balance_unit constraints.

        Called only when the producer is done for this partition.  Whatever is
        left in the ready pool — a sub-balance_unit remainder, even a partial
        GRPO group — is token-balanced across DPs and dumped into their buckets
        so every produced sample is guaranteed to be consumed (otherwise
        all_consumed never becomes True → livelock).  Group integrity is not
        needed (advantages are precomputed per-sample); cross-DP count
        imbalance from this uneven flush is handled by the iterator's dummy-pad.
        """
        leftover = sorted(i for i in available_ready if i not in assigned)
        if not leftover:
            return

        # Resolve lengths (custom_meta, with token-budget fallback for missing).
        custom_meta = partition.get_custom_meta(leftover)
        fallback = self._token_budget_for_fallback
        lengths = [custom_meta.get(i, {}).get("total_lengths", fallback) for i in leftover]
        for idx, tl in zip(leftover, lengths, strict=False):
            resolved_lengths[idx] = tl

        if len(leftover) >= dp_size:
            # Token-balance the leftover across all DPs (variable count per DP).
            parts = get_seqlen_balanced_partitions(lengths, dp_size, equal_size=False)
            for dp_i, idx_list in enumerate(parts):
                dp_samples = [leftover[j] for j in idx_list]
                buckets.setdefault(dp_i, []).extend(dp_samples)
                assigned.update(dp_samples)
        else:
            # Fewer leftover than DPs — round-robin; some DPs get nothing (the
            # dummy-pad will align them).
            for n, idx in enumerate(leftover):
                dp_i = n % dp_size
                buckets.setdefault(dp_i, []).append(idx)
                assigned.add(idx)

        logger.debug(
            "[stream-sampler] tail-flush %d leftover samples across %d dps",
            len(leftover),
            dp_size,
        )

    # ------------------------------------------------------------------
    # Cache / lifecycle
    # ------------------------------------------------------------------

    def is_window_drained(
        self,
        partition_id: str,
        task_name: str,
        window_id: int,
        window_quota: int | None,
        production_done: bool,
        partition_drained: bool,
    ) -> bool:
        """Per-window end-of-stream test.

        A rollout-mini window is drained once the sampler has globally dispatched
        its quota (``mini_global_samples``) across all DPs, OR the whole partition
        is drained.  The partition fallback covers an under-produced FINAL window
        whose quota can never be reached (fewer samples produced than expected).
        """
        dispatched = self._dispatched.get((partition_id, task_name), {}).get(window_id, 0)
        if window_quota is not None and dispatched >= window_quota:
            return True
        return bool(production_done and partition_drained)

    def is_dummy_round(self, partition_id: str, task_name: str, dp_rank: int, batch_index: int) -> bool:
        """True if (dp_rank, batch_index) is a DUMMY round: at end-of-stream this
        dp's bucket was empty for this batch_index but another dp still got a real
        slice that round. The consumer emits a zero-grad dummy micro-batch so all
        dps run the same number of micro-batches. Distinguishes an empty fetch that
        means 'pad a dummy' from one that means 'end of stream'."""
        return (dp_rank, batch_index) in self._dummy_rounds.get((partition_id, task_name), set())

    def window_dispatched(self, partition_id: str, task_name: str, window_id: int) -> int:
        """Total samples dispatched (== consumed) across ALL dps for this window.

        Lets the consumer log GLOBAL window progress (global_consumed / quota)
        without a cross-DP all_reduce — the sampler already tracks this count
        centrally for the per-window quota."""
        return self._dispatched.get((partition_id, task_name), {}).get(window_id, 0)

    def clear_cache(self, partition_id: str):
        """Drop all per-DP buckets and assignment tracking for this partition."""
        super().clear_cache(partition_id)
        keys_to_remove = [k for k in self._buckets if k[0] == partition_id]
        for k in keys_to_remove:
            del self._buckets[k]
        for k in list(self._assigned_global):
            if k[0] == partition_id:
                del self._assigned_global[k]
        for k in list(self._resolved_lengths):
            if k[0] == partition_id:
                del self._resolved_lengths[k]
        for k in list(self._dispatched):
            if k[0] == partition_id:
                del self._dispatched[k]
        for k in list(self._dummy_rounds):
            if k[0] == partition_id:
                del self._dummy_rounds[k]
        # Also clear the parent GRPO sampler's internal cache used by
        # _run_balance_round (keyed under "__streaming_internal__").
        if "__streaming_internal__" in self._states:
            del self._states["__streaming_internal__"]
