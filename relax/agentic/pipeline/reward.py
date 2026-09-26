# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward computation over the original exported Sample references."""

from __future__ import annotations

import asyncio
import copy
import time
from argparse import Namespace
from typing import Any, Callable, Iterable, Literal

from relax.agentic import format_agentic_status
from relax.agentic.pipeline import GroupExport, SessionExport
from relax.agentic.pipeline.runtime import RuntimeGroupStream
from relax.agentic.profile import mark_sample_agentic_event
from relax.engine.filters.base_types import MetricGatherer, call_dynamic_filter
from relax.engine.rewards import async_rm, batched_async_rm
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


class RewardDomain:
    """Compute every reward field on the exported Sample refs.

    Ordinary RM and group RM are alternative producers of ``Sample.reward``.
    Custom advantage is an optional post-process, and the group filter observes
    the final ordered Sample list.
    """

    def __init__(
        self,
        args: Namespace,
        rollout_mode: Literal["train", "eval"],
        group_filter: Callable[[list[Sample]], Any] | None,
    ) -> None:
        self.args = args
        self.group_rm = args.group_rm

        custom_advantage_path = args.agentic_custom_advantage_path if rollout_mode == "train" else None
        self.custom_advantage_func = None
        if custom_advantage_path is not None:
            from relax.utils.utils import load_function

            self.custom_advantage_func = load_function(custom_advantage_path)
        self._group_filter = group_filter if rollout_mode == "train" else None
        self._metric_gatherer = MetricGatherer()
        self._scored_session_ids: set[str] = set()
        self._progress_callback: Callable[[], None] = lambda: None

    @property
    def scored_session_count(self) -> int:
        return len(self._scored_session_ids)

    def set_progress_callback(self, callback: Callable[[], None]) -> None:
        self._progress_callback = callback

    def finish_step_progress(self, samples: Iterable[Sample]) -> None:
        self._scored_session_ids.difference_update(
            sample.session_id for sample in samples if sample.session_id is not None
        )

    def note_scored_sessions(self, stream: RuntimeGroupStream) -> None:
        self._add_scored_sessions(result.session_id for result in stream.session_results)

    def discard_group_progress(self, stream: RuntimeGroupStream) -> None:
        previous_count = len(self._scored_session_ids)
        self._scored_session_ids.difference_update(result.session_id for result in stream.session_results)
        if len(self._scored_session_ids) != previous_count:
            self._progress_callback()

    async def score_sessions(self, stream: RuntimeGroupStream) -> None:
        """Start ordinary RM as each Session export becomes available.

        A controlled Session drop ends this helper immediately, allowing the
        sibling Runtime gather in ``_run_group()`` to trigger Group cleanup.
        """

        tasks = tuple(
            asyncio.create_task(
                self._score_session(result.session_id, result.completion),
                name=f"agentic-session-reward:{result.session_id}",
            )
            for result in stream.session_results
        )
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                # Read the whole completion batch so a controlled drop cannot hide an RM failure.
                if None in [task.result() for task in done]:
                    return
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def score_group(self, group: GroupExport) -> None:
        """Run group RM after every Session export in the Group is complete."""

        samples = group.samples
        reward_start_at = time.time()
        for sample in samples:
            mark_sample_agentic_event(sample, "reward_arrive_at", reward_start_at)
            mark_sample_agentic_event(sample, "reward_start_at", reward_start_at)
        rewards = await batched_async_rm(self.args, samples)
        reward_end_at = time.time()
        for sample, reward in zip(samples, rewards, strict=True):
            sample.reward = reward
            mark_sample_agentic_event(sample, "reward_end_at", reward_end_at)

    async def apply_custom_advantage(self, group: GroupExport) -> bool:
        """Write per-export custom advantage values or reject the whole
        Group."""

        payload = [
            {export.name: copy.deepcopy(export.sample.metadata) for export in session.exports}
            for session in group.sessions
        ]
        result = self.custom_advantage_func(payload)
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            self._log_filtered_group()
            return False
        for session_result, session in zip(result, group.sessions, strict=True):
            for export in session.exports:
                value = session_result[export.name]
                if not isinstance(value, list):
                    export.sample.custom_advantage = float(value)
                    continue
                expanded = [0.0] * export.sample.response_length
                for turn_value, (start, end) in zip(value, export.sample.custom_advantage, strict=True):
                    expanded[start:end] = [float(turn_value)] * (end - start)
                export.sample.custom_advantage = expanded
        return True

    def finalize_group(self, group: GroupExport) -> GroupExport | None:
        """Apply the final global filter and expose complete Sample refs."""

        samples = group.samples
        finalize_start_at = time.time()
        for sample in samples:
            mark_sample_agentic_event(sample, "group_finalize_start_at", finalize_start_at)
        try:
            # Normalize the configured filter once, preserving its drop reason.
            filter_output = call_dynamic_filter(self._group_filter, samples)
            if not filter_output.keep:
                self._metric_gatherer.on_dynamic_filter_drop(filter_output.reason)
                self._log_filtered_group()
                return None
            return group
        finally:
            finalize_end_at = time.time()
            for sample in samples:
                mark_sample_agentic_event(sample, "group_finalize_end_at", finalize_end_at)

    def collect_metrics(self) -> dict[str, int]:
        metrics = self._metric_gatherer.collect()
        self._metric_gatherer = MetricGatherer()
        return metrics

    async def _score_session(
        self,
        session_id: str,
        completion: "asyncio.Future[SessionExport | None]",
    ) -> SessionExport | None:
        """Score every JSONL Sample exported by one completed Session."""

        outcome = await asyncio.shield(completion)
        if outcome is None:
            return None
        reward_arrive_at = time.time()
        for export in outcome.exports:
            mark_sample_agentic_event(export.sample, "reward_arrive_at", reward_arrive_at)
        await asyncio.gather(*(self._score_sample(export.sample) for export in outcome.exports))
        self._add_scored_sessions((session_id,))
        return outcome

    async def _score_sample(self, sample: Sample) -> None:
        """Reuse an exported reward or submit one ordinary RM call."""

        if sample.reward is None:
            mark_sample_agentic_event(sample, "reward_start_at")
            try:
                sample.reward = await async_rm(self.args, sample)
            finally:
                mark_sample_agentic_event(sample, "reward_end_at")

    def _add_scored_sessions(self, session_ids: Iterable[str]) -> None:
        previous_count = len(self._scored_session_ids)
        self._scored_session_ids.update(session_ids)
        if len(self._scored_session_ids) != previous_count:
            self._progress_callback()

    def _log_filtered_group(self) -> None:
        logger.info(format_agentic_status("Filtered 1 group"))
