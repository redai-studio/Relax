# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Resident agentic rollout control flow.

Prepare, Runtime, Reward, and Transfer own their domain state. The Pipeline
owns lifecycle tasks and expresses one rollout step as collect -> close check
-> lease -> wait.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from argparse import Namespace
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple, cast

from tqdm import tqdm

from relax.agentic import format_agentic_event
from relax.agentic.pipeline import (
    GroupExport,
    GroupInput,
)
from relax.agentic.pipeline.prepare import PrepareDomain
from relax.agentic.pipeline.reward import RewardDomain
from relax.agentic.pipeline.runtime import (
    RuntimeDomain,
    RuntimeGroupStream,
    agentic_eval_concurrency_from_args,
    finish_before_cancellation,
    load_agentic_compiler_resources,
)
from relax.agentic.pipeline.transfer import TransferBatch, TransferDomain, _transfer_batch_to_data_system
from relax.agentic.profile import TRACE_KEY
from relax.distributed.ray.rollout import _log_rollout_data
from relax.engine.rollout import on_policy_distillation as opd
from relax.engine.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import finalize_rollout_explicit_metric_values
from relax.utils.profile_utils import start_sglang_profile, stop_sglang_profile
from relax.utils.training.eval_config import EvalDatasetConfig
from relax.utils.training.train_dump_utils import save_debug_rollout_data
from relax.utils.types import Sample
from relax.utils.utils import compute_dp_size


_RESIDENT_PIPELINE: Optional["AgenticResidentPipeline"] = None
_RESIDENT_ENTRY_LOCK = threading.Lock()
_RESIDENT_ASYNC_LOCK = threading.Lock()
_RESIDENT_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_RESIDENT_ASYNC_THREAD: threading.Thread | None = None
_METADATA_METRIC_SKIP = {TRACE_KEY, "id", "index", "sample_index", "group_index", "start_rollout_id", "rollout_turns"}
logger = get_logger(__name__)
_IDLE_HEARTBEAT_INTERVAL_S = 30.0


@dataclass
class _StepContext:
    """References owned by one rollout step."""

    rollout_id: int
    final_backfill: bool
    output_groups: List[List[Sample]] = field(default_factory=list)
    tq_put_tasks: List[asyncio.Task[None]] = field(default_factory=list)
    progress: Any = None
    scored: int = 0
    prepared: int = 0


async def _run_group(
    runtime_domain: RuntimeDomain,
    reward_domain: RewardDomain,
    stream: RuntimeGroupStream,
    opd_manager: opd.OpdManager | None = None,
) -> Optional[GroupExport]:
    """Run one Group through Runtime and the configured Reward barrier."""

    requires_complete_group = reward_domain.group_rm or reward_domain.custom_advantage_func is not None
    try:
        if requires_complete_group:
            group = await runtime_domain.collect_group(stream)
        else:
            group, _ = await asyncio.gather(
                runtime_domain.collect_group(stream),
                reward_domain.score_sessions(stream),
            )
        if group is None:
            reward_domain.discard_group_progress(stream)
            await runtime_domain.drop_group(stream)
            return None

        if reward_domain.group_rm:
            await reward_domain.score_group(group)
        if reward_domain.custom_advantage_func is not None and not await reward_domain.apply_custom_advantage(group):
            reward_domain.discard_group_progress(stream)
            return None
        if requires_complete_group:
            reward_domain.note_scored_sessions(stream)

        scored_group = reward_domain.finalize_group(group)
        if scored_group is None:
            reward_domain.discard_group_progress(stream)
            return None
        if opd_manager is None:
            return scored_group
        from relax.engine.rollout.sglang_rollout import _encode_multimodal_inputs

        await opd_manager.prefill(scored_group.samples, _encode_multimodal_inputs)
        return scored_group
    except BaseException:
        reward_domain.discard_group_progress(stream)
        await runtime_domain.drop_group(stream)
        raise


class AgenticResidentPipeline:
    """Resident coordinator of Domains, Group tasks, and finalized Group
    refs."""

    def __init__(
        self,
        args: Namespace,
        prepare_domain: PrepareDomain,
        runtime_domain: RuntimeDomain,
        reward_domain: RewardDomain,
        transfer_domain: TransferDomain,
        data_system_client: Any,
    ) -> None:
        self.args = args
        self.prepare_domain = prepare_domain
        self.runtime_domain = runtime_domain
        self.reward_domain = reward_domain
        self.transfer_domain = transfer_domain
        self._data_system_client = data_system_client
        self._opd_manager = opd.OpdManager(args) if opd.is_opd_enabled(args) else None

        self._changed = asyncio.Event()
        self.prepare_domain.set_progress_callback(self._changed.set)
        self.runtime_domain.set_progress_callback(self._changed.set)
        self.reward_domain.set_progress_callback(self._changed.set)
        self._active_group_tasks: List[asyncio.Task[Optional[GroupExport]]] = []
        self._finalized_groups: Deque[GroupExport] = deque()
        self._active_step: Optional[_StepContext] = None

    def snapshot(self) -> dict[str, object]:
        """Project debug state from owned refs without feeding it into
        control."""

        context = self._active_step
        warming, ready = self.prepare_domain.counts
        runtime_reward = len(self._active_group_tasks)
        finalized = len(self._finalized_groups)
        return {
            "active_rollout_id": context.rollout_id if context is not None else None,
            "warming_prepare_group_count": warming,
            "ready_prepare_group_count": ready,
            "prepare_group_count": warming + ready,
            "runtime_reward_group_count": runtime_reward,
            "finalized_group_count": finalized,
            "resident_group_count": self.runtime_domain.resident_group_count,
            "interrupted_runtime_group_count": len(self.runtime_domain.interrupted_group_ids),
        }

    async def run_step(self, rollout_id: int) -> RolloutFnTrainOutput:
        """Generate one rollout step while resident Group refs may cross
        steps."""

        started_at = time.monotonic()
        await start_sglang_profile(self.args, rollout_id)
        profile_stopped = False
        step_progress_finished = False
        final_backfill = self.args.fully_async and rollout_id >= self.args.num_rollout
        context = _StepContext(
            rollout_id=rollout_id,
            # Regular rollout IDs are [0, num_rollout). Fully async uses the
            # following ID to drain physical debt without opening a partition.
            final_backfill=final_backfill,
        )
        self._active_step = context
        try:
            self.prepare_domain.open_step(
                prelaunch_next_step=self.args.agentic_prelaunch and rollout_id + 1 < self.args.num_rollout
            )
            output = await self._run_rollout_step(context)
            self._finish_step_progress(context)
            step_progress_finished = True

            samples = [sample for group in output.samples for sample in group]
            metrics = dict(output.metrics or {})
            metrics.update(_aggregate_agentic_timing(samples))
            metrics.update(self.reward_domain.collect_metrics())
            if getattr(self.args, "agentic_program_admission", False) or getattr(
                self.args, "agentic_session_lifecycle", False
            ):
                metrics.update(await self.runtime_domain.collect_agentic_kv_metrics())
            for key, value in _collect_agentic_metadata_metrics(samples).items():
                metrics.setdefault(key, value)

            await stop_sglang_profile(self.args, rollout_id)
            profile_stopped = True
            rollout_time = time.monotonic() - started_at
            if samples:
                last_sample = samples[-1]
                logger.info(
                    "Finish rollout: %s, label: %s, reward: %s",
                    [str(last_sample.prompt) + last_sample.response],
                    str(last_sample.label)[:100],
                    last_sample.reward,
                )
                _save_train_debug_rollout_data(self.args, rollout_id, samples)
            _log_rollout_data(
                rollout_id,
                self.args,
                samples,
                metrics,
                rollout_time,
            )
            if self.args.debug_rollout_only:
                if self._data_system_client is None:
                    raise RuntimeError("Agentic debug rollout cleanup requires a data system client.")
                partition_rollout_id = rollout_id - 1 if final_backfill else rollout_id
                await self._data_system_client.async_clear_partition(partition_id=f"train_{partition_rollout_id}")
            output.metrics = metrics
            logger.info(
                format_agentic_event(
                    "ROLLOUT",
                    "accounting_end",
                    rollout=rollout_id,
                    groups=len(context.output_groups),
                    rows=len(samples),
                    debt=self.transfer_domain.total_debt,
                    resident_groups=self.runtime_domain.resident_group_count,
                    interrupted_groups=len(self.runtime_domain.interrupted_group_ids),
                )
            )
        finally:
            if not step_progress_finished:
                self._finish_step_progress(context)
            if not profile_stopped:
                await stop_sglang_profile(self.args, rollout_id)
        self._active_step = None
        return output

    def _finish_step_progress(self, context: _StepContext) -> None:
        if context.progress is not None:
            context.progress.close()
            context.progress = None
        self.reward_domain.finish_step_progress(sample for group in context.output_groups for sample in group)

    async def shutdown(self) -> None:
        """Permanently release all resident work and borrowed Runtime
        resources."""

        await finish_before_cancellation(self._shutdown(), "agentic-pipeline-shutdown")

    async def _run_rollout_step(self, context: _StepContext) -> RolloutFnTrainOutput:
        """Express rollout control as collect -> close -> lease -> fill ->
        wait."""

        if not context.final_backfill:
            self.transfer_domain.open_partition(
                context.rollout_id,
                self.args.rollout_batch_size,
                accepts_surplus=self.args.use_dynamic_global_batch_size,
            )
        target_groups = self.transfer_domain.total_debt
        target_sessions = target_groups * self.args.n_samples_per_prompt
        logger.info(
            format_agentic_event(
                "ROLLOUT",
                "accounting_start",
                rollout=context.rollout_id,
                sessions=target_sessions,
                groups=target_groups,
                final_backfill=context.final_backfill,
            )
        )
        context.progress = tqdm(
            total=target_sessions,
            desc=f"Rollout {context.rollout_id} generation",
            unit="session",
        )
        context.scored = self.reward_domain.scored_session_count
        postfix = f"scored={context.scored}"
        if self.args.agentic_prelaunch:
            context.prepared = self.prepare_domain.ready_group_count
            postfix += f", prepared={context.prepared}"
        context.progress.set_postfix_str(postfix, refresh=True)
        await self.runtime_domain.resume_generation(context.rollout_id)

        while True:
            self._changed.clear()
            self._collect_step_progress(context)
            if self._step_can_close(context):
                self._refresh_progress(context)
                return await self._close_rollout_step(context)
            await self._lease_prepared_groups(context)
            self.prepare_domain.fill(self._runtime_group_gap(context))
            self._refresh_progress(context)
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=_IDLE_HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                logger.info(
                    format_agentic_event(
                        "ROLLOUT",
                        "idle_heartbeat",
                        rollout=context.rollout_id,
                        debt=self.transfer_domain.total_debt,
                        state=self.snapshot(),
                    )
                )

    def _collect_step_progress(self, context: _StepContext) -> None:
        """Move completed Prepare, Group, and finalized refs forward."""

        for task in tuple(context.tq_put_tasks):
            if task.done():
                context.tq_put_tasks.remove(task)
                task.result()
        self.prepare_domain.collect()

        self._collect_finalized_groups()
        while self._finalized_groups and self.transfer_domain.total_debt > 0:
            group = self._finalized_groups.popleft()
            if not context.output_groups:
                sample = group.samples[0]
                logger.info(
                    "First rollout sample: %s, label: %s, reward: %s",
                    [str(sample.prompt) + sample.response],
                    str(sample.label)[:100],
                    sample.reward,
                )
            context.output_groups.append(group.samples)
            self.transfer_domain.accept(group)
        for batch in self.transfer_domain.detach_ready_batches():
            self._submit_tq_batch(context, batch)

    def _refresh_progress(self, context: _StepContext) -> None:
        progress = context.progress
        if progress is None:
            return
        group_size = self.args.n_samples_per_prompt
        completed_groups = len(context.output_groups) + len(self._finalized_groups)
        completed_sessions = completed_groups * group_size
        visible_sessions = min(completed_sessions, progress.total)
        scored = self.reward_domain.scored_session_count
        prepared = self.prepare_domain.ready_group_count if self.args.agentic_prelaunch else 0
        if visible_sessions == progress.n and scored == context.scored and prepared == context.prepared:
            return
        progress_delta = visible_sessions - progress.n
        context.scored = scored
        context.prepared = prepared
        postfix = f"scored={scored}"
        if self.args.agentic_prelaunch:
            postfix += f", prepared={prepared}"
        progress.set_postfix_str(postfix, refresh=False)
        if progress_delta > 0:
            progress.update(progress_delta)
        else:
            progress.refresh()

    def _submit_tq_batch(self, context: _StepContext, batch: TransferBatch) -> None:
        """Create one TQ transfer call owned by the active step."""

        rollout_id, groups, is_last = batch
        task = asyncio.create_task(
            _transfer_batch_to_data_system(
                args=self.args,
                batch_samples=[group.samples for group in groups],
                rollout_id=rollout_id,
                data_system_client=self._data_system_client,
                is_last=is_last,
            ),
            name=f"agentic-transfer:train_{rollout_id}:{len(groups)}-groups",
        )
        task.add_done_callback(lambda _task: self._changed.set())
        context.tq_put_tasks.append(task)

    async def _await_tq_puts(self, context: _StepContext) -> None:
        """Cross the TQ durability barrier for batches owned by this step."""

        if context.tq_put_tasks:
            await asyncio.gather(*context.tq_put_tasks)
            context.tq_put_tasks.clear()

    def _collect_finalized_groups(self) -> None:
        """Move completed Group task results into the finalized FIFO."""

        for task in tuple(self._active_group_tasks):
            if not task.done():
                continue
            self._active_group_tasks.remove(task)
            outcome = task.result()
            if outcome is not None:
                self._finalized_groups.append(outcome)

    def _step_can_close(self, context: _StepContext) -> bool:
        """Read close eligibility from physical debt and resident Group
        refs."""

        debt = self.transfer_domain.total_debt
        if debt == 0:
            return True

        # Fully async may close with unfinished physical debt only when the
        # same number of resident Groups is confirmed interrupted.
        if self.args.fully_async and not context.final_backfill:
            return debt <= len(self.runtime_domain.interrupted_group_ids)
        return False

    async def _close_rollout_step(self, context: _StepContext) -> RolloutFnTrainOutput:
        """Seal one step, await its TQ tasks, and apply resident-tail
        policy."""

        self.prepare_domain.close_step()
        dynamic_close = self.args.use_dynamic_global_batch_size
        # Normal partitions are already sealed, so their partial batch can be
        # flushed now. A dynamic current partition stays unsealed until its
        # aligned surplus is known.
        if not dynamic_close:
            for batch in self.transfer_domain.flush_batches():
                self._submit_tq_batch(context, batch)
        # Dynamic close has two durability barriers: base writes, then pause,
        # completed aligned surplus, seal, and the remaining writes.
        await self._await_tq_puts(context)
        self.transfer_domain.release_finished_partitions()

        await finish_before_cancellation(
            self.runtime_domain.pause_generation(),
            "agentic-generation-pause",
        )
        retain_resident_tail = not context.final_backfill and (self.args.partial_rollout or self.args.fully_async)
        if not retain_resident_tail:
            self._finalized_groups.clear()
            await self._cancel_active_group_tasks()
        elif dynamic_close:
            # Generation is now paused. Harvest Group tasks already complete,
            # append their aligned FIFO prefix, seal the partition, and wait for
            # its remaining TQ batches. Reward-inflight Groups stay resident.
            await self._seal_dynamic_partition(context)

        await self.runtime_domain.trim_memory()
        return RolloutFnTrainOutput(samples=context.output_groups, metrics={})

    async def _seal_dynamic_partition(self, context: _StepContext) -> None:
        """Seal with the DP-aligned FIFO prefix of completed Group tasks."""

        self._collect_finalized_groups()

        data_parallel_size = compute_dp_size(self.args)
        target_groups = self.args.rollout_batch_size
        candidate_count = target_groups + len(self._finalized_groups)
        aligned_count = candidate_count - candidate_count % data_parallel_size
        surplus = ()
        if aligned_count > target_groups:
            surplus = tuple(self._finalized_groups.popleft() for _ in range(aligned_count - target_groups))
        if context.progress is not None:
            context.progress.total += len(surplus) * self.args.n_samples_per_prompt
        context.output_groups.extend(group.samples for group in surplus)
        self._refresh_progress(context)
        for batch in self.transfer_domain.finish_current_partition(surplus):
            self._submit_tq_batch(context, batch)
        await self._await_tq_puts(context)
        self.transfer_domain.release_finished_partitions()

    async def _lease_prepared_groups(self, context: _StepContext) -> None:
        """Open generation for the ready FIFO prefix visible at this check.

        Each ready task already owns a Shard-resident Group whose Sessions have
        crossed the first-IR barrier. Leasing changes ownership to Runtime and
        opens its generation gate.
        """

        for stream in self.prepare_domain.take_ready(self._runtime_group_gap(context)):
            await self.runtime_domain.lease_group(stream, context.rollout_id)
            group_task = asyncio.create_task(
                _run_group(self.runtime_domain, self.reward_domain, stream, self._opd_manager),
                name=f"agentic-group:{stream.group_id}",
            )
            group_task.add_done_callback(lambda _task: self._changed.set())
            self._active_group_tasks.append(group_task)

    def _runtime_group_gap(self, context: _StepContext) -> int:
        oversample_groups = 0
        if not context.final_backfill:
            oversample_groups = self.args.over_sampling_batch_size - self.args.rollout_batch_size
        return (
            self.transfer_domain.total_debt
            + oversample_groups
            - len(self._active_group_tasks)
            - len(self._finalized_groups)
        )

    async def _shutdown(self) -> None:
        """Cancel and join active-step TQ writes, then release every owner."""

        self._finalized_groups.clear()
        first_error: Optional[BaseException] = None
        context = self._active_step
        tq_put_tasks: Tuple[asyncio.Task[None], ...] = ()
        if context is not None:
            tq_put_tasks = tuple(context.tq_put_tasks)
        for task in tq_put_tasks:
            task.cancel()
        for outcome in await asyncio.gather(*tq_put_tasks, return_exceptions=True):
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                first_error = outcome
                break

        self.transfer_domain.clear()
        for cleanup in (
            self.prepare_domain.shutdown,
            self._cancel_active_group_tasks,
            self.runtime_domain.shutdown,
        ):
            try:
                await cleanup()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    async def _cancel_active_group_tasks(self) -> None:
        """Drop resident Runtime/Reward tasks when the mode cannot retain
        them."""

        tasks = tuple(self._active_group_tasks)
        self._active_group_tasks.clear()
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                raise outcome


async def _init_agentic_resident_pipeline_async(
    args: Namespace,
    data_source: Any,
    data_system_client: Any,
) -> None:
    """Build the Pipeline owned by one RolloutManager.

    RolloutManager calls this exactly once per resident lifecycle. Reward owns
    RM/custom loading; the composition root binds the stable metrics-aware
    filter.
    """

    global _RESIDENT_PIPELINE
    filter_path = args.dynamic_sampling_filter_path
    if filter_path is None:
        group_filter = None
    else:
        from relax.utils.utils import load_function

        filter_function = load_function(filter_path)

        def group_filter(samples: list[Sample]) -> Any:
            return filter_function(args, samples)

    runtime_domain = None
    try:
        reward_domain = RewardDomain(args, "train", group_filter)
        transfer_domain = TransferDomain(args=args)
        concurrency = args.agentic_concurrency
        runtime_domain = await RuntimeDomain.connect(args, "train", concurrency)
        prepare_domain = PrepareDomain(
            "train",
            data_source,
            runtime_domain,
        )
        pipeline = AgenticResidentPipeline(
            args=args,
            prepare_domain=prepare_domain,
            runtime_domain=runtime_domain,
            reward_domain=reward_domain,
            transfer_domain=transfer_domain,
            data_system_client=data_system_client,
        )
        _RESIDENT_PIPELINE = pipeline
    except BaseException:
        if runtime_domain is not None:
            await runtime_domain.shutdown()
        raise


async def _shutdown_agentic_resident_pipeline_async() -> None:
    """Release one initialized RolloutManager-local resident Pipeline."""

    global _RESIDENT_PIPELINE
    pipeline = _RESIDENT_PIPELINE
    _RESIDENT_PIPELINE = None
    if pipeline is not None:
        await pipeline.shutdown()


async def _eval_groups(
    eval_id: int,
    scope_id: str,
    groups: Tuple[Tuple[Sample, ...], ...],
    runtime_domain: RuntimeDomain,
    reward_domain: RewardDomain,
    progress: Any,
) -> Tuple[GroupExport, ...]:
    """Evaluate finite source Groups through the shared Runtime/Reward core."""

    if not groups:
        return ()

    async def run_eval_group(group_input: GroupInput) -> Optional[GroupExport]:
        stream = await runtime_domain.prepare_group(group_input)
        if stream is None:
            return None
        await runtime_domain.lease_group(stream, eval_id)
        return await _run_group(runtime_domain, reward_domain, stream)

    sample_groups = groups if reward_domain.group_rm else tuple((sample,) for group in groups for sample in group)
    group_inputs = tuple(
        GroupInput(f"{scope_id}:prepare_group_{sequence}", list(sample_group))
        for sequence, sample_group in enumerate(sample_groups)
    )
    group_tasks = tuple(
        asyncio.create_task(
            run_eval_group(group_input),
            name=f"agentic-eval-group:{group_input.group_id}",
        )
        for group_input in group_inputs
    )
    first_sample_logged = False

    async def shutdown_eval() -> None:
        for task in group_tasks:
            task.cancel()
        await asyncio.gather(*group_tasks, return_exceptions=True)

    def note_completed(task: asyncio.Task[Optional[GroupExport]], sample_count: int) -> None:
        nonlocal first_sample_logged
        if task.cancelled() or task.exception() is not None:
            return
        progress.update(sample_count)
        group = task.result()
        if not first_sample_logged and group is not None and group.samples:
            sample = group.samples[0]
            logger.info(
                "eval_rollout_single_dataset example data: %s reward=%s",
                [str(sample.prompt) + sample.response],
                sample.reward,
            )
            first_sample_logged = True

    for task, sample_group in zip(group_tasks, sample_groups, strict=True):
        task.add_done_callback(lambda completed, count=len(sample_group): note_completed(completed, count))
    try:
        outcomes = await asyncio.gather(*group_tasks)
        return tuple(outcome for outcome in outcomes if outcome is not None)
    finally:
        await finish_before_cancellation(
            shutdown_eval(),
            "agentic-eval-shutdown",
        )


def _agentic_eval_runtime_concurrency(args: Namespace, eval_group_size: int) -> int:
    """Translate logical Eval Group concurrency into Runtime permits."""

    eval_group_concurrency = agentic_eval_concurrency_from_args(args, eval_group_size)
    return eval_group_concurrency if args.group_rm else eval_group_concurrency * eval_group_size


def _run_on_resident_async_loop(coro):
    """Synchronously submit work to the RolloutManager-local resident loop.

    The rollout function is synchronous, while the Pipeline owns asyncio tasks
    that may outlive one step. The dedicated thread keeps their event loop
    alive between calls instead of recreating it with asyncio.run().
    """

    global _RESIDENT_ASYNC_LOOP, _RESIDENT_ASYNC_THREAD
    with _RESIDENT_ASYNC_LOCK:
        loop = _RESIDENT_ASYNC_LOOP
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()

            def run_loop() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(
                target=run_loop,
                name="agentic-resident-loop",
                daemon=True,
            )
            thread.start()
            _RESIDENT_ASYNC_LOOP = loop
            _RESIDENT_ASYNC_THREAD = thread
        elif threading.current_thread() is _RESIDENT_ASYNC_THREAD:
            raise RuntimeError("Agentic resident loop cannot synchronously wait on itself.")
    # Preserve the synchronous rollout contract while the coroutine runs on
    # the long-lived Pipeline loop.
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _shutdown_resident_async_loop() -> None:
    global _RESIDENT_ASYNC_LOOP, _RESIDENT_ASYNC_THREAD
    with _RESIDENT_ASYNC_LOCK:
        loop = _RESIDENT_ASYNC_LOOP
        thread = _RESIDENT_ASYNC_THREAD
        _RESIDENT_ASYNC_LOOP = None
        _RESIDENT_ASYNC_THREAD = None
    if loop is None:
        return
    if not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
    if not loop.is_closed() and (thread is None or thread is not threading.current_thread()):
        loop.close()


def get_agentic_resident_pipeline() -> AgenticResidentPipeline:
    pipeline = _RESIDENT_PIPELINE
    if pipeline is None:
        raise RuntimeError(
            "Agentic resident pipeline has not been initialized. "
            "Call init_agentic_resident_pipeline before generate_rollout."
        )
    return pipeline


def init_agentic_resident_pipeline(args, data_source, data_system_client) -> None:
    """Preserve the synchronous RolloutManager composition boundary."""

    with _RESIDENT_ENTRY_LOCK:
        if _RESIDENT_PIPELINE is not None:
            return
        try:
            _run_on_resident_async_loop(
                _init_agentic_resident_pipeline_async(
                    args=args,
                    data_source=data_source,
                    data_system_client=data_system_client,
                )
            )
        except BaseException:
            _shutdown_resident_async_loop()
            raise


def shutdown_agentic_resident_pipeline() -> None:
    """Idempotently release the resident Pipeline and its event loop."""

    with _RESIDENT_ENTRY_LOCK:
        try:
            _run_on_resident_async_loop(_shutdown_agentic_resident_pipeline_async())
        finally:
            _shutdown_resident_async_loop()


def _aggregate_agentic_timing(samples: list[Sample]) -> dict[str, float]:
    turns_by_sample = [sample.metadata[TRACE_KEY]["turns"] for sample in samples]
    if not turns_by_sample:
        return {}

    phase_values = {"generate": [sum(turn["generation_elapsed_s"] for turn in turns) for turns in turns_by_sample]}
    for event_key, phase in (
        ("process_vision_info_elapsed_s", "process_vision_info"),
        ("processor_elapsed_s", "image_processor"),
        ("media_encode_elapsed_s", "mm_encode"),
    ):
        sample_totals = []
        for turns in turns_by_sample:
            observed = [turn["events"][event_key] for turn in turns if event_key in turn["events"]]
            if observed:
                sample_totals.append(sum(observed))
        if sample_totals:
            phase_values[phase] = sample_totals

    return {
        f"perf_detail/rollout/{phase}_time/{statistic}": value
        for phase, values in phase_values.items()
        for statistic, value in (("mean", sum(values) / len(values)), ("max", max(values)))
    }


def _collect_agentic_metadata_metrics(samples: list[Sample]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.metadata.items():
            if (
                isinstance(key, str)
                and key
                and not key.startswith("_")
                and key not in _METADATA_METRIC_SKIP
                and isinstance(value, (int, float))
            ):
                metric_values.setdefault(key, []).append(float(value))
    return finalize_rollout_explicit_metric_values(metric_values)


def _save_train_debug_rollout_data(
    args,
    rollout_id: int,
    samples: list[Sample],
) -> None:
    if args.save_debug_rollout_data is None:
        return
    resources = load_agentic_compiler_resources(args)
    try:
        save_debug_rollout_data(
            args,
            samples,
            rollout_id=rollout_id,
            evaluation=False,
            tokenizer=resources.tokenizer,
        )
    finally:
        resources.shutdown()


def _eval_sampling_params(args, dataset_cfg: EvalDatasetConfig, *, sample_idx: int) -> dict[str, Any]:
    params = {
        "temperature": dataset_cfg.temperature,
        "top_p": dataset_cfg.top_p,
        "top_k": dataset_cfg.top_k,
        "max_new_tokens": dataset_cfg.max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if args.sglang_enable_deterministic_inference:
        params["sampling_seed"] = args.rollout_seed + sample_idx
    return params


def _build_eval_samples(
    args,
    dataset_cfg: EvalDatasetConfig,
) -> list[Sample]:
    from relax.utils.data.data import Dataset
    from relax.utils.misc import load_function

    resources = load_agentic_compiler_resources(args)
    try:
        custom_prompt_path = getattr(args, "custom_prompt_path", None)
        custom_prompt_func = load_function(custom_prompt_path) if custom_prompt_path else None
        dataset = Dataset(
            path=dataset_cfg.path,
            tokenizer=resources.tokenizer,
            processor=resources.processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            system_prompt=args.system_prompt,
            custom_prompt_func=custom_prompt_func,
        )
        samples = []
        sample_index = 0
        for group_index, prompt_sample in enumerate(dataset.samples):
            for sample_idx in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample.group_index = group_index
                sample.metadata = dataset_cfg.inject_metadata(sample.metadata)
                sample.sampling_params = _eval_sampling_params(
                    args,
                    dataset_cfg,
                    sample_idx=sample_idx,
                )
                samples.append(sample)
                sample_index += 1
        return samples
    finally:
        resources.shutdown()


async def eval_rollout(args, rollout_id: int) -> RolloutFnEvalOutput:
    results = {}
    eval_metrics = {}
    reward_key = args.eval_reward_key or args.reward_key
    for dataset_index, dataset_cfg in enumerate(args.eval_datasets):
        samples = _build_eval_samples(args, dataset_cfg)
        progress = tqdm(total=len(samples), desc=f"Eval {dataset_cfg.name}", unit="sample")
        grouped_samples: dict[int, list[Sample]] = {}
        for sample in samples:
            grouped_samples.setdefault(cast(int, sample.group_index), []).append(sample)
        eval_group_size = dataset_cfg.n_samples_per_eval_prompt
        eval_concurrency = _agentic_eval_runtime_concurrency(args, eval_group_size)
        runtime_domain = await RuntimeDomain.connect(args, "eval", eval_concurrency)
        try:
            scored_groups = await _eval_groups(
                eval_id=rollout_id,
                scope_id=f"eval:{dataset_cfg.name}:{rollout_id}",
                groups=tuple(tuple(group) for group in grouped_samples.values()),
                runtime_domain=runtime_domain,
                reward_domain=RewardDomain(args, "eval", None),
                progress=progress,
            )
            if dataset_index + 1 == len(args.eval_datasets):
                eval_metrics = {
                    f"eval/{key}": value for key, value in (await runtime_domain.collect_agentic_kv_metrics()).items()
                }
        finally:
            try:
                await finish_before_cancellation(
                    runtime_domain.shutdown(),
                    f"agentic-eval-runtime-shutdown:{dataset_cfg.name}:{rollout_id}",
                )
            finally:
                progress.close()
        completed_samples = [sample for group in scored_groups for sample in group.samples]
        completed_samples.sort(key=lambda sample: sample.index)
        results[dataset_cfg.name] = {
            "rewards": [
                sample.reward if not reward_key else sample.reward[reward_key] for sample in completed_samples
            ],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in completed_samples],
            "samples": completed_samples,
        }
    return RolloutFnEvalOutput(data=results, metrics=eval_metrics)


def generate_rollout(args, rollout_id, data_source, data_system_client=None, evaluation=False):
    """Run one rollout through the synchronous rollout function contract."""

    with _RESIDENT_ENTRY_LOCK:
        if evaluation:
            return asyncio.run(eval_rollout(args, rollout_id))
        pipeline = get_agentic_resident_pipeline()
        finalize_pipeline = False
        try:
            output = _run_on_resident_async_loop(pipeline.run_step(rollout_id))
            finalize_pipeline = (args.fully_async and rollout_id >= args.num_rollout) or (
                rollout_id + 1 >= args.num_rollout and pipeline.transfer_domain.total_debt == 0
            )
            return output
        except BaseException:
            finalize_pipeline = True
            raise
        finally:
            if finalize_pipeline:
                try:
                    _run_on_resident_async_loop(_shutdown_agentic_resident_pipeline_async())
                finally:
                    _shutdown_resident_async_loop()
