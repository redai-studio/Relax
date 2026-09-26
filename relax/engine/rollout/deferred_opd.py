# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wire deferred OPD scoring into the rollout step.

The session stages every batch the step would have published and flushes them
after generation has finished. That ordering is not a preference: the student has
to be offloaded before the teacher can be activated on the same GPUs, and
offloading it while requests are still in flight would abort them.

Token selections that need a second pass on the student get an explicit follow-up
stage, because the student was deliberately put to sleep for the teacher.

The immediate path is untouched: when deferred scoring is off, no session is
created and ``OpdManager.prefill`` keeps running inline. The resident Agentic
pipeline stages its own transfer batches into the same session and flushes them
when its step closes.
"""

from typing import Any

from relax.engine.rollout.deferred import DeferredExecutor, DeferredState, seal_batch
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)

# Produced by generation, not by the scorer, so it is never a scoring requirement.
_GENERATION_OWNED_FIELDS = frozenset({"rollout_log_probs"})


def deferred_opd_active(args: Any) -> bool:
    """Whether this step stages OPD scoring instead of running it inline."""
    from relax.engine.inference.phase_plans import deferred_opd_enabled
    from relax.engine.rollout.on_policy_distillation import is_opd_enabled

    if not is_opd_enabled(args):
        return False
    return deferred_opd_enabled(args)


def required_scoring_fields(opd_manager: Any) -> tuple[str, ...]:
    """The training fields the scorer must produce for every eligible
    sample."""
    return tuple(field for field in opd_manager.schema_opd_transfer_data() if field not in _GENERATION_OWNED_FIELDS)


class DeferredOpdSession:
    """Stage batches during a rollout step, score and publish them after it."""

    def __init__(
        self,
        args: Any,
        rollout_id: int,
        data_system_client: Any,
        opd_manager: Any,
        *,
        publish: Any,
        encode_multimodal_inputs: Any = None,
    ) -> None:
        self.args = args
        self.rollout_id = rollout_id
        self.data_system_client = data_system_client
        self.opd_manager = opd_manager
        self._publish = publish
        self._encode_multimodal_inputs = encode_multimodal_inputs
        self.executor = DeferredExecutor(args)
        self._staged: list[tuple[str, Any, list[Sample], int, int, bool]] = []
        self._required = required_scoring_fields(opd_manager)

    @classmethod
    def maybe_create(
        cls,
        args: Any,
        rollout_id: int,
        data_system_client: Any,
        opd_manager: Any,
        *,
        publish: Any,
        encode_multimodal_inputs: Any = None,
    ) -> "DeferredOpdSession | None":
        if opd_manager is None or not deferred_opd_active(args):
            return None
        logger.info(f"Deferred OPD scoring active for rollout {rollout_id}: batches publish after scoring")
        return cls(
            args,
            rollout_id,
            data_system_client,
            opd_manager,
            publish=publish,
            encode_multimodal_inputs=encode_multimodal_inputs,
        )

    # ------------------------------------------------------------------
    # Staging.
    # ------------------------------------------------------------------
    async def transfer(
        self,
        args: Any,
        batch_samples: Any,
        batch_count: int,
        rollout_id: int,
        data_system_client: Any,
        is_last: bool = False,
        scoring_ids: Any = None,
    ) -> None:
        """Stand in for ``transfer_batch_to_data_system`` and stage instead.

        Keeps the helper's signature so the step's publication points do not
        have to know whether scoring is deferred. ``scoring_ids`` follows the
        flattened sample order when training indices are not unique.
        """
        del args, data_system_client
        flat = _flatten(batch_samples) if batch_samples else []
        if not flat:
            self._stage_empty_batch(batch_count, rollout_id, is_last)
            return
        batch_id = f"{rollout_id}:{batch_count}:{len(self._staged)}"
        self._staged.append((batch_id, batch_samples, flat, batch_count, rollout_id, is_last, scoring_ids))
        logger.info(f"Staged deferred batch {batch_id} with {len(flat)} samples; publication waits for scoring")

    def _stage_empty_batch(self, batch_count: int, rollout_id: int, is_last: bool) -> None:
        """Keep an empty batch's end-of-stream marker; there is nothing else to
        stage.

        A streaming partition only closes on ``is_last``. The last staged batch
        of the same rollout publishes last, so it carries the marker instead.
        """
        if not is_last:
            logger.warning(f"Deferred OPD got an empty batch {batch_count} for rollout {rollout_id}; nothing staged")
            return
        for index in range(len(self._staged) - 1, -1, -1):
            staged = self._staged[index]
            if staged[4] == rollout_id:
                self._staged[index] = (*staged[:5], True, *staged[6:])
                logger.warning(
                    f"Deferred OPD got an empty last batch {batch_count} for rollout {rollout_id}; "
                    f"batch {staged[0]} now closes the partition"
                )
                return
        logger.error(
            f"Deferred OPD got an empty last batch {batch_count} for rollout {rollout_id} with no staged batch "
            f"to close train_{rollout_id}; a streaming trainer will wait for an end-of-stream that never comes"
        )

    @property
    def staged_batches(self) -> int:
        return len(self._staged)

    # ------------------------------------------------------------------
    # Flush.
    # ------------------------------------------------------------------
    async def flush(self) -> None:
        """Score and publish every staged batch, in the order it was staged.

        Raises when a batch cannot be published: a rollout step that silently
        dropped part of its data would leave the trainer waiting for rows that
        are never coming.
        """
        staged, self._staged = self._staged, []
        for batch_id, payload, flat, batch_count, rollout_id, is_last, scoring_ids in staged:
            batch_ref = seal_batch(
                self.args,
                flat,
                batch_id=batch_id,
                rollout_id=rollout_id,
                required_fields=self._required,
                policy_version=str(getattr(self.args, "_opd_policy_version", "") or "") or None,
                scoring_ids=scoring_ids,
            )
            handle = self.executor.submit_deferred(
                batch_ref,
                plan_id=None,
                operation_id=f"deferred-opd:{batch_id}",
                samples=flat,
                payload=payload,
                # The staged end-of-stream marker has to survive the deferral, or
                # a streaming partition never closes.
                is_last=is_last,
            )

            # Bind this batch's loop values: a timed-out wait must not let a slow
            # batch publish under the next iteration's rollout_id.
            async def publish(
                published_payload: Any,
                published_is_last: bool,
                count: int = batch_count,
                published_rollout_id: int = rollout_id,
            ) -> None:
                await self._publish(
                    self.args,
                    published_payload,
                    count,
                    published_rollout_id,
                    self.data_system_client,
                    is_last=published_is_last,
                )

            identities = tuple(ref.identity for ref in batch_ref.samples)

            async def score(samples: list[Sample], ids: tuple[Any, ...] = identities) -> tuple[Any, ...]:
                return await self._score_batch(samples, ids)

            self.executor.start(handle, score=score, publish=publish)
            try:
                result = await self.executor.wait_deferred(handle)
            except BaseException:
                # The executor task is shielded from this waiter; stop it so a
                # cancelled step cannot publish later.
                await self.cancel()
                raise
            if result.state is not DeferredState.COMPLETED:
                raise RuntimeError(
                    f"Deferred OPD batch {batch_id} did not publish: state={result.state.value} "
                    f"scored={result.scored}/{batch_ref.eligible_count} "
                    f"missing={list(result.missing)[:8]} error={result.error}"
                )
            logger.info(f"Deferred batch {batch_id} published {result.published} samples after scoring")

    async def cancel(self) -> None:
        """Drop staged batches and stop any batch still scoring or
        publishing."""
        self._staged.clear()
        await self.executor.cancel()

    async def _score_batch(self, samples: list[Sample], identities: tuple[Any, ...]) -> tuple[Any, ...]:
        """Score one batch: teacher stage, optional student stage, assemble."""
        from relax.engine.inference.phase_plans import PHASE_TEACHER
        from relax.engine.rollout.scoring_phase import async_reactivate_generation, async_scoring_phase

        async with async_scoring_phase(self.args, PHASE_TEACHER):
            await self.opd_manager.prepare_teacher_inputs(samples)
            results = await self.opd_manager.score_teacher(samples)
        failures = tuple(
            identity
            for identity, sample, ok in zip(identities, samples, results, strict=True)
            if not ok and int(sample.response_length or 0) > 0
        )
        if failures:
            # Stop before the student stage: an incomplete teacher result cannot
            # produce a publishable batch, and the extra activation would only
            # move memory around for nothing.
            return failures
        if self.opd_manager.needs_student_prefill:
            # The student was offloaded for the teacher, so this selection needs
            # its own activation stage before the second pass.
            await async_reactivate_generation(self.args)
            await self.opd_manager.score_student_at_teacher(samples, None, self._encode_multimodal_inputs)
        self.opd_manager.assemble_transfer(samples)
        return ()


def _flatten(batch_samples: Any) -> list[Sample]:
    """Flatten the caller's nesting without changing its order."""
    flat: list[Sample] = []
    stack: list[Any] = list(batch_samples) if isinstance(batch_samples, (list, tuple)) else [batch_samples]
    for item in stack:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        else:
            flat.append(item)
    return flat


__all__ = [
    "DeferredOpdSession",
    "deferred_opd_active",
    "required_scoring_fields",
]
