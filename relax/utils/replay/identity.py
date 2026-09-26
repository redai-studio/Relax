# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Identity model and dependency closure.

Expands a sample/group/batch selection to its semantic-group closure and
refuses to guess when membership is missing (PR #65: physical batch ≠ group).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relax.utils.replay.schema import BundleIndex, SampleRecord


class ClosureError(ValueError):
    """Raised when a selection cannot be expanded to a complete cohort."""


def sample_integrity_problems(record: SampleRecord) -> list[tuple[str, str]]:
    """Return (field, message) pairs for a malformed sample record."""
    problems: list[tuple[str, str]] = []
    if record.response_length < 0 or record.total_length < record.response_length:
        problems.append(("length", f"sample {record.sample_id!r} has invalid lengths"))
    if len(record.loss_mask) != record.response_length:
        problems.append(
            (
                "loss_mask_length",
                f"sample {record.sample_id!r} loss_mask length {len(record.loss_mask)} != "
                f"response_length {record.response_length}",
            )
        )
    return problems


@dataclass
class Closure:
    """The result of expanding a selection to a replayable cohort."""

    sample_ids: list[str] = field(default_factory=list)
    group_ids: set[str] = field(default_factory=set)
    cohort_ids: set[str] = field(default_factory=set)


def _sample_to_group(record: SampleRecord) -> str:
    """Map a sample record to its semantic group id (or fail closed)."""
    if record.group_index is None:
        raise ClosureError(f"sample {record.sample_id!r} has no group_index; cannot resolve semantic group")
    return f"g-{record.group_index}"


def group_members(index: BundleIndex, group_id: str) -> list[SampleRecord]:
    """Return every sample record belonging to semantic group group_id."""
    members = [record for record in index.samples if _sample_to_group(record) == group_id]
    if not members:
        raise ClosureError(f"semantic group {group_id!r} has no members in bundle {index.bundle_id!r}")
    return members


def batch_members(index: BundleIndex, batch_id: str) -> list[SampleRecord]:
    """Return every sample record belonging to micro-batch batch_id.

    Fails closed when no sample records the batch (including the case where the
    producer never recorded micro_batch_id), so a selection never silently
    degrades into a physical-size guess.
    """
    members = [record for record in index.samples if record.micro_batch_id == batch_id]
    if not members:
        raise ClosureError(
            f"micro-batch {batch_id!r} has no members in bundle {index.bundle_id!r} "
            "(or micro_batch_id was not recorded)"
        )
    return members


def expand_selection(
    index: BundleIndex,
    *,
    sample_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
    batch_ids: list[str] | None = None,
) -> Closure:
    """Expand a selection into the full semantic-group dependency closure.

    Selecting any sample pulls in its entire semantic group, because reward and
    advantage normalization are group-level. batch_ids select by physical
    micro-batch membership, then pull each member's group the same way. Missing
    membership is a hard error, never an inference from physical batch size.
    """
    records_by_id = {record.sample_id: record for record in index.samples}

    selected_ids: set[str] = set()
    for sample_id in sample_ids or []:
        if sample_id not in records_by_id:
            raise ClosureError(f"sample {sample_id!r} not found in bundle {index.bundle_id!r}")
        selected_ids.add(sample_id)

    closure_group_ids: set[str] = set(group_ids or [])
    closure_group_ids.update(_sample_to_group(records_by_id[sample_id]) for sample_id in selected_ids)
    for batch_id in batch_ids or []:
        closure_group_ids.update(_sample_to_group(record) for record in batch_members(index, batch_id))

    # An empty selection expands to every group in the bundle, so validation of
    # the whole bundle never reports a spurious "partial cohort".
    if not closure_group_ids:
        closure_group_ids = {_sample_to_group(record) for record in index.samples}

    expanded_ids: list[str] = []
    for group_id in sorted(closure_group_ids):
        for record in group_members(index, group_id):
            if record.sample_id not in expanded_ids:
                expanded_ids.append(record.sample_id)

    # Preserve original order for the explicitly selected samples, then append
    # the remaining closure members in stable group order.
    ordered = [sample_id for sample_id in (sample_ids or []) if sample_id in expanded_ids]
    ordered.extend(sample_id for sample_id in expanded_ids if sample_id not in ordered)

    return Closure(
        sample_ids=ordered,
        group_ids=closure_group_ids,
        cohort_ids=closure_group_ids,  # GRPO CP=1: semantic group == normalization cohort
    )


def validate_identity(index: BundleIndex) -> None:
    """Verify the index identity has exactly one valid cohort anchor.

    The anchor is either an actor_step_id (rollout_id, step_id) tuple (per-
    step) or a rollout_id (per-rollout). A scalar accumulated_step_id is never
    a valid anchor (see ActorStepId).
    """
    identity = index.identity
    if (identity.actor_step_id is None) == (identity.rollout_id is None):
        raise ClosureError("identity must set exactly one of actor_step_id or rollout_id")
    if identity.actor_step_id is not None:
        step = identity.actor_step_id
        if step.rollout_id < 0 or step.step_id < 0:
            raise ClosureError(f"invalid actor step coordinate {step}")
    elif identity.rollout_id < 0:
        raise ClosureError(f"invalid rollout id {identity.rollout_id}")
