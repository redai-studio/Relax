# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline replay runner.

Executes the pipeline stages in DAG order. Only stages that (a) resolve to
recompute under the frozen V1 matrix for the bundle's topology and (b) are
declared recompute in the manifest are executed. An unsupported topology (non-
GRPO or CP!=1) raises ValueError. Per-stage unsupported / recorded-only /
inspect-only results are skipped unless the caller explicitly requested that
stage, in which case unsupported fails. Skipped stages never count toward the
first-divergent-stage determination.
"""

from __future__ import annotations

from pathlib import Path

from relax.utils.logging_utils import get_logger
from relax.utils.replay.adapters import register_all
from relax.utils.replay.bundle import BundleReader, LoadedBundle
from relax.utils.replay.report import ReplayReport, StageResult, StageStatus
from relax.utils.replay.schema import STAGE_ORDER, StageCapability, StageId
from relax.utils.replay.selection import select_bundle
from relax.utils.replay.stages import capability_for, get_adapter, topology_supported


logger = get_logger(__name__)

# Stages whose output is a reduction over the entire cohort. They cannot be
# faithfully recomputed for a partial selection (single sample/group/batch), so
# a partial replay reports them as skipped rather than comparing a subset scalar
# against a full-cohort expected value.
_COHORT_LEVEL_STAGES = frozenset({StageId.LOSS_POLICY, StageId.LOSS_VALUE})


def _requested(stage: StageId, requested_stages: frozenset[str] | None) -> bool:
    return requested_stages is not None and stage.value in requested_stages


def _skip_or_fail(
    report: ReplayReport,
    stage: StageId,
    *,
    message: str,
    requested_stages: frozenset[str] | None,
    fail_if_requested: bool,
) -> None:
    """Skip a non-recompute stage, or fail when the caller asked for it."""
    status = StageStatus.FAIL if fail_if_requested and _requested(stage, requested_stages) else StageStatus.SKIPPED
    report.add(StageResult(stage=stage.value, status=status, message=message))


def replay_bundle(
    bundle: LoadedBundle,
    *,
    partial_selection: bool = False,
    requested_stages: frozenset[str] | None = None,
) -> ReplayReport:
    """Replay every declared stage and return the divergence report."""
    register_all()
    report = ReplayReport(bundle_id=bundle.index.bundle_id)
    config = bundle.index.config
    context_parallel = bundle.index.identity.rank.get("cp", 1)
    if not topology_supported(advantage_estimator=config.advantage_estimator, context_parallel=context_parallel):
        raise ValueError(
            "unsupported replay topology "
            f"advantage_estimator={config.advantage_estimator!r} cp={context_parallel} "
            "(V1 supports GRPO with CP=1)"
        )
    ctx: dict[str, object] = {}

    for stage in STAGE_ORDER:
        if partial_selection and stage in _COHORT_LEVEL_STAGES:
            _skip_or_fail(
                report,
                stage,
                message="cohort-level stage is not replayable under a partial selection",
                requested_stages=requested_stages,
                fail_if_requested=False,
            )
            continue

        contract = bundle.manifest.stage_contracts.get(stage)
        if contract is None:
            _skip_or_fail(
                report,
                stage,
                message="no stage contract",
                requested_stages=requested_stages,
                fail_if_requested=True,
            )
            continue

        capability = capability_for(
            stage, advantage_estimator=config.advantage_estimator, context_parallel=context_parallel
        )
        if capability != StageCapability.RECOMPUTE:
            _skip_or_fail(
                report,
                stage,
                message=f"capability={capability.value}",
                requested_stages=requested_stages,
                fail_if_requested=capability == StageCapability.UNSUPPORTED,
            )
            continue
        if contract.capability != StageCapability.RECOMPUTE:
            _skip_or_fail(
                report,
                stage,
                message=f"manifest capability={contract.capability.value}",
                requested_stages=requested_stages,
                fail_if_requested=contract.capability == StageCapability.UNSUPPORTED,
            )
            continue

        adapter = get_adapter(stage)
        report.add(adapter(bundle, ctx))

    return report


def replay(
    path: str | Path,
    *,
    sample_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
    batch_ids: list[str] | None = None,
    requested_stages: frozenset[str] | None = None,
) -> ReplayReport:
    """Load a bundle from path and replay it (optionally a selection)."""
    bundle = BundleReader(path).load()

    has_selection = any((sample_ids, group_ids, batch_ids))
    partial_selection = False
    if has_selection:
        full_count = len(bundle.index.samples)
        bundle = select_bundle(bundle, sample_ids=sample_ids, group_ids=group_ids, batch_ids=batch_ids)
        partial_selection = len(bundle.index.samples) != full_count

    logger.info("replaying bundle %s (%d samples)", bundle.index.bundle_id, len(bundle.index.samples))
    return replay_bundle(bundle, partial_selection=partial_selection, requested_stages=requested_stages)
