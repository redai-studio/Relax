# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sample-stage adapter (data-layer integrity)."""

from __future__ import annotations

from typing import Any

from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.identity import sample_integrity_problems
from relax.utils.replay.report import FieldDivergence, StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter


@register_adapter(StageId.SAMPLE)
def replay_sample(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Check each sample record is well-formed (same rules as validate)."""
    result = StageResult(stage=StageId.SAMPLE.value, status=StageStatus.PASS)
    for record in bundle.index.samples:
        for field, _message in sample_integrity_problems(record):
            result.status = StageStatus.FAIL
            result.divergences.append(FieldDivergence(field=field, sample_id=record.sample_id))
    if result.status == StageStatus.FAIL:
        result.message = f"{len(result.divergences)} sample record(s) are inconsistent"
    return result
