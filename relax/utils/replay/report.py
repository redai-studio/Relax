# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Divergence report model.

A replay reports the first stage at which recomputed outputs diverge from
captured expected outputs, plus the field, sample and token offset where
possible. Skipped stages (recorded-only / inspect-only / unsupported) never
count as pass or fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StageStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass
class FieldDivergence:
    """A single field/token mismatch within a failing stage."""

    field: str
    sample_id: str | None = None
    token_offset: int | None = None
    expected: float | None = None
    actual: float | None = None
    abs_error: float | None = None


@dataclass
class StageResult:
    """Outcome of replaying one pipeline stage."""

    stage: str
    status: StageStatus
    message: str = ""
    divergences: list[FieldDivergence] = field(default_factory=list)
    max_abs_error: float | None = None
    mismatch_count: int = 0
    non_finite_count: int = 0


@dataclass
class ReplayReport:
    """Full replay report: per-stage results plus the first divergent stage."""

    bundle_id: str
    stages: list[StageResult] = field(default_factory=list)
    first_divergent_stage: str | None = None

    def add(self, result: StageResult) -> None:
        self.stages.append(result)
        if result.status == StageStatus.FAIL and self.first_divergent_stage is None:
            self.first_divergent_stage = result.stage

    @property
    def passed(self) -> bool:
        return self.first_divergent_stage is None
