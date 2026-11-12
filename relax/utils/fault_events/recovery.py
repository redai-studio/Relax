# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Evidence required by local and global recovery plans."""

from argparse import Namespace
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from relax.utils.logging_utils import get_logger

from .models import ROLES, RecoveryEvidence, service_role


logger = get_logger(__name__)


def observe_recovery(observer: Callable[..., None] | None, *, observed: int = 0, failed: int = 0) -> None:
    if observer is None:
        return
    try:
        observer(observed=observed, failed=failed)
    except Exception:
        # A caller-provided observer or a custom logger must not alter restart.
        try:
            logger.warning("Fault observation unavailable")
        except Exception:
            pass


def expected_evidence(mode: str, config: Namespace, roles: Collection[str]) -> RecoveryEvidence:
    """Read the configured recovery plan without importing model components."""
    if mode == "local":
        return RecoveryEvidence.LOCAL
    return _expected_evidence(_branch(config), roles)


def _branch(config: Namespace) -> str:
    if getattr(config, "debug_train_only", False) or getattr(config, "debug_rollout_only", False):
        return "debug"
    if getattr(config, "hybrid", False):
        return "hybrid"
    return "pure_async" if getattr(config, "fully_async", False) else "colocate"


def _expected_evidence(branch: str, roles: Collection[str]) -> RecoveryEvidence:
    expected = RecoveryEvidence.DEPLOYED | RecoveryEvidence.TASK_SUBMITTED
    if branch != "debug":
        if "actor" in roles:
            expected |= RecoveryEvidence.STEP_RESTORED
            if branch == "pure_async":
                expected |= RecoveryEvidence.ACTOR_WEIGHT_SENT
                if "actor_fwd" in roles:
                    expected |= RecoveryEvidence.ACTOR_FWD_WEIGHT_RECEIVED
                if "reference" in roles:
                    expected |= RecoveryEvidence.REFERENCE_WEIGHT_RECEIVED
    return expected


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """The role graph and configuration branch frozen before rebuilding."""

    roles: tuple[str, ...]
    branch: str
    sync_weights: bool = False

    def validate(self) -> None:
        if type(self.roles) is not tuple or len(self.roles) > len(ROLES):
            raise ValueError("INVALID_PLAN: roles must be a bounded tuple of service roles")
        if any(type(role) is not str or role not in ROLES for role in self.roles):
            raise ValueError("INVALID_PLAN: roles must contain supported service roles")
        if self.roles != tuple(sorted(set(self.roles))):
            raise ValueError("INVALID_PLAN: roles must be sorted and unique")
        if type(self.branch) is not str or self.branch not in ("debug", "colocate", "hybrid", "pure_async"):
            raise ValueError("INVALID_PLAN: branch must identify a startup mode")
        if type(self.sync_weights) is not bool or (
            self.sync_weights
            and (self.branch not in ("colocate", "hybrid") or not {"actor", "rollout"} <= set(self.roles))
        ):
            raise ValueError("INVALID_PLAN: sync_weights requires colocated actor and rollout")

    @property
    def expected(self) -> RecoveryEvidence:
        return _expected_evidence(self.branch, self.roles)


def global_plan(config: Namespace, roles: Collection[str]) -> RecoveryPlan:
    from relax.engine.sft.runtime import is_offline_mode

    graph = tuple(sorted(service_role(role) for role in roles))
    branch = _branch(config)
    sync = branch in ("colocate", "hybrid") and {"actor", "rollout"} <= set(graph) and not is_offline_mode(config)
    return RecoveryPlan(graph, branch, sync)


class GlobalStartup:
    """Observe one startup without owning business tasks or remote
    responses."""

    def __init__(self, runtime: Any, recovery: Any = None, initialized: bool = False) -> None:
        self.runtime = runtime
        self.recovery = recovery
        self.observed = RecoveryEvidence.DEPLOYED if initialized else RecoveryEvidence(0)
        self._failed_stage: str | None = None
        self.complete = False

    @contextmanager
    def weight_sync(self) -> Iterator[None]:
        self._failed_stage = None
        try:
            yield
        except BaseException:
            # Cancellation may be translated by the training event loop.
            # Preserve its location and let Controller choose superseded first.
            self._failed_stage = "weights"
            raise

    @contextmanager
    def step_restore(self) -> Iterator[None]:
        self._failed_stage = None
        try:
            yield
        except BaseException:
            self._failed_stage = "step"
            raise
        self.observed |= RecoveryEvidence.STEP_RESTORED

    def token(self) -> dict | None:
        if self.runtime is not None:
            return self.runtime.token(self.recovery)
        return None

    def handoff(self, config: Namespace, roles: Collection[str], submitted: int) -> None:
        if submitted and submitted == len(roles):
            self.observed |= RecoveryEvidence.TASK_SUBMITTED
        if self.runtime is not None:
            self.runtime.startup_handoff(self.recovery, config, roles, self.observed)
        self.complete = True

    def fail(self, restarting: bool, current: Any) -> None:
        if self.runtime is None or self.recovery is None or self.complete:
            return
        if restarting and current is not self.recovery:
            self.runtime.finish(self.recovery, "recovery_interrupted", "superseded", observed=self.observed)
        elif self._failed_stage == "weights":
            # A failed Serve response can mean worker failure or transport
            # loss. Only its completion receipt proves an operation outcome.
            self.runtime.evidence(self.recovery, observed=self.observed)
            self.runtime.note_unknown("OUTCOME_UNOBSERVED")
        elif self._failed_stage == "step":
            self.runtime.finish(
                self.recovery,
                "recovery_failed",
                "step_restore_failed",
                observed=self.observed,
                failed=RecoveryEvidence.STEP_RESTORED,
            )
        else:
            self.runtime.finish(self.recovery, "recovery_failed", "handoff_exception", observed=self.observed)

    def interrupt(self) -> None:
        if self.runtime is not None and self.recovery is not None and not self.recovery.handoff:
            self.runtime.finish(self.recovery, "recovery_interrupted", "superseded")
