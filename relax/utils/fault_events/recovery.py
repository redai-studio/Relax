# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Evidence required by local and global recovery plans."""

from argparse import Namespace
from collections.abc import Callable, Collection

from relax.utils.logging_utils import get_logger

from .models import RecoveryEvidence


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


def expected_evidence(mode: str, config: Namespace, roles: Collection[str]) -> int:
    """Read the configured recovery plan without importing model components."""
    if mode == "local":
        return int(RecoveryEvidence.LOCAL)
    expected = RecoveryEvidence.DEPLOYED | RecoveryEvidence.TASK_SUBMITTED
    if not (getattr(config, "debug_train_only", False) or getattr(config, "debug_rollout_only", False)):
        if "actor" in roles:
            expected |= RecoveryEvidence.STEP_RESTORED
            if getattr(config, "fully_async", False) and not getattr(config, "hybrid", False):
                expected |= RecoveryEvidence.ACTOR_WEIGHT_SENT
                if "actor_fwd" in roles:
                    expected |= RecoveryEvidence.ACTOR_FWD_WEIGHT_RECEIVED
                if "reference" in roles:
                    expected |= RecoveryEvidence.REFERENCE_WEIGHT_RECEIVED
    return int(expected)
