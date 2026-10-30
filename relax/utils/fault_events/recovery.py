# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Evidence required by local and global recovery plans."""

from argparse import Namespace
from collections.abc import Collection

from .models import RecoveryEvidence


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
