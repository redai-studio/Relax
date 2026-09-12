# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Map a terminal NeMo Gym result into Relax's managed-session output."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .protocol import TrialResult, TrialStatus


def to_relax_output(result: TrialResult) -> dict[str, Any]:
    if result.status not in {TrialStatus.COMPLETED, TrialStatus.TRUNCATED}:
        raise ValueError(f"Cannot materialize non-success terminal status: {result.status.value}")
    if result.reward is None:
        # A Gateway deadline can end the trial before its verifier runs. Exit
        # as an agent failure so Relax drops the group instead of invoking RM.
        raise ValueError(
            f"NeMo Gym trial has no reward: request_id={result.request_id} "
            f"status={result.status.value} error_code={result.error_code or 'missing_reward'}"
        )

    reward = copy.deepcopy(result.reward)
    metadata: dict[str, Any] = {
        "nemo_gym": {
            "request_id": result.request_id,
            "status": result.status.value,
            "metrics": copy.deepcopy(result.metrics),
        }
    }
    if result.artifact_ref is not None:
        metadata["nemo_gym"]["artifact_ref"] = result.artifact_ref
    if result.error is not None:
        safe_error = {
            key: value for key in ("code", "type") if isinstance((value := result.error.get(key)), str) and value
        }
        if safe_error:
            metadata["nemo_gym"]["error"] = safe_error

    if isinstance(reward, dict) and "scalar" in reward:
        scalar = reward.get("scalar")
        if scalar is not None and not isinstance(scalar, bool) and isinstance(scalar, (int, float)):
            metadata["nemo_gym"]["reward_components"] = copy.deepcopy(reward.get("components", {}))
            reward = float(scalar)

    return {
        "reward": reward,
        "metadata": metadata,
    }


def write_relax_output(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically publish output, tolerating Relax removing a cancelled session
    dir."""

    output_path = Path(path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except FileNotFoundError:
        temporary_path.unlink(missing_ok=True)
        return
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
