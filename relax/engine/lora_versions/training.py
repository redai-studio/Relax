# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collective training-side export and publication of immutable adapters."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from relax.utils.logging_utils import get_logger

from .artifact import write_adapter_artifact
from .identity import snapshot_adapter


if TYPE_CHECKING:
    from relax.backends.megatron.weight_update.lora_adapter_sync import LoraAdapterSync


logger = get_logger(__name__)


@dataclass(frozen=True)
class LoRATrainingConfig:
    """One run's shared directory and the identity of its frozen base."""

    artifact_dir: str
    base_fingerprint: str
    probe_input_ids: tuple[int, ...]
    max_versions: int = 2
    reconcile_interval_s: float = 1.0
    engine_actor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not Path(self.artifact_dir).is_absolute() or not self.base_fingerprint:
            raise ValueError("an absolute shared artifact directory and base content fingerprint are required")
        if not self.probe_input_ids or any(type(token) is not int or token < 0 for token in self.probe_input_ids):
            raise ValueError("model-valid nonnegative probe token IDs are required")
        if self.max_versions < 2:
            raise ValueError("at least two immutable LoRA versions are required")
        if not math.isfinite(self.reconcile_interval_s) or self.reconcile_interval_s <= 0:
            raise ValueError("reconciliation interval must be finite and positive")


def run_on_training_rank_zero(function: Callable[[], Any], *, phase: str) -> Any:
    """Share a control operation's result before any rank can raise."""
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    outcome = [None]
    if dist.get_rank(group) == 0:
        try:
            outcome[0] = {"value": function(), "error": None}
        except Exception as error:
            outcome[0] = {"value": None, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(outcome, src=0, group=group)
    if outcome[0]["error"] is not None:
        raise RuntimeError(f"{phase} failed: {outcome[0]['error']}")
    return outcome[0]["value"]


def agree_training_error(error: Exception | None, phase: str) -> None:
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, f"{type(error).__name__}: {error}" if error is not None else None, group=group)
    if any(errors):
        raise RuntimeError(f"LoRA {phase} failed: {errors}")


class TrainingLoRAPublisher:
    """All ranks export; only rank zero writes and calls the control plane.

    An unconfirmed RPC retains its original artifact. A later call confirms
    that exact publication before exporting any newer training weights.
    Internal bridge/collective failures still rely on distributed timeouts; the
    agreement points handle failures between collective stages.
    """

    def __init__(
        self, sync: "LoraAdapterSync", config: LoRATrainingConfig, rollout_manager: Any, rpc_timeout_s: float
    ) -> None:
        if not math.isfinite(rpc_timeout_s) or rpc_timeout_s <= 0:
            raise ValueError("publication RPC timeout must be finite and positive")
        self.sync = sync
        self.config = config
        self.rollout_manager = rollout_manager
        self.rpc_timeout_s = rpc_timeout_s
        self.sync_sequence = -1
        self.pending_version: int | None = None
        self._pending_artifact: str | None = None
        self.published_version: int | None = None
        self.skipped_total = 0
        self.last_result: dict[str, Any] | None = None

    def publish(self, all_params: Mapping[str, Any]) -> dict[str, Any]:
        import ray
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        self.sync_sequence += 1
        group = get_gloo_group()
        rank = dist.get_rank(group)
        version = self.pending_version
        if version is None:
            version = self.sync_sequence
            local = full = None
            error = None
            try:
                local = self.sync.export_local_adapter(all_params)
            except Exception as exc:
                error = exc
            agree_training_error(error, "export")
            error = None
            try:
                full = self.sync.gather_full_adapter(local, all_gather=False)
            except Exception as exc:
                error = exc
            agree_training_error(error, "pipeline gather")

            def freeze() -> str:
                snapshot = snapshot_adapter(
                    version, full, self.sync.config_dict(), base_fingerprint=self.config.base_fingerprint
                )
                return str(write_adapter_artifact(snapshot, self.config.artifact_dir).path)

            artifact = run_on_training_rank_zero(freeze, phase="LoRA artifact write")
            del local, full
            self.pending_version = version
            if rank == 0:
                self._pending_artifact = artifact

        outcome = [None]
        if rank == 0:
            try:
                outcome[0] = ray.get(
                    self.rollout_manager.publish_training_lora.remote(self._pending_artifact),
                    timeout=self.rpc_timeout_s,
                )
            except Exception as error:
                # The runtime shields publication; a lost reply does not
                # establish whether commit happened.
                outcome[0] = {"status": "unconfirmed", "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(outcome, src=0, group=group)
        result = outcome[0]
        self.last_result = result
        status = result["status"]
        if status == "unconfirmed":
            raise RuntimeError(f"LoRA publication {version} is unconfirmed: {result['error']}")
        if status == "failed":
            raise RuntimeError(f"LoRA publication {version} failed: {result['error']}")
        if status == "published":
            if result["version_id"] != version:
                raise RuntimeError("LoRA publication reply does not match the pending version")
            self.published_version = version
        elif status == "capacity_skipped" and self.published_version is not None:
            self.skipped_total += 1
        else:
            raise RuntimeError(f"Invalid LoRA publication outcome: {status}")
        self.pending_version = None
        self._pending_artifact = None
        if rank == 0:
            logger.info(
                "LoRA publication status=%s version=%s lora_publication_skipped_total=%d "
                "lora_publication_lag_steps=%d",
                status,
                version,
                self.skipped_total,
                self.sync_sequence - self.published_version,
            )
        return result
