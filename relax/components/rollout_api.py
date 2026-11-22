# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Request models used by Rollout and the inference Gateway."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ScaleOutAPIRequest(BaseModel):
    """API request model for scale-out operation.

    Idempotency guarantees:
    - If num_replicas > 0: ray_native mode. num_replicas is the target *absolute* total engine count.
      If the current engine count (including in-flight requests) already meets or
      exceeds the target, no scale-out occurs (returns immediately with noop status).
    - If engine_urls is provided: external mode. engine_urls are filtered to exclude engines already active
      in the system or currently being processed by in-flight scale-out requests (status in
      PENDING, CREATING, CONNECTING, HEALTH_CHECKING, WEIGHT_SYNCING, READY, ACTIVE).
    - num_gpus_per_engine is always taken from args.rollout_num_gpus_per_engine.
    """

    model_name: str = Field(default="default", description="Target model name")
    num_replicas: int = Field(
        default=0,
        ge=0,
        description=(
            "Target absolute total engine count for ray_native mode (num_replicas > 0). If current engines >= target, no-op."
        ),
    )
    engine_urls: List[str] = Field(
        default_factory=list,
        description=(
            "External engine URLs to add (external mode, when num_replicas == 0). "
            "URLs already active or in-flight are automatically filtered out (idempotent)."
        ),
    )
    timeout_secs: Optional[float] = Field(default=None, gt=0, description="Total timeout for the operation")


class CancelAllScaleOutRequestsRequest(BaseModel):
    """Request model for cancel all scale-out requests."""

    model_name: Optional[str] = Field(None, description="Filter: only cancel requests for this model")
    status_filter: Optional[str] = Field(None, description="Filter: only cancel requests with this status")
    dry_run: bool = Field(False, description="If true, preview what would be cancelled without actually cancelling")


class ScaleInAPIRequest(BaseModel):
    """API request model for scale-in operation."""

    model_name: str = Field(default="default", description="Target model name")
    num_replicas: int = Field(
        default=0,
        ge=0,
        description="Target number of remaining replicas after scale-in. If > 0, takes priority over engine_urls.",
    )
    engine_urls: List[str] = Field(
        default_factory=list,
        description="Engine URLs to remove (e.g., ['http://localhost:30000', 'http://localhost:30001'])",
    )
    force: bool = Field(default=False, description="Force removal without waiting for drain")
    timeout_secs: Optional[float] = Field(default=None, gt=0, description="Total timeout for the operation")
    dry_run: bool = Field(default=False, description="Preview which engines would be removed without removing them")
