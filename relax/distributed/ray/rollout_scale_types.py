# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scale request data and status values, independent of the Ray runtime."""

import dataclasses
import enum
import time
import uuid
from typing import Optional


class ScaleOutStatus(str, enum.Enum):
    """Status states for scale-out requests."""

    PENDING = "PENDING"  # Request received, waiting to process
    CREATING = "CREATING"  # (ray_native) Creating Ray actors and starting SGLang
    CONNECTING = "CONNECTING"  # (external) Connecting to external engines
    HEALTH_CHECKING = "HEALTH_CHECKING"  # Engines started, running health checks
    WEIGHT_SYNCING = "WEIGHT_SYNCING"  # Engines healthy, syncing weights
    READY = "READY"  # Weight sync complete, can accept requests
    ACTIVE = "ACTIVE"  # Registered to router, processing requests
    PARTIAL = "PARTIAL"  # Partial success: some replicas scaled, others timed out
    FAILED = "FAILED"  # Scale-out failed
    REMOVING = "REMOVING"  # Being removed (scale-in or rollback)
    CANCELLED = "CANCELLED"  # Request cancelled by user


class ScaleOutMode(str, enum.Enum):
    """Scale-out mode."""

    RAY_NATIVE = "ray_native"  # Create new engines in the same Ray cluster
    EXTERNAL = "external"  # Connect to pre-existing external engines


@dataclasses.dataclass
class ScaleOutRequest:
    """Represents a scale-out request and its state.

    State Machine:
        PENDING → CREATING/CONNECTING → HEALTH_CHECKING → WEIGHT_SYNCING → READY → ACTIVE
            ↓            ↓                    ↓                ↓           ↓
        CANCELLED    CANCELLED/FAILED      FAILED           FAILED      REMOVING

    PENDING and CREATING states can be cancelled (e.g. when waiting for Ray
    autoscaler to provision resources).  FAILED state can retry with a new
    request.

    Mode is auto-detected: if num_replicas > 0, it's ray_native; if engine_urls is provided, it's external.
    """

    request_id: str  # UUID, unique identifier
    status: ScaleOutStatus  # Current state in state machine
    model_name: str = "default"  # Target model name

    # ray_native mode parameters — num_replicas is the *effective* delta to create
    # (after idempotency filtering: target_total - effective_current)
    num_replicas: int = 0  # Number of engines to actually create (post-idempotency delta)

    # external mode parameters — URLs already filtered for idempotency
    engine_urls: list[str] = dataclasses.field(default_factory=list)  # External engine URLs

    # Timeout settings
    timeout_secs: float = 600.0  # Total timeout for the scale-out operation

    # Metadata
    created_at: float = 0.0  # Creation timestamp
    updated_at: float = 0.0  # Last update timestamp

    # Tracking information
    engine_ids: list[str] = dataclasses.field(default_factory=list)  # Created/connected engine IDs
    failed_engines: list[str] = dataclasses.field(default_factory=list)  # Failed engine IDs
    error_message: Optional[str] = None  # Error details if failed
    weight_version: Optional[str] = None  # Weight version after sync
    # Deduped category names (e.g. "PROVISION_TIMEOUT") for the monitor to bucket.
    failure_categories: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at

    def update_status(self, status: ScaleOutStatus, error_message: Optional[str] = None) -> None:
        """Update the status and timestamp."""
        self.status = status
        self.updated_at = time.time()
        if error_message:
            self.error_message = error_message

    def is_terminal(self) -> bool:
        """Check if the request is in a terminal state."""
        return self.status in (
            ScaleOutStatus.FAILED,
            ScaleOutStatus.ACTIVE,
            ScaleOutStatus.PARTIAL,
            ScaleOutStatus.CANCELLED,
        )

    def can_cancel(self) -> bool:
        """Check if the request can be cancelled.

        Cancellation is allowed in PENDING and CREATING states.  During
        CREATING, the request may be waiting for Ray to schedule a placement
        group (e.g. waiting for cluster auto-scaling).  Setting the status to
        CANCELLED will be detected by the polling loop in
        ``_scale_out_ray_native`` which will then clean up and abort.
        """
        return self.status in (ScaleOutStatus.PENDING, ScaleOutStatus.CREATING)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "model_name": self.model_name,
            "num_replicas": self.num_replicas,
            "engine_urls": self.engine_urls,
            "engine_ids": self.engine_ids,
            "failed_engines": self.failed_engines,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error_message": self.error_message,
            "weight_version": self.weight_version,
            "failure_categories": self.failure_categories,
        }


class ScaleInStatus(str, enum.Enum):
    PENDING = "PENDING"
    DRAINING = "DRAINING"
    REMOVING = "REMOVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclasses.dataclass
class ScaleInRequest:
    request_id: str
    status: ScaleInStatus
    model_name: str = "default"
    num_replicas: int = 0
    engine_urls: list[str] = dataclasses.field(default_factory=list)
    timeout_secs: float = 120.0
    force: bool = False
    dry_run: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    selected_engines: list[str] = dataclasses.field(default_factory=list)
    removed_engines: list[str] = dataclasses.field(default_factory=list)
    failed_engines: list[str] = dataclasses.field(default_factory=list)
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at

    def update_status(self, status: ScaleInStatus, error_message: Optional[str] = None) -> None:
        self.status = status
        self.updated_at = time.time()
        if error_message:
            self.error_message = error_message

    def is_terminal(self) -> bool:
        return self.status in (ScaleInStatus.COMPLETED, ScaleInStatus.FAILED)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "model_name": self.model_name,
            "num_replicas": self.num_replicas,
            "engine_urls": self.engine_urls,
            "timeout_secs": self.timeout_secs,
            "force": self.force,
            "dry_run": self.dry_run,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "selected_engines": self.selected_engines,
            "removed_engines": self.removed_engines,
            "failed_engines": self.failed_engines,
            "error_message": self.error_message,
        }
