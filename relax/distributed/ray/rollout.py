# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# pyright: reportMissingTypeArgument=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUninitializedInstanceVariable=false, reportOptionalIterable=false, reportFunctionMemberAccess=false, reportPossiblyUnboundVariable=false, reportOperatorIssue=false, reportIndexIssue=false, reportReturnType=false

import asyncio
import dataclasses
import enum
import logging
import multiprocessing
import os
import random
import threading
import time
import uuid
from typing import Any, Optional

import numpy as np
import ray
import transfer_queue as tq
import yaml
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.core.node_group_affinity import with_control_plane_affinity
from relax.distributed.ray.rollout_validation import validate_server_group_gpu_indices
from relax.engine.rollout.base_types import call_rollout_fn
from relax.utils import device as device_utils
from relax.utils import scale_utils, tracking_utils
from relax.utils.env import Envs
from relax.utils.health_monitor import RolloutHealthMonitor
from relax.utils.http_utils import (
    _wrap_ipv6,
    find_available_port,
    get,
    get_host_info,
    init_http_client,
    post,
    router_worker_base_urls,
)
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_checker import MetricChecker
from relax.utils.metrics.metric_utils import (
    compute_pass_rate,
    compute_rollout_reward_metrics,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from relax.utils.misc import group_by, load_function
from relax.utils.multimodal.stats import get_sample_multimodal_stats
from relax.utils.opd.opd_utils import compute_mopd_metrics
from relax.utils.reload_utils import ReloadableMixin
from relax.utils.s3_model_loader import (
    build_runai_streamer_env_for_load,
    prepare_model_maybe_update_args,
)
from relax.utils.scale_utils import PrecheckProbeCategory, ScaleOutFailure, ScaleOutFailureCategory
from relax.utils.tracking_utils import init_tracking
from relax.utils.training.train_dump_utils import (
    save_debug_rollout_data,
    save_eval_summary_jsonl,
    save_rollout_result_jsonl,
)
from relax.utils.types import Sample
from relax.utils.utils import get_ray_accelerator_kwargs

from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = get_logger(__name__)


def _resolve_rollout_engine_class(args):
    """Return the rollout engine class for the configured run.

    ``rollout_engine_class_path`` (a dotpath) selects a custom engine such as
    the native-diffusion generation engine; when unset the default text
    ``SGLangEngine`` is returned so existing runs are unaffected (design doc
    8.3).
    """
    class_path = getattr(args, "rollout_engine_class_path", None)
    if class_path:
        return load_function(class_path)
    return SGLangEngine


@dataclasses.dataclass
class EngineGroupConfig:
    """Configuration for a single engine group.

    Attributes:
        worker_type: One of "regular", "prefill", "decode", or "placeholder".
                     "placeholder" reserves GPU slots without creating engines.
        num_gpus: Total number of GPUs for this group.
        num_gpus_per_engine: GPUs per engine for this group.  Overrides the
                             model-level or global ``--rollout-num-gpus-per-engine``.
        overrides: Optional dict of SGLang ``ServerArgs`` field overrides.
                   These are applied on top of the base CLI ``--sglang-*``
                   arguments in ``_compute_server_args``.
    """

    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        valid_types = {"regular", "prefill", "decode", "placeholder"}
        assert self.worker_type in valid_types, (
            f"Invalid worker_type '{self.worker_type}', must be one of {valid_types}"
        )
        assert self.num_gpus > 0, f"num_gpus must be > 0, got {self.num_gpus}"


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single model deployment.

    Attributes:
        name: Unique name for this model (e.g. "actor", "reward").
        model_path: HF checkpoint path.  Falls back to ``args.hf_checkpoint``.
        num_gpus_per_engine: Default GPUs per engine for all groups in this
                             model.  Individual groups can override.
        engine_groups: Engine group configurations for this model.
    """

    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    engine_groups: list[EngineGroupConfig] = dataclasses.field(default_factory=list)

    def resolve(self, args) -> None:
        """Resolve per-group defaults from model-level then args-level
        values."""
        default_gpus_per_engine = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
        # `args.sglang_hf_checkpoint` lets INT4 QAT runs point SGLang at the
        # source compressed-tensors directory while training-side consumers
        # keep using the auto-cast `args.hf_checkpoint` (BF16 cache).
        default_model_path = self.model_path or args.sglang_hf_checkpoint or args.hf_checkpoint
        for g in self.engine_groups:
            if g.num_gpus_per_engine is None:
                g.num_gpus_per_engine = default_gpus_per_engine
            # Inject model_path into overrides so _compute_server_args picks it up.
            if "model_path" not in g.overrides:
                g.overrides["model_path"] = default_model_path

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(g.worker_type in ("prefill", "decode") for g in self.engine_groups)

    @property
    def total_num_gpus(self) -> int:
        return sum(g.num_gpus for g in self.engine_groups)


@dataclasses.dataclass
class SglangConfig:
    """Configuration for SGLang engine deployment.

    Loaded from ``--sglang-config`` YAML file.

    **Config format**::

        sglang:
          - name: actor
            model_path: /path/to/actor
            num_gpus_per_engine: 2
            engine_groups:
              - worker_type: prefill
                num_gpus: 4
                num_gpus_per_engine: 2
              - worker_type: decode
                num_gpus: 8
                num_gpus_per_engine: 4
          - name: reward
            model_path: /path/to/reward
            engine_groups:
              - worker_type: regular
                num_gpus: 4

    Each model gets its own router.  ``placeholder`` groups reserve GPU
    slots without creating engines.  ``overrides`` are ``ServerArgs``
    field names applied on top of the base ``--sglang-*`` CLI args.
    """

    models: list[ModelConfig]

    @staticmethod
    def from_yaml(path: str) -> "SglangConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        assert "sglang" in data, (
            f"sglang config must have a 'sglang' key, got {list(data.keys())}. "
            f"Wrap your engine_groups inside a model entry under 'sglang'."
        )
        models = []
        for m in data["sglang"]:
            groups = [EngineGroupConfig(**g) for g in m.get("engine_groups", [])]
            models.append(
                ModelConfig(
                    name=m["name"],
                    model_path=m.get("model_path"),
                    num_gpus_per_engine=m.get("num_gpus_per_engine"),
                    engine_groups=groups,
                )
            )
        return SglangConfig(models=models)

    @staticmethod
    def from_prefill_num_servers(args) -> "SglangConfig":
        """Build a config equivalent to the legacy --prefill-num-servers
        flag."""
        total_gpus = args.rollout_num_gpus
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = total_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {total_gpus}, prefill {prefill_gpus}"
        return SglangConfig(
            models=[
                ModelConfig(
                    name="default",
                    engine_groups=[
                        EngineGroupConfig(worker_type="prefill", num_gpus=prefill_gpus),
                        EngineGroupConfig(worker_type="decode", num_gpus=decode_gpus),
                    ],
                )
            ]
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(m.has_pd_disaggregation for m in self.models)

    @property
    def total_num_gpus(self) -> int:
        return sum(m.total_num_gpus for m in self.models)


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

    def __post_init__(self):
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

    def __post_init__(self):
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


class EngineGroupLifecycle(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    REMOVING = "REMOVING"
    REMOVED = "REMOVED"


@dataclasses.dataclass
class EngineGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg. A
    RolloutServer may contain multiple EngineGroups (e.g. prefill vs decode in
    PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", or "decode"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    router_ip: str | None = None
    router_port: int | None = None
    is_scaled_out: bool = False  # True for groups added via scale-out, False for initial groups
    skip_dcs_registration: bool = False  # Skip DCS registration for scaled-out engines
    skip_router_registration: bool = False  # Skip router registration until weight sync completes
    lifecycle_status: EngineGroupLifecycle = EngineGroupLifecycle.ACTIVE
    eviction_requested: bool = False

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()``
        without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next engine group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpu_per_engine=num_gpu_per_engine,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(_resolve_rollout_engine_class(self.args))

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1") | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    # OPD per-position token_ids logprob patch: default off, enabled
                    # via env (affects the student engine too, like the teacher).
                    "RELAX_OPD_PER_POS_TOKEN_IDS": "0",
                    # OPD pre-expanded multimodal patch: passed through to the
                    # student engine too, because advantage mode sends
                    # opd_preexpanded_raw requests to the student (student-at-
                    # teacher-topk prefill).  The patch makes the student SGLang
                    # skip retokenize/detokenize for expanded input_ids + image_data.
                    "RELAX_OPD_PREEXPANDED_PATCH": "0",
                    # The TP memory-imbalance check is overly conservative under
                    # colocate (the actor occupies GPUs at engine init and is
                    # offloaded before rollout), so disable it. Recent SGLang reads
                    # SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK (default True); the old
                    # SGL(ANG)_DISABLE_* vars are no longer honored — worse, the
                    # deprecation shim value-copies SGL_DISABLE_* into the ENABLE var,
                    # so setting them re-enables the check. Set ENABLE=false directly.
                    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    "SLIME_ENABLE_PROFILING": "true",
                    # NOTE(sunhuo): disable custom all-reduce-v2 temporarily, as it may cause custom_all_reduce.cuh:37: CUDA error: invalid argument.
                    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
                }.items()
            }
            effective_model_path = self.sglang_overrides.get("model_path", self.args.hf_checkpoint)
            effective_load_format = self.sglang_overrides.get(
                "load_format", getattr(self.args, "sglang_load_format", "auto")
            )
            env_vars.update(
                build_runai_streamer_env_for_load(
                    getattr(self.args, "model_source", None),
                    effective_model_path,
                    effective_load_format,
                )
            )
            if getattr(self.args, "fp16", False):
                env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"

            accelerator_kwargs = get_ray_accelerator_kwargs(num_gpus)
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
                **accelerator_kwargs,
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
                register_sigterm_handler=self.is_scaled_out,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            # Compute base_port from the maximum cursor across all nodes that
            # this group's engines may land on (conservative: just use global max).
            base_port = max(port_cursors.values()) if port_cursors else find_available_port(15000)
            addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=rollout_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
                skip_dcs_registration=self.skip_dcs_registration,
                skip_router_registration=self.skip_router_registration,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def healthcheck_engines(self, timeout: float = 5.0) -> set[int]:
        """Check health of engines in this group.

        Returns:
            Set of indices of failed engines.
        """
        failed_indices = set()
        for i, engine in enumerate(self.all_engines):
            if engine is None:
                continue
            try:
                ray.get(engine.health_generate.remote(timeout=timeout))
            except Exception as e:
                logger.warning(f"Engine {i} healthcheck failed: {e}")
                failed_indices.add(i)
        return failed_indices

    def shutdown_engines(self, indices: set[int]) -> None:
        """Shutdown engines at the given indices.

        This removes the engines from the group and unregisters them from
        router and DCS.
        """
        for i in indices:
            engine = self.all_engines[i]
            if engine is not None:
                try:
                    ray.get(engine.shutdown.remote(), timeout=10)
                except Exception as e:
                    logger.warning(f"Failed to shutdown engine {i}: {e}")
                try:
                    ray.get(engine.unregister_dcs.remote(), timeout=10)
                except Exception as e:
                    logger.warning(f"Failed to unregister engine {i} from DCS: {e}")
                try:
                    ray.get(engine.unregister_from_router.remote(), timeout=10)
                except Exception as e:
                    logger.warning(f"Failed to unregister engine {i} from router: {e}")
                try:
                    ray.kill(engine)
                except Exception as e:
                    logger.warning(f"Failed to kill engine {i}: {e}")
            self.all_engines[i] = None
            logger.info(f"Shutdown engine at index {i}")


@dataclasses.dataclass
class EngineFinalizeResult:
    """Outcome of finalizing a new engine group's registration."""

    success: bool
    group: "EngineGroup | None" = None
    reason: "ScaleOutFailure | None" = None


@dataclasses.dataclass
class ScaleResult:
    """Outcome of a scale-out sub-step: success flag + optional classified
    failure."""

    success: bool
    reason: "ScaleOutFailure | None" = None


class _WeightSyncTimeoutError(TimeoutError):
    """A remote weight-sync operation timed out and must not be retried."""


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more engine groups.

    Each RolloutServer represents one model deployed behind a single router. A
    server may contain multiple EngineGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    engine_groups: list[EngineGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute
        nothing)."""
        return [e for g in self.engine_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.engine_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.engine_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.engine_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to
        ``engines``."""
        return [g.num_gpus_per_engine for g in self.engine_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to
        ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating
        engines.
        """
        offsets = []
        for g in self.engine_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.

        Only valid when all active groups share the same value.
        """
        values = {g.nodes_per_engine for g in self.engine_groups}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        groups = list(self.engine_groups)
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in groups]

        all_handles = []
        port_cursors: dict[int, int] = {}
        groups_to_remove = []

        for g_idx, g in enumerate(groups):
            if g.is_scaled_out and g.pg is not None and g.lifecycle_status is not EngineGroupLifecycle.ACTIVE:
                if any(engine is None for engine in g.all_engines):
                    logger.warning(
                        "Skipping recovery for non-active scaled engine group at rank offset %s",
                        g.rank_offset,
                    )
                continue
            if g.pg is None:
                failed_indices = g.healthcheck_engines()
                if failed_indices:
                    logger.warning(
                        f"External engine group {g_idx} has {len(failed_indices)} unhealthy engines, shutting down..."
                    )
                    g.shutdown_engines(failed_indices)

                if all(engine is None for engine in g.all_engines):
                    logger.warning(f"External engine group {g_idx} is completely dead, marking for removal")
                    groups_to_remove.append(g_idx)
                continue
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)

        for g_idx in reversed(groups_to_remove):
            group = groups.pop(g_idx)
            if group in self.engine_groups:
                self.engine_groups.remove(group)
            dead_per_group.pop(g_idx)
            logger.info(f"Removed dead external engine group {g_idx}")

        if all_handles:
            ray.get(all_handles)

        release_handles = []
        new_engines_all = []
        for g, dead_indices in zip(groups, dead_per_group, strict=True):
            if g.pg is None or (g.is_scaled_out and g.lifecycle_status is not EngineGroupLifecycle.ACTIVE):
                continue
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.args.offload_rollout and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                new_engines_all.extend(new_engines)

        if release_handles:
            ray.get(release_handles)
            ray.get(
                [engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines_all]
            )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.engine_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.engine_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []


# In-process singleton set inside RolloutManager.__init__. Only meaningful
# from code running inside the RolloutManager actor's own process (e.g., a
# custom_reward_post_process function loaded there). Cross-process access
# must go through the Ray handle.
_LOCAL_ROLLOUT_MANAGER: "RolloutManager | None" = None


def get_local_rollout_manager() -> "RolloutManager":
    # Read via sys.modules to defend against the case where an importer of
    # this module gets a different module-object namespace than the one
    # RolloutManager.__init__ wrote into (cloudpickle can create parallel
    # namespaces when materializing @ray.remote classes in workers).
    import sys as _sys

    mgr = getattr(_sys.modules[__name__], "_LOCAL_ROLLOUT_MANAGER", None)
    if mgr is None:
        raise RuntimeError(
            "get_local_rollout_manager() called outside the RolloutManager "
            "actor process, or before it finished __init__. "
            f"module_id={id(_sys.modules[__name__])}, pid={os.getpid()}"
        )
    return mgr


@ray.remote(
    concurrency_groups={
        "health_monitoring": 1,
        "scale_out": 8,
        "scale_in": 8,
        "scale_coordination": 1,
        "recover_rollout_engines": 1,
    }
)
class RolloutManager(ReloadableMixin):
    """The class to run rollout and convert rollout data to training data.

    Inherits ReloadableMixin to provide hot-reload capabilities, supporting:
    - reload_module(module_name): Reload the specified module
    - get_loaded_modules(): Retrieve information about loaded modules
    """

    # Scale-out weight-sync tunables; see relax.utils.env.Envs for semantics/defaults.
    _WEIGHT_SYNC_PORT_BASE = Envs.RELAX_WEIGHT_SYNC_PORT_BASE
    _WEIGHT_SYNC_PORT_MAX = Envs.RELAX_WEIGHT_SYNC_PORT_MAX
    _SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE = Envs.RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE
    _SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX = Envs.RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX
    _SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS = Envs.RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS
    _WEIGHT_SYNC_MAX_INIT_ATTEMPTS = Envs.RELAX_WEIGHT_SYNC_MAX_INIT_ATTEMPTS

    def __init__(self, args, pg, data_source=None):
        self.pg = pg
        self.args = args
        self._dynamic_global_batch_size = None

        init_tracking(args, primary=False)

        self.data_source = data_source

        tq.init(self.args.tq_config)
        self.data_system_client = tq.get_client()

        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )

        if self.args.use_agentic_rollout:
            from relax.agentic.rollout import init_agentic_resident_pipeline

            init_agentic_resident_pipeline(self.args, self.data_source, self.data_system_client)

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
        self.rollout_engine_lock = Lock.options(
            **with_control_plane_affinity(self.args, {"num_cpus": 1, "num_gpus": 0})
        ).remote()
        self.rollout_id = -1
        self._metric_checker = MetricChecker.maybe_create(args)
        self._tokenizer = None  # Lazy-initialized tokenizer for debug data saving
        self._engine_lifecycle_lock = threading.RLock()
        existing_groups = [group for srv in self.servers.values() for group in srv.engine_groups]
        self._next_engine_rank = max(
            (group.rank_offset + len(group.all_engines) for group in existing_groups),
            default=0,
        )

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.engine_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection
        self.status = None

        # In-process singleton so user code running inside this actor (notably
        # custom_reward_post_process_func loaded and invoked on the rollout
        # actor's event loop) can call offload()/onload() directly without a
        # self-remote call that would deadlock.
        # We must write into sys.modules[__name__] explicitly: when Ray/cloudpickle
        # reconstructs the @ray.remote class in the worker, __init__.__globals__
        # can be a namespace distinct from sys.modules['relax.distributed.ray.rollout'],
        # so a plain `global _LOCAL_ROLLOUT_MANAGER` write becomes invisible to
        # any code that imports the module the normal way.
        import sys as _sys

        _sys.modules[__name__]._LOCAL_ROLLOUT_MANAGER = self

        # Elastic scale-out tracking
        self._scale_out_requests: dict[str, ScaleOutRequest] = {}
        self._port_cursors: dict[int, int] = {}

        # Elastic scale-in tracking
        self._scale_in_requests: dict[str, ScaleInRequest] = {}
        self._training_weight_updating: bool = False
        self._scale_out_weight_updating: bool = False
        # Distributed mutex shared with the Actor process to ensure DCS weight
        # sync (update_weights_fully_async) and sglang remote instance weight sync
        # (_sync_weights_from_seed_engine) never run concurrently.
        # Both paths use the seed engine's NCCL stack and cannot overlap.
        self._weight_sync_lock = Lock.options(
            **with_control_plane_affinity(self.args, {"num_cpus": 1, "num_gpus": 0})
        ).remote()

        # GC config: max terminal requests to keep per dict
        self._max_terminal_requests = 100

        # Eviction monitoring: periodically check if any engine received SIGTERM
        self._eviction_monitor_stop = threading.Event()
        self._eviction_monitor_thread = None
        self._eviction_check_interval = getattr(args, "eviction_check_interval", 10.0)
        if not self.args.debug_train_only:
            self._start_eviction_monitor()

    def _reserve_engine_ranks(self, count: int, alignment: int = 1) -> int:
        """Allocate ranks monotonically so a removed elastic rank is not
        reused."""
        with self._engine_lifecycle_lock:
            rank_offset = ((self._next_engine_rank + alignment - 1) // alignment) * alignment
            self._next_engine_rank = rank_offset + count
            return rank_offset

    @property
    def _is_weight_updating(self) -> bool:
        with self._engine_lifecycle_lock:
            return self._training_weight_updating or self._scale_out_weight_updating

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is
        running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.engine_groups[0].all_engines and self.server.engine_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.engine_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        self._stop_eviction_monitor()
        for monitor in self._health_monitors:
            monitor.stop()
        self._shutdown_all_engines()

    def _shutdown_all_engines(self, timeout: float = 15.0):
        """Shut down all SGLang engine actors and their child processes.

        This method is called during dispose() to ensure SGLang subprocesses
        (scheduler, detokenizer, etc.) are properly terminated when training
        completes, instead of being orphaned.
        """
        if not self.servers:
            return

        all_engines = []
        for srv in self.servers.values():
            for group in srv.engine_groups:
                for engine in group.all_engines:
                    if engine is not None:
                        all_engines.append(engine)

        if not all_engines:
            return

        logger.info(f"[Shutdown] Shutting down {len(all_engines)} SGLang engine(s)...")

        # Step 1: Call shutdown() on each engine to kill child processes
        # (sglang::scheduler, sglang::detokenizer, etc.)
        shutdown_refs = []
        for engine in all_engines:
            try:
                ref = engine.shutdown.remote()
                shutdown_refs.append((engine, ref))
            except Exception as e:
                logger.warning(f"[Shutdown] Failed to call shutdown on engine: {e}")

        # Wait for shutdown calls to complete with a timeout
        for engine, ref in shutdown_refs:
            try:
                ray.get(ref, timeout=timeout)
            except Exception as e:
                logger.warning(f"[Shutdown] Engine shutdown timed out or failed: {e}")
                # Force-kill the Ray actor as a fallback
                try:
                    ray.kill(engine)
                except Exception:
                    pass

        logger.info(f"[Shutdown] All {len(all_engines)} SGLang engine(s) shut down.")

    @property
    def server(self) -> RolloutServer | None:
        """Default server (first model).

        For backward compatibility.
        """
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_server(self, model_name: str | None = None) -> RolloutServer | None:
        if model_name is None:
            return self.server
        return self.servers.get(model_name)

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_rollout_engines_and_lock(self, model_name: str | None = None):
        srv = self._get_server(model_name)
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_weight_sync_lock(self):
        """Return the distributed Lock actor used to serialise DCS weight sync
        and remote instance weight sync.

        Called once by the Actor during initialisation so it can hold the same
        lock around update_weights_fully_async.
        """
        return self._weight_sync_lock

    def get_dynamic_global_batch_size(self):
        """Return the actual sample count from the last rollout step.

        Used by training side to compute correct batch_size for TQ fetch when
        use_dynamic_global_batch_size is enabled.
        """
        assert self._dynamic_global_batch_size is not None, (
            "get_dynamic_global_batch_size called before first generate()"
        )
        return self._dynamic_global_batch_size

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return ray.get(self.data_source.lengths.remote()) // self.args.rollout_batch_size

    async def generate(self, rollout_id):
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        output = await asyncio.to_thread(
            call_rollout_fn,
            self.generate_rollout,
            self.args,
            rollout_id,
            self.data_source,
            self.data_system_client,
            evaluation=False,
        )
        if self.args.partial_rollout and self.args.use_dynamic_global_batch_size:
            self._dynamic_global_batch_size = len(
                {sample.index for sample_group in output.samples for sample in sample_group}
            )

    async def eval(self, rollout_id):
        self.health_monitoring_resume()

        # TODO: add fault tolerance to eval
        result = await asyncio.to_thread(
            call_rollout_fn,
            self.eval_generate_rollout,
            self.args,
            rollout_id,
            self.data_source,
            self.data_system_client,
            evaluation=True,
        )
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    async def run_predict(self, train_step: int) -> None:
        """Periodic SFT predict pass entry point.

        Ensures KV/cuda-graph is onloaded (no-op if already on), then delegates
        to ``run_predict_loop`` which renders the eval set, batches calls to
        ``self.generate_predict``, and writes
        ``<args.save>/predict/predictions_step_<train_step>.jsonl``.

        Mirrors the ``eval`` method: does NOT proactively offload afterward —
        the next training step's actor↔rollout coordination drives state
        transitions, same as PPL eval.
        """
        from relax.engine.sft.predict.loop import run_predict_loop

        await self.onload_kv()
        await run_predict_loop(self, self.args, train_step)

    async def generate_predict(
        self,
        prompts: list[str],
        multimodal_inputs_list: list[dict | None] | None = None,
    ) -> list[str]:
        """Generate completions for ``prompts`` concurrently.

        POSTs all prompts at once in round-robin order directly to engine
        workers, bypassing the router. Predict prompts share a long fixed
        prefix (``<|vision_start|><|image_pad|><|vision_end|>...``); cache-
        aware routing would pin every request to the engine that first
        cached the prefix, defeating multi-engine throughput.

        ``multimodal_inputs_list`` is a parallel list of dicts (or ``None``)
        carrying images/videos/audio for each prompt; encoded inline and
        merged into the payload, mirroring the RL ``generate()`` path.
        """
        import sglang_router
        from packaging.version import parse

        from relax.engine.rollout.sglang_rollout import _encode_multimodal_inputs

        self.health_monitoring_resume()

        router_base = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        if parse(sglang_router.__version__) <= parse("0.2.1") or getattr(self.args, "use_slime_router", False):
            response = await get(f"{router_base}/list_workers")
            worker_urls = response["urls"]
        else:
            response = await get(f"{router_base}/workers")
            worker_urls = [w["url"] for w in response["workers"]]
        worker_urls = router_worker_base_urls(worker_urls)
        if not worker_urls:
            worker_urls = [router_base]

        # Reuse the shared --eval-* sampling args (no SFT-predict-specific
        # flags). Defaults preserve the original SFT predict behaviour
        # (greedy, max_new_tokens=512) when --eval-* is not provided.
        eval_temperature = getattr(self.args, "eval_temperature", None)
        eval_max_response_len = getattr(self.args, "eval_max_response_len", None)
        eval_top_p = getattr(self.args, "eval_top_p", None)
        sampling_params = {
            "temperature": 0.0 if eval_temperature is None else eval_temperature,
            "max_new_tokens": 512 if eval_max_response_len is None else eval_max_response_len,
            "top_p": 1.0 if eval_top_p is None else eval_top_p,
        }
        if multimodal_inputs_list is None:
            multimodal_inputs_list = [None] * len(prompts)

        async def _one(idx: int, prompt: str, mm: dict | None) -> str:
            url = f"{worker_urls[idx % len(worker_urls)]}/generate"
            payload: dict[str, Any] = {"text": prompt, "sampling_params": sampling_params}
            if mm:
                encoded_mm, _ = await _encode_multimodal_inputs(mm)
                payload.update(encoded_mm)
            output = await post(url, payload)
            if isinstance(output, dict) and "text" in output:
                return output["text"]
            if isinstance(output, str):
                return output
            return str(output)

        return await asyncio.gather(
            *[_one(i, p, m) for i, (p, m) in enumerate(zip(prompts, multimodal_inputs_list, strict=True))]
        )

    async def save(self, rollout_id):
        await self.data_source.save.remote(rollout_id)

    async def load(self, rollout_id=None):
        try:
            await self.data_source.load.remote(rollout_id)
        except Exception as e:
            logger.warning(f"Failed to load data source: {e}")

    async def offload(self):
        self._offload_local()

    def _offload_local(self):
        """Sync body of offload(); safe to call directly from code running
        inside this actor's process (e.g. custom_reward_post_process)."""
        if self.status == "offload":
            logger.info("Rollout already offloaded; skipping")
            return
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()
        self.status = "offload"

    async def onload(self, tags: list[str] | None = None):
        self._onload_local(tags)

    def _onload_local(self, tags: list[str] | None = None):
        """Sync body of onload(); safe to call directly from code running
        inside this actor's process (e.g. custom_reward_post_process)."""
        for srv in self.servers.values():
            srv.onload(tags)
        # Full onload transitions status; per-tag calls leave status for the
        # dedicated wrappers below (onload_weights / onload_kv).
        if tags is None:
            self.status = "onload"

    async def onload_weights(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
        self.status = "onload"

    def get_status(self):
        return self.status

    @ray.method(concurrency_group="recover_rollout_engines")
    def recover_rollout_engines(self, model_name: str | None = None):
        """Restart any dead rollout engines and update num_new_engines for
        update_weights detection."""
        self.health_monitoring_pause()
        srv = self._get_server(model_name)
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_num_new_engines(self, model_name: str | None = None):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_server(model_name)
        if srv:
            srv.num_new_engines = 0

    @ray.method(concurrency_group="health_monitoring")
    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    @ray.method(concurrency_group="health_monitoring")
    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    async def check_weights(self, action: str):
        refs = [engine.check_weights.remote(action=action) for engine in self.rollout_engines]
        return await asyncio.gather(*refs)

    @ray.method(concurrency_group="health_monitoring")
    def set_force_unhealthy(self, engine_id: int) -> None:
        """Set ``_force_unhealthy`` on a specific rollout engine for debug.

        Args:
            engine_id: Index into ``self.rollout_engines`` (node-0 engines).
        """
        engines = self.rollout_engines
        if engine_id < 0 or engine_id >= len(engines):
            raise IndexError(f"engine_id {engine_id} out of range [0, {len(engines)})")
        engine = engines[engine_id]
        if engine is None:
            raise ValueError(f"Engine {engine_id} is already None (dead)")
        ray.get(engine.set_force_unhealthy.remote(True))
        logger.info(f"set_force_unhealthy: engine {engine_id} marked as unhealthy")

    @property
    def tokenizer(self):
        """Lazy-initialized tokenizer for debug data saving."""
        if self._tokenizer is None:
            try:
                from relax.utils.data.processing_utils import load_tokenizer

                prepare_model_maybe_update_args(self.args, completeness="metadata")
                self._tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                logger.info(f"Loaded tokenizer from {self.args.hf_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        return self._tokenizer

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        """Save debug rollout data using shared utility function."""
        save_debug_rollout_data(self.args, data, rollout_id, evaluation, tokenizer=self.tokenizer)

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    # ===================== Elastic Rollout Scale-Out Methods =====================

    def _gc_terminal_requests(self) -> None:
        """Garbage-collect terminal scale-out/in requests to prevent unbounded
        memory growth.

        Keeps at most ``self._max_terminal_requests`` terminal entries in each
        dict, evicting the oldest (by ``updated_at``) first.
        """
        for store in (self._scale_out_requests, self._scale_in_requests):
            terminal = [r for r in store.values() if r.is_terminal()]
            if len(terminal) > self._max_terminal_requests:
                # Sort by updated_at ascending (oldest first) and remove excess
                terminal.sort(key=lambda r: r.updated_at)
                to_remove = terminal[: len(terminal) - self._max_terminal_requests]
                for r in to_remove:
                    del store[r.request_id]
                logger.debug(f"[GC] Evicted {len(to_remove)} terminal requests")

    @staticmethod
    def _normalize_engine_addr(addr: str) -> str:
        """Strip ``http://`` or ``https://`` scheme prefix from an engine
        address.

        Engine URLs stored internally are bare ``host:port`` strings (e.g.
        ``10.0.0.1:30000``), but callers of the scale-out / scale-in APIs may
        pass fully-qualified URLs like ``http://10.0.0.1:30000``.  This helper
        ensures consistent comparison by removing the scheme if present.

        Note: ``get_url()`` on ``SGLangEngine`` returns ``http://host:port``,
        so we also strip that when collecting existing engine addresses.
        """
        for prefix in ("http://", "https://"):
            if addr.startswith(prefix):
                return addr[len(prefix) :]
        return addr

    @staticmethod
    def _parse_host_port(addr: str) -> tuple[str, int]:
        """Parse a host:port string, supporting IPv6 bracket notation.

        Automatically strips ``http://`` / ``https://`` scheme prefixes
        before parsing.

        Args:
            addr: Address string like "host:port", "[::1]:port",
                  or "http://host:port"

        Returns:
            Tuple of (host, port)

        Raises:
            ValueError: If the address format is invalid
        """
        # Strip scheme prefix so callers can pass full URLs
        addr = RolloutManager._normalize_engine_addr(addr)
        if addr.startswith("["):
            # IPv6 bracket notation: [::1]:8000
            bracket_end = addr.find("]")
            if bracket_end == -1 or bracket_end + 1 >= len(addr) or addr[bracket_end + 1] != ":":
                raise ValueError(f"Invalid IPv6 address format: '{addr}', expected '[host]:port'")
            host = addr[1:bracket_end]
            port_str = addr[bracket_end + 2 :]
        elif addr.count(":") == 1:
            host, port_str = addr.split(":")
        else:
            raise ValueError(f"Invalid address format: '{addr}', expected 'host:port' or '[ipv6]:port'")
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port in address '{addr}': '{port_str}' is not a valid integer")
        return host, port

    def _collect_existing_engine_addrs(self, srv: "RolloutServer") -> set[str]:
        """Collect normalized addresses of all live engines in the given
        server.

        Iterates over every ``EngineGroup`` and calls ``engine.get_url.remote()``
        on each non-None engine reference.  Returns the set of *normalized*
        address strings (scheme stripped) so that the caller can check whether
        a candidate external address is already active in the system.

        Only physically live engines (non-None entries in ``EngineGroup.all_engines``)
        are considered.  URLs from ACTIVE ``ScaleOutRequest`` records are intentionally
        **not** included because those records are never updated when engines are
        evicted or removed.  Including them would prevent re-adding an engine URL
        after eviction (the request stays ACTIVE but the engine is gone).

        This is a *synchronous* helper (uses ``ray.get`` with a short timeout)
        and is expected to be called from the ``scale_out`` concurrency group
        where blocking is acceptable.
        """
        addrs: set[str] = set()
        for group in srv.engine_groups:
            for engine in group.all_engines:
                if engine is not None:
                    try:
                        url = ray.get(engine.get_url.remote(), timeout=5)
                        if url:
                            addrs.add(self._normalize_engine_addr(url))
                    except Exception:
                        pass

        return addrs

    def _collect_in_flight_engine_addrs(self, model_name: str) -> set[str]:
        """Collect normalized engine addresses from non-terminal in-flight
        scale-out requests.

        Scans ``self._scale_out_requests`` for external-mode requests
        (engine_urls provided) targeting *model_name* that have not yet reached
        a terminal state (FAILED, ACTIVE, CANCELLED).  Returns the union of
        their normalized ``engine_urls`` so that the caller can avoid issuing
        duplicate scale-out work.
        """
        addrs: set[str] = set()
        for r in self._scale_out_requests.values():
            if r.model_name == model_name and r.engine_urls and not r.is_terminal():
                addrs.update(self._normalize_engine_addr(u) for u in r.engine_urls)
        return addrs

    def _find_active_scale_request(self) -> Optional[dict]:
        """Return info about any active (non-terminal) scale-out or scale-in
        request or claimed graceful eviction.

        Returns None if no active request exists, otherwise a dict with
        ``type``, ``request_id``, and ``status`` of the blocking request.
        """
        with self._engine_lifecycle_lock:
            # Explicit requests take precedence because an active scale-in may
            # itself have already moved a group into DRAINING.
            for r in self._scale_out_requests.values():
                if not r.is_terminal():
                    return {"type": "scale_out", "request_id": r.request_id, "status": r.status.value}
            for r in self._scale_in_requests.values():
                if not r.is_terminal():
                    return {"type": "scale_in", "request_id": r.request_id, "status": r.status.value}
            for model_name, srv in self.servers.items():
                for group in srv.engine_groups:
                    if (
                        group.is_scaled_out
                        and group.eviction_requested
                        and group.lifecycle_status is EngineGroupLifecycle.DRAINING
                    ):
                        return {
                            "type": "graceful_eviction",
                            "request_id": f"{model_name}:{group.rank_offset}",
                            "status": group.lifecycle_status.value,
                        }
        return None

    @ray.method(concurrency_group="scale_coordination")
    def create_scale_out_request(
        self,
        model_name: str = "default",
        num_replicas: int = 0,
        engine_urls: Optional[list[str]] = None,
        timeout_secs: Optional[float] = None,
    ) -> dict:
        """Create and validate a scale-out request with idempotency guarantees.

        Idempotency semantics:
        - **ray_native** (num_replicas > 0): ``num_replicas`` is the target *absolute* total engine count.
          If the current engine count (including in-flight requests) already meets or
          exceeds the target, a NOOP response is returned immediately (no request is
          created).  Otherwise, only the *delta* engines are requested.
        - **external** (engine_urls provided): ``engine_urls`` are filtered against engines already
          active in the system (by URL) and URLs present in non-terminal in-flight
          scale-out requests.  Only genuinely new URLs proceed.  If all URLs
          are already covered, a NOOP response is returned.
        - ``num_gpus_per_engine`` is always taken from ``self.args.rollout_num_gpus_per_engine``.

        Args:
            model_name: Target model name (default: "default")
            num_replicas: Target absolute total engine count (ray_native mode, num_replicas > 0)
            engine_urls: External engine URLs to add (external mode, when num_replicas == 0)
            timeout_secs: Total timeout for the operation

        Returns:
            Dict with request_id and initial status (or NOOP if idempotent no-op)
        """
        # Auto-detect mode: if num_replicas > 0, use ray_native; otherwise use external
        if num_replicas > 0:
            scale_mode = ScaleOutMode.RAY_NATIVE
        elif engine_urls:
            scale_mode = ScaleOutMode.EXTERNAL
        else:
            raise ValueError("Either num_replicas > 0 or engine_urls must be provided")

        # Mutual exclusion: reject if any scale operation is in progress
        active = self._find_active_scale_request()
        if active is not None:
            return {
                "request_id": str(uuid.uuid4()),
                "status": "CONFLICT",
                "message": (
                    f"Another {active['type']} request is in progress: "
                    f"request_id={active['request_id']}, status={active['status']}"
                ),
            }

        srv = self._get_server(model_name)
        if srv is None:
            raise ValueError(f"Model '{model_name}' not found")

        if scale_mode == ScaleOutMode.RAY_NATIVE:
            if num_replicas <= 0:
                raise ValueError("num_replicas must be > 0 for ray_native mode")

            # Idempotency: num_replicas is the absolute target total engine count.
            # Count logical engines the same way _fetch_engines and scale-in do
            # (node-0 engines, excluding dead ``None`` slots) so multi-node groups
            # and post-scale-in empty slots don't over-count into a false NOOP.
            target_total = num_replicas
            current_total = sum(1 for g in srv.engine_groups for e in g.engines if e is not None)

            # Also count engines in non-terminal in-flight requests for this model
            in_flight_engines = sum(
                r.num_replicas
                for r in self._scale_out_requests.values()
                if r.model_name == model_name and r.num_replicas > 0 and not r.is_terminal()
            )
            effective_current = current_total + in_flight_engines

            if effective_current >= target_total:
                logger.info(
                    f"[ScaleOut] Idempotent no-op for ray_native: target_total={target_total}, "
                    f"effective_current={effective_current} (active={current_total} + in_flight={in_flight_engines})"
                )
                return {
                    "request_id": str(uuid.uuid4()),
                    "status": "NOOP",
                    "message": (
                        f"Already at or above target: effective_current={effective_current} >= "
                        f"target_total={target_total}"
                    ),
                }

            effective_delta = target_total - effective_current
            logger.info(
                f"[ScaleOut] ray_native idempotency: target_total={target_total}, "
                f"effective_current={effective_current}, effective_delta={effective_delta}"
            )

            request = ScaleOutRequest(
                request_id=str(uuid.uuid4()),
                status=ScaleOutStatus.PENDING,
                model_name=model_name,
                num_replicas=effective_delta,
                timeout_secs=timeout_secs or self.args.scale_out_timeout,
            )

        else:  # external mode
            if not engine_urls:
                raise ValueError("engine_urls is required for external mode")

            # Normalize input URLs once so that "http://host:port" and "host:port"
            # are treated identically when compared against existing/in-flight addrs.
            normalized_input = [self._normalize_engine_addr(u) for u in engine_urls]

            # Idempotency: filter out URLs already in the system or in-flight
            # (_collect_existing/in_flight already return normalized addrs)
            existing_urls = self._collect_existing_engine_addrs(srv)
            in_flight_urls = self._collect_in_flight_engine_addrs(model_name)
            already_known = existing_urls | in_flight_urls

            new_urls = [url for url in normalized_input if url not in already_known]
            if not new_urls:
                logger.info(
                    f"[ScaleOut] Idempotent no-op for external: all {len(engine_urls)} URLs "
                    f"are already active or in-flight (existing={len(existing_urls)}, "
                    f"in_flight={len(in_flight_urls)})"
                )
                return {
                    "request_id": str(uuid.uuid4()),
                    "status": "NOOP",
                    "message": (f"All {len(engine_urls)} URLs are already active or in-flight; no new engines to add"),
                }

            if len(new_urls) < len(normalized_input):
                filtered = set(normalized_input) - set(new_urls)
                logger.info(f"[ScaleOut] Idempotency filtered {len(filtered)} URLs already known: {filtered}")

            request = ScaleOutRequest(
                request_id=str(uuid.uuid4()),
                status=ScaleOutStatus.PENDING,
                model_name=model_name,
                engine_urls=new_urls,
                timeout_secs=timeout_secs or self.args.scale_out_timeout,
            )

        # Recheck and insert under the same lifecycle lock used by graceful
        # eviction claims.  Validation above may be slow enough for eviction
        # to win after the initial fast-path check.
        with self._engine_lifecycle_lock:
            active = self._find_active_scale_request()
            if active is not None:
                return {
                    "request_id": str(uuid.uuid4()),
                    "status": "CONFLICT",
                    "message": (
                        f"Another {active['type']} request is in progress: "
                        f"request_id={active['request_id']}, status={active['status']}"
                    ),
                }
            self._scale_out_requests[request.request_id] = request
            self._gc_terminal_requests()
        return request.to_dict()

    @ray.method(concurrency_group="scale_out")
    async def execute_scale_out(self, request_id: str) -> None:
        """Execute a previously created scale-out request. Meant to be called
        via .remote() (fire-and-forget).

        Runs in the ``scale_out`` concurrency group so that long-running
        scale-out operations (which involve waiting for Ray placement groups,
        engine init, health checks, and weight sync) do not block the default
        concurrency group where ``generate``, ``eval``, and status-query
        methods run.

        Args:
            request_id: The request ID returned by create_scale_out_request
        """
        request = self._scale_out_requests.get(request_id)
        if request is None:
            logger.error(f"Scale-out request {request_id} not found")
            return
        if request.status != ScaleOutStatus.PENDING:
            logger.warning(f"Scale-out request {request_id} is not in PENDING state: {request.status}")
            return

        # Auto-detect mode: if num_replicas > 0, it's ray_native; if engine_urls is provided, it's external
        if request.num_replicas > 0:
            await self._scale_out_ray_native(request)
        elif request.engine_urls:
            await self._scale_out_external(request)
        else:
            request.update_status(
                ScaleOutStatus.FAILED, "Invalid request: neither num_replicas nor engine_urls provided"
            )
            logger.error(
                f"[ScaleOut] Invalid request {request.request_id}: neither num_replicas nor engine_urls provided"
            )

    async def _scale_out_ray_native(self, request: ScaleOutRequest) -> None:
        """Execute scale-out in ray_native mode with incremental resource
        acquisition.

        Creates **one independent Placement Group per replica** and processes
        them incrementally.  Replicas whose PG becomes ready within the timeout
        are fully brought up (engine start → health check → weight sync →
        register).  If the timeout expires before all PGs are ready, the
        already-successful replicas are kept (not rolled back) and the request
        finishes with PARTIAL status.

        This design ensures that:
        - Available resources are consumed immediately instead of blocking on
          the slowest replica.
        - Partial success is preserved: if the user requests 3 replicas but
          only 1 can be scheduled, that 1 replica becomes active while the
          remaining 2 are reported as failed.
        - Ray's auto-scaler still gets the PG requests and can provision new
          nodes for the remaining replicas.

        Flow (per replica):
        1. Create a per-replica Placement Group
        2. Wait for PG to be ready (incremental, with global timeout)
        3. Probe GPU topology inside the PG
        4. Create EngineGroup
        5. Start engines (skip router registration during init)
        6. Health check
        7. Weight sync
        8. Register to router (AFTER weight sync to ensure correct weights from first request)
        9. Register to server and health monitor
        """
        logger.info(
            f"[ScaleOut] Starting ray_native scale-out: request_id={request.request_id}, "
            f"model_name={request.model_name}, num_replicas={request.num_replicas}, "
            f"num_gpus_per_engine={self.args.rollout_num_gpus_per_engine}, timeout={request.timeout_secs}s"
        )

        srv = self._get_server(request.model_name)
        if srv is None:
            request.update_status(ScaleOutStatus.FAILED, f"Model '{request.model_name}' not found")
            logger.error(f"[ScaleOut] Model '{request.model_name}' not found")
            return

        try:
            # Step 1: Update status
            request.update_status(ScaleOutStatus.CREATING)
            logger.info(
                f"[ScaleOut] Status: PENDING → CREATING for request {request.request_id}, "
                f"target: {request.num_replicas} replicas ({self.args.rollout_num_gpus_per_engine} GPUs each)"
            )

            if not srv.engine_groups:
                request.update_status(ScaleOutStatus.FAILED, "No existing engine groups to use as template")
                logger.error(f"[ScaleOut] No existing engine groups for model '{request.model_name}'")
                return

            gpus_per_engine = self.args.rollout_num_gpus_per_engine
            actors_per_replica = max(1, gpus_per_engine // self.args.num_gpus_per_node)

            # Step 2: Create one PG per replica so that replicas with available
            # resources can proceed immediately without waiting for the others.
            per_replica_pgs = []
            for i in range(request.num_replicas):
                num_gpus = gpus_per_engine
                accel_resource = device_utils.get_ray_accelerator_name()
                bundles = [{accel_resource: 1, "CPU": 1} for _ in range(num_gpus)]
                pg = ray.util.placement_group(bundles, strategy="PACK")
                per_replica_pgs.append(pg)

            logger.info(
                f"[ScaleOut] Created {len(per_replica_pgs)} per-replica placement groups "
                f"({gpus_per_engine} GPUs each). Waiting for resources (timeout={request.timeout_secs}s)..."
            )

            # Step 3: Incrementally wait for PGs to become ready.
            # We poll all pending PGs and process each one as soon as it is ready.
            # This ensures that if only some resources are available, we scale
            # those replicas without blocking on the rest.
            from relax.distributed.ray.placement_group import InfoActor, sort_key

            start_time = time.time()
            poll_interval = 5.0
            # Track which replicas are still pending PG readiness
            pending_indices = set(range(request.num_replicas))
            succeeded_engine_ids: list[str] = []
            failed_replica_ids: list[str] = []
            # Aggregated per-replica root-cause reasons for failed replicas so the
            # final error_message can surface the real cause (provision timeout /
            # NCCL transport mismatch / weight sync / health) instead of a
            # generic "All N replicas failed".
            failure_reasons: list[ScaleOutFailure] = []

            logger.info(f"[ScaleOut] Next monotonic engine rank: {self._next_engine_rank}")

            # Phase A: Wait for PGs to become ready, with incremental processing
            while pending_indices:
                elapsed = time.time() - start_time
                remaining = request.timeout_secs - elapsed

                if remaining <= 0:
                    break

                # Check for cancellation
                if request.status == ScaleOutStatus.CANCELLED:
                    logger.info(f"[ScaleOut] Request {request.request_id} cancelled during PG wait")
                    for idx in pending_indices:
                        try:
                            ray.util.remove_placement_group(per_replica_pgs[idx])
                        except Exception:
                            pass
                    # Keep already-succeeded replicas (don't roll back)
                    break

                # Check readiness of all pending PGs concurrently
                newly_ready = []
                check_futures = {}
                for idx in list(pending_indices):
                    fut = asyncio.ensure_future(per_replica_pgs[idx].ready())
                    check_futures[idx] = fut

                done_set, _ = await asyncio.wait(
                    check_futures.values(),
                    timeout=min(poll_interval, remaining),
                )

                for idx, fut in check_futures.items():
                    if fut in done_set and not fut.cancelled():
                        try:
                            fut.result()  # Raise if PG creation itself failed
                            newly_ready.append(idx)
                        except Exception as e:
                            logger.warning(f"[ScaleOut] PG for replica {idx} failed: {e}")
                            pending_indices.discard(idx)
                            failed_replica_ids.append(f"replica_{idx}")
                            failure_reasons.append(ScaleOutFailure(ScaleOutFailureCategory.PROVISION_FAILED))
                            try:
                                ray.util.remove_placement_group(per_replica_pgs[idx])
                            except Exception:
                                pass

                if newly_ready:
                    logger.info(f"[ScaleOut] {len(newly_ready)} PGs became ready: indices={newly_ready}")

                # Process all newly-ready replicas concurrently through the full pipeline.
                # Pre-allocate engine offsets so each coroutine gets a unique rank_offset
                # without racing on srv.engine_groups mutations.
                if newly_ready:
                    replica_offsets = {
                        idx: self._reserve_engine_ranks(
                            actors_per_replica,
                            alignment=actors_per_replica,
                        )
                        for idx in newly_ready
                    }

                    async def _bring_up_one(idx: int) -> tuple[int, bool, "ScaleOutFailure | None", int]:
                        r = await self._bring_up_single_replica(
                            request=request,
                            srv=srv,
                            pg=per_replica_pgs[idx],
                            replica_idx=idx,
                            num_gpus=gpus_per_engine,
                            gpus_per_engine=gpus_per_engine,
                            engine_offset=replica_offsets[idx],
                            sort_key=sort_key,
                            InfoActor=InfoActor,
                        )
                        return (idx, r.success, r.reason, replica_offsets[idx])

                    results = await asyncio.gather(
                        *[_bring_up_one(idx) for idx in newly_ready],
                        return_exceptions=True,
                    )

                    for i, result in enumerate(results):
                        replica_idx_for_result = newly_ready[i]
                        pending_indices.discard(replica_idx_for_result)
                        if isinstance(result, BaseException):
                            # Should not normally happen since _bring_up_single_replica
                            # catches exceptions internally, but handle defensively.
                            logger.exception(f"[ScaleOut] Unexpected exception during parallel bring-up: {result}")
                            failed_replica_ids.append(f"replica_{replica_idx_for_result}")
                            failure_reasons.append(ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED))
                            try:
                                ray.util.remove_placement_group(per_replica_pgs[replica_idx_for_result])
                            except Exception:
                                pass
                            continue
                        idx, success, reason, engine_offset = result
                        if success:
                            engine_id = f"engine_{engine_offset}"
                            succeeded_engine_ids.append(engine_id)
                            logger.info(f"[ScaleOut] ✅ Replica {idx} successfully brought up as {engine_id}")
                        else:
                            failed_replica_ids.append(f"replica_{idx}")
                            if reason:
                                failure_reasons.append(reason)
                            logger.warning(f"[ScaleOut] ❌ Replica {idx} failed during bring-up: {reason}")
                            try:
                                ray.util.remove_placement_group(per_replica_pgs[idx])
                            except Exception:
                                pass

                if not newly_ready and pending_indices:
                    elapsed_now = time.time() - start_time
                    remaining_now = request.timeout_secs - elapsed_now
                    if remaining_now > 0:
                        logger.info(
                            f"[ScaleOut] Waiting for {len(pending_indices)} PGs "
                            f"(elapsed={elapsed_now:.0f}s, remaining={remaining_now:.0f}s)"
                        )

            # Phase B: Handle remaining pending PGs (timed out)
            for idx in list(pending_indices):
                logger.warning(f"[ScaleOut] Replica {idx} timed out waiting for resources")
                failed_replica_ids.append(f"replica_{idx}")
                failure_reasons.append(ScaleOutFailure(ScaleOutFailureCategory.PROVISION_TIMEOUT))
                try:
                    ray.util.remove_placement_group(per_replica_pgs[idx])
                except Exception:
                    pass

            # Phase C: Determine final status
            self._update_scale_out_final_status(
                request, srv, succeeded_engine_ids, failed_replica_ids, failure_reasons
            )

        except Exception as e:
            # Surface only the exception type, never its args (may leak); full exc logged below.
            failure = ScaleOutFailure(ScaleOutFailureCategory.UNKNOWN, type(e).__name__)
            request.failure_categories = scale_utils._scale_out_failure_categories([failure])
            request.update_status(ScaleOutStatus.FAILED, f"Scale-out failed: {failure.message()}")
            logger.exception(f"Scale-out failed for request {request.request_id}")

    async def _bring_up_single_replica(
        self,
        request: ScaleOutRequest,
        srv: RolloutServer,
        pg: Any,
        replica_idx: int,
        num_gpus: int,
        gpus_per_engine: int,
        engine_offset: int,
        sort_key: Any,
        InfoActor: Any,
    ) -> ScaleResult:
        """Bring up a single replica: probe topology, create engine, then
        finalize registration.

        Returns ``ScaleResult(True)`` on success, or ``ScaleResult(False, reason)``
        on failure where ``reason`` describes the specific failure (engine init /
        provision timeout / finalize root cause). On failure, engines are rolled back
        but the placement group is NOT removed (caller is responsible for PG cleanup).

        Args:
            request: The parent scale-out request.
            srv: The target RolloutServer.
            pg: The Ray placement group for this replica.
            replica_idx: Index of this replica within the scale-out request (for logging).
            num_gpus: Total GPUs in the PG.
            gpus_per_engine: GPUs per engine.
            engine_offset: Rank offset for the new engine.
            sort_key: Sort key function for GPU topology reordering.
            InfoActor: Ray actor class for GPU topology probing.

        Returns:
            ``ScaleResult(True)`` if the replica was successfully brought up and
            registered, otherwise ``ScaleResult(False, reason)``.
        """
        info_actors = []
        new_group = None
        try:
            # Step 1: Probe GPU topology
            logger.info(f"[ScaleOut] Replica {replica_idx}: probing GPU topology for {num_gpus} GPUs...")
            accelerator_kwargs = get_ray_accelerator_kwargs(1)
            for i in range(num_gpus):
                info_actors.append(
                    InfoActor.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=i,
                        ),
                        **accelerator_kwargs,
                    ).remote()
                )
            gpu_ids = await asyncio.gather(*[actor.get_ip_and_gpu_id.remote() for actor in info_actors])
            for actor in info_actors:
                ray.kill(actor)
            info_actors = []

            bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_gpus)]
            sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
            pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
            pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
            pg_tuple = (pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids)

            logger.info(
                f"[ScaleOut] Replica {replica_idx}: GPU topology probed, "
                f"bundle_indices={pg_reordered_bundle_indices}, gpu_ids={pg_reordered_gpu_ids}"
            )

            # Step 2: Create EngineGroup (skip router registration during init)
            new_group = EngineGroup(
                args=self.args,
                pg=pg_tuple,
                all_engines=[None],  # Single engine per replica
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type="regular",
                rank_offset=engine_offset,
                gpu_offset=0,
                router_ip=srv.router_ip,
                router_port=srv.router_port,
                is_scaled_out=True,
                skip_dcs_registration=True,  # Will be done in _finalize_engine_group_registration
                skip_router_registration=True,  # Will be done in _finalize_engine_group_registration
            )

            # Step 3: Start engines
            init_handles, self._port_cursors = new_group.start_engines(self._port_cursors)
            if not init_handles:
                logger.error(f"[ScaleOut] Replica {replica_idx}: no init handles returned")
                return ScaleResult(False, ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_NO_HANDLES))

            # Step 4: Wait for engine init
            remaining_timeout = max(10.0, request.timeout_secs - (time.time() - request.created_at))
            try:
                await asyncio.wait_for(asyncio.gather(*init_handles), timeout=remaining_timeout)
            except asyncio.TimeoutError:
                # Elastic node couldn't bring the engine up in time → provision timeout.
                logger.error(f"[ScaleOut] Replica {replica_idx}: engine init timed out after {remaining_timeout:.0f}s")
                await self._rollback_engines(new_group)
                return ScaleResult(False, ScaleOutFailure(ScaleOutFailureCategory.PROVISION_TIMEOUT))
            except Exception as e:
                logger.error(f"[ScaleOut] Replica {replica_idx}: engine init failed: {e}")
                await self._rollback_engines(new_group)
                return ScaleResult(False, ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED, str(e)))

            # Step 5: Finalize registration (health check → weight sync → DCS → router → server)
            result = await self._finalize_engine_group_registration(
                request=request,
                srv=srv,
                engines=new_group.engines,
                engine_group=new_group,
                replica_idx=replica_idx,
                log_prefix="[ScaleOut]",
            )
            if not result.success:
                # Keep the pre-created group: the finalizer returns None on
                # failure, but these engine actors still require rollback.
                await self._rollback_engines(new_group)
                return ScaleResult(
                    False, result.reason or ScaleOutFailure(ScaleOutFailureCategory.UNKNOWN, "finalize failed")
                )
            new_group = result.group

            return ScaleResult(True)

        except Exception as e:
            logger.exception(f"[ScaleOut] Replica {replica_idx}: unexpected error: {e}")
            if new_group is not None:
                if new_group in srv.engine_groups:
                    srv.engine_groups.remove(new_group)
                await self._rollback_engines(new_group)
            for actor in info_actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            return ScaleResult(
                False, ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED, f"{type(e).__name__}: {e}")
            )

    async def _scale_out_external(self, request: ScaleOutRequest) -> None:
        """Execute scale-out in external mode.

        Flow:
        1. Parse and validate engine URLs
        2. Connect to external engines (skip router registration)
        3. Apply partial success policy
        4. Finalize registration: health check → DCS → weight sync → router → server
        """
        srv = self._get_server(request.model_name)
        if srv is None:
            request.update_status(ScaleOutStatus.FAILED, f"Model '{request.model_name}' not found")
            return

        new_engines = []
        failed_engine_actors = []
        failure_reasons: list[ScaleOutFailure] = []
        try:
            request.update_status(ScaleOutStatus.CONNECTING)

            router_ip = srv.router_ip
            router_port = srv.router_port
            total_engines = sum(len(g.all_engines) for g in srv.engine_groups)

            # Step 1: Connect to external engines
            for i, addr in enumerate(request.engine_urls):
                try:
                    host, port = self._parse_host_port(addr)
                except ValueError as e:
                    request.failed_engines.append(f"engine_{i}")
                    failure_reasons.append(ScaleOutFailure(ScaleOutFailureCategory.INVALID_ENGINE_ADDRESS, str(e)))
                    logger.warning(f"Invalid address for external engine {addr}: {e}")
                    continue

                # Create SGLangEngine actor (connecting mode).
                # No GPU needed: this actor is an RPC proxy to the external engine;
                # NCCL weight sync is orchestrated via HTTP to the remote SGLang process.
                RolloutRayActor = ray.remote(_resolve_rollout_engine_class(self.args))
                accelerator_kwargs = get_ray_accelerator_kwargs(0.2)
                engine = RolloutRayActor.options(num_cpus=0.2, **accelerator_kwargs).remote(
                    self.args,
                    rank=total_engines + i,
                    worker_type="regular",
                    base_gpu_id=0,
                    num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
                )

                # Initialize connection (skip router registration until weight sync)
                per_engine_timeout = min(request.timeout_secs, 300)
                try:
                    init_handle = engine.init.remote(
                        dist_init_addr=addr,
                        port=port,
                        nccl_port=None,
                        host=host,
                        router_ip=router_ip,
                        router_port=router_port,
                        init_external_kwargs={
                            "external_engine_need_check_fields": ["tp_size", "dp_size", "pp_size", "ep_size", "dtype"],
                            "timeout": 10,
                        },
                        skip_dcs_registration=True,
                        skip_router_registration=True,
                    )
                    await asyncio.wait_for(init_handle, timeout=per_engine_timeout)
                    new_engines.append(engine)
                except Exception as e:
                    request.failed_engines.append(f"engine_{i}")
                    failed_engine_actors.append(engine)
                    failure_reasons.append(
                        ScaleOutFailure(ScaleOutFailureCategory.EXTERNAL_ENGINE_CONNECT_FAILED, str(e))
                    )
                    logger.warning(f"Failed to connect to external engine {addr}: {e}")

            # Step 2: Apply partial success policy
            if request.failed_engines and new_engines:
                policy = self.args.scale_out_partial_success_policy
                if policy == "rollback_all":
                    logger.warning(
                        f"Partial failure ({len(request.failed_engines)} failed), "
                        f"rolling back all per policy '{policy}'"
                    )
                    await self._rollback_engines(new_engines + failed_engine_actors)
                    # rollback_all commits 0 engines, so treat it like a full
                    # failure: surface the aggregated root cause, not just the policy.
                    reason_str = scale_utils._aggregate_scale_out_reasons(failure_reasons)
                    request.failure_categories = scale_utils._scale_out_failure_categories(failure_reasons)
                    message = (
                        f"Partial failure rolled back: {len(request.failed_engines)}/{len(request.engine_urls)} "
                        f"engines failed (policy: {policy})"
                    )
                    if reason_str:
                        message += f". Reasons: {reason_str}"
                    request.update_status(ScaleOutStatus.FAILED, message)
                    return
                else:
                    logger.warning(
                        f"Partial failure ({len(request.failed_engines)} failed), "
                        f"keeping {len(new_engines)} successful engines per policy '{policy}'"
                    )
                    await self._rollback_engines(failed_engine_actors)

            if not new_engines:
                await self._rollback_engines(failed_engine_actors)
                connect_reason = scale_utils._aggregate_scale_out_reasons(failure_reasons)
                request.failure_categories = scale_utils._scale_out_failure_categories(failure_reasons)
                request.update_status(
                    ScaleOutStatus.FAILED,
                    f"Failed to connect to any external engines: {connect_reason}"
                    if connect_reason
                    else "Failed to connect to any external engines",
                )
                return

            # Step 3: Finalize registration (health check → DCS → weight sync → router → server)
            result = await self._finalize_engine_group_registration(
                request=request,
                srv=srv,
                engines=new_engines,
                rank_offset=total_engines,
                router_ip=router_ip,
                router_port=router_port,
                log_prefix="[ScaleOut] External:",
            )

            if result.success:
                request.engine_ids = [f"engine_{total_engines + i}" for i in range(len(new_engines))]
                self._update_scale_out_final_status(
                    request, srv, request.engine_ids, request.failed_engines, failure_reasons
                )
                # Override to ACTIVE since external mode handles partial success differently
                if request.engine_ids:
                    request.update_status(ScaleOutStatus.ACTIVE)
                    logger.info(
                        f"External scale-out completed: {len(new_engines)} engines connected to model '{request.model_name}'"
                    )
            else:
                await self._rollback_engines(new_engines)
                if result.reason is not None:
                    request.failure_categories = scale_utils._scale_out_failure_categories([result.reason])
                request.update_status(
                    ScaleOutStatus.FAILED,
                    f"Engine registration failed: {result.reason}" if result.reason else "Engine registration failed",
                )

        except Exception as e:
            # Surface only the exception type, never its args (may leak); full exc logged below.
            failure = ScaleOutFailure(ScaleOutFailureCategory.UNKNOWN, type(e).__name__)
            request.failure_categories = scale_utils._scale_out_failure_categories([failure])
            request.update_status(ScaleOutStatus.FAILED, f"External scale-out failed: {failure.message()}")
            logger.exception(f"External scale-out failed for request {request.request_id}")
            # Clean up all engines
            all_actors = new_engines + failed_engine_actors
            if all_actors:
                await self._rollback_engines(all_actors)

    # ===================== Scale-Out Helper Methods =====================

    async def _finalize_engine_group_registration(
        self,
        request: ScaleOutRequest,
        srv: RolloutServer,
        engines: list,
        *,
        engine_group: EngineGroup | None = None,
        rank_offset: int = 0,
        router_ip: str | None = None,
        router_port: int | None = None,
        replica_idx: int | None = None,
        log_prefix: str = "[ScaleOut]",
    ) -> EngineFinalizeResult:
        """Finalize engine group registration: health check → weight sync →
        DCS.

        → router → server → health monitor.

        This is a shared helper for both ray_native and external scale-out modes.

        Args:
            request: The scale-out request.
            srv: The target RolloutServer.
            engines: List of engine actors (already started/connected).
            engine_group: Pre-created EngineGroup (for ray_native mode). If None, creates one.
            rank_offset: Rank offset for new engines (used when creating EngineGroup).
            router_ip: Router IP address.
            router_port: Router port.
            replica_idx: Replica index for logging (optional).
            log_prefix: Prefix for log messages.

        Returns:
            An ``EngineFinalizeResult``. On failure ``group`` is None and
            ``reason`` describes the specific failure (health / weight sync /
            precheck transport / DCS / router) so the caller can surface the
            real root cause instead of a generic message.
        """
        replica_str = f"Replica {replica_idx}" if replica_idx is not None else "Engines"
        remaining_timeout = max(10.0, request.timeout_secs - (time.time() - request.created_at))

        # Step 1: Health check
        healthy = await self._health_check_engines(engines, timeout=remaining_timeout)
        if not healthy:
            logger.error(f"{log_prefix} {replica_str}: health check failed")
            return EngineFinalizeResult(False, reason=ScaleOutFailure(ScaleOutFailureCategory.HEALTH_CHECK_FAILED))

        # Step 2: Sync weights from seed engine before publishing the engines
        # to DCS. A failed scale-out engine must never enter the actor's normal
        # DCS topology, otherwise the next full weight update can fail and
        # escalate an isolated scale-out failure into a global restart.
        request.update_status(ScaleOutStatus.WEIGHT_SYNCING)
        logger.info(f"{log_prefix} {replica_str}: starting weight sync from seed engine...")
        sync_timeout = max(30.0, request.timeout_secs - (time.time() - request.created_at))
        sync_reasons: list[ScaleOutFailure] = []
        sync_ok = await self._sync_weights_from_seed_engine(
            engines,
            timeout=sync_timeout,
            model_name=request.model_name,
            run_precheck=(engine_group is not None and getattr(self.args, "scale_weight_sync_precheck", True)),
            reason_sink=sync_reasons,
        )
        if not sync_ok:
            logger.error(f"{log_prefix} {replica_str}: weight sync failed")
            return EngineFinalizeResult(
                False,
                reason=(
                    sync_reasons[-1] if sync_reasons else ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED)
                ),
            )
        logger.info(f"{log_prefix} {replica_str}: weight sync completed")

        # Step 3: Register to DCS coordinator only after weight sync succeeds.
        logger.info(f"{log_prefix} {replica_str}: registering {len(engines)} engines to DCS...")
        try:
            register_handles = [engine.register_dcs.remote() for engine in engines if engine is not None]
            if register_handles:
                remaining_dcs = max(10.0, request.timeout_secs - (time.time() - request.created_at))
                await asyncio.wait_for(asyncio.gather(*register_handles), timeout=remaining_dcs)
            logger.info(f"{log_prefix} {replica_str}: DCS registration completed")
        except Exception as e:
            logger.error(f"{log_prefix} {replica_str}: DCS registration failed: {e}")
            return EngineFinalizeResult(
                False, reason=ScaleOutFailure(ScaleOutFailureCategory.DCS_REGISTRATION_FAILED, str(e))
            )

        # Step 4: Register to router (AFTER weight sync and DCS registration)
        logger.info(f"{log_prefix} {replica_str}: registering to router...")
        try:
            register_router_handles = [engine.register_to_router.remote() for engine in engines if engine is not None]
            if register_router_handles:
                register_results = await asyncio.wait_for(asyncio.gather(*register_router_handles), timeout=30)
                if not all(result is True for result in register_results):
                    rejected_count = sum(result is not True for result in register_results)
                    raise RuntimeError(f"Router rejected {rejected_count} engine(s)")
            logger.info(f"{log_prefix} {replica_str}: router registration completed")
        except Exception as e:
            logger.error(f"{log_prefix} {replica_str}: router registration failed: {e}")
            return EngineFinalizeResult(
                False, reason=ScaleOutFailure(ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED, str(e))
            )

        # Step 5: Create or use existing EngineGroup
        if engine_group is None:
            engine_group = EngineGroup(
                args=self.args,
                pg=None,  # External mode doesn't need placement group
                all_engines=engines,
                num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
                num_new_engines=len(engines),
                worker_type="regular",
                rank_offset=rank_offset,
                gpu_offset=0,
                router_ip=router_ip,
                router_port=router_port,
                is_scaled_out=True,
                skip_dcs_registration=False,  # Already registered above
            )
        else:
            # Mark DCS registration as done (for ray_native mode)
            engine_group.skip_dcs_registration = False

        # Step 6: Add to server
        srv.engine_groups.append(engine_group)
        engine_group.num_new_engines = 0

        # Step 7: Register health monitor
        if self.args.use_fault_tolerance:
            monitor = RolloutHealthMonitor(engine_group, self.args)
            monitor.start()
            self._health_monitors.append(monitor)

        logger.info(
            f"{log_prefix} {replica_str}: fully registered. "
            f"Total engine_groups: {len(srv.engine_groups)}, "
            f"total engines: {sum(len(g.all_engines) for g in srv.engine_groups)}"
        )
        return EngineFinalizeResult(True, group=engine_group)

    def _update_scale_out_final_status(
        self,
        request: ScaleOutRequest,
        srv: RolloutServer,
        succeeded_engine_ids: list[str],
        failed_engine_ids: list[str],
        failure_reasons: list[ScaleOutFailure] | None = None,
    ) -> None:
        """Update the final status of a scale-out request.

        Args:
            request: The scale-out request.
            srv: The target RolloutServer.
            succeeded_engine_ids: List of successfully added engine IDs.
            failed_engine_ids: List of failed engine/replica IDs.
            failure_reasons: Per-replica root-cause reasons (deduped for the
                error_message). Lets the TUI categorize the real cause
                (provision timeout / NCCL transport mismatch / weight sync /
                health) instead of a generic "All N replicas failed".
        """
        request.engine_ids = succeeded_engine_ids
        request.failed_engines = failed_engine_ids

        total_requested = request.num_replicas or len(request.engine_urls)

        # Canonicalize, dedupe (by category), and bound reasons so the
        # error_message never leaks raw exception text / addresses / tracebacks
        # and stays length-bounded.
        reason_str = scale_utils._aggregate_scale_out_reasons(failure_reasons)
        request.failure_categories = scale_utils._scale_out_failure_categories(failure_reasons)

        if not succeeded_engine_ids:
            if reason_str:
                message = f"scale-out failed: {reason_str}. Failed engines: {failed_engine_ids}"
            else:
                message = f"All {total_requested} replicas failed to scale out. Failed: {failed_engine_ids}"
            request.update_status(ScaleOutStatus.FAILED, message)
            logger.error(f"[ScaleOut] ❌ Scale-out completely failed: 0/{total_requested} replicas succeeded")
        elif len(succeeded_engine_ids) == total_requested:
            request.update_status(ScaleOutStatus.ACTIVE)
            logger.info(
                f"[ScaleOut] ✅ Scale-out completed successfully: {total_requested}/{total_requested} "
                f"engines (IDs: {succeeded_engine_ids}) added to model '{request.model_name}'. "
                f"Total engines now: {sum(len(g.all_engines) for g in srv.engine_groups)}"
            )
        else:
            partial_message = (
                f"Partial scale-out: {len(succeeded_engine_ids)}/{total_requested} replicas succeeded. "
                f"Succeeded: {succeeded_engine_ids}. Failed: {failed_engine_ids}."
            )
            if reason_str:
                partial_message += f" Reasons: {reason_str}."
            request.update_status(ScaleOutStatus.PARTIAL, partial_message)
            logger.warning(
                f"[ScaleOut] ⚠️ Partial scale-out: {len(succeeded_engine_ids)}/{total_requested} "
                f"replicas succeeded (IDs: {succeeded_engine_ids}). "
                f"Failed replicas: {failed_engine_ids}. Keeping successful replicas active. "
                f"Total engines now: {sum(len(g.all_engines) for g in srv.engine_groups)}"
            )

    async def _get_current_weight_version(self) -> str | None:
        """Get the current weight version from an existing healthy engine.

        Returns:
            Current weight version string, or None if not available
        """
        for srv in self.servers.values():
            for group in srv.engine_groups:
                for engine in group.engines:
                    if engine is not None:
                        try:
                            version = await asyncio.wait_for(engine.get_weight_version.remote(), timeout=5)
                            if version is not None:
                                return version
                        except Exception:
                            continue
        return None

    def _get_healthy_seed_engines(self, model_name: str = "default") -> list:
        """Return a list of healthy seed engines, preferring initial (non-
        scaled-out) engines."""
        srv = self._get_server(model_name)
        if srv is None:
            return []
        initial_candidates = []
        scaled_candidates = []
        for group in srv.engine_groups:
            target = initial_candidates if not group.is_scaled_out else scaled_candidates
            for engine in group.engines:
                if engine is not None:
                    try:
                        version = ray.get(engine.get_weight_version.remote(), timeout=5)
                        if version is not None and version != "default":
                            target.append(engine)
                    except Exception:
                        continue
        return initial_candidates + scaled_candidates

    def _get_healthy_seed_engine(self, model_name: str = "default"):
        candidates = self._get_healthy_seed_engines(model_name)
        return candidates[0] if candidates else None

    async def _sync_single_engine_weights(
        self,
        seed_engine,
        new_engine,
        engine_index: int,
        total_engines: int,
        master_address: str,
        tp_size: int,
        timeout: float,
    ) -> bool:
        """Sync weights from seed engine to a single new engine via NCCL.

        Returns True on success and False on completed failures. Raises
        ``_WeightSyncTimeoutError`` when remote work may still be running.

        Robustness:
        - Ports for the weight-send group's rank-0 bind are allocated on the
          seed node within a bounded window below the OS ephemeral range to
          avoid port collisions. TODO(agent): for "external" engines the actor
          may be a proxy on a different node than the bind target, so the probe
          could run on the wrong node; out of scope here.
        - Completed init/send failures are retried with a fresh, rotated port
          window. Timeouts fail immediately because the underlying remote work
          may still be running and cannot be retried transactionally.
        - Result validation is fail-closed: only ``success is True`` passes.
        """
        next_start_port = None
        for attempt in range(1, self._WEIGHT_SYNC_MAX_INIT_ATTEMPTS + 1):
            try:
                ports = await self._allocate_weight_sync_ports(
                    seed_engine,
                    tp_size,
                    start_port=next_start_port,
                )
                next_start_port = int(ports[-1]) + 1
                if next_start_port + tp_size > self._WEIGHT_SYNC_PORT_MAX:
                    next_start_port = self._WEIGHT_SYNC_PORT_BASE
            except Exception as e:
                logger.warning(
                    f"[ScaleOut][WeightSync] Port allocation failed for engine "
                    f"{engine_index + 1} (attempt {attempt}/{self._WEIGHT_SYNC_MAX_INIT_ATTEMPTS}): {e}"
                )
                continue
            ports_str = ",".join(ports)
            group_name = f"direct_sync_{uuid.uuid4().hex[:8]}"

            logger.info(
                f"[ScaleOut][WeightSync] Syncing engine {engine_index + 1}/{total_engines} "
                f"(attempt {attempt}/{self._WEIGHT_SYNC_MAX_INIT_ATTEMPTS}): "
                f"master={master_address}, ports={ports_str}, group={group_name}"
            )

            try:
                phase = "NCCL group init"
                dist_backend = device_utils.get_dist_backend()
                init_seed_ref = seed_engine.init_weights_send_group_for_remote_instance.remote(
                    master_address=master_address,
                    ports=ports_str,
                    group_rank=0,
                    world_size=2,
                    group_name=group_name,
                    backend=dist_backend,
                )
                init_new_ref = new_engine.init_weights_send_group_for_remote_instance.remote(
                    master_address=master_address,
                    ports=ports_str,
                    group_rank=1,
                    world_size=2,
                    group_name=group_name,
                    backend=dist_backend,
                )
                init_results = await asyncio.wait_for(
                    asyncio.gather(init_seed_ref, init_new_ref, return_exceptions=True),
                    timeout=min(timeout, 120),
                )

                for j, result in enumerate(init_results):
                    side = "seed" if j == 0 else "new"
                    if not (isinstance(result, dict) and result.get("success") is True):
                        msg = result.get("message", "unknown") if isinstance(result, dict) else repr(result)
                        raise RuntimeError(f"Failed to init NCCL group on {side}: {msg}")

                logger.info(f"[ScaleOut][WeightSync] NCCL group initialized for engine {engine_index + 1}")

                phase = "weight send"
                send_seed_ref = seed_engine.send_weights_to_remote_instance.remote(
                    master_address=master_address,
                    ports=ports_str,
                    group_name=group_name,
                )
                send_new_ref = new_engine.send_weights_to_remote_instance.remote(
                    master_address=master_address,
                    ports=ports_str,
                    group_name=group_name,
                )
                send_results = await asyncio.wait_for(
                    asyncio.gather(send_seed_ref, send_new_ref, return_exceptions=True),
                    timeout=min(timeout, 300),
                )

                for j, result in enumerate(send_results):
                    side = "seed" if j == 0 else "new"
                    if not (isinstance(result, dict) and result.get("success") is True):
                        msg = result.get("message", "unknown") if isinstance(result, dict) else repr(result)
                        raise RuntimeError(f"Failed to send weights on {side}: {msg}")

                logger.info(
                    f"[ScaleOut][WeightSync] Weight sync completed for engine {engine_index + 1}/{total_engines}"
                )
                return True

            except asyncio.TimeoutError as exc:
                # wait_for cancellation does not guarantee cancellation of the
                # underlying Ray/HTTP work. Retrying could overlap two NCCL
                # operations on the same engines, so fail and let the caller
                # roll back the new engine.
                message = (
                    f"[ScaleOut][WeightSync] {phase} timed out for engine {engine_index + 1}; "
                    "not retrying because the remote operation may still be running"
                )
                logger.error(message)
                raise _WeightSyncTimeoutError(message) from exc
            except Exception as e:
                logger.warning(
                    f"[ScaleOut][WeightSync] Failed to sync engine {engine_index + 1} "
                    f"(attempt {attempt}/{self._WEIGHT_SYNC_MAX_INIT_ATTEMPTS}): {e}"
                )
                # Retry with a fresh, rotated port window on the next attempt.
                continue

        logger.warning(
            f"[ScaleOut][WeightSync] Giving up on engine {engine_index + 1} "
            f"after {self._WEIGHT_SYNC_MAX_INIT_ATTEMPTS} attempt(s)"
        )
        return False

    def _next_weight_sync_start_port(self, tp_size: int) -> int:
        """Return a rotating start port inside the bounded weight-sync window.

        The cursor is advanced by ``tp_size`` on every call so concurrent syncs
        (all driven on the same event loop, hence not truly parallel) and
        successive retries request DISJOINT windows, minimising overlap without
        a stateful cross-process lease. It wraps back to the base when the
        window would be exceeded.
        """
        cursor = getattr(self, "_weight_sync_port_cursor", None)
        if cursor is None or cursor + tp_size > self._WEIGHT_SYNC_PORT_MAX:
            cursor = self._WEIGHT_SYNC_PORT_BASE
        self._weight_sync_port_cursor = cursor + tp_size
        return cursor

    async def _allocate_weight_sync_ports(
        self,
        seed_engine,
        tp_size: int,
        start_port: int | None = None,
    ) -> list[str]:
        """Allocate ``tp_size`` consecutive ports ON THE SEED NODE (NCCL rank-0
        bind node) within the bounded window, returned as strings.

        Raises if the seed node has no free block inside the window (rather
        than scanning into the ephemeral range).
        """
        allocation_lock = getattr(self, "_weight_sync_port_allocation_lock", None)
        if allocation_lock is None:
            allocation_lock = asyncio.Lock()
            self._weight_sync_port_allocation_lock = allocation_lock

        async with allocation_lock:
            cursor = getattr(self, "_weight_sync_port_cursor", None)
            if start_port is None:
                start_port = self._next_weight_sync_start_port(tp_size)
            elif cursor is not None:
                start_port = max(start_port, cursor)

            _, base_port = await asyncio.wait_for(
                seed_engine._get_current_node_ip_and_free_port.remote(
                    start_port=start_port,
                    consecutive=tp_size,
                    max_port=self._WEIGHT_SYNC_PORT_MAX - 1,
                ),
                timeout=30,
            )
            next_port = base_port + tp_size
            if next_port + tp_size > self._WEIGHT_SYNC_PORT_MAX:
                next_port = self._WEIGHT_SYNC_PORT_BASE
            # Advance from the block actually resolved by the remote scan, not
            # merely its requested start. This also keeps concurrent retries
            # from converging on the same first-free block.
            self._weight_sync_port_cursor = next_port
        return [str(base_port + i) for i in range(tp_size)]

    async def _allocate_scale_weight_sync_precheck_ports(self, seed_engine, tp_size: int) -> list[str]:
        """Allocate an independent, bounded port block for NCCL precheck."""
        allocation_lock = getattr(self, "_scale_weight_sync_precheck_port_allocation_lock", None)
        if allocation_lock is None:
            allocation_lock = asyncio.Lock()
            self._scale_weight_sync_precheck_port_allocation_lock = allocation_lock

        async with allocation_lock:
            start_port = getattr(
                self, "_scale_weight_sync_precheck_port_cursor", self._SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE
            )
            if start_port + tp_size > self._SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX:
                start_port = self._SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE
            _, base_port = await asyncio.wait_for(
                seed_engine._get_current_node_ip_and_free_port.remote(
                    start_port=start_port,
                    consecutive=tp_size,
                    max_port=self._SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX - 1,
                ),
                timeout=30,
            )
            next_port = base_port + tp_size
            self._scale_weight_sync_precheck_port_cursor = (
                next_port
                if next_port + tp_size <= self._SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX
                else self._SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE
            )
        return [str(base_port + offset) for offset in range(tp_size)]

    async def _run_scale_weight_sync_precheck(
        self,
        seed_engine,
        new_engines: list,
        master_address: str,
        tp_size: int,
        timeout: float,
    ) -> ScaleResult:
        """Gate scale-out engines before touching the ModelRunner NCCL stack.

        Stage 1 (cheap, no NCCL): reject if the seed/new NCCL transport env
        fingerprints disagree — the common elastic-node misconfiguration. Stage
        2 (NCCL probe): a real 2-rank collective for env-matched engines; only
        launch transients retry, everything else fails closed and propagates
        the raw NCCL error.
        """
        # Stage 1: cheap env-fingerprint gate (no NCCL, no subprocess).
        try:
            seed_fingerprint = await asyncio.wait_for(
                seed_engine.get_scale_weight_sync_transport_fingerprint.remote(), timeout=15
            )
        except Exception as exc:
            logger.error(f"[ScaleOut][Precheck] Seed transport fingerprint unavailable: {exc}")
            return ScaleResult(
                False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "seed fingerprint unavailable")
            )
        for engine_index, new_engine in enumerate(new_engines):
            if new_engine is None:
                continue
            try:
                new_fingerprint = await asyncio.wait_for(
                    new_engine.get_scale_weight_sync_transport_fingerprint.remote(), timeout=15
                )
            except Exception as exc:
                logger.error(
                    f"[ScaleOut][Precheck] New engine {engine_index + 1} transport fingerprint unavailable: {exc}"
                )
                return ScaleResult(
                    False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "new fingerprint unavailable")
                )
            match, mismatch_token = scale_utils._scale_weight_sync_precheck_fingerprints_match(
                {"fingerprint": seed_fingerprint}, {"fingerprint": new_fingerprint}
            )
            if not match:
                logger.error(
                    f"[ScaleOut][Precheck] env fingerprint mismatch ({mismatch_token}): "
                    f"seed={seed_fingerprint}, new={new_fingerprint}"
                )
                return ScaleResult(
                    False,
                    ScaleOutFailure(
                        ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH,
                        mismatch_token or "transport_fingerprint_mismatch",
                    ),
                )

        # Stage 2: NCCL connectivity probe (env already validated above).
        retryable = {PrecheckProbeCategory.LAUNCH_TRANSIENT}
        for engine_index, new_engine in enumerate(new_engines):
            if new_engine is None:
                continue
            for attempt in range(1, self._SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS + 1):
                try:
                    ports = await self._allocate_scale_weight_sync_precheck_ports(seed_engine, tp_size)
                except Exception as exc:
                    logger.error(f"[ScaleOut][Precheck] Port allocation failed for engine {engine_index + 1}: {exc}")
                    return ScaleResult(
                        False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "port allocation error")
                    )
                ports_str = ",".join(ports)
                run_token = f"precheck-{uuid.uuid4().hex}"
                per_actor_timeout = min(max(timeout, 10.0), 120.0)
                logger.info(
                    f"[ScaleOut][Precheck] Probing engine {engine_index + 1}/{len(new_engines)} "
                    f"(attempt {attempt}/{self._SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS}): "
                    f"master={master_address}, ports={ports_str}, token={run_token}"
                )
                seed_ref = seed_engine.run_scale_weight_sync_precheck.remote(
                    master_address=master_address,
                    ports=ports_str,
                    group_rank=0,
                    run_token=run_token,
                    tp_size=tp_size,
                    timeout_secs=per_actor_timeout,
                )
                new_ref = new_engine.run_scale_weight_sync_precheck.remote(
                    master_address=master_address,
                    ports=ports_str,
                    group_rank=1,
                    run_token=run_token,
                    tp_size=tp_size,
                    timeout_secs=per_actor_timeout,
                )
                try:
                    seed_result, new_result = await asyncio.wait_for(
                        asyncio.gather(seed_ref, new_ref, return_exceptions=True),
                        timeout=per_actor_timeout + 15,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[ScaleOut][Precheck] Manager timeout for token {run_token}")
                    return ScaleResult(
                        False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "manager timeout")
                    )

                if isinstance(seed_result, Exception):
                    logger.error(f"[ScaleOut][Precheck] Seed actor failed for token {run_token}: {seed_result}")
                    return ScaleResult(
                        False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "seed actor error")
                    )
                if isinstance(new_result, Exception):
                    logger.error(f"[ScaleOut][Precheck] New actor failed for token {run_token}: {new_result}")
                    return ScaleResult(
                        False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "new actor error")
                    )

                # A multi-node TP topology (tp_size spans >1 node) cannot be
                # exercised by the single-node probe, but the real direct sync
                # handles it. Skip the precheck instead of fail-closing the whole
                # scale-out.
                seed_probe = PrecheckProbeCategory.from_wire(seed_result.get("category"))
                new_probe = PrecheckProbeCategory.from_wire(new_result.get("category"))
                if seed_probe.is_skip or new_probe.is_skip:
                    logger.info("[ScaleOut][Precheck] Skipped (unsupported topology); proceeding to real weight sync")
                    return ScaleResult(True)

                if seed_result.get("success") and new_result.get("success"):
                    logger.info(f"[ScaleOut][Precheck] PASS engine {engine_index + 1}")
                    break

                probe_categories = {
                    PrecheckProbeCategory.from_wire(result.get("category"))
                    for result in (seed_result, new_result)
                    if not result.get("success")
                }
                # Surface a clear reason from whichever side failed: the probe's
                # raw exception (``error``) or a pre-launch check's ``message``
                # (e.g. gpu-mapping / insufficient memory — also how a non-NCCL
                # accelerator surfaces here: the GPU-memory query fails).
                raw_error = next(
                    (
                        f"{result.get('error_type')}: {result.get('error')}"
                        if result.get("error")
                        else str(result.get("message") or "")
                        for result in (seed_result, new_result)
                        if not result.get("success") and (result.get("error") or result.get("message"))
                    ),
                    "",
                )
                sorted_values = sorted(category.value for category in probe_categories)
                logger.warning(
                    f"[ScaleOut][Precheck] FAIL engine {engine_index + 1}, categories={sorted_values}: {raw_error}"
                )
                if attempt >= self._SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS or not probe_categories.issubset(
                    retryable
                ):
                    detail = raw_error or ",".join(sorted_values)
                    # Full NCCL detail (fingerprints + per-rank log tails) goes to
                    # this stable log, searchable by run_token; only the concise
                    # ``detail`` reaches the TUI/error_message.
                    seed_tails = " | ".join(r.get("log_tail", "") for r in seed_result.get("results", []))
                    new_tails = " | ".join(r.get("log_tail", "") for r in new_result.get("results", []))
                    logger.error(
                        f"[ScaleOut][Precheck] FAILED token={run_token} engine={engine_index + 1} "
                        f"categories={sorted_values} detail={detail!r}\n"
                        f"  seed fingerprint={seed_result.get('fingerprint')} log_tail={seed_tails}\n"
                        f"  new  fingerprint={new_result.get('fingerprint')} log_tail={new_tails}"
                    )
                    return ScaleResult(False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, detail))
            else:
                return ScaleResult(
                    False, ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "exhausted retry attempts")
                )
        return ScaleResult(True)

    async def _validate_seed_engine(self, seed_engine, timeout: float = 10.0):
        """Validate a seed engine and return (weight_version, master_address)
        or None."""
        try:
            seed_weight_version = await asyncio.wait_for(seed_engine.get_weight_version.remote(), timeout=timeout)
            if not seed_weight_version or seed_weight_version == "default":
                return None
            seed_url = await asyncio.wait_for(seed_engine.get_url.remote(), timeout=5)
            if not seed_url:
                return None
            from urllib.parse import urlparse

            parsed = urlparse(seed_url)
            master_address = parsed.hostname
            if not master_address:
                return None
            return seed_weight_version, master_address
        except Exception as e:
            logger.warning(f"[ScaleOut][WeightSync] Seed validation failed: {e}")
            return None

    async def _sync_weights_from_seed_engine(
        self,
        new_engines: list,
        timeout: float = 180.0,
        model_name: str = "default",
        run_precheck: bool = False,
        reason_sink: list[ScaleOutFailure] | None = None,
    ) -> bool:
        if not new_engines:
            logger.info("[ScaleOut][WeightSync] No new engines to sync")
            return True

        def _record(failure: ScaleOutFailure) -> None:
            if reason_sink is not None:
                reason_sink.append(failure)

        logger.info(f"[ScaleOut][WeightSync] Starting weight sync for {len(new_engines)} engines")

        seed_candidates = self._get_healthy_seed_engines(model_name)
        if not seed_candidates:
            logger.warning("[ScaleOut][WeightSync] No healthy seed engines found, weight sync failed")
            _record(ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "no healthy seed engine"))
            return False

        sync_succeeded = False
        scale_out_owner_acquired = False
        weight_sync_lock_acquired = False
        try:
            # Claim the lifecycle owner before taking the distributed lock.
            # Otherwise scale-out can hold that lock while waiting for a
            # DRAINING group whose eviction is waiting for a training update
            # blocked on the same lock.
            owner_deadline = time.monotonic() + timeout
            while True:
                with self._engine_lifecycle_lock:
                    has_draining_elastic_group = any(
                        group.is_scaled_out and group.lifecycle_status is EngineGroupLifecycle.DRAINING
                        for srv in self.servers.values()
                        for group in srv.engine_groups
                    )
                    if not has_draining_elastic_group:
                        self._scale_out_weight_updating = True
                        scale_out_owner_acquired = True
                        break
                if time.monotonic() >= owner_deadline:
                    _record(
                        ScaleOutFailure(
                            ScaleOutFailureCategory.WEIGHT_SYNC_FAILED,
                            "timed out waiting for elastic removal",
                        )
                    )
                    return False
                await asyncio.sleep(0.5)

            # Prevent Actor-side DCS weight sync from overlapping with this
            # remote instance sync. Both use the seed engine's NCCL stack.
            while not weight_sync_lock_acquired:
                weight_sync_lock_acquired = await asyncio.to_thread(ray.get, self._weight_sync_lock.acquire.remote())
                if not weight_sync_lock_acquired:
                    await asyncio.sleep(0.5)

            # Pause generation on all new engines before weight sync
            # This ensures no pending requests during flush_cache
            logger.info("[ScaleOut][WeightSync] Pausing generation on new engines...")
            pause_refs = []
            for engine in new_engines:
                if engine is not None:
                    pause_refs.append(engine.pause_generation.remote())
            if pause_refs:
                try:
                    pause_results = await asyncio.wait_for(
                        asyncio.gather(*pause_refs, return_exceptions=True),
                        timeout=60,
                    )
                    pause_errors = [result for result in pause_results if isinstance(result, Exception)]
                    if pause_errors:
                        logger.warning(f"[ScaleOut][WeightSync] Failed to pause {len(pause_errors)} new engine(s)")
                        _record(
                            ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "could not pause new engines")
                        )
                        return False
                except Exception as e:
                    logger.warning(f"[ScaleOut][WeightSync] Failed to pause new engines: {e}")
                    _record(ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, f"pause error: {e}"))
                    return False

            pp_size = max(getattr(self.args, "sglang_pp_size", 1), 1)
            rollout_gpus_per_engine = max(getattr(self.args, "rollout_num_gpus_per_engine", 1), 1)
            tp_size = max(rollout_gpus_per_engine // pp_size, 1)

            if run_precheck:
                precheck_seed = None
                precheck_address = None
                for candidate in seed_candidates:
                    validated = await self._validate_seed_engine(candidate)
                    if validated is not None:
                        _, precheck_address = validated
                        precheck_seed = candidate
                        break
                if precheck_seed is None or precheck_address is None:
                    logger.error("[ScaleOut][Precheck] No valid seed available")
                    _record(ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "no valid seed available"))
                    return False
                precheck_result = await self._run_scale_weight_sync_precheck(
                    precheck_seed,
                    new_engines,
                    master_address=precheck_address,
                    tp_size=tp_size,
                    timeout=timeout,
                )
                if not precheck_result.success:
                    logger.error("[ScaleOut][Precheck] Failed; real direct sync will not run")
                    _record(precheck_result.reason or ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL))
                    return False
                # The real sync must use the exact seed whose node and runtime
                # were admitted by precheck.
                seed_candidates = [precheck_seed]

            start_time = time.time()
            pending_engines = [(i, e) for i, e in enumerate(new_engines) if e is not None]
            synced_indices: set[int] = set()
            total_live = len(pending_engines)

            # Iterate over seed candidates with fallback: if the current seed
            # fails (crash, network issue), retry remaining engines with the
            # next candidate.
            for seed_idx, seed_engine in enumerate(seed_candidates):
                if not pending_engines:
                    break
                if time.time() - start_time > timeout:
                    logger.warning("[ScaleOut][WeightSync] Overall timeout reached")
                    break

                validated = await self._validate_seed_engine(seed_engine)
                if validated is None:
                    logger.warning(f"[ScaleOut][WeightSync] Seed candidate {seed_idx} invalid, trying next")
                    continue
                seed_weight_version, master_address = validated

                logger.info(
                    f"[ScaleOut][WeightSync] Using seed candidate {seed_idx} "
                    f"(version={seed_weight_version}) for {len(pending_engines)} engine(s)"
                )

                # Parallel sync with concurrency limit
                max_concurrent = getattr(self.args, "scale_out_max_concurrent_weight_syncs", 4)
                semaphore = asyncio.Semaphore(max_concurrent)
                remaining = max(30.0, timeout - (time.time() - start_time))

                async def _sync_one(idx, engine, _seed=seed_engine, _addr=master_address, _rem=remaining):
                    async with semaphore:
                        return await self._sync_single_engine_weights(
                            seed_engine=_seed,
                            new_engine=engine,
                            engine_index=idx,
                            total_engines=len(new_engines),
                            master_address=_addr,
                            tp_size=tp_size,
                            timeout=_rem,
                        )

                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(
                            *[_sync_one(i, e) for i, e in pending_engines],
                            return_exceptions=True,
                        ),
                        timeout=remaining + 30,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[ScaleOut][WeightSync] Batch sync timed out")
                    _record(ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "batch sync timed out"))
                    return False

                timeout_error = next(
                    (result for result in results if isinstance(result, _WeightSyncTimeoutError)),
                    None,
                )
                if timeout_error is not None:
                    logger.error(f"[ScaleOut][WeightSync] Aborting seed fallback after timeout: {timeout_error}")
                    _record(ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "remote operation timed out"))
                    return False

                next_pending = []
                for (i, e), result in zip(pending_engines, results):
                    if isinstance(result, Exception) or not result:
                        next_pending.append((i, e))
                    else:
                        synced_indices.add(i)
                pending_engines = next_pending

                if pending_engines:
                    logger.info(
                        f"[ScaleOut][WeightSync] {len(pending_engines)} engine(s) failed "
                        f"with seed {seed_idx}, trying next candidate"
                    )

            success_count = len(synced_indices)

            if success_count == 0 and total_live > 0:
                logger.warning("[ScaleOut][WeightSync] Failed to sync any engines")
                _record(
                    ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "NCCL sync failed for all engines")
                )
                return False

            logger.info("[ScaleOut][WeightSync] Flushing cache on new engines")
            flush_refs = []
            for engine in new_engines:
                if engine is not None:
                    flush_refs.append(engine.flush_cache.remote())
            if flush_refs:
                try:
                    await asyncio.wait_for(asyncio.gather(*flush_refs, return_exceptions=True), timeout=30)
                except Exception:
                    logger.warning("[ScaleOut][WeightSync] Some flush_cache calls timed out")

            logger.info(f"[ScaleOut][WeightSync] Weight sync completed: {success_count}/{total_live} engines synced")
            sync_succeeded = success_count == total_live
            if not sync_succeeded:
                _record(
                    ScaleOutFailure(
                        ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, f"only {success_count}/{total_live} engines synced"
                    )
                )
            return sync_succeeded

        finally:
            # Failed engines remain paused until the caller rolls them back, so
            # they cannot serve requests or participate in normal DCS updates.
            if sync_succeeded:
                logger.info("[ScaleOut][WeightSync] Resuming generation on new engines...")
                resume_refs = []
                for engine in new_engines:
                    if engine is not None:
                        resume_refs.append(engine.continue_generation.remote())
                if resume_refs:
                    try:
                        await asyncio.wait_for(asyncio.gather(*resume_refs, return_exceptions=True), timeout=30)
                    except Exception as e:
                        logger.warning(f"[ScaleOut][WeightSync] Some continue_generation calls failed: {e}")
            else:
                logger.warning("[ScaleOut][WeightSync] Keeping failed new engines paused for rollback")
            if scale_out_owner_acquired:
                with self._engine_lifecycle_lock:
                    self._scale_out_weight_updating = False
            if weight_sync_lock_acquired:
                # This is the SAME lock every training weight update acquires;
                # retaining it would spin the next training update forever.
                ray.get(self._weight_sync_lock.release.remote())

    async def _health_check_engines(self, engines: list, timeout: float = 60.0) -> bool:
        """Check health of engines.

        Args:
            engines: List of engine Ray actors
            timeout: Timeout in seconds

        Returns:
            True if all engines are healthy
        """
        logger.info(f"[HealthCheck] Starting health check for {len(engines)} engines (timeout={timeout}s)")
        try:
            health_refs = [engine.health_generate.remote(timeout=timeout) for engine in engines]
            results = await asyncio.wait_for(asyncio.gather(*health_refs), timeout=timeout + 10)
            logger.info(f"[HealthCheck] Results: {results}")
            all_healthy = all(results)
            if all_healthy:
                logger.info(f"[HealthCheck] All {len(engines)} engines are healthy")
            else:
                failed_indices = [i for i, r in enumerate(results) if not r]
                logger.warning(f"[HealthCheck] Some engines unhealthy: failed indices={failed_indices}")
            return all_healthy
        except Exception as e:
            logger.warning(f"[HealthCheck] Health check failed with exception: {e}")
            return False

    async def _rollback_engines(self, engines_or_group) -> None:
        """Rollback failed engines.

        Steps:
        1. Unregister from DCS coordinator
        2. Graceful shutdown (unregisters from router + kills sglang process)
        3. Force-kill Ray actor only if shutdown fails

        Args:
            engines_or_group: List of engines or EngineGroup
        """
        if isinstance(engines_or_group, EngineGroup):
            engines = engines_or_group.all_engines
        else:
            engines = engines_or_group

        for engine in engines:
            if engine is None:
                continue
            try:
                # Unregister from DCS coordinator
                await asyncio.wait_for(engine.unregister_dcs.remote(), timeout=10)
            except Exception as e:
                logger.warning(f"Failed to unregister DCS for engine during rollback: {e}")
            shutdown_ok = False
            try:
                # Graceful shutdown (unregisters from router + kills sglang process)
                await asyncio.wait_for(engine.shutdown.remote(), timeout=30)
                shutdown_ok = True
            except Exception as e:
                logger.warning(f"Failed to shutdown engine during rollback: {e}")
            # Force-kill Ray actor only as a last resort (if shutdown failed or
            # we need to ensure the actor is fully cleaned up).
            if not shutdown_ok:
                try:
                    ray.kill(engine)
                except Exception as e:
                    logger.warning(f"Failed to kill engine actor during rollback: {e}")

    @ray.method(concurrency_group="scale_out")
    def get_scale_out_status(self, request_id: str) -> Optional[dict]:
        """Get the status of a scale-out request.

        Args:
            request_id: The request ID to query

        Returns:
            Request status dict or None if not found
        """
        request = self._scale_out_requests.get(request_id)
        return request.to_dict() if request else None

    @ray.method(concurrency_group="scale_out")
    def cancel_scale_out(self, request_id: str) -> Optional[dict]:
        """Cancel a scale-out request.

        Supports cancellation in both PENDING and CREATING states.
        When cancelled during CREATING, the polling loop inside
        ``_scale_out_ray_native`` will detect the status change and
        abort the placement group wait, cleaning up resources.

        Args:
            request_id: The request ID to cancel

        Returns:
            Updated request status or None if not found/cancellable
        """
        request = self._scale_out_requests.get(request_id)
        if request is None:
            return None
        if not request.can_cancel():
            return {"error": "Request cannot be cancelled in current state", "status": request.status.value}
        previous_status = request.status.value
        request.update_status(ScaleOutStatus.CANCELLED)
        logger.info(f"Scale-out request {request_id} cancelled (was in {previous_status} state)")
        return request.to_dict()

    def get_router_address(self, model_name: str = "default") -> dict:
        srv = self.servers.get(model_name)
        if srv is None:
            return {"router_ip": None, "router_port": None}
        return {"router_ip": srv.router_ip, "router_port": srv.router_port}

    @ray.method(concurrency_group="scale_out")
    def get_engines_info(self, model_name: Optional[str] = None) -> dict:
        """Get information about all engines.

        Args:
            model_name: Model name to query (optional, default: all models)

        Returns:
            Dict with engine information
        """
        result = {"models": {}, "total_engines": 0}

        models_to_query = {model_name: self._get_server(model_name)} if model_name else self.servers

        for name, srv in models_to_query.items():
            if srv is None:
                continue

            model_info = {
                "router_ip": srv.router_ip,
                "router_port": srv.router_port,
                "engine_groups": [],
                "total_engines": 0,
            }

            for i, group in enumerate(srv.engine_groups):
                group_info = {
                    "group_index": i,
                    "worker_type": group.worker_type,
                    "num_gpus_per_engine": group.num_gpus_per_engine,
                    "num_new_engines": group.num_new_engines,
                    "engines": [],
                }

                # Batch-fetch URLs and pid/node_id for all live engines in this group via remote calls
                live_indices = [j for j, e in enumerate(group.all_engines) if e is not None]
                engine_urls = {}
                engine_pids = {}
                engine_node_ids = {}
                if live_indices:
                    try:
                        url_refs = [group.all_engines[j].get_url.remote() for j in live_indices]
                        pid_node_refs = [group.all_engines[j].get_pid_and_node_id.remote() for j in live_indices]
                        urls = ray.get(url_refs, timeout=10)
                        pid_nodes = ray.get(pid_node_refs, timeout=10)
                        engine_urls = dict(zip(live_indices, urls))
                        engine_pids = {idx: pn["pid"] for idx, pn in zip(live_indices, pid_nodes)}
                        engine_node_ids = {idx: pn["node_id"] for idx, pn in zip(live_indices, pid_nodes)}
                    except Exception:
                        logger.debug("Failed to batch-fetch engine URLs/pids, skipping URL info")

                for j, engine in enumerate(group.all_engines):
                    engine_info = {
                        "rank": group.rank_offset + j,
                        "status": "active" if engine is not None else "dead",
                    }
                    if j in engine_urls and engine_urls[j] is not None:
                        engine_info["url"] = engine_urls[j]
                    if j in engine_pids:
                        engine_info["pid"] = engine_pids[j]
                    if j in engine_node_ids:
                        engine_info["node_id"] = engine_node_ids[j]
                    group_info["engines"].append(engine_info)

                model_info["engine_groups"].append(group_info)
                model_info["total_engines"] += len(group.all_engines)

            result["models"][name] = model_info
            result["total_engines"] += model_info["total_engines"]

        return result

    @ray.method(concurrency_group="scale_out")
    def list_all_scale_out_requests(
        self, model_name: Optional[str] = None, status_filter: Optional[str] = None
    ) -> list[dict]:
        """List all scale-out requests with optional filtering.

        Args:
            model_name: Filter by target model name (e.g., "actor", "reward").
                       None returns requests for all models.
            status_filter: Filter by status (e.g., "PENDING", "ACTIVE", "FAILED").
                          None returns requests in all states.
                          Must be a valid ScaleOutStatus value.

        Returns:
            list[dict]: Requests (sorted by created_at descending), each is a
                       ScaleOutRequest.to_dict(). Empty list [] if none match.

        Raises:
            ValueError: If status_filter is not a valid ScaleOutStatus value.

        Examples:
            # All requests
            requests = ray.get(
                rollout_manager.list_all_scale_out_requests.remote()
            )

            # All PENDING requests
            pending = ray.get(
                rollout_manager.list_all_scale_out_requests.remote(
                    status_filter="PENDING"
                )
            )

            # All ACTIVE requests for the 'actor' model
            actor_active = ray.get(
                rollout_manager.list_all_scale_out_requests.remote(
                    model_name="actor",
                    status_filter="ACTIVE"
                )
            )
        """
        requests = list(self._scale_out_requests.values())

        # Filter by model_name
        if model_name is not None:
            requests = [r for r in requests if r.model_name == model_name]

        # Filter by status
        if status_filter is not None:
            try:
                status = ScaleOutStatus(status_filter)
                requests = [r for r in requests if r.status == status]
            except ValueError:
                raise ValueError(
                    f"Invalid status: '{status_filter}'. "
                    f"Must be one of: {', '.join([s.value for s in ScaleOutStatus])}"
                )

        # Sort by created_at descending (newest first)
        requests.sort(key=lambda r: r.created_at, reverse=True)

        logger.info(
            f"[ListRequests] model_name={model_name}, status_filter={status_filter}, found={len(requests)} requests"
        )

        return [r.to_dict() for r in requests]

    @ray.method(concurrency_group="scale_out")
    async def cancel_all_scale_out_requests(
        self, model_name: Optional[str] = None, status_filter: Optional[str] = None, dry_run: bool = False
    ) -> dict:
        """Cancel all scale-out requests matching criteria.

        Only requests in PENDING or CREATING state can be cancelled.
        Requests in other states are skipped and recorded in the result.

        Args:
            model_name: Only cancel requests for this model (optional).
            status_filter: Only cancel requests in this status (optional).
            dry_run: If True, only preview which requests would be cancelled.

        Returns:
            dict: With the following fields:
                - succeeded (List[str]): IDs of successfully cancelled requests
                - skipped (List[dict]): Requests that could not be cancelled and why
                  Format: [{"request_id": "...", "reason": "..."}, ...]
                - total_count (int): Total requests matching the filters
                - dry_run (bool): Whether this was a dry run
                - filters (dict): The filters that were applied

        Examples:
            # Preview which PENDING requests would be cancelled (no changes)
            result = ray.get(
                rollout_manager.cancel_all_scale_out_requests.remote(
                    status_filter="PENDING",
                    dry_run=True
                )
            )
            print(f"Would cancel: {result['succeeded']}")
            print(f"Would skip: {result['skipped']}")

            # Actually cancel all PENDING requests
            result = ray.get(
                rollout_manager.cancel_all_scale_out_requests.remote(
                    status_filter="PENDING",
                    dry_run=False
                )
            )
        """
        requests = list(self._scale_out_requests.values())

        # Apply model_name filter
        if model_name is not None:
            requests = [r for r in requests if r.model_name == model_name]

        # Apply status filter
        if status_filter is not None:
            try:
                status = ScaleOutStatus(status_filter)
                requests = [r for r in requests if r.status == status]
            except ValueError:
                raise ValueError(
                    f"Invalid status: '{status_filter}'. "
                    f"Must be one of: {', '.join([s.value for s in ScaleOutStatus])}"
                )

        succeeded = []
        skipped = []

        for request in requests:
            if dry_run:
                # Dry-run mode: preview only, no changes
                if request.can_cancel():
                    succeeded.append(request.request_id)
                    logger.info(f"[DryRun] Would cancel request {request.request_id}")
                else:
                    reason = (
                        f"Cannot cancel in {request.status.value} state. "
                        f"Only PENDING and CREATING states are cancellable."
                    )
                    skipped.append(
                        {"request_id": request.request_id, "reason": reason, "current_status": request.status.value}
                    )
                    logger.debug(f"[DryRun] Would skip request {request.request_id}: {reason}")
            else:
                # Live mode: actually cancel the request
                if request.can_cancel():
                    request.update_status(ScaleOutStatus.CANCELLED)
                    succeeded.append(request.request_id)
                    logger.info(f"[CancelAll] Cancelled scale-out request {request.request_id}")
                else:
                    reason = (
                        f"Cannot cancel in {request.status.value} state. "
                        f"Only PENDING and CREATING states are cancellable."
                    )
                    skipped.append(
                        {"request_id": request.request_id, "reason": reason, "current_status": request.status.value}
                    )
                    logger.warning(f"[CancelAll] Skipped request {request.request_id}: {reason}")

        result = {
            "succeeded": succeeded,
            "skipped": skipped,
            "total_count": len(requests),
            "dry_run": dry_run,
            "filters": {"model_name": model_name, "status_filter": status_filter},
        }

        logger.info(
            f"[CancelAll] Processed {len(requests)} requests: "
            f"cancelled={len(succeeded)}, skipped={len(skipped)}, dry_run={dry_run}"
        )

        return result

    @ray.method(concurrency_group="scale_in")
    def set_weight_updating(self, is_updating: bool) -> bool:
        """Acquire or release the fully-async topology lease.

        A DRAINING group owns the scale-in fence, so a new update must retry.
        An update admitted first may finish; scale-in waits for this owner to
        release instead of mutating the live NCCL topology underneath it.
        """
        with self._engine_lifecycle_lock:
            if is_updating and any(
                group.is_scaled_out and group.lifecycle_status is EngineGroupLifecycle.DRAINING
                for srv in self.servers.values()
                for group in srv.engine_groups
            ):
                return False
            self._training_weight_updating = is_updating
            return True

    @ray.method(concurrency_group="scale_in")
    async def sync_weights_for_scaled_out_engines(
        self,
        model_name: str = "default",
        timeout: float = 180.0,
    ) -> dict:
        """Sync weights for scaled-out engines after DCS weight sync completes.

        This method should be called by Actor after DCS weight update completes.
        It syncs weights to engines that were added via scale-out (is_scaled_out=True)
        and are not registered in DCS topology.

        Args:
            model_name: Model name to sync weights for.
            timeout: Timeout in seconds for weight sync.

        Returns:
            Dict with sync result:
                - success: bool - whether sync completed successfully
                - synced_count: int - number of engines synced
                - failed_engines: list - engine IDs that failed to sync
                - error_message: str | None - error details if failed
        """
        srv = self._get_server(model_name)
        if srv is None:
            return {
                "success": False,
                "synced_count": 0,
                "failed_engines": [],
                "error_message": f"Model '{model_name}' not found",
            }

        # Collect scaled-out engines that are not in DCS topology
        scaled_out_engines = []
        for group in srv.engine_groups:
            if not group.is_scaled_out:
                continue
            for engine in group.engines:
                if engine is not None:
                    scaled_out_engines.append(engine)

        if not scaled_out_engines:
            logger.info("[ScaleOut][WeightSync] No scaled-out engines to sync")
            return {
                "success": True,
                "synced_count": 0,
                "failed_engines": [],
                "error_message": None,
            }

        logger.info(f"[ScaleOut][WeightSync] Starting weight sync for {len(scaled_out_engines)} scaled-out engines")

        # Sync from seed engine
        try:
            sync_reasons: list[ScaleOutFailure] = []
            success = await self._sync_weights_from_seed_engine(
                scaled_out_engines,
                timeout=timeout,
                model_name=model_name,
                reason_sink=sync_reasons,
            )

            if success:
                logger.info(f"[ScaleOut][WeightSync] Successfully synced {len(scaled_out_engines)} scaled-out engines")
                return {
                    "success": True,
                    "synced_count": len(scaled_out_engines),
                    "failed_engines": [],
                    "error_message": None,
                }
            else:
                logger.warning("[ScaleOut][WeightSync] Failed to sync some scaled-out engines")
                return {
                    "success": False,
                    "synced_count": 0,
                    "failed_engines": [f"engine_{i}" for i in range(len(scaled_out_engines))],
                    "error_message": (
                        sync_reasons[-1].message() if sync_reasons else "Weight sync failed for some engines"
                    ),
                }

        except Exception as e:
            logger.exception(f"[ScaleOut][WeightSync] Exception during weight sync: {e}")
            return {
                "success": False,
                "synced_count": 0,
                "failed_engines": [f"engine_{i}" for i in range(len(scaled_out_engines))],
                "error_message": str(e),
            }

    @ray.method(concurrency_group="scale_coordination")
    def create_scale_in_request(
        self,
        model_name: str = "default",
        num_replicas: int = 0,
        engine_urls: Optional[list] = None,
        timeout_secs: Optional[float] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        # Mutual exclusion: reject if any scale operation is in progress
        active = self._find_active_scale_request()
        if active is not None:
            return {
                "request_id": str(uuid.uuid4()),
                "status": "CONFLICT",
                "message": (
                    f"Another {active['type']} request is in progress: "
                    f"request_id={active['request_id']}, status={active['status']}"
                ),
            }

        srv = self._get_server(model_name)
        if srv is None:
            raise ValueError(f"Model '{model_name}' not found")

        # Count initial (non-scaled-out) engines to enforce the lower bound.
        initial_count = sum(1 for g in srv.engine_groups if not g.is_scaled_out for e in g.engines if e is not None)

        if num_replicas > 0:
            if num_replicas < initial_count:
                return {
                    "request_id": str(uuid.uuid4()),
                    "status": "REJECTED",
                    "message": (
                        f"Cannot scale below initial engine count: "
                        f"num_replicas={num_replicas} < initial_engines={initial_count}"
                    ),
                }
            current_total = sum(1 for g in srv.engine_groups for e in g.engines if e is not None)
            if current_total <= num_replicas:
                return {
                    "request_id": str(uuid.uuid4()),
                    "status": "NOOP",
                    "message": f"Already at or below target replicas: current_total={current_total}, target={num_replicas}",
                }
        elif engine_urls:
            pass
        else:
            # num_replicas <= 0 and no engine_urls: semantically equals "target 0 < initial
            # engine count". Return a clean REJECTED instead of raising ValueError (which the
            # HTTP handler would surface as a 500 Internal Server Error).
            return {
                "request_id": str(uuid.uuid4()),
                "status": "REJECTED",
                "message": (
                    f"Cannot scale below initial engine count: "
                    f"num_replicas={num_replicas} < initial_engines={initial_count}"
                ),
            }

        request = ScaleInRequest(
            request_id=str(uuid.uuid4()),
            status=ScaleInStatus.PENDING,
            model_name=model_name,
            num_replicas=num_replicas,
            engine_urls=engine_urls or [],
            timeout_secs=timeout_secs or getattr(self.args, "scale_in_drain_timeout", 30.0) + 60.0,
            force=force,
            dry_run=dry_run,
        )
        # Recheck and insert atomically against graceful eviction claims.
        with self._engine_lifecycle_lock:
            active = self._find_active_scale_request()
            if active is not None:
                return {
                    "request_id": str(uuid.uuid4()),
                    "status": "CONFLICT",
                    "message": (
                        f"Another {active['type']} request is in progress: "
                        f"request_id={active['request_id']}, status={active['status']}"
                    ),
                }
            self._scale_in_requests[request.request_id] = request
            self._gc_terminal_requests()
        return request.to_dict()

    @ray.method(concurrency_group="scale_in")
    async def execute_scale_in(self, request_id: str) -> None:
        request = self._scale_in_requests.get(request_id)
        if request is None:
            logger.error(f"Scale-in request {request_id} not found")
            return
        if request.status != ScaleInStatus.PENDING:
            logger.warning(f"Scale-in request {request_id} is not in PENDING state: {request.status}")
            return

        await self._scale_in(request)

    async def _scale_in(self, request: ScaleInRequest) -> None:
        logger.info(
            f"[ScaleIn] Starting scale-in: request_id={request.request_id}, "
            f"model_name={request.model_name}, num_replicas={request.num_replicas}, "
            f"dry_run={request.dry_run}, force={request.force}"
        )
        srv = self._get_server(request.model_name)
        if srv is None:
            request.update_status(ScaleInStatus.FAILED, f"Model '{request.model_name}' not found (no rollout server)")
            return

        selected_groups: dict[int, EngineGroup] = {}
        completed_eviction_groups: set[int] = set()
        try:
            url_candidates = None
            if request.engine_urls:
                url_candidates = await self._resolve_scale_in_url_candidates(request, srv)

            with self._engine_lifecycle_lock:
                # Selection and lifecycle claim are one transaction with the
                # SIGTERM handler. Whichever enters first determines whether
                # a pending eviction is adopted into this target scale-in.
                if request.selected_engines:
                    logger.warning("[ScaleIn] Request %s already has claimed targets", request.request_id)
                    return
                engine_infos = self._select_engines_for_removal(request, srv, url_candidates=url_candidates)
                if not engine_infos:
                    if request.num_replicas > 0:
                        request.update_status(ScaleInStatus.COMPLETED)
                        logger.info(
                            f"[ScaleIn] No-op: already at or below target replicas (target={request.num_replicas})"
                        )
                        return
                    request.update_status(ScaleInStatus.FAILED, "No engines selected for removal")
                    return

                request.selected_engines = [
                    f"group_{group.rank_offset}_engine_{node0_idx}" for group, node0_idx in engine_infos
                ]
                logger.info(f"[ScaleIn] Selected {len(engine_infos)} engines for removal: {request.selected_engines}")

                if request.dry_run:
                    request.update_status(ScaleInStatus.COMPLETED)
                    logger.info("[ScaleIn] Dry-run complete, no engines removed")
                    return

                selected_groups = {id(group): group for group, _ in engine_infos}
                if any(
                    group.lifecycle_status not in (EngineGroupLifecycle.ACTIVE, EngineGroupLifecycle.DRAINING)
                    for group in selected_groups.values()
                ):
                    request.update_status(ScaleInStatus.FAILED, "Selected engine group is already being removed")
                    return
                for group in selected_groups.values():
                    group.lifecycle_status = EngineGroupLifecycle.DRAINING

            # Publish DRAINING before waiting. Existing transfers may finish,
            # but a new fully-async lease cannot enter while removal is pending.
            wait_start = time.monotonic()
            while self._is_weight_updating:
                elapsed = time.monotonic() - wait_start
                if elapsed >= request.timeout_secs:
                    request.update_status(
                        ScaleInStatus.FAILED,
                        f"Timed out waiting {request.timeout_secs}s for the weight-update fence",
                    )
                    logger.warning("[ScaleIn] Timed out waiting for weight update; aborting scale-in")
                    return
                await asyncio.sleep(min(1.0, request.timeout_secs - elapsed))

            drain_timeout = getattr(self.args, "scale_in_drain_timeout", 30.0)
            shutdown_timeout = getattr(self.args, "scale_in_shutdown_timeout", 20.0)

            request.update_status(ScaleInStatus.DRAINING)
            removed, failed = await self._remove_live_engines(
                srv,
                engine_infos,
                drain_timeout=drain_timeout,
                shutdown_timeout=shutdown_timeout,
                force=request.force,
            )

            request.update_status(ScaleInStatus.REMOVING)
            request.removed_engines = removed
            request.failed_engines = failed
            completed_eviction_groups = {
                id(group)
                for group, node0_idx in engine_infos
                if group.eviction_requested and f"group_{group.rank_offset}_engine_{node0_idx}" in removed
            }

            if failed:
                failure_prefix = "Scale-in partially failed" if removed else "Scale-in failed"
                request.update_status(
                    ScaleInStatus.FAILED,
                    f"{failure_prefix}: {len(failed)} engines could not be removed",
                )
            else:
                request.update_status(ScaleInStatus.COMPLETED)
                logger.info(
                    f"[ScaleIn] ✅ Scale-in completed: removed {len(removed)} engines, "
                    f"remaining groups: {len(srv.engine_groups)}"
                )

        except Exception as e:
            request.update_status(ScaleInStatus.FAILED, f"Scale-in failed: {e}")
            logger.exception(f"[ScaleIn] Unhandled error in scale-in for request {request.request_id}")
        finally:
            with self._engine_lifecycle_lock:
                for group in selected_groups.values():
                    if id(group) in completed_eviction_groups:
                        group.eviction_requested = False
                    if not group.eviction_requested and group.lifecycle_status is EngineGroupLifecycle.DRAINING:
                        group.lifecycle_status = EngineGroupLifecycle.ACTIVE

    async def _resolve_scale_in_url_candidates(self, request: ScaleInRequest, srv) -> list:
        """Resolve URL targets without holding the lifecycle lock."""
        target_urls = {self._normalize_engine_addr(url) for url in request.engine_urls}
        with self._engine_lifecycle_lock:
            snapshots = [
                (group, node0_idx, engine)
                for group in srv.engine_groups
                if group.is_scaled_out
                and group.lifecycle_status in (EngineGroupLifecycle.ACTIVE, EngineGroupLifecycle.DRAINING)
                for node0_idx, engine in enumerate(group.engines)
                if engine is not None
            ]

        async def _resolve(group, node0_idx, engine):
            try:
                url = await asyncio.wait_for(engine.get_url.remote(), timeout=5)
                if url and self._normalize_engine_addr(url) in target_urls:
                    return group, node0_idx, engine
            except Exception as e:
                logger.warning(f"Failed to get URL for engine group_{group.rank_offset}_engine_{node0_idx}: {e}")
            return None

        results = await asyncio.gather(*[_resolve(*snapshot) for snapshot in snapshots])
        return [result for result in results if result is not None]

    def _select_engines_for_removal(self, request: ScaleInRequest, srv, *, url_candidates: list | None = None) -> list:
        """Select engines eligible for removal during scale-in.

        Only engines belonging to groups that were added via scale-out
        (``is_scaled_out=True``) are candidates.  Initial groups are never
        touched, regardless of how ``num_replicas`` / ``engine_urls`` are
        specified.
        """
        # Collect one coherent snapshot.  A DRAINING group can be a SIGTERM
        # intent claimed after this request was inserted; prefer it so a
        # target-based scale-in does not remove a second elastic engine.
        with self._engine_lifecycle_lock:
            engine_snapshots = []
            for group in srv.engine_groups:
                if not group.is_scaled_out or group.lifecycle_status not in (
                    EngineGroupLifecycle.ACTIVE,
                    EngineGroupLifecycle.DRAINING,
                ):
                    continue
                for node0_idx, engine in enumerate(group.engines):
                    if engine is not None:
                        engine_snapshots.append((group, node0_idx, engine))
            current_total = sum(1 for group in srv.engine_groups for engine in group.engines if engine is not None)

        if request.num_replicas > 0:
            # Count ALL live engines (initial + scaled-out) to decide how many
            # to remove so the cluster reaches the target size.
            num_to_remove = current_total - request.num_replicas
            if num_to_remove <= 0:
                return []
            draining = [item for item in engine_snapshots if item[0].lifecycle_status is EngineGroupLifecycle.DRAINING]
            active = [item for item in engine_snapshots if item[0].lifecycle_status is EngineGroupLifecycle.ACTIVE]
            # Adopt SIGTERM intents first, then remove the most recently added
            # ACTIVE engines only for the remaining target delta.
            engine_snapshots = draining[:num_to_remove]
            remaining = num_to_remove - len(engine_snapshots)
            if remaining > 0:
                engine_snapshots.extend(active[-remaining:])
        elif request.engine_urls:
            # URL probes happen outside the lifecycle lock. Revalidate actor
            # identity against the current topology before claiming removal.
            resolved_by_slot = {(id(group), node0_idx): engine for group, node0_idx, engine in (url_candidates or [])}
            engine_snapshots = [
                (group, node0_idx, engine)
                for group, node0_idx, engine in engine_snapshots
                if resolved_by_slot.get((id(group), node0_idx)) is engine
            ]

        return [(group, node0_idx) for group, node0_idx, _ in engine_snapshots]

    async def _remove_live_engines(
        self,
        srv,
        engine_infos: list,
        *,
        drain_timeout: float,
        shutdown_timeout: float,
        force: bool,
    ) -> tuple[list[str], list[str]]:
        """Remove live actors in Router -> drain -> DCS -> shutdown -> PG
        order."""
        unregistered_engine_infos, router_failed = await self._drain_engines(
            engine_infos,
            timeout=drain_timeout,
            force=force,
        )

        removal_targets = [
            (
                group,
                node0_idx,
                f"group_{group.rank_offset}_engine_{node0_idx}",
                self._get_live_engine_actors(group, node0_idx),
            )
            for group, node0_idx in unregistered_engine_infos
        ]
        dcs_results = await asyncio.gather(
            *[self._unregister_engine_dcs(engine_id, live_actors) for _, _, engine_id, live_actors in removal_targets],
            return_exceptions=True,
        )
        removed = []
        failed = list(router_failed)
        shutdown_targets = []
        for target, result in zip(removal_targets, dcs_results):
            _, _, engine_id, _ = target
            if isinstance(result, BaseException):
                logger.warning(f"[ScaleIn] Failed to unregister DCS for engine {engine_id}: {result}")
            shutdown_targets.append(target)

        await asyncio.gather(
            *[
                self._shutdown_engine_actors(group, engine_id, live_actors, shutdown_timeout)
                for group, _, engine_id, live_actors in shutdown_targets
            ]
        )
        for _, _, engine_id, _ in shutdown_targets:
            removed.append(engine_id)
            logger.info(f"[ScaleIn] Removed engine {engine_id}")

        self._cleanup_engine_groups(srv)
        return removed, failed

    async def _drain_engines(self, engine_infos: list, timeout: float, force: bool) -> tuple[list, list[str]]:
        """Remove all engines from the router, then wait once for the drain
        period.

        Previous implementation waited ``timeout`` seconds per engine (serial),
        causing total drain time of ``N * timeout``. Now we remove all engines
        from routing concurrently and then do a single ``asyncio.sleep(timeout)``
        so total drain time is always just ``timeout`` regardless of engine count.

        Returns the engine infos confirmed absent from the router and the IDs
        of engines that must remain alive.
        """

        # Step 1: Stop routing new requests to all target engines.
        async def _mark_one_engine(group, node0_idx):
            engine = group.engines[node0_idx]
            if engine is None:
                return
            engine_id = f"group_{group.rank_offset}_engine_{node0_idx}"

            router_removal_timeout = max(5.0, min(timeout, 30.0))
            removed = await asyncio.wait_for(
                engine.unregister_from_router.remote(
                    wait_for_removal=True,
                    timeout=router_removal_timeout,
                ),
                timeout=router_removal_timeout + 5.0,
            )
            if not removed:
                raise RuntimeError(f"Router did not remove engine {engine_id}")
            logger.info(f"[ScaleIn] Removed engine {engine_id} from router")

            for monitor in self._health_monitors:
                if monitor._engine_group is group:
                    monitor.mark_intentionally_removed(node0_idx)

        # Stop routing to all engines concurrently. Only engines with confirmed
        # removal are safe to drain and shut down.
        removal_results = await asyncio.gather(
            *[_mark_one_engine(g, idx) for g, idx in engine_infos],
            return_exceptions=True,
        )
        unregistered_engine_infos = []
        failed_engine_ids = []
        for (group, node0_idx), result in zip(engine_infos, removal_results):
            if isinstance(result, BaseException):
                engine_id = f"group_{group.rank_offset}_engine_{node0_idx}"
                failure_detail = str(result) or type(result).__name__
                logger.warning(
                    f"[ScaleIn] Keeping engine {engine_id} alive because router removal failed: {failure_detail}"
                )
                failed_engine_ids.append(engine_id)
            else:
                unregistered_engine_infos.append((group, node0_idx))

        # Step 2: Single drain wait for engines that are safe to remove.
        drain_time = timeout if not force else 0
        if drain_time > 0 and unregistered_engine_infos:
            engine_ids = [f"group_{g.rank_offset}_engine_{idx}" for g, idx in unregistered_engine_infos]
            logger.info(
                f"[ScaleIn] Draining {len(unregistered_engine_infos)} engines "
                f"for {drain_time}s (force={force}): {engine_ids}"
            )
            await asyncio.sleep(drain_time)

        return unregistered_engine_infos, failed_engine_ids

    @staticmethod
    def _get_live_engine_actors(group, node0_idx: int) -> list[tuple[int, object]]:
        nodes_per_engine = group.nodes_per_engine
        indices = range(node0_idx * nodes_per_engine, (node0_idx + 1) * nodes_per_engine)
        return [
            (i, group.all_engines[i])
            for i in indices
            if i < len(group.all_engines) and group.all_engines[i] is not None
        ]

    @staticmethod
    async def _unregister_engine_dcs(engine_id: str, live_actors: list[tuple[int, object]]) -> None:
        async def _unregister_dcs(engine):
            return await asyncio.wait_for(engine.unregister_dcs.remote(), timeout=10)

        dcs_results = await asyncio.gather(
            *[_unregister_dcs(engine) for _, engine in live_actors],
            return_exceptions=True,
        )
        dcs_failures = [
            (i, result) for (i, _), result in zip(live_actors, dcs_results) if isinstance(result, BaseException)
        ]
        if dcs_failures:
            failed_indices = ", ".join(str(i) for i, _ in dcs_failures)
            raise RuntimeError(f"Failed to unregister DCS for engine {engine_id}[{failed_indices}]")

    async def _shutdown_engine_actors(
        self,
        group,
        engine_id: str,
        live_actors: list[tuple[int, object]],
        shutdown_timeout: float,
    ) -> None:
        async def _shutdown(engine):
            return await asyncio.wait_for(engine.shutdown.remote(), timeout=shutdown_timeout)

        shutdown_results = await asyncio.gather(
            *[_shutdown(engine) for _, engine in live_actors],
            return_exceptions=True,
        )
        for (i, engine), result in zip(live_actors, shutdown_results):
            if isinstance(result, BaseException):
                logger.warning(f"[ScaleIn] Failed to shutdown engine {engine_id}[{i}]: {result}")
                try:
                    ray.kill(engine)
                except Exception as e:
                    logger.warning(f"[ScaleIn] Failed to kill engine actor {engine_id}[{i}]: {e}")

        with self._engine_lifecycle_lock:
            for i, _ in live_actors:
                group.all_engines[i] = None

    async def _remove_engine(self, group, node0_idx: int, shutdown_timeout: float) -> None:
        engine_id = f"group_{group.rank_offset}_engine_{node0_idx}"
        live_actors = self._get_live_engine_actors(group, node0_idx)
        try:
            await self._unregister_engine_dcs(engine_id, live_actors)
        except Exception as e:
            logger.warning(f"[ScaleIn] Failed to unregister DCS for engine {engine_id}: {e}")
        await self._shutdown_engine_actors(group, engine_id, live_actors, shutdown_timeout)

    def _cleanup_engine_groups(self, srv) -> None:
        with self._engine_lifecycle_lock:
            groups_to_remove = [
                group
                for group in srv.engine_groups
                if group.lifecycle_status in (EngineGroupLifecycle.ACTIVE, EngineGroupLifecycle.DRAINING)
                and all(engine is None for engine in group.all_engines)
            ]
            monitors_to_remove = [
                monitor
                for monitor in self._health_monitors
                if any(monitor._engine_group is group for group in groups_to_remove)
            ]
            for group in groups_to_remove:
                group.lifecycle_status = EngineGroupLifecycle.REMOVING
                srv.engine_groups.remove(group)
            for monitor in monitors_to_remove:
                self._health_monitors.remove(monitor)

        for monitor in monitors_to_remove:
            monitor.stop()

        for group in groups_to_remove:
            if group.pg is not None:
                try:
                    ray.util.remove_placement_group(group.pg[0])
                except Exception as e:
                    logger.warning(f"[ScaleIn] Failed to remove placement group: {e}")
            with self._engine_lifecycle_lock:
                group.lifecycle_status = EngineGroupLifecycle.REMOVED

        if groups_to_remove:
            logger.info(f"[ScaleIn] Cleaned up {len(groups_to_remove)} empty engine groups")

    @ray.method(concurrency_group="scale_in")
    def get_scale_in_status(self, request_id: str) -> Optional[dict]:
        request = self._scale_in_requests.get(request_id)
        return request.to_dict() if request else None

    @ray.method(concurrency_group="scale_in")
    def list_all_scale_in_requests(
        self, model_name: Optional[str] = None, status_filter: Optional[str] = None
    ) -> list[dict]:
        """List all scale-in requests with optional filtering.

        Args:
            model_name: Filter by target model name (e.g., "actor", "reward").
                       None returns requests for all models.
            status_filter: Filter by status (e.g., "PENDING", "DRAINING", "REMOVING", "COMPLETED", "FAILED").
                          None returns requests in all states.
                          Must be a valid ScaleInStatus value.

        Returns:
            list[dict]: Requests (sorted by created_at descending), each is a
                       ScaleInRequest.to_dict(). Empty list [] if none match.

        Raises:
            ValueError: If status_filter is not a valid ScaleInStatus value.

        Examples:
            # All requests
            requests = ray.get(
                rollout_manager.list_all_scale_in_requests.remote()
            )

            # All PENDING requests
            pending = ray.get(
                rollout_manager.list_all_scale_in_requests.remote(
                    status_filter="PENDING"
                )
            )

            # All COMPLETED requests for the 'actor' model
            actor_completed = ray.get(
                rollout_manager.list_all_scale_in_requests.remote(
                    model_name="actor",
                    status_filter="COMPLETED"
                )
            )
        """
        requests = list(self._scale_in_requests.values())

        # Filter by model_name
        if model_name is not None:
            requests = [r for r in requests if r.model_name == model_name]

        # Filter by status
        if status_filter is not None:
            try:
                status = ScaleInStatus(status_filter)
                requests = [r for r in requests if r.status == status]
            except ValueError:
                raise ValueError(
                    f"Invalid status: '{status_filter}'. Must be one of: {', '.join([s.value for s in ScaleInStatus])}"
                )

        # Sort by created_at descending (newest first)
        requests.sort(key=lambda r: r.created_at, reverse=True)

        logger.info(
            f"[ListScaleInRequests] model_name={model_name}, status_filter={status_filter}, found={len(requests)} requests"
        )

        return [r.to_dict() for r in requests]

    # ===================== Eviction Monitoring =====================

    def _start_eviction_monitor(self):
        """Start a background thread that periodically polls engines for
        SIGTERM eviction."""
        if self._eviction_monitor_thread is not None:
            return

        self._eviction_monitor_stop.clear()
        self._eviction_monitor_thread = threading.Thread(
            target=self._eviction_monitor_loop,
            name="EvictionMonitor",
            daemon=True,
        )
        self._eviction_monitor_thread.start()
        logger.info("[Eviction] Eviction monitor started (interval=%.1fs)", self._eviction_check_interval)

    def _stop_eviction_monitor(self):
        """Stop the eviction monitor thread."""
        if self._eviction_monitor_thread is None:
            return
        self._eviction_monitor_stop.set()
        self._eviction_monitor_thread.join(timeout=self._eviction_check_interval + 5)
        if self._eviction_monitor_thread.is_alive():
            logger.warning("[Eviction] Eviction monitor thread did not terminate in time")
        self._eviction_monitor_thread = None
        logger.info("[Eviction] Eviction monitor stopped")

    def _eviction_monitor_loop(self):
        """Background loop: poll all engines for eviction status."""
        while not self._eviction_monitor_stop.is_set():
            if self._eviction_monitor_stop.wait(timeout=self._eviction_check_interval):
                break

            try:
                self._check_and_handle_evictions()
            except Exception:
                logger.exception("[Eviction] Unhandled error in eviction monitor loop")

    def _check_and_handle_evictions(self):
        """Poll live elastic actors independently for graceful SIGTERM
        intent."""
        pending = []
        with self._engine_lifecycle_lock:
            for server_name, srv in self.servers.items():
                for group in srv.engine_groups:
                    if (
                        not group.is_scaled_out
                        or group.pg is None
                        or group.lifecycle_status not in (EngineGroupLifecycle.ACTIVE, EngineGroupLifecycle.DRAINING)
                    ):
                        continue
                    for actor_idx, engine in enumerate(group.all_engines):
                        if engine is None:
                            continue
                        try:
                            pending.append(
                                (
                                    engine.is_evicted.remote(),
                                    server_name,
                                    group,
                                    actor_idx // group.nodes_per_engine,
                                )
                            )
                        except Exception as e:
                            logger.debug("[Eviction] Failed to submit graceful probe: %s", e)

        if not pending:
            return

        refs = [item[0] for item in pending]
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=5)
        ready_set = set(ready)
        handled = set()
        evicted_engine_infos = []
        for ref, server_name, group, node0_idx in pending:
            if ref not in ready_set or (id(group), node0_idx) in handled:
                continue
            try:
                if ray.get(ref):
                    handled.add((id(group), node0_idx))
                    evicted_engine_infos.append((server_name, group, node0_idx))
            except Exception as e:
                # Hard actor failure belongs to the fault-recovery follow-up,
                # not this graceful-eviction path.
                logger.debug("[Eviction] Graceful probe failed: %s", e)

        if evicted_engine_infos:
            self._handle_evictions(evicted_engine_infos)

    def _handle_evictions(self, eviction_infos: list[tuple[str, EngineGroup, int]]) -> None:
        """Fence a ready eviction batch before performing one live removal."""
        claimed = []
        with self._engine_lifecycle_lock:
            for srv_name, group, node0_idx in eviction_infos:
                srv = self.servers.get(srv_name)
                if (
                    srv is None
                    or group not in srv.engine_groups
                    or group.lifecycle_status not in (EngineGroupLifecycle.ACTIVE, EngineGroupLifecycle.DRAINING)
                    or not group.is_scaled_out
                    or group.pg is None
                    or node0_idx >= len(group.engines)
                    or group.engines[node0_idx] is None
                ):
                    continue
                group.eviction_requested = True
                group.lifecycle_status = EngineGroupLifecycle.DRAINING
                claimed.append((srv_name, srv, group, node0_idx))

            if not claimed:
                return

            # Signal-first fence: every ready intent is DRAINING before any
            # owner wait or cleanup.  An explicit scale-in remains the sole
            # removal owner and adopts matching DRAINING engines.
            active = self._find_active_scale_request()
            if active is not None and active["type"] == "scale_in":
                engine_ids = [f"group_{group.rank_offset}_engine_{idx}" for _, _, group, idx in claimed]
                logger.info(
                    "[Eviction] Scale-in request %s will adopt pending evictions %s (%s)",
                    active["request_id"],
                    engine_ids,
                    active["status"],
                )
                return
            if active is not None and active["type"] not in ("scale_out", "graceful_eviction"):
                return

        engine_ids = [f"group_{group.rank_offset}_engine_{idx}" for _, _, group, idx in claimed]
        logger.info("[Eviction] Gracefully removing engines %s", engine_ids)
        try:
            wait_timeout = getattr(self.args, "scale_in_drain_timeout", 30.0) + 60.0
            deadline = time.monotonic() + wait_timeout
            while self._is_weight_updating:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("[Eviction] Timed out waiting for the weight-update fence for %s", engine_ids)
                    return
                time.sleep(min(1.0, remaining))

            batches: dict[str, tuple[RolloutServer, list[tuple[EngineGroup, int]]]] = {}
            for srv_name, srv, group, node0_idx in claimed:
                if srv_name not in batches:
                    batches[srv_name] = (srv, [])
                batches[srv_name][1].append((group, node0_idx))

            async def _remove_batches():
                batch_items = list(batches.items())
                results = await asyncio.gather(
                    *[
                        self._remove_live_engines(
                            srv,
                            infos,
                            drain_timeout=getattr(self.args, "scale_in_drain_timeout", 30.0),
                            shutdown_timeout=getattr(self.args, "scale_in_shutdown_timeout", 20.0),
                            force=False,
                        )
                        for _, (srv, infos) in batch_items
                    ],
                    return_exceptions=True,
                )
                return [(srv_name, result) for (srv_name, _), result in zip(batch_items, results)]

            batch_results = asyncio.run(_remove_batches())
        except Exception:
            logger.exception("[Eviction] Graceful removal failed for %s", engine_ids)
            return

        removed_by_server = {}
        for srv_name, result in batch_results:
            if isinstance(result, BaseException):
                logger.warning("[Eviction] Graceful removal batch failed for %s: %s", srv_name, result)
                removed_by_server[srv_name] = set()
            else:
                removed, _ = result
                removed_by_server[srv_name] = set(removed)

        claimed_by_group: dict[int, tuple[EngineGroup, list[tuple[str, str]]]] = {}
        for srv_name, _, group, node0_idx in claimed:
            engine_id = f"group_{group.rank_offset}_engine_{node0_idx}"
            claimed_by_group.setdefault(id(group), (group, []))[1].append((srv_name, engine_id))

        with self._engine_lifecycle_lock:
            # Restore a surviving group only when every eviction claimed for
            # that group completed. Any timeout/failure remains fail-closed.
            for group, group_claims in claimed_by_group.values():
                completed = all(
                    engine_id in removed_by_server.get(srv_name, set()) for srv_name, engine_id in group_claims
                )
                if completed:
                    group.eviction_requested = False
                if completed and group.lifecycle_status is EngineGroupLifecycle.DRAINING:
                    group.lifecycle_status = EngineGroupLifecycle.ACTIVE


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different engine groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sgl router or slime router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values. When
    ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    # Determine the bind address (can be 0.0.0.0 / wildcard) and the connection
    # address (must be a reachable IP for engines).  When SLIME_HOST_IP is set
    # to a wildcard ("0.0.0.0" / "::") the bind is fine but the wildcard is not
    # a usable connection target, so fall back to the real local IP for the
    # cross-node connection.  For any other explicit value (including the
    # single-node default 127.0.0.1) honor it for both bind and connect so the
    # two stay consistent.
    real_local_ip = _wrap_ipv6(get_host_info()[1])
    env_overwrite_local_ip = Envs.SLIME_HOST_IP
    if env_overwrite_local_ip:
        bind_ip = _wrap_ipv6(env_overwrite_local_ip)
        is_wildcard = env_overwrite_local_ip.strip("[]") in ("0.0.0.0", "::")
        router_ip = real_local_ip if is_wildcard else bind_ip
    else:
        bind_ip = real_local_ip
        router_ip = real_local_ip

    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        assert not has_pd_disaggregation, "slime router does not support PD disaggregation."
        import copy

        from relax.engine.router.router import run_router

        router_args = copy.copy(args)
        router_args.sglang_router_ip = bind_ip
        router_args.sglang_router_port = router_port

    else:
        from sglang_router.launch_router import RouterArgs

        from relax.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = bind_ip
        router_args.port = router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if hasattr(args, "sglang_router_policy") and args.sglang_router_policy:
            router_args.policy = args.sglang_router_policy

        if has_pd_disaggregation:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    time.sleep(3)
    assert process.is_alive()
    _LAUNCHED_ROUTER_PROCESSES.append(process)
    logger.info(f"Router launched locally at {bind_ip}:{router_port} (connection address: {router_ip})")

    return router_ip, router_port


# Router processes launched via `_start_router`. Retained so
# `Controller._global_restart` can terminate them before re-initializing —
# otherwise the daemon router survives with stale pre-restart engine URLs and
# spams "Connection refused" health-check WARNs against dead ports forever.
_LAUNCHED_ROUTER_PROCESSES: list[multiprocessing.Process] = []


def stop_launched_routers() -> int:
    """Terminate every router process launched in this Controller lifetime.

    Returns the number of processes that were alive and successfully killed —
    callers use this to decide whether to reset cached router endpoint state.
    """
    global _LAUNCHED_ROUTER_PROCESSES
    killed = 0
    survivors: list[multiprocessing.Process] = []
    for proc in _LAUNCHED_ROUTER_PROCESSES:
        try:
            if not proc.is_alive():
                continue
            logger.info(f"Terminating router process pid={proc.pid}")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            if proc.is_alive():
                logger.warning(f"Router pid={proc.pid} still alive after kill; leaking")
                survivors.append(proc)
            else:
                killed += 1
        except Exception as e:
            logger.warning(f"Failed to terminate router pid={getattr(proc, 'pid', None)}: {e}")
    _LAUNCHED_ROUTER_PROCESSES = survivors
    return killed


def _wait_engine_init_with_progress(
    init_handles: list,
    model_name: str,
    timeout: float,
    log_interval: float,
) -> None:
    """Soft barrier across all engine init() handles with periodic progress
    logs.

    Acts as a single rendezvous point at training startup: blocks until every
    engine has finished server launch + weight loading, so that stragglers
    caused by storage/IO jitter do not leak into downstream NCCL collectives.
    Logs the remaining engine ranks every ``log_interval`` seconds so slow
    nodes are visible without grepping per-engine logs.
    """
    total = len(init_handles)
    pending = {h: rank for rank, h in enumerate(init_handles)}
    deadline = time.monotonic() + timeout
    next_log = time.monotonic() + log_interval

    logger.info(f"[engine-init-barrier:{model_name}] waiting for {total} engines (timeout={timeout:.0f}s)")

    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"[engine-init-barrier:{model_name}] timed out after {timeout:.0f}s; "
                f"{len(pending)}/{total} engines still initializing, "
                f"slow ranks={sorted(pending.values())}"
            )
        wait_slice = min(remaining, max(0.1, next_log - time.monotonic()))
        done, _ = ray.wait(list(pending.keys()), num_returns=len(pending), timeout=wait_slice)
        for h in done:
            try:
                ray.get(h)
            except Exception as e:
                slow = sorted(pending.values())
                raise RuntimeError(
                    f"[engine-init-barrier:{model_name}] engine rank={pending[h]} init failed: {e}; "
                    f"other ranks still pending={slow}"
                ) from e
            pending.pop(h)
        if time.monotonic() >= next_log and pending:
            ready = total - len(pending)
            slow = sorted(pending.values())
            preview = slow if len(slow) <= 10 else slow[:10] + ["..."]
            logger.info(f"[engine-init-barrier:{model_name}] ready {ready}/{total}, still-waiting ranks={preview}")
            next_log = time.monotonic() + log_interval

    logger.info(f"[engine-init-barrier:{model_name}] all {total} engines ready")


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of engine groups.  Engine groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        engine_groups: list[EngineGroup] = []
        all_init_handles: list = []
        port_cursors: dict[int, int] = {}

        for group_cfg in model_cfg.engine_groups:
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group = EngineGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=group_cfg.overrides,
                router_ip=router_ip,
                router_port=router_port,
            )
            handles, port_cursors = group.start_engines(port_cursors)
            all_init_handles.extend(handles)
            engine_groups.append(group)

            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus

        if all_init_handles:
            _wait_engine_init_with_progress(
                all_init_handles,
                model_name=model_cfg.name,
                timeout=getattr(args, "rollout_engine_init_timeout", 3600.0),
                log_interval=60.0,
            )

        servers[model_cfg.name] = RolloutServer(
            engine_groups=engine_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
        )

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if getattr(args, "prefill_num_servers", None) is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                engine_groups=[EngineGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    save_eval_summary_jsonl(args, rollout_id, data)

    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        if not rewards:
            # Every rollout for this eval dataset failed or was dropped (e.g. all
            # sandbox containers failed setup), so there are no rewards to average.
            # Surface it loudly but don't crash the whole run on a division by zero.
            logger.warning(
                f"Eval dataset {key!r} produced no valid samples (all rollouts failed or were "
                f"dropped); skipping eval metrics for this dataset."
            )
            continue
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(
                compute_metrics_from_samples(args, samples, include_rloo_diagnostics=False),
                f"eval/{key}/",
            )
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")
    tracking_utils.flush_metrics(args, step)

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    save_rollout_result_jsonl(args, rollout_id, samples)

    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    if samples:
        log_dict |= dict_add_prefix(
            compute_metrics_from_samples(args, samples, rollout_id=rollout_id),
            "rollout/",
        )
        log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    rollout_log_dict = {key: value for key, value in log_dict.items() if not key.startswith(("perf/", "perf_detail/"))}
    perf_log_dict = {key: value for key, value in log_dict.items() if key.startswith(("perf/", "perf_detail/"))}
    logger.info(f"rollout {rollout_id}: {rollout_log_dict}")
    logger.info(f"perf {rollout_id}: {perf_log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    tracking_utils.flush_metrics(args, step)


def compute_metrics_from_samples(
    args,
    samples,
    *,
    rollout_id: int | None = None,
    include_rloo_diagnostics: bool = True,
):
    rewarded_samples = [sample for sample in samples if sample.reward is not None]
    reward_cat_key = args.log_reward_category
    reward_category_samples = (
        [sample for sample in rewarded_samples if isinstance(sample.reward, dict) and reward_cat_key in sample.reward]
        if reward_cat_key is not None
        else rewarded_samples
    )
    response_lengths = [sample.effective_response_length for sample in samples]
    multimodal_stats = [get_sample_multimodal_stats(sample) for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_min_mean_max_stats([s["image_count"] for s in multimodal_stats], "image_count/")
    log_dict |= _compute_min_mean_max_stats(
        [s["multimodal_token_count"] for s in multimodal_stats], "multimodal_token_count/"
    )
    log_dict |= compute_rollout_reward_metrics(
        args,
        rewarded_samples,
        include_rloo_diagnostics=include_rloo_diagnostics,
    )
    log_dict |= _compute_zero_std_metrics(args, rewarded_samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, reward_category_samples)
    log_dict |= compute_mopd_metrics(args, rewarded_samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    log_dict["num_turn/mean"] = np.mean([s.metadata.get("rollout_turns", 1) for s in samples]).item()
    log_dict["num_turn/max"] = np.max([s.metadata.get("rollout_turns", 1) for s in samples]).item()
    log_dict["num_turn/min"] = np.min([s.metadata.get("rollout_turns", 1) for s in samples]).item()
    if rollout_id is not None and args.partial_rollout and not args.fully_async:
        staleness_gaps = [rollout_id - sample.metadata.get("start_rollout_id", rollout_id) for sample in samples]
        log_dict["staleness/avg"] = np.mean(staleness_gaps).item()
        log_dict["staleness/max"] = np.max(staleness_gaps).item()
        log_dict["staleness/min"] = np.min(staleness_gaps).item()
        log_dict["global_batch_size"] = len({sample.index for sample in samples})
    return log_dict


def _compute_min_mean_max_stats(values: list[int], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}mean": np.mean(values).item(),
        f"{prefix}max": np.max(values).item(),
        f"{prefix}min": np.min(values).item(),
    }


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
