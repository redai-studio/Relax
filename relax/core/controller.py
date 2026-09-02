# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import concurrent.futures
import copy
import os
import threading
import time
import traceback
from argparse import Namespace
from enum import Enum, IntEnum
from functools import partial
from typing import Any

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from ray import serve
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from transfer_queue import SeqlenBalancedSampler

from relax import utils as relax_utils
from relax.agentic.session.service import (
    deploy_agentic_chat_api_services,
    shutdown_agentic_chat_api_services,
)
from relax.core.node_group_affinity import require_control_plane_resource, with_control_plane_affinity
from relax.core.optional_roles import register_extra_roles
from relax.core.registry import ALGOS, ROLES, process_role
from relax.core.service import Service, create_placement_group
from relax.distributed.checkpoint_service.coordinator.service import create_dcs_deployment
from relax.distributed.coordination import PeerStepBarrier, RolloutOffloadBarrier
from relax.engine.sft.bootstrap import resolve_sft_algo_key, resolve_sft_num_rollout, validate_sft_resource
from relax.utils import device as device_utils
from relax.utils.async_utils import run, shutdown_async_loop
from relax.utils.data.identity_window_sampler import IdentityWindowSampler
from relax.utils.env import Envs
from relax.utils.health_system import HealthManager
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.opd.opd_utils import (
    maybe_start_managed_opd_teacher,
    set_managed_opd_teacher_on_actor_service,
    shutdown_managed_opd_teacher,
)
from relax.utils.s3_model_loader import (
    cleanup_s3_model_weights_from_shm,
    is_s3_uri,
    prepare_local_model,
    read_s3_model_config,
    remove_stale_s3_model_caches,
)
from relax.utils.training.ppo_utils import validate_ppo_config
from relax.utils.utils import compute_dp_size, recovery_load_path


def create_data_source_actor(config: Namespace, data_source_cls: Any) -> Any:
    actor_cls = ray.remote(num_cpus=1)(data_source_cls)
    return actor_cls.options(**with_control_plane_affinity(config)).remote(config)


def _needs_rollout_manager_setup(serve_dict: dict) -> bool:
    """Skip rollout_manager wiring in SFT-only mode (no rollout role)."""
    return ROLES.rollout in serve_dict


def _is_colocate(config: Namespace) -> bool:
    """True when actor/rollout (and optionally critic) share GPUs."""
    return not getattr(config, "fully_async", False) and not getattr(config, "hybrid", False)


logger = get_logger(__name__)

ACTOR_ROLLOUT_PG_ROLES = ["actor", "rollout", "genrm"]
# ``auto`` may resolve to a node-local full checkpoint on a single-node engine,
# while ``runai_streamer`` keeps the model source on raw S3.  Be conservative:
# cleanup is enabled only when the configured plan cannot retain SHM weights.
_S3_MODEL_CLEANUP_SAFE_LOAD_FORMATS = {"dummy", "runai_streamer"}


class ModelCompleteness(IntEnum):
    NONE = 0
    METADATA = 1
    FULL = 2


class ServiceStartupPhase(str, Enum):
    ALLOCATE_RESOURCES = "allocate resources"
    DEPLOY = "deploy"


def _uses_s3_model_prefetch(config: Namespace) -> bool:
    source = getattr(config, "model_source", None)
    return source is not None and is_s3_uri(source.uri)


def _build_train_start_config(config: Namespace) -> Namespace:
    start_config = copy.copy(config)
    if not _uses_s3_model_prefetch(config):
        return start_config
    try:
        start_config._model_source_config = read_s3_model_config(config)
    except Exception as exc:
        logger.warning(
            "Unable to read remote model config for train START telemetry "
            f"({type(exc).__name__}); continuing without it."
        )
    return start_config


def _require_positive_timeout(value: float, env_name: str) -> float:
    if value <= 0:
        raise ValueError(f"{env_name} must be greater than 0, got {value}")
    return value


def _cleanup_s3_model_weights_on_node(config: Namespace) -> tuple[int, int]:
    return cleanup_s3_model_weights_from_shm(config)


def _current_node_id() -> str:
    return ray.get_runtime_context().get_node_id()


def _remove_stale_s3_model_caches_on_node(config: Namespace) -> tuple[int, int]:
    return remove_stale_s3_model_caches(config)


def _prefetch_s3_model_on_node(config: Namespace, completeness: ModelCompleteness) -> str:
    return prepare_local_model(config, completeness=completeness.name.lower()).path


def _can_cleanup_s3_model_weights(config: Namespace, serve_dict: dict) -> bool:
    model_source = getattr(config, "model_source", None)
    if model_source is None:
        return False
    if getattr(config, "debug_rollout_only", False):
        return False
    has_rollout = any(str(role) == "rollout" for role in serve_dict)
    if not has_rollout:
        return True
    if getattr(config, "rollout_external", False):
        return False
    if getattr(config, "sglang_config", None) is not None:
        return False
    return getattr(config, "sglang_load_format", "auto") in _S3_MODEL_CLEANUP_SAFE_LOAD_FORMATS


def _actor_rollout_pg_roles(config: Namespace) -> list[str]:
    """Roles that share the actor/rollout placement group.

    PPO co-hosts the critic on the same GPUs only when its resource entry
    matches the actor's; otherwise critic runs on its own placement group.
    """
    roles = list(ACTOR_ROLLOUT_PG_ROLES)
    if getattr(config, "advantage_estimator", None) != "ppo":
        return roles
    resource = getattr(config, "resource", None) or {}
    if resource.get("critic") == resource.get("actor"):
        roles.append("critic")
    return roles


class Controller:
    def __init__(self, config: Namespace, runtime_env: dict = None) -> None:
        self.config = config
        self.serve_dict = {}
        self._teacher_manager = None
        # Initialize health management system
        self.runtime_env = runtime_env
        self._health_check_enabled = getattr(config, "use_health_check", False)
        self._max_global_restart = getattr(config, "max_global_restart", 3)
        require_control_plane_resource(config)
        self._health_manager = HealthManager(check_interval=1.0, config=config)
        self._restarting = False  # Flag to indicate a restart is in progress
        self._restart_done_event = threading.Event()  # Signals main thread that global restart Phase 1+2 is done
        self._restart_error = None  # Stores any error from the global restart thread
        # Preserve across __init__ calls during global restart (same pattern as _global_restart_count)
        if not hasattr(self, "_pending_task_refs"):
            self._pending_task_refs: list = []
            self._pending_task_refs_lock = threading.Lock()
        if not hasattr(self, "_global_restart_count"):
            self._global_restart_count = 0
        self._model_cache_node_bindings: dict[str, tuple[Any, int]] = {}
        self._model_cache_node_ids: set[str] = set()

        # SFT: fill in num_rollout / num_rollout_per_epoch before any actor
        # is launched (RL is resolved later in placement_group.py).
        resolve_sft_num_rollout(self.config)

        # Initialize data management system
        self._initialize_data_system()
        self.dcs, self.config.coordinator_url = create_dcs_deployment(
            ray_actor_options=with_control_plane_affinity(config, {"num_cpus": 1})
        )

        self._metrics_service_enabled = getattr(config, "use_metrics_service", False)
        if self._metrics_service_enabled:
            self._deploy_metrics_service()

        if self.config.use_agentic_rollout and not self.config.debug_train_only:
            deploy_agentic_chat_api_services(
                config=self.config,
                runtime_env=self.runtime_env,
            )
        self._autoscaler_config = None
        try:
            self.register_all_serve()
        except Exception as e:
            self._report_error_to_metrics_service(e)
            raise

        autoscaler_config_path = getattr(config, "autoscaler_config", None)
        if autoscaler_config_path:
            from relax.utils.autoscaler.config import AutoscalerConfig
            from relax.utils.utils import get_serve_url

            rollout_service_url = get_serve_url("/rollout")
            self._autoscaler_config = AutoscalerConfig.from_yaml(autoscaler_config_path, rollout_service_url)
            self._deploy_autoscaler_service()

        # Start health management with service restart callback
        if self._health_check_enabled:
            self._health_manager.start(
                on_unhealthy=self._on_service_unhealthy,
                on_fatal=self._on_service_fatal,
            )
            logger.info("Global health check system enabled")
        else:
            logger.info("Global health check system disabled (use --use-health-check to enable)")

    def _cleanup_s3_model_weights_after_init(self, *, force: bool = False) -> None:
        """Remove policy weight shards after every startup consumer is
        ready."""
        if getattr(self.config, "disable_s3_model_cleanup", False):
            logger.info("S3 model SHM cleanup is disabled")
            return
        if not _uses_s3_model_prefetch(self.config):
            return
        if not force and not _can_cleanup_s3_model_weights(self.config, self.serve_dict):
            logger.info("Keeping S3 model weights in SHM because the rollout load plan may retain SHM-backed weights")
            return

        cleanup_task = ray.remote(num_cpus=0, max_retries=0)(_cleanup_s3_model_weights_on_node)
        refs = []
        alive_node_ids = {node["NodeID"] for node in ray.nodes() if node.get("Alive") and node.get("NodeID")}
        target_node_ids = getattr(self, "_model_cache_node_ids", set()) or alive_node_ids
        for node_id in sorted(target_node_ids & alive_node_ids):
            refs.append(
                cleanup_task.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                ).remote(self.config)
            )
        if not refs:
            return

        task_timeout = _require_positive_timeout(
            Envs.RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S,
            "RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S",
        )
        terminal_refs, pending_refs = ray.wait(
            refs,
            num_returns=len(refs),
            timeout=task_timeout,
        )
        if pending_refs:
            for ref in pending_refs:
                ray.cancel(ref, force=True)
            cancel_timeout = _require_positive_timeout(
                Envs.RELAX_S3_MODEL_CLEANUP_CANCEL_TIMEOUT_S,
                "RELAX_S3_MODEL_CLEANUP_CANCEL_TIMEOUT_S",
            )
            _, uncancelled_refs = ray.wait(
                pending_refs,
                num_returns=len(pending_refs),
                timeout=cancel_timeout,
            )
            if uncancelled_refs:
                raise TimeoutError(
                    "S3 model SHM cleanup timed out and cancellation did not reach "
                    f"{len(uncancelled_refs)} of {len(pending_refs)} pending node tasks"
                )
            raise TimeoutError(
                f"S3 model SHM cleanup timed out with {len(pending_refs)} of {len(refs)} node tasks pending"
            )
        results = ray.get(terminal_refs)
        removed_files = sum(result[0] for result in results)
        removed_bytes = sum(result[1] for result in results)
        logger.info(
            f"S3 model SHM cleanup completed on {len(results)} nodes: "
            f"removed_files={removed_files}, removed_bytes={removed_bytes}"
        )

    def _initialize_data_system(self):
        algo_key = resolve_sft_algo_key(self.config)
        dp_size = compute_dp_size(self.config)
        use_sft_prepack = algo_key == "sft" and getattr(self.config, "sft_async_prepack", False)
        if (
            getattr(self.config, "fully_async", False)
            and getattr(self.config, "use_dynamic_batch_size", False)
            and not use_sft_prepack
        ):
            sampler = IdentityWindowSampler(
                dp_size=dp_size,
                placement="streaming",
                balance_unit_multiplier=self.config.n_samples_per_prompt,
            )
            logger.info("Using IdentityWindowSampler (fully_async + dynamic batch)")
        elif algo_key == "sft":
            sampler = SeqlenBalancedSampler(
                n_samples_per_prompt=self.config.n_samples_per_prompt,
                dp_size=dp_size,
            )
            logger.info(f"Using SeqlenBalancedSampler with dp_size={dp_size}")
        elif getattr(self.config, "balance_data", False):
            sampler = IdentityWindowSampler(dp_size=dp_size, placement="balanced")
            logger.info(f"Using IdentityWindowSampler with balanced row placement, dp_size={dp_size}")
        else:
            sampler = IdentityWindowSampler(dp_size=dp_size, placement="sequential")
            logger.info(f"Using IdentityWindowSampler with sequential row placement, dp_size={dp_size}")

        tq_config = OmegaConf.create(
            {
                "controller": {
                    "sampler": sampler,
                    "polling_mode": self.config.polling_mode,
                },
                "backend": {
                    "SimpleStorage": {
                        "num_data_storage_units": self.config.num_data_storage_units,
                    },
                },
            },
            flags={"allow_objects": True},
        )
        tq_config = tq.init(conf=tq_config) or tq_config
        self.config.tq_config = tq_config

    def _deploy_metrics_service(self):
        """Deploy the MetricsService as a lightweight Ray Serve deployment.

        The metrics service runs on CPU only, collecting and forwarding metrics
        to configured backends (TensorBoard, W&B, ClearML). It must be deployed
        before other services so they can connect to it during initialization.
        """
        from relax.utils.metrics.service import MetricsService

        deployment = MetricsService.options(
            ray_actor_options=with_control_plane_affinity(self.config, {"num_cpus": 1})
        ).bind(
            healthy=self._health_manager.status,
            pg=None,
            config=self.config,
            role="metrics",
        )
        serve.run(deployment, name="metrics", route_prefix="/metrics")
        logger.info("MetricsService deployed at /metrics")

    def _deploy_autoscaler_service(self):
        """Deploy the AutoscalerService as a lightweight Ray Serve deployment.

        The autoscaler service runs on CPU only, monitoring engine metrics and
        triggering scale-out/scale-in operations through the Rollout service
        API. It must be deployed after Rollout service is available.
        """
        from relax.utils.autoscaler.autoscaler_service import AutoscalerService

        deployment = AutoscalerService.options(
            ray_actor_options=with_control_plane_affinity(self.config, {"num_cpus": 1})
        ).bind(
            healthy=self._health_manager.status,
            pg=None,
            autoscaler_config=self._autoscaler_config,
            role="autoscaler",
        )
        handle = serve.run(deployment, name="autoscaler", route_prefix="/autoscaler")
        logger.info("AutoscalerService deployed at /autoscaler")

        try:
            handle.start.remote()
            logger.info("AutoscalerService started successfully")
        except Exception as e:
            logger.exception(f"Failed to start AutoscalerService: {e}")
            self._report_error_to_metrics_service(e)
            raise

    def _cancel_pending_tasks(self) -> None:
        """Cancel all pending service ObjectRefs to unblock the main thread.

        Must be called BEFORE ray.shutdown() during global restart. Without
        this, the main thread remains blocked awaiting ObjectRefs that become
        dangling after ray.shutdown(), causing a fatal C++ crash:
        ``TryReadObjectRefStream API can be used only when the stream has been
        created and not removed.``
        """
        with self._pending_task_refs_lock:
            refs_to_cancel = list(self._pending_task_refs)
            self._pending_task_refs.clear()

        if not refs_to_cancel:
            return

        logger.info(f"[Global Restart] Cancelling {len(refs_to_cancel)} pending task ref(s)...")
        for ref in refs_to_cancel:
            try:
                ray.cancel(ref, force=True)
            except Exception as e:
                logger.debug(f"[Global Restart] Failed to cancel task ref (may already be done): {e}")
        logger.info("[Global Restart] All pending task refs cancelled")

    def _on_service_unhealthy(self, role: str) -> None:
        """Callback when a service becomes unhealthy. Initiates service
        restart.

        Args:
            role: Service role name that became unhealthy.
        """
        logger.warning(f"Service '{role}' detected as unhealthy, initiating restart...")
        self.restart_serve(role)

    def _on_service_fatal(self, role: str, error_msg: str) -> None:
        """Callback when a service reports a fatal (non-recoverable) error.

        Runs immediately before the HealthChecker calls ``os._exit(1)`` — used
        to push the error to the metrics service for Apprise so the operator
        gets a notification before the process dies.
        """
        logger.error(f"Fatal error from service '{role}': {error_msg}")
        try:
            self._report_error_to_metrics_service(RuntimeError(f"{role}: {error_msg}"))
        except Exception as e:
            logger.warning(f"Failed to report fatal error to metrics service: {e}")

    def _create_service_task(
        self,
        role,
        cls,
        num_gpus,
        data_source,
        actor_rollout_pgs,
        actor_rollout_pg_roles=None,
        defer_deploy=False,
    ):
        """Create a single service.

        Returns (role, service, error).
        """
        if actor_rollout_pg_roles is None:
            actor_rollout_pg_roles = ACTOR_ROLLOUT_PG_ROLES
        try:
            service = Service(
                cls,
                role=role,
                healthy=self._health_manager.status,
                config=self.config,
                num_gpus=num_gpus,
                data_source=data_source,
                actor_rollout_pgs=actor_rollout_pgs if actor_rollout_pgs and role in actor_rollout_pg_roles else None,
                defer_deploy=defer_deploy,
                runtime_env=self.runtime_env,
            )
            logger.info(f"Service {role} has been created successfully")
            return (role, service, None)
        except Exception as e:
            logger.exception(f"Failed to create service {role}: {e}")
            return (role, None, str(e))

    @staticmethod
    def _deploy_service_task(role, service):
        try:
            service.deploy()
            return (role, service, None)
        except Exception as e:
            logger.exception(f"Failed to deploy service {role}: {e}")
            return (role, None, str(e))

    def _run_service_phase(
        self,
        phase: ServiceStartupPhase,
        task: Any,
        task_args: list[tuple],
    ) -> dict[Any, Service]:
        """Run one prepare/deploy phase with the existing startup
        concurrency."""
        if not task_args:
            return {}
        if self.config.fully_async:
            logger.info(f"Using parallel service {phase.value} mode with {len(task_args)} services")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_args)) as executor:
                futures = [executor.submit(task, *args) for args in task_args]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            logger.info(f"Using serial service {phase.value} mode (fully_async=False)")
            results = [task(*args) for args in task_args]

        failed_roles = [(role, error) for role, _service, error in results if error is not None]
        if failed_roles:
            error_msg = "; ".join(f"{role}: {error}" for role, error in failed_roles)
            raise RuntimeError(f"Failed to {phase.value} {len(failed_roles)} services: {error_msg}")
        return {role: service for role, service, _error in results}

    def _start_services(self, service_args: list[tuple]) -> list[Any]:
        """Start services, staging deployment only when S3 prefetch needs
        placement groups."""
        if not _uses_s3_model_prefetch(self.config):
            services = self._run_service_phase(
                ServiceStartupPhase.DEPLOY,
                self._create_service_task,
                service_args,
            )
            self.serve_dict.update(services)
            return []

        prepared_services = self._run_service_phase(
            ServiceStartupPhase.ALLOCATE_RESOURCES,
            partial(self._create_service_task, defer_deploy=True),
            service_args,
        )
        service_pgs = {role: service.pgs for role, service in prepared_services.items()}
        model_prefetch_refs = self._start_model_prefetch(service_pgs)
        services = self._run_service_phase(
            ServiceStartupPhase.DEPLOY,
            self._deploy_service_task,
            list(prepared_services.items()),
        )
        self.serve_dict.update(services)
        return model_prefetch_refs

    @staticmethod
    def _policy_model_completeness(
        config: Namespace,
        role: Any,
        *,
        shares_actor_pg: bool,
    ) -> ModelCompleteness:
        role_name = str(role)
        if role_name in {"actor", "critic", "reference", "actor_fwd"}:
            return ModelCompleteness.FULL
        if role_name != "rollout" or getattr(config, "rollout_external", False):
            return ModelCompleteness.NONE
        if getattr(config, "sglang_config", None) is not None:
            return ModelCompleteness.FULL
        load_format = getattr(config, "sglang_load_format", "auto")
        if load_format == "dummy":
            return ModelCompleteness.METADATA
        if load_format == "runai_streamer":
            return ModelCompleteness.NONE
        if load_format == "auto":
            return ModelCompleteness.FULL if shares_actor_pg else ModelCompleteness.NONE
        return ModelCompleteness.FULL

    def _start_model_prefetch(self, service_pgs: dict[Any, Any]) -> list[Any]:
        cleanup_task = ray.remote(num_cpus=0, max_retries=0)(_remove_stale_s3_model_caches_on_node)
        accelerator = device_utils.get_ray_accelerator_name()
        cleanup_node_ids = sorted(
            node["NodeID"]
            for node in ray.nodes()
            if node.get("Alive") and node.get("NodeID") and float(node.get("Resources", {}).get(accelerator, 0)) > 0
        )
        cleanup_refs = [
            cleanup_task.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(self.config)
            for node_id in cleanup_node_ids
        ]
        if cleanup_refs:
            ray.get(cleanup_refs)

        source = getattr(self.config, "model_source", None)
        if source is None or not is_s3_uri(source.uri):
            return []

        probe = ray.remote(num_cpus=0, max_retries=0)(_current_node_id)
        role_nodes: dict[Any, set[str]] = {}
        seen_pgs: dict[int, set[str]] = {}
        for role, pgs in service_pgs.items():
            if pgs is None:
                role_nodes[role] = set()
                continue
            pg, bundle_indices, _gpu_ids = pgs
            if id(pg) not in seen_pgs:
                refs = [
                    probe.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=bundle_index,
                        )
                    ).remote()
                    for bundle_index in bundle_indices
                ]
                resolved_node_ids = ray.get(refs)
                node_ids = set(resolved_node_ids)
                seen_pgs[id(pg)] = node_ids
                for node_id, bundle_index in zip(resolved_node_ids, bundle_indices, strict=True):
                    self._model_cache_node_bindings.setdefault(node_id, (pg, bundle_index))
            role_nodes[role] = seen_pgs[id(pg)]

        actor_pgs = next((pgs for role, pgs in service_pgs.items() if str(role) == "actor"), None)
        node_completeness: dict[str, ModelCompleteness] = {}
        for role, node_ids in role_nodes.items():
            role_pgs = service_pgs[role]
            shares_actor_pg = actor_pgs is not None and role_pgs is not None and role_pgs[0] is actor_pgs[0]
            completeness = self._policy_model_completeness(
                self.config,
                role,
                shares_actor_pg=shares_actor_pg,
            )
            if completeness == ModelCompleteness.NONE:
                continue
            for node_id in node_ids:
                node_completeness[node_id] = max(
                    completeness,
                    node_completeness.get(node_id, ModelCompleteness.NONE),
                )

        self._model_cache_node_ids = set(node_completeness)

        prefetch_task = ray.remote(num_cpus=0, max_retries=0)(_prefetch_s3_model_on_node)
        return [
            prefetch_task.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._model_cache_node_bindings[node_id][0],
                    placement_group_bundle_index=self._model_cache_node_bindings[node_id][1],
                )
            ).remote(self.config, completeness)
            for node_id, completeness in node_completeness.items()
        ]

    def _validate_gpu_resources(self, roles_to_create, colocate, actor_rollout_pg_roles):
        """Validate that the cluster has enough GPUs before creating placement
        groups.

        In colocate mode, roles in actor_rollout_pg_roles share one placement group,
        so only the max GPU count among them is needed. In non-colocate (fully_async)
        mode, each role gets its own placement group and GPUs are summed.

        Raises:
            RuntimeError: If the cluster does not have enough GPUs.
        """
        if colocate:
            shared_gpu = 0
            independent_gpu = 0
            for role, _cls, num_gpus, _ds in roles_to_create:
                if role in actor_rollout_pg_roles:
                    shared_gpu = max(shared_gpu, num_gpus)
                else:
                    independent_gpu += num_gpus
            total_required = shared_gpu + independent_gpu
        else:
            total_required = sum(num_gpus for _, _, num_gpus, _ in roles_to_create)

        cluster_resources = ray.cluster_resources()
        accel_resource = device_utils.get_ray_accelerator_name()
        total_available = int(cluster_resources.get(accel_resource, 0))

        logger.info(
            f"Resource validation: required GPUs={total_required}, cluster GPUs={total_available}, colocate={colocate}"
        )

        if total_required > total_available:
            role_details = ", ".join(f"{role}={num_gpus}" for role, _, num_gpus, _ in roles_to_create)
            raise RuntimeError(
                f"Insufficient GPU resources: {total_required} GPUs required but only "
                f"{total_available} available in the cluster. "
                f"Role breakdown: [{role_details}]. "
                f"Either add more GPU nodes, reduce GPU allocation per role, "
                f"or switch to colocate mode to share GPUs between actor and rollout."
            )

    def _maybe_resolve_num_rollout(self, roles_to_create):
        # Resolve --num-rollout from --num-epoch here, before any service is
        # deployed: services pickle self.config, so resolving later (inside the
        # Rollout actor) leaves Actor/Critic with num_rollout=None and breaks
        # args.train_iters. SFT is already pre-resolved in __init__, so this only
        # fills the RL "--num-epoch without --num-rollout" case.
        if self.config.num_rollout is not None:
            return
        if getattr(self.config, "num_epoch", None) is None:
            return
        rollout_data_source = None
        for role, _cls, _num_gpus, ds in roles_to_create:
            if str(role) == "rollout" and ds is not None:
                rollout_data_source = ds
                break
        if rollout_data_source is None:
            return
        num_rollout_per_epoch = ray.get(rollout_data_source.lengths.remote()) // self.config.rollout_batch_size
        self.config.num_rollout = num_rollout_per_epoch * self.config.num_epoch
        assert self.config.num_rollout > 0, (
            f"Resolved num_rollout={self.config.num_rollout} from num_epoch={self.config.num_epoch} and "
            f"num_rollout_per_epoch={num_rollout_per_epoch}; check dataset size vs rollout_batch_size."
        )
        logger.info(
            f"Resolved --num-rollout={self.config.num_rollout} from --num-epoch={self.config.num_epoch} "
            f"(num_rollout_per_epoch={num_rollout_per_epoch})"
        )

    def register_all_serve(self):
        validate_ppo_config(self.config)

        actor_rollout_pgs, self._teacher_manager = maybe_start_managed_opd_teacher(
            self.config,
            runtime_env=self.runtime_env,
        )

        algo_key = resolve_sft_algo_key(self.config)
        if algo_key not in ALGOS:
            raise ValueError(f"Algorithm key '{algo_key}' not registered in ALGOS. Available: {list(ALGOS.keys())}")
        algo: dict = ALGOS.get(algo_key).copy()
        ROLES = process_role(self.config)

        # Fail-fast on missing SFT producer role; without this the train
        # workers silently block on TransferQueue forever.
        validate_sft_resource(self.config)
        # Register optional services (e.g., GenRM, SFT-side rollout)
        extra_roles = register_extra_roles(self.config, algo)

        roles_iter = list(ROLES) + extra_roles
        role_names = {str(r) for r in roles_iter}
        # SFT path returns ROLES_SFT_ONLY (no `rollout` attr); rollout may be
        # injected via extra_roles — gate on the unified name set, not the enum.
        colocate = self.config.colocate and "rollout" in role_names and "actor" in role_names

        roles_to_create = []
        for role in roles_iter:
            cls = algo.get(role)
            if cls is None:
                logger.warning(f"No class registered for role '{role}', skipping")
                continue
            if str(role) == "rollout":
                data_source_cls = load_function(self.config.data_source_path)
                data_source = create_data_source_actor(self.config, data_source_cls)
            else:
                data_source = None
            # Optional roles (e.g. reference) may be absent from resource config
            if role not in self.config.resource:
                logger.warning(
                    f"Role '{role}' not in resource config (available: {list(self.config.resource.keys())}), skipping"
                )
                continue
            num_serves, num_gpus = self.config.resource.get(role)
            assert num_serves == 1, f"Currently only support num_serves=1 for {role}, but received {num_serves=}"
            self._health_manager.mark_healthy(role)
            logger.info(f"Service {role} start creating.")

            roles_to_create.append((role, cls, num_gpus, data_source))

        self._maybe_resolve_num_rollout(roles_to_create)
        relax_utils.report_train_start(_build_train_start_config(self.config))

        actor_rollout_pg_roles = _actor_rollout_pg_roles(self.config)
        self._validate_gpu_resources(roles_to_create, colocate, actor_rollout_pg_roles)

        if colocate and not self.config.hybrid:
            # Sync colocate: actor and rollout share GPUs via time-sharing (offload/onload)
            if actor_rollout_pgs is None:
                num_gpus = self.config.resource.get(ROLES.actor)[1]
                actor_rollout_pgs = create_placement_group(
                    num_gpus=num_gpus,
                    node_group_affinity=self.config.enable_affinity,
                )
        else:
            # fully_async (pure or hybrid): actor and rollout use separate GPUs
            actor_rollout_pgs = None

        service_args = [
            (
                role,
                cls,
                num_gpus,
                data_source,
                actor_rollout_pgs,
                actor_rollout_pg_roles,
            )
            for role, cls, num_gpus, data_source in roles_to_create
        ]
        # S3 prefetch needs the final placement groups before Ray Serve starts
        # the consumers. Other model sources preserve constructor-immediate
        # deployment inside Service.
        model_prefetch_refs = self._start_services(service_args)

        if model_prefetch_refs:
            ray.get(model_prefetch_refs)
            logger.info(f"S3 model prefetch completed on {len(model_prefetch_refs)} consumer nodes")

        logger.info(f"All {len(self.serve_dict)} services registered successfully: {list(self.serve_dict.keys())}")

    def _report_error_to_metrics_service(self, error: Exception):
        """Report error to metrics service for Apprise notification.

        Args:
            error: The exception that occurred
        """
        if not self._metrics_service_enabled:
            return

        try:
            import requests

            # Get error message and traceback
            error_message = str(error)
            error_traceback = traceback.format_exc()

            # Send to metrics service
            response = requests.post(
                "http://localhost:8000/metrics/log_error",
                json={"error_message": error_message, "error_traceback": error_traceback},
                timeout=5.0,
            )

            if response.status_code == 200:
                logger.info("Error reported to metrics service successfully")
            else:
                logger.warning(f"Failed to report error to metrics service: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to report error to metrics service: {e}")

    def _shutdown_agentic_rollout_services(self) -> None:
        if not self.config.use_agentic_rollout:
            return
        shutdown_agentic_chat_api_services()

    def training_loop(self):
        # Start all services in parallel without blocking on their completion
        # Each service runs independently: rollout, actor, critic, etc.
        async def run_all_services():
            if not (self.config.debug_train_only or self.config.debug_rollout_only):
                # Pass genRM manager to actor for coordinated offload/onload
                if "genrm" in self.serve_dict and not self.config.fully_async:
                    genrm_manager = await self.serve_dict["genrm"].get_genrm_manager()
                    await self.serve_dict[ROLES.actor].set_genrm_manager(genrm_manager)

                await set_managed_opd_teacher_on_actor_service(
                    self.serve_dict.get(ROLES.actor),
                    self._teacher_manager,
                    self.config,
                )

                # Always set rollout_manager for both sync and async modes
                # (needed for scaled-out engine weight sync in fully_async mode)
                if _needs_rollout_manager_setup(self.serve_dict) and ROLES.actor in self.serve_dict:
                    rollout_manager = await self.serve_dict[ROLES.rollout].get_rollout_manager()
                    await self.serve_dict[ROLES.actor].set_rollout_manager(rollout_manager)

                    # Colocate wiring topology:
                    #   - actor.rollout_barrier: always, so wake_up doesn't
                    #     collide with SGLang's static KV pool.
                    #   - critic.rollout_barrier / actor.peer_barrier /
                    #     rollout.peer_barrier: only when critic is co-hosted
                    #     (PPO colocate). GRPO has no third GPU claimant.
                    # In fully_async / hybrid nothing is wired and every
                    # barrier-guarded site short-circuits.
                    if _is_colocate(self.config):
                        rollout_barrier = RolloutOffloadBarrier(rollout_manager, logger=logger)
                        actor_set_kwargs: dict[str, Any] = {"rollout": rollout_barrier}
                        if ROLES.critic in self.serve_dict:
                            critic_handle = self.serve_dict[ROLES.critic].handle
                            actor_handle = self.serve_dict[ROLES.actor].handle
                            await self.serve_dict[ROLES.critic].set_barriers(rollout=rollout_barrier)
                            actor_set_kwargs["peers"] = PeerStepBarrier(
                                {ROLES.critic.value: critic_handle}, logger=logger
                            )
                            await self.serve_dict[ROLES.rollout].set_barriers(
                                peers=PeerStepBarrier(
                                    {
                                        ROLES.actor.value: actor_handle,
                                        ROLES.critic.value: critic_handle,
                                    },
                                    logger=logger,
                                ),
                            )
                        await self.serve_dict[ROLES.actor].set_barriers(**actor_set_kwargs)

                if self.config.fully_async and not self.config.hybrid and ROLES.actor in self.serve_dict:
                    # Pure fully_async: actor sends weights to separate actor_fwd/reference services
                    # via checkpoint engine. Hybrid mode skips this because the actor handles
                    # ref/actor_fwd internally via _switch_model.
                    handles = [self.serve_dict[ROLES.actor].update_weights_fully_async()]
                    if ROLES.actor_fwd in self.serve_dict:
                        handles.append(self.serve_dict[ROLES.actor_fwd].recv_weight_fully_async())
                    if ROLES.reference in self.serve_dict:
                        handles.append(self.serve_dict[ROLES.reference].recv_weight_fully_async())
                    [await handle for handle in handles]
                if ROLES.actor in self.serve_dict:
                    step = await self.serve_dict[ROLES.actor].get_step()
                    for service in self.serve_dict.values():
                        await service.set_step(step)

            # All startup consumers have now initialized and the first policy
            # weights have been synchronized. Remove source weight shards
            # before any service starts its training/rollout loop.
            self._cleanup_s3_model_weights_after_init()

            task_refs = []
            service_names = []
            for role, service in self.serve_dict.items():
                task_ref = service.run()
                if task_ref is not None:
                    task_refs.append(task_ref)
                    service_names.append(service.role)

            with self._pending_task_refs_lock:
                self._pending_task_refs = list(task_refs)

            if task_refs:
                logger.info(f"Started {len(task_refs)} services in parallel: {service_names}")

                try:
                    [await task_ref for task_ref in task_refs]
                    logger.info("Service task completed successfully")
                except Exception as e:
                    raise RuntimeError(f"Service task failed: {e}")
                finally:
                    with self._pending_task_refs_lock:
                        self._pending_task_refs.clear()

        while True:
            try:
                run(run_all_services())
                logger.info("All services running successfully")
                return  # Normal completion, exit
            except Exception as e:
                if self._restarting:
                    # A restart is in progress (triggered by HealthChecker callback thread).
                    # The service task failure is expected because restart kills the old replica.
                    # Block here until the global restart completes (teardown + re-init),
                    # then loop back to re-run run_all_services() with the freshly
                    # initialized Controller state.
                    logger.warning(
                        f"Training loop interrupted by ongoing restart, waiting for restart to complete: {e}"
                    )
                    self._restart_done_event.wait()  # Block until _global_restart signals done
                    self._restart_done_event.clear()  # Reset for next restart cycle

                    if self._restart_error is not None:
                        # Global restart itself failed — nothing more we can do
                        logger.exception(f"Global restart failed, cannot recover: {self._restart_error}")
                        self._report_error_to_metrics_service(self._restart_error)
                        raise RuntimeError(f"Global restart failed: {self._restart_error}") from self._restart_error

                    # Global restart succeeded — self.__init__() has been called,
                    # all services are re-registered. Loop back to re-run
                    # run_all_services() with fresh state.
                    self._restarting = False
                    logger.info("Global restart completed, re-running training loop")
                    continue
                logger.exception(f"Training loop failed: {e}")
                # Report error to metrics service for Apprise notification
                self._report_error_to_metrics_service(e)
                raise

    def shutdown(self) -> None:
        """Gracefully shut down all services, cleaning up SGLang engine
        processes.

        Must be called before ``serve.shutdown()`` / ``ray.shutdown()`` to
        ensure SGLang child processes (scheduler, detokenizer) are terminated
        instead of being orphaned.
        """
        logger.info("Controller shutting down — cleaning up engine processes...")
        self.stop_health_check()

        # Shut down rollout engines via RolloutManager.dispose()
        if ROLES.rollout in self.serve_dict:
            try:
                rollout_manager = run(self.serve_dict[ROLES.rollout].get_rollout_manager())
                ray.get(rollout_manager.dispose.remote(), timeout=30)
                logger.info("RolloutManager disposed — SGLang engines shut down.")
            except Exception as e:
                logger.warning(f"Failed to dispose RolloutManager: {e}")

        shutdown_managed_opd_teacher(self._teacher_manager)

        self._shutdown_agentic_rollout_services()

        try:
            self._cleanup_s3_model_weights_after_init(force=True)
        except Exception as e:
            logger.warning(f"Failed to clean S3 model SHM cache during shutdown: {e}")

        logger.info("Controller shutdown complete.")

    def add_serve(self, role: str) -> None:
        """Placeholder for future dynamic service addition."""
        raise NotImplementedError("Dynamic service addition not yet implemented")

    def del_serve(self, role: str) -> None:
        """Placeholder for future dynamic service removal."""
        raise NotImplementedError("Dynamic service removal not yet implemented")

    def stop_health_check(self, timeout: float = 2.0) -> None:
        """Stop the health management system.

        Args:
            timeout: Seconds to wait for health checker to stop.
        """
        self._health_manager.stop(timeout)

    def restart_serve(self, role: str) -> None:
        """Restart a service after it becomes unhealthy.

        Global restart (full Controller re-initialization from zero) is triggered when:
        - role is "actor" (actor is the core training service), OR
        - restart_count >= 3 for any role (too many restarts, need a clean slate)

        For other cases, delegates to Service.restart() for in-place restart.

        Args:
            role: Service role name to restart.
        """
        logger.info(f"Restarting service '{role}'...")
        serve.delete(role)
        # Must mirror register_all_serve's algo-key resolution. SFT mode is
        # identified by ``loss_type == "sft"``, not by ``advantage_estimator``
        # (Megatron's parser doesn't accept "sft" as an --advantage-estimator
        # choice), so looking the algo up under ``advantage_estimator`` here
        # silently misses the {sft, actor} dict, hits the cls-is-None skip
        # below, leaves the unhealthy flag set, and the health checker spins
        # restart attempts forever on a service that was already deleted.
        algo_key = resolve_sft_algo_key(self.config)
        if algo_key not in ALGOS:
            raise ValueError(f"Algorithm key '{algo_key}' not registered in ALGOS. Available: {list(ALGOS.keys())}")
        algo: dict = ALGOS.get(algo_key).copy()
        # Include dynamically added optional roles.
        register_extra_roles(self.config, algo)
        cls = algo.get(role)
        if cls is None:
            logger.warning(f"No class registered for role '{role}', skipping")
            return

        self._restarting = True
        restart_count = self._health_manager.increment_restart_count(role)
        logger.info(f"Restarting {role}, restart count: {restart_count}")

        # TODO(yuzhe) remove rollout and actor_fwd from global_restart.
        if role in [ROLES.actor, ROLES.rollout, ROLES.actor_fwd] or restart_count >= 3:
            # Perform full Controller re-initialization from zero when:
            # 1. Actor fails (core training service, all other services depend on it)
            # 2. Any service has been restarted >= 3 times (system is unstable)
            reason = "actor failure" if role == ROLES.actor else f"restart_count({restart_count}) >= 3 for '{role}'"
            logger.warning(f"Triggering global restart due to: {reason}")
            self._global_restart()
            # _restarting is reset by the main thread after it processes the restart_done_event
        else:
            # Delegate in-place restart to Service (reuses PG, restores step, syncs weights, re-runs task)
            service = self.serve_dict[role]
            service.restart()

            self._restarting = False
            self._health_manager.mark_healthy(role)
            logger.info(f"Service '{role}' restarted successfully")

    def _global_restart(self) -> None:
        """Perform a full global restart by re-initializing the Controller from
        zero.

        This completely tears down ALL existing resources and re-executes the
        full Controller.__init__ initialization sequence. After completion,
        it signals the main thread (blocked in training_loop) to re-run the
        training loop with the freshly initialized state.

        Steps:
        Phase 1 — Teardown:
          1. Stop health management to prevent further callbacks
          2. Cancel pending ObjectRefs to unblock the main thread
          3. Tear down all existing Ray Serve deployments (services + metrics + DCS)
          4. Tear down data system (storage units + controller)
          5. Stop the async event loop (prevents C++ crash on ObjectRefStream)
          6. Shutdown Ray Serve and Ray completely
          7. Re-initialize Ray and Ray Serve

        Phase 2 — Re-initialize:
          8. Call self.__init__() to re-create all subsystems from zero
          9. Signal the main thread to re-run training_loop()
        """
        # --- Check global restart limit ---
        self._global_restart_count += 1
        logger.info(
            f"=== Starting GLOBAL restart #{self._global_restart_count} "
            f"(max={self._max_global_restart}, full Controller re-initialization from zero) ==="
        )
        if self._global_restart_count > self._max_global_restart:
            error_msg = (
                f"Global restart count ({self._global_restart_count}) exceeded "
                f"maximum allowed ({self._max_global_restart}). "
                f"Refusing to restart — terminating process."
            )
            logger.error(error_msg)
            # Stop the HealthChecker thread to prevent further restart attempts
            # (this thread IS the checker thread, but setting stop_event ensures
            # it exits immediately when control returns to _check_loop).
            if self._health_manager._checker is not None and hasattr(self._health_manager._checker, "_stop_event"):
                self._health_manager._checker._stop_event.set()
            # Report error and force-exit the entire process.
            # os._exit is necessary because:
            # 1. raise only exits the main thread; Ray Serve / daemon threads keep the process alive
            # 2. The old HealthChecker thread would otherwise loop and re-trigger _global_restart
            self._report_error_to_metrics_service(RuntimeError(error_msg))
            logger.error("Calling os._exit(1) to force-terminate the process")
            os._exit(1)

        # Save references needed after __init__ overwrites them
        config = self.config
        recovery_load_path(config)  # Ensure config has the correct checkpoint paths after restart
        runtime_env = self.runtime_env

        # Save the old HealthChecker's stop event. Since _global_restart is
        # called FROM the old HealthChecker thread (via on_unhealthy callback),
        # _health_manager.stop() cannot actually stop the thread (join times
        # out because the thread is running this very function). We must
        # explicitly set the stop event so that when control returns to the
        # old _check_loop after _global_restart finishes, the loop exits
        # immediately instead of trying to use stale Ray actor handles from
        # the destroyed cluster (which would cause "Can't find actor" and
        # potentially a fatal C++ TryReadObjectRefStream crash).
        old_checker_stop_event = None
        if self._health_manager._checker is not None and hasattr(self._health_manager._checker, "_stop_event"):
            old_checker_stop_event = self._health_manager._checker._stop_event

        # =====================================================================
        # Phase 1: Tear down everything
        # =====================================================================

        # --- 1.1 Stop health management to prevent further callbacks ---
        try:
            self._health_manager.stop(timeout=2.0)
            logger.info("[Global Restart] Health manager stopped")
        except Exception as e:
            logger.warning(f"[Global Restart] Failed to stop health manager: {e}")

        # --- 1.2 Cancel pending ObjectRefs to unblock the main thread ---
        # This MUST happen while Ray is still alive so ray.cancel() can reach
        # the workers.  Without this, the main thread stays blocked on stale
        # ObjectRef streams and ray.shutdown() triggers a fatal C++ crash.
        self._cancel_pending_tasks()

        # --- 1.3 Tear down all service deployments ---
        for svc_role, service in self.serve_dict.items():
            try:
                service._stop_heartbeat_thread()
                logger.info(f"[Global Restart] Stopped heartbeat for '{svc_role}'")
            except Exception as e:
                logger.warning(f"[Global Restart] Failed to stop heartbeat for '{svc_role}': {e}")

            try:
                serve.delete(svc_role)
                logger.info(f"[Global Restart] Deleted Ray Serve deployment '{svc_role}'")
            except Exception as e:
                logger.warning(f"[Global Restart] Failed to delete deployment '{svc_role}': {e}")

        self.serve_dict.clear()
        logger.info("[Global Restart] All service references cleared")

        # --- 1.4 Tear down metrics deployment ---
        if self._metrics_service_enabled:
            try:
                serve.delete("metrics")
                logger.info("[Global Restart] Deleted metrics deployment")
            except Exception as e:
                logger.warning(f"[Global Restart] Failed to delete metrics deployment: {e}")

        self._shutdown_agentic_rollout_services()

        # --- 1.5 Tear down autoscaler deployment ---
        if self._autoscaler_config is not None:
            try:
                serve.delete("autoscaler")
                logger.info("[Global Restart] Deleted autoscaler deployment")
            except Exception as e:
                logger.warning(f"[Global Restart] Failed to delete autoscaler deployment: {e}")

        # --- 1.6 Tear down DCS coordinator ---
        try:
            serve.delete("dcs_coordinator")
            logger.info("[Global Restart] Deleted DCS coordinator deployment")
        except Exception as e:
            logger.warning(f"[Global Restart] Failed to delete DCS coordinator: {e}")

        # --- 1.7 Tear down data system (storage units + controller) ---
        try:
            tq.close()
        except Exception as e:
            logger.warning(f"[Global Restart] Failed to tear down data system: {e}")

        # --- 1.8 Stop the global async event loop BEFORE ray.shutdown() ---
        # The AsyncLoopThread may still hold internal watchers on Ray
        # ObjectRefStreams.  If we call ray.shutdown() while the loop is alive,
        # those watchers touch destroyed streams → fatal C++ crash
        # (TryReadObjectRefStream on a removed stream).
        shutdown_async_loop()
        logger.info("[Global Restart] Async event loop stopped")

        # --- 1.9 Terminate router processes and reset cached endpoint ---
        # The router runs as a daemon subprocess of this main process, so it
        # survives ray.shutdown().  If we don't kill it, the pre-restart engine
        # URLs stay in its worker list forever (endless "Connection refused"
        # health WARNs on dead ports), and post-restart `_start_router` also
        # short-circuits because `sglang_router_ip`/`_port` were written back
        # to the config on first launch.
        try:
            from relax.distributed.ray.rollout import stop_launched_routers

            killed = stop_launched_routers()
            # Only clear cached endpoint if we actually owned & killed a router.
            # An externally-provided router (--sglang-router-ip) was never in
            # our launched list, so we leave the endpoint set to reuse it.
            if killed > 0:
                self.config.sglang_router_ip = None
                self.config.sglang_router_port = None
                logger.info(f"[Global Restart] Terminated {killed} router process(es), endpoint reset")
        except Exception as e:
            logger.warning(f"[Global Restart] Failed to terminate router: {e}")

        # --- 1.10 Shutdown Ray Serve and Ray to kill all processes ---
        try:
            serve.shutdown()
            logger.info("[Global Restart] Ray Serve shutdown completed")
        except Exception as e:
            logger.warning(f"[Global Restart] Failed to shutdown Ray Serve: {e}")

        # Ensure the old HealthChecker thread's stop event is set BEFORE
        # ray.shutdown() destroys the old HealthStatus actor. This way, even
        # if the old thread resumes after _global_restart returns, it will
        # see _stop_event set and exit without touching stale actor handles.
        if old_checker_stop_event is not None:
            old_checker_stop_event.set()
            logger.info("[Global Restart] Old HealthChecker stop event explicitly set")

        try:
            ray.shutdown()
            logger.info("[Global Restart] Ray shutdown completed")
        except Exception as e:
            logger.warning(f"[Global Restart] Failed to shutdown Ray: {e}")

        # Wait for all processes to fully terminate
        time.sleep(5)
        logger.info("[Global Restart] Waited 5s for resource release")

        # --- 1.11 Re-initialize Ray and Ray Serve (same as train.py) ---
        ray.init(runtime_env=runtime_env)
        logger.info("[Global Restart] Ray re-initialized")
        try:
            serve.start(
                http_options={"host": "0.0.0.0", "port": "8000"},
                detached=True,
            )
        except RuntimeError:
            pass
        logger.info("[Global Restart] Ray Serve re-started")

        # =====================================================================
        # Phase 2: Re-initialize from zero by calling __init__
        # =====================================================================
        # IMPORTANT: Save a reference to the event BEFORE calling __init__(),
        # because __init__() will overwrite self._restart_done_event with a new
        # Event object. The main thread is waiting on the OLD event.
        restart_done_event = self._restart_done_event
        # Also save _restarting and _global_restart_count since __init__ resets them
        saved_restarting = True  # Must remain True so main thread knows to wait
        saved_global_restart_count = self._global_restart_count

        try:
            self.__init__(config, runtime_env)
            # Restore fields that __init__ overwrites but must survive across restarts
            self._restarting = saved_restarting
            self._global_restart_count = saved_global_restart_count
            logger.info("[Global Restart] Controller re-initialized from zero via __init__")
        except Exception as e:
            logger.exception(f"[Global Restart] Failed to re-initialize Controller: {e}")
            self._restart_error = e
            restart_done_event.set()  # Unblock main thread so it can handle the error
            return

        # Signal the main thread that global restart is done.
        # The main thread (blocked in training_loop's while-loop) will wake up
        # and re-run run_all_services() with the freshly initialized state.
        self._restart_error = None
        restart_done_event.set()
        logger.info("=== Global restart (full re-initialization) completed, main thread signaled ===")
