# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import threading
import time
from argparse import Namespace
from typing import Any, Optional

import ray
import requests
from ray import serve
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.distributed.ray.placement_group import InfoActor, sort_key
from relax.utils import device as device_utils
from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_ray_accelerator_kwargs, get_serve_url, recovery_load_path


logger = get_logger(__name__)


class Service:
    def __init__(
        self,
        cls: Any,
        role: str,
        healthy: Any,
        config: Namespace,
        num_gpus: int = 0,
        data_source: Optional[Any] = None,
        actor_rollout_pgs: Optional[Any] = None,
        runtime_env=None,
        *,
        defer_deploy: bool = False,
    ) -> None:
        """Service wrapper that deploys a Ray Serve deployment.

        Args:
            cls: The serve-deployment class (callable) to bind and deploy.
            role: Name of the role (e.g. "actor", "rollout").
            healthy: Remote health manager actor handle.
            config: Runtime configuration (Namespace or DictConfig).
            num_gpus: Number of GPUs to allocate for this service.
            data_source: Optional data source actor or factory used by rollout.
            actor_rollout_pgs: Optional placement group for colocated actor-rollout.
            runtime_env: Optional Ray runtime environment dict for the service.
            defer_deploy: Allocate resources without deploying until ``deploy`` is called.
        """
        logger.info(
            f"[{role}] Initializing service with num_gpus={num_gpus}, actor_rollout_pgs={actor_rollout_pgs is not None}"
        )
        self.config = config
        self.role = role
        self.healthy = healthy
        self.num_gpus = num_gpus
        self.cls = cls
        self.data_source = data_source
        self.runtime_env = runtime_env
        self._is_shared_pgs = actor_rollout_pgs is not None
        self._deployed = False
        self._task_ref: Optional[Any] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        if actor_rollout_pgs is not None:
            pgs = actor_rollout_pgs
        elif num_gpus == 0:
            pgs = None
        else:
            pgs = create_placement_group(
                num_gpus=num_gpus,
                node_group_affinity=self.config.enable_affinity,
            )
        self.pgs = pgs
        logger.info(f"[{role}] Placement group initialized: {pgs}")

        if not defer_deploy:
            self.deploy()

    def deploy(self) -> None:
        """Deploy a prepared service exactly once."""
        if self._deployed:
            raise RuntimeError(f"[{self.role}] Service has already been deployed")
        self._deploy(self.pgs)
        self._deployed = True
        logger.info(f"[{self.role}] Service deployed successfully")

    def _deploy(self, pgs: Optional[Any] = None) -> None:
        """Bind and deploy the Ray Serve deployment with the given placement
        group.

        Args:
            pgs: Placement group tuple or None.
        """
        ray_actor_options = with_control_plane_affinity(
            self.config,
            {"runtime_env": self.runtime_env},
        )
        if self.data_source is not None:
            self.service = self.cls.options(ray_actor_options=ray_actor_options).bind(
                self.healthy, pgs, self.config, data_source=self.data_source, runtime_env=self.runtime_env
            )
        else:
            self.service = self.cls.options(ray_actor_options=ray_actor_options).bind(
                self.healthy, pgs, self.num_gpus, self.config, self.role, runtime_env=self.runtime_env
            )
        logger.info(f"[{self.role}] Deploying service...")
        self.handle = serve.run(self.service, name=self.role, route_prefix=f"/{self.role}")

    def _start_heartbeat(self) -> None:
        """Start background heartbeat thread to report health status."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._stop_heartbeat.clear()

        def _heartbeat_loop():
            while not self._stop_heartbeat.is_set():
                try:
                    self.healthy.update_heartbeat.remote(self.role, 0)
                except Exception as e:
                    logger.debug(f"Heartbeat error for {self.role}: {e}")
                time.sleep(10)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info(f"[{self.role}] Heartbeat thread started")

    def _stop_heartbeat_thread(self) -> None:
        """Stop the heartbeat thread."""
        if self._heartbeat_thread is not None:
            self._stop_heartbeat.set()
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
            logger.info(f"[{self.role}] Heartbeat thread stopped")

    def _http_call(
        self, role: str, path: str, method: str = "GET", params: Optional[dict] = None, timeout: float = 30
    ) -> Any:
        """Make an HTTP request to a Ray Serve deployment endpoint.

        This bypasses the Ray Serve handle and the shared AsyncLoopThread event
        loop, avoiding the single-loop deadlock that occurs when restart() is
        invoked from the HealthChecker callback.

        Args:
            role: Service role name (used as route prefix, e.g. "actor").
            path: HTTP path on the deployment (e.g. "/get_step").
            method: HTTP method ("GET" or "POST").
            params: Query parameters (GET) or JSON body (POST).
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response.
        """
        base_url = get_serve_url(route_prefix=f"/{role}")
        url = f"{base_url}{path}"
        logger.debug(f"[{self.role}] HTTP {method} {url} params={params}")
        if method.upper() == "GET":
            resp = requests.get(url, params=params, timeout=timeout)
        else:
            resp = requests.post(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def run(self) -> Any:
        """Run the service with fault supervision.

        Returns:
            A Ray ObjectRef for the async task.
        """
        self.healthy.set_task_status.remote(self.role, True)
        self._start_heartbeat()
        self._task_ref = self.handle.run.remote()
        return self._task_ref

    async def set_rollout_manager(self, rollout_manager: Any) -> None:
        await self.handle.set_rollout_manager.remote(rollout_manager)

    async def get_rollout_manager(self) -> Any:
        return await self.handle.get_rollout_manager.remote()

    async def set_barriers(self, *, rollout: Any = None, peers: Any = None) -> None:
        await self.handle.set_barriers.remote(rollout=rollout, peers=peers)

    async def set_genrm_manager(self, genrm_manager: Any) -> None:
        await self.handle.set_genrm_manager.remote(genrm_manager)

    async def get_genrm_manager(self, route_key: Optional[str] = None) -> Any:
        """Get the GenRM manager selected by ``route_key``."""
        return await self.handle.get_genrm_manager.remote(route_key)

    async def set_step(self, set_step: int) -> None:
        await self.handle.set_step.remote(set_step)

    async def get_step(self) -> int:
        return await self.handle.get_step.remote()

    async def train(self, step: int, clear_data: bool = True) -> None:
        """Execute a single training step for Actor service.

        Args:
            step: The training step number to execute
            clear_data: Whether to clear data partition after training
        """
        await self.handle.train.remote(step, clear_data=clear_data)

    async def update_weights_fully_async(self, rollout_only: bool = False, actor_fwd_only: bool = False):
        """Trigger a fully asynchronous weight update for Actor service."""
        return self.handle.update_weights_fully_async.remote(rollout_only=rollout_only, actor_fwd_only=actor_fwd_only)

    async def recv_weight_fully_async(self):
        """Trigger a fully asynchronous weight receive for Actor FWD
        service."""
        return self.handle.recv_weight_fully_async.remote()

    def restart(self) -> None:
        """Restart this service in-place: reuse placement groups and dynamic
        state.

        The restart flow:
        1. Save current dynamic state (step) from the running deployment.
        2. Stop heartbeat thread.
        3. Try to gracefully stop the old deployment.
        4. Delete the Ray Serve deployment to release replica resources.
        5. Validate and reuse existing placement group (rebuild only if broken).
        6. Redeploy with the same cls, config, and reused PG.
        7. Restore saved dynamic state (step).
        8. Sync weights from peer services if this role requires it (rollout/actor_fwd).
        9. Restart heartbeat and re-run the service task.
        """
        logger.info(f"[{self.role}] Starting in-place restart...")

        self._stop_heartbeat_thread()

        try:
            self._http_call(self.role, "/stop_service", method="POST")
            logger.info(f"[{self.role}] Old deployment stopped gracefully")
        except Exception as e:
            logger.warning(f"[{self.role}] Failed to gracefully stop old deployment: {e}")
        try:
            serve.delete(self.role)
            logger.info(f"[{self.role}] Ray Serve deployment deleted")
        except Exception as e:
            logger.warning(f"[{self.role}] Failed to delete Ray Serve deployment: {e}")

        # Wait for old deployment resources to be fully released
        time.sleep(3)
        logger.info(f"[{self.role}] Waited 3s for old deployment resource release")

        pgs = self._ensure_placement_group()

        recovery_load_path(self.config)  # Ensure config has the correct checkpoint paths after restart
        self._deploy(pgs)
        logger.info(f"[{self.role}] Service redeployed successfully")

        current_step = 0
        try:
            resp = self._http_call("actor", "/get_step")
            current_step = resp.get("step", 0)
        except Exception as e:
            current_step = ray.get(self.healthy.get_current_step.remote(role=self.role))
            logger.warning(f"[{self.role}] Failed to get current step, defaulting to step: {current_step}: {e}")

        try:
            self._http_call(self.role, "/set_step", method="POST", params={"step": current_step})
            logger.info(f"[{self.role}] Restored step to {current_step}")
        except Exception as e:
            logger.warning(f"[{self.role}] Failed to restore step: {e}")

        self._task_ref = None
        task_ref = self.run()
        logger.info(f"[{self.role}] Service task re-launched after restart, task_ref={task_ref}")

    def _ensure_placement_group(self) -> Optional[Any]:
        """Validate existing placement group or rebuild it if broken.

        For shared PGs (colocate scenario), always reuse without rebuilding.
        For self-owned PGs, check health and rebuild if needed.

        Returns:
            The validated (or newly created) placement group tuple, or None.
        """
        if self.pgs is None:
            # No PG needed (e.g. num_gpus=0)
            return None

        if self._is_shared_pgs:
            # Shared PG (colocate): always reuse, never destroy
            logger.info(f"[{self.role}] Reusing shared (colocate) placement group")
            return self.pgs

        # Self-owned PG: validate by checking if it's still alive
        try:
            pg = self.pgs[0] if isinstance(self.pgs, tuple) else self.pgs
            ready, _ = ray.wait([pg.ready()], timeout=5)
            if ready:
                logger.info(f"[{self.role}] Existing placement group is healthy, reusing")
                return self.pgs
            else:
                logger.warning(f"[{self.role}] Placement group health check timed out, rebuilding")
        except Exception as e:
            logger.warning(f"[{self.role}] Placement group validation failed: {e}, rebuilding")

        # Cleanup broken PG
        try:
            pg = self.pgs[0] if isinstance(self.pgs, tuple) else self.pgs
            remove_placement_group(pg)
            logger.info(f"[{self.role}] Old placement group removed")
        except Exception as e:
            logger.warning(f"[{self.role}] Failed to remove old placement group: {e}")

        # Keep recovered baseline roles on the stable worker group.
        new_pgs = create_placement_group(
            num_gpus=self.num_gpus,
            node_group_affinity=self.config.enable_affinity,
        )
        self.pgs = new_pgs
        logger.info(f"[{self.role}] New placement group created")
        return new_pgs


def _require_node_group_markers(node_group: str, retries: int = 3, retry_delay: float = 2.0) -> None:
    """Raise if the node-group marker resources remain unavailable."""
    required = {f"{node_group}_gpu", f"{node_group}_cpu"}
    last_available: dict = {}
    for attempt in range(retries):
        cluster_resources = ray.cluster_resources()
        last_available = cluster_resources
        missing = {r for r in required if cluster_resources.get(r, 0) <= 0}
        if not missing:
            return
        if attempt < retries - 1:
            logger.warning(
                f"[Affinity] Node-group marker resources {sorted(missing)} not yet present "
                f"(attempt {attempt + 1}/{retries}); retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)

    missing = sorted(r for r in required if last_available.get(r, 0) <= 0)
    raise RuntimeError(
        f"Node-group affinity is enabled (RELAX_INITIAL_NODE_GROUP='{node_group}'), but the "
        f"required custom marker resources {missing} are not declared anywhere in this cluster. "
        f"Baseline placement groups would hang forever waiting for them. This usually means the "
        f"cluster has no '{node_group}' worker-group declaring '{node_group}_gpu'/'{node_group}_cpu' "
        f"custom resources. Please check your cluster environment; if this cluster intentionally has "
        f"no such marker resources, disable affinity by passing --no-enable-affinity."
    )


def create_placement_group(num_gpus, node_group_affinity=True):
    """Create a packed GPU placement group with optional node-group
    affinity."""
    accel_resource = device_utils.get_ray_accelerator_name()
    base_bundle = {accel_resource: 1, "CPU": 1}
    node_group = os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip()
    if node_group_affinity and node_group:
        _require_node_group_markers(node_group)
        base_bundle = {**base_bundle, f"{node_group}_gpu": 1, f"{node_group}_cpu": 1}
    bundles = [dict(base_bundle) for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)
    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    accelerator_kwargs = get_ray_accelerator_kwargs(1)
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
                **accelerator_kwargs,
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids
