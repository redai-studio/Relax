# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
Checkpoint Engine Client - Data plane client for DCS.

Provides:
- Node registration with the coordinator
- Peer discovery
- Tensor sending/receiving through backends
- Checkpoint save/load operations
- Automatic fault recovery
"""

import asyncio
import time
from typing import Any, Dict, Optional, Sequence

import httpx

from relax.distributed.checkpoint_service.backends import CommBackend, DeviceDirectBackend
from relax.distributed.checkpoint_service.config import BackendType, DCSConfig, RoleInfo
from relax.distributed.checkpoint_service.metrics import MetricsCollector
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class CheckpointEngineClient:
    """Distributed Checkpoint Engine Client - Data plane client for DCS.

    Handles node registration, peer discovery, tensor communication, and checkpoint operations
    with a DCS coordinator via automatic heartbeat and topology updates.

    Example:
        client = CheckpointEngineClient(
            args=args,
            coordinator_url="http://localhost:8000",
            role="actor",
            rank=0,
        )
        await client.start()
        await client.stop()
    """

    def __init__(
        self,
        args,
        coordinator_url: str,
        role: str,
        rank: int,
        config: Optional[DCSConfig] = None,
        backend_type: BackendType = BackendType.NCCL,
        device_id: int = 0,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[Sequence] = None,
        model_name: Optional[str] = None,
        quantization_config: Optional[Dict[str, int | str | list[str]]] = None,
        lock: Any = None,
    ):
        """Initialize checkpoint engine client.

        Args:
            args: Command-line arguments object
            coordinator_url (str): URL of DCS coordinator (e.g., "http://localhost:8000")
            role (str): Role name (e.g., "actor", "rollout", "trainer", "reference")
            rank (int): Rank within role (starts from 0)
            config (Optional[DCSConfig]): Checkpoint engine config. Default: None
            backend_type (BackendType): Communication backend type. Default: NCCL
            device_id (int): GPU/device ID. Default: 0
            ip (Optional[str]): Node IP address. If None, auto-detected. Default: None
            port (Optional[int]): Communication port. If None, auto-assigned. Default: None
            metadata (Optional[Dict]): Additional node metadata. Default: None
            model (Optional[Sequence]): Model object reference. Default: None
            model_name (Optional[str]): Model name (e.g., "qwen3-4B"). Default: None
            quantization_config (Optional[Dict]): Quantization configuration. Default: None
            lock :
        """
        self.coordinator_url = coordinator_url.rstrip("/")
        self.config = config or DCSConfig()

        self.role_info = RoleInfo(
            role_name=role,
            rank=rank,
            ip=ip,
            port=port,
            device_id=device_id,
            metadata=metadata or {},
        )
        self.args = args
        self.backend_type = backend_type
        self._backend: Optional[CommBackend] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.lock = lock

        # State
        self._registered = False
        self._rank = -1
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._topology_cache: Dict[str, Dict[int, RoleInfo]] = {}
        self._last_topology_update: float = 0

        # Metrics
        self._metrics = MetricsCollector() if self.config.enable_metrics else None

    @property
    def role(self) -> str:
        """Get the role name of current node."""
        return self.role_info.role_name

    @property
    def rank(self) -> int:
        """Get the rank within the same role."""
        return self.role_info.rank

    @property
    def world_size(self) -> int:
        """Get the total number of nodes in the same role."""
        return self.role_info.world_size

    @property
    def node_id(self) -> str:
        """Get unique node identifier in format '{role_name}-{rank}'."""
        return self.role_info.node_id

    @property
    def is_registered(self) -> bool:
        """Check if node is registered with coordinator."""
        return self._registered

    @property
    def backend(self) -> CommBackend:
        """Get communication backend instance."""
        if self._backend is None:
            raise RuntimeError("Backend not initialized. Call start() first.")
        return self._backend

    async def start(self) -> None:
        """Start the checkpoint engine client.

        Registers with coordinator, initializes backend, and starts heartbeat
        task.
        """
        if self._running:
            logger.warning("Client already running")
            return

        # Create HTTP client
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Register with coordinator
        self.role_info.rank = await self._register()

        # Initialize backend
        await self._init_backend(self.lock)

        # # Start heartbeat
        # self._running = True
        # self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Checkpoint engine client started: {self.node_id}")

    async def stop(self) -> None:
        """Stop the checkpoint engine client and release resources."""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close backend
        if self._backend:
            await self._backend.close_async()

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()

        logger.info(f"Checkpoint engine client stopped: {self.node_id}")

    async def _register(self) -> int:
        """Register node with coordinator and return assigned rank."""
        payload = {
            "role_name": self.role_info.role_name,
            "rank": self.role_info.rank,
            "ip": self.role_info.ip,
            "port": self.role_info.port,
            "device_id": self.role_info.device_id,
            "metadata": self.role_info.metadata,
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._http_client.post(
                    f"{self.coordinator_url}/register",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                self._rank = data["rank"]
                self._registered = True

                logger.info(f"Registered with coordinator: {self.node_id} -> rank={self._rank}")
                break
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(f"Registration failed, retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    raise RuntimeError(f"Failed to register with coordinator: {e}")
        return self._rank

    async def unregister(self) -> None:
        """Unregister node from coordinator."""
        if not self._registered:
            return

        try:
            response = await self._http_client.delete(
                f"{self.coordinator_url}/unregister",
                params={"role": self.role, "rank": self.rank},
            )
            response.raise_for_status()
            logger.info(f"Unregistered from coordinator: {self.node_id}")
        except Exception as e:
            logger.warning(f"Failed to unregister from coordinator: {e}")

    async def _init_backend(self, lock) -> None:
        """Initialize communication backend (currently DeviceDirectBackend)."""
        if self.backend_type == BackendType.TCP:
            raise TypeError(f"Not support {BackendType.TCP} now!")
        else:
            self._backend = DeviceDirectBackend(
                args=self.args,
                backend_type=self.backend_type,
                role_info=self.role_info,
                model=self.model,
                model_name=self.model_name,
                quantization_config=self.quantization_config,
                coordinator_url=self.coordinator_url,
                lock=lock,
            )

    async def init_process_groups_for_actor_fwd_ref(self, rollout_id: int) -> None:
        """Update reference model or forward actor weights based on rollout
        interval.

        Args:
            rollout_id (int): Current rollout ID for checking update period.
        """
        self.need_update_ref = (
            self.args.ref_update_interval is not None and (rollout_id + 1) % self.args.ref_update_interval == 0
        )
        if not self.need_update_ref and self.role == "reference":
            return
        response = await self._http_client.get(
            f"{self.coordinator_url}/get_model_update_group_ranks",
            params={"role": self.role, "rank": self.rank, "need_update_ref": self.need_update_ref},
        )
        response.raise_for_status()
        data = response.json()
        logger.info(f"Topology data for actor_fwd_ref weight update: {data}")
        self._backend.init_process_groups_for_actor_fwd_ref(data)
        logger.info(f"Weights updated for actor_fwd_ref weight update, role: {self.role}.")

    async def recv_weight_fully_async(self):
        if not self.need_update_ref and self.role == "reference":
            return
        self._backend.recv_weight()

    async def update_weights_for_rollout(self, rollout_only=False, actor_fwd_only=False) -> set[str] | None:
        """Update weights for rollout role from trainer."""
        response = await self._http_client.get(f"{self.coordinator_url}/topology")
        response.raise_for_status()
        data = response.json()
        logger.info(f"get topology: {data}")
        if not actor_fwd_only:
            self._backend.init_process_group_for_rollout(data)
        self._backend.update_weights_for_rollout(rollout_only, actor_fwd_only)
        logger.info("Weights updated for rollout role.")
        if not actor_fwd_only:
            return self._backend.updated_rollout_urls()
        return None

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat signals to coordinator."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

                response = await self._http_client.get(
                    f"{self.coordinator_url}/heartbeat",
                    params={"role": self.role, "rank": self.rank},
                )
                response.raise_for_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    async def _refresh_topology(self) -> None:
        """Fetch and cache network topology from coordinator."""
        try:
            response = await self._http_client.get(f"{self.coordinator_url}/topology")
            response.raise_for_status()
            data = response.json()

            # Update cache
            for role_name, ranks in data["nodes"].items():
                if role_name not in self._topology_cache:
                    self._topology_cache[role_name] = {}

                for rank_str, info in ranks.items():
                    rank = int(rank_str)
                    role_info = RoleInfo(**info)
                    self._topology_cache[role_name][rank] = role_info

                    # Register peer with backend
                    if self._backend and role_info.node_id != self.node_id:
                        self._backend.register_peer(role_info)

            self._last_topology_update = time.time()
            logger.debug(f"Topology refreshed: {len(data['nodes'])} roles")

        except Exception as e:
            logger.error(f"Failed to refresh topology: {e}")


async def create_client(
    args,
    coordinator_url: str,
    role: str,
    rank: Optional[int] = None,
    **kwargs,
) -> CheckpointEngineClient:
    """Create and start a checkpoint engine client.

    Args:
        args: Command-line arguments
        coordinator_url (str): DCS coordinator URL (e.g., "http://localhost:8000")
        role (str): Role name (e.g., "actor", "rollout", "trainer", "reference")
        rank (Optional[int]): Rank within role. If None, assigned by coordinator. Default: None
        **kwargs: Additional args forwarded to CheckpointEngineClient.__init__()

    Returns:
        CheckpointEngineClient: Started and ready-to-use client instance

    Raises:
        RuntimeError: If client startup fails (e.g., coordinator registration fails)

    Example:
        client = await create_client(
            args=args,
            coordinator_url="http://localhost:8000",
            role="actor",
        )
        await client.stop()
    """
    client = CheckpointEngineClient(
        args=args,
        coordinator_url=coordinator_url,
        role=role,
        rank=rank,
        **kwargs,
    )
    await client.start()
    return client
