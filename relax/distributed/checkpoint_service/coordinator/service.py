# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
DCS Coordinator Service - Control plane for the Distributed Checkpoint Service.

Responsibilities:
- Role registration and discovery
- Topology management
- Heartbeat monitoring
- Elastic scaling coordination
- Checkpoint version management

Deployment options:
- Standalone: FastAPI/uvicorn
- Ray Serve: For integration with Ray clusters
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.distributed.checkpoint_service.config import DCSConfig, RoleInfo, TopologyConfig
from relax.distributed.checkpoint_service.coordinator.topology import TopologyManager
from relax.utils.env import Envs
from relax.utils.http_utils import _wrap_ipv6
from relax.utils.utils import get_serve_url


logger = logging.getLogger(__name__)


# ============================================================================
# API Models
# ============================================================================


class Response(BaseModel):
    """Response model for heartbeat."""

    status: str
    message: str


class RegisterRequest(BaseModel):
    """Request model for node registration."""

    role_name: str | None = None
    rank: int | None = None
    world_size: int | None = None
    ip: str | None = None
    port: int | None = None
    device_id: int | None = None
    metadata: Dict[str, Any] | None = None


class RegisterResponse(BaseModel):
    """Response model for node registration."""

    status: str
    message: str
    rank: int
    node_id: str


class HeartbeatResponse(BaseModel):
    """Response model for heartbeat."""

    status: str
    timestamp: float


class TopologyResponse(BaseModel):
    """Response model for topology query."""

    nodes: Dict[str, Dict[int, Dict[str, Any]]]
    world_size: int


class SendWeightMetaRequest(BaseModel):
    names: list
    dtypes: list
    shapes: list
    group_name: str
    update_id: str = ""


class GroupRanksResponse(BaseModel):
    """Response model for resharding operation."""

    global_rank: int
    world_size: int
    train_pp_size: int
    pp_groups: dict
    topology_id: str = ""
    update_id: str = ""
    includes_reference: bool = False


# ============================================================================
# Coordinator Service
# ============================================================================

app = FastAPI(
    title="DCS Coordinator",
    description="Distributed Checkpoint Service - Control Plane",
    version="0.1.0",
)


DCS_SERVE_MAX_ONGOING_REQUESTS = Envs.DCS_SERVE_MAX_ONGOING_REQUESTS


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=DCS_SERVE_MAX_ONGOING_REQUESTS,
    ray_actor_options={"num_cpus": 1},
    logging_config=LoggingConfig(
        log_level="WARNING",
        enable_access_log=False,  # 关闭 HTTP 访问日志
    ),
)
@serve.ingress(app)
class DCSCoordinator:
    """
    DCS Coordinator - The control plane service.

    Central management service for the Distributed Checkpoint Service:

    **Responsibilities:**
    - Role registration and rank assignment
    - Topology discovery and peer lookup
    - Heartbeat monitoring and node health tracking
    - Model update communication group formation
    - Checkpoint version management
    - Elastic scaling coordination

    **Deployment Options:**
    1. Standalone (FastAPI + uvicorn)
    2. Ray Serve (integrated with Ray cluster)

    **Key Properties:**
    - Stateless: Can be replicated for high availability
    - Non-blocking: All endpoints are async
    - Fault-tolerant: Clients retry registration/heartbeat

    Example (standalone):
        coordinator = DCSCoordinator()

        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

    Example (Ray Serve):
        from ray import serve
        serve.start()
        handle = serve.run(DCSCoordinator.bind())
        print(handle)
    """

    def __init__(
        self,
        config: Optional[DCSConfig] = None,
        topology_config: Optional[TopologyConfig] = None,
    ):
        """Initialize the coordinator.

        Args:
            config: DCS configuration with coordinator settings
            topology_config: Role mapping configuration (defines peer connections)
        """
        self.config = config or DCSConfig()
        self.topology_manager = TopologyManager(
            config=topology_config,
            heartbeat_timeout=self.config.heartbeat_timeout_seconds,
        )

        self.weight_meta_buffer = []
        self.weight_updates = {}
        self.recv_index_cache = 0
        self.target_world_size = 0
        self.recv_count = 0

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = threading.Lock()

    @app.post("/register", response_model=RegisterResponse)
    async def register_node(self, request: RegisterRequest):
        """Register a new node with the coordinator.

        This is the first API call a client makes when joining the DCS cluster.
        The coordinator assigns a rank within the role and returns it.

        Args:
            request: RegisterRequest with node information
                - role_name: Logical role (e.g., "actor", "rollout", "trainer")
                - rank: Optional desired rank (auto-assigned if None)
                - ip: IP address for P2P communication
                - port: Port for P2P communication
                - device_id: GPU device ID if applicable
                - metadata: Custom attributes (parallelism sizes, cluster info, etc.)

        Returns:
            RegisterResponse with:
                - status: "success" or "error"
                - message: Human-readable message
                - rank: Assigned rank for this node
                - node_id: Unique identifier "{role_name}_{rank}"

        Raises:
            HTTPException: If registration fails
        """
        role_info = RoleInfo(
            role_name=request.role_name,
            rank=request.rank,
            world_size=request.world_size,
            ip=request.ip,
            port=request.port,
            device_id=request.device_id,
            metadata=request.metadata,
        )

        rank = self.topology_manager.register(role_info)

        return RegisterResponse(
            status="success",
            message=f"Registered {role_info.node_id}",
            rank=rank,
            node_id=role_info.node_id,
        )

    @app.delete("/unregister")
    async def unregister_node(self, role: str, rank: int):
        """Unregister a node from the coordinator.

        Called when a node is shutting down or being removed from the cluster.
        Updates topology so peers know the node is no longer available.

        Args:
            role: Role name
            rank: Rank within the role

        Returns:
            Dict with status message

        Raises:
            HTTPException(404): If node not found
        """
        success = self.topology_manager.unregister(role, rank)
        if not success:
            raise HTTPException(status_code=404, detail="Node not found")
        return {"status": "success", "message": f"Unregistered {role}_{rank}"}

    @app.get("/heartbeat", response_model=HeartbeatResponse)
    async def heartbeat(self, role: str, rank: int):
        """Update heartbeat for a node.

        Clients call this periodically to signal they are alive.
        If heartbeat is not received for heartbeat_timeout_seconds,
        the node is declared dead.

        Args:
            role: Role name
            rank: Rank within the role

        Returns:
            HeartbeatResponse with status and timestamp

        Raises:
            HTTPException(404): If node not found
        """
        success = self.topology_manager.heartbeat(role, rank)
        if not success:
            raise HTTPException(status_code=404, detail="Node not found")
        return HeartbeatResponse(
            status="ok",
            timestamp=time.time(),
        )

    @app.get("/topology", response_model=TopologyResponse)
    async def get_topology(self, role_filter: Optional[str] = None):
        """Get the current cluster topology.

        Returns all registered nodes organized by role.
        Clients use this to discover peers and their addresses.

        Args:
            role_filter: Optional role name to filter results

        Returns:
            TopologyResponse with:
                - nodes: Dict of {role: {rank: {node_info}}}
                - world_size: Total number of nodes
        """
        if role_filter:
            nodes = {role_filter: self.topology_manager.get_role_nodes(role_filter)}
        else:
            nodes = self.topology_manager.get_all_nodes()

        # Convert RoleInfo to dict
        nodes_dict = {role: {rank: info.model_dump() for rank, info in ranks.items()} for role, ranks in nodes.items()}

        return TopologyResponse(
            nodes=nodes_dict,
            world_size=self.topology_manager.get_world_size(),
        )

    @app.get("/peer")
    async def get_peer(self, role: str, rank: int, peer_role: Optional[str] = None):
        """Get the peer node for a given node.

        Returns the peer that should receive tensors based on role mappings.
        For example, if role mapping is {"actor": "rollout"},
        get_peer("actor", 5) returns "rollout" rank 5.

        Args:
            role: Source role name
            rank: Source rank
            peer_role: Optional target role (if not using topology mapping)

        Returns:
            RoleInfo dict of the peer

        Raises:
            HTTPException(404): If peer not found
        """
        peer = self.topology_manager.get_peer(role, rank, peer_role)
        if peer is None:
            raise HTTPException(status_code=404, detail="Peer not found")
        return peer.model_dump()

    @app.get("/node")
    async def get_node(self, role: str, rank: int):
        """Get information about a specific node.

        Args:
            role: Role name
            rank: Rank within the role

        Returns:
            RoleInfo dict

        Raises:
            HTTPException(404): If node not found
        """
        node = self.topology_manager.get_node(role, rank)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return node.model_dump()

    @app.get("/global_rank")
    async def get_global_rank(self, role: str, rank: int):
        """Get the global rank for a node.

        Global rank is a unique identifier across all roles.
        Used internally for communication group formation.

        Args:
            role: Role name
            rank: Rank within the role

        Returns:
            Dict with global_rank

        Raises:
            HTTPException(404): If node not found
        """
        global_rank = self.topology_manager.get_global_rank(role, rank)
        if global_rank < 0:
            raise HTTPException(status_code=404, detail="Node not found")
        return {"global_rank": global_rank}

    @app.get("/get_model_update_group_ranks", response_model=GroupRanksResponse)
    async def get_model_update_group_ranks(
        self, role, rank, need_update_ref, rollout_id: int = 0, occurrence: int = 0
    ):
        train_nodes = self.topology_manager.get_role_nodes("actor")
        receivers = [
            node
            for target_role in (("actor_fwd", "reference") if need_update_ref == "true" else ("actor_fwd",))
            for _, node in sorted(self.topology_manager.get_role_nodes(target_role).items())
        ]
        members = [node.model_dump() for _, node in sorted(train_nodes.items())] + [n.model_dump() for n in receivers]
        topology_id = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()[:24]
        update_id = f"{topology_id}:{rollout_id}:{occurrence}"
        expected = {node.node_id for node in receivers}
        self.weight_updates.setdefault(update_id, {"expected": expected, "finished": set(), "metadata": []})
        node_id = f"{role}_{int(rank)}"
        if role == "actor":
            global_rank = 0
        elif node_id in expected:
            global_rank = 1 + [node.node_id for node in receivers].index(node_id)
        else:
            raise HTTPException(409, "Role is outside this update's receiver set")
        includes_reference = any(node.role_name == "reference" for node in receivers)
        pp_groups = {}
        for _, node in sorted(train_nodes.items()):
            metadata = node.metadata
            if metadata.get("is_pp_src_rank"):
                port = metadata.get("reference_master_port") if includes_reference else metadata.get("master_port")
                port = port or metadata.get("master_port")
                name = f"update_actor_pp_{metadata['pp_rank']}_{topology_id}"
                pp_groups[name] = f"tcp://{_wrap_ipv6(metadata['master_address'])}:{port}"
        return GroupRanksResponse(
            global_rank=global_rank,
            world_size=1 + len(receivers),
            train_pp_size=train_nodes[0].metadata.get("pp_size", 1),
            pp_groups=pp_groups,
            topology_id=topology_id,
            update_id=update_id,
            includes_reference=includes_reference,
        )

    @app.get("/recv_weight_meta")
    async def recv_weight_meta(self, index: int, wait_timeout_s: float = 0.0, update_id: str = ""):
        deadline = time.monotonic() + wait_timeout_s
        while True:
            buffer = (
                self.weight_updates.get(update_id, {}).get("metadata", []) if update_id else self.weight_meta_buffer
            )
            data = buffer[index:]
            if data or time.monotonic() >= deadline:
                return data
            await asyncio.sleep(0.01)

    @app.get("/clear_weight_meta")
    async def clear_weight_meta(self, update_id: str = "", receiver: str = ""):
        if not update_id:
            self.weight_meta_buffer.clear()
            return
        update = self.weight_updates.get(update_id)
        if update is None:
            return
        if receiver not in update["expected"]:
            raise HTTPException(409, "Unknown receiver for this update")
        update["finished"].add(receiver)
        if update["finished"] == update["expected"]:
            del self.weight_updates[update_id]

    @app.post("/send_weight_meta", response_model=Response)
    async def send_weight_meta(self, request: SendWeightMetaRequest):
        with self._lock:
            if request.update_id:
                update = self.weight_updates.get(request.update_id)
                if update is None:
                    raise HTTPException(409, "Unknown or completed update")
                update["metadata"].append(request)
            else:
                self.weight_meta_buffer.append(request)
        return Response(status="success", message="send_weight_meta")

    @app.get("/health")
    async def health_check(self):
        """Health check endpoint."""
        dead_nodes = self.topology_manager.check_health()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "world_size": self.topology_manager.get_world_size(),
            "dead_nodes": dead_nodes,
        }

    @app.get("/debug/topology")
    async def debug_topology(self):
        """Debug endpoint to get full topology details."""
        return self.topology_manager.to_dict()


# ============================================================================
# Ray Serve deployment (optional)
# ============================================================================


def create_dcs_deployment(
    config: Optional[DCSConfig] = None,
    ray_actor_options: Optional[Dict[str, Any]] = None,
):
    """Create a Ray Serve deployment for the coordinator.

    Usage:
        deployment = create_dcs_deployment()
    """
    deployment = DCSCoordinator
    if ray_actor_options is not None:
        deployment = deployment.options(ray_actor_options=ray_actor_options)
    coordinator = serve.run(deployment.bind(config=config), name="dcs_coordinator", route_prefix="/dcs_coordinator")
    coordinator_url = get_serve_url("dcs_coordinator")
    return coordinator, coordinator_url
