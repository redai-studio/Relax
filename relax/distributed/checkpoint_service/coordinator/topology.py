# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
Topology Manager - Manages the distributed topology for DCS.

Responsibilities:
- Maintain role-to-rank mappings
- Calculate logical connections between roles
- Support dynamic topology updates for elastic scaling
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from relax.distributed.checkpoint_service.config import RoleInfo, TopologyConfig


logger = logging.getLogger(__name__)


@dataclass
class TopologyNode:
    """A node in the topology graph."""

    role_info: RoleInfo
    last_heartbeat: float = field(default_factory=time.time)
    is_alive: bool = True
    connections: Set[str] = field(default_factory=set)  # Set of node_ids

    @property
    def node_id(self) -> str:
        return self.role_info.node_id


class TopologyManager:
    """Manages the distributed system topology for DCS.

    **Responsibilities:**
    - Maintain role-to-rank mappings (e.g., "actor" has ranks 0-7)
    - Assign global ranks for distributed communication
    - Calculate logical connections between roles based on topology config
    - Support dynamic topology updates for elastic scaling
    - Monitor node health via heartbeats

    **Thread-Safety:**
    All methods are thread-safe via internal RLock.

    **State:**
    - `_nodes`: Dict[role_name -> Dict[rank -> TopologyNode]]
    - `_role_name_ranks`: Dict[role_name -> List[rank]] for auto-assignment
    - `_global_ranks`: Dict[node_id -> global_rank] for mapping

    **Example:**
        manager = TopologyManager(
            config=TopologyConfig(
                role_mappings={"actor": "rollout"}
            ),
            heartbeat_timeout=30.0,
        )

        # Register nodes
        manager.register(RoleInfo(role_name="actor", rank=0, ...))
        manager.register(RoleInfo(role_name="rollout", rank=0, ...))

        # Query
        peer = manager.get_peer("actor", 0, "rollout")
        world_size = manager.get_world_size()
    """

    def __init__(
        self,
        config: Optional[TopologyConfig] = None,
        heartbeat_timeout: float = 30.0,
    ):
        """Initialize the topology manager.

        Args:
            config: Topology configuration with role mappings
            heartbeat_timeout: Seconds after which a node is declared dead
        """
        self.config = config or TopologyConfig()
        self.heartbeat_timeout = heartbeat_timeout

        # {role_name: {rank: TopologyNode}}
        self._nodes: Dict[str, List[TopologyNode] | Dict[int, TopologyNode]] = {}

        self._role_name_ranks: Dict[str, List[int]] = {}

        # {node_id: global_rank} - maps logical node IDs to global ranks
        self._global_ranks: Dict[str, int] = {}
        self._next_global_rank = 0

        self._lock = threading.RLock()

    def register(self, role_info: RoleInfo) -> int:
        """Register a node in the topology.

        Args:
            role_info: Information about the node

        Returns:
            Global rank assigned to this node
        """
        with self._lock:
            role_name = role_info.role_name
            rank = role_info.rank
            # Initialize role group if needed
            if role_name not in self._nodes:
                self._nodes[role_name] = {}
            if rank is None:
                if role_name not in self._role_name_ranks:
                    self._role_name_ranks[role_name] = []
                role_info.rank = len(self._role_name_ranks[role_name])
                self._role_name_ranks[role_name].append(role_info.rank)

            rank = role_info.rank
            node_id = role_info.node_id

            # Check if node already exists
            if rank in self._nodes[role_name]:
                existing = self._nodes[role_name][rank]
                # Update existing node
                existing.role_info = role_info
                existing.last_heartbeat = time.time()
                existing.is_alive = True
                logger.info(f"Updated existing node: {node_id}")
                return rank

            # Create new node
            node = TopologyNode(role_info=role_info)
            self._nodes[role_name][rank] = node
            return rank

    def unregister(self, role_name: str, rank: int, actor_id: str | None = None) -> bool:
        """Unregister a node from the topology.

        Args:
            role_name: Role name
            rank: Rank within the role
            actor_id: If provided, only remove the registration owned by this actor

        Returns:
            True if node was removed
        """
        with self._lock:
            if role_name not in self._nodes:
                return False

            if rank not in self._nodes[role_name]:
                return False

            node = self._nodes[role_name][rank]
            if actor_id is not None and (node.role_info.metadata or {}).get("actor_id") != actor_id:
                return False
            node_id = node.node_id

            # Remove from connections of other nodes
            for other_id in node.connections:
                other = self._get_node_by_id(other_id)
                if other:
                    other.connections.discard(node_id)

            # Remove node
            del self._nodes[role_name][rank]
            if node_id in self._global_ranks:
                del self._global_ranks[node_id]

            logger.info(f"Unregistered node: {node_id}")
            return True

    def heartbeat(self, role_name: str, rank: int) -> bool:
        """Update heartbeat for a node.

        Args:
            role_name: Role name
            rank: Rank within the role

        Returns:
            True if node exists and was updated
        """
        with self._lock:
            if role_name not in self._nodes:
                return False

            if rank not in self._nodes[role_name]:
                return False

            node = self._nodes[role_name][rank]
            node.last_heartbeat = time.time()
            node.is_alive = True
            return True

    def get_node(self, role_name: str, rank: int) -> Optional[RoleInfo]:
        """Get role info for a specific node."""
        with self._lock:
            if role_name not in self._nodes:
                return None

            if rank not in self._nodes[role_name]:
                return None

            return self._nodes[role_name][rank].role_info

    def get_role_nodes(self, role_name: str) -> Dict[int, RoleInfo]:
        """Get all nodes for a role."""
        with self._lock:
            if role_name not in self._nodes:
                return {}

            return {rank: node.role_info for rank, node in self._nodes[role_name].items() if node.is_alive}

    def get_all_nodes(self) -> Dict[str, Dict[int, RoleInfo]]:
        """Get the full topology."""
        with self._lock:
            result = {}
            for role_name, ranks in self._nodes.items():
                result[role_name] = {rank: node.role_info for rank, node in ranks.items() if node.is_alive}
            return result

    def get_global_rank(self, role_name: str, rank: int) -> int:
        """Get the global rank for a node."""
        node_id = f"{role_name}_{rank}"
        return self._global_ranks.get(node_id, -1)

    def get_peer(
        self,
        role_name: str,
        rank: int,
        peer_role: Optional[str] = None,
    ) -> Optional[RoleInfo]:
        """Get the peer node for a given node.

        Uses role mappings from config. If no mapping exists,
        defaults to same-rank peer in the specified role.

        Args:
            role_name: Current role name
            rank: Current rank
            peer_role: Target peer role (optional, uses config mapping)

        Returns:
            Peer's RoleInfo or None
        """
        with self._lock:
            # Determine peer role
            if peer_role is None:
                peer_role = self.config.get_peer_role(role_name)

            if peer_role is None:
                return None

            # Default: same rank in peer role
            return self.get_node(peer_role, rank)

    def get_all_peers(self, role_name: str, rank: int) -> List[RoleInfo]:
        """Get all connected peers for a node."""
        with self._lock:
            if role_name not in self._nodes:
                return []

            if rank not in self._nodes[role_name]:
                return []

            node = self._nodes[role_name][rank]
            peers = []

            for peer_id in node.connections:
                peer = self._get_node_by_id(peer_id)
                if peer and peer.is_alive:
                    peers.append(peer.role_info)

            return peers

    def check_health(self) -> List[str]:
        """Check health of all nodes based on heartbeat.

        Returns:
            List of dead node IDs
        """
        current_time = time.time()
        dead_nodes = []

        with self._lock:
            for role_name, ranks in self._nodes.items():
                for rank, node in ranks.items():
                    if node.is_alive:
                        if current_time - node.last_heartbeat > self.heartbeat_timeout:
                            node.is_alive = False
                            dead_nodes.append(node.node_id)
                            logger.warning(f"Node {node.node_id} marked as dead")

        return dead_nodes

    def get_world_size(self, role_name: Optional[str] = None) -> int:
        """Get the world size (total number of alive nodes)."""
        with self._lock:
            if role_name:
                if role_name not in self._nodes:
                    return 0
                return sum(1 for n in self._nodes[role_name].values() if n.is_alive)
            else:
                return sum(sum(1 for n in ranks.values() if n.is_alive) for ranks in self._nodes.values())

    def _update_connections(self, node: TopologyNode) -> None:
        """Update connections for a node based on role mappings."""
        role_name = node.role_info.role_name
        rank = node.role_info.rank

        # Get peer role from config
        peer_role = self.config.get_peer_role(role_name)

        if peer_role and peer_role in self._nodes:
            # Connect to same-rank peer
            if rank in self._nodes[peer_role]:
                peer = self._nodes[peer_role][rank]
                node.connections.add(peer.node_id)
                peer.connections.add(node.node_id)

    def _get_node_by_id(self, node_id: str) -> Optional[TopologyNode]:
        """Get a node by its ID."""
        parts = node_id.rsplit("_", 1)
        if len(parts) != 2:
            return None

        role_name, rank_str = parts
        try:
            rank = int(rank_str)
        except ValueError:
            return None

        if role_name not in self._nodes:
            return None

        return self._nodes[role_name].get(rank)

    def to_dict(self) -> Dict:
        """Export topology as a dictionary."""
        with self._lock:
            return {
                "nodes": {
                    role: {
                        rank: {
                            "role_info": node.role_info.model_dump(),
                            "global_rank": self._global_ranks.get(node.node_id, -1),
                            "is_alive": node.is_alive,
                            "connections": list(node.connections),
                        }
                        for rank, node in ranks.items()
                    }
                    for role, ranks in self._nodes.items()
                },
                "global_ranks": self._global_ranks.copy(),
                "world_size": self.get_world_size(),
            }
