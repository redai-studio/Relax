# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP weight-sync control without importing the training tensor runtime."""

import json
import time
from typing import Any, Dict, Optional

import requests
from urllib3.exceptions import NewConnectionError

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
EXPERT_IPC_LOADER = "relax.backends.sglang.shard_p2p_weight_loader.control"
TP_IPC_LOADER = "relax.backends.sglang.shard_p2p_weight_loader.control_tp"
COLOCATE_NIXL_LOADER = "relax.backends.sglang.shard_p2p_weight_loader.control_remote"


def normalize_shard_control_response(response: dict | None, load_format: str | None) -> dict | None:
    """DP schedulers can repeat the same full TP-group reply, joined by |."""
    if not (
        load_format in (COLOCATE_NIXL_LOADER, EXPERT_IPC_LOADER, TP_IPC_LOADER)
        and response
        and response.get("success") is True
        and response.get("message", "").lstrip().startswith("{")
    ):
        return response
    decoder = json.JSONDecoder()
    remaining = response["message"].strip()
    reply, end = decoder.raw_decode(remaining)
    while remaining := remaining[end:].strip():
        if not remaining.startswith("|"):
            raise ValueError("Invalid SGLang shard control reply separator")
        remaining = remaining[1:].lstrip()
        duplicate, end = decoder.raw_decode(remaining)
        if duplicate != reply:
            raise ValueError("SGLang DP schedulers disagree on the shard control reply")
    return {**response, "message": json.dumps(reply, separators=(",", ":"))}


class RolloutEngine:
    """Ray Actor for handling HTTP requests to rollout nodes.

    Encapsulates HTTP communication with a specific rollout endpoint.
    """

    def __init__(self, rank: int, node_info: Dict[str, Any]):
        """Initialize RolloutEngine actor.

        Args:
            rank: Rank/index of this rollout node
            node_info: Dict with 'ip' and 'port' keys
        """
        self.rank = rank
        self.node_info = node_info
        self.base_url = f"http://{node_info['ip']}:{node_info['port']}"
        logger.info(f"RolloutEngine #{self.rank} initialized for {self.base_url}")

    def health(self, timeout: float = 5.0, generate: bool = True) -> bool:
        endpoint = "health_generate" if generate else "get_server_info"
        response = requests.get(f"{self.base_url}/{endpoint}", timeout=timeout)
        response.raise_for_status()
        return True

    def make_request(self, endpoint: str, payload: Optional[Dict] = None) -> Any:
        """Send a synchronous HTTP POST to the rollout node and return JSON.

        Args:
            endpoint: Path on the node (e.g. '/init_weights_update_group').
            payload: Optional JSON payload.

        Returns:
            Parsed JSON response from the remote node.
        """
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def update_weights_from_tensor(
        self, serialized_named_tensors: list[str], load_format: str, flush_cache: bool = False
    ) -> dict:
        """Use the same checked native-loader RPC as colocated shard
        sessions."""
        response = requests.post(
            self.base_url + "/update_weights_from_tensor",
            json=dict(
                serialized_named_tensors=serialized_named_tensors, load_format=load_format, flush_cache=flush_cache
            ),
            timeout=600,
        )
        response.raise_for_status()
        return normalize_shard_control_response(response.json(), load_format)

    def flush_cache(self) -> None:
        """Poll the remote server until its cache is flushed or timeout.

        Retries for a short period and raises on timeout.
        """
        # flush_cache may return non-200 while there are pending requests
        url = f"{self.base_url}/flush_cache"
        for _ in range(60):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
            except NewConnectionError:
                raise
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")
