# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""
DeviceDirectBackend - Communication backend using PyTorch distributed (NCCL/GLOO).

Supports:
- NCCL: For GPU-to-GPU communication, high efficiency
- GLOO: For CPU-based communication, fully async with device computation

Features:
- Process group management
- Broadcast, send/recv operations
- Async operations with CUDA streams
"""

import asyncio
import logging
import re
import socket
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import ray
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm
from urllib3.exceptions import NewConnectionError

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.distributed.checkpoint_service.backends.base import CommBackend, TensorFusion
from relax.distributed.checkpoint_service.config import BackendType, RoleInfo
from relax.utils import device as device_utils
from relax.utils.distributed_utils import get_gloo_group, init_process_group
from relax.utils.env import Envs
from relax.utils.http_utils import _wrap_ipv6
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_peft_utils import (
    LORA_ADAPTER_NAME,
    is_lora_adapter_mode,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
)


logging.getLogger("httpx").setLevel(logging.WARNING)

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_megatron_dependencies() -> SimpleNamespace:
    """Load DeviceDirect's optional Megatron implementation on first use."""
    try:
        from megatron.core import mpu

        from relax.backends.megatron.weight_conversion import convert_to_hf
        from relax.backends.megatron.weight_update.common import all_gather_param, named_params_and_buffers
        from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import (
            _adapter_base_prefix,
            _base_param_prefix,
        )
        from relax.backends.megatron.weight_update.lora_adapter_sync import LoraAdapterSync
        from relax.distributed.checkpoint_service.utils import load_weight
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "megatron" or missing.startswith("megatron."):
            raise ModuleNotFoundError(
                "DeviceDirectBackend requires the optional Megatron dependencies; "
                "install or use the Relax Megatron training environment."
            ) from exc
        raise
    return SimpleNamespace(
        mpu=mpu,
        convert_to_hf=convert_to_hf,
        all_gather_param=all_gather_param,
        named_params_and_buffers=named_params_and_buffers,
        adapter_base_prefix=_adapter_base_prefix,
        base_param_prefix=_base_param_prefix,
        LoraAdapterSync=LoraAdapterSync,
        load_weight=load_weight,
    )


def bucket_tensor_counts(sizes: Sequence[int], max_bytes: int) -> list[int]:
    """Split a tensor sequence into buckets of at most ``max_bytes``.

    Returns the NUMBER OF TENSORS per bucket (not byte totals), because that is what
    both ends of a broadcast need to stay in lockstep: the sender walks its tensors and
    the receiver walks the matching ``names``/``shapes`` metadata, so an explicit count
    list removes any chance of the two deriving different boundaries from the same rule.

    A tensor larger than ``max_bytes`` gets a bucket of its own rather than being split
    — the transport is per-tensor, so this is the smallest achievable unit.

    Args:
        sizes: Byte size of each tensor, in broadcast order.
        max_bytes: Soft cap per bucket.

    Returns:
        Tensor counts per bucket; ``sum(result) == len(sizes)``.
    """
    counts: list[int] = []
    count = 0
    total = 0
    for size in sizes:
        if count and total + size > max_bytes:
            counts.append(count)
            count = 0
            total = 0
        count += 1
        total += size
    if count:
        counts.append(count)
    return counts


class DeviceDirectBackend(CommBackend):
    """PyTorch distributed communication backend using NCCL (GPU) or GLOO
    (CPU).

    Example:
        backend = DeviceDirectBackend(
            backend_type=BackendType.GLOO,
            role_info=RoleInfo(...)
        )
        backend.init_process_group()
        backend.send({"weight": tensor}, dst=1)
        tensors = backend.recv(src=0)
    """

    def __init__(
        self,
        args,
        backend_type: BackendType,
        role_info: Optional[RoleInfo],
        model: Sequence[torch.nn.Module],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        coordinator_url=None,
        lock: Any = None,
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize DeviceDirectBackend.

        Args:
            args: Backend arguments
            backend_type: GLOO or NCCL
            role_info: Current node information
            model: Model instance(s)
            model_name: Model identifier
            quantization_config: Optional quantization settings
            coordinator_url: URL of the coordinator service
            lock: Remote lock for coordinating weight updates
            timeout_seconds: Operation timeout (default 300)
        """
        super().__init__(backend_type, role_info)
        self._megatron = _load_megatron_dependencies()
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.http_client = httpx.Client(timeout=30.0)
        self.coordinator_url = coordinator_url
        self.lock = lock
        self.timeout_seconds = timeout_seconds
        self.device = next(model[0].parameters()).device if model else device_utils.current_device()

        self._comm_stream: Optional[Any] = None  # CUDA stream
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._tensor_fusion = TensorFusion()

        # For recv, we need to know tensor shapes in advance or use a metadata channel
        self._pending_recvs: Dict[str, asyncio.Future] = {}
        self._model_update_groups = None
        self._model_update_groups_for_actor_fwd_ref = None

        # World size for process group initialization
        self.world_size: Optional[int] = None

        # Ray actors for rollout communication
        self.rollout_engines: Dict[int, Any] = {}  # rank -> Ray actor handle
        # Signature of the rollout topology the current engines + NCCL weight-update
        # group were built for. When unchanged (and engines healthy), the group and
        # proxy actors are reused across weight updates instead of being torn down
        # and rebuilt every step. None forces a (re)build on the next update.
        self._rollout_topology_signature: Optional[frozenset] = None
        device_utils.set_device(self.device)

        # Bridge-based HF weight converter (lazy-initialized on first use)
        self._use_bridge = getattr(args, "megatron_to_hf_mode", None) == "bridge"
        if self._use_bridge:
            from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter

            self._bridge_converter = BridgeConverter(args=args, model=model, quantization_config=quantization_config)

        # LoRA weight-sync state. Mirrors the colocate UpdateWeightFromTensor fields so the
        # fully-async path supports both merge mode (fold adapter into base, reuse the NCCL
        # broadcast) and adapter mode (base synced once, adapter pushed to SGLang each step).
        self._lora_enabled = is_lora_enabled(args)
        self._lora_merge_mode = is_lora_merge_mode(args) if self._lora_enabled else False
        self._lora_adapter_mode = is_lora_adapter_mode(args) if self._lora_enabled else False
        # Cross-call state (the backend instance is long-lived across update_weights_for_rollout).
        # Adapter-mode incremental-sync state (base-once + adapter-delta protocol) lives in the
        # shared LoraAdapterSync helper; the merge-mode fields below stay on the backend.
        self._lora_sync = self._megatron.LoraAdapterSync(args, model) if self._lora_adapter_mode else None
        self._lora_adapter_full = None  # merge mode: per-call {base_prefix: {"in","out"}} full tensors
        self._lora_skip_rollout_base = False  # adapter mode: set per-call once base is synced
        self._lora_moe_etp_checked = False  # guard the (actor-side) MoE ETP=1 assertion to run once

    @staticmethod
    def _rollout_topology_signature_of(rollout_topology: Dict[Any, Dict[str, Any]]) -> frozenset:
        """Build a stable signature of the rollout topology.

        Two topologies with the same signature describe the same set of engines
        (same rank, endpoint and per-engine GPU count, the latter affecting the
        NCCL group world size), so the existing engines + weight-update group
        can be reused without a teardown/rebuild.
        """
        sig = set()
        for rank, info in rollout_topology.items():
            meta = (info.get("metadata") or {}) if isinstance(info, dict) else {}
            sig.add(
                (
                    int(rank),
                    info.get("ip") if isinstance(info, dict) else None,
                    info.get("port") if isinstance(info, dict) else None,
                    meta.get("num_gpus_per_engine"),
                )
            )
        return frozenset(sig)

    def _create_rollout_engines(self, rollout_topology: Dict[int, Dict[str, Any]]) -> None:
        """Create Ray actors for each rollout node.

        Args:
            rollout_topology: Mapping of rank -> node_info (contains 'ip' and 'port').
        """
        logger.info(f"Creating {len(rollout_topology)} RolloutEngine actors...")
        for rank, node_info in rollout_topology.items():
            actor = RolloutEngine.options(**with_control_plane_affinity(self.args)).remote(int(rank), node_info)
            self.rollout_engines[int(rank)] = actor
            logger.info(f"Created RolloutEngine actor for rank {rank}")

    def _batch_request(self, endpoint: str, payload: Optional[Dict] = None, get_rank: bool = False) -> List[Any]:
        """Send HTTP requests to all rollout engines and collect futures.

        Args:
            endpoint: Endpoint path (e.g. '/init_weights_update_group').
            payload: Optional JSON payload to send.
            get_rank: If True, payload is expected to be a dict keyed by rank.

        Returns:
            List of Ray futures for the remote calls.
        """
        if not self.rollout_engines:
            logger.warning("No rollout engines available for batch request")
            return []

        futures = []
        for rank, engine in self.rollout_engines.items():
            if get_rank:
                payload_cur = payload.get(int(rank), {}) if payload else None
            else:
                payload_cur = payload
            future = engine.make_request.remote(endpoint, payload_cur)
            futures.append(future)
        return futures

    def _healthcheck_rollout_engines(self, timeout_seconds: int = 5) -> set[int]:
        # Allow raising the cold-start healthcheck timeout via env for slow starts.
        env_timeout = Envs.RELAX_ROLLOUT_HEALTHCHECK_TIMEOUT
        if env_timeout is not None:
            timeout_seconds = env_timeout
        failed_ranks = set()
        futures_to_rank = {}

        for rank, engine in list(self.rollout_engines.items()):
            try:
                future = engine.health.remote(timeout=float(timeout_seconds))
                futures_to_rank[future] = rank
            except Exception as e:
                logger.warning(f"RolloutEngine #{rank} failed to schedule healthcheck: {e}")
                failed_ranks.add(rank)

        if not futures_to_rank:
            return failed_ranks

        ready_futures, _ = ray.wait(
            list(futures_to_rank.keys()), timeout=timeout_seconds, num_returns=len(futures_to_rank)
        )

        for future in ready_futures:
            rank = futures_to_rank[future]
            try:
                ray.get(future)
            except Exception as e:
                logger.warning(f"RolloutEngine #{rank} healthcheck failed: {e}")
                failed_ranks.add(rank)

        for future in futures_to_rank:
            if future not in ready_futures:
                logger.warning(f"RolloutEngine #{futures_to_rank[future]} healthcheck timed out")
                failed_ranks.add(futures_to_rank[future])

        return failed_ranks

    def _remove_failed_engines(self, failed_ranks: set[int]) -> None:
        for rank in failed_ranks:
            if rank in self.rollout_engines:
                try:
                    ray.kill(self.rollout_engines[rank])
                except Exception as e:
                    logger.warning(f"Error killing failed RolloutEngine #{rank}: {e}")
                del self.rollout_engines[rank]

            for key in [str(rank), rank]:
                if key in self.rollout_topology:
                    del self.rollout_topology[key]

        if failed_ranks:
            logger.info(f"Removed {len(failed_ranks)} failed engines: {failed_ranks}")

    def _cleanup_rollout_engines(self) -> None:
        """Cleanup Ray actors for rollout communication."""
        for rank, actor in self.rollout_engines.items():
            try:
                ray.kill(actor)
                logger.debug(f"Killed RolloutEngine actor for rank {rank}")
            except Exception as e:
                logger.warning(f"Error killing RolloutEngine #{rank}: {e}")
        self.rollout_engines.clear()

    def _update_rollout_engines(self, max_retries: int = 30, retry_interval: float = 10.0) -> None:
        """Wait for rollout engines to be ready with retries.

        In fully-async mode, Rollout engines may still be initializing (loading model)
        when Actor attempts to sync weights. This method retries health checks until
        engines are ready, without modifying topology during retries.

        Args:
            max_retries: Maximum number of retry attempts (default: 30).
            retry_interval: Seconds to wait between retries (default: 10.0).

        Raises:
            RuntimeError: Only if **no** healthy engine remains after all
                retries. Engines that stay unhealthy are pruned and the weight
                sync proceeds with whatever engines are still healthy.
        """
        if not self.rollout_topology:
            raise RuntimeError("No rollout engines configured")

        for attempt in range(max_retries):
            failed_ranks = self._healthcheck_rollout_engines()

            if not failed_ranks:
                if attempt > 0:
                    logger.info(f"Rollout engines ready after {attempt + 1} attempts")
                return

            logger.warning(
                f"Healthcheck failed for engines: {failed_ranks}, "
                f"retrying in {retry_interval}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(retry_interval)

        # All retries exhausted. Prune whatever is still unhealthy and continue
        # with the engines that remain healthy, instead of unconditionally
        # raising. Only give up (and let the caller trigger recovery) when *no*
        # healthy engine is left. Previously this raised even when healthy
        # engines remained (e.g. a dead scale-out engine while the seed was
        # fine), which failed the actor weight sync and escalated to a full
        # global restart -- losing all training progress. Pruning here also
        # updates ``self.rollout_topology`` (via ``_remove_failed_engines``), so
        # the caller rebuilds the weight-update group over the surviving set.
        final_failed = self._healthcheck_rollout_engines()
        if final_failed:
            logger.error(f"Pruning engines still unhealthy after {max_retries} retries: {final_failed}")
            self._remove_failed_engines(final_failed)

        if not self.rollout_engines:
            raise RuntimeError(f"No healthy rollout engines available after {max_retries} retries")

        if final_failed:
            logger.warning(
                f"Proceeding with {len(self.rollout_engines)} healthy rollout engine(s) after pruning {final_failed}"
            )

    _MASTER_PORT_MIN = 11000
    _MASTER_PORT_MAX = 11999

    @staticmethod
    def _find_free_port_in_range(port_min: int, port_max: int) -> int:
        """Find a free port within [port_min, port_max] by attempting to bind.

        Raises RuntimeError if no free port is found in the range.
        """
        import random

        ports = list(range(port_min, port_max + 1))
        random.shuffle(ports)
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No free port available in range [{port_min}, {port_max}]")

    def init_process_group_for_rollout(self, topology_data: Optional[Dict] = None) -> None:
        """Initialize PyTorch distributed process group for rollout
        communication."""

        if self.role_info is None:
            raise RuntimeError("Role info not set. Cannot initialize process group.")
        self._is_pp_src_rank = (
            self._megatron.mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and self._megatron.mpu.get_tensor_model_parallel_rank() == 0
        )
        if self._is_pp_src_rank:
            pp_rank = self._megatron.mpu.get_pipeline_model_parallel_rank()
            master_address = ray._private.services.get_node_ip_address()
            self._group_name = f"slime-pp_{pp_rank}"

            if topology_data is None:
                raise RuntimeError("topology_data is required for init_process_group_for_rollout")

            self.rollout_topology = topology_data.get("nodes", {}).get("rollout", {})

            # Fast path: when the rollout topology is unchanged and the engines are
            # still healthy, reuse the existing proxy actors and NCCL weight-update
            # group. We skip create/destroy/sleep/reinit entirely, and crucially do
            # NOT send /destroy_weights_update_group or /init_weights_update_group to
            # the rollout side either, so both ends keep the same persistent group.
            new_sig = self._rollout_topology_signature_of(self.rollout_topology)
            if (
                self._model_update_groups is not None
                and new_sig == self._rollout_topology_signature
                and self.rollout_engines
                and not self._healthcheck_rollout_engines()
            ):
                logger.info("Reusing rollout weight-update group (topology unchanged).")
                return

            # Rebuild path (first update, topology change, or an unhealthy engine):
            # drop any stale engines/group before recreating, and invalidate the
            # signature until the new group is successfully established.
            if self.rollout_engines:
                self._cleanup_rollout_engines()
            self._rollout_topology_signature = None

            self._create_rollout_engines(self.rollout_topology)
            self._update_rollout_engines()

            if self._model_update_groups is not None:
                try:
                    logger.info("Destroying old process group...")
                    destroy_payload = {"group_name": self._group_name}
                    futures = self._batch_request("/destroy_weights_update_group", destroy_payload)
                    dist.destroy_process_group(self._model_update_groups)
                    ray.get(futures)
                    self._model_update_groups = None
                    # Wait for NCCL socket ports to be released by the OS
                    time.sleep(2.0)
                except Exception as e:
                    logger.warning(f"Error destroying old process group: {e}")
                    self._model_update_groups = None

            default_gpus = self.args.rollout_num_gpus_per_engine
            cumulative_offset = 1
            rank_offsets: dict[int, int] = {}
            for rank, role_info in sorted(self.rollout_topology.items(), key=lambda kv: int(kv[0])):
                metadata = role_info.get("metadata") if isinstance(role_info, dict) else {}
                gpus_for_node = (metadata or {}).get("num_gpus_per_engine", default_gpus)
                rank_offsets[int(rank)] = cumulative_offset
                cumulative_offset += gpus_for_node
            world_size = cumulative_offset

            max_retries = 3
            last_error = None
            for attempt in range(1, max_retries + 1):
                master_port = self._find_free_port_in_range(self._MASTER_PORT_MIN, self._MASTER_PORT_MAX)

                init_payloads = {}
                for rank, role_info in self.rollout_topology.items():
                    init_payloads[int(rank)] = {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": rank_offsets[int(rank)],
                        "world_size": world_size,
                        "group_name": self._group_name,
                        "backend": self.backend_type,
                    }

                logger.info(
                    f"Sending init_weights_update_group to {len(self.rollout_topology)} rollout nodes "
                    f"(attempt {attempt}/{max_retries}, port={master_port})..."
                )
                futures = self._batch_request("/init_weights_update_group", init_payloads, get_rank=True)

                try:
                    self._model_update_groups = init_process_group(
                        backend=self.backend_type,
                        init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
                        world_size=world_size,
                        rank=0,
                        group_name=self._group_name,
                        timeout=timedelta(seconds=180),
                    )
                    ray.get(futures)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Failed to init process group for rollout (attempt {attempt}/{max_retries}, "
                        f"port={master_port}): {e}",
                        exc_info=(attempt == max_retries),
                    )
                    self._model_update_groups = None
                    try:
                        ray.get(futures, timeout=5)
                    except Exception:
                        pass
                    if attempt < max_retries:
                        time.sleep(5.0 * attempt)

            if last_error is not None:
                raise RuntimeError(
                    f"Failed to init process group for rollout after {max_retries} attempts"
                ) from last_error

            # Group successfully (re)built — remember the topology it actually
            # serves so the next update with the same topology takes the reuse
            # fast path above. Recompute from ``self.rollout_topology`` rather
            # than the pre-check ``new_sig`` because ``_update_rollout_engines``
            # may have pruned dead engines from it; storing the stale full
            # signature would let the fast path reuse a group that is missing a
            # since-recovered engine (orphaning it).
            self._rollout_topology_signature = self._rollout_topology_signature_of(self.rollout_topology)

    def init_process_groups_for_actor_fwd_ref(self, topology_data) -> None:
        """Initialize process groups used for actor -> actor_fwd weight sync.

        This sets up deterministic groups so actor (source) ranks broadcast
        updated weights and actor_fwd ranks receive them.
        """
        if self.role_info is None:
            raise RuntimeError("Role info not set. Cannot initialize process group.")

        # Determine if this rank is the PP source rank (for weight gathering)
        self._is_pp_src_rank = (
            self._megatron.mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and self._megatron.mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = self._megatron.mpu.get_pipeline_model_parallel_rank()
        global_rank = topology_data.get("global_rank")
        pp_groups = topology_data.get("pp_groups")
        world_size = topology_data.get("world_size", 1)

        if self.role_info.role_name == "actor":
            # Actor side: PP source rank (rank 0) manages connection for this PP stage

            if self._is_pp_src_rank:
                if self._model_update_groups_for_actor_fwd_ref is not None:
                    # TODO: This case should not happen in current design since we only init once, but if we want to support dynamic re-init in the future we need to destroy old groups before creating new ones
                    return

                    # dist.destroy_process_group(self._model_update_groups_for_actor_fwd_ref)
                    # time.sleep(1)  # Ensure all ranks have destroyed before creating new group
                # This actor's PP rank 0 establishes the master address/port for this PP group
                group_name = f"update_actor_pp_{pp_rank}"
                init_method = pp_groups.get(group_name)
                # Create the process group for this actor PP stage
                # Rank 0 is always the actor PP source rank
                self._model_update_groups_for_actor_fwd_ref = init_process_group(
                    backend=self.backend_type,
                    init_method=init_method,
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                    timeout=timedelta(seconds=180),
                )
                logger.info(
                    f"Actor PP{pp_rank} initialized process group {group_name} "
                    f"(world_size={world_size}) at {init_method}"
                )

        else:  # actor_fwd side
            # Actor_fwd side: Each rank joins all groups corresponding to actor PP stages
            # to receive weights for inference reference model updates
            if self._model_update_groups_for_actor_fwd_ref is not None:
                return
                # for group_name, group in self._model_update_groups_for_actor_fwd_ref.items():
                #     dist.destroy_process_group(group)
                # time.sleep(1)  # Ensure all ranks have destroyed before creating new group

            self._model_update_groups_for_actor_fwd_ref = {}

            # Calculate this actor_fwd rank's position in the cluster

            # For each actor PP stage, join the corresponding group in deterministic order
            # This ensures all ranks call init_process_group in the same order
            for group_name, init_method in pp_groups.items():
                # Actor_fwd ranks join with rank = actor_fwd_rank + 1 (rank 0 is actor)
                group = init_process_group(
                    backend=self.backend_type,
                    init_method=init_method,
                    world_size=world_size,
                    rank=global_rank,
                    group_name=group_name,
                    timeout=timedelta(seconds=180),
                )
                self._model_update_groups_for_actor_fwd_ref[group_name] = group
                logger.info(
                    f"Actor_fwd PP{pp_rank} joined group {group_name} as rank {global_rank} (world_size={world_size})"
                )

    @torch.no_grad()
    def update_weights_for_rollout(self, rollout_only=False, actor_fwd_only=False) -> None:
        """Update weights used by rollout nodes.

        Sequence: pause rollout generation, flush caches, gather and broadcast
        model parameters (non-expert then expert), then resume generation.
        """
        self.weight_version += 1

        # LoRA: in merge mode, pre-gather every adapter to a full tensor so each base weight can
        # be folded before conversion (reuses the existing NCCL broadcast). In adapter mode, the
        # base is broadcast to rollout only on the first sync; afterwards only the adapter is
        # refreshed via /update_lora_from_distributed (see the adapter push below).
        if self._lora_merge_mode:
            self._assert_moe_etp_supported()
            self._lora_adapter_full = self._collect_full_adapter_tensors()
            # Grouped-expert (MoE) merge folds the adapter into each expert base weight. That
            # single merged base then serves BOTH consumers: the rollout (SGLang) path converts
            # it to HF, and the actor_fwd/reference path receives it raw via the EP-gathered
            # bucket. The reference model's own grouped-expert adapters are zeroed at init in
            # merge mode (see actor _init), so folding delta into the base does not double-count.
            if dist.get_rank() == 0:
                n_expert = sum(1 for prefix in self._lora_adapter_full if ".experts." in prefix)
                n_other = len(self._lora_adapter_full) - n_expert
                logger.info(
                    "[lora-merge] fully-async sync v=%d: folding %d non-expert + %d expert adapters "
                    "into base weights (rank0-local); reference gets merged base, rollout gets HF-converted",
                    self.weight_version,
                    n_other,
                    n_expert,
                )
        elif self._lora_adapter_mode and getattr(self.args, "num_experts", 0) and not rollout_only:
            # Adapter mode + MoE, off-policy (actor_fwd present): the rollout gets expert deltas as
            # a pure adapter via /update_lora_from_distributed, but actor_fwd has no adapter transport
            # of its own, so its grouped-expert delta is folded into the base (reuse the merge math;
            # ETP=1 / tp_size=1). Its own expert adapters are zeroed at init, so no double-count.
            self._assert_moe_etp_supported()
            self._lora_adapter_full = self._collect_full_adapter_tensors()
            if dist.get_rank() == 0:
                n_expert = sum(1 for prefix in self._lora_adapter_full if ".experts." in prefix)
                logger.info(
                    "[lora-adapter] fully-async sync v=%d: folding %d expert adapters into base for "
                    "actor_fwd (rollout gets the pure adapter via /update_lora_from_distributed)",
                    self.weight_version,
                    n_expert,
                )
        self._lora_skip_rollout_base = self._lora_adapter_mode and self._lora_sync.base_sync_done

        if not actor_fwd_only:
            if dist.get_rank() == 0:
                # Pause generation on all rollout nodes
                logger.info("Pausing generation on all rollout nodes...")
                ray.get(self._batch_request("/pause_generation"))

                # Flush cache on all rollout nodes
                logger.info("Flushing cache on all rollout nodes...")
                for rank, engine in self.rollout_engines.items():
                    ray.get(engine.flush_cache.remote())

            dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        origin_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights") if self._is_pp_src_rank else None

        for name, param in self._megatron.named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name,
                param,
                converted_named_tensors,
                origin_named_tensors,
                buffer_size,
                rollout_only,
                actor_fwd_only,
                pbar=pbar,
            )

        if converted_named_tensors or origin_named_tensors:
            if not rollout_only:
                self._update_bucket_weights_from_distributed_for_actor_fwd_ref(origin_named_tensors)
            if converted_named_tensors and not actor_fwd_only:
                self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
                converted_named_tensors.clear()
            origin_named_tensors.clear()
        dist.barrier(group=get_gloo_group())

        if self._lora_adapter_mode and getattr(self.args, "num_experts", 0):
            # Adapter mode + MoE: the expert delta reaches the rollout as a pure adapter via the
            # /update_lora_from_distributed push, so the expert base is converted for rollout only on
            # the first sync (base is frozen in LoRA training). actor_fwd/reference has no adapter
            # transport of its own, so every sync it receives the current expert adapter folded
            # into the base weight{N} (its own expert adapters are zeroed at init). Two passes,
            # each with the specialized rollout_only/actor_fwd_only flag, express the pure-vs-folded
            # divergence without changing the expert-bucket arity (merge/dense stay single-pass).
            if not actor_fwd_only and not self._lora_skip_rollout_base:
                self._run_expert_pass(rollout_only=True, actor_fwd_only=False, pbar=pbar)
            if not rollout_only:
                self._run_expert_pass(rollout_only=False, actor_fwd_only=True, pbar=pbar)
        else:
            self._run_expert_pass(rollout_only=rollout_only, actor_fwd_only=actor_fwd_only, pbar=pbar)
        dist.barrier(group=get_gloo_group())
        if not rollout_only:
            if dist.get_rank() == 0:
                payload = {
                    "names": [
                        "weight_updated_stop",
                    ],
                    "dtypes": [],
                    "shapes": [],
                    "group_name": "end",
                }
                logger.info("start post end send_weight_meta to actor fwd nodes...")
                response = self.http_client.post(
                    f"{self.coordinator_url}/send_weight_meta",
                    json=payload,
                )
                response.raise_for_status()
            dist.barrier(group=get_gloo_group())

        # Adapter mode: (re)register the trained LoRA adapter on every rollout engine. The base
        # was already broadcast above on the first sync; the adapter itself is broadcast over the
        # same NCCL group (no disk). Must run on ALL ranks (the export/gather inside are
        # collective); only rank 0 issues the HTTP fan-out and drives the broadcast.
        #
        # The failure is captured rather than propagated on the spot: generation is currently
        # PAUSED (see /pause_generation above) and only rank 0 can fail here, so an immediate
        # raise would strand the engines paused forever AND leave every other rank blocked in
        # the barrier below waiting for a rank that already unwound. The verdict is shared
        # across ranks first, generation is resumed, and only then does everyone raise together.
        push_error: Exception | None = None
        if self._lora_adapter_mode and not actor_fwd_only:
            try:
                self._push_lora_adapter_distributed(first_sync=not self._lora_sync.adapter_loaded)
            except Exception as e:  # noqa: BLE001 - re-raised below, after the engines are resumed
                logger.exception("LoRA adapter push failed; resuming generation before aborting")
                push_error = e
            else:
                self._lora_sync.adapter_loaded = True
                self._lora_sync.base_sync_done = True

        if not actor_fwd_only:
            # MAX over gloo: rank 0 is the only rank that can observe the push failure, so
            # without this the other ranks would take the success path and desync. Gated on
            # the (rank-uniform) adapter-mode flag so no other path pays for the collective.
            push_failed = False
            if self._lora_adapter_mode:
                flag = torch.tensor([1 if push_error is not None else 0], dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=get_gloo_group())
                push_failed = bool(flag.item())
            if dist.get_rank() == 0:
                # Continue generation on all rollout nodes
                logger.info("Resuming generation on all rollout nodes...")
                self._batch_request("/continue_generation")
            dist.barrier(group=get_gloo_group())
            if push_failed:
                # Abort rather than roll out with a stale/absent adapter: in adapter mode the
                # engine's base weights are frozen, so a dropped adapter means the rollout
                # policy silently stops tracking the trained one.
                raise RuntimeError(
                    "LoRA adapter push to the rollout engines failed; aborting the weight update "
                    "instead of generating with a stale or missing adapter. Generation has been "
                    "resumed so the engines are not left paused."
                ) from push_error
            # NOTE: rollout proxy actors are intentionally kept alive across weight
            # updates so init_process_group_for_rollout can reuse them (and the NCCL
            # group) when the topology is unchanged. They are torn down only when the
            # topology actually changes (inside init_process_group_for_rollout's
            # rebuild path).

        # Free the per-call merge-mode adapter cache (full tensors) before reclaiming CUDA memory.
        self._lora_adapter_full = None
        # Release fragmented CUDA reserved memory left behind by the
        # all_gather + HF-convert buffers that were allocated and freed
        # during the weight update loop.  Without this, the caching
        # allocator keeps large reserved blocks that are internally
        # fragmented, which can cause OOM when the optimizer later tries
        # to allocate contiguous Adam state buffers.
        device_utils.empty_cache()

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        origin_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        rollout_only=False,
        actor_fwd_only=False,
        pbar: tqdm | None = None,
    ) -> int | None:
        """Gather parameter across TP, convert to HF format and buffer it.

        Returns updated buffer size on the source rank, otherwise None.
        """
        param = self._megatron.all_gather_param(self.args, name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            if converted_named_tensors or origin_named_tensors:
                if not rollout_only:
                    self._update_bucket_weights_from_distributed_for_actor_fwd_ref(origin_named_tensors)
                if converted_named_tensors and not actor_fwd_only:
                    self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
                    converted_named_tensors.clear()
                origin_named_tensors.clear()
                buffer_size = 0
        origin_named_tensors += [(name, param)]
        if not actor_fwd_only and not self._lora_skip_rollout_base:
            if self._lora_enabled and is_lora_adapter_param(name):
                # Adapter params never go to the rollout via convert: merge mode folds them into
                # the base weight below; adapter mode pushes them via /update_lora_from_distributed.
                # (They ARE still sent raw to actor_fwd via origin_named_tensors above.)
                pass
            else:
                convert_param = param
                if self._lora_merge_mode:
                    convert_param = self._merge_full_base(name, param)
                if self._use_bridge:
                    converted_named_tensors += self._bridge_converter.convert(name, convert_param)
                else:
                    converted_named_tensors += self._megatron.convert_to_hf(
                        self.args, self.model_name, name, convert_param, self.quantization_config
                    )
        buffer_size += param_size
        return buffer_size

    def _run_expert_pass(self, *, rollout_only: bool, actor_fwd_only: bool, pbar: tqdm | None) -> None:
        """One full pass over the grouped-expert params: gather, (optionally)
        fold, bucket-flush.

        Extracted so adapter mode can run it twice (a pure-base rollout pass
        and a folded-base actor_fwd pass) while merge/dense keep a single pass.
        The ``rollout_only`` / ``actor_fwd_only`` flags thread through to the
        per-param handler and the bucket flush unchanged.
        """
        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []
        for name, param in self._megatron.named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, rollout_only, actor_fwd_only, pbar=pbar
            )
        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(
                named_tensors, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only, pbar=pbar
            )

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        rollout_only: bool = False,
        actor_fwd_only: bool = False,
        pbar: tqdm | None = None,
    ) -> int:
        """Gather expert parameter across expert-parallel group and buffer it.

        HF conversion is deferred until bucket flush.
        """
        if self._lora_enabled and is_lora_adapter_param(name):
            # Grouped-expert adapters never travel on their own:
            # - merge mode: folded into every base ``weight{N}`` below (serves rollout + actor_fwd).
            # - adapter mode: rollout gets the delta via the ``/update_lora_from_distributed`` push, and
            #   actor_fwd gets it folded into the base on its dedicated pass (see below). Either way
            #   the reference model's own expert adapters are zeroed at init (actor _init), so no
            #   double-count. Skip the raw adapter param here in both modes.
            return buffer_size
        param = self._megatron.all_gather_param(self.args, name, param)

        # Fold this expert's LoRA adapter into its base weight on the owning EP rank (ETP=1, so
        # tp_size=1 and no collective) before it enters the EP-gather + convert path, exactly as
        # the colocate HfWeightIteratorBridge does. Unpaired weights pass through.
        # - merge mode: always fold (the merged base serves both consumers).
        # - adapter mode: fold only on the actor_fwd pass (``actor_fwd_only``); the rollout pass
        #   ships the PURE base once and the delta reaches SGLang via ``/update_lora_from_distributed``.
        if self._lora_merge_mode or (self._lora_adapter_mode and actor_fwd_only):
            param = self._merge_full_base(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * self._megatron.mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            if named_tensors:
                self._update_expert_bucket_weights_from_distributed(
                    named_tensors, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only, pbar=pbar
                )
                buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        rollout_only: bool = False,
        actor_fwd_only: bool = False,
        pbar: tqdm | None = None,
    ) -> None:
        """Gather expert partitions, convert to HF format, and broadcast.

        Clears the input buffer when complete.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * self._megatron.mpu.get_expert_model_parallel_world_size()

        dist.all_gather_object(all_names, names, group=self._megatron.mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(self._megatron.mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=self.device)
                for _ in range(self._megatron.mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(
                params, param.data, group=self._megatron.mpu.get_expert_model_parallel_group(), async_op=True
            )
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        if not rollout_only:
            self._update_bucket_weights_from_distributed_for_actor_fwd_ref(all_gathered_params)
        if not actor_fwd_only:
            converted_hf_tensors = []
            for name, param in all_gathered_params:
                if self._use_bridge:
                    converted_hf_tensors += self._bridge_converter.convert(name, param)
                else:
                    converted_hf_tensors += self._megatron.convert_to_hf(
                        self.args, self.model_name, name, param, self.quantization_config
                    )
            self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)
            converted_hf_tensors.clear()
        all_gathered_params.clear()

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Broadcast a bucket of converted tensors to rollout nodes.

        A remote lock is acquired to avoid NCCL deadlocks during concurrent
        broadcasts. This function blocks until all broadcasts and remote
        updates complete.
        """

        while not ray.get(self.lock.acquire.remote()):
            time.sleep(0.1)
        # Prepare payload for weight update
        weight_payload = {
            "names": [name for name, _ in converted_named_tensors],
            "dtypes": [str(param.dtype).replace("torch.", "") for _, param in converted_named_tensors],
            "shapes": [param.shape for _, param in converted_named_tensors],
            "group_name": self._group_name,
            "weight_version": str(self.weight_version),
            "flush_cache": False,
        }
        # Send weight update to all rollout nodes via Ray actors
        futures = self._batch_request("/update_weights_from_distributed", weight_payload)

        # Broadcast weights via PyTorch distributed
        handles = []
        for _, param in converted_named_tensors:
            handles.append(dist.broadcast(param.data, 0, group=self._model_update_groups, async_op=True))
        for handle in handles:
            handle.wait()
        ray.get(futures)  # Ensure remote update completes

        ray.get(self.lock.release.remote())
        if pbar is not None:
            pbar.update(1)

    def _update_bucket_weights_from_distributed_for_actor_fwd_ref(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Broadcast weights to actor_fwd reference models using dist.

        Metadata describing names/shapes/dtypes is sent to a coordinator so
        receiving nodes can allocate buffers, then weights are broadcast using
        the process group set up for actor_fwd reception.
        """
        # Prepare metadata for weight transfer
        pp_rank = self._megatron.mpu.get_pipeline_model_parallel_rank()
        group_name = f"update_actor_pp_{pp_rank}"
        payload = {
            "names": [name for name, _ in named_tensors],
            "dtypes": [str(param.dtype).replace("torch.", "") for _, param in named_tensors],
            "shapes": [list(param.shape) for _, param in named_tensors],
            "group_name": group_name,
        }
        response = self.http_client.post(
            f"{self.coordinator_url}/send_weight_meta",
            json=payload,
        )
        response.raise_for_status()
        handles = []
        for _, param in named_tensors:
            handles.append(
                dist.broadcast(param.data, 0, group=self._model_update_groups_for_actor_fwd_ref, async_op=True)
            )
        for handle in handles:
            handle.wait()

    def recv_weight(self):
        """Poll coordinator for weight metadata and receive broadcasts.

        This method is intended for actor_fwd processes: it queries the
        coordinator for pending weight metadata, allocates receives, and then
        performs dist.broadcast to get actual tensors into local models. The
        loop ends when a special 'weight_updated_stop' marker is seen.
        """
        index = 0
        long_poll_wait_s = float(getattr(self.args, "dcs_recv_weight_meta_wait_timeout_s", 20.0))
        # Ensure read timeout is longer than long-poll wait duration.
        recv_timeout = httpx.Timeout(connect=5.0, read=max(long_poll_wait_s + 5.0, 10.0), write=30.0, pool=30.0)
        while True:
            try:
                response = self.http_client.get(
                    f"{self.coordinator_url}/recv_weight_meta",
                    params={"index": index, "wait_timeout_s": long_poll_wait_s},
                    timeout=recv_timeout,
                )
            except httpx.ReadTimeout:
                # Long-poll timed out without new metadata; continue waiting.
                continue
            response.raise_for_status()
            data = response.json()
            if not data:
                continue
            for metadata in data:
                index += 1
                names = metadata.get("names")
                # termination marker
                if names and names[0] == "weight_updated_stop":
                    dist.barrier(get_gloo_group())
                    if dist.get_rank() == 0:
                        response = self.http_client.get(f"{self.coordinator_url}/clear_weight_meta")
                        response.raise_for_status()

                    logger.info("Received final weight update marker for actor_fwd nodes")
                    return

                dtypes = metadata.get("dtypes")
                shapes = metadata.get("shapes")
                group_name = metadata.get("group_name")
                weights: list[tuple[str, torch.Tensor]] = []
                handles = []
                for name, dtype, shape in zip(names, dtypes, shapes):
                    target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                    weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                    handles.append(
                        torch.distributed.broadcast(
                            weight,
                            src=0,
                            group=self._model_update_groups_for_actor_fwd_ref[group_name],
                            async_op=True,
                        )
                    )
                    weights.append((name, weight))
                for handle in handles:
                    handle.wait()

                self._megatron.load_weight(self.args, self.model, weights)

    # ------------------------------------------------------------------
    # LoRA weight sync (fully-async)
    # ------------------------------------------------------------------

    def _assert_moe_etp_supported(self) -> None:
        """Fail loud (once) if MoE LoRA expert folding is combined with expert-
        TP > 1.

        Grouped-expert folding merges each adapter into its base locally on the
        owning EP rank with tp_size=1 (no expert-TP collective). This happens
        in merge mode always, and in adapter mode on the actor_fwd fold pass.
        Expert-TP > 1 would need an ETP all-gather reached by only the single
        owning rank -> deadlock; mirror the colocate constraint
        (HfWeightIteratorBridge.__init__). EP (expert-model-parallel) may still
        be > 1; dense models are unaffected. This lives on the actor-side push
        path (not __init__), because the SAME backend class is also constructed
        on rollout/SGLang engines where the Megatron expert groups are not
        initialized (get_expert_tensor_parallel_world_size() -> None).
        """
        if self._lora_moe_etp_checked or not getattr(self.args, "num_experts", 0):
            return
        etp = self._megatron.mpu.get_expert_tensor_parallel_world_size()
        assert etp == 1, (
            f"MoE LoRA expert folding requires --expert-tensor-parallel-size 1 (got {etp}). "
            "This applies to merge mode and to adapter mode when actor_fwd/reference is present "
            "(off-policy) — the actor_fwd expert delta is folded into the base with tp_size=1. "
            "Set ETP=1 (EP may stay > 1); adapter mode with ETP > 1 is only supported fully "
            "on-policy (no actor_fwd, so no fold)."
        )
        self._lora_moe_etp_checked = True

    def _collect_full_adapter_tensors(self) -> dict[str, dict[str, torch.Tensor]]:
        """TP-gather every LoRA adapter param to a FULL tensor, keyed by base-
        weight prefix.

        Merge mode uses this so each base weight can be folded with its adapter
        (tp_size=1, no further collective) inside the PP-src-only convert step.
        Adapter mode reuses it for the actor_fwd expert fold pass (experts
        only; non-expert entries are collected but never looked up there).
        Called on ALL ranks in identical param order at the top of
        ``update_weights_for_rollout`` so the TP all-gathers stay in lockstep.
        Adapter tensors are small; gathering them here (in addition to the main
        loop's gather for actor_fwd) is cheap.
        """
        adapter_full: dict[str, dict[str, torch.Tensor]] = {}
        for name, param in self._megatron.named_params_and_buffers(self.args, self.model):
            if not is_lora_adapter_param(name):
                continue
            if ".experts." in name or not getattr(param, "tensor_model_parallel", False):
                full = param.data
            else:
                full = self._megatron.all_gather_param(self.args, name, param)
            slot = adapter_full.setdefault(self._megatron.adapter_base_prefix(name), {})
            slot["in" if ".linear_in." in name else "out"] = full
        return adapter_full

    def _merge_full_base(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Return ``base + (alpha/dim)·(B @ A)`` as a NEW full tensor (merge mode).

        ``param`` is the already-TP-gathered full base weight; the paired adapter tensors were
        gathered to full in ``_collect_full_adapter_tensors``. Uses ``LoRAMerge`` with
        ``tp_size=1`` so no collective runs here (safe on the PP-src-only path). Never mutates
        ``param`` (which may alias the model's live weight when TP=1). Unpaired weights
        (layernorm/embed/router) pass through unchanged.

        For grouped MoE experts, ``name`` is a per-expert base ``...experts.<proj>.to_wrap.weight{N}``
        (global index N) while its adapter is a single grouped tensor on the owning EP rank. When
        that adapter is per-expert (3D) the local expert slice is
        selected before merging; a 2D shared adapter is used as-is by every local expert. Mirrors
        the colocate ``HfWeightIteratorBridge._merge_base_with_adapter`` (with tp_size=1 since
        MoE merge requires ETP=1).
        """
        slot = self._lora_adapter_full.get(self._megatron.base_param_prefix(name)) if self._lora_adapter_full else None
        if not slot or "in" not in slot or "out" not in slot:
            return param

        try:
            # Megatron-Bridge >= 0.6.0 moved LoRAMerge into its own module.
            from megatron.bridge.peft.lora_merge import LoRAMerge
        except ImportError:  # bridge <= 0.5.x
            from megatron.bridge.peft.lora import LoRAMerge

        linear_in = slot["in"].float()
        linear_out = slot["out"].float()
        if ".experts." in name and linear_in.ndim > 2:
            ep_size = self._megatron.mpu.get_expert_model_parallel_world_size()
            ep_rank = self._megatron.mpu.get_expert_model_parallel_rank()
            global_idx = int(re.search(r"weight(\d+)$", name).group(1))
            local_idx = global_idx - ep_rank * self.args.num_experts // ep_size
            linear_in = linear_in[local_idx]
            linear_out = linear_out[local_idx]

        return (
            LoRAMerge()
            .merge(
                param.float(),
                linear_out,
                linear_in,
                self.args.lora_alpha,
                self.args.lora_rank,
                tp_size=1,
                tp_group=None,
            )
            .to(param.dtype)
        )

    def _push_lora_adapter_distributed(self, *, first_sync: bool) -> None:
        """Export the HF adapter and broadcast it to every rollout SGLang
        engine over the existing NCCL weight-update group — no disk IO.

        Mirrors the base-weight NCCL path (``_update_bucket_weights_from_distributed``):
        rank 0 fans out metadata via ``_batch_request("/update_lora_from_distributed")``
        then broadcasts the adapter tensors ``src=0`` on ``self._model_update_groups``,
        which each engine receives and hands to its ``LoRAManager``. This removes the
        network-FS write (rank 0) + per-engine read (``/load_lora_adapter``) round-trip
        of the previous disk path. Same-name replacement is handled server-side.

        The broadcast is bucketed by ``--update-weight-buffer-size`` so peak device memory
        is one bucket rather than the whole (multi-GB, per-expert) adapter, on this rank and
        on every engine. The bucket boundaries travel in the payload as ``bucket_sizes``.

        Collective contract: every rank runs the export and the PP gather in lockstep;
        only rank 0 (== group src) issues the HTTP fan-out and the broadcast.
        """
        # Delta-skip: skip the whole push when no adapter param changed beyond threshold. MUST be
        # a collective decision (each rank owns different adapter shards) or the gather would hang.
        all_params = dict(self._megatron.named_params_and_buffers(self.args, self.model, convert_to_global_name=False))
        unchanged, new_state = self._lora_sync.should_skip(all_params)
        if not first_sync and unchanged:
            logger.debug("LoRA adapter unchanged on all ranks, skipping adapter push")
            self._lora_sync.prev_state = new_state
            return

        # Export full HF-format adapter tensors (bridge TP-gathers internally), then gather across
        # PP to rank 0 (== PP-rank 0, the group src) for a single broadcast; other ranks get None.
        local_adapter = self._lora_sync.export_local_adapter(all_params)
        merged = self._lora_sync.gather_full_adapter(local_adapter, all_gather=False)

        # Only rank 0 holds the full adapter and is src (rank 0) of self._model_update_groups
        # (== "slime-pp_0"); it drives the NCCL broadcast. Other ranks do not participate.
        if dist.get_rank() == 0:
            names = list(merged.keys())
            # Bucket like the base-weight path (_update_bucket_weights_from_distributed): a MoE
            # adapter is multi-GB (e.g. 40 layers x 256 experts x rank 32 ~= 3.4 GiB in BF16), so
            # staging the whole thing on the accelerator would spike this rank AND every receiving
            # engine by the full adapter size — on the engine side that lands on top of the KV
            # cache and OOMs mid-broadcast, which wedges every other participant until the NCCL
            # timeout. The engine buckets by the same explicit counts, so both ends stay in step.
            bucket_sizes = bucket_tensor_counts(
                [merged[name].numel() * merged[name].element_size() for name in names],
                self.args.update_weight_buffer_size,
            )

            # Serialize with the base-weight broadcast lock to avoid NCCL deadlocks.
            while not ray.get(self.lock.acquire.remote()):
                time.sleep(0.1)
            try:
                payload = {
                    "lora_name": LORA_ADAPTER_NAME,
                    "config_dict": self._lora_sync.config_dict(),
                    "names": names,
                    "dtypes": [str(merged[name].dtype).replace("torch.", "") for name in names],
                    "shapes": [list(merged[name].shape) for name in names],
                    "bucket_sizes": bucket_sizes,
                    "group_name": self._group_name,
                    "pinned": False,
                }
                # Fan out metadata (non-blocking) so every engine enters the broadcast recv,
                # then broadcast bucket by bucket, then confirm the remote loads completed.
                futures = self._batch_request("/update_lora_from_distributed", payload)
                offset = 0
                for count in bucket_sizes:
                    # NCCL needs contiguous device tensors; the bridge exports to CPU. Make
                    # contiguous BEFORE the transfer so a non-contiguous tensor costs a host
                    # copy rather than a second device allocation.
                    bucket = [
                        merged[name].contiguous().to(device=self.device) for name in names[offset : offset + count]
                    ]
                    handles = [dist.broadcast(t, 0, group=self._model_update_groups, async_op=True) for t in bucket]
                    for handle in handles:
                        handle.wait()
                    # Release this bucket's device memory before staging the next one.
                    handles.clear()
                    bucket.clear()
                    offset += count
                ray.get(futures)
                logger.info(
                    "[lora-adapter] broadcast %d tensors in %d bucket(s) over %s",
                    len(names),
                    len(bucket_sizes),
                    self._group_name,
                )
            finally:
                ray.get(self.lock.release.remote())
        self._lora_sync.prev_state = new_state


@ray.remote
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

    def health(self, timeout: float = 5.0) -> bool:
        response = requests.get(f"{self.base_url}/health_generate", timeout=timeout)
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
