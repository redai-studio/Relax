# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Manager for Generative Reward Model Service.

This module implements a simplified manager for genRM engines, built on top of
``MultiEngineManager`` (parallel bring-up, health check, dead-engine recovery,
onload/offload) with GenRM-specific placement and engine wiring.
"""

import logging

import ray

from relax.backends.sglang.sglang_engine import GenRMEngine
from relax.core.node_group_affinity import with_control_plane_affinity
from relax.distributed.ray.engine_pool import _allocate_genrm_engine_addr_and_ports
from relax.distributed.ray.multi_engine_manager import MultiEngineManager, _is_engine_dead  # noqa: F401
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from relax.utils.http_utils import init_http_client


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class GenRMManager(MultiEngineManager):
    """Manager for GenRM engines.

    This is a simplified version of RolloutManager focused on:
    - Initializing genRM engines
    - Health checking
    - Onload/offload operations
    """

    def __init__(self, args, pg, bundle_offset: int = 0, port_window_index: int = 0, *, dedicated_pg: bool = False):
        init_http_client(args)

        num_gpu_per_engine = min(args.genrm_num_gpus_per_engine, args.num_gpus_per_node)
        num_slots = 0 if args.debug_train_only else args.genrm_num_gpus // num_gpu_per_engine
        nodes_per_engine = max(1, args.genrm_num_gpus_per_engine // args.num_gpus_per_node)

        self.pg = pg
        self.dedicated_pg = dedicated_pg
        self.num_gpu_per_engine = num_gpu_per_engine
        self.bundle_offset = bundle_offset
        self.port_window_index = port_window_index

        super().__init__(
            args,
            num_slots=num_slots,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=GenRMEngine,
            skip_init=args.debug_train_only,
            log_prefix="GenRM",
        )
        self.num_new_engines = len(self.engines)
        self.genrm_engine_lock = Lock.options(
            **with_control_plane_affinity(self.args, {"num_cpus": 1, "num_gpus": 0})
        ).remote()

    def get_genrm_engines_and_lock(self):
        return self.engines, self.genrm_engine_lock, self.num_new_engines

    def get_engine_hosts_ports(self):
        """Return a list of (host, port) tuples for each live genRM engine.

        This is used by the GenRM service to send HTTP generation requests
        directly to the underlying SGLang servers.

        The host/port information is captured during engine initialization
        from the addr_and_ports dict passed to ``_allocate_engine_addr_and_ports``.

        ``all_engines`` holds one entry per *node*, so a multi-node engine
        (genrm_num_gpus_per_engine > num_gpus_per_node) occupies
        ``nodes_per_engine`` consecutive ranks. Only node_rank 0 of each group
        runs the SGLang HTTP server -- the followers are compute-only workers
        and answer /generate with 404 -- so stride the same way the
        ``engines`` property does and return head nodes only.

        The list is also compacted over dead engines, so callers must swap it
        and any derived round-robin state together.
        """
        results = []
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            engine = self.all_engines[rank]
            if engine is not None and rank not in self._unready_ranks and rank in self._engine_addr_and_ports:
                info = self._engine_addr_and_ports[rank]
                results.append((info["host"], info["port"]))
        return results

    # ------------------------------------------------------------------
    # MultiEngineManager hooks.
    # ------------------------------------------------------------------

    def _resolve_placement(self, rank):
        gpu_idx = rank * self.num_gpu_per_engine + self.bundle_offset
        shared_with_rollout = getattr(self.args, "_genrm_colocate_with_rollout", False)
        if not self.args.fully_async and not shared_with_rollout and not self.dedicated_pg:
            gpu_idx += self.args.rollout_num_gpus

        return self.pg, False, gpu_idx

    def _ray_resource_kwargs(self, rank):
        # Lower default fractional-GPU footprint when sharing bundles with
        # rollout (rollout uses 0.2 per actor; 0.2 + 0.2 risks Ray scheduler
        # rejection).
        shared_with_rollout = getattr(self.args, "_genrm_colocate_with_rollout", False)
        default_ray_num_gpus = 0.1 if shared_with_rollout else 0.2
        num_gpus = getattr(self.args, "genrm_ray_num_gpus", default_ray_num_gpus)
        return {"num_cpus": num_gpus, "num_gpus": num_gpus}

    def _build_engine_env_vars(self):
        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            # See rollout.py: recent SGLang reads SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK
            # (default True) and the deprecation shim value-copies SGL_DISABLE_* into it,
            # so the old DISABLE vars re-enable the check. Set ENABLE=false directly.
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            # NOTE: disable custom all-reduce-v2, same as rollout.py — avoids
            # custom_all_reduce.cuh:37: CUDA error: invalid argument during CUDA graph capture.
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
        }
        if getattr(self.args, "fp16", False):
            env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
        return env_vars

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        return _allocate_genrm_engine_addr_and_ports(
            args=self.args,
            new_engines=new_engines,
            port_window_index=self.port_window_index,
        )
