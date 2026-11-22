# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local static engine lifecycle and role-specific launch configuration."""

from typing import Any, Optional

import ray

from relax.distributed.ray.multi_engine_manager import _ENGINE_SHUTDOWN_TIMEOUT_S, MultiEngineManager
from relax.distributed.ray.policy_engine_pool import _allocate_rollout_engine_addr_and_ports_normal
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.utils.env import Envs
from relax.utils.http_utils import find_available_port, init_http_client
from relax.utils.logging_utils import get_logger
from relax.utils.opd.opd_utils import build_teacher_engine_args, build_teacher_overrides


logger = get_logger(__name__)


def _resolve_teacher_gpu_index(
    *, args, replica: int, gpus_per_replica: int, shared_pg: bool, bundle_offset: int = 0
) -> int:
    if not shared_pg:
        # Dedicated (per-teacher own PG) path: each replica creates its OWN
        # placement group of size gpus_per_replica (see _resolve_placement), so the
        # index is always 0 within that per-replica PG — a replica*gpus_per_replica
        # offset would overflow it (only valid when all replicas share one big PG).
        return 0
    # Shared (colocate) actor PG: rollout lives at the front [0, rollout_num_gpus);
    # teachers occupy the bundles after it. ``bundle_offset`` is this teacher's
    # slice start within the teacher region so multiple teachers (MOPD) sharing the
    # one actor PG do not collide.
    return int(args.rollout_num_gpus) + bundle_offset + replica * gpus_per_replica


def _build_teacher_engine_env(args) -> dict[str, str]:
    env_vars = dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1") | {
        # Native multimodal OPD must preserve the supplied response token IDs.
        "SGLANG_MM_AVOID_RETOKENIZE": "true",
        "RELAX_OPD_PER_POS_TOKEN_IDS": str(int(Envs.RELAX_OPD_PER_POS_TOKEN_IDS)),
        "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
    }
    if getattr(args, "fp16", False):
        env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
    return env_vars


_GENRM_PORT_BASE = 16000
_GENRM_PORT_WINDOW_SIZE = 1000
_MAX_PORT = 65535


def _allocate_genrm_engine_addr_and_ports(*, args, new_engines, port_window_index=0):
    """Allocate network addresses and ports for genRM engines.

    Similar to _allocate_rollout_engine_addr_and_ports_normal but for genRM.

    ``port_window_index`` selects a disjoint range per GenRM instance (multi-instance
    --genrm-instances): every model has its own engine pool that probes
    for free ports independently and in parallel during Serve replica
    initialization, so two instances starting from the same base port can both
    see a given port as free (probe-then-bind race) and then collide when
    their SGLang servers actually bind it.
    """
    window_start = _GENRM_PORT_BASE + port_window_index * _GENRM_PORT_WINDOW_SIZE
    window_end = window_start + _GENRM_PORT_WINDOW_SIZE - 1
    if port_window_index < 0 or window_end > _MAX_PORT:
        raise ValueError(
            f"GenRM port window index {port_window_index} is out of range; "
            f"at most {(_MAX_PORT - _GENRM_PORT_BASE + 1) // _GENRM_PORT_WINDOW_SIZE} instances are supported."
        )

    num_engines_per_node = max(1, min(args.num_gpus_per_node, args.genrm_num_gpus) // args.genrm_num_gpus_per_engine)
    addr_and_ports = {}

    visited_nodes = set()
    for rank, engine in new_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # Use a different, bounded port range from rollout (15000).
            start_port = window_start

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                        max_port=window_end,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

        if args.genrm_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.genrm_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # First node in the engine, allocate dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports.setdefault(rank + i, {})
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for rank, _ in new_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[rank], f"GenRM engine rank={rank} {key} is not set."
        logger.info(f"Ports for genRM engine rank={rank}: {addr_and_ports[rank]}")

    return addr_and_ports


class EnginePool(MultiEngineManager):
    """Local engine lifecycle shared by static inference roles; never a Ray
    owner."""

    def __init__(self, role: str, args: Any, binding: Any, options: dict[str, Any]) -> None:
        self.role = role
        self.router = None
        if role == "teacher":
            options = dict(options, pg=binding)
        if role == "genrm":
            self._init_genrm(args, binding, **options)
        elif role == "teacher":
            self._init_teacher(args, **options)
        else:
            raise ValueError(f"Unsupported static engine role: {role}")

    def _init_genrm(self, args, pg, bundle_offset: int = 0, port_window_index: int = 0, *, dedicated_pg: bool = False):
        from relax.backends.sglang.sglang_engine import SGLangEngine

        init_http_client(args)

        num_gpu_per_engine = min(args.genrm_num_gpus_per_engine, args.num_gpus_per_node)
        num_slots = 0 if args.debug_train_only else args.genrm_num_gpus // num_gpu_per_engine
        nodes_per_engine = max(1, args.genrm_num_gpus_per_engine // args.num_gpus_per_node)

        self.pg = pg
        self.dedicated_pg = dedicated_pg
        self.num_gpu_per_engine = num_gpu_per_engine
        self.bundle_offset = bundle_offset
        self.port_window_index = port_window_index

        self._initialize(
            args,
            num_slots=num_slots,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=SGLangEngine,
            skip_init=args.debug_train_only,
            log_prefix="GenRM",
        )
        self.num_new_engines = len(self.engines)

    def _init_teacher(
        self,
        args,
        num_replicas: int,
        gpus_per_replica: int,
        pg: tuple | None = None,
        shared_pg: bool = False,
        bundle_offset: int = 0,
    ) -> None:
        from relax.backends.sglang.sglang_engine import SGLangEngine

        assert num_replicas >= 1, f"num_replicas must be >= 1, got {num_replicas}."
        assert gpus_per_replica > 0, f"gpus_per_replica must be > 0, got {gpus_per_replica}."
        if gpus_per_replica > args.num_gpus_per_node:
            assert gpus_per_replica % args.num_gpus_per_node == 0, (
                f"Multi-node teacher GPUs ({gpus_per_replica}) must be divisible by "
                f"num_gpus_per_node ({args.num_gpus_per_node})."
            )
        nodes_per_engine = max(1, gpus_per_replica // args.num_gpus_per_node)
        if shared_pg:
            assert pg is not None, "shared_pg=True requires the full actor/rollout placement group."
            _pg, bundle_indices, gpu_ids = pg
            required = int(args.rollout_num_gpus) + bundle_offset + gpus_per_replica * num_replicas
            assert len(bundle_indices) >= required and len(gpu_ids) >= required, (
                f"shared teacher PG too small: bundles={len(bundle_indices)}, "
                f"gpu_ids={len(gpu_ids)}, required={required} (rollout_num_gpus={args.rollout_num_gpus} + "
                f"bundle_offset={bundle_offset} + gpus_per_replica={gpus_per_replica} * num_replicas={num_replicas})."
            )

        self.gpus_per_replica = gpus_per_replica
        self._shared_pg = shared_pg
        self._shared_pg_tuple = pg
        self._bundle_offset = bundle_offset

        overrides = build_teacher_overrides(args, colocate_sync=shared_pg)
        self._overrides = overrides
        self._teacher_args = build_teacher_engine_args(args, overrides)
        logger.info(
            f"[OPD teacher] launching {num_replicas} replica(s), "
            f"TP={gpus_per_replica}, nodes_per_engine={nodes_per_engine}, model={overrides['model_path']}, "
            f"shared_pg={shared_pg}, mem_fraction_static={overrides.get('mem_fraction_static')}"
        )

        self._initialize(
            args,
            num_slots=num_replicas * nodes_per_engine,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=SGLangEngine,
            log_prefix="[OPD teacher]",
        )

    def get_urls(self) -> list[str]:
        return [f"{ray.get(engine.get_url.remote())}/generate" for engine in self.engines if engine is not None]

    def get_engine_hosts_ports(self) -> list[tuple[str, int]]:
        return [
            (info["host"], info["port"])
            for rank, info in sorted(self._engine_addr_and_ports.items())
            if rank % self.nodes_per_engine == 0
            and self.all_engines[rank] is not None
            and rank not in self._unready_ranks
        ]

    def _initialize(
        self,
        args: Any,
        *,
        num_slots: int,
        nodes_per_engine: int = 1,
        engine_actor_cls: type,
        skip_init: bool = False,
        log_prefix: str = "",
    ) -> None:
        super().__init__(
            args,
            num_slots=num_slots,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=engine_actor_cls,
            skip_init=skip_init,
            log_prefix=log_prefix,
        )
        if getattr(self, "role", None) in {"teacher", "genrm"} and hasattr(self, "router") and not skip_init:
            from relax.distributed.ray.model_router import ModelRouter

            self.router = ModelRouter()
            self.router.start(args)
            self.router.activate(self.engines)

    def _resolve_placement(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_resolve_placement(*args, **kwargs)
        return self._genrm_resolve_placement(*args, **kwargs)

    def _teacher_resolve_placement(self, rank: int):
        replica = rank // self.nodes_per_engine
        node_gpu_offset = (rank % self.nodes_per_engine) * self.args.num_gpus_per_node
        if self._shared_pg:
            # Colocate: teachers share the actor placement group, which the
            # controller owns and removes → owns_pg=False.
            gpu_index = _resolve_teacher_gpu_index(
                args=self.args,
                replica=replica,
                gpus_per_replica=self.gpus_per_replica,
                shared_pg=True,
                bundle_offset=self._bundle_offset,
            )
            return self._shared_pg_tuple, False, gpu_index + node_gpu_offset

        # Dedicated static models borrow the task's per-model binding. Every
        # node of a logical replica occupies its own full-node slice; only the
        # head exposes an HTTP endpoint.
        if self._shared_pg_tuple is None:
            raise ValueError("Dedicated teacher requires a model placement binding")
        return self._shared_pg_tuple, False, self._bundle_offset + replica * self.gpus_per_replica + node_gpu_offset

    def _genrm_resolve_placement(self, rank):
        gpu_idx = rank * self.num_gpu_per_engine + self.bundle_offset
        shared_with_rollout = getattr(self.args, "_genrm_colocate_with_rollout", False)
        if not self.args.fully_async and not shared_with_rollout and not self.dedicated_pg:
            gpu_idx += self.args.rollout_num_gpus

        return self.pg, False, gpu_idx

    def _ray_resource_kwargs(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_ray_resource_kwargs(*args, **kwargs)
        return self._genrm_ray_resource_kwargs(*args, **kwargs)

    def _teacher_ray_resource_kwargs(self, rank: int) -> dict:
        return {"num_cpus": 0.2, "num_gpus": 0.2}

    def _genrm_ray_resource_kwargs(self, rank):
        # Lower default fractional-GPU footprint when sharing bundles with
        # rollout (rollout uses 0.2 per actor; 0.2 + 0.2 risks Ray scheduler
        # rejection).
        shared_with_rollout = getattr(self.args, "_genrm_colocate_with_rollout", False)
        default_ray_num_gpus = 0.1 if shared_with_rollout else 0.2
        num_gpus = getattr(self.args, "genrm_ray_num_gpus", default_ray_num_gpus)
        return {"num_cpus": num_gpus, "num_gpus": num_gpus}

    def _allocate_engine_addr_and_ports(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_allocate_engine_addr_and_ports(*args, **kwargs)
        return self._genrm_allocate_engine_addr_and_ports(*args, **kwargs)

    def _teacher_allocate_engine_addr_and_ports(self, *, new_engines: list[tuple]) -> dict[int, dict]:
        addr_and_ports: dict[int, dict] = {}
        unallocated_replicas: dict[int, list[tuple]] = {}
        for rank, engine in new_engines:
            # OPD consumers receive teacher URLs once during startup. Preserve
            # the original endpoint across recovery instead of silently moving
            # a rebuilt engine to a port those consumers never learn about.
            if self._shared_pg and rank in self._engine_addr_and_ports:
                addr_and_ports[rank] = dict(self._engine_addr_and_ports[rank])
                continue
            unallocated_replicas.setdefault(rank // self.nodes_per_engine, []).append((rank, engine))
        next_port = 15000
        for replica, engines in unallocated_replicas.items():
            base_port = find_available_port(next_port)
            per_engine_addr_and_ports, cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self._teacher_args,
                rollout_engines=engines,
                worker_type="regular",
                num_gpus_per_engine=self.gpus_per_replica,
                rank_offset=replica * self.nodes_per_engine,
                base_port=base_port,
            )
            addr_and_ports.update({rank: per_engine_addr_and_ports[rank] for rank, _ in engines})
            next_port = max(cursors.values(), default=base_port)
        return addr_and_ports

    def _genrm_allocate_engine_addr_and_ports(self, *, new_engines):
        return _allocate_genrm_engine_addr_and_ports(
            args=self.args,
            new_engines=new_engines,
            port_window_index=self.port_window_index,
        )

    def _build_engine_env_vars(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_build_engine_env_vars(*args, **kwargs)
        return self._genrm_build_engine_env_vars(*args, **kwargs)

    def _teacher_build_engine_env_vars(self) -> dict[str, str]:
        return _build_teacher_engine_env(self.args)

    def _genrm_build_engine_env_vars(self):
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

    def _engine_ctor_args(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_engine_ctor_args(*args, **kwargs)
        return self._genrm_engine_ctor_args(*args, **kwargs)

    def _teacher_engine_ctor_args(self, rank: int):
        return self._teacher_args

    def _genrm_engine_ctor_args(self, rank: int) -> Any:
        """First positional argument passed to the engine actor constructor."""
        return self.args

    def _build_engine_ctor_kwargs(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_build_engine_ctor_kwargs(*args, **kwargs)
        return self._genrm_build_engine_ctor_kwargs(*args, **kwargs)

    def _teacher_build_engine_ctor_kwargs(self, rank: int) -> dict:
        return {
            "role": "teacher",
            "sglang_overrides": self._overrides,
            "num_gpus_per_engine": self.gpus_per_replica,
            "register_sigterm_handler": False,
        }

    def _genrm_build_engine_ctor_kwargs(self, rank: int) -> dict:
        """Extra keyword arguments passed to the engine actor constructor
        (beyond rank/worker_type/base_gpu_id)."""
        return {
            "role": "genrm",
        }

    def _build_engine_init_kwargs(self, *args, **kwargs):
        if self.role == "teacher":
            return self._teacher_build_engine_init_kwargs(*args, **kwargs)
        return self._genrm_build_engine_init_kwargs(*args, **kwargs)

    def _teacher_build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        # The teacher is standalone: do NOT register to the rollout router and
        # do NOT register to DCS (it receives no weight sync).
        return {
            **addr_and_ports,
            "router_ip": None,
            "router_port": None,
            "skip_dcs_registration": True,
            "skip_router_registration": True,
        }

    def _genrm_build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        """Keyword arguments passed to ``engine.init.remote(...)``."""
        return dict(addr_and_ports)

    def onload(self, tags: Optional[list[str]] = None) -> None:
        """Load engine weights to GPU.

        Also the recovery point for engines that died since the last step: a
        freshly built engine comes up onloaded, which is exactly the state this
        phase wants.
        """
        self._ensure_router_running()
        rebuilt = self.recover()
        if self._onloaded and tags is None:
            if getattr(self, "router", None) is not None:
                self.router.activate(self.engines)
            logger.info(f"{self._log_prefix} engines already onloaded; skipping")
            return
        logger.info(f"{self._log_prefix} engines onload started with tags={tags}")
        # Engines rebuilt just above are already onloaded -- resuming them
        # again would be a double-resume, so only touch the ones that survived.
        dead = self._fanout("resume_memory_occupation", skip_ranks=rebuilt, tags=tags)
        if dead:
            # An engine that died while offloaded is only discovered here
            # (offload() short-circuits when already offloaded), so it missed
            # the recover() above. Rebuild now rather than leaving the pool a
            # man down for the whole next phase.
            self._retire_engines(dead)
            self.recover()
        if getattr(self, "router", None) is not None:
            self.router.activate(self.engines)
        self._onloaded = True
        logger.info(f"{self._log_prefix} engines onload completed")

    def offload(self) -> None:
        """Offload engine weights from GPU to free memory.

        Dead engines are retired here but NOT rebuilt: this typically runs
        while other ranks wait on a barrier, so keep it short and leave the
        rebuild to the next onload().
        """
        if getattr(self, "router", None) is not None:
            if not self.router.is_healthy():
                self._close_router()
            else:
                try:
                    self.router.withdraw(self.engines)
                except ray.exceptions.ActorDiedError:
                    self._close_router()
        pending = [rank for rank in self._unready_ranks if self.all_engines[rank] is not None]
        if pending:
            self._retire_engines(pending)
        if not self._onloaded:
            logger.info(f"{self._log_prefix} engines already offloaded; skipping")
            return
        logger.info(f"{self._log_prefix} engines offload started")
        dead = self._fanout("release_memory_occupation")
        self._retire_engines(dead)
        # Unconditional: the surviving engines did release, so the manager
        # must not claim to still be onloaded just because one engine died.
        self._onloaded = False
        logger.info(f"{self._log_prefix} engines offload completed (retired {len(dead)} dead)")

    def recover(self) -> set[int]:
        rebuilt = super().recover()
        if rebuilt and self._onloaded and getattr(self, "router", None) is not None:
            self.router.activate(self.engines)
        return rebuilt

    def shutdown(self) -> None:
        if getattr(self, "router", None) is not None:
            self._close_router()
        super().shutdown()

    def _close_router(self) -> None:
        endpoint = self.router.endpoint
        self.router.close()
        # Only detach registrations after the owning process has exited.
        for engine in self.all_engines[:: self.nodes_per_engine]:
            if engine is not None and endpoint is not None:
                try:
                    ray.get(engine.detach_model_router.remote(*endpoint), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                except ray.exceptions.ActorDiedError:
                    pass  # Retirement independently confirms the server session's exit.
        self.router.endpoint = None

    def _ensure_router_running(self) -> None:
        router = getattr(self, "router", None)
        if router is not None and not router.is_healthy():
            self._close_router()
            router.start(self.args)
