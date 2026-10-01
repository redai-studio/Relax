# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import ray

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.core.service import create_placement_group
from relax.distributed.ray.multi_engine_manager import MultiEngineManager
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.utils.env import Envs
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
    return (
        int(getattr(args, "_teacher_bundle_start", args.rollout_num_gpus)) + bundle_offset + replica * gpus_per_replica
    )


def _build_teacher_engine_env(args) -> dict[str, str]:
    env_vars = dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1") | {
        # OPD patches default off; enabled only when the corresponding env flag is
        # passed through from the driver. RELAX_OPD_PREEXPANDED_PATCH affects the
        # teacher engine only; RELAX_OPD_PER_POS_TOKEN_IDS affects teacher + student.
        "RELAX_OPD_PREEXPANDED_PATCH": str(int(Envs.RELAX_OPD_PREEXPANDED_PATCH)),
        "RELAX_OPD_PER_POS_TOKEN_IDS": str(int(Envs.RELAX_OPD_PER_POS_TOKEN_IDS)),
        "RELAX_OPD_TOKEN_IDS_LOGPROB_K": Envs.RELAX_OPD_TOKEN_IDS_LOGPROB_K,
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


@ray.remote(concurrency_groups={"discovery": 4})
class TeacherManager(MultiEngineManager):
    """Launch and own Relax-managed OPD teacher SGLang engine(s)."""

    def __init__(
        self,
        args,
        num_replicas: int,
        gpus_per_replica: int,
        pg: tuple | None = None,
        shared_pg: bool = False,
        bundle_offset: int = 0,
    ) -> None:
        assert num_replicas >= 1, f"num_replicas must be >= 1, got {num_replicas}."
        assert gpus_per_replica > 0, f"gpus_per_replica must be > 0, got {gpus_per_replica}."
        if shared_pg:
            assert pg is not None, "shared_pg=True requires the full actor/rollout placement group."
            _pg, bundle_indices, gpu_ids = pg
            required = (
                int(getattr(args, "_teacher_bundle_start", args.rollout_num_gpus))
                + bundle_offset
                + gpus_per_replica * num_replicas
            )
            assert len(bundle_indices) >= required and len(gpu_ids) >= required, (
                f"shared teacher PG too small: bundles={len(bundle_indices)}, "
                f"gpu_ids={len(gpu_ids)}, required={required} (rollout_num_gpus={args.rollout_num_gpus} + "
                f"bundle_offset={bundle_offset} + gpus_per_replica={gpus_per_replica} * num_replicas={num_replicas})."
            )

        self.gpus_per_replica = gpus_per_replica
        self._replica_pgs: dict[int, tuple] = {}
        nodes_per_engine = max(1, gpus_per_replica // args.num_gpus_per_node)
        self.num_gpu_per_engine = min(gpus_per_replica, args.num_gpus_per_node)
        self._shared_pg = shared_pg
        self._shared_pg_tuple = pg
        self._bundle_offset = bundle_offset

        overrides = build_teacher_overrides(args, colocate_sync=shared_pg)
        self._overrides = overrides
        self._teacher_args = build_teacher_engine_args(args, overrides)
        logger.info(
            f"[OPD teacher] launching {num_replicas} replica(s), "
            f"TP={gpus_per_replica}, model={overrides['model_path']}, "
            f"shared_pg={shared_pg}, mem_fraction_static={overrides.get('mem_fraction_static')}"
        )

        super().__init__(
            args,
            num_slots=num_replicas * nodes_per_engine,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=SGLangEngine,
            log_prefix="[OPD teacher]",
            role="teacher",
            skip_init=getattr(args, "opd_teacher_defer", False),
        )
        if getattr(args, "opd_teacher_defer", False):
            self._onloaded = False
            self.inference.state = "sleeping"
            self.inference.phase = "teacher"

    def get_urls(self) -> list[str]:
        urls = []
        for engine in self.engines:
            if engine is None:
                continue
            base_url = ray.get(engine.get_url.remote())
            urls.append(f"{base_url}/generate")
        return urls

    def _resolve_placement(self, rank: int):
        replica = rank // self.nodes_per_engine
        node_offset = (rank % self.nodes_per_engine) * self.num_gpu_per_engine
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
            return self._shared_pg_tuple, False, gpu_index + node_offset

        # Dedicated: this replica creates and owns its own placement group.
        if replica not in self._replica_pgs:
            self._replica_pgs[replica] = create_placement_group(
                num_gpus=self.gpus_per_replica,
                node_group_affinity=getattr(self.args, "enable_affinity", True),
            )
        return self._replica_pgs[replica], True, node_offset

    def _remove_owned_pg(self, rank: int) -> None:
        super()._remove_owned_pg(rank)
        replica = rank // self.nodes_per_engine
        if not any(i // self.nodes_per_engine == replica for i in self._engine_placements):
            self._replica_pgs.pop(replica, None)

    def _ray_resource_kwargs(self, rank: int) -> dict:
        fraction = getattr(self.args, "_inference_ray_gpu_fraction", 0.2)
        return {"num_cpus": fraction, "num_gpus": fraction}

    def _build_engine_env_vars(self) -> dict[str, str]:
        return _build_teacher_engine_env(self.args)

    def _engine_ctor_args(self, rank: int):
        return self._teacher_args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        return {
            "role": "teacher",
            "sglang_overrides": self._overrides,
            "num_gpus_per_engine": self.gpus_per_replica,
            "register_sigterm_handler": False,
        }

    def _build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        # The teacher is standalone: do NOT register to the rollout router and
        # do NOT register to DCS (it receives no weight sync).
        return {
            **addr_and_ports,
            "router_ip": None,
            "router_port": None,
            "skip_dcs_registration": True,
            "skip_router_registration": True,
        }

    def _allocate_engine_addr_and_ports(self, *, new_engines: list[tuple]) -> dict[int, dict]:
        from relax.distributed.ray.inference_ports import allocate_inference_ports

        # Legacy clients cache raw URLs. Planned deployments use discovery and
        # may safely publish a replacement endpoint after recovery.
        if not getattr(getattr(self, "args", None), "_inference_placement", None):
            cached = getattr(self, "_engine_addr_and_ports", {})
            if self._shared_pg and all(rank in cached for rank, _ in new_engines):
                return {rank: dict(cached[rank]) for rank, _ in new_engines}
        addresses, _ = allocate_inference_ports(
            new_engines,
            nodes_per_engine=self.nodes_per_engine,
            base_port=30000 + getattr(self.args, "_teacher_port_window", 0) * 1000,
            max_port=30999 + getattr(self.args, "_teacher_port_window", 0) * 1000,
            dp_size=self._teacher_args.sglang_dp_size,
        )
        return addresses

    def recover(self) -> set:
        if (
            not getattr(getattr(self, "args", None), "_inference_placement", None)
            and not self._shared_pg
            and any(engine is None for engine in self.all_engines)
        ):
            raise RuntimeError("Legacy dedicated Teacher URLs require a global restart after engine failure")
        return super().recover()
