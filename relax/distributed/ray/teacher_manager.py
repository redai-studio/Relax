# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import ray
from ray.util.placement_group import remove_placement_group

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.core.service import create_placement_group
from relax.distributed.ray.multi_engine_manager import (
    SNAPSHOT_CONCURRENCY_GROUP,
    SNAPSHOT_CONCURRENCY_GROUPS,
    MultiEngineManager,
)
from relax.distributed.ray.placement_physical import validate_engine_bundles
from relax.distributed.ray.placement_planner import ACTOR_POOL
from relax.distributed.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.utils.env import Envs
from relax.utils.http_utils import find_available_port
from relax.utils.logging_utils import get_logger
from relax.utils.opd.opd_utils import build_teacher_engine_args, build_teacher_overrides


logger = get_logger(__name__)


def _resolve_teacher_gpu_index(*, replica: int, gpus_per_replica: int, shared_pg: bool, bundle_offset: int = 0) -> int:
    if not shared_pg:
        # Dedicated (per-teacher own PG) path: each replica creates its OWN
        # placement group of size gpus_per_replica (see _resolve_placement), so the
        # index is always 0 within that per-replica PG — a replica*gpus_per_replica
        # offset would overflow it (only valid when all replicas share one big PG).
        return 0
    # Shared (colocate) actor PG: ``bundle_offset`` is the absolute index of this
    # teacher's first bundle, as planned by placement_planner.plan_placement --
    # after the rollout region and after any earlier teacher (MOPD) sharing the
    # one actor PG.
    return bundle_offset + replica * gpus_per_replica


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


@ray.remote(concurrency_groups=SNAPSHOT_CONCURRENCY_GROUPS)
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
            required = bundle_offset + gpus_per_replica * num_replicas
            assert len(bundle_indices) >= required and len(gpu_ids) >= required, (
                f"shared teacher PG too small: bundles={len(bundle_indices)}, "
                f"gpu_ids={len(gpu_ids)}, required={required} "
                f"(bundle_offset={bundle_offset} + gpus_per_replica={gpus_per_replica} * num_replicas={num_replicas})."
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
            f"TP={gpus_per_replica}, model={overrides['model_path']}, "
            f"shared_pg={shared_pg}, mem_fraction_static={overrides.get('mem_fraction_static')}"
        )

        super().__init__(
            args,
            num_slots=num_replicas,
            nodes_per_engine=1,
            engine_actor_cls=SGLangEngine,
            log_prefix="[OPD teacher]",
        )

    @ray.method(concurrency_group=SNAPSHOT_CONCURRENCY_GROUP)
    def get_inference_snapshot(self) -> dict:
        return super().get_inference_snapshot()

    def get_urls(self) -> list[str]:
        urls = []
        for engine in self.engines:
            if engine is None:
                continue
            base_url = ray.get(engine.get_url.remote())
            urls.append(f"{base_url}/generate")
        return urls

    # ------------------------------------------------------------------
    # MultiEngineManager hooks.
    # ------------------------------------------------------------------

    def _resolve_placement(self, rank: int):
        replica = rank
        if self._shared_pg:
            # Colocate: teachers share the actor placement group, which the
            # controller owns and removes → owns_pg=False.
            gpu_index = _resolve_teacher_gpu_index(
                replica=replica,
                gpus_per_replica=self.gpus_per_replica,
                shared_pg=True,
                bundle_offset=self._bundle_offset,
            )
            return self._shared_pg_tuple, False, gpu_index

        # Dedicated: this replica creates and owns its own placement group.
        pg_tuple = create_placement_group(
            num_gpus=self.gpus_per_replica,
            node_group_affinity=getattr(self.args, "enable_affinity", True),
        )
        try:
            # PACK is best effort: on a busy cluster the group may straddle nodes or skip GPUs.
            validate_engine_bundles(
                pg_tuple, 0, self.gpus_per_replica, label=f"teacher replica {replica} (dedicated placement group)"
            )
        except Exception:
            remove_placement_group(pg_tuple[0])
            raise
        gpu_index = _resolve_teacher_gpu_index(
            replica=replica,
            gpus_per_replica=self.gpus_per_replica,
            shared_pg=False,
        )
        return pg_tuple, True, gpu_index

    def _physical_placement(self):
        # Shared: checked once against the actor placement group. A dedicated
        # replica's group only exists once _resolve_placement creates it.
        return (ACTOR_POOL, self._shared_pg_tuple) if self._shared_pg else None

    def _ray_resource_kwargs(self, rank: int) -> dict:
        # A deferred teacher may sit on rollout's own bundles. Like a GenRM that
        # shares them, it then asks Ray for a smaller slice so the bundle still
        # fits the training actor, the rollout engine and the teacher.
        shares_rollout_bundles = self._shared_pg and self._bundle_offset < int(
            getattr(self.args, "rollout_num_gpus", None) or 0
        )
        fraction = 0.1 if shares_rollout_bundles else 0.2
        return {"num_cpus": fraction, "num_gpus": fraction}

    def _build_engine_env_vars(self) -> dict[str, str]:
        return _build_teacher_engine_env(self.args)

    def _engine_ctor_args(self, rank: int):
        return self._teacher_args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        return {
            "sglang_overrides": self._overrides,
            "num_gpus_per_engine": self.gpus_per_replica,
            "register_sigterm_handler": False,
            # Static weights: the teacher never joins weight sync.
            "profile": "teacher",
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
        addr_and_ports: dict[int, dict] = {}
        for rank, engine in new_engines:
            # A shared-PG engine is rebuilt on the bundles it had, so keep its
            # endpoint: callers that only hold the URLs captured at startup
            # keep working. A dedicated engine gets a fresh placement group that
            # may land on another node; its new endpoint is published through
            # the topology snapshot, which OPD follows via the teacher gateway.
            if self._shared_pg and rank in self._engine_addr_and_ports:
                addr_and_ports[rank] = dict(self._engine_addr_and_ports[rank])
                continue
            base_port = find_available_port(15000)
            per_engine_addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
                args=self._teacher_args,
                rollout_engines=[(0, engine)],
                worker_type="regular",
                num_gpus_per_engine=self.gpus_per_replica,
                rank_offset=0,
                base_port=base_port,
            )
            addr_and_ports[rank] = per_engine_addr_and_ports[0]
        return addr_and_ports
