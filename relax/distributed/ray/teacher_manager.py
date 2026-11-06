# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
from pathlib import Path
from time import monotonic

import aiohttp
import ray

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.distributed.ray.multi_engine_manager import MultiEngineManager
from relax.distributed.ray.placement_group import create_placement_group
from relax.distributed.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.utils.env import Envs
from relax.utils.http_utils import find_available_port
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


@ray.remote
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

        super().__init__(
            args,
            num_slots=num_replicas * nodes_per_engine,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=SGLangEngine,
            log_prefix="[OPD teacher]",
        )

    def get_urls(self) -> list[str]:
        urls = []
        for engine in self.engines:
            if engine is None:
                continue
            base_url = ray.get(engine.get_url.remote())
            urls.append(f"{base_url}/generate")
        return urls

    def recover(self) -> set:
        """Recover in place only when the teacher's endpoint can stay
        stable."""
        dead = [rank for rank, engine in enumerate(self.all_engines) if engine is None]
        if dead and not self._shared_pg:
            # A dedicated replacement PG may land on another node. OPD callers
            # hold URLs captured at startup, so rebuilding here could advertise
            # success while every caller keeps targeting the old host. Escalate
            # to Controller restart, which rebuilds and re-injects the routes.
            raise RuntimeError(
                f"Dedicated OPD teacher engines died at ranks={dead}; global restart is required to refresh URLs."
            )
        return super().recover()

    # ------------------------------------------------------------------
    # MultiEngineManager hooks.
    # ------------------------------------------------------------------

    def _resolve_placement(self, rank: int):
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

        # Only the replica head owns its PG; follower slots borrow that same PG.
        owns_pg = rank % self.nodes_per_engine == 0
        if owns_pg:
            pg_tuple = create_placement_group(
                num_gpus=self.gpus_per_replica,
                node_group_affinity=getattr(self.args, "enable_affinity", True),
            )
        else:
            pg_tuple = self._engine_placements[replica * self.nodes_per_engine][0]
        gpu_index = _resolve_teacher_gpu_index(
            args=self.args,
            replica=replica,
            gpus_per_replica=self.gpus_per_replica,
            shared_pg=False,
        )
        return pg_tuple, owns_pg, gpu_index + node_gpu_offset

    def _ray_resource_kwargs(self, rank: int) -> dict:
        return {"num_cpus": 0.2, "num_gpus": 0.2}

    def _build_engine_env_vars(self) -> dict[str, str]:
        return _build_teacher_engine_env(self.args)

    def _engine_ctor_args(self, rank: int):
        return self._teacher_args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        return {
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
        for replica, engines in unallocated_replicas.items():
            base_port = find_available_port(15000)
            per_engine_addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
                args=self._teacher_args,
                rollout_engines=engines,
                worker_type="regular",
                num_gpus_per_engine=self.gpus_per_replica,
                rank_offset=replica * self.nodes_per_engine,
                base_port=base_port,
            )
            addr_and_ports.update({rank: per_engine_addr_and_ports[rank] for rank, _ in engines})
        return addr_and_ports


@ray.remote
class SharedTeacherManager:
    """Own admission and checkpoint changes for one physical Teacher engine."""

    def __init__(self, args, manager, url: str, routes: dict[str, str]) -> None:
        self.manager = manager
        self.url = url
        self.base_url = url.removesuffix("/generate")
        self.routes = routes
        self.current_teacher = next(iter(routes))
        self._timeout = float(args.opd_teacher_timeout_s)
        self._limit = int(getattr(args, "opd_teacher_connector_limit", 256))
        if self._limit <= 0:
            raise ValueError("Shared Teacher request capacity must be positive")

        self._session: aiohttp.ClientSession | None = None
        self._control = asyncio.Lock()
        self._state = "starting"
        self._worker: asyncio.Task | None = None
        # A running batch finishes before any switch; these bounds apply between batches.
        self._batch_tokens = 131072
        self._max_wait_s = 2.0
        self._queues: dict[str, dict[asyncio.Future, tuple[float, dict]]] = {}
        self._pending = 0
        self._capacity = asyncio.Event()

    async def initialize(self) -> None:
        async with self._control:
            if self._state != "starting":
                raise RuntimeError(f"Shared Teacher cannot initialize while {self._state}")
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self._limit), timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            try:
                await self._verify_checkpoint(self.current_teacher)
                self._state = "ready"
            except BaseException:
                self._state = "failed"
                await self._session.close()
                raise

    async def score(self, teacher: str, payload: dict) -> dict:
        if self._state != "ready":
            raise RuntimeError(f"Shared Teacher is {self._state}")
        if teacher not in self.routes:
            raise KeyError(f"Unknown shared Teacher: {teacher}")

        input_ids = payload.get("input_ids")
        if not isinstance(input_ids, (tuple, list)) or not input_ids or isinstance(input_ids[0], (tuple, list)):
            raise ValueError("Shared Teacher expects one nonempty input_ids sequence per request")
        # Wait before putting the payload into a Teacher queue. Closing admission
        # replaces this event so an old RPC cannot slip through a later onload.
        capacity = self._capacity
        while self._pending >= self._limit:
            capacity.clear()
            await capacity.wait()
            if capacity is not self._capacity:
                raise RuntimeError("Shared Teacher stopped accepting requests while waiting for capacity")

        reply = asyncio.get_running_loop().create_future()
        self._queues.setdefault(teacher, {})[reply] = monotonic(), payload
        self._pending += 1
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_batches())
        try:
            return await reply
        except asyncio.CancelledError:
            # A queued request can be removed; a dispatched request must still drain.
            queue = self._queues.get(teacher)
            if queue is not None and reply in queue:
                del queue[reply]
                self._pending -= 1
                self._capacity.set()
                if not queue:
                    del self._queues[teacher]
            raise

    async def _verify_checkpoint(self, teacher: str, *, check_version: bool = False) -> None:
        async with self._session.get(self.base_url + "/model_info") as response:
            response.raise_for_status()
            info = await response.json()
        path = info.get("model_path")
        if not path or Path(path).resolve() != Path(self.routes[teacher]).resolve():
            raise RuntimeError(
                f"Shared Teacher {teacher!r} expected checkpoint {self.routes[teacher]!r}, got {path!r}"
            )
        if check_version and info.get("weight_version") != teacher:
            raise RuntimeError(f"Shared Teacher {teacher!r} got weight_version={info.get('weight_version')!r}")

    async def _load(self, teacher: str) -> None:
        started = monotonic()
        payload = {"model_path": self.routes[teacher], "weight_version": teacher, "flush_cache": True}
        async with self._session.post(self.base_url + "/update_weights_from_disk", json=payload) as response:
            response.raise_for_status()
            result = await response.json()
        if result.get("success") is not True:
            raise RuntimeError(f"Shared Teacher checkpoint update failed for {teacher!r}: {result.get('message')}")
        await self._verify_checkpoint(teacher, check_version=True)
        logger.info("Shared Teacher reload %s -> %s took %.2fs", self.current_teacher, teacher, monotonic() - started)
        self.current_teacher = teacher

    async def _generate(self, payload: dict) -> dict:
        from relax.engine.rollout.on_policy_distillation import _aiohttp_json_post_kwargs

        async with self._session.post(self.url, **_aiohttp_json_post_kwargs(payload)) as response:
            response.raise_for_status()
            return await response.json()

    def _take_batch(self) -> tuple[str, dict[asyncio.Future, dict]]:
        # Prefer the loaded Teacher until the oldest waiting request needs its turn.
        oldest = min(self._queues, key=lambda key: next(iter(self._queues[key].values()))[0])
        age = monotonic() - next(iter(self._queues[oldest].values()))[0]
        teacher = self.current_teacher if self.current_teacher in self._queues and age < self._max_wait_s else oldest
        queue = self._queues[teacher]
        batch = {}
        tokens = 0
        while queue:
            reply = next(iter(queue))
            _, payload = queue[reply]
            length = len(payload["input_ids"])
            if batch and tokens + length > self._batch_tokens:
                break
            batch[reply] = payload
            tokens += length
            del queue[reply]
        if not queue:
            del self._queues[teacher]
        return teacher, batch

    async def _run_batches(self) -> None:
        batch = {}
        try:
            while self._queues:
                teacher, batch = self._take_batch()
                if self.routes[teacher] != self.routes[self.current_teacher]:
                    await self._load(teacher)
                else:
                    self.current_teacher = teacher
                results = await asyncio.gather(*map(self._generate, batch.values()), return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                for reply, result in zip(batch, results, strict=True):
                    if not reply.cancelled():
                        result["_relax_teacher_route"] = teacher
                        reply.set_result(result)
                self._pending -= len(batch)
                self._capacity.set()
                batch = {}
        except BaseException as exc:
            # After a transport or reload failure, GPU completion/identity is unknown.
            self._stop_admission("failed")
            for queue in (batch, *self._queues.values()):
                for reply in queue:
                    if not reply.done():
                        reply.set_exception(exc)
            self._queues.clear()
            self._pending = 0
            logger.exception("Shared Teacher failed; rebuild the physical engine before reusing it")
        finally:
            self._worker = None

    def _stop_admission(self, state: str) -> None:
        self._state = state
        self._capacity.set()
        self._capacity = asyncio.Event()

    def get_status(self) -> dict:
        return {
            "state": self._state,
            "teacher_id": None if self._state == "failed" else self.current_teacher,
            "pending": self._pending,
        }

    async def offload(self) -> None:
        async with self._control:
            if self._state == "offloaded":
                return
            if self._state != "ready":
                raise RuntimeError(f"Shared Teacher cannot offload while {self._state}")
            self._stop_admission("draining")
            try:
                if self._worker is not None:
                    await asyncio.shield(self._worker)
                if self._state == "failed":
                    raise RuntimeError("Shared Teacher failed while draining")
                await self.manager.offload.remote()
                self._state = "offloaded"
            except BaseException:
                self._state = "failed"
                raise

    async def onload(self) -> None:
        async with self._control:
            if self._state == "ready":
                return
            if self._state != "offloaded":
                raise RuntimeError(f"Shared Teacher cannot onload while {self._state}")
            try:
                await self.manager.onload.remote()
                # The physical manager may have rebuilt the engine from its initial checkpoint.
                await self._load(self.current_teacher)
                self._state = "ready"
            except BaseException:
                self._state = "failed"
                raise

    async def shutdown(self) -> None:
        async with self._control:
            if self._state == "closed":
                return
            self._stop_admission("closing")
            worker = self._worker
            try:
                if worker is not None:
                    await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Even a cancelled shutdown RPC must drain accepted GPU work.
                await asyncio.shield(worker)
                raise
            finally:
                try:
                    if self._session is not None:
                        await self._session.close()
                    await self.manager.shutdown.remote()
                finally:
                    ray.kill(self.manager)
                    self._state = "closed"
