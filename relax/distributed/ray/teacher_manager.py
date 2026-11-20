# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
from collections import Counter
from collections.abc import Coroutine
from enum import Enum
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

    def engine_identities(self) -> dict[str, str]:
        """Map scoring URLs to the actors holding their pinned packages."""
        return {
            f"{ray.get(engine.get_url.remote())}/generate": engine._actor_id.hex()
            for engine in self.engines
            if engine is not None
        }

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


class _TeacherState(str, Enum):
    """Only READY accepts new scores."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    OFFLOADED = "offloaded"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@ray.remote
class SharedTeacherManager:
    """Dispatch scores and weight switches to shared Teacher engines."""

    def __init__(self, args, manager, urls, *, initial_teacher, teacher_to_adts_id, tp_size):
        self.manager = manager
        self.urls = urls
        self._package_dir = args.opd_teacher_delta_dir
        self._teacher_ids = teacher_to_adts_id
        self._base = teacher_to_adts_id[initial_teacher]
        self._resident = dict.fromkeys(urls, self._base)
        self._prepared_engines = {}
        self._tp_size = tp_size
        self._timeout = args.opd_teacher_timeout_s
        self._limit = getattr(args, "opd_teacher_connector_limit", 256)
        self._session = None
        self._control = asyncio.Lock()
        self._state = _TeacherState.STARTING
        self._worker = None
        self._wakeup = asyncio.Event()
        self._token_budget = 131072  # per endpoint; a single longer request can still run alone
        self._max_wait_s = 2.0
        self._queues: dict[str, dict[asyncio.Future, tuple[float, dict]]] = {}

    async def initialize(self) -> None:
        async with self._control:
            if self._state is not _TeacherState.STARTING:
                raise RuntimeError(f"Shared Teacher cannot initialize while {self._state.value}")
            self._session = aiohttp.ClientSession(
                # Retire idle connections before the SGLang HTTP server closes them (5s by default).
                connector=aiohttp.TCPConnector(limit=self._limit, keepalive_timeout=2),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
            try:
                await self._restore_replicas()
                self._state = _TeacherState.READY
            except BaseException:
                self._state = _TeacherState.FAILED
                await self._session.close()
                raise

    async def score(self, teacher: str, payload: dict) -> dict:
        if self._state is not _TeacherState.READY:
            raise RuntimeError(f"Shared Teacher is {self._state.value}")
        teacher = self._teacher_ids[teacher]

        reply = asyncio.get_running_loop().create_future()
        self._queues.setdefault(teacher, {})[reply] = monotonic(), payload
        self._wakeup.set()
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_batches())
        try:
            return await reply
        except asyncio.CancelledError:
            # A queued request can be removed; a dispatched request must still drain.
            queue = self._queues.get(teacher)
            if queue is not None and reply in queue:
                del queue[reply]
                if not queue:
                    del self._queues[teacher]
            raise

    async def _send_command(self, url: str, operation: str, teacher: str) -> None:
        from relax.backends.sglang.adts import control_payload

        endpoint = url.removesuffix("/generate")
        target = self._package_dir if operation == "prepare" else teacher
        payload = control_payload(operation, target, self._tp_size, weight_version=teacher)
        async with self._session.post(endpoint + "/update_weights_from_tensor", json=payload) as response:
            response.raise_for_status()
            result = await response.json()
        if not result["success"]:
            raise RuntimeError(f"Teacher {operation} failed at {url}: {result.get('message')}")

        async with self._session.get(endpoint + "/get_weight_version") as response:
            response.raise_for_status()
            info = await response.json()
        if info.get("weight_version") != teacher:
            raise RuntimeError(f"Expected Teacher {teacher} at {url}, got {info.get('weight_version')}")

    async def _restore_replicas(self) -> None:
        identities = await self.manager.engine_identities.remote()

        async def restore(url):
            teacher = self._resident[url]
            if identities[url] == self._prepared_engines.get(url):
                await self._send_command(url, "restore", teacher)
            else:
                # A new actor starts from the base checkpoint and must prepare its own rank-local packages.
                await self._send_command(url, "prepare", self._base)
                if teacher != self._base:
                    await self._send_command(url, "switch", teacher)

        results = await asyncio.gather(*(restore(url) for url in self.urls), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        self._prepared_engines = identities

    async def _generate(self, url: str, payload: dict) -> dict:
        from relax.engine.rollout.on_policy_distillation import _aiohttp_json_post_kwargs

        async with self._session.post(url, **_aiohttp_json_post_kwargs(payload)) as response:
            response.raise_for_status()
            return await response.json()

    def _assign_replicas(self, serving: dict[str, str]) -> dict[str, list[str]]:
        idle = [url for url in self.urls if url not in serving]
        if not self._queues or not idle:
            return {}
        tokens = {t: sum(len(payload["input_ids"]) for _, payload in q.values()) for t, q in self._queues.items()}
        queued_at = {t: next(iter(q.values()))[0] for t, q in self._queues.items()}
        assigned = {t: [] for t in self._queues}
        counts = {t: sum(active == t for active in serving.values()) for t in self._queues}
        now = monotonic()
        while idle:
            waiting = [t for t, q in self._queues.items() if len(q) > len(assigned[t])]
            if not waiting:
                break
            unserved = [t for t in waiting if counts[t] == 0]
            if unserved:
                # Preserve affinity for fresh requests. The oldest overdue, unserved route wins even at saturation.
                overdue = [t for t in unserved if now - queued_at[t] >= self._max_wait_s]
                resident = [t for t in unserved if any(self._resident[url] == t for url in idle)]
                teacher = min(overdue or resident or unserved, key=queued_at.__getitem__)
            else:
                teacher = max(waiting, key=lambda t: (tokens[t] / (counts[t] + 1), -queued_at[t]))

            # Borrow an unused or duplicate resident before displacing another waiting Teacher's only match.
            matching = [url for url in idle if self._resident[url] == teacher]
            unused = [url for url in idle if self._resident[url] not in waiting]
            copies = Counter(self._resident[url] for url in idle)
            spare = [url for url in idle if self._resident[url] not in unserved or copies[self._resident[url]] > 1]
            url = (matching or unused or spare or idle)[0]
            idle.remove(url)
            assigned[teacher].append(url)
            counts[teacher] += 1
        return {t: urls for t, urls in assigned.items() if urls}

    def _take_batch(self, teacher: str, replicas_left: int = 1, *, in_flight: int = 0) -> dict[asyncio.Future, dict]:
        queue = self._queues[teacher]
        available = self._token_budget - in_flight
        if replicas_left > 1:
            # Share a short queue too, reserving at least one request for each remaining replica.
            total = sum(len(payload["input_ids"]) for _, payload in queue.values())
            available = min(available, (total + replicas_left - 1) // replicas_left)

        batch = {}
        for _ in range(len(queue) - replicas_left + 1):
            reply = next(iter(queue))
            _, payload = queue[reply]
            length = len(payload["input_ids"])
            # An idle replica may take one oversized request. A busy one must wait for space.
            if length > available and (in_flight or batch):
                break
            batch[reply] = payload
            available -= length
            del queue[reply]
        if not queue:
            del self._queues[teacher]
        return batch

    async def _run_batches(self) -> None:
        scoring: dict[asyncio.Task, tuple[str, asyncio.Future, int]] = {}
        switching: dict[asyncio.Task, tuple[str, str, dict]] = {}
        try:
            while self._queues or scoring or switching:
                self._wakeup.clear()
                self._finish_completed(scoring, switching)
                serving = self._dispatch_idle_replicas(scoring, switching)
                self._refill_busy_replicas(scoring, serving)
                if scoring or switching:
                    await self._wakeup.wait()
        except BaseException as exc:
            await self._fail_pending(exc, scoring, switching)
        finally:
            self._worker = None

    def _finish_completed(self, scoring: dict, switching: dict) -> None:
        for task in tuple(scoring):
            if not task.done():
                continue
            # Keep the reply registered until success so a lost connection fails its caller too.
            result = task.result()
            _, reply, _ = scoring.pop(task)
            if not reply.done():
                reply.set_result(result)

        for task in tuple(switching):
            if not task.done():
                continue
            task.result()
            url, teacher, batch = switching.pop(task)
            self._resident[url] = teacher
            self._dispatch_scores(url, batch, scoring)

    def _dispatch_idle_replicas(self, scoring: dict, switching: dict) -> dict[str, str]:
        serving = {url: self._resident[url] for url, _, _ in scoring.values()}
        serving.update((url, teacher) for url, teacher, _ in switching.values())
        for teacher, urls in self._assign_replicas(serving).items():
            for index, url in enumerate(urls):
                batch = self._take_batch(teacher, len(urls) - index)
                serving[url] = teacher
                if self._resident[url] == teacher:
                    self._dispatch_scores(url, batch, scoring)
                    continue
                task = self._start_engine_task(self._send_command(url, "switch", teacher))
                switching[task] = (url, teacher, batch)
        return serving

    def _refill_busy_replicas(self, scoring: dict, serving: dict[str, str]) -> None:
        if not self._queues:
            return
        in_flight = Counter()
        for url, _, length in scoring.values():
            in_flight[url] += length

        drain = None
        if self._queues.keys() - set(serving.values()):
            # Prefer a spare replica, then the one with the fewest tokens. Pool order breaks ties.
            counts = Counter(serving.values())
            active = [url for url in self.urls if url in in_flight]
            spare = [url for url in active if serving[url] not in self._queues or counts[serving[url]] > 1]
            drain = min(spare or active, key=in_flight.__getitem__, default=None)

        for url, tokens in in_flight.items():
            teacher = serving[url]
            if url != drain and tokens and teacher in self._queues:
                self._dispatch_scores(url, self._take_batch(teacher, in_flight=tokens), scoring)

    def _dispatch_scores(self, url: str, batch: dict, scoring: dict) -> None:
        for reply, payload in batch.items():
            task = self._start_engine_task(self._generate(url, payload))
            scoring[task] = (url, reply, len(payload["input_ids"]))

    def _start_engine_task(self, coro: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda _: self._wakeup.set())
        return task

    async def _fail_pending(self, exc: BaseException, scoring: dict, switching: dict) -> None:
        self._state = _TeacherState.FAILED
        replies = [reply for _, reply, _ in scoring.values()]
        replies.extend(reply for _, _, batch in switching.values() for reply in batch)
        replies.extend(reply for queue in self._queues.values() for reply in queue)
        for reply in replies:
            if not reply.done():
                reply.set_exception(exc)
        self._queues.clear()
        await asyncio.gather(*scoring, *switching, return_exceptions=True)
        logger.exception("Shared Teacher failed; rebuild the physical engine before reusing it")

    async def offload(self) -> None:
        async with self._control:
            if self._state is _TeacherState.OFFLOADED:
                return
            if self._state is not _TeacherState.READY:
                raise RuntimeError(f"Shared Teacher cannot offload while {self._state.value}")
            self._state = _TeacherState.DRAINING
            try:
                if self._worker is not None:
                    await asyncio.shield(self._worker)
                if self._state is _TeacherState.FAILED:
                    raise RuntimeError("Shared Teacher failed while draining")
                await self.manager.offload.remote()
                self._state = _TeacherState.OFFLOADED
            except BaseException:
                self._state = _TeacherState.FAILED
                raise

    async def onload(self) -> None:
        async with self._control:
            if self._state is _TeacherState.READY:
                return
            if self._state is not _TeacherState.OFFLOADED:
                raise RuntimeError(f"Shared Teacher cannot onload while {self._state.value}")
            try:
                await self.manager.onload.remote()
                # The physical manager may have rebuilt the engine from its initial checkpoint.
                await self._restore_replicas()
                self._state = _TeacherState.READY
            except BaseException:
                self._state = _TeacherState.FAILED
                raise

    async def shutdown(self) -> None:
        async with self._control:
            if self._state is _TeacherState.CLOSED:
                return
            if self._state is not _TeacherState.FAILED:
                self._state = _TeacherState.CLOSING
            try:
                if self._worker is not None:
                    await self._worker
            finally:
                try:
                    if self._session is not None:
                        await self._session.close()
                    await self.manager.shutdown.remote()
                finally:
                    ray.kill(self.manager)
                    self._state = _TeacherState.CLOSED
