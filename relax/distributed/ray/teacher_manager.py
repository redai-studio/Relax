# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
from collections import Counter
from enum import Enum
from itertools import cycle
from pathlib import Path
from time import monotonic

import aiohttp
import ray
from ray.util.placement_group import placement_group_table

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
    if getattr(args, "opd_teacher_delta_dir", None):
        env_vars["RELAX_ADP_TEACHER_DIR"] = args.opd_teacher_delta_dir
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
        """Associate each scoring endpoint with the process that owns its in-
        memory weights."""
        return {url: engine._actor_id.hex() for url, engine in zip(self.get_urls(), self.engines, strict=True)}

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

    def _build_engine_ctor_kwargs(self, pg_tuple: tuple, gpu_index: int) -> dict:
        server_gpu_ids = None
        if self.nodes_per_engine == 1:
            pg, bundle_indices, gpu_ids = pg_tuple
            replica = slice(gpu_index, gpu_index + self.gpus_per_replica)
            nodes = placement_group_table(pg)["bundles_to_node_id"]
            replica_nodes = {nodes[index] for index in bundle_indices[replica]}
            if len(replica_nodes) != 1 or not all(replica_nodes):
                raise ValueError(f"Single-node Teacher replica spans invalid placement-group nodes: {replica_nodes}")
            server_gpu_ids = gpu_ids[replica]
        return {
            "sglang_overrides": self._overrides,
            "num_gpus_per_engine": self.gpus_per_replica,
            "server_gpu_ids": server_gpu_ids,
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
        port_cursors = {}
        for replica, engines in unallocated_replicas.items():
            # All replicas are allocated before any engine binds its ports.
            base_port = max(port_cursors.values()) if port_cursors else find_available_port(15000)
            per_engine_addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
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
    """Score queued requests on replicas that can each hold a different Teacher
    checkpoint."""

    def __init__(self, args, manager, urls: list[str] | str, routes: dict[str, str]) -> None:
        self.manager = manager
        self.urls = [urls] if isinstance(urls, str) else urls
        self._lockstep = getattr(args, "opd_teacher_scheduler", "affinity") == "lockstep"
        self._score_urls = cycle(self.urls)
        self.routes = routes
        # Route aliases for one checkpoint share both a FIFO and a resident identity.
        checkpoints = {}
        self._route_teacher = {
            route: checkpoints.setdefault(Path(path).resolve(), route) for route, path in routes.items()
        }
        self._replica_teacher: dict[str, str | None] = dict.fromkeys(self.urls, next(iter(routes)))
        self._delta_ids = getattr(args, "opd_teacher_delta_ids", None)
        if self._delta_ids:
            from relax.backends.sglang.adp import delta_tp_size

            self._delta_tp_size = delta_tp_size(args)
        self._prepared_engines: dict[str, str] = {}
        self._timeout = float(args.opd_teacher_timeout_s)
        self._limit = int(getattr(args, "opd_teacher_connector_limit", 256))
        if self._limit <= 0:
            raise ValueError("Shared Teacher request capacity must be positive")

        self._session: aiohttp.ClientSession | None = None
        self._control = asyncio.Lock()
        self._state = _TeacherState.STARTING
        self._worker: asyncio.Task | None = None
        self._scheduler_wakeup = asyncio.Event()
        self._batch_tokens = 131072 * len(self.urls)  # nominal token budget across the whole pool
        self._max_wait_s = 2.0
        self._queues: dict[str, dict[asyncio.Future, tuple[float, dict]]] = {}
        self._pending = 0
        self._capacity = asyncio.Event()

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
                await self._restore_replicas(initializing=True)
                self._state = _TeacherState.READY
            except BaseException:
                self._state = _TeacherState.FAILED
                await self._session.close()
                raise

    async def score(self, teacher: str, payload: dict) -> dict:
        if self._state is not _TeacherState.READY:
            raise RuntimeError(f"Shared Teacher is {self._state.value}")
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

        teacher = self._route_teacher[teacher]
        reply = asyncio.get_running_loop().create_future()
        self._queues.setdefault(teacher, {})[reply] = monotonic(), payload
        self._pending += 1
        self._scheduler_wakeup.set()
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

    async def _verify_checkpoint(
        self, url: str, teacher: str, *, version: str | None = None, check_path: bool = True
    ) -> None:
        async with self._session.get(url.removesuffix("/generate") + "/model_info") as response:
            response.raise_for_status()
            info = await response.json()
        # SGLang changes model_path only for a disk reload. Tensor updates publish their package ID as weight_version.
        path = info.get("model_path")
        if check_path and (not path or Path(path).resolve() != Path(self.routes[teacher]).resolve()):
            raise RuntimeError(
                f"Shared Teacher {teacher!r} at {url} expected checkpoint {self.routes[teacher]!r}, got {path!r}"
            )
        if version is not None and info.get("weight_version") != version:
            raise RuntimeError(
                f"Shared Teacher {teacher!r} at {url} expected weight_version={version!r}, "
                f"got {info.get('weight_version')!r}"
            )

    async def _update(self, url: str, endpoint: str, payload: dict, teacher: str) -> None:
        async with self._session.post(url.removesuffix("/generate") + endpoint, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
        if result.get("success") is not True:
            raise RuntimeError(
                f"Shared Teacher checkpoint update failed for {teacher!r} at {url}: {result.get('message')}"
            )

    async def _disk(self, url: str, teacher: str) -> None:
        payload = {"model_path": self.routes[teacher], "weight_version": teacher, "flush_cache": True}
        await self._update(url, "/update_weights_from_disk", payload, teacher)
        await self._verify_checkpoint(url, teacher, version=teacher)

    async def _delta(self, url: str, operation: str, teacher: str) -> None:
        from relax.backends.sglang.adp import control_payload

        package_id = self._delta_ids[teacher]
        payload = control_payload(operation, package_id, self._delta_tp_size)
        await self._update(url, "/update_weights_from_tensor", payload, teacher)
        await self._verify_checkpoint(url, teacher, version=package_id, check_path=False)

    async def _prepare_delta(self, url: str) -> None:
        initial = next(iter(self.routes))
        await self._verify_checkpoint(url, initial)
        await self._delta(url, "prepare", initial)
        # Prove that each configured checkpoint has the bytes expected by its package before accepting scores.
        seen = {self._delta_ids[initial]}
        for route, package_id in self._delta_ids.items():
            if package_id not in seen:
                await self._disk(url, route)
                await self._delta(url, "adopt", route)
                seen.add(package_id)

    async def _restore_replicas(self, *, initializing: bool = False) -> None:
        identities = {}
        if self._delta_ids:
            identities = await self.manager.engine_identities.remote()
            if set(identities) != set(self.urls) or len(set(identities.values())) != len(self.urls):
                raise RuntimeError("Shared Teacher replica identities do not match the scoring endpoints")

        async def restore(url: str, teacher: str | None) -> None:
            if teacher is None:
                raise RuntimeError(f"Cannot restore a shared Teacher with unknown identity at {url}")
            if self._delta_ids:
                if identities[url] != self._prepared_engines.get(url):
                    # Replacement engines boot at the initial checkpoint without a pinned bank.
                    await self._prepare_delta(url)
                await self._delta(url, "restore", teacher)
            elif initializing:
                await self._verify_checkpoint(url, teacher)
            else:
                await self._disk(url, teacher)
            logger.info(
                "Shared Teacher %s endpoint=%s teacher=%s", "initialize" if initializing else "restore", url, teacher
            )

        # A failed restore closes admission; still wait for every other physical writer.
        results = await asyncio.gather(
            *(restore(url, self._replica_teacher[url]) for url in self.urls), return_exceptions=True
        )
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
                resident = [t for t in unserved if any(self._replica_teacher[url] == t for url in idle)]
                teacher = min(overdue or resident or unserved, key=queued_at.__getitem__)
            else:
                teacher = max(waiting, key=lambda t: (tokens[t] / (counts[t] + 1), -queued_at[t]))

            # Borrow an unused or duplicate resident before displacing another waiting Teacher's only match.
            matching = [url for url in idle if self._replica_teacher[url] == teacher]
            unused = [url for url in idle if self._replica_teacher[url] not in waiting]
            copies = Counter(self._replica_teacher[url] for url in idle)
            spare = [
                url
                for url in idle
                if self._replica_teacher[url] not in unserved or copies[self._replica_teacher[url]] > 1
            ]
            url = (matching or unused or spare or idle)[0]
            idle.remove(url)
            assigned[teacher].append(url)
            counts[teacher] += 1
        return {t: urls for t, urls in assigned.items() if urls}

    def _take_batch(self, teacher: str, replicas_left: int) -> dict[asyncio.Future, dict]:
        queue = self._queues[teacher]
        # Each endpoint now owns its batch. An oversized first request can still run alone.
        budget = self._batch_tokens // len(self.urls)
        if replicas_left > 1:
            # Split a short queue too; reserve at least one request per remaining assigned endpoint.
            total = sum(len(payload["input_ids"]) for _, payload in queue.values())
            budget = min(budget, (total + replicas_left - 1) // replicas_left)
        batch = {}
        tokens = 0
        for _ in range(len(queue) - replicas_left + 1):
            reply = next(iter(queue))
            _, payload = queue[reply]
            length = len(payload["input_ids"])
            if batch and tokens + length > budget:
                break
            batch[reply] = payload
            tokens += length
            del queue[reply]
        if not queue:
            del self._queues[teacher]
        return batch

    async def _switch_replica(self, url: str, teacher: str) -> None:
        previous = self._replica_teacher[url]
        self._replica_teacher[url] = None
        started = monotonic()
        if self._delta_ids:
            await self._delta(url, "switch", teacher)
        else:
            await self._disk(url, teacher)
        self._replica_teacher[url] = teacher
        logger.info(
            "Shared Teacher %s %s -> %s took %.3fs endpoint=%s",
            "delta" if self._delta_ids else "reload",
            previous,
            teacher,
            monotonic() - started,
            url,
        )

    async def _dispatch_lockstep(self, scoring, tokens, dispatch, preferred_url) -> None:
        if not self._queues:
            return
        oldest = min(self._queues, key=lambda route: next(iter(self._queues[route].values()))[0])
        current = self._replica_teacher[self.urls[0]]
        age = monotonic() - next(iter(self._queues[oldest].values()))[0]
        teacher = current if current in self._queues and age < self._max_wait_s else oldest
        if teacher != current:
            if scoring:
                return
            # All replicas must finish loading before any of them accepts work on the new route.
            results = await asyncio.gather(
                *(self._switch_replica(url, teacher) for url in self.urls), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return  # A caller may have cancelled its queued request while the pool switched.

        counts = Counter(url for url, _, _, _ in scoring.values())
        total = sum(tokens.values())
        queue = self._queues[teacher]
        while queue:
            reply = next(iter(queue))
            _, payload = queue[reply]
            length = len(payload["input_ids"])
            if total and total + length > self._batch_tokens:
                break
            url = min(self.urls, key=lambda u: (counts[u], u != preferred_url))
            del queue[reply]
            dispatch(url, teacher, {reply: payload})
            counts[url] += 1
            total += length
            preferred_url = None
        if not queue:
            del self._queues[teacher]

    async def _run_batches(self) -> None:
        switching = {}
        scoring = {}
        tokens = dict.fromkeys(self.urls, 0)
        budget = self._batch_tokens // len(self.urls)
        preferred_url = next(self._score_urls)

        def wake(_):
            self._scheduler_wakeup.set()

        def dispatch(url, teacher, requests):
            for reply, payload in requests.items():
                length = len(payload["input_ids"])
                task = asyncio.create_task(self._generate(url, payload))
                scoring[task] = (url, teacher, reply, length)
                tokens[url] += length
                task.add_done_callback(wake)

        try:
            while self._queues or switching or scoring:
                self._scheduler_wakeup.clear()
                for task in tuple(scoring):
                    if task.done():
                        result = task.result()
                        url, _, reply, length = scoring.pop(task)
                        tokens[url] -= length
                        preferred_url = url
                        if not reply.done():
                            reply.set_result(result)
                        self._pending -= 1
                        self._capacity.set()

                if self._lockstep:
                    await self._dispatch_lockstep(scoring, tokens, dispatch, preferred_url)
                    if scoring:
                        await self._scheduler_wakeup.wait()
                    continue

                for task in tuple(switching):
                    if task.done():
                        task.result()
                        url, teacher, batch = switching.pop(task)
                        dispatch(url, teacher, batch)

                serving = {url: teacher for url, teacher, _, _ in scoring.values()}
                serving.update({url: teacher for url, teacher, _ in switching.values()})
                for teacher, urls in self._assign_replicas(serving).items():
                    for index, url in enumerate(urls):
                        batch = self._take_batch(teacher, len(urls) - index)
                        serving[url] = teacher
                        if self._replica_teacher[url] == teacher:
                            dispatch(url, teacher, batch)
                        else:
                            task = asyncio.create_task(self._switch_replica(url, teacher))
                            switching[task] = (url, teacher, batch)
                            task.add_done_callback(wake)

                # Leave a path for a waiting, uncovered Teacher to acquire an engine.
                # Other resident routes can refill without draining their current scores.
                uncovered = self._queues.keys() - set(serving.values())
                drain_url = None
                if uncovered:
                    counts = Counter(serving.values())
                    candidates = {url for url, _, _, _ in scoring.values()}
                    if candidates:
                        drain_url = min(
                            candidates,
                            key=lambda u: (
                                serving[u] in self._queues and counts[serving[u]] == 1,
                                tokens[u],
                                self.urls.index(u),
                            ),
                        )
                for url, teacher in serving.items():
                    if url == drain_url or self._replica_teacher[url] != teacher or not tokens[url]:
                        continue
                    queue = self._queues.get(teacher)
                    while queue:
                        reply = next(iter(queue))
                        _, payload = queue[reply]
                        if tokens[url] + len(payload["input_ids"]) > budget:
                            break
                        del queue[reply]
                        dispatch(url, teacher, {reply: payload})
                        if not queue:
                            del self._queues[teacher]
                if scoring or switching:
                    await self._scheduler_wakeup.wait()
        except BaseException as exc:
            self._stop_admission(_TeacherState.FAILED)
            replies = [reply for _, _, reply, _ in scoring.values()]
            replies.extend(reply for _, _, batch in switching.values() for reply in batch)
            replies.extend(reply for queue in self._queues.values() for reply in queue)
            for reply in replies:
                if not reply.done():
                    reply.set_exception(exc)
            self._queues.clear()
            self._pending = 0
            await asyncio.gather(*scoring, *switching, return_exceptions=True)
            logger.exception("Shared Teacher failed; rebuild the physical engine before reusing it")
        finally:
            self._worker = None

    def _stop_admission(self, state: _TeacherState) -> None:
        self._state = state
        self._capacity.set()
        self._capacity = asyncio.Event()

    def get_status(self) -> dict:
        teachers = self._replica_teacher if self._state is not _TeacherState.FAILED else dict.fromkeys(self.urls)
        ids = set(teachers.values())
        status = {
            "state": self._state.value,
            "teacher_id": next(iter(ids)) if len(ids) == 1 else None,
            "pending": self._pending,
        }
        if len(self.urls) > 1:
            status["replica_teachers"] = dict(teachers)
        return status

    async def offload(self) -> None:
        async with self._control:
            if self._state is _TeacherState.OFFLOADED:
                return
            if self._state is not _TeacherState.READY:
                raise RuntimeError(f"Shared Teacher cannot offload while {self._state.value}")
            self._stop_admission(_TeacherState.DRAINING)
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
                self._stop_admission(_TeacherState.CLOSING)
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
