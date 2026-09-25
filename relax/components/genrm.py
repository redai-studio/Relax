# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Service Implementation.

This module provides a Ray Serve deployment for Generative Reward Model
(genRM), which evaluates responses using LLM-based preference prediction.

A single Serve deployment can host multiple genRM instances (distinct
models/configs), routed by a caller-supplied ``route_key`` -- typically the
name of the reward/scoring task invoking it. Requests that omit ``route_key``
fall back to the sole "__default__" instance (the legacy single-model config).
"""

import asyncio
import time
from argparse import Namespace
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.ray.placement_group import create_genrm_managers
from relax.utils.data.processing_utils import load_tokenizer
from relax.utils.env import Envs
from relax.utils.genrm_scale_registry import GenRMScaleRegistry


app = FastAPI()

# Max concurrent in-flight requests per GenRM Serve replica. Ray Serve's default
# of 5 throttles judge dispatch and leaves the SGLang engines idle. The replica
# is a pure-async CPU proxy (tokenize + forward), so a high cap lets one replica
# saturate the engines. Override via env for tuning.
GENRM_SERVE_MAX_ONGOING_REQUESTS = Envs.GENRM_SERVE_MAX_ONGOING_REQUESTS

# NOTE: GENRM_SERVE_MAX_ONGOING_REQUESTS above must stay module-level — it feeds
# the @serve.deployment decorator, which runs at import. The retry count has no
# such constraint, so it is read at the call site to stay lazy.

# Sentinel instance key for the legacy single-model config (--genrm-model-path)
# and for requests that don't pass a route_key.
_DEFAULT_INSTANCE_KEY = "__default__"


class Message(BaseModel):
    """Single chat message."""

    role: str
    content: str


class GenerateRequest(BaseModel):
    """Request model for genRM generation (OpenAI chat format).

    Accepts a list of messages in OpenAI format with optional sampling params.
    """

    messages: Union[List[Message], List[dict]]
    sampling_params: Optional[dict] = None
    route_key: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response model for genRM generation.

    Returns the raw model response text plus, when available, which engine
    served the request and the SGLang finish metadata. This makes per-engine
    reward-consistency experiments and truncation checks possible without
    touching the training path; clients that only read ``response`` are
    unaffected.
    """

    response: str
    engine_host: Optional[str] = None
    engine_port: Optional[int] = None
    finish_reason: Optional[str] = None
    completion_tokens: Optional[int] = None


class GenRMScaleRequest(BaseModel):
    """Request model for GenRM scale-out / scale-in.

    ``num_replicas`` is the target *absolute* total engine count (not a delta).
    Idempotency is keyed on the request-body fingerprint (model_name,
    num_replicas, timeout_secs): same key + same fingerprint returns the
    original operation; a keyed NOOP decision is replayed verbatim; a different
    fingerprint is rejected with 409.
    """

    model_name: str = Field(default="default", description="GenRM instance (route key) to scale")
    num_replicas: int = Field(..., gt=0, description="Target absolute total engine count")
    timeout_secs: Optional[float] = Field(default=None, gt=0, description="Total timeout for the operation")
    idempotency_key: Optional[str] = Field(default=None, description="Idempotency key for safe retries")


class GenRMScaleResponse(BaseModel):
    """Response model for GenRM scale operations.

    NOOP responses intentionally carry no ``request_id`` (the decision is
    replayed verbatim for keyed retries, per the Task 4 contract).
    """

    status: str
    request_id: Optional[str] = None
    current: Optional[int] = None
    ready: Optional[int] = None
    detail: Optional[str] = None


class GenRMScaleStatusResponse(BaseModel):
    """Response model for GenRM scale operation status queries."""

    request_id: str
    direction: str
    status: str
    model_name: str
    target: int
    timeout_secs: Optional[float] = None
    idempotency_key: Optional[str] = None
    created_at: float
    updated_at: float
    current: Optional[int] = None
    ready: Optional[int] = None
    created: int = 0
    removed: int = 0
    failed: int = 0
    cleanup_required: bool = False
    error_message: Optional[str] = None
    detail: Optional[str] = None


class GenRMScaleReconcileResponse(BaseModel):
    """Response model for GenRM scale-operation reconcile.

    Reconcile continues the *original* operation (no new request ID, no victim
    re-selection, no re-scaling); it only retries unfinished resource cleanup.
    ``cleanup_required`` therefore reports whether the model is still blocked
    from new scale operations after the attempt.
    """

    request_id: str
    direction: str
    status: str
    cleanup_required: bool
    victim_cleared: bool = False
    detail: Optional[str] = None


class _EngineCacheState:
    """Per-instance round-robin cache over a GenRMManager's live engine list.

    Isolated per route_key so a dead/rebuilt engine on one instance never
    perturbs another instance's cycle.
    """

    def __init__(self) -> None:
        self.hosts_ports: Optional[list] = None
        self.cycle: Optional[Any] = None
        self.refreshed_at: float = 0.0

    def invalidate(self) -> None:
        now = time.monotonic()
        if now - self.refreshed_at < Envs.GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S:
            return
        self.hosts_ports = None

    def needs_refresh(self) -> bool:
        if self.hosts_ports is None:
            return True
        if self.hosts_ports:
            return False
        return time.monotonic() - self.refreshed_at >= Envs.GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S

    def refresh(self, hosts_ports: list) -> None:
        self.refreshed_at = time.monotonic()
        # Swap the list and its cycle together: the manager compacts the list
        # over dead engines, so a cycle built for the old length would hand
        # back an out-of-range index.
        self.cycle = cycle(range(len(hosts_ports)))
        self.hosts_ports = hosts_ports

    def force_refresh(self) -> None:
        """Drop the cached list immediately, bypassing the invalidation
        cooldown.

        Reserved for scale operations, where the routing table changed and the
        next pick must observe it.
        """
        self.hosts_ports = None


@serve.deployment(
    max_ongoing_requests=GENRM_SERVE_MAX_ONGOING_REQUESTS,
    logging_config=LoggingConfig(
        log_level="WARNING",
        enable_access_log=False,  # 关闭 HTTP 访问日志
    ),
)
@serve.ingress(app)
class GenRM(Base):
    """GenRM Service for generative reward model evaluation.

    This service uses SGLang engines to perform preference evaluation by
    comparing model responses against ground truth or standards. It may host
    one or more genRM instances (models/configs), routed by ``route_key``.
    """

    def __init__(
        self,
        healthy: Any,
        pg: Optional[Any],
        num_gpus: int,
        config: Namespace,
        role: str,
        runtime_env: Optional[dict] = None,
    ) -> None:
        """Initialize GenRM service.

        Args:
            healthy: Remote health manager actor handle.
            pg: Placement group for resource allocation.
            num_gpus: Number of GPUs allocated (used by Service framework).
            config: Runtime configuration namespace.
            role: Role name (should be "genrm").
            runtime_env: Optional Ray runtime environment dict.
        """
        super().__init__()
        self.config = config
        self.healthy = healthy
        self.role = role

        # {route_key: GenRMManager handle}. Single-instance configs (the legacy
        # --genrm-model-path path) resolve to exactly {"__default__": manager}.
        self.genrm_managers = create_genrm_managers(config, pg, runtime_env=runtime_env)
        self.instance_specs = config._genrm_instances_resolved

        self._engine_caches: dict[str, _EngineCacheState] = {key: _EngineCacheState() for key in self.genrm_managers}
        # Per-engine (key, host, port) counters, updated only on this replica's
        # event loop. ``inflight`` backs the scale-in drain proof; ``served``
        # exposes routing distribution via /genrm/engines.
        self._engine_inflight: dict = {}
        self._engine_served: dict = {}
        # Task 4 elastic-scaling contract core: operation registry with
        # state machine / idempotency / mutual exclusion. The protected
        # ``initial`` lower bound per instance is its engine count.
        self._scale_registry = GenRMScaleRegistry()
        for key, spec in self.instance_specs.items():
            self._scale_registry.register_initial(key, spec["num_gpus"] // spec["num_gpus_per_engine"])
        self._logger.info(f"GenRM service initialized successfully: instances={list(self.genrm_managers)}")
        # Shared HTTP client for engine calls (avoids per-request connection overhead).
        # Raise pool limits well above httpx's default 100 so one replica can fan out
        # many concurrent engine requests; keepalive_expiry >> the default 5s so
        # idle-then-reused connections aren't reaped mid-burst (avoids ReadError/500).
        self._http_client = httpx.AsyncClient(
            timeout=1800,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048, keepalive_expiry=600),
        )

        # Load one tokenizer per instance -- distinct instances may be distinct
        # models with distinct tokenizers/chat templates.
        self.tokenizers = {
            key: load_tokenizer(spec["model_path"], trust_remote_code=True)
            for key, spec in self.instance_specs.items()
        }

    def run(self):
        """GenRM is a passive HTTP service, no background loop needed.

        Unlike Actor or Rollout, GenRM only responds to incoming requests and
        does not actively produce work. Return None so the Controller training
        loop does not block on it.
        """
        return None

    @app.post("/generate")
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate response for given chat messages.

        Takes OpenAI-style messages as input, sends to SGLang engine,
        and returns the raw model response. The caller is responsible
        for formatting the prompt and parsing the response.

        Args:
            request: GenerateRequest containing messages list, optional
                sampling_params, and an optional route_key selecting which
                genRM instance to use (defaults to the sole instance).

        Returns:
            GenerateResponse containing raw model response text
        """
        try:
            output = await self._call_engine(request.route_key, request.messages, request.sampling_params)
            response = output.get("text", "").strip()
            meta_info = output.get("meta_info") or {}
            finish_reason = meta_info.get("finish_reason") if isinstance(meta_info, dict) else None
            return GenerateResponse(
                response=response,
                engine_host=output.get("engine_host"),
                engine_port=output.get("engine_port"),
                finish_reason=(finish_reason.get("type") if isinstance(finish_reason, dict) else None),
                completion_tokens=(meta_info.get("completion_tokens") if isinstance(meta_info, dict) else None),
            )

        except Exception as e:
            self._logger.error(f"GenRM generation failed (route_key={request.route_key}): {e}")
            raise

    def _resolve_instance_key(self, route_key: Optional[str]) -> str:
        if route_key is None and len(self.genrm_managers) == 1:
            return next(iter(self.genrm_managers))
        key = route_key or _DEFAULT_INSTANCE_KEY
        if key not in self.genrm_managers:
            raise RuntimeError(
                f"No GenRM instance registered for route_key={key!r}; available={list(self.genrm_managers)}"
            )
        return key

    def _pick_engine(self, route_key: Optional[str]) -> tuple[str, int, str, int]:
        """Round-robin one live engine of the instance selected by
        ``route_key``, refreshing that instance's cache if it was dropped."""
        key = self._resolve_instance_key(route_key)
        cache = self._engine_caches[key]
        if cache.needs_refresh():
            hosts_ports = ray.get(self.genrm_managers[key].get_engine_hosts_ports.remote())
            cache.refresh(hosts_ports)

        hosts_ports = cache.hosts_ports
        if not hosts_ports:
            raise RuntimeError(f"No genRM engines available for instance '{key}'")

        # Thread-safe round-robin via itertools.cycle (next() is atomic in CPython).
        # Re-read the local alias, not cache.*, so a concurrent invalidation
        # can't make the index and the list disagree.
        idx = next(cache.cycle) % len(hosts_ports)
        host, port = hosts_ports[idx]
        return key, idx, host, port

    async def _call_engine(
        self, route_key: Optional[str], messages: list, sampling_params: Optional[dict] = None
    ) -> dict:
        """Call an SGLang engine for text generation.

        Uses the engine addresses obtained from the selected instance's
        GenRMManager to send HTTP requests to the underlying SGLang server.

        Args:
            route_key: Selects which genRM instance to use.
            messages: List of chat messages in OpenAI format.
            sampling_params: Optional per-request sampling params that override defaults.

        Returns:
            Dict containing at least {"text": str} from the SGLang server.
        """
        key, idx, host, port = self._pick_engine(route_key)
        inflight_key = (key, host, port)
        self._engine_inflight[inflight_key] = self._engine_inflight.get(inflight_key, 0) + 1
        # The inner call may re-pick on retry; the holder carries the final
        # engine so this finally decrements the engine actually used.
        inflight_holder = [inflight_key]
        try:
            output = await self._call_engine_tracked(
                route_key, key, host, port, inflight_holder, messages, sampling_params
            )
            # Attribute the reply to the engine that actually served it (the
            # holder's final key follows retry re-picks), so per-engine
            # consistency evidence does not have to infer routing.
            final_key = inflight_holder[0]
            output["engine_host"] = final_key[1]
            output["engine_port"] = final_key[2]
            return output
        finally:
            final_key = inflight_holder[0]
            remaining = self._engine_inflight.get(final_key, 0) - 1
            self._engine_inflight[final_key] = max(0, remaining)
            self._engine_served[final_key] = self._engine_served.get(final_key, 0) + 1

    async def _call_engine_tracked(
        self,
        route_key: Optional[str],
        key: str,
        host: str,
        port: int,
        inflight_holder: list,
        messages: list,
        sampling_params: Optional[dict] = None,
    ) -> dict:
        """Engine call with routing already accounted in ``_engine_inflight``.

        On a retry re-pick the counters move to the new engine atomically, so
        the drain proof never misses a request that switched engines.
        """
        spec = self.instance_specs[key]
        # ensure plain list — some tokenizers return BatchEncoding which is not JSON-serializable
        # Tokenization (chat-template render + encode) is synchronous CPU work; run it in a
        # worker thread so it does not block this replica's event loop. Fast (Rust) tokenizers
        # release the GIL during encode, so concurrent requests tokenize in parallel instead of
        # serializing — without this a single replica throttles dispatch and starves the engines.
        # Forward chat_template_kwargs from the instance's sampling_config through to
        # the jinja template — e.g. `{"enable_thinking": false}` for Qwen3+ to
        # suppress the default <think> block. Keys unused by the template are
        # silently dropped by transformers, so this is safe across model families.
        sampling_config = spec["sampling_config"]
        chat_template_kwargs = sampling_config.get("chat_template_kwargs", {}) or {}
        input_ids = await asyncio.to_thread(
            self.tokenizers[key].apply_chat_template,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        if not isinstance(input_ids, list):
            input_ids = (
                input_ids["input_ids"]
                if hasattr(input_ids, "__getitem__") and "input_ids" in input_ids
                else list(input_ids)
            )

        # Merge per-request sampling params with default config
        default_sampling = {
            "temperature": sampling_config.get("temperature", 0.2),
            "top_p": sampling_config.get("top_p", 1.0),
            "top_k": sampling_config.get("top_k", -1),
            "max_new_tokens": sampling_config.get("max_response_len", 1024),
        }
        # Override defaults with per-request params
        if sampling_params:
            default_sampling.update(sampling_params)

        payload = {
            "input_ids": input_ids,
            "sampling_params": default_sampling,
        }

        # Retry transient resets (transport-level or 5xx) with short backoff so
        # bursty colocate contention doesn't surface as a 500; 4xx is a client bug
        # (terminal) and a cancellation (caller timeout) is never retried.
        # Serve→engine retry attempts for transient resets (ReadError / 5xx under
        # bursty colocate contention), absorbing them before they surface as a 500
        # to the client. Set 1 to disable.
        retry_attempts = Envs.GENRM_ENGINE_RETRY_ATTEMPTS
        for _attempt in range(1, retry_attempts + 1):
            try:
                resp = await self._http_client.post(f"http://{host}:{port}/generate", json=payload)
                resp.raise_for_status()
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                status = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                if (status == 0 or status >= 500) and _attempt < retry_attempts:
                    # A transport error means this engine may be gone and its
                    # replacement will come back on a different port, so drop the
                    # cache and re-pick — retrying the same dead host is useless.
                    if status == 0:
                        self._engine_caches[key].invalidate()
                    old_key = inflight_holder[0]
                    key, idx, host, port = self._pick_engine(route_key)
                    new_key = (key, host, port)
                    if new_key != old_key:
                        # Move the in-flight accounting to the retry target so
                        # the drain proof tracks the engine actually serving.
                        self._engine_inflight[old_key] = max(0, self._engine_inflight.get(old_key, 0) - 1)
                        self._engine_inflight[new_key] = self._engine_inflight.get(new_key, 0) + 1
                        inflight_holder[0] = new_key
                    await asyncio.sleep(0.3 * _attempt)
                    continue
                raise
        return resp.json()

    @app.get("/health")
    async def health(self) -> dict:
        """Health check endpoint; reports per-instance status."""
        instances = {}
        for key, manager in self.genrm_managers.items():
            try:
                is_healthy = ray.get(manager.health_check.remote())
                instances[key] = {"status": "healthy" if is_healthy else "unhealthy"}
            except Exception as e:
                self._logger.error(f"GenRM health check failed for instance '{key}': {e}")
                instances[key] = {"status": "unhealthy", "error": str(e)}
        overall = "healthy" if all(v["status"] == "healthy" for v in instances.values()) else "unhealthy"
        if list(instances) == [_DEFAULT_INSTANCE_KEY]:
            result = {"status": overall, "service": "genrm"}
            if "error" in instances[_DEFAULT_INSTANCE_KEY]:
                result["error"] = instances[_DEFAULT_INSTANCE_KEY]["error"]
            return result
        return {"status": overall, "service": "genrm", "instances": instances}

    @app.get("/metrics")
    async def metrics(self) -> dict:
        """Metrics endpoint; reports per-instance stats with live capacity.

        ``num_engines`` follows the manager's actual capacity (elastic scaling
        changes it after startup); ``ready_engines``/``occupied``/
        ``pending_cleanup`` use the same accounting as ``/engines``. When the
        capacity query fails, the startup count is kept and ``capacity_error``
        is surfaced instead of fabricating numbers.
        """
        instances: Dict[str, Any] = {}
        for key, spec in self.instance_specs.items():
            entry = {
                "model_path": spec["model_path"],
                "num_gpus": spec["num_gpus"],
                "num_gpus_per_engine": spec["num_gpus_per_engine"],
                "num_engines": spec["num_gpus"] // spec["num_gpus_per_engine"],
            }
            manager = self.genrm_managers.get(key)
            if manager is not None and getattr(manager, "get_engine_capacity", None) is not None:
                try:
                    capacity = ray.get(manager.get_engine_capacity.remote())
                    entry.update(
                        {
                            "num_engines": int(capacity.get("current", entry["num_engines"])),
                            "ready_engines": int(capacity.get("ready", len(self._genrm_engine_list(key)))),
                            "occupied": int(capacity.get("occupied", entry["num_engines"])),
                            "pending_cleanup": int(capacity.get("pending_cleanup", 0)),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    self._logger.warning(f"GenRM metrics capacity query failed for '{key}': {e}")
                    entry["capacity_error"] = str(e)
            instances[key] = entry
        if list(instances) == [_DEFAULT_INSTANCE_KEY]:
            return {"service": "genrm", **instances[_DEFAULT_INSTANCE_KEY]}
        return {"service": "genrm", "instances": instances}

    # ------------------------------------------------------------------
    # Elastic scaling (Task 4). The endpoints are thin: validation,
    # idempotency, mutual exclusion and the operation state machine live in
    # ``GenRMScaleRegistry``; actual Ray lifecycle execution is delegated to
    # manager hooks (``execute_genrm_scale_out`` / ``execute_genrm_scale_in``).
    # The component owns public registry state; manager progress remains the
    # authority for physical lifecycle completion and cleanup.
    # ------------------------------------------------------------------

    @app.post("/scale_out")
    async def scale_out(self, request: GenRMScaleRequest) -> GenRMScaleResponse:
        """Scale one GenRM instance to an absolute target engine count."""
        return await self._genrm_scale("scale_out", request)

    @app.post("/scale_in")
    async def scale_in(self, request: GenRMScaleRequest) -> GenRMScaleResponse:
        """Scale one GenRM instance down to an absolute target engine count."""
        return await self._genrm_scale("scale_in", request)

    @app.get("/engines")
    async def get_engines(self) -> Dict[str, Any]:
        """Discover engines per instance (``current`` = capacity, ``ready`` = routable).

        Per-engine ``inflight``/``served`` counters make routing visible to
        scale-operation evidence: after a scale-out both engines' ``served``
        counts grow, after a scale-in the victim disappears from the list."""
        instances: Dict[str, Any] = {}
        for key in self.genrm_managers:
            engines = self._genrm_engine_list(key)
            engine_views = []
            for host, port in engines:
                stats_key = (key, host, port)
                engine_views.append(
                    {
                        "host": host,
                        "port": port,
                        "inflight": self._engine_inflight.get(stats_key, 0),
                        "served": self._engine_served.get(stats_key, 0),
                    }
                )
            # ``current`` is service capacity (including a draining victim),
            # while physical occupancy and pending cleanup are separate.
            current = len(engines)
            occupied = current
            pending_cleanup = 0
            manager = self.genrm_managers[key]
            if getattr(manager, "get_engine_capacity", None) is not None:
                try:
                    capacity = ray.get(manager.get_engine_capacity.remote())
                    current = int(capacity.get("current", len(engines)))
                    occupied = int(capacity.get("occupied", current))
                    pending_cleanup = int(capacity.get("pending_cleanup", 0))
                except Exception as e:
                    self._logger.warning(f"GenRM capacity query failed for '{key}': {e}")
            instances[key] = {
                "engines": engine_views,
                "current": current,
                "ready": len(engines),
                "occupied": occupied,
                "pending_cleanup": pending_cleanup,
            }
        if list(instances) == [_DEFAULT_INSTANCE_KEY]:
            return {"service": "genrm", **instances[_DEFAULT_INSTANCE_KEY]}
        return {"service": "genrm", "instances": instances}

    @app.get("/scale_out/{request_id}", response_model=GenRMScaleStatusResponse)
    async def get_scale_out_status(self, request_id: str) -> GenRMScaleStatusResponse:
        return self._genrm_scale_status("scale_out", request_id)

    @app.get("/scale_in/{request_id}", response_model=GenRMScaleStatusResponse)
    async def get_scale_in_status(self, request_id: str) -> GenRMScaleStatusResponse:
        return self._genrm_scale_status("scale_in", request_id)

    @app.post("/scale_out/{request_id}/reconcile", response_model=GenRMScaleReconcileResponse)
    async def reconcile_scale_out(self, request_id: str) -> GenRMScaleReconcileResponse:
        """Retry unfinished cleanup of a scale-out (no re-scaling)."""
        return await self._reconcile_scale("scale_out", request_id)

    @app.post("/scale_in/{request_id}/reconcile", response_model=GenRMScaleReconcileResponse)
    async def reconcile_scale_in(self, request_id: str) -> GenRMScaleReconcileResponse:
        """Retry unfinished cleanup of a scale-in (fixed victim, no re-
        selection)."""
        return await self._reconcile_scale("scale_in", request_id)

    def _resolve_scale_model(self, model_name: Optional[str]) -> str:
        """Map the API's ``model_name`` onto a configured instance key."""
        if model_name in (None, "default") and len(self.genrm_managers) == 1:
            return next(iter(self.genrm_managers))
        return self._resolve_instance_key(None if model_name == "default" else model_name)

    def _genrm_engine_list(self, key: str) -> List[Tuple[str, int]]:
        """Observe the live engine list of one instance from its manager."""
        manager = self.genrm_managers[key]
        return list(ray.get(manager.get_engine_hosts_ports.remote()))

    def _genrm_capacity(self, key: str, ready: int) -> Dict[str, int]:
        """Read capacity from the real manager while preserving fake
        support."""
        manager = self.genrm_managers[key]
        if getattr(manager, "get_engine_capacity", None) is None:
            return {"current": ready, "ready": ready, "occupied": ready, "pending_cleanup": 0}
        try:
            data = ray.get(manager.get_engine_capacity.remote())
            return {
                "current": int(data.get("current", ready)),
                "ready": int(data.get("ready", ready)),
                "occupied": int(data.get("occupied", ready)),
                "pending_cleanup": int(data.get("pending_cleanup", 0)),
            }
        except Exception as e:
            self._logger.warning(f"GenRM capacity query failed for '{key}': {e}")
            return {"current": ready, "ready": ready, "occupied": ready, "pending_cleanup": 0}

    async def _genrm_scale(self, direction: str, request: GenRMScaleRequest) -> GenRMScaleResponse:
        try:
            key = self._resolve_scale_model(request.model_name)
            engines = self._genrm_engine_list(key)
            capacity = self._genrm_capacity(key, len(engines))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self._logger.error(f"GenRM engine discovery failed for model '{request.model_name}': {e}")
            raise HTTPException(status_code=503, detail="GenRM engine discovery failed")

        decision = self._scale_registry.submit(
            direction,
            model_name=key,
            target=request.num_replicas,
            timeout_secs=request.timeout_secs,
            idempotency_key=request.idempotency_key,
            current=capacity["current"],
            ready=capacity["ready"],
        )
        if decision.get("http", 200) != 200:
            raise HTTPException(
                status_code=decision["http"],
                detail=decision.get("detail") or decision.get("status", "request rejected"),
            )
        if decision["status"] == "PENDING" and decision.get("dispatch"):
            request_id = decision["request_id"]
            detail = self._dispatch_genrm_scale(direction, key, request_id, target=request.num_replicas)
            if detail is not None:
                self._scale_registry.set_detail(request_id, detail)
            return GenRMScaleResponse(
                status="PENDING",
                request_id=request_id,
                current=capacity["current"],
                ready=capacity["ready"],
                detail=detail,
            )
        # Replay of an existing operation (no dispatch) or a fresh/replayed
        # NOOP -- the latter carries no request_id by contract.
        return GenRMScaleResponse(
            status=decision["status"],
            request_id=decision.get("request_id"),
            current=decision.get("current"),
            ready=decision.get("ready"),
        )

    def _dispatch_genrm_scale(
        self, direction: str, model: str, request_id: str, target: Optional[int] = None
    ) -> Optional[str]:
        """Fire-and-forget handoff to the manager lifecycle hook.

        Returns ``None`` when the hook accepted the operation, else a detail
        string explaining why execution is deferred (operation stays PENDING).
        Real managers also receive the operation parameters via
        ``begin_scale_op``; fakes without that method keep the hook-only
        contract.
        """
        manager = self.genrm_managers[model]
        hook_name = "execute_genrm_scale_out" if direction == "scale_out" else "execute_genrm_scale_in"
        hook = getattr(manager, hook_name, None)
        if hook is None:
            return "manager_scale_not_implemented"
        try:
            begin = getattr(manager, "begin_scale_op", None)
            if begin is not None and target is not None:
                begin.remote(request_id, direction, target)
            hook.remote(request_id)
        except Exception as e:
            self._logger.error(f"GenRM scale hook {hook_name} failed for {request_id}: {e}")
            return "manager_scale_not_implemented"
        if getattr(manager, "get_scale_progress", None) is not None:
            # Real manager: watch its progress and drive the registry.
            asyncio.create_task(self._watch_scale_operation(direction, model, request_id))
        return None

    _SCALE_OUT_CHAIN = ["PENDING", "CREATING", "HEALTH_CHECKING", "READY", "ACTIVE"]
    _SCALE_IN_CHAIN = ["PENDING", "DRAINING", "REMOVING", "COMPLETED"]

    async def _watch_scale_operation(self, direction: str, model: str, request_id: str) -> None:
        """Drive the registry from the manager's physical lifecycle progress.

        The manager reports oscillating per-engine phases; this watcher maps
        them onto the registry's monotonic chain, closes the component-side
        admission cache for draining victims, proves the drain (in-flight count
        zero) and confirms it, then finishes the registry with the manager's
        snapshot counts. ``/scale_out|in/{id}`` status queries see only
        registry state, so this loop is the single writer.
        """
        manager = self.genrm_managers[model]
        terminal = {"ACTIVE", "PARTIAL", "FAILED"} if direction == "scale_out" else {"COMPLETED", "FAILED"}
        op = self._scale_registry.get_status(direction, request_id) or {}
        # timeout_secs is the operation's TOTAL timeout, counted from submit
        # time (registry created_at), not from watcher start.
        timeout_secs = float(op.get("timeout_secs") or 600.0)
        deadline = float(op.get("created_at") or time.time()) + timeout_secs
        try:
            while time.time() < deadline:
                progress = await asyncio.to_thread(self._manager_progress, manager, request_id)
                phase = progress.get("phase")

                if direction == "scale_in" and phase == "DRAINING":
                    victim = progress.get("victim")
                    if victim:
                        self._close_victim_admission(model, victim)
                        if self._victim_inflight_zero(model, victim):
                            await asyncio.to_thread(ray.get, manager.confirm_scale_drained.remote(request_id))

                # A manager terminal phase is only a *reported* result until
                # its lifecycle thread has stopped.  In particular, aborting
                # a placement-group wait can report FAILED long before the
                # worker which owns that PG has returned.  Fail closed when a
                # legacy/failed poll omits the flag.
                if phase in terminal and progress.get("physical_done") is True:
                    self._finish_scale_from_progress(direction, request_id, phase, progress)
                    return
                # Do not advance the public state machine into a terminal
                # state before the physical-completion fence above.  A
                # terminal registry state without cleanup_required releases
                # the per-model mutual-exclusion gate.
                if phase not in terminal:
                    self._advance_registry_towards(direction, request_id, phase)
                await asyncio.sleep(0.5)
            # Deadline exceeded: abort the physical lifecycle, then wait up to
            # 60s for the manager thread to observe the abort (PG waits and
            # engine init are not interruptible; a late healthy replica is
            # discarded there rather than published). Only then fail the
            # registry with the manager's last snapshot.
            await asyncio.to_thread(ray.get, manager.abort_scale_op.remote(request_id))
            progress = {}
            for _ in range(120):
                progress = await asyncio.to_thread(self._manager_progress, manager, request_id)
                if progress.get("physical_done") is True:
                    break
                await asyncio.sleep(0.5)
            if progress.get("physical_done") is not True:
                # The lifecycle thread still owns a PG/actor transition. Mark
                # this terminal operation dirty so registry mutual exclusion
                # remains in force until reconcile can prove cleanup.
                progress["cleanup_required"] = True
                progress.setdefault("error", "abort requested; physical lifecycle still stopping")
            self._finish_scale_from_progress(direction, request_id, "FAILED", progress or {})
        except Exception as e:
            self._logger.error(f"GenRM scale watcher for {request_id} crashed: {e}")
            self._finish_scale_from_progress(
                direction,
                request_id,
                "FAILED",
                {
                    "error": f"watcher crashed: {e}",
                    # A watcher failure gives us no proof that the manager
                    # has released actors or its placement group.  Keep the
                    # model blocked until reconcile obtains that proof.
                    "cleanup_required": True,
                },
            )

    def _manager_progress(self, manager: Any, request_id: str) -> dict:
        try:
            return ray.get(manager.get_scale_progress.remote(request_id), timeout=15)
        except Exception as e:
            self._logger.warning(f"GenRM scale progress poll failed for {request_id}: {e}")
            return {}

    def _advance_registry_towards(self, direction: str, request_id: str, phase: Optional[str]) -> None:
        """Advance the registry monotonically towards the manager's phase.

        Manager phases may oscillate per engine (CREATING -> HEALTH_CHECKING ->
        CREATING for the next one); the registry only ever moves forward along
        its legal transition chain. FAILED is legal from any live status,
        PARTIAL from HEALTH_CHECKING or READY.
        """
        if not phase or phase in ("PENDING",):
            return
        current = (self._scale_registry.get_status(direction, request_id) or {}).get("status", "PENDING")
        # Terminal phases are deliberately finished only by the watcher after
        # it has seen ``physical_done is True``.  See the fence in
        # _watch_scale_operation.
        if phase in {"FAILED", "PARTIAL", "ACTIVE", "COMPLETED"}:
            return
        chain = self._SCALE_OUT_CHAIN if direction == "scale_out" else self._SCALE_IN_CHAIN
        if phase not in chain or current not in chain:
            return
        target_idx = chain.index(phase)
        current_idx = chain.index(current)
        for idx in range(current_idx + 1, target_idx + 1):
            self._safe_advance(direction, request_id, chain[idx])

    def _safe_advance(self, direction: str, request_id: str, status: str) -> None:
        try:
            self._scale_registry.advance(request_id, status)
        except (KeyError, ValueError) as e:
            # Progress oscillation can propose an already-passed status; the
            # registry stays the authority, never crash the watcher on it.
            self._logger.debug(f"GenRM scale advance {request_id} -> {status} skipped: {e}")

    def _finish_scale_from_progress(self, direction: str, request_id: str, phase: str, progress: dict) -> None:
        try:
            self._scale_registry.finish(
                request_id,
                status=phase,
                current=int(progress.get("current", 0) or 0),
                ready=int(progress.get("ready", 0) or 0),
                created=int(progress.get("created", 0) or 0),
                removed=int(progress.get("removed", 0) or 0),
                failed=int(progress.get("failed", 0) or 0),
                cleanup_required=bool(progress.get("cleanup_required", False)),
                error_message=progress.get("error"),
            )
        except (KeyError, ValueError) as e:
            self._logger.error(f"GenRM scale finish {request_id} as {phase} rejected: {e}")
        model = self._scale_registry_operation_model(request_id)
        if model is not None:
            self._engine_caches[model].force_refresh()

    def _scale_registry_operation_model(self, request_id: str) -> Optional[str]:
        for direction in ("scale_out", "scale_in"):
            op = self._scale_registry.get_status(direction, request_id) or {}
            if op:
                return op.get("model_name")
        return None

    def _close_victim_admission(self, model: str, victim) -> None:
        """Drop the cached engine list so no new request picks the victim.

        The manager already excludes draining victims from its published list;
        this closes the component-side staleness window. After this call any
        pick refreshes from the manager and cannot select the victim.
        """
        host, port = victim[0], victim[1]
        cache = self._engine_caches[model]
        if cache.hosts_ports and (host, port) in [tuple(x) for x in cache.hosts_ports]:
            cache.force_refresh()

    def _victim_inflight_zero(self, model: str, victim) -> bool:
        host, port = victim[0], victim[1]
        return self._engine_inflight.get((model, host, port), 0) == 0

    def _genrm_scale_status(self, direction: str, request_id: str) -> GenRMScaleStatusResponse:
        result = self._scale_registry.get_status(direction, request_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Scale request {request_id} not found")
        return GenRMScaleStatusResponse(**result)

    async def _reconcile_scale(self, direction: str, request_id: str) -> GenRMScaleReconcileResponse:
        """Retry unfinished cleanup for one operation (RFC reconcile).

        Continues the original operation: no new request ID, no re-selection of
        a scale-in victim, no re-scaling. Idempotent -- an operation whose
        cleanup already completed replays as a no-op success. The terminal
        status and the prior ``removed`` count are preserved; only the
        ``cleanup_required`` gate is cleared once the manager confirms the
        resources were released.
        """
        op = self._scale_registry.get_status(direction, request_id)
        if op is None:
            raise HTTPException(status_code=404, detail=f"Scale request {request_id} not found")
        if not op.get("cleanup_required"):
            # Nothing withheld: replay the decision verbatim (idempotent).
            return GenRMScaleReconcileResponse(
                request_id=request_id,
                direction=direction,
                status=op["status"],
                cleanup_required=False,
            )
        model = op["model_name"]
        manager = self.genrm_managers[model]
        reconcile = getattr(manager, "reconcile_scale_op", None)
        if reconcile is None:
            raise HTTPException(status_code=503, detail="manager does not support reconcile")
        if direction == "scale_in":
            # The manager retires the *fixed* victim only once the component
            # proves its in-flight count is zero. Refuse rather than break the
            # drain contract (and never pick a different victim).
            progress = await asyncio.to_thread(self._manager_progress, manager, request_id)
            victim = progress.get("victim")
            if victim and not self._victim_inflight_zero(model, victim):
                raise HTTPException(
                    status_code=409,
                    detail="victim still has in-flight requests; retry reconcile after drain",
                )
            if victim:
                self._close_victim_admission(model, victim)
        result = await asyncio.to_thread(ray.get, reconcile.remote(request_id))
        if not result.get("known"):
            raise HTTPException(status_code=503, detail="manager lost lifecycle state; cleanup remains blocked")
        if result.get("in_progress"):
            raise HTTPException(status_code=409, detail="physical lifecycle is still stopping; retry reconcile")
        if not result.get("cleanup_required"):
            self._scale_registry.clear_cleanup(request_id)
            self._engine_caches[model].force_refresh()
        refreshed = self._scale_registry.get_status(direction, request_id) or op
        return GenRMScaleReconcileResponse(
            request_id=request_id,
            direction=direction,
            status=refreshed["status"],
            cleanup_required=bool(refreshed.get("cleanup_required", False)),
            victim_cleared=bool(result.get("victim_cleared", False)),
        )

    def get_genrm_manager(self, route_key: Optional[str] = None) -> Any:
        """Get one GenRM manager by route key.

        Omitting ``route_key`` remains supported when exactly one instance is
        configured.
        """
        return self.genrm_managers[self._resolve_instance_key(route_key)]

    def onload(self) -> None:
        """Load genRM model weights to GPU, for every instance."""
        self._logger.info("GenRM onload requested")
        ray.get([m.onload.remote() for m in self.genrm_managers.values()])

    def offload(self) -> None:
        """Offload genRM model weights from GPU, for every instance."""
        self._logger.info("GenRM offload requested")
        ray.get([m.offload.remote() for m in self.genrm_managers.values()])


# ── Compatibility wrapper for old imports ─────────────────────────────────
GENRM_ROLE = "genrm"


def register_genrm(config, algo: dict) -> list[str]:
    """Compatibility wrapper; optional-role wiring lives in ``relax.core``."""
    from relax.core.optional_roles import register_genrm as _register_genrm

    return _register_genrm(config, algo)
