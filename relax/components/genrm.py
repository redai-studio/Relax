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
from typing import Any, List, Optional, Union

import httpx
import ray
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.ray.placement_group import create_genrm_managers
from relax.utils.data.processing_utils import load_tokenizer
from relax.utils.env import Envs


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

    Returns the raw model response text.
    """

    response: str


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
            return GenerateResponse(response=response)

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
                    key, idx, host, port = self._pick_engine(route_key)
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
        """Metrics endpoint; reports per-instance stats."""
        instances = {
            key: {
                "model_path": spec["model_path"],
                "num_gpus": spec["num_gpus"],
                "num_engines": spec["num_gpus"] // spec["num_gpus_per_engine"],
            }
            for key, spec in self.instance_specs.items()
        }
        if list(instances) == [_DEFAULT_INSTANCE_KEY]:
            return {"service": "genrm", **instances[_DEFAULT_INSTANCE_KEY]}
        return {"service": "genrm", "instances": instances}

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
