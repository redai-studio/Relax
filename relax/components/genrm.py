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
from argparse import Namespace
from typing import Any, List, Optional, Union

import httpx
import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.ray.placement_group import create_genrm_managers
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.discovery import RoleSnapshot, role_snapshot_from_payloads
from relax.engine.inference.gateway import SNAPSHOT_FETCH_TIMEOUT_S, InferenceGateway
from relax.engine.inference.messages import genrm_payload
from relax.engine.inference.routing import RoutingError, select_model
from relax.utils.data.processing_utils import load_tokenizer
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


app = FastAPI()
logger = get_logger(__name__)

# Max concurrent in-flight requests per GenRM Serve replica. Ray Serve's default
# of 5 throttles judge dispatch and leaves the SGLang engines idle. The replica
# forwards asynchronously, but HTTP connection-pool assignment runs synchronously
# on its event loop. Keep admission bounded instead of queueing inside that pool.
GENRM_SERVE_MAX_ONGOING_REQUESTS = Envs.GENRM_SERVE_MAX_ONGOING_REQUESTS
_GENRM_HTTP_MAX_CONNECTIONS = 2048


def _validate_genrm_concurrency(max_ongoing_requests: int) -> None:
    """Reject pool oversubscription before allocating GenRM engines."""
    if not 0 < max_ongoing_requests <= _GENRM_HTTP_MAX_CONNECTIONS:
        raise ValueError(
            f"GENRM_SERVE_MAX_ONGOING_REQUESTS={max_ongoing_requests} must be positive and must not exceed "
            f"the GenRM HTTP pool capacity ({_GENRM_HTTP_MAX_CONNECTIONS}). "
            "Oversubscription can stall the event loop in HTTP connection-pool assignment and leave GPUs idle. "
            "Set GENRM_SERVE_MAX_ONGOING_REQUESTS=256 to queue excess requests in Ray Serve."
        )
    if max_ongoing_requests > 256:
        logger.warning(
            "GENRM_SERVE_MAX_ONGOING_REQUESTS=%s exceeds the recommended default of 256. "
            "Fitting within the HTTP pool capacity (%s) does not guarantee good throughput: "
            "pool scans run on the event loop. Monitor GenRM latency and CPU utilization when tuning this limit.",
            max_ongoing_requests,
            _GENRM_HTTP_MAX_CONNECTIONS,
        )


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
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response model for genRM generation.

    Returns the raw model response text.
    """

    response: str


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
        _validate_genrm_concurrency(GENRM_SERVE_MAX_ONGOING_REQUESTS)
        self._logger.info(
            "GenRM concurrency limits: max_ongoing_requests=%s, http_max_connections=%s",
            GENRM_SERVE_MAX_ONGOING_REQUESTS,
            _GENRM_HTTP_MAX_CONNECTIONS,
        )
        self.config = config
        self.healthy = healthy
        self.role = role

        # {route_key: GenRMManager handle}. Single-instance configs (the legacy
        # --genrm-model-path path) resolve to exactly {"__default__": manager}.
        self.genrm_managers = create_genrm_managers(config, pg, runtime_env=runtime_env)
        self.instance_specs = config._genrm_instances_resolved

        self._gateway = InferenceGateway(
            role,
            self._fetch_inference_snapshot,
            upstream_name="GenRM engine",
            request_client=InferenceRequestClient(role, self.get_inference_resources),
            proxy_client=httpx.AsyncClient(
                timeout=1800,
                limits=httpx.Limits(
                    max_connections=_GENRM_HTTP_MAX_CONNECTIONS,
                    max_keepalive_connections=_GENRM_HTTP_MAX_CONNECTIONS,
                    keepalive_expiry=600,
                ),
            ),
        )
        self._logger.info(f"GenRM service initialized successfully: instances={list(self.genrm_managers)}")
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

    async def get_inference_resources(self) -> dict:
        snapshot = await asyncio.to_thread(self._fetch_inference_snapshot)
        return {"snapshot": snapshot.to_dict(), "managers": self.genrm_managers, "specs": self.instance_specs}

    def _fetch_inference_snapshot(self) -> RoleSnapshot:
        payloads = ray.get(
            [manager.get_inference_snapshot.remote() for manager in self.genrm_managers.values()],
            timeout=SNAPSHOT_FETCH_TIMEOUT_S,
        )
        return role_snapshot_from_payloads(
            self.role,
            dict(zip(self.genrm_managers, payloads, strict=True)),
            default_model=_DEFAULT_INSTANCE_KEY if _DEFAULT_INSTANCE_KEY in self.genrm_managers else None,
        )

    @app.post("/generate")
    async def generate_endpoint(self, request: Request):
        """``/generate`` entry point.

        A body with ``messages`` is the long-standing GenRM contract and is
        handled by ``generate`` unchanged. Anything else is a native engine
        payload, forwarded to a replica chosen from the topology snapshot.
        """
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
        if isinstance(body, dict) and "messages" in body:
            try:
                legacy_request = GenerateRequest.model_validate(body)
            except ValidationError as e:
                raise RequestValidationError(e.errors())
            return await self.generate(legacy_request)
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid request body: expected a JSON object")
        return await self._gateway.generate(body)

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
            output = await self._call_engine(
                request.route_key, request.messages, request.sampling_params, model=request.model
            )
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
            detail = f"No GenRM instance registered for route_key={key!r}; available={list(self.genrm_managers)}"
            if route_key is None:
                raise RuntimeError(detail)  # Preserve ambiguous programmatic lookup.
            raise HTTPException(status_code=400, detail=detail)
        return key

    async def _call_engine(
        self,
        route_key: Optional[str],
        messages: list,
        sampling_params: Optional[dict] = None,
        *,
        model: Optional[str] = None,
    ) -> dict:
        try:
            resources = await self._gateway._requests.resources()
            selected = select_model(RoleSnapshot.from_dict(resources["snapshot"]), model, route_key)
        except RoutingError as exc:
            raise HTTPException(
                exc.status_code, str(exc), headers={"Retry-After": "5"} if exc.status_code == 503 else None
            ) from exc
        payload = await genrm_payload(
            self.tokenizers[selected.name],
            self.instance_specs[selected.name]["sampling_config"],
            [message.model_dump() if isinstance(message, BaseModel) else message for message in messages],
            sampling_params,
        )
        for attempt in range(1, Envs.GENRM_ENGINE_RETRY_ATTEMPTS + 1):
            try:
                return await self._gateway.generate({"model": selected.name, **payload})
            except HTTPException as exc:
                if exc.status_code < 500 or attempt >= Envs.GENRM_ENGINE_RETRY_ATTEMPTS:
                    raise
                await asyncio.sleep(0.3 * attempt)

    @app.get("/engines")
    async def get_engines(self) -> dict:
        return await self._gateway.engines()

    @app.get("/v1/models")
    async def list_models(self) -> dict:
        return await self._gateway.models()

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(self, request: Request):
        return await self._gateway.chat_completions(await request.body(), dict(request.headers))

    @app.get("/health")
    async def health(self) -> dict:
        """Health check endpoint; reports per-instance status."""
        instances = {}
        for key, manager in self.genrm_managers.items():
            try:
                is_healthy = await asyncio.wait_for(manager.health_check.remote(), timeout=SNAPSHOT_FETCH_TIMEOUT_S)
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

    def onload(self, *, lease=None) -> None:
        """Load genRM model weights to GPU, for every instance."""
        self._logger.info("GenRM onload requested")
        ray.get([m.onload.remote(lease=lease) for m in self.genrm_managers.values()])

    def offload(self, *, lease=None) -> None:
        """Offload genRM model weights from GPU, for every instance."""
        self._logger.info("GenRM offload requested")
        ray.get([m.offload.remote(lease=lease) for m in self.genrm_managers.values()])


# ── Compatibility wrapper for old imports ─────────────────────────────────
GENRM_ROLE = "genrm"


def register_genrm(config, algo: dict) -> list[str]:
    """Compatibility wrapper; optional-role wiring lives in ``relax.core``."""
    from relax.core.optional_roles import register_genrm as _register_genrm

    return _register_genrm(config, algo)
