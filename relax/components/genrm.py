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
from uuid import uuid4

import httpx
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.ray.inference_manager import ModelHandle
from relax.distributed.ray.placement_group import create_genrm_role
from relax.engine.inference.executor import InferenceRequestExecutor, assign_backend_rids
from relax.engine.inference.routing import RoutingError
from relax.engine.inference.types import Role
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
# the @serve.deployment decorator, which runs at import. The retry count
# (GENRM_ENGINE_RETRY_ATTEMPTS) has no such constraint; the request executor
# reads it at send time.

# Sentinel instance key for the legacy single-model config (--genrm-model-path)
# and for requests that don't pass a route_key.
_DEFAULT_INSTANCE_KEY = "__default__"


def _http_error(exc: RoutingError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=str(exc),
        headers={"Retry-After": "1"} if exc.status_code == 503 else None,
    )


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
        inference_manager_handle: Any = None,
    ) -> None:
        """Initialize GenRM service.

        Args:
            healthy: Remote health manager actor handle.
            pg: Placement group for resource allocation.
            num_gpus: Number of GPUs allocated (used by Service framework).
            config: Runtime configuration namespace.
            role: Role name (should be "genrm").
            runtime_env: Optional Ray runtime environment dict.
            inference_manager_handle: The task's inference manager, which holds
                every GenRM model's engines, state and lifecycle.
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

        # {route_key: handle}. Every instance is one GenRM model on the inference
        # manager; single-instance configs (the legacy --genrm-model-path path)
        # resolve to exactly {"__default__": handle}.
        self._inference_manager = inference_manager_handle
        self.genrm_managers = {
            key: ModelHandle(inference_manager_handle, Role.GENRM, key)
            for key in create_genrm_role(config, pg, inference_manager_handle)
        }
        self.instance_specs = config._genrm_instances_resolved
        # Request defaults and template arguments are part of each model's
        # registered configuration on the manager.
        self.model_configs = {
            key: ray.get(inference_manager_handle.model_spec.remote(Role.GENRM, key)) for key in self.genrm_managers
        }

        self._logger.info(f"GenRM service initialized successfully: instances={list(self.genrm_managers)}")
        # Shared HTTP client for engine calls (avoids per-request connection overhead).
        # Raise pool limits well above httpx's default 100 so one replica can fan out
        # many concurrent engine requests; keepalive_expiry >> the default 5s so
        # idle-then-reused connections aren't reaped mid-burst (avoids ReadError/500).
        http_client = httpx.AsyncClient(
            timeout=1800,
            limits=httpx.Limits(
                max_connections=_GENRM_HTTP_MAX_CONNECTIONS,
                max_keepalive_connections=_GENRM_HTTP_MAX_CONNECTIONS,
                keepalive_expiry=600,
            ),
        )
        # Requests run the same admission, target, retry and abort protocol as
        # the Gateway's.
        self._executor = InferenceRequestExecutor(Role.GENRM, inference_manager_handle, http_client)

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
        key = self._resolve_instance_key(request.route_key)
        try:
            admitted = await self._executor.admit(uuid4().hex, model=key)
        except RoutingError as exc:
            self._logger.warning(f"GenRM request to instance '{key}' is not admitted: {exc}")
            raise _http_error(exc) from exc
        try:
            payload = await self.prepare_generate_payload(key, request.messages, request.sampling_params)
            assign_backend_rids(admitted, payload)
            output = await self._executor.post(admitted, "generate", payload)
        except RoutingError as exc:
            raise _http_error(exc) from exc
        except httpx.RequestError as exc:
            self._logger.error(f"GenRM generation failed (route_key={request.route_key}): {exc}")
            raise HTTPException(status_code=502, detail=f"GenRM engine request failed: {exc}") from exc
        finally:
            await self._executor.release(admitted)
        if output.is_error:
            self._logger.error(f"GenRM engine returned {output.status_code} (route_key={request.route_key})")
            raise HTTPException(status_code=output.status_code, detail=output.text)
        return GenerateResponse(response=output.json().get("text", "").strip())

    def _resolve_instance_key(self, route_key: Optional[str]) -> str:
        if route_key is None and len(self.genrm_managers) == 1:
            return next(iter(self.genrm_managers))
        key = route_key or _DEFAULT_INSTANCE_KEY
        if key not in self.genrm_managers:
            raise RuntimeError(
                f"No GenRM instance registered for route_key={key!r}; available={list(self.genrm_managers)}"
            )
        return key

    async def prepare_generate_payload(
        self, route_key: Optional[str], messages: list, sampling_params: Optional[dict] = None
    ) -> dict:
        """Adapt messages on CPU using the selected instance's existing
        tokenizer."""
        key = self._resolve_instance_key(route_key)
        messages = [message.model_dump() if isinstance(message, Message) else message for message in messages]
        model_config = self.model_configs[key]
        # ensure plain list — some tokenizers return BatchEncoding which is not JSON-serializable
        # Tokenization (chat-template render + encode) is synchronous CPU work; run it in a
        # worker thread so it does not block this replica's event loop. Fast (Rust) tokenizers
        # release the GIL during encode, so concurrent requests tokenize in parallel instead of
        # serializing — without this a single replica throttles dispatch and starves the engines.
        # Forward the model's chat_template_kwargs through to the jinja template
        # — e.g. `{"enable_thinking": false}` for Qwen3+ to suppress the default
        # <think> block. Keys unused by the template are silently dropped by
        # transformers, so this is safe across model families.
        input_ids = await asyncio.to_thread(
            self.tokenizers[key].apply_chat_template,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **model_config.chat_template_kwargs,
        )

        if not isinstance(input_ids, list):
            input_ids = (
                input_ids["input_ids"]
                if hasattr(input_ids, "__getitem__") and "input_ids" in input_ids
                else list(input_ids)
            )

        # Per-request sampling params override the model's defaults
        default_sampling = dict(model_config.sampling_defaults)
        if sampling_params:
            default_sampling.update(sampling_params)

        return {
            "input_ids": input_ids,
            "sampling_params": default_sampling,
        }

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
        """Get one GenRM model handle by route key.

        Omitting ``route_key`` remains supported when exactly one instance is
        configured.
        """
        return self.genrm_managers[self._resolve_instance_key(route_key)]

    def onload(self) -> None:
        """Load genRM model weights to GPU, for every instance."""
        self._logger.info("GenRM onload requested")
        ray.get([m.activate.remote() for m in self.genrm_managers.values()])

    def offload(self) -> None:
        """Offload genRM model weights from GPU, for every instance."""
        self._logger.info("GenRM offload requested")
        ray.get([m.deactivate.remote() for m in self.genrm_managers.values()])
