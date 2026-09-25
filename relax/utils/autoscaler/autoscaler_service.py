# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Autoscaler service for dynamic Rollout engine scaling.

This module implements the main AutoscalerService which is deployed as a Ray
Serve application. It periodically collects metrics from SGLang engines,
evaluates scaling conditions, and triggers scale-out/scale-in operations
through the Rollout service API.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from ray import serve

from relax.components.base import Base
from relax.utils.autoscaler.config import AutoscalerConfig, ServiceScalingPolicy
from relax.utils.autoscaler.metrics_collector import MetricsCollector
from relax.utils.autoscaler.scaling_decision import (
    ScalingAction,
    ScalingDecision,
    ScalingDecisionEngine,
    is_scale_request_terminal,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

app = FastAPI()


# ===================== API Models =====================


class ScaleHistoryItem(BaseModel):
    """Single scale history record."""

    request_id: Optional[str] = None
    action: str
    status: str
    triggered_at: float
    completed_at: Optional[float] = None
    from_engines: Optional[int] = None
    to_engines: Optional[int] = None
    delta: int = 0
    reason: str = ""
    triggered_conditions: List[str] = Field(default_factory=list)
    metrics_snapshot: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    # Carried through so the monitor buckets without re-parsing error_message.
    failure_categories: List[str] = Field(default_factory=list)


class ScaleHistoryResponse(BaseModel):
    """Response model for scale history endpoint."""

    history: List[ScaleHistoryItem]
    total_count: int
    action_filter: Optional[str] = None
    limit: int = 100


class AutoscalerStatusResponse(BaseModel):
    """Response model for autoscaler status endpoint."""

    enabled: bool
    running: bool
    current_engines: int
    min_engines: int
    max_engines: int
    last_scale_time: Optional[float] = None
    last_scale_action: Optional[str] = None
    last_decision: Optional[Dict[str, Any]] = None
    pending_requests: List[Dict[str, Any]] = Field(default_factory=list)
    recent_metrics: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    total_scale_operations: int = 0
    services: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ConditionStatusResponse(BaseModel):
    """Response model for condition status endpoint."""

    conditions: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]
    observations: Dict[str, Any] = Field(default_factory=dict)


class EnableRequest(BaseModel):
    """Request model for enable/disable endpoint."""

    enabled: bool = Field(..., description="Enable or disable autoscaler")


class ScaleOutPolicyUpdate(BaseModel):
    """Request model for updating scale-out policy."""

    token_usage_threshold: Optional[float] = Field(None, gt=0, le=1, description="Token usage threshold for scale-out")
    queue_depth_per_engine: Optional[int] = Field(None, ge=0, description="Queue depth per engine threshold")
    queue_time_p95_threshold: Optional[float] = Field(None, gt=0, description="P95 queue time threshold in seconds")
    ttft_p95_threshold: Optional[float] = Field(None, gt=0, description="P95 TTFT threshold in seconds")
    condition_duration_secs: Optional[float] = Field(None, ge=0, description="Condition duration before triggering")
    max_delta: Optional[int] = Field(None, ge=1, description="Maximum engines to add in single scale-out")


class ScaleInPolicyUpdate(BaseModel):
    """Request model for updating scale-in policy."""

    token_usage_threshold: Optional[float] = Field(None, gt=0, lt=1, description="Token usage threshold for scale-in")
    queue_depth_threshold: Optional[int] = Field(None, ge=0, description="Queue depth threshold for scale-in")
    throughput_variance_threshold: Optional[float] = Field(None, ge=0, description="Throughput variance threshold")
    condition_duration_secs: Optional[float] = Field(None, ge=0, description="Condition duration before triggering")
    max_delta: Optional[int] = Field(None, ge=1, description="Maximum engines to remove in single scale-in")
    projected_usage_max: Optional[float] = Field(
        None, gt=0, le=1, description="Maximum projected usage after scale-in"
    )


class ConfigUpdateRequest(BaseModel):
    """Request model for updating autoscaler configuration."""

    min_engines: Optional[int] = Field(None, ge=1, description="Minimum number of engines")
    max_engines: Optional[int] = Field(None, ge=1, description="Maximum number of engines")
    scale_out_cooldown_secs: Optional[float] = Field(None, ge=0, description="Cooldown after scale-out in seconds")
    scale_in_cooldown_secs: Optional[float] = Field(None, ge=0, description="Cooldown after scale-in in seconds")
    metrics_interval_secs: Optional[float] = Field(None, gt=0, description="Metrics collection interval in seconds")
    evaluation_interval_secs: Optional[float] = Field(None, gt=0, description="Evaluation interval in seconds")
    condition_window_secs: Optional[float] = Field(None, gt=0, description="Condition window in seconds")
    min_coverage_scale_out: Optional[float] = Field(
        None, gt=0, le=1, description="Minimum metrics coverage required to allow scale-out"
    )
    min_coverage_scale_in: Optional[float] = Field(
        None, gt=0, le=1, description="Minimum metrics coverage required to allow scale-in"
    )
    scale_out_request_timeout_secs: Optional[float] = Field(
        None, gt=0, description="Per-request scale-out timeout in seconds forwarded to the Rollout service"
    )
    rollout_service_url: Optional[str] = Field(None, description="Rollout service URL")
    service_targets: Optional[Dict[str, str]] = Field(
        None, description="Per-service target URLs, e.g. {rollout: url, genrm: url}"
    )
    service_policies: Optional[Dict[str, ServiceScalingPolicy]] = Field(
        None, description="Per-service scaling policy overrides"
    )
    scale_out_policy: Optional[ScaleOutPolicyUpdate] = Field(None, description="Scale-out policy updates")
    scale_in_policy: Optional[ScaleInPolicyUpdate] = Field(None, description="Scale-in policy updates")


class ConfigUpdateResponse(BaseModel):
    """Response model for config update endpoint."""

    status: str
    message: str
    config: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


# ===================== Autoscaler State =====================


@dataclass
class AutoscalerState:
    """Runtime state for the autoscaler."""

    enabled: bool = False
    running: bool = False
    last_scale_time: Optional[float] = None
    last_scale_action: Optional[ScalingAction] = None
    last_decision: Optional[ScalingDecision] = None
    pending_requests: List[Dict[str, Any]] = field(default_factory=list)
    scale_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1000))
    total_scale_operations: int = 0
    last_error: Optional[str] = None


@dataclass
class ServiceRuntime:
    """Independent mutable autoscaling state for one service target."""

    name: str
    config: AutoscalerConfig
    metrics_collector: MetricsCollector
    decision_engine: ScalingDecisionEngine
    state: AutoscalerState = field(default_factory=AutoscalerState)
    # GenRM's API is model-scoped.  First release deliberately supports one
    # instance per service target; a mixed instance response is rejected at
    # discovery rather than silently scaling an arbitrary "default" model.
    model_name: Optional[str] = None


# ===================== Autoscaler Service =====================


@serve.deployment
@serve.ingress(app)
class AutoscalerService(Base):
    """Ray Serve deployment for autoscaling Rollout engines.

    This service monitors SGLang engine metrics and automatically scales
    the engine pool based on configurable policies. It integrates with
    the Rollout service API to execute scaling operations.

    Features:
        - Multi-condition scale-out (any condition triggers)
        - All-condition scale-in (all conditions must be met)
        - Cooldown periods to prevent thrashing
        - Comprehensive observability endpoints

    Endpoints:
        GET /status: Get current autoscaler status
        POST /enable: Enable or disable autoscaler
        GET /conditions: Get current condition evaluation status
        GET /scale_history: Get history of scale operations
        GET /health: Health check endpoint
        GET /metrics_history: Get metrics history for debugging
        GET /config: Get current autoscaler configuration
        PATCH /config: Update autoscaler configuration (partial updates supported)
    """

    def __init__(
        self,
        healthy: Any,
        pg: Optional[Any],
        autoscaler_config: "AutoscalerConfig",
        role: str = "autoscaler",
    ) -> None:
        super().__init__()
        self.role = role
        self.config = autoscaler_config

        self._services: Dict[str, ServiceRuntime] = {}
        self._rebuild_service_runtimes()
        # Compatibility aliases: external callers and legacy tests still see
        # the original rollout runtime at these attributes.
        rollout = self._services["rollout"]
        self.metrics_collector = rollout.metrics_collector
        self.decision_engine = rollout.decision_engine
        self._state = rollout.state
        self._main_task: Optional[asyncio.Task] = None
        self._http_session: Optional[Any] = None  # aiohttp.ClientSession

        logger.info(
            f"AutoscalerService initialized: min={self.config.min_engines}, "
            f"max={self.config.max_engines}, enabled={self.config.enabled}"
        )

    def _rebuild_service_runtimes(self) -> None:
        """Construct independent policy/decision state for every configured
        target."""
        names = {"rollout", *self.config.service_targets}
        previous = getattr(self, "_services", {})
        runtimes: Dict[str, ServiceRuntime] = {}
        for name in sorted(names):
            policy = self.config.get_effective_policies(name)
            effective = replace(
                self.config,
                min_engines=policy.min_engines,
                max_engines=policy.max_engines,
                scale_out_policy=policy.scale_out_policy,
                scale_in_policy=policy.scale_in_policy,
            )
            old = previous.get(name)
            collector = old.metrics_collector if old else MetricsCollector(effective)
            # Keep collected history across a config PATCH, but make its
            # subsequent polling cadence use the newly resolved service policy.
            collector.config = effective
            runtimes[name] = ServiceRuntime(
                name=name,
                config=effective,
                metrics_collector=collector,
                decision_engine=ScalingDecisionEngine(effective),
                state=old.state if old else AutoscalerState(),
            )
        self._services = runtimes

    def _runtime(self, runtime: Optional[ServiceRuntime]) -> ServiceRuntime:
        """Return a runtime, including the lightweight legacy test shape."""
        if runtime is not None:
            return runtime
        services = getattr(self, "_services", None)
        if services is not None:
            return services["rollout"]
        return ServiceRuntime(
            name="rollout",
            config=self.config,
            metrics_collector=getattr(self, "metrics_collector", MetricsCollector(self.config)),
            decision_engine=getattr(self, "decision_engine", ScalingDecisionEngine(self.config)),
            state=self._state,
        )

    async def start(self) -> None:
        """Start the autoscaler service."""
        if self._main_task is not None and not self._main_task.done():
            logger.warning("Autoscaler already running")
            return

        for runtime in self._services.values():
            await runtime.metrics_collector.start()
        self._http_session = await self._create_http_session()

        for runtime in self._services.values():
            runtime.state.enabled = self.config.enabled
            runtime.state.running = True
        self._main_task = asyncio.ensure_future(self._main_loop())

        logger.info(f"Autoscaler service started, enabled={self._state.enabled}")

    async def stop(self) -> None:
        """Stop the autoscaler service."""
        for runtime in self._services.values():
            runtime.state.running = False
            runtime.state.enabled = False

        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        for runtime in self._services.values():
            await runtime.metrics_collector.stop()

        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

        logger.info("Autoscaler service stopped")

    async def _create_http_session(self) -> Any:
        """Create an aiohttp client session."""
        try:
            import aiohttp

            return aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0),
            )
        except ImportError:
            logger.error("aiohttp not installed, HTTP operations will fail")
            raise

    async def _main_loop(self) -> None:
        """Main autoscaler loop: collect metrics, evaluate, and scale."""
        logger.info("Autoscaler main loop started")

        while self._state.running:
            try:
                # Only evaluate if enabled
                if self._state.enabled:
                    await self._evaluate_and_scale()

                # Wait for next evaluation interval
                await asyncio.sleep(self.config.evaluation_interval_secs)

            except asyncio.CancelledError:
                logger.info("Autoscaler main loop cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in autoscaler loop: {e}")
                self._state.last_error = str(e)
                await asyncio.sleep(self.config.evaluation_interval_secs)

        logger.info("Autoscaler main loop exited")

    async def _evaluate_and_scale(self) -> None:
        """Evaluate each configured service with independent mutable state."""
        services = getattr(self, "_services", None)
        runtimes = services.values() if services is not None else [self._runtime(None)]
        for runtime in runtimes:
            try:
                await self._evaluate_service(runtime)
            except Exception as exc:
                runtime.state.last_error = str(exc)
                logger.exception(f"Autoscaler evaluation failed for service={runtime.name}: {exc}")

    async def _evaluate_service(self, runtime: ServiceRuntime) -> None:
        if getattr(self, "_services", None) is None:
            # Test-only legacy construction overrides these methods with their
            # former no-argument shape.
            await self._update_pending_requests()
            engines = await self._fetch_engines()
        else:
            await self._update_pending_requests(runtime)
            engines = await self._fetch_engines(runtime.name)
        if runtime.name == "genrm" and engines:
            runtime.model_name = engines[0].get("model_name")
        if not engines:
            logger.warning(f"No {runtime.name} engines found, skipping evaluation")
            runtime.decision_engine.reset_condition_trackers()
            return
        snapshot = await runtime.metrics_collector.collect_all(engines)
        runtime.metrics_collector.add_snapshot(snapshot, num_candidates=len(engines))
        aggregated = runtime.metrics_collector.get_aggregated_metrics()
        pending_out = sum(
            req.get("delta", 0)
            for req in runtime.state.pending_requests
            if req.get("action") == "scale_out" and not is_scale_request_terminal("scale_out", req.get("status"))
        )
        decision = runtime.decision_engine.evaluate(
            aggregated_metrics=aggregated,
            current_engines=len(engines) + pending_out,
            last_scale_time=runtime.state.last_scale_time,
            last_scale_action=runtime.state.last_scale_action,
            pending_requests=runtime.state.pending_requests,
        )
        runtime.state.last_decision = decision
        if decision.action == ScalingAction.SCALE_OUT:
            await self._execute_scale_out(decision, len(engines), runtime)
        elif decision.action == ScalingAction.SCALE_IN:
            await self._execute_scale_in(decision, len(engines), runtime)

    async def _fetch_engines(self, service: str = "rollout") -> List[Dict[str, str]]:
        """Fetch active engine list from Rollout service.

        Only engines with ``status == "active"`` and a usable URL are returned.
        Returns:
            List of dicts with 'id', 'url', and 'model_name' keys.
        """
        if self._http_session is None:
            return []

        url = f"{self.config.get_service_url(service)}/engines"

        try:
            async with self._http_session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch engines: HTTP {response.status}")
                    return []

                data = await response.json()
                engines = []

                if service == "genrm":
                    instances = data.get("instances") or {"default": data}
                    active_models = [
                        model_name for model_name, instance in instances.items() if instance.get("engines", [])
                    ]
                    if len(active_models) > 1:
                        logger.error(
                            "GenRM autoscaler target contains multiple model instances; "
                            "configure one target per model before enabling automatic scaling"
                        )
                        return []
                    for model_name, instance in instances.items():
                        for engine in instance.get("engines", []):
                            host, port = engine.get("host"), engine.get("port")
                            if host and port:
                                engines.append(
                                    {
                                        "id": f"{model_name}:{host}:{port}",
                                        "url": f"http://{host}:{port}",
                                        "model_name": model_name,
                                        "status": "active",
                                    }
                                )
                    return engines
                for model_name, model_info in data.get("models", {}).items():
                    for engine_group in model_info.get("engine_groups", []):
                        for engine in engine_group.get("engines", []):
                            if engine.get("status") != "active":
                                continue
                            if not engine.get("url"):
                                continue
                            engines.append(
                                {
                                    "id": f"engine_{engine.get('rank', 'unknown')}",
                                    "url": engine.get("url", ""),
                                    "model_name": model_name,
                                    "status": engine.get("status", "unknown"),
                                }
                            )

                return engines

        except Exception as e:
            logger.warning(f"Error fetching engines: {e}")
            return []

    async def _execute_scale_out(
        self, decision: ScalingDecision, current_engines: int, runtime: Optional[ServiceRuntime] = None
    ) -> None:
        runtime = self._runtime(runtime)
        if self._http_session is None:
            return

        target_count = current_engines + decision.delta
        url = f"{self.config.get_service_url(runtime.name)}/scale_out"
        payload: Dict[str, Any] = {
            "model_name": runtime.model_name or "default",
            "num_replicas": target_count,
        }
        if runtime.config.scale_out_request_timeout_secs is not None:
            payload["timeout_secs"] = runtime.config.scale_out_request_timeout_secs

        logger.info(
            f"[Autoscaler] Executing scale-out: {current_engines} -> {target_count} engines "
            f"(+{decision.delta}), reason: {decision.reason}"
        )

        try:
            async with self._http_session.post(url, json=payload) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    request_id = data.get("request_id")
                    status = data.get("status", "PENDING")

                    # NOOP request IDs are not persisted and must not enter pending.
                    if status == "NOOP":
                        logger.info(
                            f"[Autoscaler] Scale-out NOOP (idempotent no-op), not tracking as pending: "
                            f"request_id={request_id}"
                        )
                        self._record_noop("scale_out", decision, data, current_engines, target_count, runtime)
                        return

                    if status == "CONFLICT":
                        logger.warning(
                            "[Autoscaler] Scale-out reported CONFLICT with 2xx body, not tracking as pending"
                        )
                        return

                    logger.info(f"[Autoscaler] Scale-out request accepted: request_id={request_id}, status={status}")

                    runtime.state.pending_requests.append(
                        {
                            "request_id": request_id,
                            "action": "scale_out",
                            "triggered_at": time.time(),
                            "status": status,
                            "from_engines": current_engines,
                            "to_engines": target_count,
                            "delta": decision.delta,
                            "reason": decision.reason,
                            "triggered_conditions": decision.triggered_conditions,
                            "metrics_snapshot": decision.metrics_snapshot,
                        }
                    )
                    runtime.state.last_scale_time = time.time()
                    runtime.state.last_scale_action = ScalingAction.SCALE_OUT
                else:
                    text = await response.text()
                    logger.warning(f"[Autoscaler] Scale-out request failed: HTTP {response.status} - {text}")

        except Exception as e:
            logger.exception(f"[Autoscaler] Error executing scale-out: {e}")

    async def _execute_scale_in(
        self, decision: ScalingDecision, current_engines: int, runtime: Optional[ServiceRuntime] = None
    ) -> None:
        runtime = self._runtime(runtime)
        if self._http_session is None:
            return

        target_count = current_engines - decision.delta
        url = f"{self.config.get_service_url(runtime.name)}/scale_in"
        payload = {
            "model_name": runtime.model_name or "default",
            "num_replicas": target_count,
        }

        logger.info(
            f"[Autoscaler] Executing scale-in: {current_engines} -> {target_count} engines "
            f"(-{decision.delta}), reason: {decision.reason}"
        )

        try:
            async with self._http_session.post(url, json=payload) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    request_id = data.get("request_id")
                    status = data.get("status", "PENDING")

                    # NOOP request IDs are not persisted and must not enter pending.
                    if status == "NOOP":
                        logger.info(
                            f"[Autoscaler] Scale-in NOOP (idempotent no-op), not tracking as pending: "
                            f"request_id={request_id}"
                        )
                        self._record_noop("scale_in", decision, data, current_engines, target_count, runtime)
                        return

                    if status == "CONFLICT":
                        logger.warning(
                            "[Autoscaler] Scale-in reported CONFLICT with 2xx body, not tracking as pending"
                        )
                        return

                    logger.info(f"[Autoscaler] Scale-in request accepted: request_id={request_id}, status={status}")

                    runtime.state.pending_requests.append(
                        {
                            "request_id": request_id,
                            "action": "scale_in",
                            "triggered_at": time.time(),
                            "status": status,
                            "from_engines": current_engines,
                            "to_engines": target_count,
                            "delta": decision.delta,
                            "reason": decision.reason,
                            "triggered_conditions": decision.triggered_conditions,
                            "metrics_snapshot": decision.metrics_snapshot,
                        }
                    )
                    runtime.state.last_scale_time = time.time()
                    runtime.state.last_scale_action = ScalingAction.SCALE_IN
                else:
                    text = await response.text()
                    logger.warning(f"[Autoscaler] Scale-in request failed: HTTP {response.status} - {text}")

        except Exception as e:
            logger.exception(f"[Autoscaler] Error executing scale-in: {e}")

    def _record_noop(
        self,
        action: str,
        decision: ScalingDecision,
        data: Dict[str, Any],
        from_engines: int,
        to_engines: int,
        runtime: Optional[ServiceRuntime] = None,
    ) -> None:
        """Record a NOOP response without changing cooldown state."""
        now = time.time()
        record = {
            "request_id": data.get("request_id"),
            "action": action,
            "status": "NOOP",
            "triggered_at": now,
            "completed_at": now,
            "from_engines": from_engines,
            "to_engines": to_engines,
            "delta": decision.delta,
            "reason": decision.reason,
            "triggered_conditions": decision.triggered_conditions,
            "metrics_snapshot": decision.metrics_snapshot,
        }
        runtime = self._runtime(runtime)
        runtime.state.scale_history.appendleft(record)
        runtime.state.total_scale_operations += 1

    async def _update_pending_requests(self, runtime: Optional[ServiceRuntime] = None) -> None:
        runtime = self._runtime(runtime)
        if self._http_session is None:
            return

        completed = []

        for req in runtime.state.pending_requests:
            action = req.get("action", "scale_out")

            status = req.get("status")
            if is_scale_request_terminal(action, status) and not req.get("cleanup_required"):
                completed.append(req)
                continue

            try:
                endpoint = "scale_out" if action == "scale_out" else "scale_in"
                url = f"{self.config.get_service_url(runtime.name)}/{endpoint}/{req['request_id']}"

                async with self._http_session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        new_status = data.get("status")
                        req["status"] = new_status
                        req["error_message"] = data.get("error_message")
                        req["failure_categories"] = data.get("failure_categories") or []
                        req["actual_current"] = data.get("current")
                        req["actual_ready"] = data.get("ready")
                        req["cleanup_required"] = bool(data.get("cleanup_required", False))

                        if is_scale_request_terminal(action, new_status) and req["cleanup_required"]:
                            # A terminal dirty operation is still a hard
                            # per-service gate. Ask the server to reconcile
                            # the original operation; never issue a new scale
                            # request while it owns unresolved resources.
                            reconcile_url = f"{url}/reconcile"
                            async with self._http_session.post(reconcile_url) as reconcile_response:
                                if reconcile_response.status == 200:
                                    reconcile_data = await reconcile_response.json()
                                    req["cleanup_required"] = bool(reconcile_data.get("cleanup_required", True))
                        if is_scale_request_terminal(action, new_status) and not req["cleanup_required"]:
                            completed.append(req)
                            req["completed_at"] = time.time()
                            logger.info(
                                f"[Autoscaler] Scale request {req['request_id']} completed: "
                                f"status={new_status}, action={action}, "
                                f"from={req.get('from_engines')} -> to={req.get('to_engines')}"
                            )

            except Exception as e:
                logger.warning(f"Error checking request {req.get('request_id')}: {e}")

        for req in completed:
            runtime.state.pending_requests.remove(req)
            req["status"] = req.get("status", "UNKNOWN")
            runtime.state.scale_history.appendleft(req)
            runtime.state.total_scale_operations += 1

    # ===================== HTTP Endpoints =====================

    @app.get("/status", response_model=AutoscalerStatusResponse)
    async def get_autoscaler_status(self) -> AutoscalerStatusResponse:
        engines = await self._fetch_engines()

        # Collect real-time metrics if no history or for fresh status
        if engines and not self.metrics_collector.get_history():
            logger.info("[Autoscaler] No metrics history, collecting real-time metrics for /status")
            realtime_metrics = await self.metrics_collector.collect_all(engines)
            # Bind denominator to this call's engine count (see _evaluate_and_scale).
            self.metrics_collector.add_snapshot(realtime_metrics, num_candidates=len(engines))

        aggregated = self.metrics_collector.get_aggregated_metrics()

        services = {}
        for name, runtime in getattr(self, "_services", {}).items():
            service_engines = engines if name == "rollout" else await self._fetch_engines(name)
            service_metrics = runtime.metrics_collector.get_aggregated_metrics()
            services[name] = {
                "current_engines": len(service_engines),
                "min_engines": runtime.config.min_engines,
                "max_engines": runtime.config.max_engines,
                "last_decision": runtime.state.last_decision.to_dict() if runtime.state.last_decision else None,
                "pending_requests": list(runtime.state.pending_requests),
                "recent_metrics": service_metrics.to_dict(),
                "total_scale_operations": runtime.state.total_scale_operations,
            }
        return AutoscalerStatusResponse(
            enabled=self._state.enabled,
            running=self._state.running,
            current_engines=len(engines),
            min_engines=self.config.min_engines,
            max_engines=self.config.max_engines,
            last_scale_time=self._state.last_scale_time,
            last_scale_action=(self._state.last_scale_action.value if self._state.last_scale_action else None),
            last_decision=(self._state.last_decision.to_dict() if self._state.last_decision else None),
            pending_requests=self._state.pending_requests,
            recent_metrics=aggregated.to_dict(),
            config=self.config.to_dict(),
            total_scale_operations=self._state.total_scale_operations,
            services=services,
        )

    @app.get("/scale_history", response_model=ScaleHistoryResponse)
    async def get_scale_history(
        self,
        limit: int = 100,
        action: Optional[str] = None,
        service: str = "rollout",
    ) -> ScaleHistoryResponse:
        runtime = self._services.get(service)
        if runtime is None:
            raise ValueError(f"Unknown autoscaler service '{service}'")
        history = list(runtime.state.scale_history)

        if action:
            action = action.lower()
            if action not in ("scale_out", "scale_in"):
                action = None
            else:
                history = [h for h in history if h.get("action") == action]

        history = history[:limit]

        return ScaleHistoryResponse(
            history=[ScaleHistoryItem(**h) for h in history],
            total_count=len(runtime.state.scale_history),
            action_filter=action,
            limit=limit,
        )

    @app.post("/enable")
    async def set_enabled(self, request: EnableRequest) -> Dict[str, Any]:
        self._state.enabled = request.enabled

        if self._state.enabled and not self._state.running:
            await self.start()
        elif not self._state.enabled:
            logger.info("Autoscaler disabled by request")

        return {
            "status": "ok",
            "enabled": self._state.enabled,
            "message": f"Autoscaler {'enabled' if request.enabled else 'disabled'}",
        }

    @app.get("/conditions", response_model=ConditionStatusResponse)
    async def get_conditions(self, service: str = "rollout") -> ConditionStatusResponse:
        runtime = self._services.get(service)
        if runtime is None:
            raise ValueError(f"Unknown autoscaler service '{service}'")
        aggregated = runtime.metrics_collector.get_aggregated_metrics()
        conditions = runtime.decision_engine.get_condition_status(aggregated)
        observations = runtime.decision_engine.condition_observation()

        return ConditionStatusResponse(
            conditions=conditions,
            metrics=aggregated.to_dict(),
            observations=observations,
        )

    @app.get("/health")
    async def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._state.running else "stopped",
            "enabled": self._state.enabled,
            "service": "autoscaler",
        }

    @app.get("/metrics_history")
    async def get_metrics_history(self, limit: int = 10, service: str = "rollout") -> Dict[str, Any]:
        """Get metrics history for debugging.

        Args:
            limit: Maximum number of historical snapshots to return.

        Returns:
            Recent metrics history.
        """
        runtime = self._services.get(service)
        if runtime is None:
            raise ValueError(f"Unknown autoscaler service '{service}'")
        history = runtime.metrics_collector.get_history()
        return {
            "count": len(history),
            "snapshots": [
                {
                    "timestamp": h["timestamp"],
                    "engine_count": len(h["metrics"]),
                    "engines": {eid: m.to_dict() for eid, m in list(h["metrics"].items())[:5]},
                }
                for h in history[-limit:]
            ],
        }

    @app.post("/clear_history")
    async def clear_history(self, service: str = "rollout") -> Dict[str, Any]:
        """Clear metrics history.

        Returns:
            Confirmation message.
        """
        runtime = self._services.get(service)
        if runtime is None:
            raise ValueError(f"Unknown autoscaler service '{service}'")
        runtime.metrics_collector.clear_history()
        return {"status": "ok", "message": f"Metrics history cleared for {service}"}

    @app.get("/config")
    async def get_config(self) -> Dict[str, Any]:
        """Get current autoscaler configuration.

        Returns:
            Current configuration as dictionary.
        """
        return {
            "status": "ok",
            "config": self.config.to_dict(),
        }

    @app.patch("/config", response_model=ConfigUpdateResponse)
    async def update_config(self, request: ConfigUpdateRequest) -> ConfigUpdateResponse:
        """Update autoscaler configuration.

        Allows partial updates to configuration values. Only provided fields
        will be updated; others remain unchanged.

        Args:
            request: Configuration update request with optional fields.

        Returns:
            Updated configuration and any warnings.
        """
        warnings: List[str] = []
        updates_made: List[str] = []

        if request.min_engines is not None:
            if request.max_engines is not None and request.min_engines > request.max_engines:
                raise ValueError(
                    f"min_engines ({request.min_engines}) cannot be greater than max_engines ({request.max_engines})"
                )
            if request.min_engines > self.config.max_engines:
                raise ValueError(
                    f"min_engines ({request.min_engines}) cannot be greater than current max_engines ({self.config.max_engines})"
                )
            self.config.min_engines = request.min_engines
            updates_made.append(f"min_engines={request.min_engines}")

        if request.max_engines is not None:
            if request.max_engines < self.config.min_engines:
                raise ValueError(
                    f"max_engines ({request.max_engines}) cannot be less than current min_engines ({self.config.min_engines})"
                )
            self.config.max_engines = request.max_engines
            updates_made.append(f"max_engines={request.max_engines}")

        if request.scale_out_cooldown_secs is not None:
            self.config.scale_out_cooldown_secs = request.scale_out_cooldown_secs
            updates_made.append(f"scale_out_cooldown_secs={request.scale_out_cooldown_secs}")

        if request.scale_in_cooldown_secs is not None:
            self.config.scale_in_cooldown_secs = request.scale_in_cooldown_secs
            updates_made.append(f"scale_in_cooldown_secs={request.scale_in_cooldown_secs}")

        if request.metrics_interval_secs is not None:
            if request.evaluation_interval_secs is not None:
                if request.metrics_interval_secs > request.evaluation_interval_secs:
                    warnings.append(
                        f"metrics_interval_secs ({request.metrics_interval_secs}) should be <= "
                        f"evaluation_interval_secs ({request.evaluation_interval_secs})"
                    )
            elif request.metrics_interval_secs > self.config.evaluation_interval_secs:
                warnings.append(
                    f"metrics_interval_secs ({request.metrics_interval_secs}) should be <= "
                    f"evaluation_interval_secs ({self.config.evaluation_interval_secs})"
                )
            self.config.metrics_interval_secs = request.metrics_interval_secs
            updates_made.append(f"metrics_interval_secs={request.metrics_interval_secs}")

        if request.evaluation_interval_secs is not None:
            self.config.evaluation_interval_secs = request.evaluation_interval_secs
            updates_made.append(f"evaluation_interval_secs={request.evaluation_interval_secs}")

        if request.condition_window_secs is not None:
            self.config.condition_window_secs = request.condition_window_secs
            updates_made.append(f"condition_window_secs={request.condition_window_secs}")

        if request.min_coverage_scale_out is not None:
            self.config.min_coverage_scale_out = request.min_coverage_scale_out
            updates_made.append(f"min_coverage_scale_out={request.min_coverage_scale_out}")

        if request.min_coverage_scale_in is not None:
            self.config.min_coverage_scale_in = request.min_coverage_scale_in
            updates_made.append(f"min_coverage_scale_in={request.min_coverage_scale_in}")

        if request.scale_out_request_timeout_secs is not None:
            self.config.scale_out_request_timeout_secs = request.scale_out_request_timeout_secs
            updates_made.append(f"scale_out_request_timeout_secs={request.scale_out_request_timeout_secs}")

        if request.rollout_service_url is not None:
            self.config.rollout_service_url = request.rollout_service_url
            updates_made.append(f"rollout_service_url={request.rollout_service_url}")

        if request.service_targets is not None:
            self.config.service_targets = dict(request.service_targets)
            updates_made.append(f"service_targets={sorted(request.service_targets)}")

        if request.service_policies is not None:
            self.config.service_policies = dict(request.service_policies)
            updates_made.append(f"service_policies={sorted(request.service_policies)}")

        if request.scale_out_policy is not None:
            policy = request.scale_out_policy
            if policy.token_usage_threshold is not None:
                if self.config.scale_in_policy.token_usage_threshold >= policy.token_usage_threshold:
                    warnings.append(
                        f"scale_out_policy.token_usage_threshold ({policy.token_usage_threshold}) should be > "
                        f"scale_in_policy.token_usage_threshold ({self.config.scale_in_policy.token_usage_threshold}) "
                        "to avoid thrashing"
                    )
                self.config.scale_out_policy.token_usage_threshold = policy.token_usage_threshold
                updates_made.append(f"scale_out_policy.token_usage_threshold={policy.token_usage_threshold}")

            if policy.queue_depth_per_engine is not None:
                self.config.scale_out_policy.queue_depth_per_engine = policy.queue_depth_per_engine
                updates_made.append(f"scale_out_policy.queue_depth_per_engine={policy.queue_depth_per_engine}")

            if policy.queue_time_p95_threshold is not None:
                self.config.scale_out_policy.queue_time_p95_threshold = policy.queue_time_p95_threshold
                updates_made.append(f"scale_out_policy.queue_time_p95_threshold={policy.queue_time_p95_threshold}")

            if policy.ttft_p95_threshold is not None:
                self.config.scale_out_policy.ttft_p95_threshold = policy.ttft_p95_threshold
                updates_made.append(f"scale_out_policy.ttft_p95_threshold={policy.ttft_p95_threshold}")

            if policy.condition_duration_secs is not None:
                self.config.scale_out_policy.condition_duration_secs = policy.condition_duration_secs
                updates_made.append(f"scale_out_policy.condition_duration_secs={policy.condition_duration_secs}")

            if policy.max_delta is not None:
                self.config.scale_out_policy.max_delta = policy.max_delta
                updates_made.append(f"scale_out_policy.max_delta={policy.max_delta}")

        if request.scale_in_policy is not None:
            policy = request.scale_in_policy
            if policy.token_usage_threshold is not None:
                if policy.token_usage_threshold >= self.config.scale_out_policy.token_usage_threshold:
                    warnings.append(
                        f"scale_in_policy.token_usage_threshold ({policy.token_usage_threshold}) should be < "
                        f"scale_out_policy.token_usage_threshold ({self.config.scale_out_policy.token_usage_threshold}) "
                        "to avoid thrashing"
                    )
                self.config.scale_in_policy.token_usage_threshold = policy.token_usage_threshold
                updates_made.append(f"scale_in_policy.token_usage_threshold={policy.token_usage_threshold}")

            if policy.queue_depth_threshold is not None:
                self.config.scale_in_policy.queue_depth_threshold = policy.queue_depth_threshold
                updates_made.append(f"scale_in_policy.queue_depth_threshold={policy.queue_depth_threshold}")

            if policy.throughput_variance_threshold is not None:
                self.config.scale_in_policy.throughput_variance_threshold = policy.throughput_variance_threshold
                updates_made.append(
                    f"scale_in_policy.throughput_variance_threshold={policy.throughput_variance_threshold}"
                )

            if policy.condition_duration_secs is not None:
                self.config.scale_in_policy.condition_duration_secs = policy.condition_duration_secs
                updates_made.append(f"scale_in_policy.condition_duration_secs={policy.condition_duration_secs}")

            if policy.max_delta is not None:
                self.config.scale_in_policy.max_delta = policy.max_delta
                updates_made.append(f"scale_in_policy.max_delta={policy.max_delta}")

            if policy.projected_usage_max is not None:
                self.config.scale_in_policy.projected_usage_max = policy.projected_usage_max
                updates_made.append(f"scale_in_policy.projected_usage_max={policy.projected_usage_max}")

        # Validate cross-field constraints after applying the patch.
        self.config.validate()

        self._rebuild_service_runtimes()
        rollout = self._services["rollout"]
        self.metrics_collector = rollout.metrics_collector
        self.decision_engine = rollout.decision_engine
        self._state = rollout.state

        message = f"Configuration updated: {', '.join(updates_made)}" if updates_made else "No changes applied"
        logger.info(f"[Autoscaler] Config update: {message}")
        if warnings:
            logger.warning(f"[Autoscaler] Config update warnings: {warnings}")

        return ConfigUpdateResponse(
            status="ok",
            message=message,
            config=self.config.to_dict(),
            warnings=warnings,
        )
