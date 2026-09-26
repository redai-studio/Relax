# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import threading
from argparse import Namespace
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import wandb
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.adapters.apprise import _AppriseAdapter
from relax.utils.metrics.adapters.clearml import _ClearMLAdapter
from relax.utils.metrics.adapters.tensorboard import _TensorboardAdapter
from relax.utils.metrics.adapters.wandb import _is_offline_mode
from relax.utils.metrics.timeline_trace import TimelineTraceAdapter


logger = get_logger(__name__)

app = FastAPI()


# Special key to mark timeline events in metrics
TIMELINE_EVENTS_KEY = "__timeline_events__"


# Pydantic models for request bodies
class LogMetricRequest(BaseModel):
    step: int
    metric_name: str
    metric_value: Union[float, int, str, dict, list[dict]]
    tags: Optional[Dict[str, str]] = None


class LogMetricsBatchRequest(BaseModel):
    step: int
    metrics: Dict[str, Union[float, int, str, dict, List[dict]]]
    tags: Optional[Dict[str, str]] = None


class ReportStepRequest(BaseModel):
    step: int


class ClearMetricsRequest(BaseModel):
    step: Optional[int] = None


class LogErrorRequest(BaseModel):
    error_message: str
    error_traceback: Optional[str] = None


class MetricsBuffer:
    """Thread-safe buffer for step-based metric aggregation."""

    def __init__(self):
        self._buffer = defaultdict(list)
        self._lock = threading.Lock()

    def add_metric(self, step: int, metric_name: str, metric_value: Any, tags: Optional[Dict[str, str]] = None):
        with self._lock:
            self._buffer[step].append({"name": metric_name, "value": metric_value, "tags": tags or {}})

    def get_metrics_for_step(self, step: int) -> List[Dict[str, Any]]:
        with self._lock:
            return self._buffer.get(step, [])

    def clear_step(self, step: int):
        with self._lock:
            if step in self._buffer:
                del self._buffer[step]

    def has_metrics_for_step(self, step: int) -> bool:
        with self._lock:
            return step in self._buffer and len(self._buffer[step]) > 0


def is_timeline_event(metric_value: Any) -> bool:
    """Check if a metric value is a timeline event (Chrome Trace Event
    format)."""
    if not isinstance(metric_value, list):
        return False
    if not metric_value:
        return True
    # Check for Chrome Trace Event required fields
    return "ph" in metric_value[0] and "ts" in metric_value[0]


METRICS_SERVE_MAX_ONGOING_REQUESTS = Envs.METRICS_SERVE_MAX_ONGOING_REQUESTS


@serve.deployment(max_ongoing_requests=METRICS_SERVE_MAX_ONGOING_REQUESTS)
@serve.ingress(app)
class MetricsService:
    """Centralized metrics collection and reporting service.

    Deployed as a Ray Serve endpoint, this service collects metrics from
    training/inference processes and forwards them to configured backends
    (TensorBoard, W&B, ClearML) at step boundaries.
    """

    def __init__(self, healthy: Any, pg: Optional[Any], config: Namespace, role: str = "metrics") -> None:
        self.config = config
        self.healthy = healthy
        self.role = role

        self.metrics_buffer = MetricsBuffer()
        self._adapters = {}

        if getattr(config, "use_tensorboard", False):
            try:
                self._adapters["tensorboard"] = _TensorboardAdapter(config)
                logger.info("TensorBoard adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TensorBoard adapter: {e}")

        if getattr(config, "use_clearml", False):
            try:
                self._adapters["clearml"] = _ClearMLAdapter(config)
                logger.info("ClearML adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ClearML adapter: {e}")

        if getattr(config, "notify_urls", None):
            try:
                self._adapters["apprise"] = _AppriseAdapter(config)
                logger.info("Apprise adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Apprise adapter: {e}")

        self._use_wandb = getattr(config, "use_wandb", False)
        if self._use_wandb:
            try:
                self._init_wandb(config)
                logger.info("W&B adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize W&B adapter: {e}")
                self._use_wandb = False

        # Initialize TimelineTrace adapter
        timeline_dump_dir = getattr(config, "timeline_dump_dir", None)
        if timeline_dump_dir:
            self._timeline_adapter = TimelineTraceAdapter(timeline_dump_dir)
            logger.info(f"TimelineTrace adapter enabled, dumping to: {timeline_dump_dir}")
        else:
            self._timeline_adapter = None

        logger.info(f"MetricsService initialized with adapters: {list(self._adapters.keys())}")

    @staticmethod
    def _init_wandb(config: Namespace) -> None:
        """Initialize W&B for the MetricsService.

        Unlike init_wandb_primary (designed for training workers with
        rank/group), MetricsService is a single Ray Serve replica that only
        needs basic project and run name configuration.
        """

        if config.wandb_mode:
            os.environ["WANDB_MODE"] = config.wandb_mode

        offline = _is_offline_mode(config)

        if (not offline) and getattr(config, "wandb_key", None) is not None:
            wandb.login(key=config.wandb_key, host=getattr(config, "wandb_host", None))

        project = getattr(config, "wandb_project", None) or getattr(config, "tb_project_name", None)
        run_name = getattr(config, "tb_experiment_name", None) or "metrics-service"

        init_kwargs = {
            "project": project,
            "name": run_name,
            "entity": getattr(config, "wandb_team", None),
        }

        if offline:
            init_kwargs["settings"] = wandb.Settings(mode="offline")

        wandb_dir = getattr(config, "wandb_dir", None)
        if wandb_dir:
            os.makedirs(wandb_dir, exist_ok=True)
            init_kwargs["dir"] = wandb_dir

        wandb.init(**init_kwargs)

        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("rollout/step")
        wandb.define_metric("rollout/*", step_metric="rollout/step")
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.define_metric("perf/*", step_metric="rollout/step")

    @app.post("/log_metric")
    async def log_metric(self, request: LogMetricRequest) -> Dict[str, Any]:
        try:
            # Check if this is a timeline event
            if is_timeline_event(request.metric_value):
                if self._timeline_adapter:
                    self._timeline_adapter.add_event_dicts([request.metric_value])
                logger.debug(f"Timeline event logged: {request.metric_name}")
            else:
                self.metrics_buffer.add_metric(request.step, request.metric_name, request.metric_value, request.tags)
                logger.debug(
                    f"Metric logged: step={request.step}, name={request.metric_name}, value={request.metric_value}"
                )
            return {"status": "success", "message": f"Metric {request.metric_name} logged for step {request.step}"}
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
            return {"status": "error", "message": str(e)}

    @app.post("/log_metrics_batch")
    async def log_metrics_batch(self, request: LogMetricsBatchRequest) -> Dict[str, Any]:
        try:
            timeline_events = []
            regular_metrics = {}

            for metric_name, metric_value in request.metrics.items():
                # Check if this is a timeline event
                if is_timeline_event(metric_value):
                    timeline_events += metric_value
                else:
                    regular_metrics[metric_name] = metric_value

            # Handle regular metrics
            if regular_metrics:
                for metric_name, metric_value in regular_metrics.items():
                    self.metrics_buffer.add_metric(request.step, metric_name, metric_value, request.tags)

            # Handle timeline events
            if timeline_events:
                if self._timeline_adapter:
                    self._timeline_adapter.add_event_dicts(timeline_events)
                logger.debug(f"Batch timeline events logged: step={request.step}, count={len(timeline_events)}")

            logger.debug(f"Batch metrics logged: step={request.step}, count={len(request.metrics)}")
            return {
                "status": "success",
                "message": f"{len(request.metrics)} metrics logged for step {request.step}",
            }
        except Exception as e:
            logger.error(f"Failed to log batch metrics: {e}")
            return {"status": "error", "message": str(e)}

    @app.post("/report_step")
    async def report_step(self, request: ReportStepRequest) -> Dict[str, Any]:
        try:
            step = request.step
            metrics = self.metrics_buffer.get_metrics_for_step(step)

            report_results = {}

            # Handle regular metrics reporting
            if metrics:
                metrics_dict = {}
                for metric in metrics:
                    metric_name = metric["name"]
                    metric_value = metric["value"]
                    metrics_dict[metric_name] = metric_value

                if self._use_wandb:
                    try:
                        wandb.log(metrics_dict, step=step)
                        report_results["wandb"] = "success"
                        logger.debug(f"Reported {len(metrics_dict)} metrics to W&B for step {step}")
                    except Exception as e:
                        report_results["wandb"] = f"error: {e}"
                        logger.exception(f"Failed to report to W&B: {e}")

                if "tensorboard" in self._adapters:
                    try:
                        self._adapters["tensorboard"].log(data=metrics_dict, step=step)
                        report_results["tensorboard"] = "success"
                        logger.debug(f"Reported {len(metrics_dict)} metrics to TensorBoard for step {step}")
                    except Exception as e:
                        report_results["tensorboard"] = f"error: {e}"
                        logger.exception(f"Failed to report to TensorBoard: {e}")

                if "clearml" in self._adapters:
                    try:
                        self._adapters["clearml"].log(data=metrics_dict, step=step)
                        report_results["clearml"] = "success"
                        logger.debug(f"Reported {len(metrics_dict)} metrics to ClearML for step {step}")
                    except Exception as e:
                        report_results["clearml"] = f"error: {e}"
                        logger.exception(f"Failed to report to ClearML: {e}")

                if "apprise" in self._adapters:
                    try:
                        self._adapters["apprise"].log(data=metrics_dict, step=step)
                        report_results["apprise"] = "success"
                        logger.debug(f"Reported {len(metrics_dict)} metrics to Apprise for step {step}")
                    except Exception as e:
                        report_results["apprise"] = f"error: {e}"
                        logger.exception(f"Failed to report to Apprise: {e}")

            # Handle timeline events dumping
            # Timeline events are accumulated directly in the adapter, dump on step report
            if self._timeline_adapter and self._timeline_adapter.get_event_count() > 0:
                try:
                    event_count = self._timeline_adapter.get_event_count()
                    self._timeline_adapter.dump(step)
                    report_results["timeline"] = f"dumped {event_count} events"
                    logger.info(f"Dumped {event_count} timeline events for step {step}")
                except Exception as e:
                    report_results["timeline"] = f"error: {e}"
                    logger.exception(f"Failed to dump timeline: {e}")

            self.metrics_buffer.clear_step(step)
            logger.info(f"Reported {len(metrics)} metrics for step {step}")
            return {
                "status": "success",
                "message": f"Reported {len(metrics)} metrics for step {step}",
                "results": report_results,
            }

        except Exception as e:
            logger.error(f"Failed to report step {request.step}: {e}")
            return {"status": "error", "message": str(e)}

    @app.get("/health")
    async def health(self) -> Dict[str, Any]:
        """Health check endpoint for the metrics service."""
        return {
            "status": "healthy",
            "service": "metrics",
            "adapters": list(self._adapters.keys()),
            "wandb_enabled": self._use_wandb,
        }

    @app.get("/query_metrics")
    async def query_metrics(self, step: Optional[int] = None) -> Dict[str, Any]:
        return {"status": "success", "message": "Metrics retrieval not fully implemented", "metrics": {}}

    @app.post("/clear_metrics")
    async def clear_metrics(self, request: ClearMetricsRequest) -> Dict[str, Any]:
        return {"status": "success", "message": "Metrics cleared"}

    @app.post("/log_error")
    async def log_error(self, request: LogErrorRequest) -> Dict[str, Any]:
        """Log error with traceback to Apprise notification service.

        This endpoint is used to report runtime errors during training/inference.
        Only Apprise adapter is notified (not TensorBoard, W&B, or ClearML).

        Args:
            request: LogErrorRequest containing error_message and optional error_traceback

        Returns:
            Dict with status and message
        """
        try:
            # Only send to Apprise adapter
            if "apprise" in self._adapters:
                self._adapters["apprise"].log_error(
                    error_message=request.error_message, error_traceback=request.error_traceback
                )
                logger.info("Error notification sent to Apprise")
                return {"status": "success", "message": "Error notification sent"}
            else:
                logger.warning("Apprise adapter not configured, error notification not sent")
                return {"status": "warning", "message": "Apprise adapter not configured"}
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return {"status": "error", "message": str(e)}

    def get_service_metrics(self) -> Dict[str, Any]:
        return {
            "service": "metrics",
            "adapters_configured": list(self._adapters.keys()),
            "use_wandb": self._use_wandb,
            "buffer_size": len(self.metrics_buffer._buffer),
        }
