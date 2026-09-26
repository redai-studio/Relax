# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import requests


def _sanitize_for_json(value: Any) -> Any:
    """Recursively coerce a metric value into a JSON-serializable form.

    The JSON encoder used by ``requests`` rejects non-finite floats (``inf`` /
    ``nan``) and non-native objects such as ``torch.Tensor`` / ``numpy`` scalars,
    causing the whole metrics batch to be dropped. This normalizes:

    - ``inf`` / ``nan`` floats -> ``None`` (JSON ``null``)
    - tensors / ndarrays / numpy scalars -> Python scalars or lists
    - nested dicts / lists / tuples -> sanitized element-wise

    Unknown objects are returned unchanged (best effort).
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]

    # torch.Tensor / numpy.ndarray / numpy scalar all expose ``tolist`` which
    # returns Python scalars or nested lists; re-sanitize to catch inf/nan.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _sanitize_for_json(tolist())
        except Exception:
            return value
    return value


class MetricsClient:
    """Client for interacting with the Metrics Service.

    Provides a convenient interface for logging metrics and reporting them at
    the end of steps. Supports both synchronous and asynchronous operation
    modes.
    """

    def __init__(self, service_url: str = "http://localhost:8000/metrics"):
        """Initialize the metrics client.

        Args:
            service_url: URL of the metrics service
        """
        self.service_url = service_url.rstrip("/")
        self._buffer = defaultdict(list)
        self._lock = threading.Lock()

    def log_metric(
        self,
        step: int,
        metric_name: str,
        metric_value: Union[float, int, str, dict],
        tags: Optional[Dict[str, str]] = None,
        immediate: bool = False,
    ) -> bool:
        """Log a single metric.

        Args:
            step: The step number
            metric_name: Name of the metric
            metric_value: Value of the metric (can also be a timeline event dict)
            tags: Optional tags for the metric
            immediate: If True, send immediately. If False, buffer for batch reporting.

        Returns:
            True if successful, False otherwise
        """
        if immediate:
            # Send immediately
            return self._send_metric(step, metric_name, metric_value, tags)
        else:
            # Buffer for later batch reporting
            with self._lock:
                self._buffer[step].append({"name": metric_name, "value": metric_value, "tags": tags or {}})
            return True

    def log_metrics_batch(
        self,
        step: int,
        metrics: Dict[str, Union[float, int, str, dict, List[dict]]],
        tags: Optional[Dict[str, str]] = None,
        immediate: bool = False,
    ) -> bool:
        """Log multiple metrics at once.

        Args:
            step: The step number
            metrics: Dictionary of metric names to values (can include timeline events)
            tags: Optional tags for all metrics
            immediate: If True, send immediately. If False, buffer for batch reporting.

        Returns:
            True if successful, False otherwise
        """
        if immediate:
            # Send immediately as batch
            return self._send_metrics_batch(step, metrics, tags)
        else:
            # Buffer for later batch reporting
            with self._lock:
                for metric_name, metric_value in metrics.items():
                    self._buffer[step].append({"name": metric_name, "value": metric_value, "tags": tags or {}})
            return True

    def report_step(self, step: int) -> Dict[str, Any]:
        """Report all buffered metrics for a specific step.

        This is the main entry point for step-end reporting.
        All metrics buffered for the given step will be sent to the
        metrics service for reporting to all configured backends.

        Args:
            step: The step number to report

        Returns:
            Dict with status information from the service
        """
        # Get buffered metrics for this step
        with self._lock:
            metrics_list = self._buffer.get(step, [])

            # Convert to metrics dict
            metrics_dict = {}
            for metric in metrics_list:
                metrics_dict[metric["name"]] = metric["value"]

            # Clear buffer for this step
            if step in self._buffer:
                del self._buffer[step]

        if metrics_dict:
            # First, send the metrics as a batch
            batch_success = self._send_metrics_batch(step, metrics_dict)
            if not batch_success:
                return {"status": "error", "message": f"Failed to send metrics batch for step {step}"}

        # Send to service
        return self._send_report_step(step)

    def clear_buffer(self, step: Optional[int] = None):
        """Clear the metrics buffer.

        Args:
            step: Optional step number. If None, clears all buffered metrics.
        """
        with self._lock:
            if step is None:
                self._buffer.clear()
            elif step in self._buffer:
                del self._buffer[step]

    def get_buffered_metrics_count(self, step: Optional[int] = None) -> int:
        """Get the number of buffered metrics.

        Args:
            step: Optional step number. If None, returns total count for all steps.

        Returns:
            Number of buffered metrics
        """
        with self._lock:
            if step is None:
                return sum(len(metrics) for metrics in self._buffer.values())
            else:
                return len(self._buffer.get(step, []))

    def _send_metric(
        self,
        step: int,
        metric_name: str,
        metric_value: Union[float, int, str, dict],
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Send a single metric to the service.

        Args:
            step: The step number
            metric_name: Name of the metric
            metric_value: Value of the metric
            tags: Optional tags for the metric

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.service_url}/log_metric",
                json={
                    "step": step,
                    "metric_name": metric_name,
                    "metric_value": _sanitize_for_json(metric_value),
                    "tags": tags,
                },
                timeout=5,
            )
            if response.status_code != 200:
                print(f"[MetricsClient] log_metric failed: HTTP {response.status_code}, response: {response.text}")
                return False
            result = response.json()
            if result.get("status") != "success":
                print(f"[MetricsClient] log_metric failed: {result}")
                return False
            return True
        except Exception as e:
            print(f"[MetricsClient] log_metric exception: {e}")
            return False

    def _send_metrics_batch(
        self,
        step: int,
        metrics: Dict[str, Union[float, int, str, dict, List[dict]]],
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Send multiple metrics as a batch to the service.

        Args:
            step: The step number
            metrics: Dictionary of metric names to values
            tags: Optional tags for all metrics

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.service_url}/log_metrics_batch",
                json={"step": step, "metrics": _sanitize_for_json(metrics), "tags": tags},
                timeout=5,
            )
            if response.status_code != 200:
                print(
                    f"[MetricsClient] log_metrics_batch failed: HTTP {response.status_code}, response: {response.text}"
                )
                return False
            result = response.json()
            if result.get("status") != "success":
                print(f"[MetricsClient] log_metrics_batch failed: {result}")
                return False
            return True
        except Exception as e:
            print(f"[MetricsClient] log_metrics_batch exception: {e}")
            return False

    def _send_report_step(self, step: int) -> Dict[str, Any]:
        """Send a report step request to the service.

        Args:
            step: The step number
            metrics: Dictionary of metrics to report

        Returns:
            Dict with status information from the service
        """
        try:
            # Then, trigger reporting
            response = requests.post(f"{self.service_url}/report_step", json={"step": step}, timeout=10)

            if response.status_code != 200:
                print(f"[MetricsClient] report_step failed: HTTP {response.status_code}, response: {response.text}")
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
            result = response.json()
            if result.get("status") != "success":
                print(f"[MetricsClient] report_step failed: {result}")
            return result
        except Exception as e:
            print(f"[MetricsClient] report_step exception: {e}")
            return {"status": "error", "message": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the metrics service.

        Returns:
            Dict with health status information
        """
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"[MetricsClient] health_check failed: HTTP {response.status_code}, response: {response.text}")
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
            return response.json()
        except Exception as e:
            print(f"[MetricsClient] health_check exception: {e}")
            return {"status": "error", "message": str(e)}


# Global metrics client instance (optional)
_global_metrics_client: Optional[MetricsClient] = None


def get_metrics_client(service_url: str = "http://localhost:8000/metrics") -> MetricsClient:
    """Get the global metrics client singleton.

    Creates a new instance on first call and reuses it for subsequent calls.
    Useful for accessing metrics client from anywhere in the codebase.

    Note: If called with a different service_url after initialization,
    it will recreate the client with the new URL.

    Args:
        service_url: URL of the metrics service

    Returns:
        MetricsClient: The global metrics client instance
    """
    global _global_metrics_client
    if _global_metrics_client is None or _global_metrics_client.service_url != service_url:
        _global_metrics_client = MetricsClient(service_url)
    return _global_metrics_client
