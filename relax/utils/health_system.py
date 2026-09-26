# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.utils.async_utils import run
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class ServiceHealthState:
    """Health state for a single service."""

    healthy: bool = True
    error: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    current_step: int = 0
    task_running: bool = False
    restart_count: int = 0
    # Set by ``report_error(fatal=True)`` for deterministic failures (e.g. SFT
    # data schema mismatches) that won't recover from a restart. The
    # HealthChecker short-circuits the retry ladder and exits the process when
    # it sees this flag, instead of grinding through ~12 in-place restarts and
    # 4 global restarts before ``_global_restart`` finally hits its limit.
    fatal: bool = False


@ray.remote
class HealthStatus:
    """Remote actor that stores and manages service health state.

    Single responsibility: Store and query service health status. Thread-safe
    operations via Ray actor.
    """

    HEARTBEAT_TIMEOUT = 120.0

    def __init__(self) -> None:
        """Initialize with empty health state."""
        self.state: Dict[str, ServiceHealthState] = {}

    def mark_healthy(self, role: str) -> None:
        """Mark a service as healthy.

        Args:
            role: Service role name (e.g., 'actor', 'rollout')
        """
        if role not in self.state:
            self.state[role] = ServiceHealthState()
        self.state[role].healthy = True
        self.state[role].error = None
        self.state[role].last_heartbeat = time.time()

    def mark_unhealthy(self, role: str, error: Optional[str] = None) -> None:
        """Mark a service as unhealthy.

        Args:
            role: Service role name (e.g., 'actor', 'rollout')
            error: Optional error message describing the failure
        """
        if role not in self.state:
            self.state[role] = ServiceHealthState()
        self.state[role].healthy = False
        if error:
            self.state[role].error = error

    def update_heartbeat(self, role: str, step: int = 0) -> None:
        """Update heartbeat for a service.

        Args:
            role: Service role name
            step: Current step of the service
        """
        if role not in self.state:
            self.state[role] = ServiceHealthState()
        self.state[role].last_heartbeat = time.time()
        if step > self.state[role].current_step:
            self.state[role].current_step = step
        if self.state[role].healthy:
            self.state[role].task_running = True

    def set_task_status(self, role: str, running: bool) -> None:
        """Update task running status for a service.

        Args:
            role: Service role name
            running: Whether the task is currently running
        """
        if role not in self.state:
            self.state[role] = ServiceHealthState()
        self.state[role].task_running = running

    def report_error(self, role: str, error: str, fatal: bool = False) -> None:
        """Report an error for a service.

        Args:
            role: Service role name
            error: Error message
            fatal: If True, the failure is non-recoverable (e.g. SFT data
                schema mismatch). The HealthChecker will skip restart and
                terminate the process instead of cycling through the retry
                ladder.
        """
        if role not in self.state:
            self.state[role] = ServiceHealthState()
        self.state[role].healthy = False
        self.state[role].error = error
        self.state[role].task_running = False
        if fatal:
            self.state[role].fatal = True

    def get_service_health(self, role: str) -> Dict:
        """Get detailed health info for a service.

        Args:
            role: Service role name

        Returns:
            Dict with healthy, error, last_heartbeat, current_step, task_running
        """
        if role not in self.state:
            return {
                "healthy": True,
                "error": None,
                "last_heartbeat": time.time(),
                "current_step": 0,
                "task_running": False,
                "fatal": False,
            }
        state = self.state[role]
        return {
            "healthy": state.healthy,
            "error": state.error,
            "last_heartbeat": state.last_heartbeat,
            "current_step": state.current_step,
            "task_running": state.task_running,
            "fatal": state.fatal,
        }

    def get_all_health(self) -> Dict[str, Dict]:
        """Get health info for all services.

        Returns:
            Dict mapping role to health info
        """
        result = {}
        for role in self.state:
            result[role] = self.get_service_health(role)
        return result

    def get_unhealthy_services(self) -> List[str]:
        """Get list of currently unhealthy services.

        Returns:
            List of service role names with unhealthy status (not True).
        """
        return [role for role, state in self.state.items() if not state.healthy]

    def get_stale_services(self) -> List[str]:
        """Get services that have stale heartbeat (timeout).

        Returns:
            List of service roles with stale heartbeat
        """
        current_time = time.time()
        stale = []
        for role, state in self.state.items():
            if state.task_running and (current_time - state.last_heartbeat) > self.HEARTBEAT_TIMEOUT:
                stale.append(role)
        return stale

    def get_current_step(self, role: str) -> int:
        """Get the current step for a service.

        Args:
            role: Service role name

        Returns:
            Current step (0 if not found)
        """
        if role in self.state:
            return self.state[role].current_step
        return 0

    def increment_restart_count(self, role: str) -> int:
        """Increment and return restart count for a service.

        Args:
            role: Service role name

        Returns:
            New restart count
        """
        if role not in self.state:
            self.state[role] = ServiceHealthState()
        self.state[role].restart_count += 1
        return self.state[role].restart_count


class HealthChecker:
    """Background health checking thread that monitors services.

    Single responsibility: Periodically check health and trigger callbacks.
    Runs in a background daemon thread, decoupled from health state storage.

    Args:
        health_status: Remote HealthStatus actor for querying health
        on_unhealthy: Callback function(role: str) when service becomes unhealthy
        check_interval: Seconds between health checks (default: 1.0)
        on_fatal: Optional callback(role: str, error_msg: str) invoked when a
            service reports a fatal (non-recoverable) error. After the callback
            returns, the checker calls ``os._exit(1)`` to terminate the
            process, bypassing the restart ladder.
    """

    def __init__(
        self,
        health_status: HealthStatus,
        on_unhealthy: Callable[[str], None],
        check_interval: float = 1.0,
        on_fatal: Optional[Callable[[str, str], None]] = None,
    ):
        self.health_status = health_status
        self.on_unhealthy = on_unhealthy
        self.on_fatal = on_fatal
        self.check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background health check thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("Health checker thread already running")
            return

        self._stop_event.clear()

        def _runner():
            try:
                run(self._check_loop())
            except Exception as e:
                logger.error(f"Health checker error: {e}")

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()
        logger.info("Health checker started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop background health check thread.

        Args:
            timeout: Seconds to wait for thread to stop.
        """
        if self._thread is None:
            return

        self._stop_event.set()
        logger.debug("Signaling health checker to stop")
        self._thread.join(timeout)

        if not self._thread.is_alive():
            self._thread = None
            logger.info("Health checker stopped")
        else:
            logger.warning("Health checker did not stop within timeout")

    async def _check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            if self._stop_event.is_set():
                logger.debug("Health check loop stopping")
                break

            try:
                unhealthy = ray.get(self.health_status.get_unhealthy_services.remote())
                for role in unhealthy:
                    health_info = ray.get(self.health_status.get_service_health.remote(role))
                    error_msg = health_info.get("error", "Unknown error")
                    if health_info.get("fatal"):
                        # Skip the restart ladder entirely: this error is
                        # deterministic, retrying just delays the inevitable
                        # ``os._exit`` from ``_global_restart`` by ~12 wasted
                        # service restarts. Notify metrics, then exit hard.
                        logger.error(
                            f"Service {role} reported FATAL error: {error_msg}. "
                            f"Skipping restart and terminating process."
                        )
                        if self.on_fatal is not None:
                            try:
                                self.on_fatal(role, error_msg)
                            except Exception as cb_exc:
                                logger.exception(f"on_fatal callback raised, exiting anyway: {cb_exc}")
                        self._stop_event.set()
                        os._exit(1)
                    logger.warning(f"Service {role} is unhealthy: {error_msg}, triggering restart")
                    self.on_unhealthy(role)
                    # on_unhealthy may trigger _global_restart which sets
                    # _stop_event. Check immediately to avoid using stale
                    # Ray actor handles from the destroyed cluster.
                    if self._stop_event.is_set():
                        logger.info("Stop event set after on_unhealthy callback, exiting check loop")
                        return

                stale = ray.get(self.health_status.get_stale_services.remote())
                for role in stale:
                    logger.warning(f"Service {role} heartbeat stale (timeout), triggering restart")
                    ray.get(self.health_status.mark_unhealthy.remote(role, "Heartbeat timeout"))
                    self.on_unhealthy(role)
                    if self._stop_event.is_set():
                        logger.info("Stop event set after on_unhealthy callback (stale), exiting check loop")
                        return

            except Exception as e:
                if self._stop_event.is_set():
                    logger.info("Stop event set, ignoring health check error and exiting check loop")
                    return
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(self.check_interval)

    def is_running(self) -> bool:
        """Check if health checker is active."""
        return self._thread is not None and self._thread.is_alive()


class HealthManager:
    """Top-level health management coordinator for Controller.

    Composite responsibility: Owns and coordinates HealthStatus and HealthChecker.
    Single public interface for Controller to interact with health system.
    Internal components are decoupled.

    Args:
        check_interval: Seconds between health checks (default: 1.0)
    """

    def __init__(self, check_interval: float = 1.0, config: Optional[Any] = None):
        """Initialize health manager.

        Args:
            check_interval: Interval for periodic health checks.
        """
        self.status = HealthStatus.options(**with_control_plane_affinity(config)).remote()
        self._checker: Optional[HealthChecker] = None
        self._check_interval = check_interval
        logger.info("HealthManager initialized")

    def start(
        self,
        on_unhealthy: Callable[[str], None],
        on_fatal: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Start the health management system.

        Args:
            on_unhealthy: Callback function(role: str) when service becomes unhealthy.
            on_fatal: Optional callback(role, error_msg) invoked when a service
                reports a fatal (non-recoverable) error, immediately before the
                checker terminates the process.
        """
        if self._checker is not None:
            logger.warning("Health checker already running")
            return

        self._checker = HealthChecker(
            health_status=self.status,
            on_unhealthy=on_unhealthy,
            check_interval=self._check_interval,
            on_fatal=on_fatal,
        )
        self._checker.start()
        logger.info("Health management system started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the health management system.

        Args:
            timeout: Seconds to wait for checker to stop.
        """
        if self._checker is None:
            return

        self._checker.stop(timeout)
        self._checker = None
        logger.info("Health management system stopped")

    def is_running(self) -> bool:
        """Check if health management is active.

        Returns:
            True if health checker is running.
        """
        return self._checker is not None and self._checker.is_running()

    def mark_healthy(self, role: str) -> None:
        """Mark a service as healthy.

        Args:
            role: Service role name.
        """
        self.status.mark_healthy.remote(role)

    def mark_unhealthy(self, role: str, error: Optional[str] = None) -> None:
        """Mark a service as unhealthy.

        Args:
            role: Service role name.
            error: Optional error message.
        """
        self.status.mark_unhealthy.remote(role, error)

    def update_heartbeat(self, role: str, step: int = 0) -> None:
        """Update heartbeat for a service.

        Args:
            role: Service role name.
            step: Current step.
        """
        self.status.update_heartbeat.remote(role, step)

    def report_error(self, role: str, error: str, fatal: bool = False) -> None:
        """Report an error for a service.

        Args:
            role: Service role name.
            error: Error message.
            fatal: If True, mark the failure as non-recoverable so the
                HealthChecker terminates the process instead of restarting.
        """
        self.status.report_error.remote(role, error, fatal=fatal)

    def get_service_health(self, role: str) -> Dict:
        """Get detailed health info for a service.

        Args:
            role: Service role name.

        Returns:
            Dict with health info.
        """
        return ray.get(self.status.get_service_health.remote(role))

    def get_all_health(self) -> Dict[str, Dict]:
        """Get health info for all services.

        Returns:
            Dict mapping role to health info.
        """
        return ray.get(self.status.get_all_health.remote())

    def get_current_step(self, role: str) -> int:
        """Get current step for a service.

        Args:
            role: Service role name.

        Returns:
            Current step.
        """
        return ray.get(self.status.get_current_step.remote(role))

    def increment_restart_count(self, role: str) -> int:
        """Increment and get restart count for a service.

        Args:
            role: Service role name.

        Returns:
            New restart count.
        """
        return ray.get(self.status.increment_restart_count.remote(role))
