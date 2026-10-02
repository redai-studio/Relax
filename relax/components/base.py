# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Base classes and mixins for Relax Service implementations.

This module provides:
- ServiceState: Enum for service lifecycle states
- ServiceStatus: Pydantic model for service status reporting
- Base: Legacy base class for backward compatibility
"""

import threading


try:
    from enum import StrEnum
except ImportError:
    # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from typing import Any, Dict, Optional

from pydantic import BaseModel


class ServiceState(StrEnum):
    """Enum representing the lifecycle states of a Service.

    States:
        CREATED: Service instance created but not yet initialized
        STARTING: Service is in the process of starting
        RUNNING: Service is actively running
        PAUSED: Service is temporarily paused
        STOPPING: Service is in the process of stopping
        STOPPED: Service has been stopped
        ERROR: Service encountered an error
    """

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ServiceStatus(BaseModel):
    """Pydantic model representing the status of a Service.

    Attributes:
        role: The role name of the service (e.g., "actor", "rollout")
        state: Current lifecycle state of the service
        step: Current training/rollout step
        healthy: Whether the service is healthy
        error_message: Optional error message if in ERROR state
        metrics: Dictionary of runtime metrics
    """

    role: str
    state: ServiceState
    step: int
    healthy: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class Base:
    """Legacy base class for backward compatibility.

    This class maintains the original interface used by existing Server
    implementations. New implementations should consider using ServiceBase
    for the full lifecycle interface.

    Attributes:
        _healthy: Health status flag
        step: Current processing step
        _logger_instance: Cached logger instance (lazily created)
    """

    # Class-level cache for loggers to avoid repeated creation
    _logger_cache: Dict[str, Any] = {}

    def __init__(self) -> None:
        self._healthy = True
        self.step = 0
        self._logger_instance = None
        self._lock = threading.Lock()

    def __del__(self) -> None:
        # Ray Serve calls the destructor on replica shutdown (normal stop,
        # global restart, in-place restart). Components that attached a
        # TransferQueue client must detach so a MooncakeStore segment
        # deregisters before client_ttl instead of leaving a stale endpoint.
        if getattr(self, "data_system_client", None) is None:
            return
        try:
            from relax.utils.tq.lifecycle import detach_tq_client

            detach_tq_client()
            self.data_system_client = None
        except Exception:  # destructor must never raise (interpreter shutdown)
            return

    @property
    def _logger(self):
        """Lazily create and cache a logger for this instance.

        This property creates the logger on first access to avoid pickle issues
        with Ray Serve. The logger is cached at the class level using the
        module name as key.

        Returns:
            Logger: A logger instance for this class's module
        """
        if self._logger_instance is None:
            # Get the actual module name from the subclass
            module_name = self.__class__.__module__
            if module_name not in Base._logger_cache:
                from relax.utils.logging_utils import get_logger

                Base._logger_cache[module_name] = get_logger(module_name)
            self._logger_instance = Base._logger_cache[module_name]
        return self._logger_instance

    def run(self):
        """Run method to be implemented by subclasses.

        Subclasses should implement a synchronous or asynchronous `run` method
        depending on their runtime semantics.

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def set_step(self, step):
        """Set the current processing step.

        Args:
            step: Step number to set
        """
        self.step = step

    def get_step(self):
        """Get the current processing step.

        Returns:
            int: Current step number
        """
        return self.step

    def get_status(self) -> ServiceStatus:
        """Get the current service status.

        This method provides compatibility with ServiceController by returning
        a ServiceStatus object.

        Returns:
            ServiceStatus: Current status
        """
        return ServiceStatus(
            role=getattr(self, "role", "unknown"),
            state=ServiceState.RUNNING if self._healthy else ServiceState.ERROR,
            step=self.step,
            healthy=self._healthy,
            error_message=None,
            metrics={},
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics.

        Returns:
            Dict[str, Any]: Empty dict by default, subclasses can override
        """
        return {}

    async def pause(self) -> None:
        """Pause the service (no-op for legacy base)."""
        pass

    async def resume(self) -> None:
        """Resume the service (no-op for legacy base)."""
        pass

    async def stop(self) -> None:
        """Stop the service (no-op for legacy base)."""
        pass
