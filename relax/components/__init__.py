# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Implementation package for Relax Servers.

This package provides:
    - Base classes for Server implementations
    - Concrete Server implementations (Actor, Rollout, Critic, Advantages)
    - Mixins for common functionality

Base Classes:
    - Base: Legacy base class for backward compatibility
    - ServiceState: Enum for service lifecycle states
    - ServiceStatus: Pydantic model for service status

Server Implementations:
    - Actor: Training actor server
    - Rollout: Rollout generation server
    - Advantages: Advantage computation server

Note:
    MetricsService has been moved to the `relax.metrics` package.
"""

from relax.components.base import (
    Base,
    ServiceState,
    ServiceStatus,
)
from relax.components.inference_gateway import InferenceGateway, InferenceGatewayDeployment


__all__ = [
    "Base",
    "ServiceState",
    "ServiceStatus",
    "InferenceGateway",
    "InferenceGatewayDeployment",
]
