# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Autoscaler configuration module.

This module defines the configuration dataclass and validation logic for the
autoscaler service. Configuration is loaded from a YAML file specified via
--autoscaler-config command-line argument.
"""

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ScaleOutPolicy:
    """Configuration for scale-out (expansion) decisions.

    Attributes:
        token_usage_threshold: Token usage ratio above which to consider scaling out.
        queue_depth_per_engine: Queue depth per engine that triggers scale-out.
        queue_time_p95_threshold: P95 queue time in seconds that triggers scale-out.
        ttft_p95_threshold: P95 time-to-first-token in seconds that triggers scale-out.
        condition_duration_secs: How long conditions must persist before triggering.
        max_delta: Maximum number of engines to add in a single scale-out.
    """

    token_usage_threshold: float = 0.85
    queue_depth_per_engine: int = 10
    queue_time_p95_threshold: float = 5.0
    ttft_p95_threshold: float = 10.0
    condition_duration_secs: float = 30.0
    max_delta: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaleOutPolicy":
        """Create ScaleOutPolicy from dictionary."""
        return cls(
            token_usage_threshold=data.get("token_usage_threshold", 0.85),
            queue_depth_per_engine=data.get("queue_depth_per_engine", 10),
            queue_time_p95_threshold=data.get("queue_time_p95_threshold", 5.0),
            ttft_p95_threshold=data.get("ttft_p95_threshold", 10.0),
            condition_duration_secs=data.get("condition_duration_secs", 30.0),
            max_delta=data.get("max_delta", 4),
        )


@dataclass
class ScaleInPolicy:
    """Configuration for scale-in (contraction) decisions.

    Attributes:
        token_usage_threshold: Token usage ratio below which to consider scaling in.
        queue_depth_threshold: Queue depth below which to consider scaling in.
        throughput_variance_threshold: Maximum throughput variance for stability.
        condition_duration_secs: How long ALL conditions must persist before triggering.
        max_delta: Maximum number of engines to remove in a single scale-in (conservative).
        projected_usage_max: Maximum projected usage after scale-in to allow removal.
    """

    token_usage_threshold: float = 0.3
    queue_depth_threshold: int = 0
    throughput_variance_threshold: float = 0.1
    condition_duration_secs: float = 120.0
    max_delta: int = 1
    projected_usage_max: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaleInPolicy":
        """Create ScaleInPolicy from dictionary."""
        return cls(
            token_usage_threshold=data.get("token_usage_threshold", 0.3),
            queue_depth_threshold=data.get("queue_depth_threshold", 0),
            throughput_variance_threshold=data.get("throughput_variance_threshold", 0.1),
            condition_duration_secs=data.get("condition_duration_secs", 120.0),
            max_delta=data.get("max_delta", 1),
            projected_usage_max=data.get("projected_usage_max", 0.5),
        )


@dataclass
class AutoscalerConfig:
    """Complete configuration for the Autoscaler service.

    This configuration controls the behavior of the autoscaler including
    engine bounds, cooldown periods, timing intervals, and scaling policies.

    Attributes:
        enabled: Whether autoscaler is enabled.
        min_engines: Minimum number of engines to maintain.
        max_engines: Maximum number of engines allowed.
        scale_out_cooldown_secs: Cooldown period after scale-out.
        scale_in_cooldown_secs: Cooldown period after scale-in.
        metrics_interval_secs: Interval between metrics collections.
        evaluation_interval_secs: Interval between scaling evaluations.
        condition_window_secs: Time window for condition history.
        min_coverage_scale_out: Minimum metrics coverage (reporting/active) required
            to allow scale-out. Lower (looser) than scale-in because a few engines
            reporting high load are enough to justify adding capacity.
        min_coverage_scale_in: Minimum metrics coverage required to allow scale-in.
            Defaults to 1.0 (full coverage) because removing capacity based on a
            partial sample can wrongly drop an engine that is actually overloaded.
        scale_out_request_timeout_secs: Optional per-request scale-out timeout.
            forwarded to the Rollout service as ``timeout_secs``. Defaults to None,
            which makes the server inherit its own ``--scale-out-timeout`` (300s) so
            default behavior is unchanged and large-model bring-up is not killed
            prematurely. Ops may set a shorter value to bound how long a stuck pending
            scale-out blocks subsequent decisions.
        rollout_service_url: URL of the Rollout service for scaling API calls.
        scale_out_policy: Policy configuration for scale-out decisions.
        scale_in_policy: Policy configuration for scale-in decisions.
    """

    enabled: bool = True
    min_engines: int = 1
    max_engines: int = 32
    scale_out_cooldown_secs: float = 60.0
    scale_in_cooldown_secs: float = 300.0
    metrics_interval_secs: float = 10.0
    evaluation_interval_secs: float = 30.0
    condition_window_secs: float = 60.0
    min_coverage_scale_out: float = 0.5
    min_coverage_scale_in: float = 1.0
    scale_out_request_timeout_secs: Optional[float] = None
    rollout_service_url: str = "http://localhost:8000/rollout"
    scale_out_policy: ScaleOutPolicy = field(default_factory=ScaleOutPolicy)
    scale_in_policy: ScaleInPolicy = field(default_factory=ScaleInPolicy)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.min_engines < 1:
            raise ValueError(f"min_engines must be >= 1, got {self.min_engines}")

        if self.max_engines < self.min_engines:
            raise ValueError(f"max_engines ({self.max_engines}) must be >= min_engines ({self.min_engines})")

        if self.metrics_interval_secs <= 0:
            raise ValueError(f"metrics_interval_secs must be > 0, got {self.metrics_interval_secs}")

        if self.evaluation_interval_secs < self.metrics_interval_secs:
            raise ValueError(
                f"evaluation_interval_secs ({self.evaluation_interval_secs}) should be >= "
                f"metrics_interval_secs ({self.metrics_interval_secs})"
            )

        if self.scale_out_cooldown_secs < 0:
            raise ValueError(f"scale_out_cooldown_secs must be >= 0, got {self.scale_out_cooldown_secs}")

        if self.scale_in_cooldown_secs < 0:
            raise ValueError(f"scale_in_cooldown_secs must be >= 0, got {self.scale_in_cooldown_secs}")

        if not 0 < self.min_coverage_scale_out <= 1:
            raise ValueError(f"min_coverage_scale_out must be in (0, 1], got {self.min_coverage_scale_out}")

        if not 0 < self.min_coverage_scale_in <= 1:
            raise ValueError(f"min_coverage_scale_in must be in (0, 1], got {self.min_coverage_scale_in}")

        if self.scale_out_request_timeout_secs is not None and self.scale_out_request_timeout_secs <= 0:
            raise ValueError(
                f"scale_out_request_timeout_secs must be None or > 0, got {self.scale_out_request_timeout_secs}"
            )

        if self.scale_out_policy.condition_duration_secs < 0:
            raise ValueError(
                f"scale_out_policy.condition_duration_secs must be >= 0, "
                f"got {self.scale_out_policy.condition_duration_secs}"
            )

        if self.scale_in_policy.condition_duration_secs < 0:
            raise ValueError(
                f"scale_in_policy.condition_duration_secs must be >= 0, "
                f"got {self.scale_in_policy.condition_duration_secs}"
            )

        if not 0 < self.scale_out_policy.token_usage_threshold <= 1:
            raise ValueError(
                f"scale_out_policy.token_usage_threshold must be in (0, 1], "
                f"got {self.scale_out_policy.token_usage_threshold}"
            )

        if not 0 < self.scale_in_policy.token_usage_threshold < 1:
            raise ValueError(
                f"scale_in_policy.token_usage_threshold must be in (0, 1), "
                f"got {self.scale_in_policy.token_usage_threshold}"
            )

        if self.scale_in_policy.token_usage_threshold >= self.scale_out_policy.token_usage_threshold:
            raise ValueError(
                f"scale_in_policy.token_usage_threshold ({self.scale_in_policy.token_usage_threshold}) "
                f"should be < scale_out_policy.token_usage_threshold "
                f"({self.scale_out_policy.token_usage_threshold}) to avoid thrashing"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str, rollout_service_url: Optional[str] = None) -> "AutoscalerConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            rollout_service_url: Optional override for rollout service URL.

        Returns:
            AutoscalerConfig instance with values from YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Autoscaler config file not found: {yaml_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        scale_out_policy = ScaleOutPolicy.from_dict(data.get("scale_out_policy", {}))
        scale_in_policy = ScaleInPolicy.from_dict(data.get("scale_in_policy", {}))

        return cls(
            enabled=data.get("enabled", True),
            min_engines=data.get("min_engines", 1),
            max_engines=data.get("max_engines", 32),
            scale_out_cooldown_secs=data.get("scale_out_cooldown_secs", 60.0),
            scale_in_cooldown_secs=data.get("scale_in_cooldown_secs", 300.0),
            metrics_interval_secs=data.get("metrics_interval_secs", 10.0),
            evaluation_interval_secs=data.get("evaluation_interval_secs", 30.0),
            condition_window_secs=data.get("condition_window_secs", 60.0),
            min_coverage_scale_out=data.get("min_coverage_scale_out", 0.5),
            min_coverage_scale_in=data.get("min_coverage_scale_in", 1.0),
            scale_out_request_timeout_secs=data.get("scale_out_request_timeout_secs", None),
            rollout_service_url=rollout_service_url
            or data.get("rollout_service_url", "http://localhost:8000/rollout"),
            scale_out_policy=scale_out_policy,
            scale_in_policy=scale_in_policy,
        )

    @classmethod
    def from_args(cls, args: Namespace) -> Optional["AutoscalerConfig"]:
        """Create configuration from command-line arguments.

        If --autoscaler-config is not provided or is None, returns None
        indicating autoscaler should not be enabled.

        Args:
            args: Parsed argument namespace containing --autoscaler-config argument.

        Returns:
            AutoscalerConfig instance if config path is provided, None otherwise.
        """
        config_path = getattr(args, "autoscaler_config", None)
        if config_path is None:
            return None

        rollout_service_url = getattr(args, "rollout_service_url", None)
        return cls.from_yaml(config_path, rollout_service_url)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "enabled": self.enabled,
            "min_engines": self.min_engines,
            "max_engines": self.max_engines,
            "scale_out_cooldown_secs": self.scale_out_cooldown_secs,
            "scale_in_cooldown_secs": self.scale_in_cooldown_secs,
            "metrics_interval_secs": self.metrics_interval_secs,
            "evaluation_interval_secs": self.evaluation_interval_secs,
            "condition_window_secs": self.condition_window_secs,
            "min_coverage_scale_out": self.min_coverage_scale_out,
            "min_coverage_scale_in": self.min_coverage_scale_in,
            "scale_out_request_timeout_secs": self.scale_out_request_timeout_secs,
            "rollout_service_url": self.rollout_service_url,
            "scale_out_policy": {
                "token_usage_threshold": self.scale_out_policy.token_usage_threshold,
                "queue_depth_per_engine": self.scale_out_policy.queue_depth_per_engine,
                "queue_time_p95_threshold": self.scale_out_policy.queue_time_p95_threshold,
                "ttft_p95_threshold": self.scale_out_policy.ttft_p95_threshold,
                "condition_duration_secs": self.scale_out_policy.condition_duration_secs,
                "max_delta": self.scale_out_policy.max_delta,
            },
            "scale_in_policy": {
                "token_usage_threshold": self.scale_in_policy.token_usage_threshold,
                "queue_depth_threshold": self.scale_in_policy.queue_depth_threshold,
                "throughput_variance_threshold": self.scale_in_policy.throughput_variance_threshold,
                "condition_duration_secs": self.scale_in_policy.condition_duration_secs,
                "max_delta": self.scale_in_policy.max_delta,
                "projected_usage_max": self.scale_in_policy.projected_usage_max,
            },
        }
