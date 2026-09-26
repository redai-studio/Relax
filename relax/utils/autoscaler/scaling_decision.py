# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scaling decision engine for autoscaler.

This module implements the decision logic for determining when to scale out
(add engines) or scale in (remove engines) based on collected metrics. It
implements a multi-condition evaluation strategy with cooldown periods and
conservative scaling behavior.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from relax.utils.autoscaler.config import AutoscalerConfig
from relax.utils.autoscaler.metrics_collector import AggregatedMetrics
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


SCALE_OUT_TERMINAL_STATUSES = frozenset({"ACTIVE", "PARTIAL", "FAILED", "CANCELLED"})
SCALE_IN_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED"})


def is_scale_request_terminal(action: Optional[str], status: Optional[str]) -> bool:
    """Return True if a scale request has reached a terminal status.

    Args:
        action: "scale_out" or "scale_in". Anything other than "scale_in" is
            treated as scale-out (matching the existing default in the service).
        status: The request status string (may be None).

    Returns:
        True if the status is terminal for the given action.
    """
    if action == "scale_in":
        return status in SCALE_IN_TERMINAL_STATUSES
    return status in SCALE_OUT_TERMINAL_STATUSES


class ScalingAction(Enum):
    """Enumeration of possible scaling actions.

    Attributes:
        NONE: No scaling action needed.
        SCALE_OUT: Add more engines.
        SCALE_IN: Remove engines.
    """

    NONE = "none"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"


@dataclass
class ScalingDecision:
    """Represents a scaling decision made by the decision engine.

    Attributes:
        action: The scaling action to take.
        delta: Number of engines to add (positive) or remove (positive).
        reason: Human-readable reason for the decision.
        confidence: Confidence level of the decision (0.0 - 1.0).
        metrics_snapshot: The metrics that triggered this decision.
        triggered_conditions: List of condition names that were triggered.
    """

    action: ScalingAction
    delta: int = 0
    reason: str = ""
    confidence: float = 1.0
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    triggered_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action.value,
            "delta": self.delta,
            "reason": self.reason,
            "confidence": self.confidence,
            "metrics_snapshot": self.metrics_snapshot,
            "triggered_conditions": self.triggered_conditions,
        }

    def is_noop(self) -> bool:
        """Check if this is a no-op decision."""
        return self.action == ScalingAction.NONE or self.delta == 0


class ScalingDecisionEngine:
    """Evaluates metrics and decides scaling actions.

    This class implements the core decision logic for autoscaling:
    - Scale-out: Triggered when ANY scale-out condition is met
    - Scale-in: Triggered when ALL scale-in conditions are met

    The engine respects cooldown periods and prevents overlapping
    scaling operations.

    Attributes:
        config: AutoscalerConfig instance.
        scale_out_conditions: Dict of condition name to evaluation function.
        scale_in_conditions: Dict of condition name to evaluation function.
    """

    def __init__(self, config: AutoscalerConfig, *, clock: Callable[[], float] = time.monotonic):
        """Initialize the decision engine.

        Args:
            config: Autoscaler configuration.
            clock: Monotonic clock used to measure condition duration.
        """
        self.config = config

        self._clock = clock
        self._condition_state: Dict[tuple, Dict[str, Any]] = {}
        # A longer gap represents an unobserved interval and breaks the streak.
        self._max_gap_secs = 2.0 * config.evaluation_interval_secs

        # Define scale-out conditions (any one triggers scale-out)
        self.scale_out_conditions: Dict[str, Callable[[AggregatedMetrics], bool]] = {
            "token_usage_high": lambda m: m.avg_token_usage > config.scale_out_policy.token_usage_threshold,
            "queue_backlog": lambda m: (
                m.total_queue_reqs > m.num_engines * config.scale_out_policy.queue_depth_per_engine
            ),
            "queue_latency_high": lambda m: (
                m.max_queue_time_p95 > config.scale_out_policy.queue_time_p95_threshold
                or m.max_queue_time_p95_overflow
            ),
            "ttft_high": lambda m: (
                m.max_ttft_p95 > config.scale_out_policy.ttft_p95_threshold or m.max_ttft_p95_overflow
            ),
        }

        # Define scale-in conditions (all must be true)
        self.scale_in_conditions: Dict[str, Callable[[AggregatedMetrics], bool]] = {
            "token_usage_low": lambda m: m.avg_token_usage < config.scale_in_policy.token_usage_threshold,
            "no_queue": lambda m: m.total_queue_reqs <= config.scale_in_policy.queue_depth_threshold,
            "throughput_stable": lambda m: (
                m.throughput_variance < config.scale_in_policy.throughput_variance_threshold
            ),
        }

    def evaluate(
        self,
        aggregated_metrics: AggregatedMetrics,
        current_engines: int,
        last_scale_time: Optional[float],
        last_scale_action: Optional[ScalingAction],
        pending_requests: List[Dict[str, Any]],
    ) -> ScalingDecision:
        """Evaluate current state and return a scaling decision.

        This method implements the main decision logic:
        1. Check cooldown period
        2. Check for pending operations
        3. Evaluate scale-out conditions
        4. Evaluate scale-in conditions

        Args:
            aggregated_metrics: Current aggregated metrics from all engines.
            current_engines: Current number of active engines.
            last_scale_time: Timestamp of the last scaling operation.
            last_scale_action: Action of the last scaling operation.
            pending_requests: List of pending scaling requests.

        Returns:
            ScalingDecision with the recommended action.
        """
        now = time.time()

        # Update debounce state before gates so invalid observations reset streaks.
        self._update_condition_trackers(aggregated_metrics, self._clock())

        # Missing metrics are not equivalent to an idle engine.
        if aggregated_metrics.is_empty:
            return ScalingDecision(
                action=ScalingAction.NONE,
                delta=0,
                reason="Metrics snapshot empty (no engine reported); freezing scaling (fail-safe)",
                confidence=1.0,
                metrics_snapshot=aggregated_metrics.to_dict(),
            )

        # 1. Check cooldown period
        if last_scale_time is not None:
            cooldown_secs = (
                self.config.scale_out_cooldown_secs
                if last_scale_action == ScalingAction.SCALE_OUT
                else self.config.scale_in_cooldown_secs
            )
            time_since_scale = now - last_scale_time

            if time_since_scale < cooldown_secs:
                logger.debug(
                    f"[Autoscaler] In cooldown: {time_since_scale:.1f}s < {cooldown_secs}s "
                    f"(last action: {last_scale_action.value if last_scale_action else 'none'})"
                )
                return ScalingDecision(
                    action=ScalingAction.NONE,
                    delta=0,
                    reason=f"In cooldown period ({time_since_scale:.1f}s < {cooldown_secs}s)",
                    confidence=1.0,
                    metrics_snapshot=aggregated_metrics.to_dict(),
                )

        # 2. Check for pending scale operations
        active_pending = [
            r for r in pending_requests if not is_scale_request_terminal(r.get("action", "scale_out"), r.get("status"))
        ]
        if active_pending:
            return ScalingDecision(
                action=ScalingAction.NONE,
                delta=0,
                reason=f"{len(active_pending)} pending scale operations in progress",
                confidence=1.0,
                metrics_snapshot=aggregated_metrics.to_dict(),
            )

        # 3. Evaluate scale-out conditions (any condition triggers)
        # Scale-out tolerates partial metrics coverage.
        if current_engines < self.config.max_engines:
            if aggregated_metrics.coverage < self.config.min_coverage_scale_out:
                logger.debug(
                    f"[Autoscaler] Scale-out suppressed: coverage {aggregated_metrics.coverage:.2f} "
                    f"< min_coverage_scale_out {self.config.min_coverage_scale_out}"
                )
            else:
                triggered_out = self._evaluate_scale_out_conditions(aggregated_metrics)

                if triggered_out:
                    required = self.config.scale_out_policy.condition_duration_secs
                    sustained_out = [name for name in triggered_out if self._sustained("scale_out", name, required)]
                    if sustained_out:
                        delta = self._calculate_scale_out_delta(aggregated_metrics, current_engines)
                        if delta > 0:
                            logger.info(f"Scale-out decision: {delta} engines, triggered by: {sustained_out}")
                            return ScalingDecision(
                                action=ScalingAction.SCALE_OUT,
                                delta=delta,
                                reason=f"Conditions met: {', '.join(sustained_out)}",
                                confidence=0.8,
                                metrics_snapshot=aggregated_metrics.to_dict(),
                                triggered_conditions=sustained_out,
                            )

        # 4. Evaluate scale-in conditions (all conditions must be met)
        # Scale-in requires stricter coverage before removing capacity.
        if current_engines > self.config.min_engines:
            if aggregated_metrics.coverage < self.config.min_coverage_scale_in:
                logger.debug(
                    f"[Autoscaler] Scale-in suppressed: coverage {aggregated_metrics.coverage:.2f} "
                    f"< min_coverage_scale_in {self.config.min_coverage_scale_in}"
                )
            else:
                triggered_in = self._evaluate_scale_in_conditions(aggregated_metrics)

                if len(triggered_in) == len(self.scale_in_conditions):
                    required = self.config.scale_in_policy.condition_duration_secs
                    all_sustained = all(self._sustained("scale_in", name, required) for name in triggered_in)
                    if all_sustained:
                        delta = self._calculate_scale_in_delta(aggregated_metrics, current_engines)
                        if delta > 0:
                            logger.info(f"Scale-in decision: {delta} engines, all conditions met: {triggered_in}")
                            return ScalingDecision(
                                action=ScalingAction.SCALE_IN,
                                delta=delta,
                                reason=f"All conditions met: {', '.join(triggered_in)}",
                                confidence=0.6,  # Lower confidence for scale-in
                                metrics_snapshot=aggregated_metrics.to_dict(),
                                triggered_conditions=triggered_in,
                            )

        # No scaling needed
        return ScalingDecision(
            action=ScalingAction.NONE,
            delta=0,
            reason="No scaling conditions met",
            confidence=1.0,
            metrics_snapshot=aggregated_metrics.to_dict(),
        )

    def _update_condition_trackers(self, metrics: AggregatedMetrics, now: float) -> None:
        """Advance debounce trackers for one observation."""
        valid_out = (not metrics.is_empty) and metrics.coverage >= self.config.min_coverage_scale_out
        valid_in = (not metrics.is_empty) and metrics.coverage >= self.config.min_coverage_scale_in
        self._advance_direction("scale_out", self.scale_out_conditions, metrics, valid_out, now)
        self._advance_direction("scale_in", self.scale_in_conditions, metrics, valid_in, now)

    def _advance_direction(
        self,
        direction: str,
        conditions: Dict[str, Callable[[AggregatedMetrics], bool]],
        metrics: AggregatedMetrics,
        frame_valid: bool,
        now: float,
    ) -> None:
        """Advance all condition trackers in one direction."""
        for name, check_fn in conditions.items():
            try:
                satisfied = bool(check_fn(metrics))
            except Exception as e:
                logger.warning(f"Error evaluating {direction} condition '{name}' for tracker: {e}")
                satisfied = False

            key = (direction, name)
            st = self._condition_state.get(key)
            if st is None:
                st = {"elapsed": 0.0, "last_ts": now, "satisfied": False, "last_updated": now}
                self._condition_state[key] = st

            active = frame_valid and satisfied
            if not active:
                st["elapsed"] = 0.0
                st["satisfied"] = False
            else:
                dt = now - st["last_ts"]
                if not st["satisfied"]:
                    st["elapsed"] = 0.0
                elif dt < 0 or dt > self._max_gap_secs:
                    st["elapsed"] = 0.0
                else:
                    st["elapsed"] += dt
                st["satisfied"] = True

            st["last_ts"] = now
            st["last_updated"] = now

    def reset_condition_trackers(self) -> None:
        """Reset every debounce streak to its unsatisfied state."""
        self._update_condition_trackers(AggregatedMetrics(is_empty=True, coverage=0.0), self._clock())

    def _sustained(self, direction: str, name: str, required_secs: float) -> bool:
        """Return whether a condition has remained satisfied long enough."""
        st = self._condition_state.get((direction, name))
        if st is None or not st["satisfied"]:
            return False
        return st["elapsed"] >= required_secs

    def is_scale_in_sustained(self) -> bool:
        """Return True if ALL scale-in conditions are currently sustained.

        Read-only convenience over the tracker state (does not advance it).
        """
        required = self.config.scale_in_policy.condition_duration_secs
        return all(self._sustained("scale_in", name, required) for name in self.scale_in_conditions)

    def condition_observation(self) -> Dict[str, Any]:
        """Return a read-only snapshot of the debounce trackers."""
        observation: Dict[str, Any] = {}
        for direction, conditions, required in (
            ("scale_out", self.scale_out_conditions, self.config.scale_out_policy.condition_duration_secs),
            ("scale_in", self.scale_in_conditions, self.config.scale_in_policy.condition_duration_secs),
        ):
            cond_map: Dict[str, Any] = {}
            last_updated: Optional[float] = None
            for name in conditions:
                st = self._condition_state.get((direction, name))
                if st is None:
                    cond_map[name] = {
                        "satisfied": False,
                        "elapsed_secs": 0.0,
                        "sustained": False,
                        "last_updated": None,
                    }
                    continue
                cond_map[name] = {
                    "satisfied": st["satisfied"],
                    "elapsed_secs": st["elapsed"],
                    "sustained": st["satisfied"] and st["elapsed"] >= required,
                    "last_updated": st["last_updated"],
                }
                if last_updated is None or st["last_updated"] > last_updated:
                    last_updated = st["last_updated"]
            observation[direction] = {
                "condition_duration_secs": required,
                "last_updated": last_updated,
                "conditions": cond_map,
            }
        return observation

    def _evaluate_scale_out_conditions(self, metrics: AggregatedMetrics) -> List[str]:
        """Evaluate all scale-out conditions.

        Args:
            metrics: Current aggregated metrics.

        Returns:
            List of condition names that were triggered.
        """
        triggered = []
        condition_values = {}

        for name, check_fn in self.scale_out_conditions.items():
            try:
                is_triggered = check_fn(metrics)
                condition_values[name] = is_triggered
                if is_triggered:
                    triggered.append(name)
            except Exception as e:
                logger.warning(f"Error evaluating scale-out condition '{name}': {e}")

        logger.debug(
            f"[Autoscaler] Scale-out conditions: {condition_values}, "
            f"triggered={triggered}, "
            f"token_usage={metrics.avg_token_usage:.2%}, "
            f"queue={metrics.total_queue_reqs}, "
            f"queue_time_p95={metrics.max_queue_time_p95:.2f}s, "
            f"ttft_p95={metrics.max_ttft_p95:.2f}s"
        )

        return triggered

    def _evaluate_scale_in_conditions(self, metrics: AggregatedMetrics) -> List[str]:
        """Evaluate all scale-in conditions.

        Args:
            metrics: Current aggregated metrics.

        Returns:
            List of condition names that were triggered.
        """
        triggered = []
        condition_values = {}

        for name, check_fn in self.scale_in_conditions.items():
            try:
                is_triggered = check_fn(metrics)
                condition_values[name] = is_triggered
                if is_triggered:
                    triggered.append(name)
            except Exception as e:
                logger.warning(f"Error evaluating scale-in condition '{name}': {e}")

        logger.debug(
            f"[Autoscaler] Scale-in conditions: {condition_values}, "
            f"triggered={triggered} ({len(triggered)}/{len(self.scale_in_conditions)} required), "
            f"token_usage={metrics.avg_token_usage:.2%}, "
            f"queue={metrics.total_queue_reqs}, "
            f"throughput_variance={metrics.throughput_variance:.2%}"
        )

        return triggered

    def _calculate_scale_out_delta(self, metrics: AggregatedMetrics, current: int) -> int:
        """Calculate how many engines to add.

        The delta is computed based on multiple factors:
        - Token usage pressure: More engines when usage is very high
        - Queue backlog: Proportional to queue depth

        Args:
            metrics: Current aggregated metrics.
            current: Current number of engines.

        Returns:
            Number of engines to add (positive integer).
        """
        token_usage = metrics.avg_token_usage
        queue_depth = metrics.total_queue_reqs
        policy = self.config.scale_out_policy

        # Based on token usage pressure
        if token_usage > 0.9:
            # Critical: usage above 90%
            usage_delta = int((token_usage - 0.7) / 0.1)
        elif token_usage > policy.token_usage_threshold:
            # Above threshold but not critical
            usage_delta = 1
        else:
            usage_delta = 0

        # Based on queue backlog
        # Each engine can handle ~queue_depth_per_engine requests efficiently
        expected_queue = current * policy.queue_depth_per_engine
        if queue_depth > expected_queue:
            queue_delta = max(1, (queue_depth - expected_queue) // (policy.queue_depth_per_engine * 2))
        else:
            queue_delta = 0

        # Take the larger delta
        delta = max(usage_delta, queue_delta, 1)

        # Apply policy limits
        delta = min(delta, policy.max_delta)
        delta = min(delta, self.config.max_engines - current)

        return delta

    def _calculate_scale_in_delta(self, metrics: AggregatedMetrics, current: int) -> int:
        """Calculate how many engines to remove (conservative).

        Scale-in is conservative: we check that removing an engine
        won't cause the remaining engines to be overloaded.

        Args:
            metrics: Current aggregated metrics.
            current: Current number of engines.

        Returns:
            Number of engines to remove (positive integer), typically 1.
        """
        if current <= self.config.min_engines:
            return 0

        token_usage = metrics.avg_token_usage
        policy = self.config.scale_in_policy

        # Project utilization after removing one engine
        projected_usage = token_usage * current / (current - 1)

        # Only allow removal if projected usage is acceptable
        if projected_usage <= policy.projected_usage_max:
            # Conservative: always remove just 1 engine
            return min(1, policy.max_delta)

        return 0

    def get_condition_status(self, metrics: AggregatedMetrics) -> Dict[str, Dict[str, Any]]:
        """Get the current status of all conditions for observability.

        Args:
            metrics: Current aggregated metrics.

        Returns:
            Dictionary mapping condition name to status info.
        """
        status = {}

        for name, check_fn in self.scale_out_conditions.items():
            try:
                status[name] = {
                    "type": "scale_out",
                    "triggered": check_fn(metrics),
                }
            except Exception:
                status[name] = {
                    "type": "scale_out",
                    "triggered": False,
                    "error": True,
                }

        for name, check_fn in self.scale_in_conditions.items():
            try:
                status[name] = {
                    "type": "scale_in",
                    "triggered": check_fn(metrics),
                }
            except Exception:
                status[name] = {
                    "type": "scale_in",
                    "triggered": False,
                    "error": True,
                }

        return status
