# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scale-only pure helpers shared by rollout/sglang scale-out paths.

These functions carry no instance state; they are extracted here to keep
``relax/distributed/ray/rollout.py`` and
``relax/backends/sglang/sglang_engine.py`` focused on their service logic.
Behaviour is unchanged from the originals.
"""

import dataclasses
import enum
import os
import signal
import subprocess

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# Scale-out failure classification: producers emit a structured ``ScaleOutFailure``
# (category enum + optional short token) at the failure site; the category drives
# dedup + the monitor bucket, and ``message()`` renders the non-sensitive
# ``error_message`` (raw exception text/addresses stay in the logs).

_MAX_REASON_ITEMS = Envs.RELAX_SCALE_OUT_MAX_REASON_ITEMS
_MAX_REASON_ITEM_LEN = Envs.RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN
_MAX_REASON_TOTAL_LEN = Envs.RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN


class ScaleOutFailureCategory(enum.Enum):
    """Stable scale-out failure categories: ``.value`` is the ``error_message``
    label; ``.name`` is serialized to ``failure_categories`` for the
    monitor."""

    NCCL_PRECHECK_TRANSPORT_MISMATCH = "NCCL transport mismatch"
    NCCL_PRECHECK_FAIL = "NCCL precheck FAIL"
    PROVISION_TIMEOUT = "provision timed out (elastic node)"
    PROVISION_FAILED = "PG provision failed"
    WEIGHT_SYNC_FAILED = "weight sync failed"
    HEALTH_CHECK_FAILED = "health check failed"
    DCS_REGISTRATION_FAILED = "DCS registration failed"
    ROUTER_REGISTRATION_FAILED = "router registration failed"
    INVALID_ENGINE_ADDRESS = "invalid engine address"
    EXTERNAL_ENGINE_CONNECT_FAILED = "external engine connect failed"
    ENGINE_INIT_NO_HANDLES = "engine init: no handles"
    ENGINE_INIT_FAILED = "engine init/provision failed"
    UNKNOWN = "unknown failure"


# Precheck categories surface a readable first line of their detail (the real
# NCCL error) so the user sees the root cause; all other categories drop detail
# so raw exception text (addresses / secrets / tracebacks) can't leak.
_DETAIL_SURFACED_CATEGORIES = frozenset(
    {
        ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH,
        ScaleOutFailureCategory.NCCL_PRECHECK_FAIL,
    }
)


# Diagnostic priority (most actionable first) used when there are more distinct
# failures than we can surface: truncation keeps the highest-priority root
# causes instead of whatever arrived first, so e.g. a transport mismatch is
# never silently dropped behind provisioning noise. Categories absent here sort
# last but keep their arrival order (stable sort).
_CATEGORY_DIAGNOSTIC_PRIORITY = (
    ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH,
    ScaleOutFailureCategory.NCCL_PRECHECK_FAIL,
    ScaleOutFailureCategory.PROVISION_TIMEOUT,
    ScaleOutFailureCategory.WEIGHT_SYNC_FAILED,
    ScaleOutFailureCategory.HEALTH_CHECK_FAILED,
    ScaleOutFailureCategory.DCS_REGISTRATION_FAILED,
    ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED,
    ScaleOutFailureCategory.EXTERNAL_ENGINE_CONNECT_FAILED,
    ScaleOutFailureCategory.INVALID_ENGINE_ADDRESS,
    ScaleOutFailureCategory.PROVISION_FAILED,
    ScaleOutFailureCategory.ENGINE_INIT_NO_HANDLES,
    ScaleOutFailureCategory.ENGINE_INIT_FAILED,
    ScaleOutFailureCategory.UNKNOWN,
)
_CATEGORY_PRIORITY_INDEX = {category: index for index, category in enumerate(_CATEGORY_DIAGNOSTIC_PRIORITY)}


def _dedupe_failures_by_priority(reasons: "list[ScaleOutFailure]") -> "list[ScaleOutFailure]":
    """Deduplicate by category (first occurrence wins) then order by diagnostic
    priority, so any downstream truncation keeps the most actionable causes."""
    unique: list[ScaleOutFailure] = []
    seen: set[ScaleOutFailureCategory] = set()
    for failure in reasons:
        if failure.category in seen:
            continue
        seen.add(failure.category)
        unique.append(failure)
    unique.sort(key=lambda failure: _CATEGORY_PRIORITY_INDEX.get(failure.category, len(_CATEGORY_DIAGNOSTIC_PRIORITY)))
    return unique


def _first_line(text: str) -> str:
    """First non-empty line of ``text``, stripped and length-bounded — keeps a
    diagnostic message readable while never emitting a multi-line traceback."""
    lines = text.splitlines()
    line = lines[0].strip() if lines else ""
    if len(line) > _MAX_REASON_ITEM_LEN:
        line = line[: _MAX_REASON_ITEM_LEN - 3] + "..."
    return line


class PrecheckProbeCategory(enum.Enum):
    """NCCL weight-sync probe outcome — the wire protocol between the SGLang
    engine (emits ``.value``) and the rollout coordinator (rebuilds via
    ``from_wire`` and branches on enum identity).

    Every value is set from a structured signal (an explicit pre-launch check,
    the subprocess exit/JSON, or a caught exception) — never by scanning NCCL
    log text. Probe failures we can't attribute precisely are ``PROBE_FAILED``
    and carry the raw exception, rather than being guessed into a bucket.
    """

    # Pre-launch checks in the engine (explicit conditions, not the probe).
    UNSUPPORTED_NODE_RANK = "unsupported_node_rank"
    UNSUPPORTED_TOPOLOGY = "unsupported_topology"
    INVALID_PORTS = "invalid_ports"
    GPU_MAPPING_MISMATCH = "gpu_mapping_mismatch"
    MEMORY_CHECK_FAILED = "memory_check_failed"
    INSUFFICIENT_GPU_MEMORY = "insufficient_gpu_memory"
    # Probe subprocess outcomes.
    LAUNCH_TRANSIENT = "launch_transient"  # produced no structured output (didn't run)
    TIMEOUT = "timeout"  # exceeded the manager deadline
    PROBE_FAILED = "probe_failed"  # ran but NCCL init/collective raised
    INVALID_RESULT = "invalid_result"

    @classmethod
    def from_wire(cls, value: object) -> "PrecheckProbeCategory":
        """Rebuild from a wire value (or ``None``); unknown →
        ``INVALID_RESULT``."""
        try:
            return cls(value)
        except ValueError:
            return cls.INVALID_RESULT

    @property
    def retryable(self) -> bool:
        """Only a probe that never produced output is a transient worth
        retrying; a probe that ran and failed is deterministic and fails
        closed."""
        return self is PrecheckProbeCategory.LAUNCH_TRANSIENT

    @property
    def is_skip(self) -> bool:
        """Topology the single-node probe can't exercise; skip, let real sync
        handle it."""
        return self in (PrecheckProbeCategory.UNSUPPORTED_TOPOLOGY, PrecheckProbeCategory.UNSUPPORTED_NODE_RANK)


@dataclasses.dataclass
class ScaleOutFailure:
    """A classified scale-out failure: a category plus optional short context.

    ``message()`` renders the non-sensitive string for ``error_message``;
    ``__str__`` delegates to it so ``f"{failure}"`` / ``format(failure)`` work.
    """

    category: ScaleOutFailureCategory
    detail: str = ""

    def message(self) -> str:
        label = self.category.value
        if not self.detail:
            return label
        if self.category is ScaleOutFailureCategory.UNKNOWN:
            return _first_line(self.detail) or label
        if self.category in _DETAIL_SURFACED_CATEGORIES:
            line = _first_line(self.detail)
            return f"{label}: {line}" if line else label
        # Drop detail so addresses / secrets / tracebacks never reach the surface.
        return label

    def __str__(self) -> str:
        return self.message()


def _aggregate_scale_out_reasons(
    reasons: "list[ScaleOutFailure] | None",
    max_items: int = _MAX_REASON_ITEMS,
    max_total: int = _MAX_REASON_TOTAL_LEN,
) -> str:
    """Render, dedupe (by category), and bound failures into a single non-
    sensitive string for ``error_message``."""
    if not reasons:
        return ""
    labels = [failure.message() for failure in _dedupe_failures_by_priority(reasons)]
    if not labels:
        return ""
    extra = 0
    if len(labels) > max_items:
        extra = len(labels) - max_items
        labels = labels[:max_items]
    text = "; ".join(labels)
    if extra:
        text += f"; …and {extra} more"
    if len(text) > max_total:
        text = text[: max_total - 3] + "..."
    return text


def _scale_out_failure_categories(
    reasons: "list[ScaleOutFailure] | None",
    max_items: int = _MAX_REASON_ITEMS,
) -> list[str]:
    """Deduped category ``.name``s for ``failure_categories`` (monitor buckets
    without re-parsing ``error_message``), highest diagnostic priority first so
    truncation never hides the most actionable root cause."""
    if not reasons:
        return []
    names = [failure.category.name for failure in _dedupe_failures_by_priority(reasons)]
    return names[:max_items]


def _nccl_ib_disabled(value: object) -> bool:
    """Normalize ``NCCL_IB_DISABLE`` to its transport meaning: NCCL disables IB
    only on a nonzero int, so unset / empty / ``"0"`` all mean *IB enabled* and
    must compare equal (an unset seed vs an explicit ``"0"`` new node is not a
    mismatch)."""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return int(text) != 0
    except ValueError:
        return text.lower() not in ("false", "no", "off")


def _scale_weight_sync_precheck_fingerprints_match(seed_result: dict, new_result: dict) -> tuple[bool, str | None]:
    """Cheap env pre-check before the NCCL probe (Stage 1).

    Only ``NCCL_IB_DISABLE`` is a *guaranteed* transport-mode incompatibility —
    one side on IB and the other on socket can never share a transport — so it
    is the only hard reject here (compared by transport meaning, not raw
    string, so unset vs ``"0"`` isn't a false mismatch). Every other asymmetry
    (HCA, GID index, socket ifname) is logged as advisory and deferred to the
    real NCCL probe (Stage 2), which tests actual connectivity in practice
    rather than guessing from env.
    """
    seed_fingerprint = seed_result.get("fingerprint") or {}
    new_fingerprint = new_result.get("fingerprint") or {}

    if _nccl_ib_disabled(seed_fingerprint.get("nccl_ib_disable")) != _nccl_ib_disabled(
        new_fingerprint.get("nccl_ib_disable")
    ):
        return False, "nccl_ib_disable_mismatch"

    for key in ("nccl_ib_hca", "nccl_ib_gid_index", "nccl_socket_ifname"):
        if seed_fingerprint.get(key) != new_fingerprint.get(key):
            logger.warning(
                f"[ScaleOut][Precheck] {key} asymmetric "
                f"(seed={seed_fingerprint.get(key)!r}, new={new_fingerprint.get(key)!r}); "
                "advisory — deferring to the NCCL probe"
            )
    return True, None


def _terminate_probe_process(process: subprocess.Popen, grace_secs: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_secs)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace_secs)
            except subprocess.TimeoutExpired:
                logger.error(f"NCCL precheck process {process.pid} did not exit after SIGKILL")
