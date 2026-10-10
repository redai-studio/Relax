# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bindings to torch / Megatron, and the process-wide profiler the Megatron
actor holds.

Everything that touches the accelerator (through ``relax.utils.device``, so
CUDA and NPU alike), the process groups or Megatron's parallel state is here,
so the rest of the package runs (and is tested) on CPU with fakes.
"""

from __future__ import annotations

import socket
from functools import partial
from typing import Any, Callable

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.collector import StragglerCollector
from relax.utils.straggler.detector import DetectorConfig, RankMeta, WindowReport
from relax.utils.straggler.reporter import log_straggler_delivery, log_straggler_window, report_straggler_window
from relax.utils.straggler.timers import StragglerTimers


logger = get_logger(__name__)

# Only the role that runs training steps closes them; critic / reference /
# actor_fwd actors would only fill the pending queue and never drain it.
PROFILED_ROLES: frozenset[str] = frozenset({"actor"})


def timing_event_factory() -> Callable[[], Any]:
    """Build timing events of the active accelerator."""
    from relax.utils.device import device_module

    return partial(device_module.Event, enable_timing=True)


def stream_capture_check() -> Callable[[], bool]:
    """The active accelerator's "is the current stream being captured" check,
    resolved once because the timers call it on every bracket.

    A backend without one cannot be capturing through this API.
    """
    from relax.utils import device

    if not device.is_available():
        return _never
    return getattr(device.device_module, "is_current_stream_capturing", None) or _never


def _never() -> bool:
    return False


def gloo_all_gather(values: list[float]) -> list[list[float]]:
    """All-gather one CPU vector over Relax's Gloo group (ranks == global
    ranks)."""
    import torch
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    local = torch.tensor(values, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    return [row.tolist() for row in gathered]


def gloo_all_gather_objects(obj: Any) -> list[Any]:
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, obj, group=group)
    return gathered


def local_rank_meta() -> RankMeta:
    import torch.distributed as dist
    from megatron.core import mpu

    from relax.utils import device
    from relax.utils.distributed_utils import get_gloo_group

    return RankMeta(
        # Rank within the Gloo group == global rank; it indexes the gathered table.
        rank=dist.get_rank(get_gloo_group()),
        dp=mpu.get_data_parallel_rank(with_context_parallel=True),
        tp=mpu.get_tensor_model_parallel_rank(),
        pp=mpu.get_pipeline_model_parallel_rank(),
        cp=mpu.get_context_parallel_rank(),
        ep=mpu.get_expert_model_parallel_rank(),
        host=socket.gethostname(),
        device=device.device_module.current_device() if device.is_available() else -1,
    )


class StragglerProfiler:
    """One collector plus the per-step reporting around ``log_perf_data``.

    ``step_metrics`` and ``delivered`` never raise: a failing reporter is
    counted by the collector's health like any other profiler failure.
    """

    def __init__(self, collector: StragglerCollector, args: Any) -> None:
        self.collector = collector
        self._args = args
        self._emitted: WindowReport | None = None
        self._delivered_ms: float | None = None

    def step_metrics(self, rollout_id: int, tokens: int) -> dict[str, float] | None:
        """Close this step (all ranks; a Gloo collective every window) and
        return the scalars of every window whose analysis has finished, for
        this step's ``log_perf_data``."""
        self.collector.add_tokens(tokens)
        reports = self.collector.end_step(rollout_id, final=rollout_id + 1 == self._args.num_rollout)
        metrics: dict[str, float] = {}
        for report in reports:
            try:
                metrics.update(report_straggler_window(self._args, report))
            except Exception as exc:
                self.collector.on_error("report", exc)
        if metrics:
            self._emitted = reports[-1]
            if self._delivered_ms is not None:
                # A report cannot carry its own delivery time, so each carries the previous one's.
                metrics["straggler/latency/prev_delivered_ms"] = self._delivered_ms
        return metrics or None

    def delivered(self, rollout_id: int) -> None:
        """Log close-to-delivered latency once the ``log_perf_data`` that
        carried the last scalars has returned."""
        report, self._emitted = self._emitted, None
        if report is None:
            return
        try:
            self._delivered_ms = log_straggler_delivery(self._args, report, rollout_id)
        except Exception as exc:
            self.collector.on_error("report", exc)


_PROFILER: StragglerProfiler | None = None


def install_straggler_profiler(role: str, args: Any) -> StragglerProfiler | None:
    """Create the process-wide profiler if it is enabled and ``role`` runs
    training steps; ``None`` otherwise.

    Must run after Megatron parallel state and Relax's Gloo group exist, and
    before the optimizer is built (its config takes ``straggler_timers``).
    """
    global _PROFILER
    from relax.utils.env import Envs

    if not Envs.RELAX_STRAGGLER_PROFILER or role not in PROFILED_ROLES:
        return None
    if _PROFILER is not None:
        return _PROFILER
    from megatron.core import mpu

    # Must match ``log_perf_data``'s primary rank: the window scalars are logged
    # through it, which drops them on every other rank.
    is_primary = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )
    collector = StragglerCollector(
        rank_meta=local_rank_meta(),
        is_primary=is_primary,
        report_interval=Envs.RELAX_STRAGGLER_REPORT_INTERVAL,
        detector_config=DetectorConfig(
            z_threshold=Envs.RELAX_STRAGGLER_Z_THRESHOLD,
            rel_threshold=Envs.RELAX_STRAGGLER_REL_THRESHOLD,
            persist_windows=Envs.RELAX_STRAGGLER_PERSIST_WINDOWS,
            recover_windows=Envs.RELAX_STRAGGLER_RECOVER_WINDOWS,
        ),
        event_factory=timing_event_factory(),
        gather=gloo_all_gather,
        gather_objects=gloo_all_gather_objects,
        is_capturing=stream_capture_check(),
        on_report=partial(log_straggler_window, args),
    )
    _PROFILER = StragglerProfiler(collector, args)
    logger.info(
        "Straggler profiler enabled: report_interval=%d primary=%s meta=%s",
        collector.report_interval,
        is_primary,
        collector.rank_meta,
    )
    return _PROFILER


def _timers_for(collector: StragglerCollector, phase: str) -> StragglerTimers:
    if phase == "train":
        return collector.train_timers
    if phase == "forward_only":
        return collector.forward_only_timers
    raise ValueError(f"unknown straggler phase {phase!r}; expected 'train' or 'forward_only'")


def straggler_timers(phase: str) -> StragglerTimers | None:
    """The ``config.timers`` value for ``phase``: ``None`` without a profiler,
    which is what Relax assigns anyway."""
    return _timers_for(_PROFILER.collector, phase) if _PROFILER is not None else None
