# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bindings to torch / Megatron, and the process-wide profiler the Megatron
actor holds.

Everything that touches the accelerator (through ``relax.utils.device``, so
CUDA and NPU alike), the process groups or Megatron's parallel state is here,
so the rest of the package runs (and is tested) on CPU with fakes.
"""

from __future__ import annotations

import pickle
import socket
from functools import partial
from typing import Any, Callable

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.collector import StragglerCollector
from relax.utils.straggler.detector import DetectorConfig, RankMeta, WindowReport
from relax.utils.straggler.reporter import log_straggler_window, report_straggler_window
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


_STALL_GATHER_GROUP: Any = None


def init_straggler_gather_group(timeout_s: float) -> None:
    """Create the profiler's dedicated Gloo group, with a timeout far shorter
    than the training one.

    The generic Gloo group is shared with weight sync and other paths, so its
    timeout cannot be lowered for this; a straggler gather is ``world × 168 B``
    of CPU copies (milliseconds), and a rank that vanished must not park the
    rest of the job for ``--distributed-timeout-minutes``. Every member of the
    default group must call this in the same order, so it runs on all ranks of
    the actor group before any gather.
    """
    global _STALL_GATHER_GROUP
    from datetime import timedelta

    import torch.distributed as dist

    if _STALL_GATHER_GROUP is None:
        _STALL_GATHER_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))


def get_straggler_gather_group() -> Any:
    if _STALL_GATHER_GROUP is None:
        raise RuntimeError("the straggler Gloo group has not been initialized; call install_straggler_profiler first")
    return _STALL_GATHER_GROUP


def gloo_all_gather(values: list[float]) -> list[list[float]]:
    """All-gather one CPU vector over the straggler Gloo group (ranks == global
    ranks)."""
    import torch
    import torch.distributed as dist

    group = get_straggler_gather_group()
    local = torch.tensor(values, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    return [row.tolist() for row in gathered]


# A pickled RankMeta is under 400 B even with a 253-character host name.
_META_ROW_BYTES = 1024


def gloo_all_gather_objects(obj: Any) -> list[Any]:
    """All-gather one small picklable object per rank as a fixed-size byte row.

    ``dist.all_gather_object`` pickles before its first collective, so a rank
    whose pickling fails would skip the collectives its peers issue. Here every
    rank issues the same single collective whatever fails locally: a rank that
    cannot encode its object sends an all-zero row, and every rank raises after
    the gather.
    """
    import torch
    import torch.distributed as dist

    group = get_straggler_gather_group()
    local = torch.zeros(_META_ROW_BYTES, dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    error = None
    try:
        data = pickle.dumps(obj)
        if len(data) > _META_ROW_BYTES:
            raise ValueError(f"{len(data)} B of rank metadata does not fit the {_META_ROW_BYTES} B row")
        local[: len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    except Exception as exc:
        error = exc
    dist.all_gather(gathered, local, group=group)
    if error is not None:
        raise error
    missing = [rank for rank, row in enumerate(gathered) if not row.any()]
    if missing:
        raise RuntimeError(f"ranks {missing} sent no metadata")
    # pickle.loads ignores the zero padding after the pickled bytes.
    return [pickle.loads(row.numpy().tobytes()) for row in gathered]


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

    ``step_metrics`` never raises: a failing reporter is counted by the
    collector's health like any other profiler failure.
    """

    def __init__(self, collector: StragglerCollector, args: Any) -> None:
        self.collector = collector
        self._args = args

    def step_metrics(self, rollout_id: int, tokens: int) -> dict[str, float] | None:
        """Close this step (all ranks; a Gloo collective every window) and
        return the scalars of the latest window whose analysis has finished,
        for this step's ``log_perf_data``.

        If several windows are ready at once, each still reaches the timeline
        and logs, but only the scalars of the latest one whose reporter
        succeeded are returned (one metrics call per step carries one value per
        key). Each window it supersedes is logged as a WARNING and counted as a
        profiler error; a window whose reporter fails is counted once, as that
        failure.
        """
        self.collector.add_tokens(tokens)
        reports = self.collector.end_step(rollout_id, final=rollout_id + 1 == self._args.num_rollout)
        metrics: dict[str, float] | None = None
        returned: WindowReport | None = None
        superseded: list[WindowReport] = []
        for report in reports:
            try:
                scalars = report_straggler_window(self._args, report)
            except Exception as exc:
                self.collector.on_error("report", exc)
                continue
            if returned is not None:
                superseded.append(returned)
            returned, metrics = report, scalars
        if superseded:
            logger.warning(
                "[straggler] rollout %d: %d windows ready at once; only rollouts %d-%d reach the metrics",
                rollout_id,
                len(reports),
                returned.first_rollout,
                returned.last_rollout,
            )
            for dropped in superseded:
                self.collector.on_error(
                    "report",
                    RuntimeError(
                        f"scalars of rollouts {dropped.first_rollout}-{dropped.last_rollout} superseded by a later "
                        "window"
                    ),
                )
        return metrics


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
    from relax.utils.training.train_metric_utils import is_metrics_primary_rank

    # Every rank of the actor group reaches this call in the same order, which
    # is what ``dist.new_group`` requires.
    init_straggler_gather_group(Envs.RELAX_STRAGGLER_GATHER_TIMEOUT_S)

    # The window scalars are logged through ``log_perf_data``, which drops them
    # on every other rank.
    is_primary = is_metrics_primary_rank()
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
