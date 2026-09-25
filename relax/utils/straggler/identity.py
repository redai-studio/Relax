# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rank identity used to group observations into comparable cohorts.

Straggler detection only makes sense between ranks that execute the *same*
program: same tensor/pipeline/context/expert parallel position, differing only
in the data-parallel dimension. That equivalence class is what
:attr:`RuntimeIdentity.cohort` encodes, so a rank never compares itself against a
peer that legitimately does different work.

Discovery is deliberately defensive: every probe is optional, and a missing
Megatron/distributed runtime (unit tests, CPU-only CI) yields a single-rank
identity instead of an exception.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

#: Rank has no meaningful value for a parallel dimension that is not in use.
UNKNOWN_RANK = -1


def _probe(callable_obj: Any) -> int:
    """Call an optional rank getter, returning :data:`UNKNOWN_RANK` on
    failure."""
    try:
        return int(callable_obj())
    except Exception:
        return UNKNOWN_RANK


def _distributed_rank_and_world_size() -> Tuple[int, int]:
    """Return the *global* rank and world size, or ``(0, 1)`` when unset."""
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return 0, 1
        # group=None selects the global default group, which is the rank space a
        # cohort key is expressed in.
        return int(dist.get_rank(group=None)), int(dist.get_world_size(group=None))
    except Exception:
        return 0, 1


def _parallel_ranks() -> Tuple[int, int, int, int, int, int, Optional[int]]:
    """Return ``(tp, pp, vpp, cp, ep, dp, dp_world_size)``, probing
    defensively."""
    missing = (UNKNOWN_RANK,) * 6 + (None,)
    try:
        from megatron.core import parallel_state
    except Exception:
        return missing

    dp_world_size: Optional[int] = None
    try:
        dp_world_size = int(parallel_state.get_data_parallel_world_size())
    except Exception:
        dp_world_size = None
    return (
        _probe(parallel_state.get_tensor_model_parallel_rank),
        _probe(parallel_state.get_pipeline_model_parallel_rank),
        _probe(parallel_state.get_virtual_pipeline_model_parallel_rank),
        _probe(parallel_state.get_context_parallel_rank),
        _probe(getattr(parallel_state, "get_expert_model_parallel_rank", lambda: UNKNOWN_RANK)),
        _probe(parallel_state.get_data_parallel_rank),
        dp_world_size,
    )


def _run_id() -> str:
    """Identify the current run for cross-process envelope grouping."""
    for name in ("RELAX_STRAGGLER_RUN_ID", "SLURM_JOB_ID", "RAY_JOB_ID"):
        value = os.environ.get(name)
        if value:
            return value
    return f"pid{os.getpid()}"


@dataclass(frozen=True)
class RuntimeIdentity:
    """Where this process sits in the training topology.

    Attributes:
        run_id: identifier shared by every rank of one training run.
        rank: global rank, or 0 when distributed is not initialised.
        world_size: global world size, or 1 when distributed is not initialised.
        tensor_parallel_rank: TP position, ``-1`` when unknown.
        pipeline_parallel_rank: PP position, ``-1`` when unknown.
        virtual_pipeline_parallel_rank: VPP position, ``-1`` when unknown.
        context_parallel_rank: CP position, ``-1`` when unknown.
        expert_parallel_rank: EP position, ``-1`` when unknown.
        data_parallel_rank: DP position, ``-1`` when unknown.
        data_parallel_world_size: expected cohort size, ``None`` when unknown.
    """

    run_id: str
    rank: int
    world_size: int
    tensor_parallel_rank: int = UNKNOWN_RANK
    pipeline_parallel_rank: int = UNKNOWN_RANK
    virtual_pipeline_parallel_rank: int = UNKNOWN_RANK
    context_parallel_rank: int = UNKNOWN_RANK
    expert_parallel_rank: int = UNKNOWN_RANK
    data_parallel_rank: int = UNKNOWN_RANK
    data_parallel_world_size: Optional[int] = None

    @property
    def cohort(self) -> str:
        """Key shared by ranks that differ only in the data-parallel
        dimension."""
        return ":".join(
            str(value)
            for value in (
                self.tensor_parallel_rank,
                self.pipeline_parallel_rank,
                self.virtual_pipeline_parallel_rank,
                self.context_parallel_rank,
                self.expert_parallel_rank,
            )
        )

    @property
    def label(self) -> str:
        """Compact human-readable position, used in reports."""
        return (
            f"rank{self.rank}/tp{self.tensor_parallel_rank}/pp{self.pipeline_parallel_rank}"
            f"/vpp{self.virtual_pipeline_parallel_rank}/cp{self.context_parallel_rank}"
            f"/ep{self.expert_parallel_rank}/dp{self.data_parallel_rank}"
        )

    def as_dict(self) -> dict:
        """Return a JSON-friendly view for evidence files."""
        return {
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "cohort": self.cohort,
            "label": self.label,
            "topology": {
                "tp": self.tensor_parallel_rank,
                "pp": self.pipeline_parallel_rank,
                "vpp": self.virtual_pipeline_parallel_rank,
                "cp": self.context_parallel_rank,
                "ep": self.expert_parallel_rank,
                "dp": self.data_parallel_rank,
            },
            "data_parallel_world_size": self.data_parallel_world_size,
        }


def discover_identity() -> RuntimeIdentity:
    """Probe the current rank's position; never raises."""
    rank, world_size = _distributed_rank_and_world_size()
    tp, pp, vpp, cp, ep, dp, dp_world_size = _parallel_ranks()
    if dp == UNKNOWN_RANK and world_size > 1:
        # No Megatron parallel state (e.g. a mock trainer): fall back to a flat
        # data-parallel view so peers are still comparable, and keep the cohort
        # key constant across the run.
        dp = rank
        tp = pp = vpp = cp = ep = 0
        dp_world_size = dp_world_size if dp_world_size is not None else world_size
    return RuntimeIdentity(
        run_id=_run_id(),
        rank=rank,
        world_size=world_size,
        tensor_parallel_rank=tp,
        pipeline_parallel_rank=pp,
        virtual_pipeline_parallel_rank=vpp,
        context_parallel_rank=cp,
        expert_parallel_rank=ep,
        data_parallel_rank=dp,
        data_parallel_world_size=dp_world_size,
    )


__all__ = ["UNKNOWN_RANK", "RuntimeIdentity", "discover_identity"]
