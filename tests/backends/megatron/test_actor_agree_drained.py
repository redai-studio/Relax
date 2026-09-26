# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class _Rank:
    dp: int
    pp: int
    tp: int
    cp: int


def _world(dp: int = 2, pp: int = 2, tp: int = 2, cp: int = 2) -> list[_Rank]:
    return [
        _Rank(dp=dp_rank, pp=pp_rank, tp=tp_rank, cp=cp_rank)
        for dp_rank in range(dp)
        for pp_rank in range(pp)
        for tp_rank in range(tp)
        for cp_rank in range(cp)
    ]


def _tc_group(rank: _Rank) -> tuple[str, int, int]:
    return ("tc", rank.dp, rank.pp)


def _pp_group(rank: _Rank) -> tuple[str, int, int, int]:
    return ("pp", rank.dp, rank.tp, rank.cp)


def _cp_group(rank: _Rank) -> tuple[str, int, int, int]:
    return ("cp", rank.dp, rank.pp, rank.tp)


def _tp_group(rank: _Rank) -> tuple[str, int, int, int]:
    return ("tp", rank.dp, rank.pp, rank.cp)


def _load_collective_utils(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    core.mpu = mpu

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)

    module_path = Path(__file__).parents[3] / "relax" / "backends" / "megatron" / "collective_utils.py"
    spec = importlib.util.spec_from_file_location("_test_megatron_collective_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _broadcast_phase(
    ranks: list[_Rank],
    statuses: dict[_Rank, bool],
    group_fn,
    source_fn,
) -> dict[_Rank, bool]:
    next_statuses = dict(statuses)
    group_keys = {group_fn(rank) for rank in ranks}
    for group_key in group_keys:
        members = [rank for rank in ranks if group_fn(rank) == group_key]
        source = next(rank for rank in members if source_fn(rank))
        source_status = statuses[source]
        for rank in members:
            next_statuses[rank] = source_status
    return next_statuses


def _old_broadcast_drained(
    ranks: list[_Rank],
    local_statuses: dict[_Rank, bool],
    *,
    include_pipeline: bool,
) -> dict[_Rank, bool]:
    """Reference implementation of the previous broadcast-based drain sync."""
    statuses = _broadcast_phase(ranks, local_statuses, _cp_group, lambda rank: rank.cp == 0)
    statuses = _broadcast_phase(ranks, statuses, _tp_group, lambda rank: rank.tp == 0)
    if include_pipeline:
        statuses = _broadcast_phase(ranks, statuses, _pp_group, lambda rank: rank.pp == 0)
    return statuses


def _new_agree_drained(
    monkeypatch,
    collective_utils,
    ranks: list[_Rank],
    local_statuses: dict[_Rank, bool],
    *,
    include_pipeline: bool,
) -> dict[_Rank, bool]:
    """Run _agree_drained for every fake rank with deterministic MIN
    collectives."""
    current_rank = {"value": ranks[0]}

    tc_reduced = {
        rank: min(local_statuses[member] for member in ranks if _tc_group(member) == _tc_group(rank)) for rank in ranks
    }
    pp_reduced = {
        rank: min(tc_reduced[member] for member in ranks if _pp_group(member) == _pp_group(rank)) for rank in ranks
    }

    def fake_all_reduce(tensor, op=None, group=None):
        assert op == collective_utils.dist.ReduceOp.MIN
        rank = current_rank["value"]
        if group == _tc_group(rank):
            tensor.fill_(int(tc_reduced[rank]))
            return
        if group == _pp_group(rank):
            tensor.fill_(int(pp_reduced[rank]))
            return
        raise AssertionError(f"unexpected group {group!r} for rank {rank!r}")

    monkeypatch.setattr(collective_utils.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(collective_utils.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        collective_utils.mpu,
        "get_tensor_and_context_parallel_group",
        lambda: _tc_group(current_rank["value"]),
        raising=False,
    )
    monkeypatch.setattr(
        collective_utils.mpu,
        "get_pipeline_model_parallel_group",
        lambda: _pp_group(current_rank["value"]),
        raising=False,
    )

    results = {}
    for rank in ranks:
        current_rank["value"] = rank
        result = collective_utils._agree_drained(local_statuses[rank], include_pipeline=include_pipeline)
        results[rank] = bool(result.item())
    return results


def test_agree_drained_matches_old_all_consumed_broadcasts(monkeypatch):
    collective_utils = _load_collective_utils(monkeypatch)
    ranks = _world()
    local_statuses = {
        rank: False if (rank.dp == 0 and rank.pp == 0 and rank.tp == 0 and rank.cp == 0) else True for rank in ranks
    }

    expected = _old_broadcast_drained(ranks, local_statuses, include_pipeline=True)
    actual = _new_agree_drained(monkeypatch, collective_utils, ranks, local_statuses, include_pipeline=True)

    assert actual == expected
    assert {actual[rank] for rank in ranks if rank.dp == 0} == {False}
    assert {actual[rank] for rank in ranks if rank.dp == 1} == {True}


def test_agree_drained_streaming_matches_old_intra_stage_broadcasts(monkeypatch):
    collective_utils = _load_collective_utils(monkeypatch)
    ranks = _world()
    stage_status = {
        (0, 0): False,
        (0, 1): True,
        (1, 0): True,
        (1, 1): False,
    }
    local_statuses = {
        rank: stage_status[(rank.dp, rank.pp)] if rank.tp == 0 and rank.cp == 0 else True for rank in ranks
    }

    expected = _old_broadcast_drained(ranks, local_statuses, include_pipeline=False)
    actual = _new_agree_drained(monkeypatch, collective_utils, ranks, local_statuses, include_pipeline=False)

    assert actual == expected
    for rank in ranks:
        assert actual[rank] is stage_status[(rank.dp, rank.pp)]
