# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the Actor audit method without loading optional Megatron
kernels."""

import ast
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.agentic.weight_audit import AUDIT_FIELD


class _Streaming:
    def __init__(self, rows):
        self.rows = rows

    def get_buffer(self):
        return [({AUDIT_FIELD: [row]}, None) for row in self.rows]


class _Legacy(_Streaming):
    def __init__(self, rows, offset, dummy_after):
        super().__init__(rows)
        self.offset = offset
        self.dummy_after = dummy_after


def _load_audit_method(namespace):
    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/actor.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_audit_agentic_consumption"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_audit_agentic_consumption"]


@pytest.fixture
def actor_audit():
    dist = MagicMock()
    dist.get_rank.return_value = 0
    mpu = MagicMock()
    mpu.get_tensor_model_parallel_rank.return_value = 0
    mpu.is_pipeline_last_stage.return_value = True
    mpu.get_context_parallel_rank.return_value = 0
    namespace = dict(
        dist=dist, mpu=mpu, get_gloo_group=lambda: "gloo", StreamingTQIterator=_Streaming, logger=MagicMock()
    )
    # Legacy iterators deliberately do not pass the streaming isinstance test.
    namespace["StreamingTQIterator"] = type("Streaming", (), {"get_buffer": _Streaming.get_buffer})
    method = _load_audit_method(namespace)
    actor = SimpleNamespace(
        args=SimpleNamespace(use_agentic_rollout=True, fully_async=False),
        weight_updater=SimpleNamespace(_agentic_publication=("run", 4, "13")),
    )
    return method, actor, dist, mpu, namespace["StreamingTQIterator"]


def _row(index):
    return {
        "sample_index": index,
        "spans": [{"effective_tokens": 2, "publication": {"epoch": "run", "source_sequence": 1}}],
    }


def test_all_streaming_mini_windows_are_real_consumption(actor_audit):
    audit, actor, dist, _, streaming = actor_audit
    iters = [streaming(), streaming()]
    iters[0].rows, iters[1].rows = [_row(0)], [_row(1)]
    data = {AUDIT_FIELD: [_row(0), _row(1)]}
    audit(actor, data, iters)
    assert [r["sample_index"] for r in data[AUDIT_FIELD]] == [0, 1]
    assert [r["weighted_age_sum"] for r in data[AUDIT_FIELD]] == [6, 6]
    dist.broadcast_object_list.assert_called_once()


@pytest.mark.parametrize("offset,dummy_after", [(1, None), (3, 1)])
def test_vpp_replays_unconsumed_and_dummy_rows_are_not_counted(actor_audit, offset, dummy_after):
    audit, actor, _, _, _ = actor_audit
    iterator = _Legacy([_row(0), _row(1), _row(2)], offset, dummy_after)
    data = {AUDIT_FIELD: [_row(0), _row(1), _row(2)] * 2}
    audit(actor, data, [iterator, iterator])
    assert data[AUDIT_FIELD][0]["known_age_tokens"] == 2
    assert data[AUDIT_FIELD][1:] == [None] * 5


def test_nonreporting_ranks_and_missing_fields_still_join_collective(actor_audit):
    audit, actor, dist, mpu, _ = actor_audit
    mpu.is_pipeline_last_stage.return_value = False
    audit(actor, {})
    dist.broadcast_object_list.assert_called_once()
    dist.reset_mock()
    mpu.is_pipeline_last_stage.return_value = True
    audit(actor, {})
    dist.broadcast_object_list.assert_called_once()


def test_missing_async_publisher_is_unknown_and_non_agentic_adds_no_collective(actor_audit):
    audit, actor, dist, _, _ = actor_audit
    actor.args.fully_async = True
    data = {AUDIT_FIELD: [_row(0)]}
    audit(actor, data)
    assert data[AUDIT_FIELD][0]["consumption_age_token_mean"] is None
    dist.broadcast_object_list.assert_called_once()
    dist.reset_mock()
    actor.args.use_agentic_rollout = False
    audit(actor, data)
    dist.broadcast_object_list.assert_not_called()


@pytest.mark.parametrize("hybrid", [False, True])
def test_dcs_consumption_uses_published_snapshot_including_hybrid(actor_audit, hybrid):
    audit, actor, _, _, _ = actor_audit
    actor.args.fully_async = True
    actor.args.hybrid = hybrid
    del actor.weight_updater
    actor.checkpoint_engine_client = SimpleNamespace(backend=SimpleNamespace(_agentic_publication=("run", 4, "13")))
    data = {AUDIT_FIELD: [_row(0)]}
    audit(actor, data)
    report = data[AUDIT_FIELD][0]
    assert report["consumption_age_token_mean"] == 3
    assert report["version_missing_token_rate"] == 0
    assert report["generation_version_span"] == 0


def _distributed_worker(rank, rendezvous, missing_publisher):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        namespace = dict(
            dist=dist,
            get_gloo_group=lambda: dist.group.WORLD,
            logger=MagicMock(),
            mpu=SimpleNamespace(
                get_tensor_model_parallel_rank=lambda: 0,
                is_pipeline_last_stage=lambda: True,
                get_context_parallel_rank=lambda: 0,
            ),
        )
        method = _load_audit_method(namespace)
        actor = SimpleNamespace(args=SimpleNamespace(use_agentic_rollout=True, fully_async=False))
        if rank == 0 and not missing_publisher:
            actor.weight_updater = SimpleNamespace(_agentic_publication=("run", 4, "13"))
        data = {AUDIT_FIELD: [_row(rank)]}
        method(actor, data)
        assert data[AUDIT_FIELD][0]["consumption_age_token_mean"] == (None if missing_publisher else 3)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("missing_publisher", [False, True])
def test_real_two_rank_gloo_broadcast_finishes_and_propagates_unknown(tmp_path, missing_publisher):
    import torch.multiprocessing as mp

    context = mp.spawn(
        _distributed_worker,
        args=((tmp_path / "rendezvous").as_uri(), missing_publisher),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 40
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Agentic audit collective did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
