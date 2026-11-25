# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for weight version span capture at the Session service layer."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np

from relax.agentic.pipeline.runtime import BackendGenerateResult
from relax.agentic.session.service import (
    AgenticSessionShard,
    GroupOwnership,
    ResidentGroup,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils.types import WeightVersionSpan


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _forest_with_initial_obs(
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    train_token_delta: list[int],
    rollout_token_delta: list[int],
    rollout_id: int = 0,
    metadata: dict[str, Any] | None = None,
    group_index: int | None = None,
    index: int | None = None,
    label: str | None = None,
    train_metadata: dict[str, Any] | None = None,
):
    forest = SessionForest.create_empty(
        session_id=session_id,
        group_index=group_index,
        index=index,
        label=label,
        train_metadata=train_metadata,
        metadata=metadata,
    )
    initial_obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=rollout_id,
        abort_count=0,
        messages_delta=check_messages(messages),
        train_token_delta=list(train_token_delta),
        rollout_token_delta=list(rollout_token_delta),
    )
    return forest, initial_obs


def _shard_new():
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(fully_async=True, partial_rollout=False)
    return shard


def _requeue_ir(abort_count: int = 0, prefix=(300,)) -> InflightRequest:
    return InflightRequest(
        request_id="r1",
        parent_state_hash="hash",
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=abort_count,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=0.0,
        pending_token_delta=list(prefix),
    )


def _router_recovery_shard():
    shard = _shard_new()
    shard.args = SimpleNamespace(
        fully_async=True,
        partial_rollout=False,
        partial_rollout_max_aborted_count=None,
        use_slime_router=False,
        router_cb_timeout_duration_secs=0,
    )
    shard._train_generation_open = True
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    group = ResidentGroup(rollout_mode="train", group_id="g1", result_cells={}, sessions=[])
    group.ownership = GroupOwnership.RUNTIME
    session = _SessionRecord(
        group=group, session_id="s1", session_sampling_params={}, result_cell=_SessionResultCell()
    )
    session.forest = SessionForest
    group.sessions.append(session)
    group.result_cells[session.session_id] = session.result_cell
    shard._groups = {group.group_id: group}
    shard._set_runtime_timeouts_active = AsyncMock()
    shard._dispatch_queued_irs_locked = lambda _session: None
    shard._notify_state_change = lambda _group=None: None

    @asynccontextmanager
    async def no_lease(*_args):
        yield

    shard._admission_lease = no_lease
    shard._sglang_request_permit = no_lease
    ir = _requeue_ir(prefix=(300,))
    ir.sampling_params = {"max_new_tokens": 8}
    session.live_irs.add(ir)
    return shard, group, session, ir


def _result(tokens, finish_type, weight_version):
    meta_info = {"prompt_tokens": 1}
    if weight_version is not None:
        meta_info["weight_version"] = weight_version
    return BackendGenerateResult(
        new_tokens=list(tokens),
        new_log_probs=[-0.5] * len(tokens),
        finish_type=finish_type,
        meta_info=meta_info,
        elapsed=0.1,
        new_topk_token_ids=np.array([[t, 100] for t in tokens], dtype=np.int32) if tokens else None,
        new_topk_log_probs=np.array([[-0.5, -2.0] for _ in tokens], dtype=np.float32) if tokens else None,
    )


def _committed(shard, session, ir):
    """Commit the terminal state and return (resp_node, sample)."""
    leaf = session.resp_state_hash_by_request_id[ir.request_id]
    return session.forest.nodes_by_hash[leaf], session.forest.build_sample(
        leaf_state_hash=leaf, tokenizer=_FakeTokenizer()
    )


async def test_apply_generate_result_records_span_per_attempt():
    shard, _group, session, ir = _router_recovery_shard()
    # Span coordinates start at the pending tokens already accumulated before
    # this attempt; drop the fixture's 300-token prefix so the first attempt
    # starts at 0 (otherwise its first span would start at 1).
    ir.pending_token_delta = []
    shard._apply_generate_result(ir, _result([97, 98], "abort", "1"))
    shard._apply_generate_result(ir, _result([99], "stop", "3"))
    assert ir.pending_weight_version_spans == [
        WeightVersionSpan(version="1", token_start=0, token_end=2, attempt_seq=0),
        WeightVersionSpan(version="3", token_start=2, token_end=3, attempt_seq=1),
    ]
    assert ir.weight_version_attempt_count == 2


async def test_apply_generate_result_records_unknown_version_as_none():
    shard, _group, session, ir = _router_recovery_shard()
    ir.pending_token_delta = []
    # The engine did not report (no weight_version key in meta_info): record version=None explicitly.
    shard._apply_generate_result(ir, _result([97], "stop", None))
    assert ir.pending_weight_version_spans == [
        WeightVersionSpan(version=None, token_start=0, token_end=1, attempt_seq=0)
    ]
    # The legacy field keeps its semantics: unknown versions are not appended.
    assert ir.pending_weight_version_delta == []


async def test_apply_generate_result_skips_empty_attempts():
    shard, _group, session, ir = _router_recovery_shard()
    ir.pending_token_delta = []
    # Aborted with no new tokens: no empty span, but the attempt counter still advances.
    shard._apply_generate_result(ir, _result([], "abort", "1"))
    assert ir.pending_weight_version_spans == []
    assert ir.weight_version_attempt_count == 1


async def test_spans_survive_abort_requeue_and_commit():
    shard, _group, session, ir = _router_recovery_shard()
    shard.args.agentic_reasoning_parser = None
    shard.args.agentic_tool_call_parser = None
    shard._generation_backend = SimpleNamespace(tokenizer=_FakeTokenizer())
    forest, observation = _forest_with_initial_obs(
        session_id=session.session_id,
        messages=[{"role": "user", "content": "p"}],
        train_token_delta=[112],
        rollout_token_delta=[112],
    )
    session.forest = forest
    ir.parent_state_hash = observation.state_hash
    ir.pending_token_delta = []
    shard._apply_generate_result(ir, _result([97], "abort", "1"))
    # Abort requeue: spans stay on the IR (same lifetime as pending_weight_version_delta).
    session.queued_irs.append(ir)
    shard._apply_generate_result(ir, _result([98], "stop", "2"))
    ir.pending_status = "completed"
    # Terminal commit: spans land on the resp node and the sample.
    shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
    node, sample = _committed(shard, session, ir)
    assert node.weight_version_spans == [
        WeightVersionSpan(version="1", token_start=0, token_end=1, attempt_seq=0),
        WeightVersionSpan(version="2", token_start=1, token_end=2, attempt_seq=1),
    ]
    # A single resp node flattens at response_start=0, so coordinates are unchanged.
    assert sample.weight_version_spans == node.weight_version_spans
