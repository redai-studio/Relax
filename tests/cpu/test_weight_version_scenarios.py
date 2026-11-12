# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Acceptance-scenario tests for weight version tracking metrics and span isolation."""

from __future__ import annotations

import asyncio
import dataclasses
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
from relax.utils.agentic_obs.version_audit import compute_sample_stats
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


def _requeue_ir(abort_count: int = 0, prefix=(300,), request_id: str = "r1") -> InflightRequest:
    return InflightRequest(
        request_id=request_id,
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


def test_acceptance_6_tool_wait_two_weight_updates_reproducible_example():
    """可复现示例：Agent 在工具等待期间经历两次权重更新后，各轮 token
    的来源及训练消费时的版本年龄。

    时间线：v1 生成第 1 轮 resp（2 token）→ 工具等待 → v2 发布 → v3 生成
    第 2 轮 resp（3 token）→ 等待 → v4、v5 发布 → 训练消费（current=5）。
    """
    spans = [
        WeightVersionSpan(version="1", token_start=0, token_end=2, attempt_seq=0),
        WeightVersionSpan(version="3", token_start=2, token_end=5, attempt_seq=0),
    ]
    stats = compute_sample_stats(
        [dataclasses.asdict(span) for span in spans],
        consumed_weight_version="5",
    )
    # 各轮 token 来源：第 1 轮 v1（年龄 5-1=4），第 2 轮 v3（年龄 5-3=2）。
    assert stats["oldest_version"] == "1"
    assert stats["newest_version"] == "3"
    assert stats["generation_span"] == 2  # 生成期间跨越 v1→v3
    assert stats["oldest_lag"] == 4
    assert stats["newest_lag"] == 2
    assert stats["post_generation_lag"] == 2  # 生成完成后老化 5-3
    assert stats["token_weighted_lag"] == (4 * 2 + 2 * 3) / 5  # 2.8
    assert stats["known_token_count"] == 5
    assert stats["unknown_token_count"] == 0


def test_metric_distinction_generation_span_vs_consumption_vs_waiting():
    """口径区分：generation_span / consumption_lag / post_generation_lag
    三者数值互不混淆（验收标准 4）。"""
    spans = [
        {"version": "2", "token_start": 0, "token_end": 1, "attempt_seq": 0},
        {"version": "4", "token_start": 1, "token_end": 2, "attempt_seq": 0},
    ]
    stats = compute_sample_stats(spans, consumed_weight_version="10")
    assert stats["generation_span"] == 2  # 生成期间跨 2 个版本（4-2）
    assert stats["oldest_lag"] == 8  # 消费滞后（oldest）：10-2
    assert stats["newest_lag"] == 6
    assert stats["post_generation_lag"] == 6  # 等待滞后 == newest_lag，≠ oldest_lag ≠ span
    assert stats["generation_span"] != stats["oldest_lag"] != stats["newest_lag"]


async def test_cancelled_ir_spans_never_reach_any_stream():
    """验收标准 2：取消或被替换的片段不混入统计。

    IR 被 session 清理后，其 pending spans 随 IR 一起消失，不存在写入任何
    forest node 的代码路径；对照 IR 的 spans 正常提交。
    """
    shard, _group, session, ir = _router_recovery_shard()
    # 对照 IR：正常提交，spans 进入 resp node。
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
    shard._apply_generate_result(ir, _result([97], "stop", "2"))
    ir.pending_status = "completed"
    shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
    leaf = session.resp_state_hash_by_request_id[ir.request_id]
    committed_spans = session.forest.nodes_by_hash[leaf].weight_version_spans
    assert committed_spans == [WeightVersionSpan(version="2", token_start=0, token_end=1, attempt_seq=0)]
    # 被丢弃的 IR：先累积 pending span，再被 session 清理（无人引用）。
    dropped = _requeue_ir(request_id="r2", prefix=())
    dropped.sampling_params = {"max_new_tokens": 8}
    session.live_irs.add(dropped)
    shard._apply_generate_result(dropped, _result([97], "abort", "1"))
    assert len(dropped.pending_weight_version_spans) == 1
    session.live_irs.discard(dropped)
    dropped = None
    # 清理路径不产生任何 node 写入：forest 中只有对照 IR 的 spans。
    for node in session.forest.nodes_by_hash.values():
        if node is session.forest.nodes_by_hash[leaf]:
            assert node.weight_version_spans == committed_spans
        else:
            assert node.weight_version_spans == []
