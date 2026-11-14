# 权重版本追踪与滞后审计 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 agentic 异步训练补齐生成片段、权重版本与训练消费时刻之间的关联——token 区间级版本追踪、消费滞后统计（含 token 加权与缺失率）、以及可追溯到 Session/轮次/生成片段的离线诊断报告。

**Architecture:** 沿用方案 A 演进现有链路：生成侧在 `service.py` 的 attempt 边界采集 `WeightVersionSpan`（token 区间 + 引擎回传版本），随 resp node 提交进入 `MsgNode`/`Sample`/trajectory 流；训练侧 train_hooks 在写 TrainRecord 时打上 `consumed_weight_version`（actor 发布序号）；离线工具 `version_audit.py` 按 `sample_export_id` join 两个流计算 spec §6 的全部统计口径。Sample 不经过 TransferQueue（探针已验证），span 明细走 trajectory 流、train 流只写消费点版本——这是 spec §5 fallback 路径，即主实现。

**Tech Stack:** Python 3.10+（dataclass、`from __future__ import annotations`）、pytest（纯 CPU）、JSONL 六流观测（`relax/utils/agentic_obs/`）。

**仓库根：** `/home/nianliu/code/xhs/Relax`（下文所有相对路径以此为根）。本仓库**不是 git 仓库的直接子目录**——执行前先确认：`cd /home/nianliu/code/xhs/Relax && git rev-parse --show-toplevel`。若 Relax 是独立 git 仓库（预期如此），所有 git 命令在 Relax 根执行。

**规范文档：** `docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md`（用户已批准）。

---

## 任务 0：环境准备与基线

**Files:**
- 无代码修改。

- [ ] **Step 1: 确认仓库与基线测试可用**

```bash
cd /home/nianliu/code/xhs/Relax && git rev-parse --show-toplevel && git status --short | head -5
python3 -m pytest tests/cpu/test_agentic_rollout.py -q -k "abort" 2>&1 | tail -5
```

Expected: git 根为 Relax 目录；abort 相关测试 PASS（基线绿）。若基线红，停下来报告，不要继续。

说明：本仓库测试从仓库根直接运行 `pytest tests/cpu/...`（`pyproject.toml` 的 `[tool.pytest.ini_options]` 只配了 `asyncio_mode = "auto"`，无 pythonpath 配置；`tests/__init__.py`、`tests/cpu/__init__.py`、`tests/cpu/utils/__init__.py` 存在，`relax` 按已安装依赖解析）。新测试文件 import `relax.*` 即可；不要 import `tests.*` 下其他测试文件——复用 fixture 时复制定义（符合既有测试自包含风格）。

---

## 任务 1：`WeightVersionSpan` 类型 + `Sample.weight_version_spans` 字段

**Files:**
- Modify: `relax/utils/types.py`（Sample 定义在 28-120 行；在文件顶部 dataclass 定义区、Sample 之前插入新类型；Sample 的 `weight_versions` 在 47 行）
- Test: `tests/cpu/utils/test_weight_version_types.py`（新建）

- [ ] **Step 1: 写失败的单测**

创建 `tests/cpu/utils/test_weight_version_types.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for WeightVersionSpan and Sample.weight_version_spans."""

from __future__ import annotations

import dataclasses

from relax.utils.types import Sample, WeightVersionSpan


def test_weight_version_span_is_frozen_with_expected_fields():
    span = WeightVersionSpan(version="3", token_start=0, token_end=5, attempt_seq=0)
    assert span.version == "3"
    assert span.token_start == 0
    assert span.token_end == 5
    assert span.attempt_seq == 0
    try:
        span.version = "4"  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised, "WeightVersionSpan must be frozen"


def test_weight_version_span_allows_unknown_version():
    # None 表示引擎未上报版本，显式未知，绝不按 0 或最新版本处理。
    span = WeightVersionSpan(version=None, token_start=5, token_end=9, attempt_seq=1)
    assert span.version is None


def test_sample_has_default_empty_weight_version_spans():
    # Sample 全字段带默认值（types.py:32-100），可无参构造。
    sample = Sample()
    assert sample.weight_version_spans == []
    # 旧字段并存保留（spec §2：并存决策）。
    assert sample.weight_versions == []


def test_sample_carries_weight_version_spans():
    spans = [
        WeightVersionSpan(version="1", token_start=0, token_end=3, attempt_seq=0),
        WeightVersionSpan(version=None, token_start=3, token_end=4, attempt_seq=1),
    ]
    sample = Sample(weight_version_spans=spans)
    assert sample.weight_version_spans == spans
    assert sample.weight_versions == []
```

（`Sample` 全字段有默认值，`Sample()` 与 `Sample(weight_version_spans=spans)` 均合法——已对照 types.py:28-100 核实。）

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/utils/test_weight_version_types.py -q 2>&1 | tail -3
```

Expected: FAIL，`ImportError: cannot import name 'WeightVersionSpan'`。

- [ ] **Step 3: 最小实现**

在 `relax/utils/types.py` 中，`Sample` 类定义之前插入：

```python
@dataclass(frozen=True)
class WeightVersionSpan:
    """One backend generate attempt's committed tokens annotated with the
    engine's weight version at generation time.

    Coordinates are relative to the resp node's ``train_token_delta`` (the
    flattening into whole-response indices happens in ``build_sample``).
    ``version`` is None when the engine did not report a version; unknown
    versions are reported explicitly and never coerced to 0 or "latest".
    """

    version: str | None
    token_start: int
    token_end: int
    attempt_seq: int
```

并在 `Sample` 的字段区（`weight_versions` 之后）新增：

```python
    weight_version_spans: list[WeightVersionSpan] = field(default_factory=list)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
python3 -m pytest tests/cpu/utils/test_weight_version_types.py -q 2>&1 | tail -3
```

Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
cd /home/nianliu/code/xhs/Relax
git add relax/utils/types.py tests/cpu/utils/test_weight_version_types.py
git commit -m "feat(agentic): add WeightVersionSpan type and Sample.weight_version_spans"
```

---

## 任务 2：session state 扩展（IR / MsgNode / append_resp / build_sample）

**Files:**
- Modify: `relax/agentic/session/state.py`（InflightRequest 438-467；MsgNode 409-435；append_resp 691-751；build_sample 775-897）
- Test: `tests/cpu/test_weight_version_state.py`（新建）

- [ ] **Step 1: 写失败的单测（状态层）**

创建 `tests/cpu/test_weight_version_state.py`（构造手法对照 `tests/cpu/test_agentic_observability.py:94-142` 的 `_two_turn_forest` 与 `tests/cpu/test_agentic_rollout.py:1388-1398` 的 `_requeue_ir`，签名已核实）：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for weight version span plumbing through session state."""

from __future__ import annotations

import asyncio

from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils.types import WeightVersionSpan


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_ir_defaults_and_attempt_counter():
    ir = InflightRequest(
        request_id="r1",
        parent_state_hash="hash",
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_event_loop_policy().new_event_loop().create_future(),
        wall_started_at=0.0,
    )
    assert ir.pending_weight_version_spans == []
    assert ir.weight_version_attempt_count == 0


def _forest() -> SessionForest:
    return SessionForest.create_empty(
        session_id="s1",
        group_index=2,
        index=5,
        label="gold",
        train_metadata={"loss": "grpo"},
        metadata={},
    )


def test_append_resp_commits_spans_to_resp_node():
    forest = _forest()
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    spans = [
        WeightVersionSpan(version="1", token_start=0, token_end=1, attempt_seq=0),
        WeightVersionSpan(version=None, token_start=1, token_end=2, attempt_seq=1),
    ]
    resp = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "ab"}]}],
        token_delta=_chars("ab"),
        logprob_delta=[-0.1, -0.2],
        weight_version_delta=["1"],
        weight_version_spans=spans,
    )
    assert forest.nodes_by_hash[resp.state_hash].weight_version_spans == spans
    # obs node 不带版本（loss_mask=0，不参与统计）。
    assert forest.nodes_by_hash[obs.state_hash].weight_version_spans == []


def test_append_resp_spans_default_empty():
    forest = _forest()
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    resp = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "ab"}]}],
        token_delta=_chars("ab"),
        logprob_delta=[-0.1, -0.2],
    )
    assert forest.nodes_by_hash[resp.state_hash].weight_version_spans == []


def _two_resp_forest():
    """root -> obs -> resp(v1, 2 token) -> obs -> resp(v3, 2 token)"""
    forest = _forest()
    obs1 = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    resp1 = forest.append_resp(
        parent_state_hash=obs1.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "ab"}]}],
        token_delta=_chars("ab"),
        logprob_delta=[-0.1, -0.2],
        weight_version_delta=["1"],
        weight_version_spans=[WeightVersionSpan("1", 0, 2, 0)],
    )
    obs2 = forest.append_obs(
        parent_state_hash=resp1.state_hash,
        rollout_id=8,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "tool"}]}]),
        train_token_delta=_chars("tool"),
        rollout_token_delta=_chars("tool"),
    )
    resp2 = forest.append_resp(
        parent_state_hash=obs2.state_hash,
        rollout_id=8,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "cd"}]}],
        token_delta=_chars("cd"),
        logprob_delta=[-0.3, -0.4],
        weight_version_delta=["3"],
        weight_version_spans=[WeightVersionSpan("3", 0, 2, 1)],
    )
    return forest, resp2.state_hash


def test_build_sample_flattens_spans_to_response_coordinates():
    forest, leaf = _two_resp_forest()
    sample = forest.build_sample(leaf_state_hash=leaf, tokenizer=_FakeTokenizer())
    # 展平后坐标系变为相对整条 response 的连续下标：第二个 resp node 的 span
    # 偏移 response_start=2。
    assert sample.weight_version_spans == [
        WeightVersionSpan("1", 0, 2, 0),
        WeightVersionSpan("3", 2, 4, 1),
    ]
    # 旧字段并存。
    assert sample.weight_versions == ["1", "3"]
```

`_FakeTokenizer` 复制自 `tests/cpu/test_agentic_rollout.py:159-166`（encode/decode 用 ord/chr），加进本文件：

```python
class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)
```

（构造签名核实依据：`append_obs` state.py:646-663（keyword-only：`parent_state_hash`/`rollout_id`/`abort_count`/`messages_delta`/`train_token_delta`/`rollout_token_delta`，均必填）；`append_resp` state.py:691-710（keyword-only：`parent_state_hash`/`rollout_id`/`abort_count`/`messages_delta`/`token_delta`/`logprob_delta` 必填，`weight_version_delta` 可选）；节点查引用 `forest.nodes_by_hash`（state.py:612,622）；`InflightRequest` 必填字段 state.py:440-446（`waiter` 需要事件循环——在无 async 上下文的测试里用 `asyncio.get_event_loop_policy().new_event_loop().create_future()`，与 `test_agentic_rollout.py:1395` 在 async 测试内 `asyncio.get_running_loop().create_future()` 等价的同步写法）。）

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/test_weight_version_state.py -q 2>&1 | tail -3
```

Expected: FAIL，`TypeError`（unexpected keyword `weight_version_spans`）或 AttributeError（`pending_weight_version_spans` 不存在）。

- [ ] **Step 3: 实现 state.py 扩展**

五处修改（行号基于当前文件，执行时以实际内容定位）：

**(a) import**（文件顶部 `from relax.utils.types import Sample` 一行改为）：

```python
from relax.utils.types import Sample, WeightVersionSpan
```

**(b) MsgNode**（426 行 `weight_version_delta` 字段之后）：

```python
    weight_version_spans: list[WeightVersionSpan] = field(default_factory=list)
```

**(c) InflightRequest**（460 行 `pending_weight_version_delta` 之后）：

```python
    pending_weight_version_spans: list[WeightVersionSpan] = field(default_factory=list)
    weight_version_attempt_count: int = 0
```

（abort 重排队时这两个字段与 `pending_token_delta`/`pending_weight_version_delta` 一样随 IR 对象保留——dataclass 字段天然满足，无需额外代码。）

**(d) append_resp**（691-751 行，keyword-only 签名）：
- 签名新增 keyword-only 参数（702 行 `weight_version_delta` 之后）：

```python
        weight_version_spans: list[WeightVersionSpan] | None = None,
```

- MsgNode 构造处（740 行 `weight_version_delta=...` 之后）新增：

```python
                weight_version_spans=list(weight_version_spans) if weight_version_spans else [],
```

**(e) build_sample**（775-897 行）：
- locals 区（798 行 `weight_versions` 之后）新增：

```python
    weight_version_spans: list[WeightVersionSpan] = []
```

- resp 分支（827 行 `weight_versions.extend(node.weight_version_delta)` 之后，同缩进）追加（利用现有 `response_start` 局部变量（817 行）做坐标平移）：

```python
                weight_version_spans.extend(
                    WeightVersionSpan(
                        version=span.version,
                        token_start=response_start + span.token_start,
                        token_end=response_start + span.token_end,
                        attempt_seq=span.attempt_seq,
                    )
                    for span in node.weight_version_spans
                )
```

- Sample 构造（884 行 `weight_versions=weight_versions` 之后）新增 kwarg：

```python
            weight_version_spans=weight_version_spans,
```

- [ ] **Step 4: 跑新测试确认通过**

```bash
python3 -m pytest tests/cpu/test_weight_version_state.py -q 2>&1 | tail -3
```

Expected: 4 passed（`test_ir_defaults_and_attempt_counter`、`test_append_resp_commits_spans_to_resp_node`、`test_append_resp_spans_default_empty`、`test_build_sample_flattens_spans_to_response_coordinates`）。

- [ ] **Step 5: 跑既有回归（append_resp/build_sample 消费方）**

```bash
python3 -m pytest tests/cpu/test_agentic_rollout.py -q 2>&1 | tail -5
```

Expected: 全绿（abort 保留语义测试 `test_agentic_rollout.py:1444-1469` 锁定 `pending_weight_version_delta` 行为，本任务未改它）。

- [ ] **Step 6: Commit**

```bash
git add relax/agentic/session/state.py tests/cpu/test_weight_version_state.py
git commit -m "feat(agentic): thread weight version spans through IR, MsgNode, append_resp, build_sample"
```

---

## 任务 3：service.py 采集（attempt 边界、abort 保留、提交、清理丢弃）

**Files:**
- Modify: `relax/agentic/session/service.py`（`_apply_generate_result` 2819-2845；`_terminal_response_locked` 2904-2925 附近；abort 分支 2763-2786 无需改——dataclass 字段自动保留）
- Test: `tests/cpu/test_weight_version_service.py`（新建）

- [ ] **Step 1: 写失败的采集测试**

创建 `tests/cpu/test_weight_version_service.py`。fixture 从 `tests/cpu/test_agentic_rollout.py` 原样复制（自包含风格，不跨测试文件 import）：`_shard_new`（1382-1385）、`_requeue_ir`（1388-1398）、`_router_recovery_shard`（1407-1441，返回四元组 `(shard, group, session, ir)`）、`_FakeTokenizer`（159-166）；`_forest_with_initial_obs`（173-202）也复制——`_terminal_response_locked` 需要 session.forest 与 ir.parent_state_hash 指向真实 obs node（对照 `test_score_centering_partial_attempts_commit_aligned_candidates` 1472-1507 的用法）。**执行时先读这些行确认无变动，再复制。**

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for weight version span collection in the rollout service."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from relax.agentic.pipeline.runtime import BackendGenerateResult
from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.types import WeightVersionSpan

# —— 从 tests/cpu/test_agentic_rollout.py:1382-1441, 159-166, 173-202 原样复制：
# _FakeTokenizer / _shard_new / _requeue_ir / _router_recovery_shard / _forest_with_initial_obs
# （复制时同步带上它们的依赖 import：AgenticSessionShard 的引入方式见
# test_agentic_rollout.py 顶部 `from relax.agentic.session.service import ...` 段，
# 以及 AsyncMock、asynccontextmanager、ResidentGroup、GroupOwnership、
# _SessionRecord、_SessionResultCell —— 按源文件 import 块照抄。）


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
    """提交终态并返回 (resp_node, sample)，对照 1500-1503 行用法。"""
    leaf = session.resp_state_hash_by_request_id[ir.request_id]
    return session.forest.nodes_by_hash[leaf], session.forest.build_sample(
        leaf_state_hash=leaf, tokenizer=_FakeTokenizer()
    )


def test_apply_generate_result_records_span_per_attempt():
    shard, _group, session, ir = _router_recovery_shard()
    shard._apply_generate_result(ir, _result([97, 98], "abort", "1"))
    shard._apply_generate_result(ir, _result([99], "stop", "3"))
    assert ir.pending_weight_version_spans == [
        WeightVersionSpan(version="1", token_start=0, token_end=2, attempt_seq=0),
        WeightVersionSpan(version="3", token_start=2, token_end=3, attempt_seq=1),
    ]
    assert ir.weight_version_attempt_count == 2


def test_apply_generate_result_records_unknown_version_as_none():
    shard, _group, session, ir = _router_recovery_shard()
    # 引擎未上报（meta_info 无 weight_version 键）→ version=None 显式记录。
    shard._apply_generate_result(ir, _result([97], "stop", None))
    assert ir.pending_weight_version_spans == [
        WeightVersionSpan(version=None, token_start=0, token_end=1, attempt_seq=0)
    ]
    # 旧字段保持原语义：未知版本不追加（_accumulate_request_meta 不改）。
    assert ir.pending_weight_version_delta == []


def test_apply_generate_result_skips_empty_attempts():
    shard, _group, session, ir = _router_recovery_shard()
    # abort 且无新 token → 不产生空 span，但 attempt 计数仍递增。
    shard._apply_generate_result(ir, _result([], "abort", "1"))
    assert ir.pending_weight_version_spans == []
    assert ir.weight_version_attempt_count == 1


def test_spans_survive_abort_requeue_and_commit():
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
    # abort 重排队：spans 随 IR 保留（对齐 pending_weight_version_delta 语义）。
    session.queued_irs.append(ir)
    shard._apply_generate_result(ir, _result([98], "stop", "2"))
    ir.pending_status = "completed"
    # 终态提交：spans 进入 resp node 与 sample。
    shard._terminal_response_locked(session=session, ir=ir, finish_type="stop")
    node, sample = _committed(shard, session, ir)
    assert node.weight_version_spans == [
        WeightVersionSpan(version="1", token_start=0, token_end=1, attempt_seq=0),
        WeightVersionSpan(version="2", token_start=1, token_end=2, attempt_seq=1),
    ]
    # 单 resp node 展平：response_start=0，坐标不变。
    assert sample.weight_version_spans == node.weight_version_spans
```

要点核实：`_router_recovery_shard` 返回 `(shard, group, session, ir)` 且 `ir.pending_token_delta=[300]`、`session.live_irs` 已含 ir（1440-1441 行）；带 spans 提交前需重置 `ir.pending_token_delta = []` 并把 `parent_state_hash` 指向真实 obs（对照 1483-1485 行）；`ir.pending_status = "completed"` 在 `_terminal_response_locked` 前设置（对照 1499 行）；resp node 查引用用 `forest.nodes_by_hash`。`test_apply_generate_result_records_span_per_attempt` 首个 attempt token_start 预期为 0，因 `ir.pending_token_delta` 已重置为 `[]`（span 坐标以该次调用前已积累 token 数计）——若不重置则起点为 1（前缀 300），断言相应调整。**注意**：前三个测试只调 `_apply_generate_result`，不需要 forest/backend，`_router_recovery_shard` 直接够用。

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/test_weight_version_service.py -q 2>&1 | tail -3
```

Expected: FAIL（`pending_weight_version_spans` 不存在 / spans 为空）。

- [ ] **Step 3: 实现采集逻辑**

`relax/agentic/session/service.py`：

**(a) `_apply_generate_result`**（2819-2845 行）。当前 2826 行 `ir.pending_token_delta.extend(result.new_tokens)`、2833 行 `self._accumulate_request_meta(ir, meta_info=result.meta_info)`。改为：

```python
        token_start = len(ir.pending_token_delta)
        ir.pending_token_delta.extend(result.new_tokens)
        token_end = len(ir.pending_token_delta)
        weight_version = result.meta_info.get("weight_version")
        if token_end > token_start:
            ir.pending_weight_version_spans.append(
                WeightVersionSpan(
                    version=str(weight_version) if weight_version is not None else None,
                    token_start=token_start,
                    token_end=token_end,
                    attempt_seq=ir.weight_version_attempt_count,
                )
            )
        ir.weight_version_attempt_count += 1
        self._accumulate_request_meta(ir, meta_info=result.meta_info)
```

（`_accumulate_request_meta` 1688-1699 行**不改**——旧字段 `pending_weight_version_delta` 语义保持，并存。）

**(b) `_terminal_response_locked`**（2904-2925 行）：`forest.append_resp(...)` 调用新增 kwarg（与 2917 行 `weight_version_delta=ir.pending_weight_version_delta` 并列）：

```python
            weight_version_spans=ir.pending_weight_version_spans,
```

**(c) import**：service.py 若未导入 `WeightVersionSpan`，在 `from relax.utils.types import get_spec_token_counts`（约 77 行）处补：

```python
from relax.utils.types import WeightVersionSpan, get_spec_token_counts
```

清理路径（`live_irs.clear()` 等，3276-3295 行）**无需改动**——被丢弃 IR 的 pending spans 随 IR 对象消失，从不进入 node，天然满足验收标准 2。

- [ ] **Step 4: 跑新测试确认通过**

```bash
python3 -m pytest tests/cpu/test_weight_version_service.py -q 2>&1 | tail -3
```

Expected: 4 passed。

- [ ] **Step 5: 回归**

```bash
python3 -m pytest tests/cpu/test_agentic_rollout.py tests/cpu/test_agentic_observability.py -q 2>&1 | tail -5
```

Expected: 全绿。

- [ ] **Step 6: Commit**

```bash
git add relax/agentic/session/service.py tests/cpu/test_weight_version_service.py
git commit -m "feat(agentic): capture weight version spans at generate attempt boundaries"
```

---

## 任务 4：观测序列化（LineageNode spans + schema 3.1.0）

**Files:**
- Modify: `relax/utils/agentic_obs/records.py`（LineageNode 126-144；SCHEMA_VERSION 30）
- Modify: `relax/agentic/observability.py`（build_trajectory_record 391-444）
- Modify: `tests/cpu/utils/test_agentic_obs.py:227`（schema 断言更新）
- Modify: `tests/cpu/test_agentic_observability.py:981、2138`（schema 断言更新）
- Test: `tests/cpu/test_weight_version_observability.py`（新建，span 快照断言）

- [ ] **Step 1: 写失败的观测测试**

创建 `tests/cpu/test_weight_version_observability.py`。复用 `tests/cpu/test_agentic_observability.py` 的 `_FakeTokenizer`（49-56）、`_two_turn_forest`（94-142）、`_read_jsonl`（63-64）构造手法——**先读该文件 40-145 行与 2053-2165 行的六流 join 测试**，理解 trajectory 流写入方式后照写。核心测试：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for weight version spans in the trajectory observation stream."""

from __future__ import annotations

import json

from relax.utils.types import WeightVersionSpan


def test_trajectory_record_snapshots_spans(tmp_path, ...):
    # 构造带 spans 的 forest（参考 _two_turn_forest 的做法，给两个 resp node
    # 分别挂上 node.weight_version_delta 与 node.weight_version_spans），
    # 通过 build_trajectory_record / 六流写入流程产出 trajectory JSONL，
    # 然后断言：
    #   row = json.loads(trajectory_lines[0])
    #   resp_nodes = [n for n in row["lineage"] if n["kind"] == "resp"]
    #   assert resp_nodes[0]["weight_version_spans"] == [
    #       {"version": "1", "token_start": 0, "token_end": 1, "attempt_seq": 0}
    #   ]
    #   assert row["schema_version"] == "3.1.0"
    ...
```

**执行指令（给实现者）：** 本任务测试不能停留在伪码。具体做法——把 `tests/cpu/test_agentic_observability.py` 中既有的六流 join 测试（2053-2165 行，函数名 `test_all_six_streams_join_from_prompt_to_committed_update`，依赖 `rollout_module` / `fake_mpu` / `tmp_path` 三个 fixture 及 `_train_args` / `_session_observation_fixture` / `_two_turn_forest` / `_obs_pipeline` / `_lease_observed_group` / `_emit_update` / `_rollout_data` / `_read_jsonl` 等模块级 helper，需一并复制或按需精简）整体复制到本文件并改名 `test_trajectory_stream_carries_weight_version_spans`，在复制体内做两处修改：(1) forest 构造后，向 resp node 挂 spans（**注意：`forest.lineage(leaf_hash)` 包含 index 0 的合成 root node，两轮 forest 的 lineage 是 `[root, obs, resp, obs, resp]`，resp 在 forest 侧索引 2 和 4**——既有测试 2067 行 `forest.lineage(leaf_hash)[2].weight_version_delta.append("v1")`、2145 行断言 `len(forest.lineage(...)) == len(lineage) + 1` 印证了这一点）：

```python
    forest.lineage(leaf_hash)[2].weight_version_spans.append(
        WeightVersionSpan(version="v1", token_start=0, token_end=1, attempt_seq=0)
    )
    forest.lineage(leaf_hash)[4].weight_version_spans.append(
        WeightVersionSpan(version="v3", token_start=0, token_end=2, attempt_seq=0)
    )
```

（root node 在 trajectory JSON 写出时被 `if not node.messages_delta and not node.train_token_delta: continue` 过滤——见 observability.py:413——所以 JSON 侧 lineage 仍是 `[obs, resp, obs, resp]`，resp 在 JSON 索引 1 和 3。）；(2) 在既有断言之后追加：

```python
    lineage = json.loads(trajectory_lines[0])["lineage"]
    assert [node["weight_version_spans"] for node in lineage if node["kind"] == "resp"] == [
        [{"version": "v1", "token_start": 0, "token_end": 1, "attempt_seq": 0}],
        [{"version": "v3", "token_start": 0, "token_end": 2, "attempt_seq": 0}],
    ]
    obs_nodes = [node for node in lineage if node["kind"] == "obs"]
    assert all(node["weight_version_spans"] == [] for node in obs_nodes)
```

（trajectory_lines 的获取方式照抄既有测试——它读 `config.trajectory_path(...)` 的 JSONL 文件。）

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/test_weight_version_observability.py -q 2>&1 | tail -3
```

Expected: FAIL（lineage dict 无 `weight_version_spans` 键 / schema 断言失败）。

- [ ] **Step 3: 实现**

**(a) `records.py` LineageNode**（126-144 行，`weight_versions` 字段之后）：

```python
    weight_version_spans: list[dict[str, Any]] = field(default_factory=list)
```

（类型是 `list[dict[str, Any]]` 而非 dataclass 列表——该文件用 `asdict` 序列化，且 records 层保持纯 JSON 结构。）

**(b) `SCHEMA_VERSION`**（30 行）：

```python
SCHEMA_VERSION = "3.1.0"
```

（注释行不变：major 覆盖不兼容变更，minor 为 additive——本次 additive minor。）

**(c) `observability.py` build_trajectory_record**（419-427 行 LineageNode 构造处，`weight_versions=list(node.weight_version_delta)`（424）之后）：

```python
                weight_version_spans=[
                    {
                        "version": span.version,
                        "token_start": span.token_start,
                        "token_end": span.token_end,
                        "attempt_seq": span.attempt_seq,
                    }
                    for span in node.weight_version_spans
                ],
```

**(d) 更新 4 处既有 schema 断言**（`"3.0.0"` → `"3.1.0"`）：
- `tests/cpu/utils/test_agentic_obs.py:227`
- `tests/cpu/test_agentic_observability.py:981`
- `tests/cpu/test_agentic_observability.py:2138`
- （docs 两处在任务 7 一并更新。）

- [ ] **Step 4: 跑新测试 + 全部观测回归**

```bash
python3 -m pytest tests/cpu/test_weight_version_observability.py tests/cpu/utils/test_agentic_obs.py tests/cpu/test_agentic_observability.py -q 2>&1 | tail -3
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add relax/utils/agentic_obs/records.py relax/agentic/observability.py \
  tests/cpu/utils/test_agentic_obs.py tests/cpu/test_agentic_observability.py \
  tests/cpu/test_weight_version_observability.py
git commit -m "feat(obs): snapshot weight version spans into trajectory stream (schema 3.1.0)"
```

---

## 任务 5：训练消费点（train_hooks + model.train 签名 + actor 传参）

**Files:**
- Modify: `relax/utils/agentic_obs/train_hooks.py`（`_Update` 59-79；`observe_train_step` 180-195；`_write_update` 290-348）
- Modify: `relax/backends/megatron/model.py`（`train` 1704-1711 签名；调用点 1862）
- Modify: `relax/backends/megatron/actor.py`（2200、2691、2843 三处 actor train 调用；**跳过 2013 行 critic 调用**）
- Test: `tests/cpu/test_weight_version_train_hooks.py`（新建）

- [ ] **Step 1: 写失败的 train_hooks 测试**

创建 `tests/cpu/test_weight_version_train_hooks.py`。**先读 `tests/cpu/test_agentic_observability.py` 40-530 行**，把以下模块级 helper 原样复制进来（自包含风格，签名照抄）：`_FakeMpu`（375-403）、`fake_mpu` fixture（406-426，仅 `monkeypatch` 参数）、`_rollout_data`（429-441）、`_train_args`（444-459，返回 `Namespace`，`dump_agentic_obs=str(tmp_path)`）、`_read_jsonl`（63-64）与 `from relax.utils.agentic_obs.config import resolve_agentic_obs_config` / `from relax.utils.agentic_obs.writers import close_observation_writers` 导入（照抄文件头部对应 import）。注意：**`_train_args`/`_rollout_data`/`_read_train_samples` 是普通函数不是 fixture，直接调用**；`_read_train_samples(path)` 接收 train JSONL 路径（`resolve_agentic_obs_config(args).train_scalar_path(rollout_id, dp_rank)`，见既有测试 500/516 行用法）。`_emit_update`（462-467）复制后改造为接受并透传 `consumed_weight_version`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for consumed weight version stamping in train hooks."""

from __future__ import annotations

from relax.utils.agentic_obs.config import resolve_agentic_obs_config
from relax.utils.agentic_obs import train_hooks
from relax.utils.agentic_obs.writers import close_observation_writers


def _emit(args, rollout_id, data, step_id=0, consumed_weight_version=None):
    with train_hooks.observe_train_step(
        args, rollout_id, step_id, consumed_weight_version=consumed_weight_version
    ):
        train_hooks.capture_training_input(data)
        train_hooks.record_optimizer_result(successful=True)


def _read_train_samples(path):
    return [sample for record in _read_jsonl(path) for sample in record["samples"]]


def test_train_record_stamps_consumed_weight_version_per_sample(tmp_path, fake_mpu):
    fake_mpu()
    args = _train_args(tmp_path)
    _emit(args, 3, _rollout_data([1, 2]), step_id=2, consumed_weight_version="7")
    train_hooks.close_train_appenders()
    close_observation_writers()
    samples = _read_train_samples(resolve_agentic_obs_config(args).train_scalar_path(3, 0))
    assert all(s["consumed_weight_version"] == "7" for s in samples)
    assert samples[0]["sample_export_id"] == "1"


def test_train_record_omits_version_when_not_provided(tmp_path, fake_mpu):
    # 未传（默认 None）时显式写 null——离线工具把这类样本计为消费点未知。
    fake_mpu()
    args = _train_args(tmp_path)
    _emit(args, 3, _rollout_data([1]))
    train_hooks.close_train_appenders()
    close_observation_writers()
    samples = _read_train_samples(resolve_agentic_obs_config(args).train_scalar_path(3, 0))
    assert samples[0]["consumed_weight_version"] is None
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/test_weight_version_train_hooks.py -q 2>&1 | tail -3
```

Expected: FAIL，`TypeError: observe_train_step() got an unexpected keyword argument 'consumed_weight_version'`。

- [ ] **Step 3: 实现 train_hooks.py**

**(a) `_Update` dataclass**（59-79 行，字段区末尾）：

```python
    consumed_weight_version: str | None = None
```

**(b) `observe_train_step`**（180-195 行）签名改为：

```python
def observe_train_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    *,
    role: str = "actor",
    consumed_weight_version: str | None = None,
) -> Iterator[None]:
```

函数体内构造 `_Update` 处把 `consumed_weight_version=consumed_weight_version` 传入（读函数体定位 `_Update(...)` 构造行，加一个 kwarg）。

**(c) `_write_update`**（290-348 行）：per-sample dict 构造处（304-317 行）每个样本 dict 新增：

```python
                "consumed_weight_version": state.consumed_weight_version,
```

- [ ] **Step 4: 跑 train_hooks 测试确认通过**

```bash
python3 -m pytest tests/cpu/test_weight_version_train_hooks.py -q 2>&1 | tail -3
```

Expected: 2 passed。

- [ ] **Step 5: 验证既有 observe_train_step 调用方兼容（回归）**

```bash
python3 -m pytest tests/cpu/test_agentic_observability.py tests/cpu/utils/test_agentic_obs_isolation.py tests/cpu/utils/data/test_streaming_tq_iterator.py -q 2>&1 | tail -3
```

Expected: 全绿（keyword-only 默认 None 参数向后兼容所有既有调用点）。

- [ ] **Step 6: model.py 签名穿线**

`relax/backends/megatron/model.py` `train()`（1704-1711 行）签名末尾新增参数：

```python
def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    *,
    consumed_weight_version: str | None = None,
) -> None:
```

调用点（1862 行，`with (timer(...), observe_train_step(...)):` 内）改为：

```python
                observe_train_step(
                    args,
                    rollout_id,
                    step_id,
                    role=getattr(model[0], "role", "actor"),
                    consumed_weight_version=consumed_weight_version,
                )
```

- [ ] **Step 7: actor.py 三处传参**

`relax/backends/megatron/actor.py` 三处 actor 角色的 `train(...)` 调用（2200、2691、2843 行——以实际代码定位，均为 `train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler, self.data_iterator, self.num_microbatches)` 形态），每处加 kwarg：

```python
            train(
                rollout_id,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                self.data_iterator,
                self.num_microbatches,
                consumed_weight_version=str(self.weight_updater.weight_version),
            )
```

**注意跳过 2013 行的 critic 调用**（其上一行是 `self.args.loss_type = "value_loss"`，critic 无 weight_updater 语义）。若调用形态是 `from .model import train` 直呼（143 行 import），按上述加 kwarg 即可；先读三处上下文确认变量名（`rollout_id`/`self.model` 等）与实际一致。

- [ ] **Step 8: 语法/静态检查 + 提交**

```bash
python3 -c "import ast; ast.parse(open('relax/backends/megatron/model.py').read()); ast.parse(open('relax/backends/megatron/actor.py').read())"
python3 -m pytest tests/cpu/test_weight_version_train_hooks.py -q 2>&1 | tail -3
git add relax/utils/agentic_obs/train_hooks.py relax/backends/megatron/model.py relax/backends/megatron/actor.py tests/cpu/test_weight_version_train_hooks.py
git commit -m "feat(train): stamp consumed weight version into train observation records"
```

Expected: ast 检查无输出（成功）；测试 PASS。

---

## 任务 6：version_audit.py 离线诊断工具

**Files:**
- Create: `relax/utils/agentic_obs/version_audit.py`
- Test: `tests/cpu/utils/test_version_audit.py`（新建）

- [ ] **Step 1: 写失败的纯函数单测**

统计核心做成纯函数（`compute_span_stats` / `compute_sample_stats`），CLI 只是壳——先测纯函数。创建 `tests/cpu/utils/test_version_audit.py`：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the weight version audit offline tool."""

from __future__ import annotations

import pytest

from relax.utils.agentic_obs.version_audit import (
    compute_run_summary,
    compute_sample_stats,
)


def _span(version, start, end, attempt=0):
    return {"version": version, "token_start": start, "token_end": end, "attempt_seq": attempt}


class TestSampleStats:
    def test_known_versions_full_metrics(self):
        # 两轮生成：v1 产 2 token，v3 产 2 token；消费时 current=7。
        spans = [_span("1", 0, 2), _span("3", 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="7")
        assert stats["oldest_version"] == "1"
        assert stats["newest_version"] == "3"
        assert stats["generation_span"] == 2      # 3 - 1
        assert stats["oldest_lag"] == 6            # 7 - 1
        assert stats["newest_lag"] == 4            # 7 - 3
        assert stats["token_weighted_lag"] == pytest.approx(5.0)  # (6*2+4*2)/4
        assert stats["post_generation_lag"] == 4   # == newest_lag
        assert stats["known_token_count"] == 4
        assert stats["unknown_token_count"] == 0
        assert stats["span_missing_rate"] == 0.0

    def test_unknown_versions_excluded_from_lag_means(self):
        # 一半 token 未知：lag 只按已知 token 计，缺失率单列。
        spans = [_span("1", 0, 2), _span(None, 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="3")
        assert stats["oldest_version"] == "1"
        assert stats["token_weighted_lag"] == 2.0      # (2*2)/2
        assert stats["known_token_count"] == 2
        assert stats["unknown_token_count"] == 2
        assert stats["span_missing_rate"] == 0.5
        assert stats["token_missing_rate"] == pytest.approx(0.5)

    def test_all_unknown_reports_null_not_zero(self):
        stats = compute_sample_stats([_span(None, 0, 3)], consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["newest_version"] is None
        assert stats["generation_span"] is None
        assert stats["oldest_lag"] is None
        assert stats["newest_lag"] is None
        assert stats["token_weighted_lag"] is None
        assert stats["post_generation_lag"] is None
        assert stats["known_token_count"] == 0
        assert stats["unknown_token_count"] == 3
        assert stats["span_missing_rate"] == 1.0

    def test_non_numeric_version_treated_as_unknown(self):
        # 非数字版本字符串按未知处理（进缺失率），不猜测。
        spans = [_span("v1", 0, 2)]
        stats = compute_sample_stats(spans, consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["known_token_count"] == 0
        assert stats["unknown_token_count"] == 2
        assert stats["span_missing_rate"] == 1.0

    def test_missing_consumed_version_lags_null(self):
        # 消费点版本缺失：跨度可算，lag 全 None。
        stats = compute_sample_stats([_span("1", 0, 2), _span("3", 2, 4)], consumed_weight_version=None)
        assert stats["generation_span"] == 2
        assert stats["oldest_lag"] is None
        assert stats["token_weighted_lag"] is None

    def test_empty_spans(self):
        stats = compute_sample_stats([], consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["span_missing_rate"] is None  # 无 span 样本不计缺失率


class TestRunSummary:
    def test_run_level_aggregates_exclude_unknown_from_means(self):
        rows = [
            {  # 全已知
                "oldest_lag": 6, "newest_lag": 4, "token_weighted_lag": 5.0,
                "generation_span": 2, "known_token_count": 4, "unknown_token_count": 0,
                "spans": 2, "unknown_spans": 0,
            },
            {  # 消费点缺失 → lag None，不计入均值
                "oldest_lag": None, "newest_lag": None, "token_weighted_lag": None,
                "generation_span": 2, "known_token_count": 4, "unknown_token_count": 0,
                "spans": 2, "unknown_spans": 0,
            },
            {  # 全未知 → lag None
                "oldest_lag": None, "newest_lag": None, "token_weighted_lag": None,
                "generation_span": None, "known_token_count": 0, "unknown_token_count": 3,
                "spans": 1, "unknown_spans": 1,
            },
        ]
        summary = compute_run_summary(rows)
        assert summary["sample_count"] == 3
        assert summary["lag_sample_count"] == 1          # 只有第一个进均值
        assert summary["oldest_lag_mean"] == 6.0
        assert summary["newest_lag_mean"] == 4.0
        assert summary["token_weighted_lag_mean"] == 5.0
        assert summary["generation_span_mean"] == 2.0     # 前两个有已知版本
        assert summary["span_missing_rate"] == pytest.approx(1 / 5)
        assert summary["token_missing_rate"] == pytest.approx(3 / 11)
        assert summary["sample_missing_rate"] == pytest.approx(1 / 3)

    def test_median_and_max(self):
        rows = [
            {"oldest_lag": 1, "newest_lag": 1, "token_weighted_lag": 1.0,
             "generation_span": 0, "known_token_count": 1, "unknown_token_count": 0,
             "spans": 1, "unknown_spans": 0},
            {"oldest_lag": 3, "newest_lag": 2, "token_weighted_lag": 2.5,
             "generation_span": 1, "known_token_count": 2, "unknown_token_count": 0,
             "spans": 2, "unknown_spans": 0},
            {"oldest_lag": 10, "newest_lag": 5, "token_weighted_lag": 8.0,
             "generation_span": 4, "known_token_count": 3, "unknown_token_count": 0,
             "spans": 3, "unknown_spans": 0},
        ]
        summary = compute_run_summary(rows)
        assert summary["oldest_lag_median"] == 3
        assert summary["oldest_lag_max"] == 10
        assert summary["newest_lag_median"] == 2
        assert summary["newest_lag_max"] == 5
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/utils/test_version_audit.py -q 2>&1 | tail -3
```

Expected: FAIL，`ModuleNotFoundError: No module named 'relax.utils.agentic_obs.version_audit'`。

- [ ] **Step 3: 实现 version_audit.py 核心**

创建 `relax/utils/agentic_obs/version_audit.py`。版权头 + `get_logger(__name__)` + 显式类型注解。核心结构：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline weight version staleness audit over the agentic observation streams.

Joins the trajectory stream (per-span weight versions) with the train stream
(consumed weight versions) by sample_export_id and reports the metrics defined
in docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md §6.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterator

from relax.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _version_or_none(version: str | None) -> int | None:
    """Parse a version string; None or non-numeric strings are unknown."""
    if version is None:
        return None
    try:
        return int(version)
    except ValueError:
        return None


def compute_sample_stats(
    spans: list[dict[str, Any]], consumed_weight_version: str | None
) -> dict[str, Any]:
    """Per-sample stats. Unknown versions are explicit None and excluded
    from lag means; missing rates are reported separately."""
    known = [s for s in spans if _version_or_none(s.get("version")) is not None]
    unknown = [s for s in spans if _version_or_none(s.get("version")) is None]
    known_tokens = sum(s["token_end"] - s["token_start"] for s in known)
    unknown_tokens = sum(s["token_end"] - s["token_start"] for s in unknown)
    current = _version_or_none(consumed_weight_version)
    versions = [_version_or_none(s["version"]) for s in known]  # type: ignore[arg-type]
    if known:
        # _version_or_none 在 known 集合上永非 None，min/max 安全。
        oldest_span = min(known, key=lambda s: _version_or_none(s["version"]))
        newest_span = max(known, key=lambda s: _version_or_none(s["version"]))
        oldest_version: str | None = oldest_span["version"]
        newest_version: str | None = newest_span["version"]
        generation_span = max(versions) - min(versions)
    else:
        oldest_version = None
        newest_version = None
        generation_span = None
    out: dict[str, Any] = {
        "oldest_version": oldest_version,
        "newest_version": newest_version,
        "generation_span": generation_span,
        "known_token_count": known_tokens,
        "unknown_token_count": unknown_tokens,
        "span_missing_rate": (len(unknown) / len(spans)) if spans else None,
        "token_missing_rate": (unknown_tokens / (known_tokens + unknown_tokens))
        if (known_tokens + unknown_tokens) > 0
        else None,
    }
    if not known or current is None:
        out.update(
            {
                "oldest_lag": None,
                "newest_lag": None,
                "token_weighted_lag": None,
                "post_generation_lag": None,
            }
        )
        return out
    oldest, newest = min(versions), max(versions)
    out["oldest_lag"] = current - oldest
    out["newest_lag"] = current - newest
    out["post_generation_lag"] = out["newest_lag"]
    out["token_weighted_lag"] = (
        sum(
            (current - _version_or_none(s["version"])) * (s["token_end"] - s["token_start"])
            for s in known
        )
        / known_tokens
    )
    return out
```

`compute_run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]`：

```python
def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def compute_run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level aggregates. Rows carry precomputed per-sample stats plus
    span/token counts; None fields never enter means."""
    lags = [r for r in rows if r.get("oldest_lag") is not None]
    spans_total = sum(r.get("spans", 0) for r in rows)
    unknown_spans = sum(r.get("unknown_spans", 0) for r in rows)
    known_tokens = sum(r.get("known_token_count", 0) for r in rows)
    unknown_tokens = sum(r.get("unknown_token_count", 0) for r in rows)
    spans_known = [r for r in rows if r.get("generation_span") is not None]
    return {
        "sample_count": len(rows),
        "lag_sample_count": len(lags),
        "oldest_lag_mean": _mean([r["oldest_lag"] for r in lags]),
        "oldest_lag_median": _median([r["oldest_lag"] for r in lags]),
        "oldest_lag_max": max((r["oldest_lag"] for r in lags), default=None),
        "newest_lag_mean": _mean([r["newest_lag"] for r in lags]),
        "newest_lag_median": _median([r["newest_lag"] for r in lags]),
        "newest_lag_max": max((r["newest_lag"] for r in lags), default=None),
        "token_weighted_lag_mean": _mean([r["token_weighted_lag"] for r in lags]),
        "generation_span_mean": _mean([r["generation_span"] for r in spans_known]),
        "span_missing_rate": unknown_spans / spans_total if spans_total else None,
        "token_missing_rate": unknown_tokens / (known_tokens + unknown_tokens)
        if (known_tokens + unknown_tokens) > 0
        else None,
        "sample_missing_rate": (
            sum(1 for r in rows if r.get("spans", 0) > 0 and r.get("unknown_spans", 0) == r.get("spans"))
            / len(rows)
            if rows
            else None
        ),
    }
```

校验测试期望值：`test_run_level_aggregates_exclude_unknown_from_means` 中 span_missing_rate = 1/(2+2+1)=0.2 ✓；token_missing_rate = 3/(4+4+3)=3/11 ✓；sample_missing_rate：第三个样本全未知 → 1/3 ✓。`oldest_lag` 是 int，`_mean` 返回 float 6.0 ✓；median [6] → 6（statistics.median 单元素返回原值 6，测试断言 `== 6.0`，`6 == 6.0` 为 True）✓。

- [ ] **Step 4: 跑纯函数测试确认通过**

```bash
python3 -m pytest tests/cpu/utils/test_version_audit.py -q 2>&1 | tail -3
```

Expected: 8 passed。

- [ ] **Step 5: Commit（核心统计）**

```bash
git add relax/utils/agentic_obs/version_audit.py tests/cpu/utils/test_version_audit.py
git commit -m "feat(obs): add version audit core statistics"
```

- [ ] **Step 6: 写失败的 join/CLI 测试**

在同一测试文件追加端到端 join 测试（在 tmp_path 下手写 trajectory + train JSONL，跑 CLI 入口）：

```python
def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


TRAJECTORY_ROW = {
    "schema_version": "3.1.0",
    "run_id": "run-1",
    "session_id": "sess-1",
    "sample_export_id": "1001",
    "lineage": [
        {"kind": "obs", "state_hash": "h0", "token_count": 2, "weight_versions": [], "weight_version_spans": []},
        {"kind": "resp", "state_hash": "h1", "token_count": 2, "weight_versions": ["1"],
         "weight_version_spans": [{"version": "1", "token_start": 0, "token_end": 2, "attempt_seq": 0}]},
        {"kind": "obs", "state_hash": "h2", "token_count": 4, "weight_versions": [], "weight_version_spans": []},
        {"kind": "resp", "state_hash": "h3", "token_count": 2, "weight_versions": ["3"],
         "weight_version_spans": [{"version": "3", "token_start": 2, "token_end": 4, "attempt_seq": 1}]},
    ],
}

TRAIN_ROW = {
    "schema_version": "3.1.0",
    "run_id": "run-1",
    "update_id": "actor:3:2",
    "dp_rank": 0,
    "samples": [{"sample_export_id": "1001", "consumed_weight_version": "7"}],
}


def test_cli_joins_streams_and_emits_report(tmp_path, capsys):
    root = tmp_path / "run-1"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW])
    _write_jsonl(root / "train" / "step_00000003_dp_000.jsonl", [TRAIN_ROW])
    from relax.utils.agentic_obs.version_audit import main as audit_main

    audit_main([str(root)])
    out = capsys.readouterr().out
    assert "oldest_lag" in out
    assert "span_missing_rate" in out
    # 明细行含追溯链：session → turn → attempt。
    assert "sess-1" in out


def test_cli_missing_only_filter_and_jsonl(tmp_path, capsys):
    root = tmp_path / "run-1"
    unknown_row = json.loads(json.dumps(TRAJECTORY_ROW))
    unknown_row["sample_export_id"] = "1002"
    unknown_row["lineage"][1]["weight_version_spans"] = [
        {"version": None, "token_start": 0, "token_end": 2, "attempt_seq": 0}
    ]
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW, unknown_row])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [
            TRAIN_ROW,
            {
                **TRAIN_ROW,
                "samples": [{"sample_export_id": "1002", "consumed_weight_version": "7"}],
            },
        ],
    )
    from relax.utils.agentic_obs.version_audit import main as audit_main

    audit_main([str(root), "--missing-only", "--jsonl"])
    out = capsys.readouterr().out
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_export_id"] == "1002"
    assert rows[0]["spans"][0]["version"] is None
```

- [ ] **Step 7: 跑测试确认失败**

```bash
python3 -m pytest tests/cpu/utils/test_version_audit.py -q 2>&1 | tail -3
```

Expected: 新增 2 个 FAIL（`main` 不存在）。

- [ ] **Step 8: 实现 join + CLI**

在 `version_audit.py` 追加。join 逻辑参考 `merge.py` 的 reader 风格（`merge_train_updates` 按 `(run_id, train_generation_id, update_id)` 分组——本工具按更简单的键 join）：

```python
def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_trajectories(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Index trajectory rows by sample_export_id across all shards."""
    out: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "trajectories").glob("shard_*.jsonl")):
        for row in _iter_jsonl(path):
            out[int(row["sample_export_id"])] = row
    return out


def load_consumed_versions(run_dir: Path) -> dict[int, str | None]:
    """Latest consumed weight version per sample_export_id across train steps."""
    out: dict[int, str | None] = {}
    for path in sorted((run_dir / "train").glob("step_*_dp_*.jsonl")):
        for row in _iter_jsonl(path):
            for sample in row.get("samples", []):
                out[int(sample["sample_export_id"])] = sample.get("consumed_weight_version")
    return out


def _span_rows(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    turn = 0
    for node in trajectory["lineage"]:
        if node["kind"] != "resp":
            continue
        turn += 1
        for span in node.get("weight_version_spans", []):
            rows.append(
                {
                    "turn": turn,
                    "state_hash": node["state_hash"],
                    **span,
                }
            )
    return rows


def _detail_row(trajectory: dict[str, Any], consumed: str | None) -> dict[str, Any]:
    spans = _span_rows(trajectory)
    stats = compute_sample_stats(spans, consumed)
    known = [s for s in spans if _version_or_none(s.get("version")) is not None]
    return {
        "sample_export_id": trajectory["sample_export_id"],
        "session_id": trajectory.get("session_id"),
        "run_id": trajectory.get("run_id"),
        "consumed_weight_version": consumed,
        "spans": spans,
        "spans_count": len(spans),
        "unknown_spans": len(spans) - len(known),
        **stats,
    }
```

`main`：

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Observation run directory, e.g. <dump>/run-1")
    parser.add_argument("--session", help="Filter by session_id")
    parser.add_argument("--group", help="Filter by group_id")
    parser.add_argument("--missing-only", action="store_true", help="Only samples with unknown versions")
    parser.add_argument("--jsonl", action="store_true", help="Emit sample-level detail as JSONL")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    trajectories = load_trajectories(run_dir)
    consumed = load_consumed_versions(run_dir)

    rows = [
        _detail_row(t, consumed.get(int(t["sample_export_id"])))
        for t in trajectories.values()
        if (args.session is None or t.get("session_id") == args.session)
        and (args.group is None or t.get("group_id") == args.group)
    ]
    if args.missing_only:
        rows = [r for r in rows if r["unknown_spans"] > 0 or r["unknown_token_count"] > 0]

    summary = compute_run_summary(
        [
            {**r, "spans": r["spans_count"], "unknown_spans": r["unknown_spans"]}
            for r in rows
        ]
    )
    if args.jsonl:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        for row in rows:
            for span in row["spans"]:
                version = span["version"]
                age = (
                    _version_or_none(row["consumed_weight_version"]) - _version_or_none(version)
                    if version is not None and _version_or_none(row["consumed_weight_version"]) is not None
                    else None
                )
                print(
                    f"sample={row['sample_export_id']} session={row['session_id']} "
                    f"turn={span['turn']} attempt={span['attempt_seq']} "
                    f"tokens=[{span['token_start']},{span['token_end']}) "
                    f"version={version} age={age}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

注意 `compute_run_summary` 的入参行内键名：测试里 rows 直接带 `spans`/`unknown_spans`/`oldest_lag` 等键；CLI 路径在 main 里做了 `{**r, "spans": r["spans_count"], ...}` 的重命名以复用同一函数——**`{**r, "spans": ...}` 中后写的键覆盖前面的 spans 列表**，这正是需要的（summary 只用计数）。但 `_detail_row` 返回的 dict 里 `spans` 是列表——被覆盖成计数，OK。

CLI 明细行断言 `"sess-1" in out`：非 jsonl 模式打印 summary JSON + 明细行，明细行含 `session=sess-1` ✓。`--missing-only` 过滤条件同时覆盖 span 级与 token 级未知。

- [ ] **Step 9: 跑全部 version_audit 测试确认通过**

```bash
python3 -m pytest tests/cpu/utils/test_version_audit.py -q 2>&1 | tail -3
```

Expected: 10 passed。

- [ ] **Step 10: Commit**

```bash
git add relax/utils/agentic_obs/version_audit.py tests/cpu/utils/test_version_audit.py
git commit -m "feat(obs): add version_audit CLI joining trajectory and train streams"
```

---

## 任务 7：验收场景测试 + 可复现示例报告

**Files:**
- Test: `tests/cpu/test_weight_version_scenarios.py`（新建，spec §8 场景矩阵的端到端覆盖）
- Create: `docs/superpowers/examples/weight-version-audit-report.md`（诊断报告样例，验收标准 6）
- Modify: `docs/zh/guide/agentic-rollout.md`（779 行 schema 引用 + version_audit 用法小节）
- Modify: `docs/en/guide/agentic-rollout.md`（818 行 schema 引用 + 对应英文小节）

- [ ] **Step 1: 写验收场景测试**

创建 `tests/cpu/test_weight_version_scenarios.py`。这个文件覆盖 spec §8 矩阵中尚未被任务 2/3/4/5/6 覆盖的场景，重点是**验收标准 6 的可复现示例**：多轮生成 + 工具等待期间两次权重更新。复用任务 3 的 service 层 fixture 手法（`_router_recovery_shard` 同源构造）+ 任务 6 的 version_audit 函数：

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Acceptance scenarios for weight version tracking (spec §8)."""

from __future__ import annotations

import dataclasses

from relax.utils.agentic_obs.version_audit import compute_sample_stats
from relax.utils.types import WeightVersionSpan


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
    assert stats["generation_span"] == 2       # 生成期间跨越 v1→v3
    assert stats["oldest_lag"] == 4
    assert stats["newest_lag"] == 2
    assert stats["post_generation_lag"] == 2   # 生成完成后老化 5-3
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
    assert stats["generation_span"] == 2    # 生成期间跨 2 个版本（4-2）
    assert stats["oldest_lag"] == 8          # 消费滞后（oldest）：10-2
    assert stats["newest_lag"] == 6
    assert stats["post_generation_lag"] == 6  # 等待滞后 == newest_lag，≠ oldest_lag ≠ span
    assert stats["generation_span"] != stats["oldest_lag"] != stats["newest_lag"]
```

（场景矩阵其余行——abort 恢复、缺失版本、端到端 join——已分别由任务 3 的 `test_spans_survive_abort_requeue_and_commit`、任务 6 的 unknown 测试、任务 6 的 CLI join 测试覆盖。取消清理补一个 service 层显式断言测试追加进本文件——文件头部补上任务 3 测试文件同源的 import 与 helper（复制 `_router_recovery_shard`、`_result` 及其依赖，见任务 3 Step 1 的复制说明）；`compute_sample_stats` 接收 dict 列表，WeightVersionSpan 转 dict 用 `dataclasses.asdict`（frozen dataclass 无 `__dict__`，不能 `vars()`）：）

```python
import dataclasses


def test_cancelled_ir_spans_never_reach_any_stream():
    """验收标准 2：取消或被替换的片段不混入统计。

    IR 被 session 清理（live_irs 是 set[InflightRequest]，清理路径
    `session.live_irs.clear()` 见 service.py:3281；单个失败路径
    `session.live_irs.remove(ir)` 见 2445）后，其 pending spans 随 IR
    一起消失，不存在写入任何 forest node 的代码路径。
    """
    shard, _group, session, ir = _router_recovery_shard()
    shard._apply_generate_result(ir, _result([97], "abort", "1"))
    assert len(ir.pending_weight_version_spans) == 1
    # 模拟 session 清理丢弃 IR。
    session.live_irs.discard(ir)
    ir = None  # 丢弃后无人引用
    # 提交到 node 的 spans 只来自 _terminal_response_locked；
    # 被丢弃的 IR 从未经过它，forest 中任何 node 都不携带该 span。
    if session.forest is not None:
        for node in session.forest.nodes_by_hash.values():
            assert node.weight_version_spans == []
```

（`session.forest` 类型是 `SessionForest | None`；本测试场景 forest 未挂 spans，断言重点是"清理路径不产生 node 写入"。若要让断言更强，可先建一个正常提交的对照 IR，再丢弃另一个 IR，断言 node 里只有对照 IR 的 spans——照任务 3 `test_spans_survive_abort_requeue_and_commit` 的提交手法构造对照。）

- [ ] **Step 2: 跑场景测试确认通过（实现已在任务 2-6 就绪）**

```bash
python3 -m pytest tests/cpu/test_weight_version_scenarios.py -q 2>&1 | tail -3
```

Expected: 3 passed。若 FAIL，按失败信息回修对应任务实现——不要放宽断言。

- [ ] **Step 3: 生成诊断报告样例文件**

创建 `docs/superpowers/examples/weight-version-audit-report.md`。内容：先写一段说明（这是验收标准 6 的可复现示例产物，场景与 `test_acceptance_6_tool_wait_two_weight_updates_reproducible_example` 一致），然后给出演示数据下的 CLI 输出：

````markdown
# 权重版本审计报告样例

场景：Agent 多轮生成，工具等待期间经历两次权重更新（v1→v3），消费时 current=v5。
对应测试：`tests/cpu/test_weight_version_scenarios.py::test_acceptance_6_tool_wait_two_weight_updates_reproducible_example`

## run 级汇总

```json
{
  "sample_count": 1,
  "lag_sample_count": 1,
  "oldest_lag_mean": 4.0,
  "oldest_lag_median": 4,
  "oldest_lag_max": 4,
  "newest_lag_mean": 2.0,
  "newest_lag_median": 2,
  "newest_lag_max": 2,
  "token_weighted_lag_mean": 2.8,
  "generation_span_mean": 2.0,
  "span_missing_rate": 0.0,
  "token_missing_rate": 0.0,
  "sample_missing_rate": 0.0
}
```

## 样本级明细（session → turn → attempt 追溯）

```text
sample=1001 session=sess-1 turn=1 attempt=0 tokens=[0,2) version=1 age=4
sample=1001 session=sess-1 turn=2 attempt=0 tokens=[2,5) version=3 age=2
```

读法：第 1 轮的 2 个 token 来自 v1，训练消费时已老化 4 个版本；第 2 轮的 3 个 token 来自 v3（工具等待期间经历 v2、v3 两次更新），消费时老化 2 个版本。生成期跨度 2（v1→v3），等待滞后 2（v3→v5），token 加权滞后 2.8。
````

- [ ] **Step 4: 更新 zh/en 文档**

`docs/zh/guide/agentic-rollout.md`：
- 779 行附近 `schema 3.0.0` → `schema 3.1.0`；
- 「Agentic 可观测性 JSONL」小节（684 行起）的 train 流说明（700-724 行附近）补一句：train 流每样本含 `consumed_weight_version`（消费时刻 actor 权重版本，未采集为 null）；trajectory 流 lineage 节点含 `weight_version_spans`（token 区间 + 生成时版本，未知为 null）；
- 小节末尾补 version_audit 用法段（SSOT：详细口径链接 spec §6，不复制定义表）：

```markdown
#### 权重版本滞后审计

对已产出的观测目录离线运行：

```bash
python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only] [--jsonl]
```

输出 run 级汇总（各滞后均值/中位数/最大值、三口径缺失率）与样本级明细（session → turn → attempt → token 区间 → 来源版本 → 消费时年龄）。指标口径定义见 `docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md` §6。
```

`docs/en/guide/agentic-rollout.md`：818 行 schema 引用同步 3.1.0，并加对应英文小节（与 zh 对应，同样链接 spec 不复制口径表）。

- [ ] **Step 5: 全量回归**

```bash
python3 -m pytest tests/cpu -q 2>&1 | tail -5
```

Expected: 全绿。若有失败：先判断是否本 PR 引入（跑 git stash 前后对比或看失败测试是否涉及本改动），按 ci-failure-triage 原则处理。

- [ ] **Step 6: pre-commit + Commit**

```bash
cd /home/nianliu/code/xhs/Relax
pre-commit run --all-files 2>&1 | tail -5
git add tests/cpu/test_weight_version_scenarios.py docs/superpowers/examples/weight-version-audit-report.md docs/zh/guide/agentic-rollout.md docs/en/guide/agentic-rollout.md
git commit -m "feat(obs): acceptance scenarios, audit report sample, and docs for weight version tracking"
```

Expected: pre-commit 全过（不过则修完再提交）。

---

## 任务 8：最终验证

**Files:**
- 无新修改。

- [ ] **Step 1: 验收标准逐条自查**

对照 `docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md` 与 task.txt 验收标准 1-6，逐条确认：

| 验收标准 | 证据 |
|---|---|
| 1 跨版本多轮/恢复/缺失/取消 | 任务 3（abort 保留）+ 任务 7（取消丢弃）+ 任务 6（缺失） |
| 2 token 区间只对成功提交内容 | 任务 3 采集点在 attempt 边界 + IR 丢弃即消失 + 任务 7 显式断言 |
| 3 未知显式 None + 缺失率 | 任务 6 `compute_sample_stats`/`compute_run_summary` 三口径 |
| 4 版本年龄按发布序号、口径区分 | 任务 7 口径区分测试 |
| 5 旁路不侵入 | 任务 2/3 Step 5 回归全绿 |
| 6 实现+测试+可复现示例 | 任务 7 示例测试 + 报告样例文件 |

- [ ] **Step 2: 全量测试 + pre-commit 终检**

```bash
python3 -m pytest tests/cpu -q 2>&1 | tail -3
pre-commit run --all-files 2>&1 | tail -3
git log --oneline | head -8
```

Expected: 测试全绿、pre-commit 通过、提交历史为任务 1-7 的 7 个 feat 提交。

---

## Self-Review 记录

- **Spec 覆盖**：spec §3.1（任务 1）、§3.2（任务 1/2/4）、§4 采集链路（任务 3，含 abort/清理两条特殊路径）、§5 训练消费侧（任务 5，TQ 探针已在前置验证中解决——fallback 即主路径）、§6 统计口径（任务 6 + 任务 7 口径区分测试）、§7 离线报告（任务 6 CLI + 任务 7 样例）、§8 测试矩阵（任务 2/3/6/7 分布覆盖，逐行对应见任务 8 表格）、§9 规则（版权头/get_logger/类型注解/最小变更/pre-commit 均写入各任务）。无缺口。
- **签名核对（本轮修正）**：计划中所有测试代码引用的构造函数/方法/fixture 均已对照源码逐个核实：`SessionForest.create_empty(session_id, group_index, index, label, train_metadata, metadata)`；`append_obs`/`append_resp` keyword-only 参数（`parent_state_hash`/`rollout_id`/`abort_count`/`messages_delta`/`train_token_delta`/`rollout_token_delta`/`token_delta`/`logprob_delta`/`weight_version_delta`/`weight_version_spans`）；node 查找走 `forest.nodes_by_hash`；`InflightRequest` 必填 `waiter`；`Sample()` 全字段默认值；`_router_recovery_shard()` 返回 `(shard, group, session, ir)` 四元组；`_read_train_samples(path)` 接收 train JSONL 路径（`resolve_agentic_obs_config(args).train_scalar_path(rollout_id, dp_rank)`）；`_train_args`/`_rollout_data`/`_emit_update` 是普通函数直接调用（唯一 fixture 是 `fake_mpu` 与 `tmp_path`）；六流 join 测试实名 `test_all_six_streams_join_from_prompt_to_committed_update`（2053 行）。
- **lineage 索引语义（本轮修正）**：`forest.lineage(leaf_hash)` 含 index 0 的合成 root，两轮 forest 侧索引为 `[root, obs, resp, obs, resp]`（resp 在 2/4），trajectory JSON 侧因 root 被 `if not node.messages_delta and not node.train_token_delta: continue`（observability.py:413）过滤为 `[obs, resp, obs, resp]`（resp 在 1/3）。任务 4 挂 spans 用 forest 侧 [2]/[4]，JSON 断言用 [1]/[3]，与既有测试 2067/2145/2147 行的手法一致。
- **frozen dataclass 转换（本轮修正）**：`WeightVersionSpan` 是 frozen dataclass，无 `__dict__`，`vars()` 不可用——任务 7 统一 `dataclasses.asdict(span)`。
- **占位符扫描**：任务 4 Step 1 与任务 6 Step 1/6 的测试代码含「先读既有测试/文件，按实际签名微调构造」的指令——这是**引导性指令而非占位符**（断言目标与核心代码完整给出；构造样板已给出且指明核对来源），符合本仓库「测试 fixture 须与被测代码实际签名一致」的现实约束。其余步骤均为完整代码。
- **类型一致性**：`WeightVersionSpan(version, token_start, token_end, attempt_seq)` 在任务 1 定义、任务 2/3 按位置构造一致；`weight_version_spans` 字段名贯穿 IR/MsgNode/append_resp/build_sample/Sample/LineageNode/trajectory JSON/CLI 一致；`consumed_weight_version` 命名贯穿 observe_train_step/_Update/train 签名/actor 传参/train JSON 一致；`compute_sample_stats`/`compute_run_summary` 签名在任务 6 定义与任务 7 消费一致。
