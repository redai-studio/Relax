# 权重版本追踪与滞后审计（Agentic 异步训练）— 设计文档

日期：2026-11-10
仓库：Relax（`/home/nianliu/code/xhs/Relax`）
任务来源：`/home/nianliu/code/xhs/task.txt`（【No.15】）
参考：radixark/miles PR #1891、#3334（token-span 版本追踪体系）；miles 现有实现 `miles/utils/types.py` 的 `WeightVersionSpan` / `GroupWeightVersionStats`。

## 1. 背景与目标

异步训练中，多轮 agentic 生成可能跨越多次权重更新，生成完成后还可能等待较长时间才被训练消费。当前 Relax 的现状：

- **生成侧版本数据已存在但从未被关联计算**：每个 resp node 记录 `weight_version_delta: list[str]`（`relax/agentic/session/state.py:426`），attempt 级粒度（每次后端 generate 调用追加一条），无 token 边界；`build_sample` 展平进 `Sample.weight_versions`（`relax/agentic/session/state.py:798-884`），但训练侧无人消费。
- **发布侧版本可获取**：actor 的 `weight_updater.weight_version` 每次 update_weights +1（每 policy 单调递增 int，`relax/backends/megatron/weight_update/update_weight_from_tensor.py:466`）；引擎响应 meta_info 回传字符串形式版本（shard 侧 `_accumulate_request_meta` 捕获，`relax/agentic/session/service.py:1689-1692`）。
- **join 骨架现成**：`sample_export_id`（六流共享）+ `update_id = actor:<rollout_id>:<step_id>`，观测数据走本地 JSONL（`relax/utils/agentic_obs/`）。

**目标**：补齐生成片段、权重版本与训练消费时刻之间的关联——为成功提交的生成片段记录 token 区间及对应权重版本，对实际进入训练的样本统计版本跨度、消费滞后、token 加权滞后及版本缺失率，并提供可追溯到 Session / 轮次 / 生成片段的诊断报告。

**非目标**：不改变现有采样、GRPO 组处理或训练语义（旁路审计）；不实现基于滞后的提交限流（miles 的 `admits()` 机制）；不接 wandb 实时指标。

## 2. 核心决策（已与用户确认）

| 决策点 | 结论 |
| --- | --- |
| 目标仓库 | Relax |
| 版本记录粒度 | token 区间级（span），非 attempt 级 |
| 指标出口 | 六流 JSONL + 离线诊断报告（CLI） |
| 版本号体系 | actor 发布序号（`weight_updater.weight_version`，每 policy 单调递增） |
| 未知版本处理 | 显式 None，不计入 lag 均值分子分母，缺失率单列；绝不按 0 或最新版本处理 |
| 消费滞后采集点 | actor 侧 train_hooks（写 TrainRecord 时计算） |
| 实现方案 | 方案 A：演进现有链路为 token-span，单一数据源 |
| 旧字段 `Sample.weight_versions` | 保留（并存），新增 `weight_version_spans`；旧字段无消费者但按用户要求保留 |
| 示例交付 | 单元测试（可复现场景）+ 诊断报告样例文件 |

## 3. 数据结构

### 3.1 WeightVersionSpan（新类型，放 `relax/utils/types.py`）

```python
@dataclass(frozen=True)
class WeightVersionSpan:
    """One backend generate attempt's committed tokens annotated with the
    engine's weight version at generation time."""
    version: str | None      # None = unknown (engine did not report; pre-recovery)
    token_start: int          # start index within the resp node's train_token_delta
    token_end: int            # end index (exclusive)
    attempt_seq: int          # 0-based backend call ordinal within the request
```

- 坐标系：**相对该 resp node 的 train_token_delta**（不是整条 trajectory 的绝对坐标），导出时样本内多 node 各自独立，诊断报告按 node + attempt 定位片段。
- `version` 保持 str（与引擎回传一致），比较/求差时转 int；非数字版本字符串按未知处理（进缺失率），不猜测。
- `attempt_seq` 用于诊断报告按生成片段归因（abort 恢复后同一 node 内多次 attempt 的区分）。

### 3.2 主链路结构扩展

- `InflightRequest`（`relax/agentic/session/state.py:440` 附近）：新增 `pending_weight_version_spans: list[WeightVersionSpan]`，与 `pending_weight_version_delta` 同生命周期（abort 重排队时跨 attempt 保留）。
- `MsgNode`（`state.py:426` 附近）：新增 `weight_version_spans: list[WeightVersionSpan]`，resp node 提交时随 `append_resp` 入库；obs node 不带版本（loss_mask=0，不参与统计）。
- `Sample`（`relax/utils/types.py:47`）：新增 `weight_version_spans: list[WeightVersionSpan]`（`build_sample` 展平所有 resp node 的 spans 顺序拼接；坐标系随展平转为相对整条 response 的连续下标）。旧字段 `weight_versions: list[str]` 原样保留。
- `LineageNode`（`relax/utils/agentic_obs/records.py:126`）：新增 `weight_version_spans` 字段，trajectory 流同步快照。
- schema：`SCHEMA_VERSION` 3.0.0 → 3.1.0（additive minor）。

## 4. 采集链路（生成侧）

采集点在 `relax/agentic/session/service.py`，演进 `_accumulate_request_meta`（:1689）及其调用处：

1. **每次 backend.generate attempt 返回时**：
   - attempt 开始前记录 `token_start = len(ir.pending_token_delta)`（该次调用前已积累的 token 数）；
   - attempt 返回后 `token_end = len(ir.pending_token_delta)`（本次新增 token 落在 [token_start, token_end)）；
   - `version = meta_info.get("weight_version")`（None 未知也照样记 span，version=None）；
   - 追加 `WeightVersionSpan(version, token_start, token_end, attempt_seq)` 到 `ir.pending_weight_version_spans`；`attempt_seq` 由 IR 上的 attempt 计数器维护。
2. **abort 恢复**：IR 重排队时 `pending_weight_version_spans` 与 token delta 一样跨 attempt 保留（与现有 `pending_weight_version_delta` 行为一致，测试 `tests/cpu/test_agentic_rollout.py:1445-1469` 已锁定该语义）；恢复后的新 attempt 追加新 span。
3. **resp node 提交**（`_terminal_response_locked` → `forest.append_resp`，service.py:2917）：spans 随 node 提交。
4. **取消/被替换的片段**：IR 被 session 清理（`live_irs.clear()` 等，service.py:3276-3295）时 pending spans 随 IR 丢弃，从不进入 node——token 区间天然只与成功提交的生成内容对应，取消或被替换的片段不混入统计。

## 5. 训练消费侧（train_hooks）

`relax/utils/agentic_obs/train_hooks.py` 写 TrainRecord 时为每个样本条目新增：

- `consumed_weight_version: str`：本次 update 消费时刻 actor 的 `weight_updater.weight_version`（train_hooks 运行在 actor 进程内，直接读，与 `relax/backends/megatron/actor.py:3064-3070` 的 ci 校验同源，无跨进程查询）。
- `weight_version_stats`（该样本的统计摘要，仅在消费点可算）：
  ```python
  {
    "oldest_version": str | None,        # min over known spans (None = 全部未知)
    "newest_version": str | None,        # max over known spans
    "generation_span": int | None,       # newest - oldest（生成期跨越的版本数）
    "token_weighted_lag": float | None,  # Σ((current−v)×span_tokens)/Σ(known tokens)
    "known_token_count": int,            # 已知版本 token 数
    "unknown_token_count": int,          # 未知版本 token 数
  }
  ```
- 不复制全量 span 进 train 流（避免膨胀），样本级明细靠 `sample_export_id` 回溯 trajectory 流。

**样本到 actor 的通路风险**：`Sample.weight_version_spans` 经 TransferQueue 是否存活需实现前验证——若 TQ 序列化按字段白名单裁剪，则 fallback：shard 侧 trajectory 流写入完整 span，train 侧只写 `consumed_weight_version`，离线工具 join 后计算 lag（两个时刻各自落盘，时序仍准确）。

## 6. 统计口径（固定定义，写入文档）

设某样本 spans 的已知版本集合 V，token 数按 span 长度计：

| 指标 | 定义 |
| --- | --- |
| 生成期跨度 generation_span | `max(V) − min(V)`，仅已知版本；无已知版本记 None |
| 消费滞后（oldest）oldest_lag | `current − min(V)`（current = consumed_weight_version） |
| 消费滞后（newest）newest_lag | `current − max(V)` |
| token 加权滞后 token_weighted_lag | `Σ((current − v_i) × len_i) / Σ len_i`，仅已知版本的 span |
| 等待滞后 post_generation_lag | `newest_lag`（生成完成后老化的部分）；与生成期跨度（生成过程中跨越）分列呈现 |
| 版本缺失率（span 级） | unknown spans / total spans |
| 版本缺失率（token 加权） | unknown tokens / total tokens |

- 所有 lag 均值**只计入有已知版本的样本/span**；缺失不进分子分母，单独报缺失率（三种口径）。
- 未知版本在报告里显式 `version: null` 行呈现，不存在按 0 或最新版本处理的路径。
- 版本年龄全部基于 actor 发布序号（同一 policy 单调递增）；多 policy 场景各自独立序号，报告按 policy 分组。

## 7. 离线诊断报告

新文件 `relax/utils/agentic_obs/version_audit.py`（reader 侧，与 merge.py 同级，复用其流读取基础设施）：

```
用法：python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only]

输出 1：run 级汇总（stdout / JSON）
  - 版本信息缺失率（span 级 / sample 级 / token 加权）
  - 生成期跨度分布（mean / median / max）
  - 消费滞后分布（oldest / newest / token 加权，mean / median / max）
  - 等待滞后分布

输出 2：样本级明细（--jsonl 或 stdout 表格）
  - 按 sample_export_id → session_id → turn（resp node 序号）→ attempt_seq 展开
  - 每行：token 区间 [start, end)、来源版本、消费时版本年龄（current − version）
  - 过滤参数：--session、--group、--missing-only
```

数据源：trajectories 流（span 明细）join train 流（consumed_weight_version、update_id）。join 键：`(run_id, sample_export_id)`；多 DP shard 按 update_id + dp_rank 聚合。

## 8. 测试与可复现示例

测试放 `tests/cpu/`（纯 CPU、无 GPU 依赖，符合 Relax 测试放置规则）：

| 场景 | 断言要点 | 对应验收标准 |
| --- | --- | --- |
| 多轮生成 + 工具等待期间两次权重更新 | 各轮 token 的 span 版本来源正确（v1/v2/v3 分属各轮）；消费时版本年龄数值正确 | 6 |
| abort 恢复 | pending spans 跨 attempt 保留；恢复后新 attempt 新 span、attempt_seq 递增 | 1 |
| 取消清理 | 被丢弃 IR 的 spans 不出现在 node / trajectory / train 任何流 | 2 |
| 缺失版本 | version=None 显式呈现；缺失率三口径正确；不进 lag 均值 | 3 |
| 口径区分 | generation_span vs consumption_lag vs post_generation_lag 三者数值互不混淆 | 4 |
| 端到端 join | trajectory ↔ train 合并产出样本级报告行（含追溯链 session → turn → attempt） | 7/9 |
| 旁路不侵入 | 现有 GRPO 组处理、采样语义测试全绿（回归） | 5 |

报告样例：单测产物或独立小脚本生成一份诊断报告示例文件，直观展示「Agent 在工具等待期间经历两次权重更新后，各轮 token 的来源及训练消费时的版本年龄」（验收标准 6 的可复现示例）。

## 9. 影响面与规则遵守

- 修改文件集中在：`relax/utils/types.py`、`relax/agentic/session/state.py`、`relax/agentic/session/service.py`、`relax/agentic/observability.py`、`relax/utils/agentic_obs/records.py`、`relax/utils/agentic_obs/train_hooks.py`、新增 `relax/utils/agentic_obs/version_audit.py`、`tests/cpu/` 下测试、docs 对应章节。
- 遵循 Relax 仓库规则：版权头、`get_logger`、显式类型注解、最小变更、pre-commit；`.claude/rules` SSOT（span 单一数据源）；新增测试按「先评估增量价值」原则限定在上表场景。
- schema bump 3.1.0 additive；旧字段 `weight_versions` 保留，向后兼容。

### 10.1 TQ 序列化风险与 fallback 的判定标准

实现第 4 步时先跑一个探针：构造带 `weight_version_spans` 的 Sample 过 TransferQueue 序列化往返，断言字段存活。失败则启用 fallback（train 流只写 `consumed_weight_version`，lag 计算移入 version_audit.py 离线 join）。判定以探针结果为准，不在设计层预设。

## 10. 实现顺序（供后续计划参考）

1. `WeightVersionSpan` 类型 + `InflightRequest`/`MsgNode`/`Sample`/`LineageNode` 扩展（纯数据结构，先写单测）
2. service.py 采集逻辑（attempt 边界、abort 保留、清理丢弃）
3. observability/records 序列化（schema 3.1.0）
4. train_hooks 消费点计算（含 TQ 存活性验证，失败则走 fallback）
5. version_audit.py 离线工具
6. 测试补全 + 报告样例 + 文档
