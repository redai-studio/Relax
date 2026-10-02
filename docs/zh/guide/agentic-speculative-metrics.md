# Agentic 投机解码指标

Agentic Rollout 的投机解码指标按当前 rollout batch 中导出样本实际覆盖的 committed generation 聚合。

## 指标口径

保留现有指标名称，但修改聚合方式：

| 指标 | 旧口径 | 新口径 |
| --- | --- | --- |
| `spec_accept_rate` | sample 接受率的算术平均 | accepted 总数 / proposed 总数 |
| `spec_accept_length` | sample completion/verify 比值的算术平均 | completion 总数 / verify 总数 |

同时新增两个 coverage 指标：

- `spec_accept_rate_coverage`：同时提供 accepted 与 proposed counters 的 accounting records 比例。
- `spec_accept_length_coverage`：同时提供 verify 与 completion counters 的 accounting records 比例。

明确返回的零值属于有效数据；counter key 不存在表示 backend 未提供该计数。若聚合后的分母为 0，则不输出对应 ratio，避免伪造 `0%`。

## Generation identity 与去重

每个 committed generation 使用 `(session_id, request_id)` 标识。

`state_hash` 表示 conversation state，而不是一次 generation execution。不同请求即使收敛到同一个 state，也仍然作为独立 generation 分别统计。

多个导出 trajectory 若共享同一个 committed generation，该 generation 在一个 rollout metric batch 内只统计一次。

只导出 response state 被当前 sample lineage 覆盖的 committed generations；未被当前导出覆盖的 sibling state 不参与统计。

## 导出 metadata

新的 Agentic export 在以下位置携带 sparse speculative accounting：

`metadata.agentic_trace.spec_generations`

每条记录包含：

- `request_id`
- `resp_state_hash`
- backend 实际提供的 speculative counters

支持的 counter 字段：

- `spec_accept_token_num`
- `spec_draft_token_num`
- `spec_verify_ct`
- `completion_token_num`

backend 未提供的 counter key 保持缺失。

## 兼容性

没有 generation-level accounting carrier 的 sample 继续通过 `Sample.spec_info` 参与 fallback 聚合。

历史 `spec_info` 中的 `0 / 0` 无法判断是 backend 明确返回零，还是旧代码默认补零，因此只有分母为正时才视为该 counter pair 有覆盖，从而避免从已经丢失的信息推导伪造的 `0%`。

## 人工核对示例

假设当前 export 包含两条 trajectory：

    A -> B
    A -> C

committed generation counters 为：

| Generation | Accepted | Proposed | Verify | Completion |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 2 | 1 | 2 |
| B | 2 | 4 | 2 | 3 |
| C | 9 | 10 | 3 | 6 |

两个 sample 分别包含 `[A, B]` 与 `[A, C]`，共享的 A 在 batch 内只计算一次。

汇总结果：

    accepted   = 1 + 2 + 9  = 12
    proposed   = 2 + 4 + 10 = 16
    verify     = 1 + 2 + 3  = 6
    completion = 2 + 3 + 6  = 11

因此：

    spec_accept_rate        = 12 / 16 = 75%
    spec_accept_length      = 11 / 6 = 1.833333...
    acceptance coverage     = 3 / 3 = 100%
    accept-length coverage  = 3 / 3 = 100%

对于 Task No.5 中两个独立 generation 的 `1 / 2` 与 `9 / 10`，聚合结果为：

    (1 + 9) / (2 + 10) = 10 / 12 = 83.33%
