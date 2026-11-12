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
