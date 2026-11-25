# 权重版本滞后审计

异步训练中，多轮 agentic 生成可能跨越多次权重更新，生成完成后还可能等待较长时间才被训练消费。权重版本滞后审计把这两类滞后分开度量：**生成期跨度**（生成过程中跨越的版本变化，来源是工具等待期间的多次权重发布）与**等待滞后**（生成完成后、训练消费前老化的版本数），并支持按有效 token 加权的滞后与版本信息缺失率。首版为旁路审计能力，不改变现有采样与 GRPO 组处理语义。

## 数据来源

审计消费 Agentic 可观测性 JSONL 的两个流（目录结构见 [Agentic Rollout](./agentic-rollout.md)，schema `3.2.0`）：

| 流 | 字段 | 含义 |
|---|---|---|
| `trajectories/*.jsonl` | `lineage[].weight_version_spans` | 每 resp 节点、每次后端 generate attempt 成功提交的 token 区间与生成时引擎权重版本；引擎未上报为 `null`。坐标是**节点内相对坐标**（每轮从 0 起算）。 |
| `train/*.jsonl` | `samples[].consumed_weight_version` | 训练实际消费该样本时 actor 的权重版本（发布序号，每 policy 单调递增；未采集为 `null`）。 |
| `train/*.jsonl` | `samples[].loss_mask.ranges` | 训练实际生效的 loss mask（response 空间左闭右开区间）。token 加权指标只统计这些 token。 |

两流按 `sample_export_id` join。审计先把 lineage 中每个 resp 节点的 span 平移到 response 坐标（加上该节点之前已提交的 response token，包括中间的工具观测），使 span 与 loss mask 指同一批 token。版本号来自同一发布序号体系：colocate/hybrid 模式取 `weight_updater.weight_version`，DCS 模式取 `DeviceDirectBackend.weight_version`（仅刷新 actor_fwd、不向 rollout 发布的同步不计入该序号，避免把一次实际发布差算成两个）。

## 统计总体

**只有训练流中有记录的样本进入训练统计总体。** Session 在 reward/filter 之前就写 trajectory，评估导出也写同一份流，因此被过滤、仅评估、或 update 尚未落盘的导出不会出现在 `train/*.jsonl` 里。这些导出在报告里单独列为 `unconfirmed`（`unconfirmed_sample_count` 与带 `reason=no-train-record` 的明细行），不计入 `sample_count`、滞后均值或缺失率。

反过来，训练流里有记录但 `consumed_weight_version` 为 `null` 的样本仍算已确认消费，只是消费点版本未知，其 lag 报 `null`。

## 快速开始

对已产出的观测目录离线运行（`<run_dir>` 是 `<dump_dir>/<run_id>`，即包含 `trajectories/` 与 `train/` 的目录）：

```bash
python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only] [--jsonl]
```

| 参数 | 含义 |
|---|---|
| `--session S` | 只审计指定 `session_id` 的样本 |
| `--group G` | 只审计指定 `group_id` 的样本 |
| `--missing-only` | 只报告存在未知 span 或未知 token 的样本（含未确认导出中的未知 span） |
| `--jsonl` | 每样本输出一行 JSON 明细，而非汇总报告 |

默认输出 run 级汇总 JSON 与样本级明细行。`--jsonl` 的行带 `confirmed` 字段区分已确认训练样本与未确认导出。

## 指标口径

设某样本所有 span 中已知版本的集合为 V，span 的 token 数按**落在有效 loss mask 内**的长度计，`current` 为 `consumed_weight_version`：

| 指标 | 定义 |
|---|---|
| 生成期跨度 `generation_span` | `max(V) − min(V)`，仅已知版本；无已知版本记 `null` |
| 消费滞后（oldest）`oldest_lag` | `current − min(V)` |
| 消费滞后（newest）`newest_lag` | `current − max(V)` |
| token 加权滞后 `token_weighted_lag` | `Σ((current − v_i) × len_i) / Σ len_i`，仅已知版本的 span，`len_i` 为有效 token 数 |
| 等待滞后 `post_generation_lag` | 等于 `newest_lag`（生成完成后老化的部分），与生成期跨度分列呈现 |
| 有效 token 数 `effective_token_count` | 所有 span 落在 loss mask 内的 token 数 |
| 缺失率（span 级） | 未知版本 span 数 / 总 span 数 |
| 缺失率（token 加权） | 未知版本有效 token 数 / 有效 token 数 |
| 缺失率（sample 级） | 全部 span 均未知的样本数 / 总样本数 |

口径要点：

- **未知版本显式呈现**：引擎未上报（`null`）或非数字版本字符串均按未知处理，绝不按版本 0 或最新版本猜测；lag 均值/中位数/最大值**只计入有已知版本的样本**，缺失不进分子分母，由三种缺失率单独量化。
- **只统计实际参与训练的 token**：`--rollout-sample-filter-path` 可以把样本的 loss mask 清零但样本仍照常进入 GRPO 组处理，训练记录也保留这类行。token 加权滞后与 token 缺失率只按 mask 内的 token 计，否则被清零的样本会用从未训练的 token 稀释滞后；版本级指标（`oldest_lag` / `newest_lag` / `generation_span`）仍覆盖该样本全部已知 span。
- **无有效 token 的样本单列**：若某样本的有效 token 数为 0（全零 mask），其 `token_weighted_lag` 与 `token_missing_rate` 为 `null`，并计入汇总的 `zero_effective_sample_count`，不参与 token 加权均值。
- **读法**：生成期跨度大说明生成过程跨越了多次发布（如工具等待期间被多次 update_weights 超过）；等待滞后大说明样本在生成完成后排队过久才被消费。
- 明细行的 token 区间是 response 坐标，可直接与训练 loss mask 对照；`Sample.weight_version_spans` 使用同一坐标系。

### 示例

Agent 多轮生成，工具等待期间经历两次权重更新（v1 → v3），消费时 `current = v5`：第 1 轮的 2 个 token 来自 v1，第 2 轮的 3 个 token 来自 v3。

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
  "effective_token_count": 5,
  "zero_effective_sample_count": 0,
  "span_missing_rate": 0.0,
  "token_missing_rate": 0.0,
  "sample_missing_rate": 0.0,
  "unconfirmed_sample_count": 0
}
```

```text
sample=1001 session=sess-1 turn=1 attempt=0 tokens=[0,2) version=1 age=4
sample=1001 session=sess-1 turn=2 attempt=0 tokens=[2,5) version=3 age=2
```

生成期跨度 2（v1 → v3），等待滞后 2（v3 → v5），token 加权滞后 `(4×2 + 2×3) / 5 = 2.8`。示例的两轮之间没有工具观测 token，因此第 2 轮从 2 起算；若中途有观测，坐标会再后移观测长度。对应测试 `tests/cpu/test_weight_version_scenarios.py`。

## 明细行格式

```text
sample=<export_id> session=<session_id> turn=<resp节点序号> attempt=<attempt_seq> tokens=[<start>,<end>) version=<来源版本|None> age=<消费时年龄|None>
sample=<export_id> session=<session_id> unconfirmed=true reason=no-train-record spans=<span数>
```

`age = current − version`，任一侧未知则为 `None`。abort 恢复后同一轮的多次 attempt 通过 `attempt` 区分，可追溯到具体生成片段。

## 容错行为

- **撕裂尾行**：writer 允许进程崩溃留下半行 JSON；读取器跳过并告警，不中断报告——崩溃后的 staleness 事后分析正是本工具的典型场景。
- **overflow 分片**：训练侧背压写入的 `overflow_step_*_dp_*.jsonl` 是同 `(rollout_id, dp_rank)` 的独立 update，按数值序读取，保证较新的消费戳生效。
- **缺失 mask**：旧记录没有 `loss_mask` 时不当作全零 mask，按 span 全长统计，避免把未知证据算作未训练。
- **缺失流**：`run_dir` 不是目录时 CLI 直接报错退出；目录下无流文件时汇总中各缺失率为 `null`（无分母），不按 0 处理。
