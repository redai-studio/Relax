# 权重版本滞后审计

异步训练中，多轮 agentic 生成可能跨越多次权重更新，生成完成后还可能等待较长时间才被训练消费。权重版本滞后审计把这两类滞后分开度量：**生成期跨度**（生成过程中跨越的版本变化，来源是工具等待期间的多次权重发布）与**等待滞后**（生成完成后、训练消费前老化的版本数），并支持按 token 加权的滞后与版本信息缺失率。首版为旁路审计能力，不改变现有采样与 GRPO 组处理语义。

## 数据来源

审计消费 Agentic 可观测性 JSONL 的两个流（目录结构见 [Agentic Rollout](./agentic-rollout.md)，schema `3.2.0`）：

| 流 | 字段 | 含义 |
|---|---|---|
| `trajectories/*.jsonl` | `lineage[].weight_version_spans` | 每 resp 节点、每次后端 generate attempt 成功提交的 token 区间与生成时引擎权重版本；引擎未上报为 `null`。坐标是**节点内相对坐标**（每轮从 0 起算）。 |
| `train/*.jsonl` | `samples[].consumed_weight_version` | 训练实际消费该样本时 actor 的权重版本（发布序号，每 policy 单调递增；未采集为 `null`）。 |

两流按 `sample_export_id` join。版本号来自同一发布序号体系：colocate/hybrid 模式取 `weight_updater.weight_version`，DCS 模式取 `DeviceDirectBackend.weight_version`。

## 快速开始

对已产出的观测目录离线运行（`<run_dir>` 是 `<dump_dir>/<run_id>`，即包含 `trajectories/` 与 `train/` 的目录）：

```bash
python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only] [--jsonl]
```

| 参数 | 含义 |
|---|---|
| `--session S` | 只审计指定 `session_id` 的样本 |
| `--group G` | 只审计指定 `group_id` 的样本 |
| `--missing-only` | 只报告存在未知 span 或未知 token 的样本 |
| `--jsonl` | 每样本输出一行 JSON 明细，而非汇总报告 |

默认输出 run 级汇总 JSON 与样本级明细行。一个可复现的完整示例（多轮生成 + 工具等待期间两次权重更新）见 `docs/superpowers/examples/weight-version-audit-report.md`，对应测试 `tests/cpu/test_weight_version_scenarios.py`。

## 指标口径

设某样本所有 span 中已知版本的集合为 V，token 数按 span 长度计，`current` 为 `consumed_weight_version`：

| 指标 | 定义 |
|---|---|
| 生成期跨度 `generation_span` | `max(V) − min(V)`，仅已知版本；无已知版本记 `null` |
| 消费滞后（oldest）`oldest_lag` | `current − min(V)` |
| 消费滞后（newest）`newest_lag` | `current − max(V)` |
| token 加权滞后 `token_weighted_lag` | `Σ((current − v_i) × len_i) / Σ len_i`，仅已知版本的 span |
| 等待滞后 `post_generation_lag` | 等于 `newest_lag`（生成完成后老化的部分），与生成期跨度分列呈现 |
| 缺失率（span 级） | 未知版本 span 数 / 总 span 数 |
| 缺失率（token 加权） | 未知版本 token 数 / 总 token 数 |
| 缺失率（sample 级） | 全部 span 均未知的样本数 / 总样本数 |

口径要点：

- **未知版本显式呈现**：引擎未上报（`null`）或非数字版本字符串均按未知处理，绝不按版本 0 或最新版本猜测；lag 均值/中位数/最大值**只计入有已知版本的样本**，缺失不进分子分母，由三种缺失率单独量化。
- **读法**：生成期跨度大说明生成过程跨越了多次发布（如工具等待期间被多次 update_weights 超过）；等待滞后大说明样本在生成完成后排队过久才被消费。
- 明细行的 token 区间是 resp 节点内相对坐标；映射到整条 response 的展平坐标由 `build_sample` 完成（`Sample.weight_version_spans`）。

## 明细行格式

```text
sample=<export_id> session=<session_id> turn=<resp节点序号> attempt=<attempt_seq> tokens=[<start>,<end>) version=<来源版本|None> age=<消费时年龄|None>
```

`age = current − version`，任一侧未知则为 `None`。abort 恢复后同一轮的多次 attempt 通过 `attempt` 区分，可追溯到具体生成片段。

## 容错行为

- **撕裂尾行**：writer 允许进程崩溃留下半行 JSON；读取器跳过并告警，不中断报告——崩溃后的 staleness 事后分析正是本工具的典型场景。
- **overflow 分片**：训练侧背压写入的 `overflow_step_*_dp_*.jsonl` 是同 `(rollout_id, dp_rank)` 的独立 update，按数值序读取，保证较新的消费戳生效。
- **缺失流**：`run_dir` 不是目录时 CLI 直接报错退出；目录下无流文件时汇总中各缺失率为 `null`（无分母），不按 0 处理。
