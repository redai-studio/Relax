# 投机解码指标

通过 `--sglang-speculative-algorithm` 开启投机解码后，每个 rollout 指标批次只统计本次导出样本覆盖的已提交生成。计数按生成去重并求和，再计算比率。

## 记录的指标

现有日志入口将训练指标记录在 `rollout/` 下，评估指标记录在 `eval/<dataset>/` 下。未开启投机解码时不记录这些指标。

| 指标 | 含义 |
|---|---|
| `spec_accept_rate` | 接受的草稿 token 总数除以提议的草稿 token 总数，只使用这两个计数均可用的生成。 |
| `spec_accept_length` | completion token 总数除以 verify 总次数，只使用这两个计数均可用且 verify 次数大于 0 的生成。 |
| `spec_nodes_total` | 去重后的计入生成数；没有生成标识的样本按一条记录计入。 |
| `spec_nodes_missing_counts` | 四个计数中至少一个不可用的记录数。 |
| `spec_coverage` | 四个计数均可用的记录占比。 |
| `spec_accept_rate_coverage` | 接受数、提议数均可用的记录占比。 |
| `spec_accept_length_coverage` | completion 数、verify 次数均可用的记录占比。 |
| `spec_legacy_samples` | 无法恢复生成标识的旧版样本数，仅在大于 0 时记录。 |

覆盖率均以 `spec_nodes_total` 对应的去重记录数为分母。明确上报的零值属于可用字段；配对分母之和为 0 时不输出对应比率。明确上报分子为 0、分母为正数时，零比率是真实有效的。缺失、`null`、负数、非整数或其他非法计数保持未知，不会被伪造成零分子。空批次返回空指标。

部分上报时，一个比率可以有效，另一个不可用。例如接受数/提议数为 1/2、verify 次数为 1，但缺少 completion 数时，输出 `spec_accept_rate=0.5`，不输出 `spec_accept_length`；完整计数、接受率和产出率的覆盖率分别为 0、1、0。

## 生成与恢复计数

生成是提交到 Session 的一次请求，以 `req_<session_id>_<sequence>` 标识。导出样本保留其路径上的稀疏请求计数记录。批次内每个标识只计一次：`A -> B` 与 `A -> C` 共享 A；生成相同文本的独立请求仍分别计数，不同 Session 也不会串计数。已提交但不在导出路径上的兄弟分支不参与统计。

被中断后恢复的请求保持同一个标识。某个字段只有在每次后端尝试中都被上报，才能作为完整请求的计数；其值为各次尝试之和。任一次缺失都会让该字段保持未知，与尝试顺序无关。例如第一次只上报 completion = 5，第二次上报完整计数且 completion = 2，那么已知 completion 总数为 7，但无法确定完整的接受数、提议数或 verify 次数，因此两个比率都不输出，避免把部分工作显示成完整覆盖。

同一已提交生成的重复导出记录可以补全缺失字段，已有字段不会重复相加。聚合不修改输入样本。

## 兼容性

`spec_accept_rate` 与 `spec_accept_length` 保留名称和量纲，但口径从逐样本比率的算术平均改为去重后计数求和的比率。小样本不再与大样本具有相同权重；新旧口径的历史曲线不宜直接拼接。覆盖率和记录数指标为新增。

`Sample.SpecInfo` 的四个整数累计字段继续保留，供现有调用方使用。新序列化格式还保留生成记录与 `available_counters`，因此即使兼容性整数字段默认值为 0，缺失状态也不会在 JSON 往返过程中丢失。

不含生成标识的旧载荷按整个样本计入，无法对其中的 Agentic 共享生成去重；`spec_legacy_samples` 显示这一限制。只有实际存在且值有效的字段可用。提议数或 verify 次数为正数，可以证明存在投机解码上报；全零或仅有 completion 的旧数据无法区分未上报与明确的零值，因此保持未知。部分旧载荷仍可参与分子、分母均已知的比率。新旧样本可以混在同一批次中。

## 可人工核对的示例

两次独立生成的接受数/提议数为 1/2 与 9/10 时：

```text
接受数 = 1 + 9 = 10
提议数 = 2 + 10 = 12
spec_accept_rate = 10 / 12 ≈ 0.8333
旧版逐样本平均结果 = (1/2 + 9/10) / 2 = 0.7
```

对于导出轨迹 `A -> B` 与 `A -> C`：

| 生成 | 接受数 | 提议数 | verify 次数 | completion 数 |
|---|---|---|---|---|
| A（共享） | 4 | 8 | 2 | 6 |
| B | 1 | 2 | 1 | 2 |
| C | 1 | 1 | 1 | 1 |
| 合计，A 只计一次 | 6 | 11 | 4 | 9 |

结果为 `spec_accept_rate=6/11≈0.5455`、`spec_accept_length=9/4=2.25`、`spec_nodes_total=3`、`spec_nodes_missing_counts=0`，三个覆盖率均为 1。旧实现对两条样本的比率 `5/10` 与 `5/9` 求平均，得到约 0.5278。

`tests/utils/test_spec_decoding_metrics.py` 对上述示例和边界行为做了断言。`tests/test_agentic_rollout.py` 中的 CPU 用例贯通后端元数据处理、请求提交、轨迹导出、`TrainingFieldArtifact` 序列化和批次聚合，包括部分计数的恢复请求，以及不在导出路径上的兄弟分支。

## 相关指南

- [MTP 训练](./mtp-rl-training.md)
- [Agentic Rollout](./agentic-rollout.md)
