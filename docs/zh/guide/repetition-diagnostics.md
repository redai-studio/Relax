# 重复检测与离线诊断

当 rollout 响应出现长段重复时，可以使用离线诊断定位可疑区域。工具扫描完整的 `response` 字段，包括其中保存的工具观察，能够定位开头、中间或结尾的重复。

## 快速开始

在仓库根目录运行，传入已有的训练/评估 JSONL 结果或可信的 `.pt` rollout dump：

```bash
python -m relax.entrypoints.diagnose_repetition /path/to/rollout_result/train/42.jsonl --output report.json
python -m relax.entrypoints.diagnose_repetition /path/to/rollout_data/42.pt --output report.json
```

省略 `--output` 时将 JSON 写入标准输出。JSONL 仅需 Python 标准库；`.pt` 还需要 PyTorch，两种格式均可在 CPU 上分析。仅加载可信来源的 `.pt` 文件，因为其序列化格式在加载时可以执行代码。

## 参数配置

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--window-size` | `10000` | 每个窗口的字符数，须为正整数。 |
| `--stride` | `5000` | 相邻窗口起点之间的字符数，须为正且不超过窗口大小。 |
| `--threshold` | `10.0` | 压缩比须严格大于该值才命中，须为有限正数。 |

这些参数用于本次离线报告，训练中的 `repetition_frac` 指标使用默认配置。工具逐窗扫描全文，非空的短响应使用一个窗口，并在需要时补充末窗以覆盖结尾。压缩比为 UTF-8 字节数除以 zlib 压缩级别 9 处理后的字节数。

指标计算在样本首次命中后停止扫描；离线报告始终扫描所有窗口，以收集完整诊断信息。使用默认配置时，两者的重复判定一致。与之前仅检查尾部的指标相比，现在响应任意位置的重复都可能被计入，包括过去始终判为不重复的 10,000 字符及以下响应。不要直接比较使用新旧检测器的训练记录中的 `repetition_frac`。

## 解读报告

| 字段 | 含义 |
|---|---|
| `summary.repetition_frac` | 至少命中一个窗口的样本占比。 |
| `has_repetition` | 当前样本是否存在命中窗口。 |
| `hit_windows` | 疑似重复窗口，每个包含 `start`、`end`、`compression_ratio`。 |
| `max_compression_ratio` | 全部已扫描窗口中的最大压缩比，空响应为 `0.0`。 |
| `window_count` | 当前样本实际扫描的窗口数。 |

使用 `source` 和 `sample_position` 定位原始记录：后者是从零开始的 JSONL 行号或 `.pt` 样本列表下标。报告也保留原始 rollout/样本标识及可用的数据集名称。

窗口偏移是从零开始、右边界不包含在内的 Unicode 码点索引。例如，`start=10000`、`end=20000` 对应 `response[10000:20000]`，表示字符位置，而非字节或 token 位置。命中窗口可能重叠，并不代表精确重复边界。结构化工具输出等易压缩内容也可能命中，应查看对应原文后再判断是否属于不期望的重复。

## 下一步

- [Rollout 结果可视化](./rollout-result-viewer.md) — 浏览原始 prompt 和响应。
- [调试指南](./debugging.md) — 排查训练精度问题。
