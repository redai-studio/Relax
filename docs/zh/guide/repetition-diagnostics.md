# 长响应全文重复诊断

## 概述

Relax 在完整 `Sample.response` 上检测高度可压缩的窗口，包括工具观察。因此，即使结尾正常，也能定位开头或中间的重复。压缩仅用于诊断度量：检测器不会压缩上下文记忆、修改响应或 reward，也不会停止生成。它不区分角色，也不识别语义重复。

现有 `repetition_frac` 指标等于至少命中一个窗口的样本数除以全部样本数。生产指标聚合位于 `relax/utils/metrics/rollout_metrics.py`，由分布式 rollout 模块导入。原有布尔接口仍可从 `relax.utils.metrics.metric_utils` 导入。

## 检测规则

| 设置 | 默认值与含义 |
|---|---|
| 窗口 | 10,000 个 Python 字符串字符 |
| 步长 | 5,000 个字符；不限制窗口数，不动态跳过窗口 |
| 压缩 | UTF-8 原始字节数除以 zlib 压缩后字节数，压缩等级为 9 |
| 命中 | 压缩比**严格大于 10**；等于阈值不命中 |
| 偏移 | 从零开始的 Unicode 码点，左闭右开：`[start, end)` |
| 短响应 | 长度不超过窗口的非空响应均扫描一次 |
| 空响应 | 无窗口、无命中、最大压缩比为 `null`、覆盖字符数为 0 |

偏移不是 UTF-8 字节、token 或字素簇：组合附加符号单独计数。应在任何显示转换之前，用原始 `response[start:end]` 查看命中区间。

对于 22,000 字符的响应，窗口为 `[0, 10000)`、`[5000, 15000)`、`[10000, 20000)` 和 `[12000, 22000)`。未对齐末段会补充一个末尾对齐的完整窗口；已经对齐的末窗不会重复扫描。无论响应多长，每个字符都会被覆盖。

::: warning 结果解释与兼容性
命中表示疑似重复窗口，不是精确重复边界。天然容易压缩的内容也可能命中，少量重复混入正常文本时可能仍低于阈值。与旧版仅检测末尾的实现不同，短响应及恰好 10,000 字符的响应现在也会检测，因此历史与新版 `repetition_frac` 的检测覆盖范围不同。
:::

详细诊断扫描全部窗口，最大压缩比取自**全部**窗口，包括未命中窗口。布尔接口可在确认命中后提前返回；未命中结果必须扫描全文。`covered_chars` 按命中窗口的区间并集计算，重叠部分只计一次；它不代表精确重复字符数。

## 快速开始

训练指标同样使用全文布尔扫描，不限制窗口数量。对于无重复样本，每个窗口都需要压缩；窗口和步长固定时，CPU 成本随批次内响应字符总数线性增长。每个 rollout step 承担该批次全部样本的扫描耗时。下文检测器基准没有测量整个批次的指标聚合或训练吞吐，评估训练开销时应结合实际批次大小与响应长度另行测量。

在可使用 Python 的仓库目录运行。JSONL 路径使用标准库，不需要 Ray、Megatron 或 PyTorch：

```bash
python -m relax.entrypoints.repetition tests/fixtures/repetition/middle_repetition.jsonl --output /tmp/repetition-report.json
```

固定样例包含中间重复、末尾正常的响应。对于既有 dump，可传入一个或多个文件或目录；目录递归搜索，文件路径会去重：

```bash
python -m relax.entrypoints.repetition /data/rollout-results --output /tmp/repetition-report.json
python -m relax.entrypoints.repetition /data/debug.pt --input-format torch --output /tmp/debug-report.json
```

## 配置

| CLI 参数 | 含义 |
|---|---|
| `inputs` | 一个或多个 dump 文件或目录 |
| `--output` | 必填，JSON 报告目标路径 |
| `--window-size` | 正整数，默认 10,000 |
| `--stride` | 不大于窗口的正整数，默认 5,000 |
| `--threshold` | 有限正数，默认 10.0 |
| `--input-format` | `auto`（默认）、`jsonl` 或 `torch` |
| `--trusted-torch` | 允许可信旧版 Torch dump 中的 pickle 对象 |

这些选项只配置本次离线执行，不是训练参数。自动识别 `.jsonl`、`.pt` 和 `.pth`；其他扩展名文件需显式指定格式。目录发现仅包含对应的已知扩展名。

## Dump 格式与报告

JSONL 每个非空行是一个样本对象，必须包含字符串 `response`，兼容 `save_rollout_result_jsonl` 和 `save_eval_summary_jsonl` 写出的训练及评估摘要。Torch 输入为 `save_debug_rollout_data` 生成的字典，含 `samples` 及可选的 `rollout_id`；仅读取该格式时导入 PyTorch，张量加载到 CPU。

报告保留 `source`、从零开始的 `record_index`、从一开始的 JSONL `line_number`，以及已有的 `rollout_id`、`sample_index`、`index`、`group_index` 和 `dataset`。缺失标识为 `null`。`sample_index` 和 `index` 保留各自原有含义，不相互替代。对于缺失或重复的 ID，使用来源文件加记录位置区分样本。

每个样本包含 `has_repetition`、`response_chars`、`scanned_windows`、`hits`（含 `start`、`end`、`compression_ratio`）、`max_compression_ratio` 和 `covered_chars`。报告不复制响应原文。顶层记录 `schema_version`、偏移约定、检测配置、来源、样本及汇总；汇总包含样本计数、`repetition_frac`、响应及覆盖字符统计、扫描窗口数、全局最大压缩比和耗时。命中摘要也会写入日志。

JSON 损坏、记录不是对象、响应缺失或不是字符串时，会带来源位置报错，不会静默丢弃。报告流式写入临时文件，成功后原子替换；失败保留已有报告。禁止用报告覆盖显式输入文件。JSONL 内存随单条记录及其命中数增长；Torch 反序列化会加载整个 dump。

在 POSIX 系统中，新报告仅允许所有者访问（`0600`）。替换已有报告时保留其读、写、执行权限；所有者、ACL 和扩展属性不会复制。需要共享访问时，应明确设置报告权限，后续更新会保留这些权限。生成过程中的临时文件保持仅所有者可访问。Windows 访问权限由文件系统 ACL 管理。

::: warning 旧版 Torch dump
默认使用受限的 `torch.load(..., weights_only=True)`。若自行生成的可信旧版 dump 含自定义 pickle 对象，可显式添加 `--trusted-torch`。该选项允许任意 pickle 代码执行，不可对不可信文件启用。受限加载失败时不会自动回退到不安全模式。
:::

## API 参考

```python
from relax.utils.repetition import RepetitionConfig, analyze_repetition, has_repetition
from relax.utils.rollout_dump import iter_rollout_records

config = RepetitionConfig(window_size=10_000, stride=5_000, threshold=10.0)
for record in iter_rollout_records("tests/fixtures/repetition/middle_repetition.jsonl"):
    detected = has_repetition(record.response, config)
    result = analyze_repetition(record.response, config)
    report_record = {**record.identity(), **result.to_dict()}
    assert detected == result.has_repetition
```

`analyze_repetition` 返回不可变的 `RepetitionResult`，`hits` 中每项为 `RepetitionWindow`。`iter_repetition_windows(length, config)` 提供相同的窗口覆盖策略，不执行压缩。`relax.entrypoints.repetition` 中的 `diagnose_dumps(inputs, output, *, config=RepetitionConfig(), input_format="auto", trusted_torch=False)` 写出完整报告并返回汇总。

## 验证与基准

```bash
python -m pytest tests/utils/test_repetition.py tests/utils/test_repetition_dump.py tests/utils/test_repetition_metrics.py
python -m relax.tools.benchmark_repetition --output benchmarks/repetition/local.json --repeats 7
```

测试覆盖开头、中间、结尾重复，固定的中间重复/结尾正常样例，空串与短响应、整窗、重叠边界、未对齐末窗、中文偏移、无重复对照、阈值相等和区间并集计数。集成测试调用生产样本指标聚合及真实 dump 写入/读取接口，包含 CLI。可选的分布式日志测试需要训练镜像中的 Ray/SGLang/TransferQueue/Megatron 环境，缺失时明确跳过；CPU 集成测试不代表完成 GPU 或多节点验证。

已提交的测量记录位于 `benchmarks/repetition/windows-python313.json`。下表摘录**详细接口、无重复对照**的数据，环境为 Windows 11、Python 3.13.9、12 个逻辑 CPU、zlib runtime 1.3.1，每项执行七个计时批次：

| 字符数 | 窗口数 | CPU 中位耗时 ms | 墙钟中位耗时 ms | 进程峰值 MiB | 扫描 tracemalloc 峰值 KiB |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1 | 0.053 | 0.068 | 24.06 | 304.14 |
| 100,000 | 19 | 1.645 | 1.690 | 25.69 | 314.23 |
| 1,000,000 | 199 | 18.229 | 18.701 | 37.19 | 314.23 |

每个长度/样例/接口组合在独立子进程运行。计时排除导入、输入构造和 tracemalloc，并批量执行完整调用以降低 CPU 时钟量化影响。原生进程峰值包含解释器、输入构造、预热及 zlib 原生内存，不是扫描新增内存。另行执行的 tracemalloc 测量只包含追踪器可见的分配。完整报告还包含布尔接口、全文重复、中间重复、中文中间重复、原始计时及 p95。详细诊断始终遍历所有窗口。10,000 字符的混合中间重复样例只有一个窗口，重复被正常文本稀释后可能低于阈值。这些是检测器基准，不是 dump I/O 或端到端训练吞吐基准。

## 故障排除

- **存在少量重复却未命中：** 查看最大压缩比及原始窗口；压缩比阈值不保证识别所有重复片段。
- **配置无效：** 要求 `0 < stride <= window_size`，阈值为有限正数。
- **格式未知或记录损坏：** 必要时显式选择格式并修复报错位置的源记录，不要通过丢弃无效样本生成报告。
- **Torch 输入占用内存较高：** 序列化格式要求加载整个 dump；JSONL 支持逐条流式处理。

## 下一步

- [Rollout 结果可视化](./rollout-result-viewer.md)
- [调试指南](./debugging.md)
- [指标服务](./metrics-service-detailed.md)
