# 重复检测

长响应有时会退化成重复文本。Relax 在线通过 `rollout/repetition_frac` 指标报告这种情况，
并提供离线入口，定位重复出现在响应的**哪个位置**。

## 检测对象

检测作用于 `Sample.response`，即完整的响应文本；对 agentic rollout 来说，其中可能包含工具观察。
判定方法是压缩比：一段文本如果被 `zlib` 压缩得远好于正常文本，就会被标为低熵的疑似重复文本；结构化工具输出也可能命中。

扫描使用重叠滑窗覆盖**全文**：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| 窗口大小 | 10,000 字符 | 每个扫描窗口的长度 |
| 步长 | 5,000 字符 | 相邻窗口起点的间隔 |
| 阈值 | 10.0 | 压缩比**严格大于**该值时判定命中 |

`stride` 不能大于窗口大小：更大的步长会在窗口之间留下未扫描的空隙，
落在空隙里的重复会被漏掉且没有任何提示。传入这种参数会抛出 `ValueError`。
阈值必须是有限正数；即使输入为空，也会拒绝 NaN、无穷大、零和负数。

短于一个窗口的非空响应按单个窗口扫描。空响应没有窗口，不会被判为重复，
且 `max_compression_ratio=None`（JSON 中为 `null`）。当最后一个对齐步长的窗口没有覆盖到文本末尾时，
会额外扫描正好末尾 10,000 字符的窗口，保证末段不被遗漏。所有窗口都会被扫描：
不会通过静默跳过中间窗口来降低成本。

::: tip 为什么用重叠滑窗
重叠滑窗可以减少窗口边界造成的漏检，但不能保证任意一个窗口长度的重复片段都会命中。
例如重复片段位于 `[2500, 12500)` 时，相邻窗口均混入正常文本，压缩比可能低于阈值。
默认参数下，连续重复区间至少长 15,000 字符才能保证其中包含一个完整扫描窗口；是否命中仍取决于压缩比。
之前的实现只看末尾 10,000 字符，因此「中间重复、结尾正常」的响应完全检测不到。
:::

## 偏移与区间

报告中的区间是对 `Sample.response` 的左闭右开切片 `[start, end)`，以**字符**
（Unicode 码点，即 `len(text)`）计数，而不是字节。`response[start:end]` 能精确还原被扫描的窗口，
这样偏移量对中文等非 ASCII 文本同样有意义。

区间表示**疑似重复的窗口**，不表示重复的精确边界 —— 实际重复片段可能短于命中它的窗口，也可能超出该窗口。

报告统计覆盖字符数时，按命中区间的**并集**计算，重叠部分不会被重复计数。

## 在线指标

`rollout/repetition_frac` 表示一个 rollout 批次中，响应任意位置存在重复的样本占比。
聚合仍然是样本占比，但检测范围扩展到了全文，并且会扫描短响应和恰好一个窗口长的响应。
旧版只检查后缀，并跳过不超过 10,000 字符的响应，因此新旧指标不可直接比较；
对比时应使用同一实现重新扫描相同 dump。

布尔接口保持不变：

```python
from relax.utils.metrics.metric_utils import has_repetition

has_repetition(sample.response)  # -> bool
```

需要详细结果时使用 `scan_repetition`：

```python
from relax.utils.metrics.metric_utils import scan_repetition

report = scan_repetition(sample.response)
report.has_repetition        # 是否命中
report.hit_intervals         # [(start, end), ...] 疑似重复窗口
report.max_compression_ratio # 最大压缩比；未扫描任何窗口时为 None
report.covered_chars         # 命中区间并集的长度
report.to_dict()             # 可直接序列化为 JSON 的摘要
```

`has_repetition` 在首次命中后短路返回；`scan_repetition` 默认扫描全部窗口，保证离线报告完整。

以上两个名字也可从 `relax.utils.repetition` 直接导入。两条路径是同一份实现，
但 `relax.utils.repetition` 只依赖标准库，适合离线脚本；
`relax.utils.metrics.metric_utils` 会连带引入 `torch` 等训练依赖。

## 离线诊断

诊断入口支持两种 dump 格式，可混合传入多个文件：

| 格式 | 产出方式 | 说明 |
| --- | --- | --- |
| `.jsonl` | 设置 `--rollout-result-dir` 后**每步常开** | 逐行流式读取，不依赖 `torch`，通常首选 |
| `.pt` | 需显式开启 `--save-debug-rollout-data` | pickle 格式，仅读取可信来源，`torch` 按需惰性导入 |

```bash
# 常开的 JSONL 结果，无需任何训练依赖即可诊断
python -m relax.entrypoints.diagnose_repetition /path/result/train/0.jsonl -o repetition.json

# 也可诊断 .pt dump，或一次传入多个文件
python relax/entrypoints/train.py ... --save-debug-rollout-data /path/dump/{rollout_id}.pt
python -m relax.entrypoints.diagnose_repetition /path/dump/*.pt --only-hits
```

检测核心位于 `relax.utils.repetition`，只依赖标准库；因此在没有 `torch`、
没有 Ray 与指标服务的纯 CPU 机器上，也能直接对 JSONL 结果做离线诊断。

可选参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `-o, --output` | stdout | 将 JSON 报告写入该路径 |
| `--window-size` | 10000 | 窗口大小（字符） |
| `--stride` | 5000 | 步长（字符） |
| `--threshold` | 10.0 | 压缩比阈值 |
| `--only-hits` | 关闭 | 只报告命中的样本 |

`--output` 不允许指向任何一个输入 dump：dump 是无法重新生成的训练产物，
覆盖写入会直接销毁正在被诊断的数据。

报告用样本在 dump 中的位置，以及 `dataset`、`index`、`sample_index`、`group_index`、`rollout_id`
标识样本；在相同样本和阈值下，其 `repetition_frac` 与在线指标一致。
每个样本必须具有字符串类型的 `response`；缺失、`null` 或其他类型会报错，
并指出文件行号或样本位置，不会静默拉低占比。空字符串合法，计为无重复样本。报告使用严格 JSON 序列化。

```json
{
  "config": {
    "window_size": 10000,
    "stride": 5000,
    "threshold": 10.0,
    "offset_unit": "characters",
    "interval_convention": "half-open [start, end)"
  },
  "num_samples": 2,
  "num_repetitive_samples": 1,
  "repetition_frac": 0.5,
  "files": [
    {
      "path": "/path/dump/7.pt",
      "rollout_id": 7,
      "num_samples": 2,
      "num_repetitive_samples": 1,
      "repetition_frac": 0.5,
      "samples": [
        {
          "position": 1,
          "index": 1,
          "group_index": 0,
          "has_repetition": true,
          "text_length": 45600,
          "threshold": 10.0,
          "max_compression_ratio": 147.05882352941177,
          "num_windows_scanned": 9,
          "hit_windows": [
            { "start": 15000, "end": 25000, "compression_ratio": 147.05882352941177 },
            { "start": 20000, "end": 30000, "compression_ratio": 147.05882352941177 },
            { "start": 25000, "end": 35000, "compression_ratio": 11.481056257175661 }
          ],
          "covered_chars": 20000
        }
      ]
    }
  ]
}
```

三个命中窗口互相重叠，共同覆盖 15,000-35,000 字符，因此 `covered_chars` 是它们的
并集（20,000），而不是各自长度之和（30,000）。

## 开销

完整扫描耗时与响应长度成线性关系。窗口边界按需生成，两个入口共用同一趟扫描：
`has_repetition` 委托 `scan_repetition` 并启用提前退出，在首次命中处停止，
且不为每个窗口保留数据；排除输入文本后，其辅助内存为 `O(window_size)`。
详细报告额外保存每个命中窗口一条记录，因此 `scan_repetition` 的辅助内存为
`O(window_size + H)`，其中 `H` 为命中窗口数——纯净文本下 `H` 为 0，与响应长度无关。

以下数据在 macOS 27.0 arm64、Python 3.14.7 下于 2026-09-18 实测，计时重复 5 次。
单响应均运行完整 `scan_repetition`，包括全重复样例，未启用短路。

| 响应长度 | 样例 | 窗口数 | 平均墙钟 ms | 平均 CPU ms | 跟踪峰值 MiB | 进程峰值 RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 无重复 | 1 | 0.09 | 0.08 | 0.30 | 25.59 |
| 10,000 | 中间重复 | 1 | 0.02 | 0.02 | 0.30 | 25.89 |
| 10,000 | 全重复 | 1 | 0.02 | 0.02 | 0.30 | 25.77 |
| 100,000 | 无重复 | 19 | 2.13 | 2.13 | 0.31 | 25.66 |
| 100,000 | 中间重复 | 19 | 1.89 | 1.80 | 0.31 | 25.95 |
| 100,000 | 全重复 | 19 | 0.26 | 0.26 | 0.31 | 25.52 |
| 1,000,000 | 无重复 | 199 | 24.08 | 23.88 | 0.31 | 28.41 |
| 1,000,000 | 中间重复 | 199 | 24.23 | 24.11 | 0.31 | 30.30 |
| 1,000,000 | 全重复 | 199 | 2.84 | 2.83 | 0.34 | 26.56 |

无重复输入是确定性的 SHA-256 十六进制填充文本。中间重复样例将居中的 `min(length, 20000)`
字符替换为重复的 `spam`；因此 1 万字符时该样例等同全重复样例。全重复样例仅包含重复的 `spam`。
墙钟时间使用 `perf_counter`，CPU 时间使用 `process_time`，均排除输入构造和预热。

内存单独测量：`tracemalloc` 在输入构造和预热后启动，仅计增量跟踪分配，不包含输入存储、
导入和 Python 未跟踪的原生分配。每个样例的峰值 RSS 在独立新进程中使用 `resource.ru_maxrss`
采集，包含解释器、导入、输入构造和完整扫描；macOS 的字节值和 Linux 的 KiB 值均换算为 MiB。
RSS 是进程生命周期峰值，不是扫描本身的增量内存。这里只实测了 macOS，未在 Linux 上复跑。

批次基准只测在线 `has_repetition` 判定及均值聚合，不包含其他 rollout 指标、Ray 或 dump I/O。
每个批次复用同一不可变响应。重复样例在指定位置用 `"spam" * 5000` 替换 20,000 字符，
其余部分仍为无重复填充文本。下表为平均墙钟时间：

| 批大小 | 响应长度 | 无重复 | 开头重复 | 中间重复 | 末尾重复 |
| --- | --- | --- | --- | --- | --- |
| 512 | 40,000 | 329.72 ms | 7.18 ms | 52.80 ms | 136.36 ms |
| 1024 | 40,000 | 710.88 ms | 14.76 ms | 120.31 ms | 310.84 ms |
| 512 | 200,000 | 2369.85 ms | 7.36 ms | 1030.33 ms | 2147.32 ms |

短路节省的时间取决于首次命中的位置；末尾才命中时仍需扫描前面的正常窗口。
绝对耗时和 RSS 会随机器及系统负载变化，不能用这些数字替代训练环境测量。

在仓库根目录运行，无需训练依赖（RSS 测量需要 macOS 或 Linux）：

```bash
PYTHONPATH=. python3 tests/utils/benchmark_repetition_scan.py --batches --json
```

输出包含环境、重复次数、单响应 CPU / 墙钟耗时、跟踪内存峰值、独立进程峰值 RSS，
以及批次 CPU / 墙钟耗时。`wall_mean_ms` 表示墙钟均值，`peak_mem_mib` 表示跟踪分配峰值。
省略 `--batches` 时只运行 1 万、10 万、100 万字符的三类单响应样例。

CPU 集成测试加载完整 rollout 模块并调用真实 `compute_metrics_from_samples`，
仅替换部署依赖，使其不再因缺少 SGLang 而跳过。测试覆盖训练与评估指标、真实 Sample、
dump 写入和读取；它不启动 Ray 或 GPU，不能替代多节点训练验证。

## 限制

- 检测基于统计特征而非语义：只能发现低熵的重复文本，无法识别改写或语义上冗余的内容。
- 不区分角色。如果响应中嵌入了工具观察，重复的观察同样会被计为重复。
- 不截断生成、不改变 reward —— 这只是一个诊断能力。

## 下一步

- [调试指南](./debugging.md)
- [配置](./configuration.md)
