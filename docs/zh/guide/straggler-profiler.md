# 慢节点分析器

慢节点分析器（Straggler Profiler）观测各数据并行 rank 上的 Megatron 训练计时器区间，并在某个 rank 相对等价对等 rank 变慢时给出报告。

## 慢节点分析器是什么

分析器会挂载一个可直接替换 Megatron `config.timers` 的对象，记录 Megatron 发出的每一个计时器区间（`forward-backward`、`forward-compute`、反向阶段、优化器阶段以及收发配对）。执行相同程序的 rank 会在同一个时间窗口内相互比较；若某个 rank 明显慢于其最快的等价对等 rank，就会连同判定所依据的实测事实一起被报告。

它是纯观测组件：不会做任何缓解、重排、中止 step 或改变训练的操作。它不会向训练路径加入集合通信或同步；每个入口都是 fail-open 的——分析器自身的失败只会被计数并退化为空操作，而不会传播到 Megatron 的调度中。

## 如何启用

分析器**默认关闭**。通过主开关启用：

```bash
export RELAX_STRAGGLER_ENABLE=1
```

只在单个进程内启用时，观测是 rank 本地的：该进程不与任何对象比较，rank 本地运行只会报告 `uncertain`。若要在 rank 之间比较，还需让每个进程指向 rank 0 的 collector socket：

```bash
export RELAX_STRAGGLER_ENABLE=1
export RELAX_STRAGGLER_COLLECTOR_ADDR=127.0.0.1:29741
export RELAX_STRAGGLER_OUTPUT_DIR=/path/to/straggler-evidence
```

关闭路径经过设计与测试验证：不新增线程、socket、CUDA event，也不产生额外工作。开关关闭时，运行时根本不会被构建，`get_straggler_timers()` 返回 `None`，训练后端保持 Megatron 的 `config.timers = None`，因此默认部署维持上游行为。开关接受 `1`、`t`、`true`、`y`、`yes`、`on`（以及对应的假值）；无法识别的取值会在启动时被拒绝，而不会悄悄解析为关闭。

## 配置

每个开关都是一个在进程启动时读取的环境变量。取值每进程解析一次；无法解析或低于最小值的取值会回退到默认值，且该调整会以 `clamped` 形式记录在启动日志中。

| 变量                                  | 默认值     | 含义                                                                                                                                                                                   |
| ------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RELAX_STRAGGLER_ENABLE`              | `False`    | 主开关。只有取值为真时分析器才生效。                                                                                                                                                   |
| `RELAX_STRAGGLER_WINDOW_S`            | `5.0`      | 一个受限**时间**窗口的秒数。窗口以该 cohort 的首次观测为锚点、按墙钟时间推进；它**不**与优化器 step 对齐，因此窗口本身没有 step 顺序。最小 `0.1`。                                     |
| `RELAX_STRAGGLER_REPORT_INTERVAL_S`   | `10.0`     | collector 周期性摘要日志行与状态文件写入线程的节奏。最小 `0.1`。                                                                                                                       |
| `RELAX_STRAGGLER_OUTPUT_DIR`          | 未设置     | 各 rank 的 JSONL 数据流与状态 JSON 文件所在目录。未设置时数据只保留在内存中（仍会被计数与汇总）。                                                                                      |
| `RELAX_STRAGGLER_COLLECTOR_ADDR`      | 未设置     | 同机 collector socket 的 `host:port`。未设置时每个进程各自保留一个 collector，观测是 rank 本地的。参考 observer 配方会导出 `127.0.0.1:29741`；该端口是配方自身的选择，并非内置默认值。 |
| `RELAX_STRAGGLER_WARMUP_WINDOWS`      | `2`        | 运行开始时永不被判定的窗口数。最初的区间包含惰性 CUDA 上下文/event 分配和首批数据，看起来就像慢 rank。最小 `0`，最大 `10`。                                                            |
| `RELAX_STRAGGLER_MIN_COHORT`          | `2`        | 一个窗口内每个 `(cohort, stage)` 对所需的最少上报 rank 数。更小的 cohort 会报告为 `uncertain` 而不做判定。最小 `2`。                                                                   |
| `RELAX_STRAGGLER_TOPOLOGY_EPOCH`      | `""`（空） | cohort key 中标识一种并行布局。它是**仅保留在 schema 中**的字段：没有任何逻辑会推导或更新它，因此除非运维人员手工设置，重新分片对 cohort key 是不可见的。                              |
| `RELAX_STRAGGLER_TIMER_LOG_LEVEL`     | `2`        | 捕获的最高 Megatron 计时器级别。Megatron 用 1 表示粗粒度阶段、2 表示细粒度阶段；默认同时捕获两者。范围 1-2。                                                                           |
| `RELAX_STRAGGLER_EVENT_POOL`          | `512`      | 每个 rank 预分配的 CUDA event 数。一个区间需要两个，因此池耗尽会将该区间降级为仅主机计时。最小 `2`。                                                                                   |
| `RELAX_STRAGGLER_QUEUE_MAX`           | `4096`     | observer 线程与 sender 之间每 rank envelope 队列的上界。最小 `1`。                                                                                                                     |
| `RELAX_STRAGGLER_WORK_TOLERANCE`      | `0.05`     | 判定某个 rank 之前用于时间偏差与工作量可比性检验的相对容差。最小 `0.0`。                                                                                                               |
| `RELAX_STRAGGLER_MIN_STAGE_MS`        | `5.0`      | 以毫秒为单位的绝对量级下限。若某 stage 的实测值或对等最快值，或其绝对差值不超过该下限，则归类为 `uncertain` 而非 `straggler`。最小 `0.0`。                                             |
| `RELAX_STRAGGLER_PERSIST_WINDOWS`     | `3`        | 某 rank 被报告为慢节点之前所需的连续异常窗口数。最小 `1`。                                                                                                                             |
| `RELAX_STRAGGLER_DEBUG_HOST_DELAY_MS` | `0.0`      | **仅调试用**注入：在实测区间内睡眠指定毫秒数。取 `0.0` 时无效。                                                                                                                        |
| `RELAX_STRAGGLER_DEBUG_RANK`          | `-1`       | **仅调试用**注入的目标 rank；`-1` 表示所有 rank。                                                                                                                                      |
| `RELAX_STRAGGLER_DEBUG_STAGE`         | `""`（空） | **仅调试用**注入的目标计时器名；空表示所有 stage。                                                                                                                                     |

三个 `RELAX_STRAGGLER_DEBUG_*` 开关用于通过受控的变慢来测量检测器灵敏度。该延迟落在实测区间内部，与真实的慢 stage 完全一致，且除睡眠之外不改变任何行为。它们不属于正常部署配置。

## 架构

数据路径包含四个阶段，与训练 step 分离、带外运行：

1. **计时器 shim**（`StragglerTimers`）替换 `config.timers`。`start`/`stop` 记录一个主机时间戳，并在 CUDA 可用时在当前流上记录一对 CUDA event。`start`/`stop` 从不调用 `torch.cuda.synchronize()`，也从不调用 `torch.distributed.barrier()`。
2. **进程内 observer**（`StragglerObserver`）持有一个固定大小的 CUDA event 池、一个受限的待处理列表和一个守护读出线程。它在训练线程之外读回 event，将每个区间分类为 `device` 或 `host_only`，并构建携带 rank 身份与可选工作量计数的 `TimingEnvelope`。
3. **rank 0 带外 collector**（`TimingCollector` 以及 `EnvelopeSender`/`EnvelopeReceiver`）通过同机 TCP socket 接收其他 rank 的 envelope。rank 0 绑定该 socket 并判定每个 rank 的 envelope；其他 rank 只发送、不做判定。不存在独立的 collector 进程，也没有跨节点传输。
4. **检测器与上报器**（`StragglerDetector`、`report_once`）。检测器在 cohort 内比较各 rank 并产生判定；上报器把运行时计数与已取出的判定转换为平台指标，并合并进既有 perf 路径（`relax/utils/training/train_metric_utils.py`）。

这里**没有新增训练路径集合通信**，也**没有新增 HTTP 请求**。计时器 shim 不加入集合通信，observer 在训练线程之外读回 event，传输是由后台守护线程持有的同机 socket，上报器只读取进程内计数。流水线中的每个失败都会被计数，而不是抛出。

## 指标

上报器在 `perf/straggler/` 前缀下输出扁平的标量 key。无法测量的取值会被省略而不是上报为零，因此缺失的 key 读作“未测量”，而永远不会被误读为“未观测到异常”。

| Key                                            | 含义                                                                                     |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `perf/straggler/active_stragglers`             | 当前被标记的 `(cohort, stage, rank)` 三元组数量。                                        |
| `perf/straggler/verdicts`                      | 截至目前产生的判定数。                                                                   |
| `perf/straggler/windows_closed`                | 截至目前关闭并判定的时间窗口数。                                                         |
| `perf/straggler/coverage`                      | cohort 覆盖率，仅在 summary 发布显式覆盖率时输出。                                       |
| `perf/straggler/judged_fraction`               | collector 已判定的 envelope 占已接收 envelope 的比例。这是传输比例，不是 cohort 覆盖率。 |
| `perf/straggler/dropped`                       | 运行时上报的所有丢弃计数之和（队列满、待处理满、输出满、检测器受限结构、畸形输入）。     |
| `perf/straggler/worst_deviation`               | 已取出判定中最大实测变慢的相对偏差。                                                     |
| `perf/straggler/worst_rank`                    | 该最大实测变慢对应的 rank。                                                              |
| `perf/straggler/rollout_id`                    | 来自已发布训练上下文的 rollout id。                                                      |
| `perf/straggler/optimizer_step`                | 来自已发布训练上下文的 rollout 内优化器 step 下标。                                      |
| `perf/straggler/step_ordinal`                  | 分析器分配的单调递增运行 step 序号。                                                     |
| `perf/straggler/workload_incomparable_windows` | 至少有一个 rank 的 token 工作量超出容差的窗口数。                                        |
| `perf/straggler/workload_missing_windows`      | 至少有一个 rank 未发布工作量的窗口数。                                                   |
| `perf/straggler/workload_publish_skipped`      | 被调用方自一致性保护拦下的工作量发布次数。                                               |
| `perf/straggler/workload_publish_errors`       | 抛异常的工作量发布尝试次数。                                                             |
| `perf/straggler/collector_status_available`    | 当前进程拥有 collector 时为 `1`，不拥有时为 `0`。                                        |

`collector_status_available` 在流水线并行下尤为重要：平台只在 Megatron 主 rank（`tp0`、pipeline 最后一个 stage、`dp0`）上合并指标，而该 rank 仅在 `pp_size == 1` 时拥有 collector。当 `pp_size > 1` 时，负责导出的 rank 不拥有 collector，显式的 `0` 就是“此处无慢节点计数”的标记，而不是静默省略。

设置 `RELAX_STRAGGLER_OUTPUT_DIR` 后，collector 还会在 `<output_dir>/run_<run_id>/` 下持久化证据：

| 文件                                                     | 写入者                                                 | 内容                                                            |
| -------------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| `straggler_envelopes.jsonl`                              | collector                                              | 每行一个已判定的计时区间。                                      |
| `straggler_verdicts.jsonl`                               | collector                                              | 每行一个判定。                                                  |
| `collector_status.json`、`collector_status_<label>.json` | rank 0                                                 | collector 摘要，包含去重计数与检测器计数。                      |
| `runtime_status.json`、`runtime_status_<label>.json`     | 无后缀文件由 rank 0 写入，各 rank 写入自己的带后缀文件 | shim、observer、sender/receiver 与 collector 计数的逐进程快照。 |

状态文件由后台状态写入线程原子写入（唯一临时文件，随后 rename），因此未优雅退出即被 kill 的进程，其状态文件最多滞后一个写入间隔。

## 判定语义

一个判定描述的是**某个 rank 在某个 stage（某个 Megatron 计时器名）的某个时间窗口内**的情况。它把实测事实与推断原因保存在不同字段中：

- `facts` 包含一切被观测或计数到的内容：rank 与 cohort、stage、窗口下标、该 rank 与各对等 rank 的样本数、实测与对等最快与对等中位的主机中位数、比值、绝对差值、容差、持续性、cohort 大小与覆盖率、device 中位数及其可用性，以及工作量差值。
- `candidate_causes` 是判定对*原因*所做的唯一前瞻性陈述。它从 `("undetermined",)` 开始，只有当事实足以支撑时才被替换。当主机区间增长而 GPU 时间线区间没有增长时，它为 `host_side_delay_possible`；当两者都增长时，它为 `device_or_stream_visible_delay_possible`。stage 名称永远不会参与原因判断。

`reason` 字段是**纯粹的测量分类**，绝不是原因：

- `gpu_stream_stall` —— 主机区间与 GPU 时间线区间都增长，因此多出的时间位于 GPU 时间线之内。
- `host_only_stall` —— 主机区间增长而 GPU 时间线区间没有增长，因此多出的时间明显在流之外（主机工作或等待）。
- `attribution_unknown` —— device 区间不可用（没有 CUDA event），无法做主机/设备切分。
- `within_tolerance`、`cohort_below_min_size`、`below_absolute_floor` 与 `workload_incomparable` 是其余测量分类。

这一区分并不能证明硬件变慢。一对 CUDA event 测量的是 **GPU 流上的流逝时间**，那是一条 GPU 时间线，而不是对 GPU 忙碌时间的陈述：让流空转的主机停顿与慢 kernel 无法区分。要把慢 kernel 与空转的流区分开，需要逐 kernel 数据，而本分析器并不采集这些数据，因此 `gpu_stream_stall` 只报告可观测的差距，到此为止。

判定的 `kind` 为 `straggler`、`recovered` 或 `uncertain`。只有当某 rank 连续 `RELAX_STRAGGLER_PERSIST_WINDOWS` 个窗口的偏差都超过 `RELAX_STRAGGLER_WORK_TOLERANCE` 时，才会被报告为慢节点。参照对象是窗口内**最快**的对等 rank，而不是均值，因此单个慢 rank 无法把基线拉向自己。比较只在仅数据并行维度不同的 rank 所构成的 cohort 内进行。

## 它不代表什么

- **它不是根因结论。** 判定报告实测差距以及事实所能支撑的原因。它不采集逐 kernel 时长或点对点传输时间，因此从不断言网络、故障或“GPU 慢”之类的原因。
- **它从不做缓解。** 分析器不会重排、中止、拖慢或以其他方式改变训练。`straggler` 判定是给运维人员的信息，而不是一个动作。
- **attention 与 MoE 分组仅保留在 schema 中。** 它们作为能力声明在 stage 分类体系中，但并未被测量：所固定的 Megatron 版本不发出 attention 或 MoE 计时器名，因此若不使用深度 hook，就无法从 `forward-compute` 中单独拆分出 attention/MoE 开销，而深度 hook 被有意排除在本阶段之外。
- **比较需要等价 cohort。** 只有当两个 rank 运行相同程序且仅数据并行维度不同（TP/PP/VPP/CP/EP/ETP 位置相同、模型 chunk 相同、topology epoch 与 stage schema 相同）时才会被比较。绝不会把某个 rank 与合法地做不同工作的对等 rank 相比；小于 `RELAX_STRAGGLER_MIN_COHORT` 的 cohort 会给出 `uncertain`。
- **绝对下限可能掩盖真实的变慢。** 差距必须同时超过相对容差与绝对下限（`RELAX_STRAGGLER_MIN_STAGE_MS`）。在下限以下，相对偏差由主机/启动抖动主导，因此非常短的 stage 上很大的*相对*变慢会被归类为 `uncertain`（`below_absolute_floor`）而非 `straggler`，从而可能被漏掉。

## 已知限制

- **重启约定。** 运行时是首次使用时构建的进程级单例，在一个进程内不会被重建；observer 的 `disabled` 状态按设计是终态。不存在用于生产环境的重启或重置钩子（重置辅助函数仅供测试使用），因此启动失败或自行禁用的分析器会一直保持关闭，直到进程重启。
- **迟到的数据包会丢失。** 当来自 `W + 1 + WINDOW_GRACE` 的 envelope 到达时，检测器关闭窗口 `W`，其中 `WINDOW_GRACE` 为一个窗口。延迟超过该宽限的数据包无法被放入其所属窗口，会作为迟到的传输噪声被丢弃而不参与判定。
- **工作量标记可能过期。** observer 在*构建* envelope 时（即在区间关闭之后、在读出线程上）才查找工作量。因此所标记的工作量可能属于比该区间更晚的 rollout 或优化器 step；envelope 唯一的时间锚点是其主机开始时间，工作量只是建议性的传输元数据。
- **topology epoch 是惰性的。** `RELAX_STRAGGLER_TOPOLOGY_EPOCH` 是 cohort key 的一部分，但没有任何逻辑会推导或更新它。除非运维人员手工设置该变量，重新分片对 cohort key 是不可见的。
- **无 CUDA 路径在训练线程上判定。** 没有 CUDA event 时，observer 没有可轮询读回的 event，因此会把每个已完成的区间直接交给 collector，检测器便在训练线程上运行。文件 I/O 与网络传输仍在各自线程上，但在该模式下判定本身并非在训练线程之外完成。
- **没有 rollout 或 topology 重置。** 跨越 rollout 边界的窗口会被排入下一个 rollout 的 perf 日志。覆盖率缺口会被计数（`incomplete_windows`）而不是靠猜测；只包含多 rank cohort 中单个 rank 的窗口不会产生判定。

## 输出示例

以下片段是**示意性的**：它们展示判定与 collector 摘要的结构，测量值以 `"..."` 省略。它们不是任何特定运行的输出。

一个判定，写入 `straggler_verdicts.jsonl` 的形式如下：

```json
{
  "kind": "straggler",
  "cohort": "topo0:dense:...",
  "name": "forward-compute",
  "rank": 3,
  "label": "rank3/tp0/pp0/.../dp3/chunk-1",
  "window_index": "...",
  "deviation": "...",
  "consecutive_windows": "...",
  "rank_host_ms": "...",
  "reference_host_ms": "...",
  "rank_device_ms": "...",
  "reference_device_ms": "...",
  "host_only": false,
  "cohort_size": "...",
  "reason": "gpu_stream_stall",
  "measurement_kind": "device_and_host",
  "candidate_causes": ["device_or_stream_visible_delay_possible"],
  "facts": {
    "stage": "forward-compute",
    "observed_ms": "...",
    "peer_fastest_ms": "...",
    "peer_median_ms": "...",
    "ratio": "...",
    "absolute_delta_ms": "...",
    "cohort_size": "...",
    "cohort_expected": "...",
    "coverage_ratio": "...",
    "device_available": true,
    "device_ms": "...",
    "peer_device_ms": "...",
    "workload_delta": "...",
    "workload_evidence_degraded": false
  }
}
```

一个 collector 摘要，写入 `collector_status.json` 的形式如下：

```json
{
  "envelopes": "...",
  "judged_packets": "...",
  "invalid_packets": "...",
  "duplicate_packets": "...",
  "late_packets": "...",
  "windows_closed": "...",
  "stragglers_reported": "...",
  "recoveries_reported": "...",
  "uncertain_judgements": "...",
  "incomplete_windows": "...",
  "warmup_windows_skipped": "...",
  "workload_incomparable_windows": "...",
  "workload_missing_windows": "...",
  "active_stragglers": [
    { "cohort": "topo0:dense:...", "name": "forward-compute", "rank": 3, "label": "..." }
  ],
  "dedup": { "new": "...", "duplicate": "...", "late": "...", "expired": "...", "evicted": "..." },
  "persist_windows": 3,
  "work_tolerance": 0.05,
  "min_stage_ms": 5.0
}
```

## 性能说明

开销通过 **AB/BA 配对实验**测量：同一配方在分析器关闭（A 臂）与开启（B 臂）下运行，并在多个会话中按两种顺序各跑一次；配对差值在会话层级估计，因此 step 间噪声不会被当作独立样本。

在进程内以 CPU 测量时，记录路径的每区间开销在微秒量级。该数值是 **CPU 代理指标，不是 GPU 测量**：它不包含设备侧影响，实验的端到端结果才是分析器对真实训练运行影响的权威测量。任何开销结论都应以该实验为准；本页有意不复述一个总括性百分比。

## 故障排查

**collector 不可达。** 检查每个进程中的 `RELAX_STRAGGLER_COLLECTOR_ADDR` 是否都指向同一个 `host:port`，该主机与端口在该机器上是否可达，以及 rank 0 是否能成功绑定。绑定失败会被记录并计数（receiver 的 `accept_errors`）。在发送侧，`sender.send_errors` 统计连接与发送失败，`sender.connected` 报告 socket 是否已建立，`sender.dropped_queue_full` 统计 collector 不在期间因发送队列已满而丢弃的 envelope 数。

**没有判定出现。** 常见原因按顺序如下：

- **cohort 过小。** 上报该 `(cohort, stage)` 对的 rank 少于 `RELAX_STRAGGLER_MIN_COHORT`，因此该窗口只会给出 `uncertain` 而不是判定。单 rank 运行按设计不报告任何内容。
- **预热。** 最初的 `RELAX_STRAGGLER_WARMUP_WINDOWS` 个窗口永不被判定；`warmup_windows_skipped` 会统计它们。
- **等价性与工作量门槛。** 某 rank 自身的 token 工作量超出其对等中位数 `RELAX_STRAGGLER_WORK_TOLERANCE` 时，会给出原因为 `workload_incomparable` 的 `uncertain` 而非慢节点判定；`workload_missing_windows` 统计没有工作量可比较的窗口。缺失工作量不会抑制时间判定，只会把工作量证据标记为降级。
- **绝对下限。** 未超过 `RELAX_STRAGGLER_MIN_STAGE_MS` 的偏差会给出原因为 `below_absolute_floor` 的 `uncertain`。较短的元数据 stage 通常落入此类。
- **持续性。** 某 rank 必须连续 `RELAX_STRAGGLER_PERSIST_WINDOWS` 个窗口都出现偏差才会被报告。
- **覆盖率缺口。** 只包含多 rank cohort 中单个 rank 的窗口会增加 `incomplete_windows` 并且不产生判定。

**状态 JSON 的位置。** 设置 `RELAX_STRAGGLER_OUTPUT_DIR` 后，collector 会在 `<output_dir>/run_<run_id>/` 下写入 `collector_status.json` 与 rank 0 的 `runtime_status.json`；每个 rank 还会写入 `runtime_status_<label>.json`，rank 0 另外写入 `collector_status_<label>.json`。这些文件由后台线程原子刷新。若未设置 `RELAX_STRAGGLER_OUTPUT_DIR`，则不会产生状态文件，同样的计数只能通过 `perf/straggler/*` 指标与日志获取。
