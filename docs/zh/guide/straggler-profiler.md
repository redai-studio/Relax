# Straggler（慢 rank）分析

Relax 的 Megatron 后端内置一个常驻、低开销的慢 rank 分析器：每个训练 rank 用计算流上的 CUDA event 记录自己每一步的 forward / backward / optimizer / 梯度同步 / 参数 all-gather 等段的 GPU 时间（区间内 CPU 侧的 launch 空隙也算在里面），每隔 K 个 rollout 把一个固定长度的统计向量 Gloo all-gather 到 primary rank，由 primary 上的一个后台线程判定哪个 rank 拖慢了整组、为什么，并把结论写进指标、timeline 和日志。

它不是 `torch.profiler`：不采 kernel、不做 `cuda.synchronize()`、不改变通算 overlap，可以在生产训练里一直开着。分析器自身的任何异常都不会抛进训练循环：它会计数，屡次失败后在所有 rank 上一起自动关闭。

## 开启

分析器默认关闭。通过 Ray 运行时环境把环境变量传给训练 Actor：

```yaml
# configs/env.yaml
env_vars:
  RELAX_STRAGGLER_PROFILER: "1"
  RELAX_STRAGGLER_REPORT_INTERVAL: "10"
```

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `RELAX_STRAGGLER_PROFILER` | bool | false | 开关。 |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | 每多少个 rollout 做一次跨 rank 汇聚与判定（一个"窗口"）。一个 rollout 可能包含多个 optimizer step；下文的 ms/step 都是每个 rollout 的累计值。 |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | 候选 rank 的 robust z 分数（基于组内中位数 / MAD）门限。 |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | 候选 rank 相对组内中位数的最小超出比例。 |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | 连续多少个窗口满足条件才告警（所有 reason 都适用），用来过滤 checkpoint、GC 峰等一次性抖动；必须 ≥ 1。 |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | 已告警的 rank 要连续多少个干净窗口才解除告警，避免告警反复开关；「不确定」的窗口不算干净；必须 ≥ 1。 |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | profiler 专用 Gloo 组的超时（秒）。单次 gather 是几毫秒的 CPU 小张量汇聚；这个值只限制「有 rank 已经消失」时其余 rank 阻塞多久。 |

开启后每个 actor 角色的进程启动时打一行 `Straggler profiler enabled: report_interval=… primary=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…)`；critic / reference / `actor_fwd` 不跑训练步，不会被采样。

## 看什么

### 日志（primary rank）

日志由 primary 上的分析线程在判定一结束就打出，`step` 是该窗口最后一个 rollout 的 step，不等下一个训练步。没有告警的窗口打一行 INFO：

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

有告警时，某 rank 的 reason 变化的那个窗口打一条 WARNING，之后每个仍有告警的窗口打一张 INFO per-rank 表。每种 reason 都要连续 `PERSIST_WINDOWS` 个窗口才成立，要连续 `RECOVER_WINDOWS` 个干净窗口才解除（解除时再打一条 WARNING：`rank 5 (…) recovered from slow_device after 2 clean windows`）：

```text
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 195.6 ms/step
after its DP peers (peers idle 201.4 ms/step = 21% of their compute, z=120.1) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.55x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   941.0 |     6.4 |     0.4 | 14802.1 |        63.6 |   1.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   743.6 |     5.8 |   195.6 | 14592.7 |        51.0 |   1.8 | late_arrival |
```

表里只列出 rank 0 和 rank 3，实际每个 rank 一行。

某个 rank 的窗口不足以下结论时，它被标为「不确定」并给出原因（原因变化时打一条 WARNING），而不是默默当作正常：

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| 原因 | 含义 |
|------|------|
| `dropped_events` / `unread_events` / `profiler_errors` | 该 rank 这个窗口的计时不完整：有事件对因队列满被丢弃、窗口结束时还没读到（会计入下一个窗口）、或分析器在该 rank 上捕获了异常。未读到的 `dp_grad_sync` 会让该 rank 看起来像晚到，所以不下结论。 |
| `no_peers` | 所在 PP stage 没有其他 rank 可比。 |
| `missing_tokens` | 看起来算得慢，但该 stage 没有 token 数，分不清是数据不均还是设备慢。 |
| `no_wait_reference` | 它在等 PP，但同 stage 的其他 rank 都被判为慢、晚到或不确定，没有可以对比的等待时间。 |

「不确定」的窗口两头都不算：它打断候选的连续计数，也不算作告警解除的干净窗口；已有的告警保持不变。

### 指标（与 `perf/*` 同一 step key，进 TensorBoard / WandB / ClearML）

这些标量随训练线程下一次 `log_perf_data` 的 `perf/*` 在同一次 `tracking_utils.log` 里发出，不额外发请求：开启 `--use-metrics-service` 时，每次 log 都是训练线程上的一次同步 HTTP 请求。判定在后台线程上跑，所以一个窗口的标量通常随下一个 rollout 的 `perf/*` 发出；`straggler/window/{first,last}_rollout` 标明它属于哪个窗口。运行的最后一个 rollout 会等最后一个窗口判定完再发。如果后台判定落后，同一步里有多个窗口同时判定完成（正常运行中不会出现），每个窗口照常写 timeline 和日志，但只有其中最新一个上报成功的窗口的标量进指标（同一次 log 每个 key 只能带一个值）；被它覆盖的每个窗口打一条 WARNING `… windows ready at once; only rollouts …-… reach the metrics`，并计一次分析器自身异常；上报失败的窗口只按这次失败计一次（见下文「分析器自身出错时」）。

| 指标 | 含义 |
|------|------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | 每段 GPU 时间（ms/step）的全局中位数、最大值；`max_rank` / `spread` 是相对**同 PP stage 其他 rank** 超出最多的那个 rank 及其超出比例（`value/peer_median − 1`）。`lp_*` 是 log-prob / forward-only 阶段的同一组段。`dp_param_gather` 只在关闭 `--overlap-param-gather` 时有值（Megatron 不给 overlap 路径计时）。 |
| `straggler/self/*`、`straggler/self_per_ktok/*` | 自身计算时间（`fwd + bwd + optim`）及其按 token 归一化后的版本；后者与 `tokens/*` 一样，任一 rank token 数为 0 时不输出。 |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | 等待别人的时间（`pp_recv + dp_grad_sync + dp_param_gather`）。 |
| `straggler/late/max_ms`、`straggler/late/max_rank`、`straggler/late/peer_idle_ms` | 最晚到达 DP 梯度同步的 rank、晚了多少（按 `bwd + dp_grad_sync` 比较，见下文 `late_arrival`）、同组其他 rank 因此每步空转多久。 |
| `straggler/tokens/{median,max,spread}` | 各 rank 每步 token 数的不均程度：`median` / `max` 为全局值，`spread` 为同 stage 内最大超出；任一 rank token 数为 0 时不输出。 |
| `straggler/pp_stage_imbalance` | 各 PP stage 计算时间中位数的 `max/min`，与是否有慢 rank 无关。 |
| `straggler/gc/{median_ms,max_ms}` | Python GC 停顿。 |
| `straggler/flagged/count`、`straggler/flagged/rank`（无则 −1）、`straggler/flagged/reason` | 告警状态。reason 编码：0 none、1 slow_device、2 data_imbalance、3 upstream_wait、4 cpu_bound、5 late_arrival。 |
| `straggler/waiting/count` | 本窗口因上游 PP stage 慢而在等的 rank 数（被上游拖慢，不是慢 rank；不计入 `flagged`）。 |
| `straggler/uncertain/count`、`straggler/uncertain/rank`（无则 −1）、`straggler/uncertain/cause` | 「不确定」的 rank 数、第一个这样的 rank 及其原因。原因编码：0 none、1 dropped_events、2 unread_events、3 profiler_errors、4 no_peers、5 missing_tokens、6 no_wait_reference。 |
| `straggler/recovered/count` | 本窗口解除告警的 rank 数。 |
| `straggler/window/first_rollout`、`straggler/window/last_rollout` | 这组标量属于哪个窗口（它们通常随下一个 rollout 发出）。 |
| `straggler/latency/analyzed_ms` | 从窗口结束（gather 之前）到判定完成（告警日志此刻打出）的毫秒数。 |
| `straggler/health/state`、`straggler/health/errors`、`straggler/health/degraded_ranks` | 分析器自身健康：各 rank 中最差的状态（0 active、1 degraded、2 disabled）、本窗口捕获的异常数、处于 degraded 的 rank 数。 |
| `straggler/self_overhead_ms`、`straggler/gather_ms`、`straggler/analyze_ms`、`straggler/dropped_events`、`straggler/unread_events` | 工具自身开销：每步 drain 读事件的 CPU 时间；primary 上每窗口 gather 的耗时（首个窗口另含一次元数据 gather）；后台线程每窗口判定的耗时；因队列满被丢弃的事件对数（正常为 0）；窗口结束时尚未读到的事件对数（正常为 0）。 |

### Timeline

已开 `--timeline-dump-dir` 时（timeline 经 metrics-service adapter 落盘，所以还需要 `--use-metrics-service`），每个窗口会给每个 rank 的每个非零段追加一条 `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` 事件（`pid` = 2³⁰ + 全局 rank，高于任何真实 pid；`tid` = 段序号），在 Perfetto 里一行一个 rank，肉眼就能看出谁的 `bwd` 条更长、谁的 `dp_grad_sync` 条最短。

## 覆盖范围

计时复用 Megatron 已有的 22 个计时点，覆盖**调度面**：

| 覆盖 | 内容 |
| --- | --- |
| `fwd` / `bwd` | 每个 micro-batch 的前向 / 反向（整块，不含层内拆分） |
| `pp_recv` / `pp_send` | PP 点对点通信（9 个上游名字并入 2 段，不区分方向与级次） |
| `dp_grad_sync` | DP 梯度同步（4 个上游名字并入 1 段，CP 组内与 DP 组内不可分） |
| `dp_param_gather` | 分布式优化器的参数 all-gather |
| `optim` | 优化器各阶段（6 个上游名字） |

**不覆盖**（上游没有对应计时点，需要新增挂点）：

- **attention、MoE、head/logits、MTP 的内部**。上游 Megatron 在这些模块里没有任何计时点，
  所以现在只能看到它们被包含在 `fwd` / `bwd` 里，无法单独归因。要在这些粒度上定位，
  需要新增挂点——这属于后续工作，当前版本不提供。
- **EP all-to-all**（MoE 专家并行的通信）。它属于「等 peer」类，目前混在计算段里。
- **真实的通算重叠比例**。`dp_grad_sync` 变小**不能**直接读成「重叠更好」——它是「等待段之和」，
  是代理量，不是重叠率本身。

**已知粒度限制**（详见 `reason` 一节）：同一 TP 组的 rank 在前向 / 反向里互相等，其中一个变慢时
整组都会被报出，分不出是哪一个；所有 rank 同时变慢时（整机降频、共享存储抖动）发现不了。

## reason 怎么读

| reason | 判定依据 | 排查方向 |
|--------|----------|----------|
| `slow_device` | 自身 GPU 时间或 forward 时间显著高于同 PP stage 的其他 rank，token 数正常 | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`、ECC、同机邻居、NVLink 拓扑 |
| `data_imbalance` | 自身时间或 forward 时间高，但按 token 归一化后正常，token 数明显偏多 | `--balance-data`、动态 batch 切分、超长样本 |
| `cpu_bound` | 自身时间或 forward 时间偏高，**且**训练步内的 Python GC 停顿占自身时间 ≥ 5% | `gc.freeze()`、数据处理线程与训练线程抢 GIL、Python 侧 per-token 循环 |
| `upstream_wait` | 自身正常，但 `pp_recv` 显著高于同 stage 中未被判为慢或晚到的 rank，且超出量 ≥ 该 stage 中位自身时间的 5%；因为等 PP 而晚到梯度同步的 rank 也归为这一类 | 看上游 stage 被 flag 的 rank，或 `pp_stage_imbalance`（层划分不均） |
| `late_arrival` | 自身 GPU 时间正常，但它的 `bwd + dp_grad_sync` 显著**短于**同 DP 组其他 rank——一次集合通信对所有人同时结束，最后到的那个 rank 等得最少，其他人的这段时间里都含着等它的时间 | GPU 之间的 CPU 侧空隙：数据取用、同步 I/O / HTTP、GIL、launch gap；对该 rank `py-spy dump --pid <pid>` 最直接 |

`late_arrival` 容易被漏掉：只看 GPU 时间的规则抓不到它。该 rank 的 GPU 时间和大家一样，只是每步晚到梯度同步；常见原因是训练线程上的同步阻塞，例如 primary rank 在训练线程上用同步 HTTP 请求上报指标。

开启 `--overlap-grad-reduce` 时，梯度 reduce-scatter 在 backward 中按 bucket 发出，同 DP 组其他 rank 等最慢 rank 的时间落在各自的 `bwd` 里，`dp_grad_sync` 只剩很短的尾段。所以晚到按 `bwd + dp_grad_sync` 比较（开不开 overlap，这段时间都在梯度同步完成时结束），慢 rank 除了自身时间还单独比较 forward（forward 里没有 DP 集合通信）。在等的 rank 自身时间也会变长，所以 `late_arrival` 告警报的是该 rank 的 forward 与其他 rank 之比，而不是自身时间之比。

## 分析器自身出错时

计时调用、读事件、窗口 gather、判定、日志回调里的异常都被捕获并计数，不会抛进训练循环。某个 rank 累计 3 次异常，或连续 3 个窗口有丢弃的事件对或异常，就请求关闭分析器。这个请求放在该 rank 的窗口向量里（`health` 字段）一起 gather，所以所有 rank 看到同一张表，在同一个窗口一起关闭，不会有 rank 单独退出而让其他 rank 卡在下一次 gather 里。关闭后计时调用直接跳过，但该 rank **仍然参与每次窗口 gather**：它发一个长度为 `NUM_FIELDS` 的全零行，只把 `health` 写成 2，因此集合通信的形状不变，关闭状态也会持续传给还没看到的 rank。

gather 本身失败时只有本 rank 知道。现在的处理是**不退出集合通信**：失败被计数，本次窗口照常重置，下一次窗口仍发一行（正常行或上面的空行）。持续失败达到上限后该 rank 本地关闭并开始发空行，其余 rank 在下一个能完成的 gather 里读到 `health = 2`，在同一个窗口一起关闭。首个窗口的元数据汇聚只做一次、不重试：其余 rank 拿到元数据后每个窗口只做数值 gather，失败的 rank 单独重试会和它们的集合通信错位，所以它照常发这一窗口的行并把 `health` 写成 2，所有 rank 在这个窗口一起关闭。元数据汇聚本身只有一次集合通信，每个 rank 发一个 1 KiB 的定长字节行：本地序列化元数据失败、或元数据超长的 rank 照常参与，只是发一个空行，所有 rank 收到后看到同一个缺失的条目，同样在这个窗口一起关闭。为了让「某 rank 进程真的消失」这种发不出任何东西的情形有界，profiler 在安装时给这类 gather 单独建一个 Gloo 组，超时由 `RELAX_STRAGGLER_GATHER_TIMEOUT_S`（默认 60 s）控制，不影响通用 Gloo 组和训练路径的超时。**已知限制**：单次 gather 会因为组超时而阻塞到该超时；如果整个组都失联，所有 rank 都只能各自本地关闭，没有跨 rank 的传播。

## 开销

- 每个 Megatron 计时点一对非阻塞的 `event.record()`（事件从池里复用）；每步末尾用 `event.query()` 惰性读取已完成的事件，不调用 `cuda.synchronize()`，耗时记在 `straggler/self_overhead_ms`。
- 每 `REPORT_INTERVAL` 个 rollout 一次 Gloo `all_gather`，耗时记在 `straggler/gather_ms`。每个 rank 每窗口贡献 168 B（21 个 float64），与 micro-batch 数无关；由于是 `all_gather`，每个 rank 都会拿到 `N × 168 B` 的完整表，全作业每窗口合计接收 `N × (N−1) × 168 B`（不含 Gloo 协议开销）。首个窗口另有一次元数据汇聚，每个 rank 1 KiB。
- 判定在 primary 的后台线程上跑，耗时记在 `straggler/analyze_ms`，复杂度 O(n log n)（n = 同 stage rank 数）：8 个 rank 每窗口几毫秒，2048 个 rank 的 CPU 基准约 35 ms。
- 在 8×A100、Qwen3-0.6B DP8（step 约 1 s）上，逐项相加的最坏上界约 1.9 ms/step（约 0.19%），端到端开 / 关对比与 0 无法区分；开启后不新增同步调用，NCCL 与计算 kernel 的重叠比例不变。测量方法和各组数据见 [RFC #362](https://github.com/redai-studio/Relax/issues/362)。

## 告警时延

判定只在窗口关闭时执行，并且要连续 `PERSIST_WINDOWS` 个窗口成为候选才告警。默认参数（`REPORT_INTERVAL=10`、`PERSIST_WINDOWS=3`）下，从开始变慢到打出告警约 21–30 个 rollout；如果变慢开始得太晚、被所在窗口的累计值稀释而没过门限，还要再多一个窗口。调小这两个参数能更早告警，代价是更容易被一次性抖动触发。

告警日志在判定完成时立即打出（`straggler/latency/analyzed_ms`，通常几毫秒）；标量随下一次 `perf/*` 上报，再由所用的输出后端（metrics service、W&B、TensorBoard 或 ClearML）展示。

## 实现位置

- `relax/utils/straggler/timers.py`：替代 Megatron `config.timers` 的非阻塞计时器（Megatron 自带的 `Timer.start/stop` 会 `cuda.synchronize()`，所以不开 profiler 时 Relax 把它设为 `None`）。
- `relax/utils/straggler/stats.py`：统计向量布局、Megatron timer 名 → 段的映射。
- `relax/utils/straggler/collector.py`：每 rank 一个，事件池、惰性读取、GC 回调、每个窗口的 gather；不依赖 torch，事件、gather 等由外部传入。
- `relax/utils/straggler/worker.py`：primary 上按顺序执行判定的后台线程。
- `relax/utils/straggler/detector.py`：纯 numpy 的窗口分析与判定（含「不确定」与告警解除），可在无 GPU 环境单测。
- `relax/utils/straggler/health.py`：分析器自身的健康状态（active → degraded → disabled）。
- `relax/utils/straggler/reporter.py`：指标 / timeline 输出（训练线程）与日志输出（后台线程）。
- `relax/utils/straggler/runtime.py`：唯一接触加速器（经 `relax.utils.device`，CUDA / NPU 通用）、进程组和 Megatron 并行状态的模块；`StragglerProfiler` 负责每步在 `log_perf_data` 前后的上报。
- 接线：`relax/backends/megatron/model.py`（`config.timers = straggler_timers(...)` 三处）、`relax/backends/megatron/actor.py`（`install_straggler_profiler`，每步 `_log_perf_data`）。
