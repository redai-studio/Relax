# Straggler（慢 rank）分析

Relax 的 Megatron 后端内置一个常驻、低开销的慢 rank 分析器：每个训练 rank 用计算流上的 CUDA event 记录自己每一步的 forward / backward / optimizer / 梯度同步 / 参数 all-gather 等段的 GPU 时间（区间内 CPU 侧的 launch 空隙也算在里面），每隔 K 个 rollout 把一个固定长度的统计向量 Gloo all-gather 到 primary rank，由它判定哪个 rank 拖慢了整组、为什么，并把结论写进指标、timeline 和日志。

它不是 `torch.profiler`：不采 kernel、不做 `cuda.synchronize()`、不改变通算 overlap，可以在生产训练里一直开着。

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
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | 连续多少个窗口满足条件才告警（所有 reason 都适用），用来过滤 checkpoint、GC 峰等一次性抖动。 |

开启后每个 actor 角色的进程启动时打一行 `Straggler profiler enabled: report_interval=… primary=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…)`；critic / reference / `actor_fwd` 不跑训练步，不会被采样。

## 看什么

### 日志（primary rank）

没有告警的窗口打一行 INFO：

```text
[straggler] step=29 no straggler; self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

有告警时，某 rank 的 reason 变化的那个窗口打一条 WARNING，之后每个仍有告警的窗口打一张 INFO per-rank 表。每种 reason 都要连续 `PERSIST_WINDOWS` 个窗口才成立：

```text
[straggler] step=14 rank 0 (rank0_dp0_tp0_pp0, host=…, gpu=0) reaches the DP grad-sync 433.0 ms/step
after its DP peers (peers idle 451.4 ms/step = 44% of their compute, z=8.9) for 3 windows;
its own GPU compute is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.26x -> late_arrival (host-side stall)
[straggler] step=14 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | reason
    0 | rank0_dp0_tp0_pp0 | …    |      0 |  1021.2 |    26.1 |   433.0 | 15610.8 |        65.4 |   1.8 | late_arrival
    1 | rank1_dp1_tp0_pp0 | …    |      1 |  1083.4 |   404.3 |    54.7 | 15534.4 |        69.7 |   2.2 | none
```

### 指标（与 `perf/*` 同一 step key，进 TensorBoard / WandB / ClearML）

这些标量和同一步的 `perf/*` 在同一次 `tracking_utils.log` 里发出，不额外发请求：开启 `--use-metrics-service` 时，每次 log 都是训练线程上的一次同步 HTTP 请求。

| 指标 | 含义 |
|------|------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | 每段 GPU 时间（ms/step）的全局中位数、最大值；`max_rank` / `spread` 是相对**同 PP stage 其他 rank** 超出最多的那个 rank 及其超出比例（`value/peer_median − 1`）。`lp_*` 是 log-prob / forward-only 阶段的同一组段。`dp_param_gather` 只在关闭 `--overlap-param-gather` 时有值（Megatron 不给 overlap 路径计时）。 |
| `straggler/self/*`、`straggler/self_per_ktok/*` | 自身计算时间（`fwd + bwd + optim`）及其按 token 归一化后的版本；后者与 `tokens/*` 一样，任一 rank token 数为 0 时不输出。 |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | 等待别人的时间（`pp_recv + dp_grad_sync + dp_param_gather`）。 |
| `straggler/late/max_ms`、`straggler/late/max_rank`、`straggler/late/peer_idle_ms` | 最晚到达 DP 梯度同步的 rank、晚了多少、同组其他 rank 因此每步空转多久。 |
| `straggler/tokens/{median,max,spread}` | 各 rank 每步 token 数的不均程度：`median` / `max` 为全局值，`spread` 为同 stage 内最大超出；任一 rank token 数为 0 时不输出。 |
| `straggler/pp_stage_imbalance` | 各 PP stage 计算时间中位数的 `max/min`，与是否有慢 rank 无关。 |
| `straggler/gc/{median_ms,max_ms}` | Python GC 停顿。 |
| `straggler/flagged/count`、`straggler/flagged/rank`（无则 −1）、`straggler/flagged/reason` | 告警状态。reason 编码：0 none、1 slow_device、2 data_imbalance、3 upstream_wait、4 cpu_bound、5 late_arrival。 |
| `straggler/waiting/count` | 本窗口因上游 PP stage 慢而在等的 rank 数（被上游拖慢，不是慢 rank；不计入 `flagged`）。 |
| `straggler/self_overhead_ms`、`straggler/gather_ms`、`straggler/analyze_ms`、`straggler/dropped_events` | 工具自身开销：每步 drain 读事件的 CPU 时间；primary 上每窗口 gather 的耗时（首个窗口另含一次元数据 gather）；primary 上每窗口判定的耗时；因队列满被丢弃的事件对数（正常为 0）。 |

### Timeline

已开 `--timeline-dump-dir` 时（timeline 经 metrics-service adapter 落盘，所以还需要 `--use-metrics-service`），每个窗口会给每个 rank 的每个非零段追加一条 `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` 事件（`pid` = 2³⁰ + 全局 rank，高于任何真实 pid；`tid` = 段序号），在 Perfetto 里一行一个 rank，肉眼就能看出谁的 `bwd` 条更长、谁的 `dp_grad_sync` 条最短。

## reason 怎么读

| reason | 判定依据 | 排查方向 |
|--------|----------|----------|
| `slow_device` | 自身 GPU 时间显著高于同 PP stage 的其他 rank，token 数正常 | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`、ECC、同机邻居、NVLink 拓扑 |
| `data_imbalance` | 自身时间高，但按 token 归一化后正常，token 数明显偏多 | `--balance-data`、动态 batch 切分、超长样本 |
| `cpu_bound` | 自身时间偏高，**且**训练步内的 Python GC 停顿占自身时间 ≥ 5% | `gc.freeze()`、数据处理线程与训练线程抢 GIL、Python 侧 per-token 循环 |
| `upstream_wait` | 自身正常，但 `pp_recv` 显著高于同 stage 其他 rank，且超出量 ≥ 该 stage 中位自身时间的 5% | 看上游 stage 被 flag 的 rank，或 `pp_stage_imbalance`（层划分不均） |
| `late_arrival` | 自身 GPU 时间正常，但它的 `dp_grad_sync` 区间显著**短于**同 DP 组其他 rank——一次集合通信对所有人同时结束，最后到的那个 rank 区间最短，其他人的区间里都含着等它的时间 | GPU 之间的 CPU 侧空隙：数据取用、同步 I/O / HTTP、GIL、launch gap；对该 rank `py-spy dump --pid <pid>` 最直接 |

`late_arrival` 容易被漏掉：只看 GPU 时间的规则抓不到它。它在原型的第一个真实作业里就出现了——rank 0 的 GPU 时间和大家一样，却每步晚 0.4–1.5 s 到梯度同步，原因是 primary rank 在训练线程上同步发送日志指标的 HTTP 请求。

## 开销

- 每个 Megatron 计时点一对 `cudaEventRecord`（非阻塞，事件从池里复用）：每个 micro-batch 的 forward / backward 各一对，每步的梯度同步、参数 all-gather、optimizer 各阶段共约十对。
- 每步末尾 `event.query()` 惰性读取已完成的事件并累加到窗口向量：`straggler/self_overhead_ms` 在 8×A100 上实测 0.12–0.15 ms/step。
- 每 `REPORT_INTERVAL` 个 rollout 一次 Gloo `all_gather`（每 rank 18 个 float64）加 primary 上的判定，分别记在 `gather_ms` 和 `analyze_ms`，摊到每步 < 1 ms。判定是 O(n log n)（n = 同 stage rank 数），单机 CPU 微基准每窗口 8 rank 2.4 ms、512 rank 10 ms、2048 rank 35 ms；其余 rank 会在下一次集合通信处等 primary。
- 每次计时调用（start / stop 与两次 `event.record()`）约 30 µs。在 A100 实验机的一张空卡上，按每步最多 17 次计时（3 个 micro-batch）回放 Megatron 调用序列：kernel 发射受限的循环里，计时调用加步末读取事件使每步墙钟增加 0.80 ms（最大 0.97 ms）；计算受限的循环里只增加 0.09 ms。
- 以上各项相加，并假设全部落在关键路径上：典型约 1.2 ms/step，最坏约 1.9 ms/step，在 step 约 1.15 s 时分别为 0.10% 和 0.17%。这是逐项相加的估算上界。
- 端到端：8×A100 / A800、Qwen3-0.6B SFT（DP8，step ≈ 1.15 s，对计时开销最敏感的小模型场景）。开启 / 关闭 profiler 各跑一次的对比受不同运行之间约 1% 的波动限制，所以改为在同一次运行内每 10 个 rollout 切换一次开关，另一次运行的相位相反，使同一个 10 步块在相同数据上一开一关。metrics service 关闭时 4 对：−0.04% / +2.04% / −0.16% / −0.15%；3 台机器上的 A/A 对照（两次都关）为 +0.41% / −0.70% / −1.41%。开 profiler 不增加第 2 代 GC 次数。逐项开销相加、并假设全部落在关键路径上时，8 个 rank 最坏约 0.17%。

## 实现位置

- `relax/utils/straggler/timers.py`：替代 Megatron `config.timers` 的非阻塞计时器（Megatron 自带的 `Timer.start/stop` 会 `cuda.synchronize()`，所以不开 profiler 时 Relax 把它设为 `None`）。
- `relax/utils/straggler/stats.py`：统计向量布局、Megatron timer 名 → 段的映射。
- `relax/utils/straggler/collector.py`：每 rank 一个，事件池、惰性读取、GC 回调、Gloo 汇聚。
- `relax/utils/straggler/detector.py`：纯 numpy 的窗口分析与判定，可在无 GPU 环境单测。
- `relax/utils/straggler/reporter.py`：指标 / timeline / 日志输出。
- 接线：`relax/backends/megatron/model.py`（`config.timers = straggler_timers(...)` 三处）、`relax/backends/megatron/actor.py`（`install_straggler_collector`、每步 `_straggler_end_step`）。
