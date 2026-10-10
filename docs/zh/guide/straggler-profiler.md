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
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | 连续多少个窗口满足条件才告警（所有 reason 都适用），用来过滤 checkpoint、GC 峰等一次性抖动。 |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | 已告警的 rank 要连续多少个干净窗口才解除告警，避免告警反复开关；「不确定」的窗口不算干净。 |

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
[straggler] step=14 rank 0 (rank0_dp0_tp0_pp0, host=…, gpu=0) reaches the DP grad-sync 433.0 ms/step
after its DP peers (peers idle 451.4 ms/step = 44% of their compute, z=8.9) for 3 windows;
its own GPU compute is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.26x -> late_arrival (host-side stall)
[straggler] step=14 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | reason
    0 | rank0_dp0_tp0_pp0 | …    |      0 |  1021.2 |    26.1 |   433.0 | 15610.8 |        65.4 |   1.8 | late_arrival
    1 | rank1_dp1_tp0_pp0 | …    |      1 |  1083.4 |   404.3 |    54.7 | 15534.4 |        69.7 |   2.2 | none
```

这条告警来自原型的第一个真实作业（见下文 `late_arrival`）；此后 `its own GPU …` 一项已从自身时间之比改为 forward 之比（`its own GPU forward is …x peers`）。

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

这些标量随训练线程下一次 `log_perf_data` 的 `perf/*` 在同一次 `tracking_utils.log` 里发出，不额外发请求：开启 `--use-metrics-service` 时，每次 log 都是训练线程上的一次同步 HTTP 请求。判定在后台线程上跑，所以一个窗口的标量通常随下一个 rollout 的 `perf/*` 发出；`straggler/window/{first,last}_rollout` 标明它属于哪个窗口。运行的最后一个 rollout 会等最后一个窗口判定完再发。

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
| `straggler/latency/analyzed_ms`、`straggler/latency/emitted_ms`、`straggler/latency/prev_delivered_ms` | 上报时延，均从窗口结束（gather 之前）算起：判定完成（告警日志此刻打出）、训练线程把标量交给 `log_perf_data`、上一个窗口的 `log_perf_data` 返回（开 metrics service 时即服务端已确认收到；一个窗口不能在自己的数据里带自己的送达时间）。每个窗口送达后另有一行 INFO：`window rollouts 10-19: closed -> analyzed +… ms (alerts logged), emitted +… ms, delivered +… ms`。 |
| `straggler/health/state`、`straggler/health/errors`、`straggler/health/degraded_ranks` | 分析器自身健康：各 rank 中最差的状态（0 active、1 degraded、2 disabled）、本窗口捕获的异常数、处于 degraded 的 rank 数。 |
| `straggler/self_overhead_ms`、`straggler/gather_ms`、`straggler/analyze_ms`、`straggler/dropped_events`、`straggler/unread_events` | 工具自身开销：每步 drain 读事件的 CPU 时间；primary 上每窗口 gather 的耗时（首个窗口另含一次元数据 gather）；后台线程每窗口判定的耗时；因队列满被丢弃的事件对数（正常为 0）；窗口结束时尚未读到的事件对数（正常为 0）。 |

### Timeline

已开 `--timeline-dump-dir` 时（timeline 经 metrics-service adapter 落盘，所以还需要 `--use-metrics-service`），每个窗口会给每个 rank 的每个非零段追加一条 `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` 事件（`pid` = 2³⁰ + 全局 rank，高于任何真实 pid；`tid` = 段序号），在 Perfetto 里一行一个 rank，肉眼就能看出谁的 `bwd` 条更长、谁的 `dp_grad_sync` 条最短。

## reason 怎么读

| reason | 判定依据 | 排查方向 |
|--------|----------|----------|
| `slow_device` | 自身 GPU 时间或 forward 时间显著高于同 PP stage 的其他 rank，token 数正常 | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`、ECC、同机邻居、NVLink 拓扑 |
| `data_imbalance` | 自身时间或 forward 时间高，但按 token 归一化后正常，token 数明显偏多 | `--balance-data`、动态 batch 切分、超长样本 |
| `cpu_bound` | 自身时间或 forward 时间偏高，**且**训练步内的 Python GC 停顿占自身时间 ≥ 5% | `gc.freeze()`、数据处理线程与训练线程抢 GIL、Python 侧 per-token 循环 |
| `upstream_wait` | 自身正常，但 `pp_recv` 显著高于同 stage 中未被判为慢或晚到的 rank，且超出量 ≥ 该 stage 中位自身时间的 5%；因为等 PP 而晚到梯度同步的 rank 也归为这一类 | 看上游 stage 被 flag 的 rank，或 `pp_stage_imbalance`（层划分不均） |
| `late_arrival` | 自身 GPU 时间正常，但它的 `bwd + dp_grad_sync` 显著**短于**同 DP 组其他 rank——一次集合通信对所有人同时结束，最后到的那个 rank 等得最少，其他人的这段时间里都含着等它的时间 | GPU 之间的 CPU 侧空隙：数据取用、同步 I/O / HTTP、GIL、launch gap；对该 rank `py-spy dump --pid <pid>` 最直接 |

`late_arrival` 容易被漏掉：只看 GPU 时间的规则抓不到它。它在原型的第一个真实作业里就出现了——rank 0 的 GPU 时间和大家一样，却每步晚 0.4–1.5 s 到梯度同步，原因是 primary rank 在训练线程上同步发送日志指标的 HTTP 请求。

开启 `--overlap-grad-reduce` 时，梯度 reduce-scatter 在 backward 中按 bucket 发出，同 DP 组其他 rank 等最慢 rank 的时间落在各自的 `bwd` 里，`dp_grad_sync` 只剩很短的尾段。所以晚到按 `bwd + dp_grad_sync` 比较（开不开 overlap，这段时间都在梯度同步完成时结束），慢 rank 除了自身时间还单独比较 forward（forward 里没有 DP 集合通信）。在等的 rank 自身时间也会变长，所以 `late_arrival` 告警报的是该 rank 的 forward 与其他 rank 之比，而不是自身时间之比。

## 分析器自身出错时

计时调用、读事件、窗口 gather、判定、日志回调里的异常都被捕获并计数，不会抛进训练循环。某个 rank 累计 3 次异常，或连续 3 个窗口有丢弃的事件对或异常，就请求关闭分析器。这个请求放在该 rank 的窗口向量里（`health` 字段）一起 gather，所以所有 rank 看到同一张表，在同一个窗口一起关闭，不会有 rank 单独退出而让其他 rank 卡在下一次 gather 里。关闭后计时调用直接跳过、不再 gather，primary 发出一组 `straggler/health/state = 2` 并打一条 WARNING 说明是哪个 rank 请求的（具体原因在那个 rank 的日志里）；训练照常继续。gather 本身失败时只有本 rank 知道，它就地停止参与后续 gather。

## 开销

- 每个 Megatron 计时点一对 `event.record()`（非阻塞，事件从池里复用）：每个 micro-batch 的 forward / backward 各一对，每步的梯度同步、参数 all-gather、optimizer 各阶段共约十对。
- 每步末尾 `event.query()` 惰性读取已完成的事件并累加到窗口向量：`straggler/self_overhead_ms` 在 8×A100 上实测 0.10–0.16 ms/step。
- 每 `REPORT_INTERVAL` 个 rollout 一次 Gloo `all_gather`（每 rank 21 个 float64），记在 `gather_ms`，摊到每步 < 1 ms。判定在 primary 的后台线程上跑，记在 `analyze_ms`，不占训练线程，其余 rank 也不用在下一次集合通信处等它；判定是 O(n log n)（n = 同 stage rank 数），单机 CPU 微基准每窗口 8 rank 2.4 ms、512 rank 10 ms、2048 rank 35 ms。
- 每次计时调用（start / stop 与两次 `event.record()`）约 30 µs。在 A100 实验机的一张空卡上，按每步最多 17 次计时（3 个 micro-batch）回放 Megatron 调用序列：kernel 发射受限的循环里，计时调用加步末读取事件使每步墙钟增加 0.80 ms（最大 0.97 ms）；计算受限的循环里只增加 0.09 ms。
- 以上各项相加，并假设全部落在关键路径上：典型约 1.2 ms/step，最坏约 1.9 ms/step，在 step 约 1.15 s 时分别为 0.10% 和 0.17%。这是逐项相加的估算上界。
- 端到端：8×A100、Qwen3-0.6B SFT（DP8，step ≈ 0.95 s，对计时开销最敏感的小模型场景），开启 `--overlap-grad-reduce --overlap-param-gather`，关闭 metrics service。开启 / 关闭 profiler 各跑一次的对比受不同运行之间约 1% 的波动限制，所以改为在同一次运行内每 10 个 rollout 切换一次开关，另一次运行的相位相反，使同一个 10 步块在相同数据上一开一关。按预先指定的规则剔除耗时尖峰（任一次运行中超过该次运行中位数 1.25 倍的步，两次运行都去掉，约 4%）后，2 对合计 +0.06%（95% CI −0.25% ~ +0.38%，56 个块）；同样方法的 A/A 对照（两次都关）为 +0.22%（−0.11% ~ +0.55%），即这种测法本身的噪声约 ±0.3%。更早在关闭通算 overlap 时测的 4 对，按同一规则事后剔除后为 −0.18% ~ +0.07%。开 profiler 不增加第 2 代 GC 次数。
- 通算 overlap：同一 recipe 开启 metrics service，关 / 开 / 再关三次各用 `torch.profiler` 采 2 个 rollout，NCCL kernel 与计算 kernel 的重叠比例为 0.1321 / 0.1321 / 0.1387，`cuda*Synchronize` 次数和 NCCL kernel 数三次相同；开启后 trace 里多出计时用的 event record 与 elapsed-time 调用，没有新增同步调用。
- 上报时延（同一次运行，`REPORT_INTERVAL=10`）：告警日志在窗口结束后 4–6 ms 打出；标量随下一个 rollout 的 `log_perf_data` 发出，metrics service 约 2.4–2.5 s 后确认收到；运行的最后一个窗口当场发出（约 0.25 s）。

## 实现位置

- `relax/utils/straggler/timers.py`：替代 Megatron `config.timers` 的非阻塞计时器（Megatron 自带的 `Timer.start/stop` 会 `cuda.synchronize()`，所以不开 profiler 时 Relax 把它设为 `None`）。
- `relax/utils/straggler/stats.py`：统计向量布局、Megatron timer 名 → 段的映射。
- `relax/utils/straggler/collector.py`：每 rank 一个，事件池、惰性读取、GC 回调、每个窗口的 gather；不依赖 torch，事件、gather 等由外部传入。
- `relax/utils/straggler/worker.py`：primary 上按顺序执行判定的后台线程。
- `relax/utils/straggler/detector.py`：纯 numpy 的窗口分析与判定（含「不确定」与告警解除），可在无 GPU 环境单测。
- `relax/utils/straggler/health.py`：分析器自身的健康状态（active → degraded → disabled）。
- `relax/utils/straggler/reporter.py`：指标 / timeline 输出（训练线程）与日志输出（后台线程）、上报时延。
- `relax/utils/straggler/runtime.py`：唯一接触加速器（经 `relax.utils.device`，CUDA / NPU 通用）、进程组和 Megatron 并行状态的模块；`StragglerProfiler` 负责每步在 `log_perf_data` 前后的上报。
- 接线：`relax/backends/megatron/model.py`（`config.timers = straggler_timers(...)` 三处）、`relax/backends/megatron/actor.py`（`install_straggler_profiler`，每步 `_log_perf_data`）。
