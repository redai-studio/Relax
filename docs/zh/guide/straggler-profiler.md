# Straggler（慢 rank）分析

Relax 的 Megatron 后端内置一个低开销的慢 rank 分析器，可以在生产训练中一直开着。它用 CUDA event 给每个训练 rank 的前向、反向、optimizer、梯度同步等阶段计时，每隔 K 个 rollout 在所有 rank 之间比较一次，报告拖慢整组的 rank 及原因。它不是 `torch.profiler`：不采集 kernel；普通训练步不调用 `cuda.synchronize()`，实测通信与计算的重叠在 A/A 运行间波动内；启用 trace 时，从第二个窗口起每个采样步开头同步一次，用于跨 rank 时间对齐。分析器自身出错不会影响训练。

## 功能

- **慢 rank 判定**：每个窗口（K 个 rollout）汇总各 rank 的阶段耗时，只在同一 PP stage 内比较，给出原因（见「reason 怎么读」）；数据不足时报告「不确定」。连续 `PERSIST_WINDOWS` 个窗口异常才告警，连续 `RECOVER_WINDOWS` 个正常窗口后解除。
- **PP 等待与 micro-batch**：非交错 1F1B 下把 PP 等待拆成填充、稳态、排空几段；记录单个 micro-batch 前向 / 反向耗时和 token 数的极值。
- **采样步 trace**：每个窗口把一个 rollout 里所有 rank 的计时放到同一条时间轴上，写成 Perfetto 文件：PP 气泡、每个 micro-batch 的前向 / 反向（VPP 下带模型块）、收发、梯度同步和 optimizer。
- **模块计时**：在采样步里给每层 attention 和 MLP / MoE 块的前向、反向计时，MoE 层内再分 router、dispatch、experts、combine。
- **运行中切换级别**：暂停、窗口统计、加 trace、加模块计时四档，训练中通过网页或接口切换。
- **出错自动关闭**：分析器自身的异常只计数，不抛进训练循环；某个 rank 在最近 10 个窗口内有 3 次异常，或连续 3 个窗口有丢失的计时或异常时，所有 rank 在同一个窗口一起关闭分析器，训练照常进行。状态见 `straggler/health/*`。

## 开启

分析器默认关闭。通过 Ray 运行时环境把环境变量传给训练 Actor：

```yaml
# configs/env.yaml
env_vars:
  RELAX_STRAGGLER_PROFILER: "1"
  RELAX_STRAGGLER_REPORT_INTERVAL: "10"
```

所有训练 rank 看到的这些变量（以及 `--timeline-dump-dir`）必须相同：窗口长度和是否写 trace 决定每个 rank 何时参加集合通信。多机时通过 Ray 运行时环境统一设置，不要在各节点上分别设置。

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `RELAX_STRAGGLER_PROFILER` | bool | false | 总开关。 |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | 每多少个 rollout 汇总并判定一次（一个窗口），怎么选见表后。下文的 ms/step 都是每个 rollout 的累计值。 |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | 判为异常的 robust z 分数门限（组内中位数和 MAD）。 |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | 判为异常所需的最小超出比例（相对组内中位数）。 |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | 连续多少个窗口异常才告警，用来过滤一次性抖动；必须 ≥ 1。 |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | 连续多少个正常窗口才解除告警；「不确定」的窗口不算正常；必须 ≥ 1。 |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | 分析器专用 Gloo 组的超时（秒），只决定某个 rank 进程消失时其余 rank 最多等多久。 |
| `RELAX_STRAGGLER_TRACE_DIR` | str | 空 | 采样步 trace 的输出目录；不设时用 `--timeline-dump-dir` 下的 `straggler/`，两者都没有就不写 trace。 |
| `RELAX_STRAGGLER_TRACE_KEEP` | int | 50 | 保留最新的多少个 trace 文件。 |
| `RELAX_STRAGGLER_LEVEL` | int | −1 | 启动级别（见「级别」）；−1 表示有 trace 目录且不超过 256 个 rank 时为 2，否则为 1。 |
| `RELAX_STRAGGLER_CONTROL_POLL_S` | float | 5.0 | primary 向 metrics service 查询级别的间隔（秒）。 |

`REPORT_INTERVAL` 是告警时延和开销之间的取舍。告警要连续 `PERSIST_WINDOWS` 个窗口，默认 10 时约 21–30 个 rollout，设为 1 时 3–4 个（Qwen3-0.6B DP8 两次运行，40 个 rollout 的正常运行没有误报）。每个窗口有一笔固定开销：一次统计 gather（8 卡约 1 ms，两机 16 卡 1.3–6.5 ms），开 trace 时再加一次设备同步（约 1 ms）、采样步的 gather 和一个 trace 文件，合计约 3 ms，摊到每个 rollout 是 3 ms / `REPORT_INTERVAL`。

- rollout 较长（例如 RL，每个几十秒到几分钟）：设为 1–2，额外开销不到 0.01%。
- rollout 较短（例如小模型 SFT，约 1 s）：保持默认 10；设为 1 时每个 rollout 额外约 3 ms（约 0.3%，多机时更多），加上直接开销会接近或超过 0.5%。窗口只有 1 个 rollout 时，采样步也可能来不及读完，这个窗口就没有 trace。

默认值不会按 rollout 时长自动调整，运行中也不能改：各 rank 必须在同一个 rollout 参加 gather，而各 rank 本地测到的时长可能不同。

开启后，每个 actor 进程启动时打印一行 `Straggler profiler enabled: report_interval=… primary=… level=… control=… trace_dir=… meta=RankMeta(…)`，`trace_dir=-` 表示不写 trace。critic、reference 和 `actor_fwd` 不跑训练步，不做分析。

## 级别

| 级别 | 内容 |
| --- | --- |
| 0 `paused` | 不计时、不判定，告警的连续计数冻结；标量只有 `straggler/control/level` |
| 1 `coarse` | 窗口统计和判定 |
| 2 `detail` | 再加采样步 trace；需要 trace 目录，没有时按 `coarse` 运行 |
| 3 `modules` | 再加模块计时：只在采样步进行，只写进 trace，不进窗口统计和判定；需要 trace 目录，只能显式设置 |

有 trace 目录时启动级别就是 `detail`，可以一直开着（开销见「开销」）；`modules` 开销更高，只在排查问题时临时打开。

启动级别由 `RELAX_STRAGGLER_LEVEL` 决定。开启 `--use-metrics-service` 时，把 `HEAD_NODE` 设为 head 节点的可访问地址；训练中可以在浏览器打开 `http://${HEAD_NODE}:8000/metrics/straggler` 点按钮切换，也可以直接发请求：

```bash
curl -X POST "http://${HEAD_NODE}:8000/metrics/straggler/control" -H 'Content-Type: application/json' -d '{"level": 0}'
```

所有 rank 从下一个窗口起一起切换，最多晚一个窗口加一次轮询间隔生效；页面上显示当前生效的级别。没开 metrics service 时级别固定为启动值，分析器因异常关闭后也不再响应切换。页面和接口由 head 节点 8000 端口的 Ray Serve 服务提供，没有鉴权，不要暴露到集群外。

页面上「Requested level」是最近一次请求的级别，「Level in effect」是 primary 上报的生效级别，在下一个窗口边界之前两者可能不同。primary 在后台线程每 `RELAX_STRAGGLER_CONTROL_POLL_S` 秒上报一次：上报照常而「Latest finished rollout」不再增长，说明训练卡住了；「Last report from the primary rank」一直变大，说明 primary 进程已经退出或连不上 metrics service。

## 看输出

### 日志（primary rank）

`step` 是窗口最后一个 rollout 的 step。没有告警的窗口打印一行 INFO：

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

有告警时，rank 的 reason 变化的窗口打印一条 WARNING，之后每个仍有告警的窗口打印一张逐 rank 的表；解除时再打印一条 WARNING，例如 `rank 5 (…) recovered from slow_device after 2 clean windows`：

```text
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 197.4 ms/step
after its DP peers (peers idle 203.3 ms/step = 22% of their compute, z=41.0) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.56x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | steady_ms | mb_fwd_max | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   933.7 |     6.4 |     6.1 | 14802.1 |        63.1 |   1.8 |       0.0 |      157.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   737.1 |     5.9 |   197.4 | 14592.7 |        50.5 |   1.8 |       0.0 |      148.0 | late_arrival |
```

表里只列出两行，实际每个 rank 一行。`steady_ms` 是稳态 PP 等待（`pp_wait_up + pp_wait_down`），`mb_fwd_max` 是本窗口最慢的单个前向（ms）。

每个窗口另打印一行 INFO，列出没有作为标量发出的统计（见「指标」的第二张表），例如 `[straggler] step=29 other metrics: analyze_ms 2.9, bwd/max_ms 401.2, …`。

数据不足以下结论的 rank 标为「不确定」并给出原因：

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| 原因 | 含义 |
|------|------|
| `dropped_events` / `unread_events` / `profiler_errors` | 该 rank 本窗口的计时不完整：有计时因队列满被丢弃、窗口结束时还没读到，或分析器在该 rank 上捕获了异常 |
| `no_peers` | 所在 PP stage 没有其他 rank 可比 |
| `missing_tokens` | 看起来算得慢，但没有 token 数，分不清是数据不均还是设备慢 |
| `no_wait_reference` | 它在等 PP，但同 stage 的其他 rank 都被判为慢、晚到或不确定，没有可对比的等待时间 |

「不确定」的窗口会打断告警的连续计数，也不算解除告警所需的正常窗口。

### 指标

每个窗口只发出下面第一张表里的标量，和 `perf/*` 使用同一个 step，写入 TensorBoard / WandB / ClearML；每多一个标量，训练线程在这一步 flush 时就多花一点时间。判定在后台线程上进行，一个窗口的标量通常随下一个 rollout 的 `perf/*` 发出，所属窗口见 `straggler/window/{first,last}_rollout`。其余统计由后台线程写进日志的 `other metrics` 一行（第二张表，键名省去 `straggler/` 前缀）。

计时复用 Megatron 已有的 22 个计时点，分成以下几段：

| 段 | 内容 |
| --- | --- |
| `fwd` / `bwd` | 每个 micro-batch 的前向 / 反向 |
| `pp_recv` / `pp_send` | PP 收发（9 个计时器合为 2 段） |
| `dp_grad_sync` | 梯度同步（4 个计时器合为 1 段，CP 组内和 DP 组内的部分分不开） |
| `dp_param_gather` | 分布式优化器的参数 all-gather，只在关闭 `--overlap-param-gather` 时有值 |
| `optim` | 优化器（6 个计时器；开启梯度裁剪时包含 grad-norm 归约） |
| `lp_fwd` / `lp_pp_recv` / `lp_pp_send` | log-prob / forward-only 阶段的同名段 |

| 标量 | 含义 |
|------|------|
| `straggler/flagged/{count,rank,reason}` | 告警状态，没有时 rank 为 −1。reason 编码：0 none、1 slow_device、2 data_imbalance、3 upstream_wait、4 cpu_bound、5 late_arrival |
| `straggler/uncertain/count` | 「不确定」的 rank 数 |
| `straggler/self/{median_ms,max_ms,max_rank}` | 自身计算时间（`fwd + bwd + optim`，ms/step）的全局中位数和最大值；`max_rank` 是相对同 PP stage 其他 rank 超出最多的 rank，不一定是最大值所在的 rank |
| `straggler/late/max_ms` | 最晚到达 DP 梯度同步的 rank 晚了多少 |
| `straggler/health/{state,errors,requested_by}` | 分析器自身的状态：各 rank 中最差的状态（0 active、1 degraded、2 disabled；degraded 指本窗口丢了计时或捕获了异常，或者 Megatron 调用计时接口的方式分析器不认识，见「限制」）、本窗口捕获的异常数，以及请求关闭的 rank（只在关闭时有） |
| `straggler/{dropped_events,unread_events}`、`straggler/trace/lost` | 因队列满被丢弃的计时数、窗口结束时还没读到的计时数、采样步 trace 丢失的计时数（正常都为 0） |
| `straggler/control/level` | 本窗口的级别：0 paused、1 coarse、2 detail、3 modules |
| `straggler/self_overhead_ms` | 分析器每步读取计时的 CPU 时间 |
| `straggler/window/{first_rollout,last_rollout}` | 这组标量所属的窗口 |
| `straggler/latency/analyzed_ms` | 从窗口结束到判定完成（告警日志打印）的毫秒数 |

| 日志中的统计 | 含义 |
|------|------|
| `<段>/{median_ms,max_ms,max_rank,spread}` | 每段 GPU 时间（ms/step）的全局中位数和最大值；`max_rank` / `spread` 是相对同 PP stage 其他 rank 超出最多的 rank 及超出比例（`value/peer_median − 1`） |
| `self/spread`、`self_per_ktok/*` | 自身计算时间相对同 PP stage 其他 rank 超出最多的比例；每千 token 的自身计算时间，在任一 rank 的 token 数为 0 时不输出 |
| `wait/{median_ms,max_ms,max_rank,spread}` | 等待其他 rank 的时间（`pp_recv + dp_grad_sync + dp_param_gather`） |
| `late/{max_rank,peer_idle_ms}` | 最晚到达 DP 梯度同步的 rank，以及同组其他 rank 每步因此空等的时间 |
| `tokens/{median,max,spread}` | 各 rank 每步 token 数的不均衡程度；任一 rank 的 token 数为 0 时不输出 |
| `pp_stage_imbalance` | 各 PP stage 计算时间中位数的 `max/min` |
| `{pp_wait_fill,pp_wait_up,pp_wait_down,pp_wait_drain}/{max_ms,max_rank}` | 训练阶段的 PP 等待按非交错 1F1B 的阶段拆开：`fill` 填充时等上游激活，`up` / `down` 稳态时等上游激活 / 下游梯度，`drain` 排空时等下游梯度。均衡的流水线在稳态不空等，`up + down` 超出传输时间的部分就是多出来的气泡；上游 rank 在一步开头停顿时体现在下游 stage 的 `fill` 上。只在开 PP 时输出 |
| `pp_bubble/ratio` | `pp_recv` 占已计时训练时间的比例，取各 PP stage 中位数的最大值；只在开 PP 时输出 |
| `micro_batch/{fwd_max_ms,fwd_max_rank,fwd_argmax,fwd_argmax_tokens,fwd_min_ms,bwd_max_ms,bwd_max_rank,tokens_max,tokens_min}` | 本窗口单个训练 micro-batch 的极值：最慢的前向及其 rank、它在所在 rollout 里的 micro-batch 序号（从 0 起）和 token 数（拿不到时为 −1），最快的前向，最慢的反向及其 rank，单个 micro-batch 的最大 / 最小 token 数（没有 token 数时不输出） |
| `step_max/{fwd,bwd,optim,pp_recv,dp_grad_sync}_{ms,rank}` | 本窗口内各段单个 rollout 的最大耗时及其 rank：窗口平均值看不出的单次变慢（某个 rollout 慢了一次，或到窗口末尾才开始变慢）在这里能看到；从未计时的段不输出 |
| `gc/{median_ms,max_ms}` | Python GC 停顿 |
| `waiting/count` | 因上游 PP stage 慢而在等待的 rank 数；它们是被拖慢的，不计入 `flagged` |
| `uncertain/{rank,cause}` | 第一个「不确定」的 rank（没有时为 −1）及其原因。cause 编码：0 none、1 dropped_events、2 unread_events、3 profiler_errors、4 no_peers、5 missing_tokens、6 no_wait_reference |
| `recovered/count` | 本窗口解除告警的 rank 数 |
| `health/degraded_ranks` | degraded 的 rank 数 |
| `gather_ms`、`analyze_ms` | primary 上每个窗口的 gather 和判定耗时 |
| `trace/{ranks,module_brackets,align_spread_ms,clock_offset_ms_max,unaligned_hosts,bubble_excess_ms_max,bubble_excess_rank}` | 采样步 trace 的统计，见下节 |

### 采样步 trace（Perfetto）

有 trace 目录时，从第二个窗口起，每个窗口的第一个 rollout（采样步）写成一个 `straggler_trace_rollout{N}.json`，用 [ui.perfetto.dev](https://ui.perfetto.dev) 打开；不需要 metrics service。

每个 rank 一行，按 (dp, pp, tp) 排序，同一条流水线的各 stage 相邻。行名带该 rank 实测的 PP 等待和按它自己的前向、反向耗时推算的均衡 1F1B 气泡，两者之差就是多出来的气泡；最慢的 stage 差为负，这时看其他 stage。行内的切片按时间嵌套：

| 层 | 内容 |
| --- | --- |
| 第 1 层（PP） | `busy`：连续的计算和通信；`bubble fill` / `up` / `down` / `drain`：PP 等待，按 1F1B 阶段命名，`waits_for` 写明在等上一个还是下一个 stage；`untimed`：超过 0.5 ms 且没有计时覆盖的时间（CPU 侧停顿） |
| 第 2 层（fwd / bwd） | `busy` 里的 `F{i}` / `B{i}`（`i` 是 micro-batch 序号，VPP 下另带模型块，记为 `F{i} chunk {c}`；前向带 token 数）、`send`、`grad sync`（DP 梯度同步，即 Megatron 的 `all-grads-sync`；其他梯度同步用 Megatron 的计时器名）、`param gather`、`optimizer`；log-prob 阶段为 `lp F{i}` 等 |
| 第 3 层（模块） | 只在 `modules` 级别出现：`F{i}` / `B{i}` 里每个 transformer 层的 attention 和 MLP（MoE 层记为 `moe`）的前向、反向，例如 `attn fwd L3`、`moe bwd L3`，参数带 `layer`、`micro_batch`（VPP 下另带 `model_chunk`）；MoE 层内另有 `router`、`dispatch`（EP all-to-all 在这里）、`experts`、`combine` 和 `shared experts`（开启 `--moe-shared-expert-overlap` 时，Megatron 不调用 shared experts 的 forward，而是把它拆开，在另一个 CUDA stream 上与 token 交换重叠执行，所以没有 `shared experts` 切片，这部分时间也不单独计）；MTP 层记为 `attn fwd MTP L1` 等（参数带 `mtp`）；出现在 `B{i}` 里的前向是激活重计算，记为 `… (recompute)`。要计其他模型的块，在 `relax/utils/straggler/megatron_adapter.py` 的 `MODULE_SPECS` 里加 spec |

primary 自己的 CPU 侧计时（例如 `actor_train`、每个 micro-batch 的 `get_data_batch_*`）单独一行 `primary CPU`。

下表中只有 `straggler/trace/lost` 是标量，其余写在日志的 `other metrics` 一行。

| 统计 | 含义 |
|------|------|
| `straggler/trace/ranks`、`straggler/trace/lost` | 提供了 trace 的 rank 数；丢失的计时数（队列满，超出每步 4096 条普通计时、8192 条模块计时的上限，或计时出错被跳过；模块计时单独排队，最多占队列的一半，队列满时先被挤掉，只计在这里，不计入 `dropped_events`） |
| `straggler/trace/module_brackets` | trace 中模块计时的条数，各 rank 求和（仅 `modules` 级别） |
| `straggler/trace/align_spread_ms` | 校正跨机时钟偏移后，同一 DP 组 `all-grads-sync` 结束时刻在各 rank 上的最大差值，即剩余的跨卡对齐误差；也包含 all-reduce 本身在各 rank 上结束的先后（两机 16 卡约 1 ms，单机 0.02–0.25 ms）；丢过计时的 rank 不计入 |
| `straggler/trace/clock_offset_ms_max` | 各机器相对 primary 所在机器的时钟偏移的最大绝对值（ms），trace 已按它平移；只在估算出至少两台机器时输出 |
| `straggler/trace/unaligned_hosts` | 没能估出时钟偏移、按自己墙钟画的机器数，这些机器上的切片和其他机器对不齐；只在不为 0 时输出 |
| `straggler/trace/bubble_excess_ms_max`、`straggler/trace/bubble_excess_rank` | 实测 PP 等待超出均衡 1F1B 气泡的最大值及其 rank |

### Timeline

同时开启 `--timeline-dump-dir` 和 `--use-metrics-service` 时，每个窗口为每个 rank 的每个非零段追加一条事件 `straggler/<段> rank{g}_dp{d}_tp{t}_pp{p}`，在 Perfetto 里每个 rank 一行。

## reason 怎么读

| reason | 判定依据 | 排查方向 |
|--------|----------|----------|
| `slow_device` | 自身计算时间或前向时间显著高于同 PP stage 的其他 rank，token 数正常 | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`、ECC、同机邻居、NVLink 拓扑 |
| `data_imbalance` | 自身计算时间或前向时间偏高，但按 token 归一化后正常，token 数明显偏多 | `--balance-data`、动态 batch 切分、超长样本 |
| `cpu_bound` | 自身计算时间或前向时间偏高，**且**训练步内的 Python GC 停顿占自身计算时间 ≥ 5% | `gc.freeze()`、数据处理线程与训练线程争抢 GIL、Python 侧逐 token 的循环 |
| `upstream_wait` | 自身计算时间正常，但 `pp_recv` 显著高于同 stage 中没被判为慢或晚到的 rank，且超出量 ≥ 该 stage 自身计算时间中位数的 5%；因为等 PP 而晚到梯度同步的 rank 也归为这一类。同 stage 没有这样的 rank 时（例如每个 stage 只有 2 个 rank、另一个已被告警），改和同一梯度同步组的 rank 比 `pp_recv`，多出的部分至少占它晚到时间的一半 | 看上游 stage 被告警的 rank，或 `pp_stage_imbalance`（层划分不均） |
| `late_arrival` | 自身 GPU 时间正常，但 `bwd + dp_grad_sync` 显著**短于**同 DP 组其他 rank：集合通信对所有 rank 同时结束，最后到的 rank 等得最少 | GPU 之间的 CPU 侧空隙：取数据、同步 I/O / HTTP、GIL、launch 间隙；对该 rank 执行 `py-spy dump --pid <pid>` 最直接 |

`late_arrival` 只看 GPU 时间发现不了：该 rank 的 GPU 时间和其他 rank 一样，只是每步都晚到梯度同步，常见原因是训练线程上的同步阻塞（例如在训练线程上用同步 HTTP 上报指标）。开启 `--overlap-grad-reduce` 时，等最慢 rank 的时间落在其他 rank 的 `bwd` 里，所以晚到按 `bwd + dp_grad_sync` 比较，慢 rank 另外单独比较前向时间。同样的原因，晚到 rank 的同伴反向会变长：只是自身计算时间偏高、前向不高的 rank，如果同一梯度同步组里晚到 rank 的晚到时间至少能解释它多出时间的一半，不报为慢。同一 TP 组整组都被报为慢时，告警会写明是整组：组内 rank 互相等待，分不出谁先变慢。已告警的 rank 换原因，要新原因连续 `PERSIST_WINDOWS` 个窗口都成立，在那之前保留原来的原因。

## 开销

- 每个计时点一对非阻塞的 CUDA event，每步结束时只读取已完成的事件；每个窗口一次 Gloo all-gather，每个 rank 发送 312 B。
- 8×A100、Qwen3-0.6B DP8（每步约 1 s）、完整功能（级别 2、trace 目录、metrics service、overlap）下，分析器在 rank 0 上的直接开销约每步 1.5 ms（约 0.15%），其中计时点约占六成；其他 rank 在下一个集合通信上等 rank 0。端到端运行内交叉对照为 +0.13%（95% CI −0.05%～+0.30%，4 组），区间上界低于 0.5%。
- 采样步 trace：每个窗口多一次设备同步和锚点（约 1.1 ms），并在窗口结束时把采样步 gather 到 primary（窗口结束的两次 gather 共约 2 ms）；默认窗口下每步最多约 0.03%。
- `modules` 级别：同样的端到端交叉对照为 +0.34%（95% CI +0.20%～+0.48%，4 组）。每个块的前向多约 21 µs、反向多约 15 µs CPU 时间（单卡微基准），每个窗口读取模块计时多约 3.5 ms；Qwen3-0.6B TP1×PP4×DP2 上采样步的前向、反向与 detail 级别分不开（±3% 以内）。
- 测量方法和数据见 [RFC #362](https://github.com/redai-studio/Relax/issues/362)。

## 限制

- 计时复用 Megatron 已有的计时点。窗口统计里 attention、MoE、head / logits、MTP 的内部都包含在 `fwd` / `bwd` 中，`modules` 级别只在 trace 里给出每层 attention、MLP / MoE 和 MoE 层内各部分的耗时；EP all-to-all 计入计算时间；不给真实的通信计算重叠比例，`dp_grad_sync` 变小不能理解为重叠变好。
- 同一 TP 组的 rank 在前向 / 反向里互相等待，其中一个变慢时整组都会被报出，分不出谁先变慢；所有 rank 同时变慢时（整机降频、共享存储抖动、整组通信变慢）发现不了。
- PP 阶段拆分、trace 里的阶段命名和气泡推算只适用于非交错 1F1B，VPP 的收发只计入 `pp_recv`。VPP 下的 micro-batch 序号和模型块按 Megatron 的交错调度表（`get_schedule_table`，用模型自己的 config 生成）由调用顺序换算；`combined_1f1b` 的训练调用把前向和反向合并执行，不编号。
- trace 的开销随 rank 数线性增长，都落在 primary 上：每个窗口把所有 rank 的采样步拼成一个文件，CPU 基准中 2048 个 rank 时约 10 MB、构建约 0.25 s（`modules` 级别约 80 MB、2 s），构建时占用 primary 的 Python 解释器，会拖慢它的训练线程。所以超过 256 个 rank 时默认不开 trace；手动设为 2 或 3 前先估算这部分开销。
- 跨卡对齐依赖各 rank 的墙钟，跨机时按机器估算时钟偏移后平移：有跨机的 DP 组时用 DP 梯度同步的结束时刻（DP 组内同时结束）；只有 PP 跨机时用跨机那一跳前向、反向的先后关系，假设两个方向的传输时间相同；丢过计时的 rank 不参与估算；两种都联系不上的机器不平移，记在 `unaligned_hosts`。两机实测与 ssh 测得的时钟偏差相比，前一种相差 0.3 ms 以内，后一种 0.2–0.6 ms。CUDA graph 捕获期间不计时。
- 告警有延迟：默认参数下从开始变慢到告警约 21–30 个 rollout，rollout 很长时调小 `REPORT_INTERVAL`（见「开启」）。调小 `PERSIST_WINDOWS` 也能更早告警，但更容易被一次性抖动触发。
- 计时依赖 Megatron 的计时接口（`config.timers`）。Megatron 版本的调用方式变了（出现不认识的计时点名字，给 `timers()` / `start` / `stop` 传入新参数，或调用 `Timers.log` / `write`），或者第一个 rollout 没有走到每步都会调用的计时点（`forward-compute`、`backward-compute`、`all-grads-sync`、`optimizer-copy-to-main-grad`；log-prob 前向为 `forward-compute`）时，训练不会中断：这次调用被忽略，打一条警告，这个 rank 此后的 `straggler/health/state` 记为 1（degraded），但不计入自动关闭的条件，相应的时间不进统计。适配层的 CPU 契约测试会对照已安装的 Megatron 检查这些变化。关闭 profiler（`RELAX_STRAGGLER_PROFILER=0`）即恢复 `config.timers = None`。
- 某个 rank 进程整个消失时，其余 rank 的每次 gather 都要等到 `RELAX_STRAGGLER_GATHER_TIMEOUT_S` 超时，失败多次后各自关闭分析器。
