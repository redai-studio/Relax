# Agentic KV 调度

两个相互独立、默认关闭的特性，用于降低 agentic rollout 期间的 KV cache 压力：**session KV lifecycle** 在 session 结束时立即释放其 KV，**program-aware admission** 在 request 边界上限制活跃工作集大小。

## 概述

在 agentic rollout 中，一个 session 是长生命周期的多轮 program，而不是单次 request。两轮之间 agent 在执行 tool——可能耗时数秒到数分钟——而该 session 的 KV prefix 一直驻留在 engine 里。并发 session 一多，KV pool 就会打满，engine 退化为 eviction 和 recompute，新到达的 request 排在这些并没有真正在 decode 的工作后面。

Relax 从 session 生命周期的两端分别处理这个问题：

| 特性 | Flag | 作用位置 | 效果 |
|---|---|---|---|
| Session KV lifecycle | `--agentic-session-lifecycle` | session 结束时 | 立即释放 session 的 KV，而不是等待 priority-aware radix-cache eviction（同 priority 内按 LRU） |
| Program-aware admission | `--agentic-program-admission` | 每次 request 开始前 | 限制集群同时承诺出去的 KV 总量 |

两个特性都**默认关闭**、**彼此独立**（可以只开其中一个）、并且都**fail-open**——任何信号缺失、过期或失败都会回退到原有行为。两者都不改变生成结果：完整的 replay payload（`input_ids`）始终会被发送，所以即使 cache 是冷的也能正确服务。

::: tip
这两个特性优化的是*调度*开销，不是模型质量。当 rollout 是 KV-bound 时才需要开启——engine `token_usage` 高、频繁 eviction、GPU 没打满但 request 在排队。
:::

## Session KV Lifecycle

### 做了什么

开启后，SGLang backend adapter 会在每次 `generate` 调用时把 Relax session ID 作为顶层 `session_id` 字段发送。同一 session 的并发 request 与 SessionForest branch 共享该 ID，每次 backend attempt 仍保留独立 request ID。session 进入终态或被 drop 时，Relax 会先 abort 或等待在途 request，再发出幂等的 `/close_session`，立即释放该 session 的 KV。Partial rollout 和 fully async 会跨 step 保留未完成 session，因此不会提前关闭其 KV。

### 路由

sgl-router **不会**代理 `/close_session`。因此 Relax 直接向每个 engine base URL 扇出，方式与 request abort 一致。engine 的 DP controller 会把释放广播到所有 DP rank；不持有该 session 的 rank 会 no-op。

```
                    ┌──────────────────┐
   generate ───────►│   sgl-router     │───────► engine (placement decided here)
   (session_id)     └──────────────────┘
                    ┌──────────────────┐
   /close_session ─X│   sgl-router     │   not proxied
                    └──────────────────┘
                             │
   /close_session ───────────┴──────────────► every engine base URL directly
                                              (DP controller broadcasts to all DP ranks)
```

### 前置条件

支持的 SGLang 目标版本为 0.5.15.post1。Server 必须同时启用 `--sglang-enable-session-radix-cache` 和 `--sglang-radix-eviction-policy priority`；Relax 会在启动时校验这一组合。

::: warning
`--sglang-enable-session-radix-cache` 和 `--sglang-radix-eviction-policy` 不是 Relax 定义的参数。它们是 SGLang `ServerArgs` 通过 `--sglang-` 前缀自动暴露出来的（见 `relax/backends/sglang/arguments.py`），因此只有当你安装的 SGLang 提供这两个参数时它们才存在。SGLang 0.5.15.post1 中两者均存在。
:::

### 失败行为

Close 只是 KV 释放优化。失败不会阻塞逻辑终态；受影响的 session 继续由 SGLang 配置的 priority-aware radix-cache eviction 处理（同 priority 内按 LRU）。可以通过 `agentic_kv/session/close_failure` 观察 close fanout 失败。

### 配置项

| Flag | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--agentic-session-lifecycle` | flag | `False` | 开启该特性 |

## Program-Aware Admission

### 做了什么

在每个 request 边界上——获取 SGLang permit 和调用 `generate()` 之前——admission 使用一条全局 FIFO 队列与 token ledger：

| 动作 | 行为 |
|---|---|
| **Bypass** | Protected work、信号 degraded 或超过最大等待时间时，不持有 lease 并继续执行 |
| **Admit** | 持有一个集群级 execution-token lease，backend attempt 结束时释放 |
| **Wait** | 按 FIFO 顺序等待，并且不占用 fleet request permit |

无论怎样配置，两条不变式始终成立：

- Admission **绝不选择 worker**。最终 placement 仍由 SGLang router 决定。
- Admission **绝不中断在途 decode**。它只拦截尚未开始的工作。

### 架构

```
┌──────────────────────┐   acquire / release   ┌────────────────────────────────┐
│  AgenticSessionShard │ ────────────────────► │  AdmissionCoordinator          │
│  (16 Ray actors)     │ ◄──────────────────── │  single Ray actor, one writer  │
│                      │        lease          │  global FIFO + BudgetState     │
└──────────┬───────────┘                       └───────────────┬────────────────┘
           │                                                   │ poll /metrics
           │ fleet permit → generate                           ▼
           ▼                                   ┌────────────────────────────────┐
┌──────────────────────┐                       │        SGLang engines          │
│     sgl-router       │ ────────────────────► └────────────────────────────────┘
└──────────────────────┘
```

| 组件 | 职责 | 实现位置 |
|---|---|---|
| **决策逻辑** | 纯粹的 Admit/Bypass 策略与 token ledger，不含 Ray 和 I/O | `relax/agentic/session/admission.py` |
| **Coordinator** | 单写者 Ray actor、全局 FIFO、`/metrics` 轮询、lease TTL 回收 | `relax/agentic/session/admission_coordinator.py` |
| **Shard 集成** | 在 fleet permit 前以 cancellation-safe 方式获取 lease | `relax/agentic/session/service.py` |

`BudgetState` 与 Ray 解耦，容量、pressure、lease、TTL 和 usage accounting 都通过注入的 monotonic `now` 进行确定性的 CPU 测试。全局 FIFO、aging、cancellation 和 metrics polling 位于 Ray-backed `AdmissionCoordinator`；测试直接使用其底层类，无需启动 Ray（`tests/test_agentic_rollout.py`）。

### Reservation 的计算

一次 reservation = **processor 展开的 training prefix + 剩余 completion budget**。纯文本的 training prefix 与 backend prefix 长度自然相同。多模态容量记账使用 processor 展开的 training length，SGLang payload 继续使用 backend tokenizer IDs 与 media data。`--agentic-admission-expected-decode-cap` 可以收紧 completion 部分，不会改变实际 generation limit。

### 决策顺序

Admission 按以下顺序执行：

1. 特性未开启或 scope 未选中 → 不经过 admission，保持现有路径
2. Protected work → **Bypass**（`protected`）
3. Capacity snapshot 缺失或过期 → **Bypass**（`degraded`）
4. 存在更早的 waiter，或者达到 capacity 或 pressure limit → 在全局 FIFO 中 **Wait**
5. 达到最大等待时间 → **Bypass**（`aged`）
6. FIFO 为空且容量充足 → **Admit** 并持有 lease

以下两种 ledger 状态会让 request 排队：

| 拒绝原因 | ledger 判定条件 | 调用方行为 |
|---|---|---|
| `pressure_guard` | 最坏情况 engine `token_usage` 达到 pressure threshold | **Wait** |
| `capacity_exhausted` | `reserved + tokens` 超过 admission ceiling | **Wait** |

ceiling = `健康 engine 的 max_total_num_tokens 之和 × headroom`。

### Lease

Lease 按 request ticket 幂等，因此重试 acquire 不会重复扣减预算。Cancellation 会原子地移除 waiter，或释放竞态中已经授予的 lease。TTL 负责回收因 shard 死亡而搁浅的 lease，worker 集合变化时 ledger epoch 会递增。

### 防饥饿

Request 在唯一的 Coordinator 队列中按最早优先顺序等待。Lease release 与周期性 metric reconciliation 会推进该队列。Ledger degraded 时，队列中的 request 会 bypass。超过 `--agentic-admission-max-wait-s` 后，最早的 request 也会 bypass，保证 admission 不会无限期阻塞进度。

### 前置条件

coordinator 通过 SGLang router 发现 engine（`/workers`，失败则回退 `/list_workers`），并抓取每个 engine 的 Prometheus `/metrics`。如果 router 地址未设置或不可达，就没有快照，ledger 会报告 `degraded`，所有请求都走 bypass。

::: tip
由于 coordinator 读取的是 engine 侧的 gauge，TP 复制会造成影响。SGLang 会为每个 rank 各发一份 `sglang:max_total_num_tokens` 之类的指标，且值完全相同，因此 Relax 用 **max** 而非 sum 聚合。若用 sum，容量会被放大、usage 会被缩小，倍数都是 TP 度，结果就是一个已经打满的 engine 看起来几乎空闲。
:::

### 配置项

| Flag | 类型 | 默认值 | 约束 | 说明 |
|---|---|---|---|---|
| `--agentic-program-admission` | flag | `False` | — | 开启该特性 |
| `--agentic-admission-headroom` | float | `0.90` | `(0, 1]` | 可用作 ceiling 的 KV 总容量比例 |
| `--agentic-admission-pressure-threshold` | float | `0.92` | `(0, 1]` | 单 worker `token_usage` 达到该值时新 request 等待 |
| `--agentic-admission-expected-decode-cap` | int | `--rollout-max-response-len` | `> 0` | 单次 reservation 的 decode token 上界 |
| `--agentic-admission-max-wait-s` | float | `30.0` | `>= 0` | Aging bypass 前的最大 FIFO wait |
| `--agentic-admission-scope` | str | `train` | `train` \| `all` | 只作用于 train，还是 train + eval |

上述数值约束仅在设置了 `--agentic-program-admission` 时才会校验。

## 快速开始

在已有的 agentic rollout 启动脚本中加入：

```bash
AGENTIC_ARGS=(
   --use-agentic-rollout
   # ... 已有的 agent flags ...

   # Session KV lifecycle：依赖 server 侧的 session radix cache
   --sglang-enable-session-radix-cache
   --sglang-radix-eviction-policy priority
   --agentic-session-lifecycle

   # Program-aware admission：默认值是一个合理的起点
   --agentic-program-admission
   --agentic-admission-headroom 0.90
   --agentic-admission-pressure-threshold 0.92
)
```

完整示例见 `examples/mini_swe_agent/run_mini_swe_agent.sh`。

## 监控指标

两个特性都按 rollout step 上报一次，与已有的 `rollout/` 和 `perf/` 指标并列。所有 series 共用 `agentic_kv/` 前缀，因此按首段路径分组的 tracker——ClearML 会以第一个 `/` 把 key 拆成 `(title, series)`——会把它们渲染成一个面板，而不是三个。

| 指标 | 含义 |
|---|---|
| `agentic_kv/session/lifecycle_enabled` | 开启 session lifecycle 时为 `1.0`，否则不上报 |
| `agentic_kv/session/close` / `close_failure` | Session close 尝试与失败次数 |
| `agentic_kv/admission/admit` / `bypass` | 每 step 的 admission outcome；没有对应 outcome 时不上报 |
| `agentic_kv/admission/wait` / `waiting` / `cancelled` | 入队、当前等待与已取消的 request 数 |
| `agentic_kv/admission/bypass_protected` / `bypass_degraded` / `bypass_aged` | Fail-open bypass 原因 |
| `agentic_kv/admission/defer_rate` | `wait / (admit + wait + bypass)`；请求因准入控制而进入 FIFO 等待队列的比例 |
| `agentic_kv/admission/degraded_rate` | `bypass_degraded / (admit + wait + bypass)`；容量信号不可用或过期时，准入控制直接放行请求的比例 |
| `agentic_kv/admission/wait_seconds_mean` | 排队后获得 lease 的 request 平均等待时间 |
| `agentic_kv/budget/ceiling` / `reserved` / `available_tokens` | Admission ceiling 与 token ledger 状态 |
| `agentic_kv/budget/reserved_utilization` | Reserved token 数占 admission ceiling 的比例 |
| `agentic_kv/budget/lease_count` / `lease_expired` | 当前 lease 与 TTL 回收的 lease 数 |
| `agentic_kv/budget/kv_token_usage_mean` / `kv_token_usage_max` | 窗口内 engine KV token usage 的均值与峰值 |
| `agentic_kv/budget/epoch` / `degraded` | Worker-set generation 与 capacity snapshot 健康状态 |

::: warning
`agentic_kv/budget/kv_token_usage_*` 采样自一个每 step 排空一次的滑动窗口，因为在打日志那一刻做瞬时读取时 rollout 已经排空，会低估真实峰值。Engine 侧的释放收益——pool 大小、强制 eviction、释放的 token 数——不在此表中，需要从 engine 自身的 Prometheus `/metrics` 读取。
:::

## 故障排除

| 现象 | 可能原因 | 检查项 |
|---|---|---|
| 全部走 bypass，`agentic_kv/admission/bypass_degraded` 持续增加 | ledger degraded | 检查 `agentic_kv/budget/degraded == 1.0`。router 地址未配置或不可达，或 worker `/metrics` 抓取失败。 |
| `agentic_kv/admission/waiting` 长期较高 | Capacity 或 pressure bound | 检查 reservation size、concurrency、headroom 与 engine KV usage。 |
| `agentic_kv/admission/bypass_aged` 持续增加 | Request 经常达到 aging deadline | 放宽 headroom、降低并发，或降低 expected decode cap。 |
| `agentic_kv/budget/reserved_utilization` 长期接近 `1.0` | 确实是容量受限 | 调高 headroom 或减少并发 session。 |
| 开了 session lifecycle 但 KV 不下降 | server 侧 cache 未开启 | 确认 engine 启动时带了 `--sglang-enable-session-radix-cache`。 |
| `agentic_kv/session/close_failure` 增加 | Worker discovery 或 close fanout 失败 | 检查 router discovery 与直接 `/close_session` 连通性。 |

## 下一步

- 阅读 [Agentic Rollout](./agentic-rollout.md) 了解这两个特性所挂载的 session 生命周期。
- 阅读 [性能调优](./performance-tuning.md) 了解更完整的 rollout 吞吐检查清单。
- 阅读 [OOM 排查](./oom-troubleshooting.md) 了解 KV 压力演变成 OOM 时的处理方式。
