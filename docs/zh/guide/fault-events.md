# 故障事件历史

故障历史由训练 driver 持有，使用有界内存，独立于 MetricsService 和性能时间线。同一 driver 内重建 Ray 或 Controller 时可保留同一份历史；driver 退出后不持久化。

HealthStatus 通过既有健康查询提供稳定故障身份和完成回执，Controller 自动记录处理选择、局部和全局恢复操作、未注册角色跳过、fatal 终止和全局重启预算耗尽。仅有 `handling_started` 或 `recovery_handoff` 不能证明恢复成功。

## 启用查询

HTTP 端点要求 Uvicorn 0.36.0 或更新版本，已在 `requirements.txt` 声明。更早版本无法加载用于连接期限控制的自定义事件循环。

在已有训练命令中增加 `--use-health-check --fault-events-enabled`。功能默认关闭；只开启故障历史而未开启健康检查时，记录固定提示并保持采集关闭。

HTTP 成功启动后，driver 日志输出查询地址和 `run_id`。`--fault-events-port` 接受 0～65535，默认 0 由系统分配端口。地址仅绑定 `127.0.0.1`。端口被占用或 HTTP 启动失败时，内存记录仍可工作，训练保留原有行为；纯 HTTP 故障不表示历史已经丢失。

在 driver 所在机器查询，将 PORT 换成日志中的端口：

```bash
curl 'http://127.0.0.1:PORT/fault-events/v1/status'
curl 'http://127.0.0.1:PORT/fault-events/v1/events?role=actor&limit=50'
```

远程查看时使用已有 SSH 转发。run ID 和游标只是标识，不承担身份认证。

## 正确读取分页

`GET /fault-events/v1/events` 接受 `role`、`fault_id`、`limit` 和 `cursor`。角色与故障筛选取交集；每页默认 50 条、最多 100 条。未知参数、重复参数和非法值均被拒绝。

第一页固定 `snapshot_end_seq`。后续携带 `next_cursor` 和相同筛选条件，直到 `done=true`；允许调整每页条数。期间新产生的事件不进入本次快照，需要另起一次查询。每页最多扫描 512 个原始序号位置，因此空的筛选页也可能需要继续翻页。

快照只固定序号上界，不锁住保留区间。例如，快照上界为 5，第一页读到 1～2，随后追加事件使序号 3 被淘汰。续页返回 4～5，并报告 `missing_range: [3, 3]`，不会混入新序号。若全部未读区间已被淘汰，最后一页可以为空，但仍报告缺口。

这些字段需要分别理解：

| 字段 | 含义 |
| --- | --- |
| `retention_truncated` | 本次运行曾因容量限制淘汰已提交记录。 |
| `missing_range` | 本页全局序号区间中尚未读取、现已缺失的位置。 |
| `gap_scope: global` | 缺口在筛选前计算。 |
| `filter_match_unknown` | 有缺口且使用了筛选，无法推算缺失记录中有多少匹配项。 |
| `admission_rejections` | 本地接纳被拒绝的次数，也包括临时满队列或繁忙。 |
| `known_lost_events` | 已确认未提交且最终被放弃的事件数；临时拒绝不立即等于最终丢失。 |
| `coverage_unknown`、`unknown_reasons` | 观测或提交不确定的固定诊断；false 不证明分布式系统绝对完整。 |

`GET /fault-events/v1/status` 返回 run ID、历史容量／条数／最高序号、队列容量与使用量、一个在途槽、`query_available` 及上述诊断，不返回配置、原始错误和训练数据。

应用错误使用固定 JSON：422 `INVALID_QUERY`、409 `RUN_MISMATCH`、503 `UNAVAILABLE`、405 `METHOD_NOT_ALLOWED`、404 `NOT_FOUND`。进入应用前，Uvicorn 也可能对非法 HTTP 或并发超限返回固定的 400／503 文本，客户端应先检查状态码和 Content-Type。不提供写接口、请求体、内容压缩或自动 API 文档端点。

## 局部恢复事件

先按角色查询，再使用返回的 `fault_id` 筛选。在没有投递丢失时，一次成功的 local 恢复具有以下序列：

```text
fault_detected
handling_started    expected=7 observed=0 failed=0
recovery_handoff    expected=7 observed=7 failed=0
recovery_succeeded  expected=7 observed=7 failed=0
```

证据位分别为重新部署（1）、恢复 step（2）、提交任务并保存引用（4）。local 始终要求三项，包括 debug 模式。handoff 表示控制流交接；成功表示这些恢复操作完成，不证明模型权重数值正确或训练已经完成一个 step。

如果原 Service 捕获 `/set_step` 异常后继续执行，时间线仍记录 `recovery_failed`，其中 `reason_code=step_restore_failed`、`observed_evidence=1`、`failed_evidence=2`。之后提交任务或 `mark_healthy` 不会反转这个终态。观测证据缺失时保留未验证结论并设置 `OUTCOME_UNOBSERVED`，不猜测成功或业务失败。

同一个连续 unhealthy 周期保持同一发现 ID（D）和首次原因。非 fatal 后升级为 fatal 时只生成一个升级 ID（U）；首报 fatal 只需要 D。重复健康查询对尚未接纳的 D/U 每轮每 ID 最多尝试一次。健康清除后，缓存 ID 仍可补发，但不会再次触发恢复。替换或退休来源时，对每个未接纳的 D/U 只结算一次；已接纳项由消费者负责。例如 D 已接纳、U 被拒绝后退休，`known_lost_events` 增加 1；两者都拒绝则增加 2；成功重试则不增加。活动发布状态也会防止已淘汰正文的 D 再次出现。

未注册角色产生 `handling_skipped`；Controller 仍将缺少恢复处理器视为错误，停止健康检查并取消待处理任务。fatal 沿用原回调及退出策略。全局成员共享 recovery ID。同一故障在后续健康轮次仍不健康时，可以用新的 recovery ID 重试。

## 全局恢复与完成回执

重建前冻结角色图及决定恢复操作的配置分支。handoff 原子地比较重建后的计划、合并本地证据并选出终态。即使证据掩码相同，不同角色图也会使比较失败。普通 checkpoint 路径、router 地址和 step 值可以改变。

| 全局启动分支 | 必需证据 |
| --- | --- |
| Colocate 或 hybrid，存在 actor | 初始化（1）、全部服务恢复 step（2）、提交任务（4）：7。 |
| 纯 fully async，只有 actor | 上述操作加 actor 发送（8）：15。 |
| 纯 fully async，actor + actor_fwd | 再加 actor_fwd 接收（16）：31。 |
| 纯 fully async，actor + reference | 再加 reference 接收（32）：47。 |
| 纯 fully async，三者都在 | 两项接收均要求：63。 |
| 没有 actor，或任一全局 debug 开关 | 初始化和提交任务：5。 |

local 在 debug 模式仍为 7。colocate/hybrid 中，仅当 actor 和 rollout 都存在、且执行原有非 offline 分支时，rollout-manager setter 才等待初次同步；不会为跳过的分支虚构权重回执。任务证据要求每个服务都有保存的引用，部分或空任务集不能证明提交，也不证明训练已经结束。

纯 async 恢复由 Actor/ActorFwd 在实际 worker-group 调用返回或抛错后报告。Service 等待实际远程响应，Controller 使用 `asyncio.gather` 并发启动发送和接收，之后才恢复 step。新 HealthStatus 在部署前已获得固定回执 token。既有 unhealthy 查询在回调前合并三个回执槽，unhealthy 列表为空时也合并。Runtime 不保留训练响应，不新增 driver 轮询 RPC。

例如 actor + actor_fwd 的 handoff 可能记录：

```json
{"event_type":"recovery_handoff","expected_evidence":31,"observed_evidence":7,"failed_evidence":0,"availability":"unverified"}
```

两项回执到达后，新建 `recovery_succeeded` 记录，expected=observed=31；旧记录保持不变。明确的操作失败产生 `recovery_failed/operation_failed` 及失败证据位。远程响应失败时，操作证据在完成回执到达前保持未验证；原始异常仍传播给训练。缺失回执保持未验证，时间流逝不会生成终态或增加 `known_lost_events`。冲突槽既不能证明成功也不能证明失败，但不能掩盖另一个操作的明确失败。`active_recoveries` 统计本次运行中的未决上下文，不提供逐 recovery 的 pending 状态接口。

fault source 和回执接收端 generation 相互独立。G0 的故障启动面向 G1 的恢复，销毁 G0 不会中断该恢复；仅启动下一次恢复也不会结束它。只有实际退休 G1，才以 `source_retired` 中断其未决结果。source 观测失败不丢弃独立的回执槽。handoff 前被放弃的启动记录 `superseded`，旧 token 不能影响新尝试。成功终态不会反转，之后的相反回执只设置 `RECEIPT_CONFLICT`。`SOURCE_RETIRED` 可以与已验证成功同时存在。

## Python 存储接口

下例不启动 Ray 或模型，可同步验证存储契约：

```python
from relax.utils.fault_events import EventDraft, FaultEventStore
from relax.utils.fault_events.models import new_id

store = FaultEventStore(new_id())
draft = EventDraft(
    run_id=store.run_id,
    fault_id=new_id(),
    event_id=new_id(),
    role="actor",
    source_generation=0,
    event_type="fault_detected",
    reason_code="reported_error",
)
assert store.append(draft) == "accepted"
assert store.append(draft) == "duplicate"
page = store.query(role="actor")
assert len(page["events"]) == 1
```

内部生产者通过 `FaultEventRuntime.record(draft)` 异步接纳。返回 true 只说明 FIFO 已接纳，不说明 Store 已提交。允许重试时应复用稳定 event ID。Store 的 ID 去重范围与正文保留范围相同；已接入的健康生命周期发布状态负责防止活动故障在正文淘汰后复活，不维护无限增长的历史 ID 表。

仍在队列或在途中的相同 ID、相同内容共用一个接纳槽；同 ID 的冲突内容被拒绝。消费者最终放弃投递前核对保留历史，避免把已经提交的重复事件算作丢失；无法确定提交状态时只报告 unknown，不猜测丢失数量。

事件只允许 ID、固定服务角色、事件／原因代码、序号／时间和证据掩码。查询记录显式包含全部字段，缺少上下文时使用 JSON null。拒绝未知字段、bool 等非原生整数、非法组合和超长记录。不保存异常正文、traceback、URL、样本、模型、配置或环境变量。字典输入可使用 `EventDraft.from_dict`，校验失败只返回固定错误。

## 通过 OpenTelemetry 导出故障日志

故障日志导出保持现有事件 schema、本地查询和恢复判据不变。不安装全局 OTel provider，不接管普通 Python 日志，也不启用其他观测信号。

driver 统一持有一个官方 SDK 的私有 `LoggerProvider`，显式资源属性为 `service.name=relax`，并使用官方 OTLP HTTP exporter。只有 Store 新接纳并提交成功的事件，才将不可变的实际记录快照交给独立有界导出队列：最多 512 条排队、1 条在途。`offer` 在 Runtime 消费者线程上、Store 和接纳锁之外构造 SDK `LogRecord`；仅由独立导出线程执行 SDK emit、`SimpleLogRecordProcessor` 及导出／网络 I/O，也不再增加 SDK 批处理队列。

依赖 Python 3.10 及更新版本的 `opentelemetry-sdk>=1.45.0,<1.46` 与 `opentelemetry-exporter-otlp-proto-http>=1.45.0,<1.46`，并在同一范围声明 `opentelemetry-exporter-otlp` 汇总包，使镜像中预装的 gRPC exporter 与 SDK 保持同一版本；导出本身只走 HTTP。约束在 1.45 系列，是为了保留 exporter 自有客户端的诊断接口，用于屏蔽可能包含敏感信息的传输错误细节；不修改全局 logger。若这一兼容契约不再满足，则拒绝启动导出，本地历史继续工作。只在启用故障日志导出时创建私有 provider 和 exporter。

导出要求同时启用故障历史的两个开关，并显式配置接收端。在已有本地 OTLP HTTP 接收端的情况下，先为 driver 设置环境，再运行既有训练命令：

```bash
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://127.0.0.1:4318/v1/logs
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL=http/protobuf
# 为既有训练命令增加 --use-health-check --fault-events-enabled。
```

日志专用 endpoint 是完整 URL；也可设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 基础地址，由导出端追加 `/v1/logs`。本接口仅支持 `http/protobuf`，遵循 `OTEL_SDK_DISABLED`，请求头和超时使用官方 exporter 的标准日志专用／通用环境变量。请求头必须符合 OTLP 环境变量语法；原始控制字符或分隔符会在 SDK 解析前被拒绝，URL 编码后的值可以使用。为避免可选认证代码阻塞训练启动，本故障专用接口拒绝 credential-provider entry point。不使用 exporter 隐式的 localhost 默认地址。未配置 endpoint 时不启动导出线程，本地历史和查询继续工作；配置或导出异常不改变本地提交结果。被拒绝的配置只记录一条警告，写明设置项和约束，例如 `INVALID_OTEL_CONFIG: headers must use the SDK name=value grammar`，不包含配置值。

每条记录的原生 `event_name` 和 `body` 均为 `relax.fault.<event_type>`。本地记录全部 17 个字段都有明确的白名单属性，不附加任意字典、异常、环境、配置或训练内容：

| 本地字段 | OTel 属性 | 值 |
| --- | --- | --- |
| `schema_version` | `relax.fault.schema_version` | 整数 |
| `run_id` | `relax.fault.run_id` | 32 位小写十六进制字符串 |
| `fault_id` | `relax.fault.fault_id` | 32 位小写十六进制字符串 |
| `event_id` | `relax.fault.event_id` | 32 位小写十六进制字符串 |
| `source_generation` | `relax.fault.source_generation` | 十进制字符串 |
| `role` | `relax.fault.role` | 固定角色字符串 |
| `event_type` | `relax.fault.event_type` | 固定事件字符串 |
| `reason_code` | `relax.fault.reason_code` | 固定原因字符串；null 时省略 |
| `recovery_id` | `relax.fault.recovery_id` | 32 位小写十六进制字符串；null 时省略 |
| `recovery_mode` | `relax.fault.recovery_mode` | `local` 或 `global`；null 时省略 |
| `expected_evidence` | `relax.fault.expected_evidence` | 整数位掩码 |
| `observed_evidence` | `relax.fault.observed_evidence` | 整数位掩码 |
| `failed_evidence` | `relax.fault.failed_evidence` | 整数位掩码 |
| `sequence` | `relax.fault.sequence` | 十进制字符串 |
| `recorded_at_ms` | `relax.fault.recorded_at_ms` | 十进制字符串 |
| `result_scope` | `relax.fault.result_scope` | 固定范围字符串；null 时省略 |
| `availability` | `relax.fault.availability` | 固定可用性字符串 |

原生严重程度也采用固定映射：`handling_started`、`recovery_handoff`、`recovery_succeeded` 为 `INFO`；`fault_detected`、`recovery_interrupted`、`handling_skipped` 为 `WARN`；`fatal_escalated`、`recovery_failed`、`termination_requested` 为 `ERROR`。

十进制字符串保留完整无符号 64 位数值，避免超过 OTLP 有符号整数范围。原生时间戳取 `recorded_at_ms * 1_000_000`；若超出 OTLP 无符号 64 位纳秒范围，则不设置该时间戳，精确值仍保留在属性中。严重程度由事件类型／原因固定映射，不接收任意文本。每个发现、fatal 升级或恢复事件在产生位置捕获有效的 trace ID、span ID 和 trace flags，随健康／Ray 旁路上下文和异步队列传递。旁路上下文每个角色最多保留两个，并在角色来源清除时清理。没有有效上下文则保留为空，且不改变本地健康状态或历史；不复制 baggage，不创建 span，不将业务 ID 当作 trace ID。该旁路上下文不改变本地业务记录 schema。

本地重复提交不产生第二次导出；网络重试仍可能让接收端多次收到相同的稳定 `event_id`，可用该 ID 关联。队列满、传输失败、关闭时放弃使用独立 `otel_*` 诊断，不增加本地 `known_lost_events`，也不停用 Store。传输成功不证明接收端已经持久化；官方 exporter 的成功返回尤其不能区分接收端通过 OTLP `partial_success` 拒绝的记录，因此成功尝试不等于逐条持久化回执。

关闭先停止导出接纳，最多等待导出线程 50 ms。`otel_shutdown_dropped` 只统计关闭时放弃的排队记录，不把在途请求计为已丢弃：50 ms 等待结束后，在途结果仍为未知，直到工作线程返回，再按实际结果更新传输计数。未决时不能声称失败或已交付；进程硬退出仍可能丢失它。不因此延长本地 Runtime 的队列排空策略，也不等待 Ray。

全局恢复在 Controller 重建前后复用同一个 driver 持有的 provider、已提交记录映射和导出队列。回执和 trace 旁路上下文共用既有健康数据包；完成证据不会创建第二个 exporter。

检查接收端时，使用其文档提供的健康检查命令或 `curl`。向 `/v1/logs` 发送普通 JSON 不是有效的检查方式，因为本接口发送的是二进制 `http/protobuf`。

## 容量、异常与退出

历史和 ID 索引各最多 4,096 项，单条最终 compact UTF-8 JSON 最多 1,024 字节。Runtime 最多保留 512 条队列与 1 条在途、18 个发布槽、9 个健康快照成员及 2 个恢复上下文。序号与诊断计数使用无符号 64 位边界，不回绕。时间只用于展示，排序以 sequence 为准。

入口等待接纳锁最多 5 ms，不调用 Store、HTTP 或 Ray；Store 取锁预算为 50 ms。这些是锁等待预算，不是整个调用耗时的硬实时保证。唯一消费者对确认未提交的繁忙结果仅用同一 ID 重试一次。提交结果不确定时停用原 Store 的追加和查询并报告 unknown，不会悄悄换成空历史。

HTTP 使用一个服务线程、四个活跃业务槽、有限 backlog／库级并发、2 KiB 请求目标、1 KiB 游标和 128 KiB 响应上限。槽位一直保留到发送及取消清理结束；连接生命周期、请求处理与发送使用三秒期限。关闭 keep-alive，并限制 h11 未完成消息缓冲。这些控制面向受信 loopback，不承诺操作系统 TCP 连接数或整个进程 RSS 的固定上限。

每个相关副本懒创建一个回执 sender，最多八条排队、一条在途，单条编码回执最多 256 字节。业务方法只尝试本地入队。确认缓慢时持续占用同一槽，不启动替代请求或线程。普通调用和关闭采集时不启动 sender。接收端丢失或队列饱和可能使证据保持未验证，但不改变训练结果。

driver 退出时向 Runtime 和 HTTP 发出关闭信号，不等待队列排空，不新增 Ray 等待。硬退出可能丢失末尾记录；缺少记录或完成回执不能被解释为业务失败。
