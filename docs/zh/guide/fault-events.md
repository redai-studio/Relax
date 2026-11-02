# 故障事件历史

故障历史由训练 driver 持有，使用有界内存，独立于 MetricsService 和性能时间线。同一 driver 内重建 Ray 或 Controller 时可保留同一份历史；driver 退出后不持久化。

本批实现事件模型、存储、异步接纳、只读查询和 driver 生命周期。HealthStatus／Controller 自动采集与操作完成回执在后续批次接入。因此，仅启用本批代码还不会产生完整的故障恢复时间线。

## 启用查询

HTTP 端点要求 Uvicorn 0.36.0 或更新版本，已在 `requirements.txt` 声明。更早版本无法加载用于连接期限控制的自定义事件循环。正式验收环境使用 Uvicorn 0.54.0。

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

内部生产者通过 `FaultEventRuntime.record(draft)` 异步接纳。返回 true 只说明 FIFO 已接纳，不说明 Store 已提交。允许重试时应复用稳定 event ID。Store 的 ID 去重范围与正文保留范围相同；后续接入的生命周期发布状态负责防止活动故障在正文淘汰后复活，不维护无限增长的历史 ID 表。

仍在队列或在途中的相同 ID、相同内容共用一个接纳槽；同 ID 的冲突内容被拒绝。消费者最终放弃投递前核对保留历史，避免把已经提交的重复事件算作丢失；无法确定提交状态时只报告 unknown，不猜测丢失数量。

事件只允许 ID、固定服务角色、事件／原因代码、序号／时间和证据掩码。查询记录显式包含全部字段，缺少上下文时使用 JSON null。拒绝未知字段、bool 等非原生整数、非法组合和超长记录。不保存异常正文、traceback、URL、样本、模型、配置或环境变量。字典输入可使用 `EventDraft.from_dict`，校验失败只返回固定错误。

## 容量、异常与退出

历史和 ID 索引各最多 4,096 项，单条最终 compact UTF-8 JSON 最多 1,024 字节。Runtime 最多保留 512 条队列与 1 条在途。序号与诊断计数使用无符号 64 位边界，不回绕。时间只用于展示，排序以 sequence 为准。

入口等待接纳锁最多 5 ms，不调用 Store、HTTP 或 Ray；Store 取锁预算为 50 ms。这些是锁等待预算，不是整个调用耗时的硬实时保证。唯一消费者对确认未提交的繁忙结果仅用同一 ID 重试一次。提交结果不确定时停用原 Store 的追加和查询并报告 unknown，不会悄悄换成空历史。

HTTP 使用一个服务线程、四个活跃业务槽、有限 backlog／库级并发、2 KiB 请求目标、1 KiB 游标和 128 KiB 响应上限。槽位一直保留到发送及取消清理结束；连接生命周期、请求处理与发送使用三秒期限。关闭 keep-alive，并限制 h11 未完成消息缓冲。这些控制面向受信 loopback，不承诺操作系统 TCP 连接数或整个进程 RSS 的固定上限。

driver 退出时向 Runtime 和 HTTP 发出关闭信号，不等待队列排空，不新增 Ray 等待。硬退出可能丢失末尾记录；缺少记录或完成回执不能被解释为业务失败。

## CPU 验证

在仓库根目录、装有既有测试及 HTTP 依赖的环境执行：

```bash
PYTHONPATH=. pytest -q \
  tests/utils/test_fault_event_store.py \
  tests/utils/test_fault_event_runtime.py \
  tests/utils/test_fault_query_server.py \
  tests/entrypoints/test_fault_events_lifecycle.py
pre-commit run --all-files
```

这些用例执行实际 Store／Runtime 和真实 loopback socket。入口测试运行生产入口代码，对模型／Ray 初始化和进程退出提供明确替身。它们验证本批基础契约和生命周期隔离，不代表 GPU 数值、完整训练或后续恢复接入已验收。既有健康线程回归作为前置检查保留，其修复已经存在于 main。
