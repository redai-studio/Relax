# Agentic Session 轨迹导出 CPU 性能报告（Task 14）

基线：Relax `ea3f0cee94b9400cfe1591beb4f2dbfe712d362c`。测试日期：2026-10-30。
对应 [official task 14](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-14)。

## 结论与实现

长轨迹导出会在 SessionShard 的事件循环上同步执行 `SessionForest.build_sample()`、token 解码和
`TrainingFieldArtifact.from_sample()`。单个请求内的列表遍历、深拷贝和 NumPy 转换期间，其他 Session
即使有空闲后端，也无法得到事件循环调度。

本次将完整的 `_build_session_export_transport()` 放入每个 Shard 的单个专用线程：

- 提交 executor 前先取得一个 semaphore 槽位，实际执行和已提交的导出合计最多一个；等待者保留已有
  Session，不复制整份 Forest，不向 executor 的无界队列累计重任务。
- 从 `ACTIVE` 切换到 `FINALIZING` 后，整个排队和组装过程继续持有该 Session 的锁。Forest、输出元数据
  和迟到的响应 trace 更新因此仍按原有锁顺序串行化；其他 Session 使用各自的锁。
- 原有 finalization 的 `finish_before_cancellation()` 边界移入 finalizer，使直接调用也具有相同的取消语义。
  已接受的终态导出会完成，再转发取消；取消 asyncio Future 不会提前释放仍在执行的线程槽位。
- drop 和 shutdown 通过原有 `finish_task` 只清理和发布一次。服务先收尾所有 Group，再关闭导出 worker，
  最后关闭后端；executor 仅在已无运行作业时 join。
- 保留原有训练字段组装、拷贝隔离、错误映射、API 和 Ray 资源配置。

采用持锁的终态所有权边界，是因为仅检查 `FINALIZING` 不足以冻结状态：
`mark_chat_service_response_ready()` 仍可在锁内更新响应节点。跨线程释放锁后直接读取原 Forest 会引入竞态。

## CPU 瓶颈证据

执行：

```bash
python -m scripts.benchmarks.benchmark_session_export \
  --profile --mode inline --tokens 131072 --exports 1
```

[profile.txt](./profile.txt) 保存该次 cProfile 输出。128K token、8 轮样本中，整个组装调用累计约
1.013 秒；artifact 转换累计约 0.980 秒，其中 `deepcopy` 累计约 0.966 秒，占约 95%。
这包含数十万次 Python 对象访问，组装完成后还需将列表转为数组。
cProfile 会显著放大 Python 调用开销，以上时间仅用于定位热点，不用于计算性能提升。
本次选择有界后台执行以保留既有数据所有权契约；没有将省略训练字段或共享可变列表作为优化手段。

参考 [miles #3736](https://github.com/radixark/miles/pull/3736) 的单 worker 与取消收尾思路，
以及 [miles #4044](https://github.com/radixark/miles/pull/4044) 的真实组装、模拟推理后端 CPU replay 方法。
Relax 的终态 Session 不需要为继续生成保留可写 Forest，因此这里沿用 Session 锁即可保持一致性。

## 可重复运行的负载与统计口径

从仓库根目录运行，使用已有的 Relax CPU 测试环境；工具需要 NumPy、PyTorch、Ray、FastAPI、httpx、
Transformers 等现有运行依赖，不增加项目依赖，不下载模型，不启动 Ray 集群或 GPU 推理服务：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m scripts.benchmarks.benchmark_session_export \
  --json-out docs/benchmarks/session-export/results.json
```

该命令覆盖 8,192 / 131,072 token × 1 / 8 个并发 finalizer，每种组合各跑三对进程。
每对均导出 16 个完整 Session。`inline` 使用原始同步调用方式，`thread` 使用生产 worker，二者调用相同且
未改动的真实组装逻辑。运行顺序交替为 inline/thread、thread/inline、inline/thread。

- 每个变体启动独立 Python 子进程；每次先用同长度样本完成一次完整导出，预热 NumPy、解码和 worker。
- 模拟后端异步返回确定性的 tokens/logprobs，经真实 `append_obs()` / `append_resp()` 提交到 Forest。
  每条轨迹 8 轮，每轮有 observation 和 response，长度包含二者；短请求使用另一个常驻 Session。
- 输入构造在计时前完成。测量起点后 50 ms，同时发出全部 16 个导出请求；并发限流等待、worker 等待、
  组装、正常 Session 清理和结果发布全部计入每个导出请求延迟。全部请求完成后才结束导出计时，吞吐为
  `16 / (最后完成时间 - 统一发送截止时间)`。
- 独立 ticker 与短请求均每 5 ms 一个计划截止时间。短请求调用真实
  `mark_chat_service_response_ready()`，包含 Session 查找、锁获取及 trace 更新。
  这是进程内 Session RPC 方法 replay，测量不包含 HTTP/Ray 传输或完整 chat 推理。
- 事件循环延迟为每个 ticker 实际执行时间减计划截止时间；短请求延迟为完成时间减计划发送时间。
  即使生产请求的协程本身被事件循环阻塞，等待时间也计入，避免 coordinated omission。
  导出完成后补齐所有过期截止时间，不跳过积压短请求。
- 主矩阵的计划窗口为 `max(1 秒, 最后一次导出完成时刻)`，随后排空探针。JSON 记录每次窗口、计划短请求
  数量及完成数量。固定窗口的丰富字段示例见下节。
- P99 使用最近秩法：排序后取第 `ceil(0.99*N)` 个样本，不插值；先求每次运行 P99，再取三次 P99 的中位数。
  表中的范围是三次独立运行的最小值和最大值，不把不同运行的样本混成一个分布。
- CPU 为测量窗口内的 `time.process_time()` 差值，包括主线程和导出线程；峰值内存为同一进程的 Linux
  `ru_maxrss`，在哈希校验前记录。该高水位包含导入、预热和已准备好的全部输入，不能解释为纯导出增量内存。
- 哈希及重建一致性验证在计时后进行。每个 payload 和重建后的 Sample 都逐字段散列，数组包括 dtype、shape
  和全部字节，仅忽略不确定的 trace `events` 时间。测试另用固定时钟精确比较包括 events 的全部字段。
  导出失败、完成数量不足、内容不一致、探针漏发或关闭后残留导出线程均使工具失败退出。

硬件：Intel Xeon Gold 6226R @ 2.90 GHz，Linux x86_64，64 个逻辑 CPU 可用，未绑单核。
Python 3.12.13，NumPy 2.4.3，PyTorch 2.6.0，Ray 2.58.0；默认 Python GIL switch interval 5 ms。
这是共享开发机上的串行变体测量，未隔离其他用户负载，故提供全部重复值。

## 主矩阵结果

[results.json](./results.json) 是 24 次运行的原始汇总，含每次分位数、工作量、CPU、内存及内容哈希。
以下均为三次中位数，时间单位 ms。

| Tokens / 并发 | 事件循环 P99，前 → 后 | 降低 | 短请求 P99，前 → 后 | 导出请求 P99，前 → 后 |
|---|---:|---:|---:|---:|
| 8K / 1 | 15.50 → 9.03 | 41.72% | 15.53 → 9.10 | 270.15 → 324.58 |
| 8K / 8 | 235.06 → 6.21 | 97.36% | 235.07 → 6.20 | 291.60 → 345.11 |
| 128K / 1 | 3695.11 → 11.69 | 99.68% | 3695.11 → 11.66 | 4485.09 → 4331.36 |
| 128K / 8 | 4017.63 → 13.92 | 99.65% | 4017.63 → 13.93 | 4102.62 → 4476.87 |

| Tokens / 并发 | 完成导出，前 / 后 | 导出总耗时 s，前 → 后 | 导出吞吐 /s，前 → 后 | CPU s，前 → 后 | 峰值 RSS MiB，前 → 后 |
|---|---:|---:|---:|---:|---:|
| 8K / 1 | 16 / 16 | 0.270 → 0.325 | 59.22 → 49.29 | 0.290 → 0.349 | 540.48 → 541.91 |
| 8K / 8 | 16 / 16 | 0.292 → 0.345 | 54.86 → 46.36 | 0.311 → 0.361 | 541.28 → 542.09 |
| 128K / 1 | 16 / 16 | 4.485 → 4.331 | 3.57 → 3.69 | 4.495 → 4.353 | 685.77 → 687.49 |
| 128K / 8 | 16 / 16 | 4.103 → 4.477 | 3.90 → 3.57 | 4.113 → 4.501 | 698.40 → 688.14 |

| Tokens / 并发 | 事件循环 P99 范围，前 | 事件循环 P99 范围，后 | 短请求 P99 范围，前 | 短请求 P99 范围，后 |
|---|---:|---:|---:|---:|
| 8K / 1 | 15.00–25.87 | 6.13–11.04 | 14.99–25.89 | 6.10–11.11 |
| 8K / 8 | 218.56–270.78 | 6.14–11.00 | 218.56–270.79 | 6.13–11.07 |
| 128K / 1 | 3192.92–3753.11 | 11.19–12.47 | 3192.93–3753.11 | 11.24–12.52 |
| 128K / 8 | 3949.05–4090.46 | 12.38–20.81 | 3949.06–4090.46 | 12.35–20.87 |

每个长轨迹变体导出 2,097,152 tokens，每个短轨迹变体导出 131,072 tokens；所有重复运行数量相同且哈希一致。
长轨迹的两种并发配置均超过事件循环 P99 降低 50% 的验收线。8K / 1 的收益较小，且线程调度会增加短任务
导出耗时；本次优化目标是 Session 响应性，不承诺降低总 CPU 或普遍提高导出吞吐。

为核对 replay 基线，另在原始 `ea3f0ce` worktree 仅复制 benchmark 和 worker 定义文件后，以 `--child --mode inline`
运行一次 128K / 8 / 16 exports / 6 秒窗口；原始 service 完全不使用 worker，走原有 finalization。
[reference.json](./reference.json) 保存结果：16 次导出、1,200 个短请求全部完成，事件循环 P99 4044.89 ms。
其全部导出与重建 Sample 哈希均与主矩阵 12 次长轨迹运行一致。

## 持续短请求与批量长导出的复现示例

以下命令在固定至少 6 秒窗口内，持续以 200 次/秒发送短 Session 请求，同时一次提交 16 条 128K、8 轮轨迹，
最多 8 个 finalizer 并发。`--rich` 额外携带 top-k=4、R3（2 层 × top-k=2）、多模态训练输入：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m scripts.benchmarks.benchmark_session_export \
  --lengths 131072 --concurrencies 8 --exports 16 --repeats 3 \
  --min-window 6 --rich \
  --json-out docs/benchmarks/session-export/rich-results.json
```

该命令自动交替运行优化前后版本，终端逐次打印事件循环 P99、短请求 P99、完成导出数和 CPU 时间。
[rich-results.json](./rich-results.json) 保存完整结果。若本机导出超过 6 秒，工具会延长窗口并处理额外计划请求，
不会通过截断长轨迹或漏发探针保持固定窗口。

固定窗口三次测量中，两边每次均完成 **16 次导出和 1,200 个短请求**，无丢弃，全部训练内容哈希一致。

| 指标（三次中位数） | 同步组装 | 单后台线程 |
|---|---:|---:|
| 事件循环 P99 ms | 4495.63 | 11.21 |
| 短请求 P99 ms | 4495.63 | 11.26 |
| 导出请求 P99 ms（含排队） | 4600.52 | 4755.85 |
| 导出总耗时 s | 4.601 | 4.756 |
| 导出吞吐 /s | 3.48 | 3.36 |
| CPU s | 4.634 | 4.817 |
| 峰值 RSS MiB | 975.33 | 935.13 |

事件循环 P99 降低 **99.75%**。三次范围：同步 4190.23–4502.86 ms，后台 11.21–11.96 ms；
短请求 P99 范围分别为 4190.24–4502.86 ms、11.23–11.94 ms。
相同工作量下，CPU 中位数增加约 3.9%，导出总耗时增加约 3.4%，吞吐降低约 3.3%；峰值 RSS 降低约 4.1%。
收益来自事件循环能在 CPU 工作期间继续调度其他 Session，全部训练数据仍完整导出。

## 正确性、取消和关闭

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 python -m pytest \
  tests/test_agentic_rollout.py tests/agentic/ -q
```

本机常规回归结果：132 passed、1 skipped；新增 HTTP/Ray 烟测默认按需启用，显式运行结果见下节。
新增 18 个 CPU 用例涵盖：

- 同步和后台路径的全部训练字段及 trace 精确相等：隐式导出、observation 终点、多分支显式导出；普通与
  top-k / R3 / 多模态输入，包含 tokens、loss mask、logprobs、状态、reward、advantage、统计及元数据。
- 重建后的 Sample 或保留的 Forest 引用被修改，均不影响已导出 payload。
- 大量排队任务取消、执行中的任务重复取消，后续任务仍只串行执行；worker 抛错后容量恢复。
- 组装异常发布 `RuntimeGroupError`，未提交的显式导出走受控丢弃；后续导出仍成功。
- 同 Session 的迟到 trace 更新和新 chat 与终态导出串行化，另一个 Session 仍可响应。
- 导出执行中和排队时 cancel、drop、shutdown（含重复取消）均等待实际线程停止，Forest 在执行期间仍有效，
  生命周期 close 和进程清理恰好一次；关闭后无专用导出线程残留，worker 拒绝新提交。
- 人为长阻塞超过最短窗口时，所有计划短请求和 ticker 仍全部处理，防止统计口径回退。
- 哈希仅忽略 trace 的时间事件；用户训练元数据中名为 `events` 的内容仍参与一致性校验。

以上 CPU 测试使用生产 finalizer、Forest、artifact 和 cleanup，模拟推理及已退出的 Agent 进程。
CPU 基准中的短请求未经过 HTTP/Ray 传输；下节单独补充实际 HTTP/Ray 集成证据。
未运行 GPU 训练或多节点集成，不能据此推断真实模型吞吐或整个训练链路加速。

资源变化：每个发生过导出的 Shard 新增一个线程，默认 16 个 Shard 最多新增 16 个线程，不新增进程。
Ray 每 Shard 的 0.25 CPU 预约未调整；这仍是调度预约而非硬 CPU 配额。GIL 限制仍存在，解码和 C 扩展长时间
持有 GIL 时仍可能出现较短阻塞。取消与关闭采用等待有限 CPU 工作完成的策略，不强行中止 Python 线程；
同一个已终态 Session 的请求会继续等待其导出锁。

## 补充：真实 HTTP/Ray 集成烟测

运行按需启用的回归用例：

```bash
RUN_AGENTIC_HTTP_RAY_SMOKE=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
python -m pytest tests/agentic/test_session_export_http_ray.py -q
```

也可直接运行工具，保存本次结果：

```bash
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m scripts.benchmarks.smoke_session_export_http_ray \
  --json-out /tmp/session-export-http-ray-smoke.json
```

烟测在独立子进程中启动本机 Ray：`address="local"`、唯一 namespace 与临时目录、2 个逻辑 CPU、0 GPU、
128 MiB object store；不连接或关闭已有集群。真实 socket 客户端经 loopback TCP 访问生产
`AgenticChatAPIService`，再通过真实 Ray RPC 进入另一个进程中的 SessionShard。

仅在 Shard 构造时注入确定性的 tokenizer/compiler、推理后端和可控 Agent 进程替身。生产
`start_group()`、首请求 barrier、`lease_group()`、chat/IR 提交、Agent watcher、finalizer、
`take_session_results()`、drop、shutdown 与 admission coordinator 的请求 permit 管理均实际执行。
因此本烟测验证 HTTP/Serve/Ray 调度与序列化边界，不包含真实模型推理、Agent launcher daemon 或 GPU 训练。

每个变体先通过实际 HTTP 进行一次短样本预热，再通过 HTTP 提交 4 个 Session、每个 8 轮对话。
每轮 observation 为 32 tokens、response 为 16,352 tokens，合计每条轨迹 131,072 tokens，携带 top-k 字段。
随后同时结束这 4 个模拟 Agent，触发生产 watcher 导出；客户端按 25 ms 的计划间隔发送 40 次短聊天
和 40 次健康请求。所有请求完成后才结束检查，延迟从计划发送时刻计算，包含排队和传输。

本机显式烟测结果：**1 passed in 36.25s**。以下是该次同步/后台方案的一对观察值；每类请求 N=40，
最近秩 P99 等于最大值。这是集成烟测，未进行重复测量，性能验收仍使用前文的重复 CPU 基准。

| 指标 | 同步导出 | 后台导出 |
|---|---:|---:|
| 完成长轨迹导出 | 4 | 4 |
| 每条轨迹 tokens | 131,072 | 131,072 |
| 完成短聊天 / 健康请求 | 40 / 40 | 40 / 40 |
| 短聊天最大延迟 ms | 1268.43 | 912.10 |
| 健康请求最大延迟 ms | 1183.67 | 533.16 |
| 导出并经 Ray 取回结果耗时 s | 1.235 | 1.194 |
| HTTP 响应完成后仍观察到终态导出 | 否 | 是 |

取回的训练 payload 经真实 Ray ObjectRef 传输，在客户端重建为 Sample。逐项检查所有 tokens、rollout tokens、
loss mask、logprobs、top-k IDs/logprobs（含 observation padding）、reward 和终态；除实际耗时及 trace
之外的 Sample 字段在两种方案间散列一致。另检查每轮 trace 已记录 HTTP-return 事件，证明实际入口的回传标记
路径运行过。上表的自然重叠结果是可选观察值：HTTP 返回后的诊断 RPC 可能错过已经完成的导出，
因此不作为测试通过条件。并发响应的强制检查放在下述 gate 场景中。

后台方案额外执行以下确定性交错检查，materialization gate 仅用于这些生命周期检查，不参与上表计时：

- 关闭仍在等待首请求租约的真实 SSE socket，再执行 `drop_group()`。断开连接本身不被解释为后端必然已中止。
- 使用真实 `ray.cancel()` 取消执行中的导出 RPC。测试 Actor 观察到目标任务的取消请求后，先让事件循环
  处理取消唤醒；gate 仍关闭时，RPC 必须尚未结束，Session 锁、Forest 和资源仍保留，结果尚未发布且
  close/terminate 尚未执行。释放 gate 后，再检查取消传播、正确导出及恰好一次清理。
- materialization 抛错后返回 `RuntimeGroupError`，下一条轨迹仍成功导出。
- 两个 Session 均进入终态导出、其中一个等待 worker 时执行 drop；CPU gate 尚未释放时健康请求须在 2 秒内响应，
  drop 必须继续等待；释放 gate 后全部清理完成。
- 在同样的执行中/排队状态下开始 shutdown，再取消该 shutdown RPC；确认清理实际完成后才转发取消，
  使用同样的 Actor 取消检查点确认 gate 释放前 RPC 和资源仍在，已关闭 Session 随后通过 HTTP 返回 404。

drop 和 shutdown 场景均在实际 HTTP 健康响应完成后，再检查 gate 仍关闭且两个 Session 仍处于
`FINALIZING`，最后才释放 gate。该确定性交错检查记录为 `http_during_gated_export`；自然负载中的
`http_while_finalizing` 仅保留观测用途。gate 不参与性能计时，也不通过增加固定等待时间制造重叠。

本轮另用测试进程中的临时变体核验断言：直接 finalizer 和 shutdown 的取消保护分别退化为普通
`asyncio.shield()` 时，均在 gate 释放前被 RPC 提前结束断言检出；仅推迟无 gate 阶段的诊断观察时，
自然重叠值为 false，烟测仍通过，受控 HTTP 重叠检查也通过。这些变体没有修改生产文件。

每个清理阶段均检查 Group、Session、IR、后端请求、模拟 Agent、permit cleanup task 和在途 permit 数量为零，
每个 Session 的后端 close 与模拟 Agent terminate 恰好一次。关闭 Shard 后专用导出线程为零；最后对本烟测
拥有的 Ray 子进程逐一等待退出，结果为 **0 个残留进程**。pytest 外层超时保护也只清理该子进程的后代。
本次原始 JSON 为临时运行产物；仓库保留可复现工具、断言及此处摘要。
