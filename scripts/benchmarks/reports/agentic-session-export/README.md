# Agentic Session 导出 CPU 性能对照

这是贡献者计划第二期 [No.14](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-14) 的验证证据，不是端到端训练吞吐保证。测量日期为 2026-11-06。

## 实现与瓶颈

基线在 Session Shard 事件循环内构建训练 Sample、深拷贝 payload 并转换数组。优化版在终止会话进入 FINALIZING 后冻结可达写入，将组装提交到每个 Shard 的单个后台线程，并在交付时采用明确的 Sample 所有权约定，省去立即被转换或交付的字段副本。没有新增工作进程、生产依赖或 GPU。取消排队任务、跟踪运行中的导出以及关闭时等待 worker 的行为由 Session 测试覆盖。

基线为 `9ca8cedce9ef5e0f3bbd5b10d38ad2e81a6a0f13`。候选源文件 SHA256 与运行环境记录在 [results.json](./results.json) 中。[独立 CPU 剖析](./profile.json) 使用一个 96 轮会话、重复 3 次，观测到 161,039 个 token；非 profiler 阶段计时中位数：

| 操作                  | 基线耗时 / ms |
| --------------------- | ------------: |
| `build_sample`        |        49.886 |
| `deepcopy_to_dict`    |       166.184 |
| `from_sample`         |       174.990 |
| `explicit_state_hash` |         4.080 |

这些是定位成本来源的诊断数据；cProfile 的开销不混入下面的性能对照。`deepcopy_to_dict` 是对同一 payload 独立计时的诊断操作，不与 `from_sample` 相加作为总耗时。

## 环境与负载

- 13th Gen Intel(R) Core(TM) i7-13650HX，WSL2 / Linux 6.6，Python 3.12.3；两版本及其子进程均固定在逻辑 CPU 0–7。宿主机为共享笔记本环境，不是独占服务器。
- PyTorch 2.6.0+cu124、NumPy 1.26.4、Ray 2.60.0、Transformers 5.22.1、Tokenizers 0.23.3；`CUDA_VISIBLE_DEVICES` 为空。
- 两侧使用同一份 Qwen3-0.6B tokenizer，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`；仅使用 tokenizer/config，不加载模型权重。文件校验值见结果 JSON。
- 使用真实的 SessionShard、SessionForest、Sample、TrainingFieldArtifact、会话退出与结果获取逻辑；推理后端、消息编译器、Agent 进程和许可服务为确定性模拟实现。Shard 运行在主线程事件循环，调用者在独立客户端线程中模拟异步 RPC；结果经过 protocol-5 pickle 往返后计算内容摘要。没有启动真实 Ray 集群或 HTTP/TCP 服务。
- 矩阵为 8 / 96 轮 × 每波 1 / 8 个并发长会话。每次共 2 波，在窗口第 1、5 秒发出退出信号；初始目标 2048 token、每次观察 1536 token、每轮响应 128 token。实际长度包含消息封装开销，见下表。
- 正式窗口固定 12 秒，8 个短会话轮转发送 50 个短 chat/s，另发送 20 个 health/s：每次必须完成 600 个 chat 和 240 个 health 请求。
- 每个独立进程先构建全部长轨迹，再导出一个不计入测量的预热会话（轮数为该场景的四分之一，至少 2 轮）、打开短会话，静默 1 秒后开始正式窗口。每组各版本重复 3 次，运行顺序按重复交替；所有计划请求必须完成。

## 统计口径

- 事件循环 probe 以 2 ms 为间隔，只统计正式 12 秒内的 deadline。循环被阻塞后，为每个错过的 deadline 补记迟延；窗口内 deadline 即使在窗口结束后才被执行，也保留其迟延，不添加窗口外的新 deadline。
- chat/health 延迟从计划发送时刻算到响应完成，含生产者迟延、排队和服务时间；导出延迟从 Agent 退出信号算到结果经过模拟传输的完成时刻，含后台任务排队。
- 每次运行分别用 NumPy `percentile` 的默认 linear 插值计算 P99；表中给出 3 次运行的中位数和 \[最小值, 最大值\]，不合并不同运行的样本计算 P99。
- 整批吞吐 = 完成导出数 /（最后一次结果传输完成 − 第一次退出信号）。另列活动排空速率 = 完成导出数 / 各波排空时长之和；后者排除波次空闲并可能重复计入重叠区间，不能当作整批吞吐。
- CPU/RSS 采样在正式窗口前紧邻启动，并持续到短请求及导出消费者排空完成，包含模拟传输、内容哈希和采样线程成本，不包含 teardown；每次实际区间长度记录于 `measurement_scope.cpu_rss_elapsed_s`。CPU 是进程全部线程的 `process_time`，RSS 为本进程每 20 ms 采样的峰值，可能漏掉更短峰值。
- 缺少核心训练字段、请求失败/漏发/未完成、导出未全部在窗口内取得、内容摘要不一致或清理后存在残留时，CLI 非零退出并保存诊断，不生成通过百分比。候选单独覆盖只允许 `export_workers`，不能降低工作负载。

## 全部矩阵结果

下表每格为 3 次的中位数；全部原始运行指标、计数与内容摘要见 [results.json](./results.json)。

| 轮数 / 每波并发 | 实际平均 tokens | 每次导出数 | 事件循环 P99 / ms，基线 → 优化 |    变化 | 短请求 P99 / ms，基线 → 优化 |
| --------------- | --------------: | ---------: | ------------------------------ | ------: | ---------------------------- |
| 8 / 1           |          13,903 |          2 | 1.138 → 1.133                  |  -0.41% | 5.082 → 4.974                |
| 8 / 8           |          13,903 |         16 | 94.323 → 1.222                 | -98.70% | 116.421 → 5.784              |
| 96 / 1          |         161,039 |          2 | 183.339 → 5.847                | -96.81% | 197.559 → 33.699             |
| 96 / 8          |         161,039 |         16 | 1874.465 → 130.460             | -93.04% | 1953.124 → 413.918           |

### 长轨迹、每波并发 8 的详细对照

| 指标                     |     基线，中位数 \[最小, 最大\] |  优化，中位数 \[最小, 最大\] |     变化 |
| ------------------------ | ------------------------------: | ---------------------------: | -------: |
| 事件循环 P99 / ms        | 1874.465 \[1869.980, 1906.445\] | 130.460 \[128.078, 167.702\] |  -93.04% |
| 短 chat P99 / ms         | 1953.124 \[1912.348, 2005.579\] | 413.918 \[406.365, 501.220\] |  -78.81% |
| health P99 / ms          | 1874.348 \[1824.128, 1878.155\] | 243.096 \[231.284, 294.802\] |  -87.03% |
| 导出 P99 / ms            | 2087.356 \[1999.478, 2118.665\] | 762.126 \[644.832, 819.912\] |  -63.49% |
| 整批墙钟导出区间 / s     |          6.059 \[5.996, 6.129\] |       4.762 \[4.587, 4.821\] |  -21.41% |
| 整批导出吞吐 / exports/s |          2.641 \[2.611, 2.668\] |       3.360 \[3.319, 3.488\] |  +27.24% |
| 活动排空速率 / exports/s |          3.849 \[3.789, 4.004\] |    11.288 \[10.684, 12.987\] | +193.24% |
| 进程 CPU / s             |          5.066 \[4.862, 5.069\] |       2.745 \[2.462, 3.005\] |  -45.82% |
| 采样峰值 RSS / MiB       |    883.328 \[840.160, 884.074\] | 841.508 \[837.543, 860.598\] |   -4.73% |
| CPU/RSS 实际采样区间 / s |       12.001 \[12.000, 12.001\] |    12.001 \[12.001, 12.001\] |   +0.00% |

本机上述长轨迹并发负载的事件循环 P99 中位数降低 **93.04%**，达到任务的 50% 数值门槛。短轨迹场景保留在矩阵中，不将其结果外推为所有负载都获得同样收益。

24 次正式运行均通过有效性检查：合计完成 216 次长导出、14,400 个短 chat 与 5,760 个 health 请求，失败数为 0；各场景六次运行的脚本列出训练字段摘要一致。所有运行 teardown 后的任务、Group、Session、导出 Future、导出 worker 线程及额外线程残留均为 0。

## 复现持续短请求与长导出

在同一 CPU Python 环境中准备基线 checkout 和同一 tokenizer。以下命令的 BASELINE_SRC、TOKENIZER_DIR 和输出目录由复跑者指定；不要为两个版本使用不同的数据或 tokenizer。

```bash
git worktree add --detach /tmp/relax-session-export-baseline 9ca8cedce9ef5e0f3bbd5b10d38ad2e81a6a0f13
hf download Qwen/Qwen3-0.6B tokenizer.json tokenizer_config.json vocab.json merges.txt config.json \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir /tmp/qwen3-tokenizer

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
BASELINE_SRC=/tmp/relax-session-export-baseline
TOKENIZER_DIR=/tmp/qwen3-tokenizer

taskset -c 0-7 python scripts/benchmarks/benchmark_agentic_session_export.py compare \
  --baseline-src "$BASELINE_SRC" --candidate-src "$PWD" \
  --tokenizer "$TOKENIZER_DIR" --preset long \
  --set groups=2 --set turns=96 --set sessions_per_group=8 \
  --set window_s=12.0 --set wave_interval_s=4.0 \
  --repeats 3 --out-dir /tmp/session-export-t96-c8
```

完整矩阵对 `turns=8,96`、`sessions_per_group=1,8` 的四种组合各执行上述比较，并分别指定输出目录。比较命令会交替版本顺序，逐进程保存 JSON 和包含每秒短请求最大延迟的摘要。

## 验证与范围

本 PR 的相关 CPU 回归和工具验证命令：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/cpu/agentic/test_session_export.py tests/cpu/test_agentic_rollout.py \
  tests/agentic/test_tool_result_images.py tests/cpu/agentic \
  tests/cpu/tools/test_agentic_session_export_benchmark.py -q
pre-commit run --all-files --show-diff-on-failure
```

相关 CPU 测试为 **159 passed**。测试包括导出内容与旧 deepcopy 路径一致、状态冻结、排队取消、运行中导出的关闭等待、异常映射、重复生命周期，以及压测 CLI 的失败判定、窗口边界和吞吐口径。

未运行整个仓库的 GPU/多节点训练与集成套件；本对照不证明真实 HTTP/Ray 网络时延、SGLang 模型推理或训练吞吐收益。原始测量仅对本机、已声明的负载与字段成立；活动资格、CI、代码审查和最终验收由维护者确认。
