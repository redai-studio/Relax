# Agentic Session 导出 CPU 性能与复现

本工具使用真实 `AgenticSessionShard` finalizer、`SessionForest.build_sample` 和 `TrainingFieldArtifact`，模拟推理后端与 Agent。只加载 tokenizer，不需要模型权重、SGLang 服务或 GPU。

## 实现

每个 Shard 用一个按需启动的后台线程处理导出。asyncio admission 在提交前限制并发，executor 内部没有待执行请求积压。等待和执行期间持有该 Session 的锁，其他 Session 与 health 请求可以继续执行；取消时等真实 worker 完成再清理状态，关闭时拒绝新提交并 drain/join。

`_compact_sample_payload` 对已知训练字段中扁平 builtin 数值列表直接创建独立 NumPy 数组，省去转换前的临时深拷贝列表。metadata 与其他对象继续保留 deepcopy、别名与循环引用语义；存在自定义复制行为时整体回退到原路径。接收端 Sample 恢复规则保持原实现。

## 环境与准备

本次测量：Linux x86_64，Xeon Platinum 8575C，Python 3.12.3，PyTorch 2.10.0+cpu、NumPy 1.26.4、Ray 2.56.1、Transformers 5.22.1、FastAPI 0.139.1、HTTPX 0.28.1。容器 CPU quota 23 核、内存上限 225 GiB；压测固定使用逻辑 CPU 0–7，0 GPU。

两版使用同一 Python 环境、依赖和 tokenizer。已有 Relax 环境可直接使用；CPU 环境可以先安装 CPU Torch，再安装本工具需要的已有项目依赖：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r docs/benchmarks/session-export-requirements.txt
```

相邻 requirements 文件记录本次使用的 CPU Python 环境，不修改生产安装依赖。Linux `/proc` 与 `prctl` 用于压测进程树监督和 RSS 采样。所有进程由私有 supervisor 创建、终止和回收，不连接已有 Ray 集群，不调用全局 `ray stop`。

从 `Qwen/Qwen3-0.6B` 下载 tokenizer 文件到一个共享本地目录：`config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`，无需下载模型权重。测量文件 SHA256 在相邻结果 JSON 中，可用于检查 tokenizer 版本。

## 原生双版本对照与短请求示例

在包含本提交的 checkout 根目录执行。`BASE` 为未优化基线；同一脚本分别启动新的子进程，导入两个 checkout 自己的生产模块。没有替换旧 finalizer 或混用优化版 compaction。

```bash
BASE=657f8fb961af9ab4b648e7d58280055dbfe4b626
git worktree add --detach ../Relax-export-baseline "$BASE"

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# TOKENIZER_DIR 指向上一步共享的本地 tokenizer 目录。
TOKENIZER_DIR=/path/to/Qwen3-0.6B-tokenizer

# 长轨迹导出时，以 100 requests/s 持续发送 500 个短请求。
taskset -c 0-7 .venv/bin/python scripts/benchmarks/benchmark_session_export.py \
  --variants baseline=../Relax-export-baseline optimized=. \
  --tokenizer "$TOKENIZER_DIR" --turns 64 --concurrency 8 \
  --exports 16 --tokens-per-turn 2048 --observation-tokens 512 \
  --short-rate 100 --short-requests 500 --probe-ms 2 \
  --warmups 1 --runs 3 --json-out outputs/session-export-long.json
```

短请求调用真实 `shard.health()`，导出请求调用真实 finalizer。两者经进程内 HTTPX ASGI 路由，没有 TCP 或 Ray RPC；`_finish_session` 被结果收集器替代。因此测量包含树遍历、解码、组装、打包和本地异步调度，不包含 Agent 进程退出、Ray 传输、训练接收端恢复或 GPU 推理。

完整矩阵将 `--turns 64 --concurrency 8` 改为 `--turns 8 64 --concurrency 1 8`。`--tokenizer synthetic` 可快速检查工具，但其数字不用于性能结论。`--profile outputs/export.prof` 会另行采集计时外的 CPU 剖析；该次结果标记为 `diagnostic_only`，不能混入正式比较。

## 负载与统计口径

- 每条轨迹 8 或 64 轮，每轮响应 2048 token，轮间观察 512 token，初始 prompt 128 token；对应完整长度 20,096 或 163,456 token。主性能矩阵 top-k 为 0；top-k、R3 和多模态另由正确性测试覆盖。
- 每次固定完成同批到达的 16 次导出、500 次短请求。导出 admission 并发为 1 或 8；短请求从绝对计划时刻按 100 requests/s 到达，不因服务阻塞降低发送量。未完成全部请求时工具报错。
- 每 case/版本先丢弃 1 次全负载预热，再保留 3 次独立进程测量。每个进程还先预热 1 次同规模组装和 health。版本运行顺序轮转；两版本、三重复时位置次数不完全相等，JSON 保留每次结果供判断波动。
- 窗口从短请求计划开始，到长短请求全部完成并 drain 50 ms。请求延迟从计划到达时刻计到响应完成，包含事件循环调度、admission 和 worker 排队。导出吞吐为完成数量除以从同批计划到达到最后一次导出响应的时间。
- 事件循环每 2 ms 一个绝对 deadline，恢复后补记全部错过的 tick。P99 为 nearest rank `ceil(0.99*N)`；先按单次运行求 P99，再报告 3 次中位数、最小/最大值和样本标准差，不把多次样本混成一个 P99。
- CPU 时间为计时窗口内 `time.process_time()` 增量，包含该 worker 全部线程。父进程每 10 ms 采样 worker VmRSS；可能遗漏更短的峰值。峰值不含 supervisor，也不是完整 Ray 进程树 RSS。另记录 `ru_maxrss` 供参考。
- 每版进程结构相同，都额外使用一个测试 supervisor；优化版每个 Shard 最多新增一个导出线程，没有额外生产进程或 GPU。等待者保留其原有 Forest，未宣称限制全部驻留 Session 数。
- 计时后比较训练字段 SHA256，排除时钟 trace metadata。记录实际导入模块路径、Git HEAD 和生产文件 SHA256，并检查测量中和重复之间源码未变。训练字段差异会使比较失败。

## 测量结果

以下沿用已完成的四组正式测量。其后的 pre-commit 仅调整 docstring 换行和 JSON 排版，执行逻辑未变；结果 JSON 同时保留测量时与提交时的源码校验值。后续重复运行已停止，未纳入统计。

2026-10-31 在同一硬件、相同 CPU affinity 和固定负载下，64 轮、并发 8 的事件循环 P99 中位数从 **3320.302 ms 降至 22.110 ms（−99.33%）**；短请求 P99 从 **4437.955 ms 降至 32.425 ms**。导出吞吐 **3.567 → 13.872 exports/s（+288.92%）**，CPU 时间 **4.609 → 1.318 s（−71.40%）**。窗口峰值 RSS **694.059 → 698.781 MiB（+4.723 MiB）**。

表格为 3 次正式重复的中位数 [最小值, 最大值]；样本标准差和每次运行的统计量见结果 JSON。

| 指标 | baseline | optimized |
| --- | --- | --- |
| 事件循环 P99 / ms | 3320.302 [3311.760, 3401.352] | 22.110 [21.432, 30.655] |
| 短请求 P99 / ms | 4437.955 [4429.162, 4452.694] | 32.425 [28.579, 47.230] |
| 导出请求 P99 / ms，含排队 | 4485.857 [4477.107, 4500.574] | 1153.426 [1141.863, 1153.881] |
| 导出吞吐 / exports/s | 3.567 [3.555, 3.574] | 13.872 [13.866, 14.012] |
| 窗口进程 CPU 时间 / s | 4.609 [4.596, 4.626] | 1.318 [1.311, 1.320] |
| 窗口峰值 RSS / MiB | 694.059 [693.059, 695.328] | 698.781 [697.094, 699.750] |
| 每次完成导出数 | 16 | 16 |
| 每次完成短请求数 | 500 | 500 |

其余矩阵如下。4 组 × 2 版本 × 3 次正式测量共完成 **384 次导出、12,000 次短请求**，各次完成量相同，训练内容摘要一致。

| 轮数 | 并发 | baseline loop P99 / ms | optimized loop P99 / ms | P99 降幅 | 吞吐变化 | CPU 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1 | 275.593 | 3.953 | 98.57% | +274.22% | -51.62% |
| 8 | 8 | 467.236 | 3.750 | 99.20% | +270.34% | -53.18% |
| 64 | 1 | 2849.091 | 25.136 | 99.12% | +283.65% | -70.91% |
| 64 | 8 | 3320.302 | 22.110 | 99.33% | +288.92% | -71.40% |

独立 CPU 剖析使用相同 64 轮轨迹、4 次真实同步 builder，位于计时窗口之外：baseline 总剖析时间 4.176 s，deepcopy 累计 3.832 s（91.75%）；optimized 总计 0.775 s，deepcopy 累计 0.018 s（2.29%）。累计时间包含嵌套调用，不能相加，也不与正式性能数据混用。

[结果 JSON](session-export-results.json) 保存正式逐次测量、汇总、训练内容摘要和生产代码/工具/tokenizer 校验值。共享元数据已去重，运行目录和 PID 已去除；统计值保留原精度。

## 正确性与收尾验证

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m pytest tests/test_agentic_rollout.py tests/agentic \
  tests/tools/test_session_benchmark_cleanup.py -q

# 独立、私有、0 GPU 的真实 HTTP/Ray Serve 集成。
RELAX_RUN_HTTP_RAY_SMOKE=1 PYTHONPATH=. OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m pytest tests/agentic/test_session_export_http_ray.py -q
```

提交版 CPU 回归 **192 passed，1 skipped**；独立启用的集成 **1 passed**。覆盖训练 payload、dtype/字节、别名/循环、未知对象 deepcopy 与 Tensor storage hooks、异常，以及取消、late chat/trace、drop、shutdown、重复导出后引用和线程释放。

真实 HTTP/Ray smoke 使用原生 Shard 构造器、Serve ingress 与 AdmissionCoordinator，只替换 compiler、generation 和 Agent launcher。覆盖正常 chat/导出、SSE 断开后显式 drop、已送达的 Ray RPC 取消、组装失败恢复、活跃/排队 drop 与 shutdown。SSE 断开本身不等同于取消推理。结果为 8 grants / 8 releases / 0 active permits，9 个 Session 的状态引用清空，线程、任务、私有 Ray 进程和临时目录全部释放。

工具故障注入另覆盖成功、失败、超时、重复中断、脱离进程组后代、文件描述符故障和相对 cwd，并确认不会清理无关子进程。

代码检查沿用仓库配置：安装 `pre-commit` 后运行 `pre-commit run --all-files --show-diff-on-failure`。提交前已通过全部检查；CPU 依赖快照不包含这一步自动创建的独立 hook 环境。

未运行多节点 GPU 训练或全仓库 GPU 集成。GIL、单次解码和大型 metadata 仍可能影响延迟，线程不能保证所有输入的最坏响应时间。共享开发机存在其他负载，3 次重复不足以推断所有硬件与生产负载的效果。
