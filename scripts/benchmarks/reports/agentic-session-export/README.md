# Agentic Session 导出 CPU 性能对照

此报告对应 [贡献者计划 No.14](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-14)。测量是 CPU 导出和请求调度，不是模型训练吞吐。

## 实现

- Session 进入 FINALIZING 后禁止其他路径写 Forest；每个 Shard 只运行一个导出 worker。拿到异步名额才提交到线程池，尚未提交的任务可取消并释放 Forest；开始执行的工作仍要收尾。结果 ID 和轨迹记录继续在 Shard 循环发布。
- 普通 Sample 打包保留原有快照隔离。Shard 专用路径明确交付刚构建的 Sample，不重复深拷贝交付字段；训练数组仍在打包时转换。
- 标准 HF fast tokenizer 无自定义解码或清理空格时，使用同一个 Rust 解码器。仅在已有 `TOKENIZERS_PARALLELISM` 配置明确关闭并行时使用释放 GIL 的单项批解码；否则用单项解码，避免启动额外 Rayon 线程池。自定义 tokenizer 和空格清理走原路径，保留 special token。
- [Miles #3736](https://github.com/radixark/miles/pull/3736) 提供了有界后台组装的参考；[Miles #4044](https://github.com/radixark/miles/pull/4044) 提供了 CPU 压测参考。参考了 [Relax #519](https://github.com/redai-studio/Relax/pull/519) 的提交前准入和 [#601](https://github.com/redai-studio/Relax/pull/601) 的解码观察。公开 HF `batch_decode` 在 [Transformers 5.16.0](https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/tokenization_utils_base.py#L2590-L2667) 仍逐条调用解码；释放 GIL 的是 [Tokenizers Rust 绑定](https://github.com/huggingface/tokenizers/blob/v0.23.2/bindings/python/src/tokenizer.rs#L1456-L1495)。

## 原压测工具的同机对照

用 [原工具](https://github.com/3516027002att-ui/Relax/blob/9a57276c7dfbc18b88686b2d5f824a40f43bd73c/scripts/benchmarks/benchmark_agentic_session_export.py) 保持负载和测量不变，对比原 PR 提交 9a57276 和本版；原工具模拟 protocol-5 序列化与缓冲区复制，不启动 Ray 服务。96 轮 / 每波并发 8，两波，固定 12 秒；每次完成 16 次导出、600 个短 chat、240 个 health。每版 3 次，交替顺序；下表为中位数 \[最小值, 最大值\]，没有跨次混合 P99。使用 2026-11-09 本机 i7-13650HX / WSL2 / CPU 0–7、Python 3.12.3、PyTorch 2.6.0、NumPy 1.26.4、Ray 2.60.0、Transformers 5.22.1 / Tokenizers 0.23.3，两侧均设 `TOKENIZERS_PARALLELISM=false`。

| 指标                    |                     原 PR |                    重写版 |
| ----------------------- | ------------------------: | ------------------------: |
| 事件循环 P99 / ms       |    66.34 \[54.80, 67.83\] |       4.89 \[4.66, 5.63\] |
| 短 chat P99 / ms        | 256.76 \[190.31, 287.30\] |    18.66 \[17.22, 19.07\] |
| 导出 P99 / ms（含排队） | 643.65 \[625.68, 648.10\] | 482.67 \[476.14, 492.65\] |
| 整批吞吐 / exports/s    |       3.44 \[3.44, 3.48\] |       3.57 \[3.57, 3.57\] |
| 进程 CPU / s            |       2.29 \[2.28, 2.37\] |       2.21 \[2.21, 2.24\] |
| 峰值 RSS / MiB          | 829.41 \[826.26, 845.19\] | 802.83 \[802.57, 803.49\] |

六次均有效，校验字段摘要一致，后台资源无残留。历史 [24 次原始数据](https://github.com/3516027002att-ui/Relax/blob/3a3461174f9e14e4435bb88cbc7ad26c5c521683/scripts/benchmarks/reports/agentic-session-export/results.json) 与 [CPU 剖析](https://github.com/3516027002att-ui/Relax/blob/3a3461174f9e14e4435bb88cbc7ad26c5c521683/scripts/benchmarks/reports/agentic-session-export/profile.json) 留在固定提交，不与本轮混用。

同一负载下不设置 `TOKENIZERS_PARALLELISM`（单项 Rust 解码），另做 2 对：事件循环 P99 **61.56 → 44.24 ms**，短 chat P99 **221.04 → 122.47 ms**，导出 P99 **617.00 → 539.18 ms**，吞吐 **3.47 → 3.53 exports/s**。进程 CPU 2.29 → 2.17 s，峰值 RSS 831.90 → 832.72 MiB（两侧范围重叠）。四次均有效、校验字段一致。明确关闭 tokenizer 并行时的降幅不能直接用于默认配置。

## 精简工具的上游基线对照

基线固定为 [46701c8](https://github.com/redai-studio/Relax/commit/46701c8f2072810b4d908ea1d1ed42a237f2514f)。同一机器、软件环境和 tokenizer，两侧均设 `TOKENIZERS_PARALLELISM=false`。每个场景各跑 3 次，基线与候选交替先跑；每次用新进程。以下结果与原工具对照独立，不跨表计算增幅。

- 使用真实 SessionShard、SessionForest、Sample 和 TrainingFieldArtifact。推理、消息编译、Agent 进程和许可服务为确定性模拟。Shard 和客户端在不同事件循环；用 protocol-5 序列化、复制缓冲区、反序列化模拟结果传输，不启动 Ray 服务。
- 预建长轨迹，预热一次导出。正式窗口 12 秒；两波退出安排在第 1、5 秒；短 chat 50/s、health 20/s。短请求按计划发起时刻起算，含排队；导出从退出信号计至模拟传输完成。
- 循环迟延按 2 ms deadline 统计，阻塞错过的 deadline 保留迟延。整批导出吞吐为完成数除以首个退出到末次取回的间隔（含波次空闲）；CPU 为进程总时间，峰值 RSS 每 20 ms 采样，含模拟传输与摘要计算，均不含 teardown。
- 缺失请求、窗口内未完成全部导出、训练字段摘要不等或清理后有残留均判失败；不生成成功摘要。比较目录必须为新路径。

| 轮数 / 并发 | 平均 token / 导出 | 导出数 / 次 |  循环 P99 / ms | 循环 P99 变化 | 短 chat P99 / ms |
| ----------- | ----------------: | ----------: | -------------: | ------------: | ---------------: |
| 8 / 1       |            13,903 |           2 |    1.25 → 1.21 |         -3.5% |      5.29 → 5.25 |
| 8 / 8       |            13,903 |          16 |   93.42 → 1.22 |        -98.7% |    103.84 → 5.28 |
| 96 / 1      |           161,039 |           2 |  172.90 → 1.28 |        -99.3% |    188.54 → 5.51 |
| 96 / 8      |           161,039 |          16 | 1807.36 → 5.77 |        -99.7% |  1849.37 → 20.75 |

### 96 轮 / 并发 8（中位数 \[最小, 最大\]）

| 指标                     |                     上游基线 |                    重写版 |
| ------------------------ | ---------------------------: | ------------------------: |
| 事件循环 P99 / ms        | 1807.36 \[1752.55, 1839.42\] |       5.77 \[5.31, 6.50\] |
| 短 chat P99 / ms         | 1849.37 \[1795.10, 1884.38\] |    20.75 \[19.57, 22.81\] |
| health P99 / ms          | 1804.78 \[1718.22, 1806.49\] |    14.02 \[10.45, 14.13\] |
| 导出 P99 / ms            | 1973.00 \[1899.81, 2022.06\] | 494.72 \[469.81, 503.97\] |
| 整批导出吞吐 / exports/s |          2.70 \[2.69, 2.71\] |       3.58 \[3.57, 3.58\] |
| 进程 CPU / s             |          4.77 \[4.73, 4.81\] |       2.31 \[2.28, 2.33\] |
| 峰值 RSS / MiB           |    883.87 \[883.32, 885.21\] | 809.11 \[808.67, 809.18\] |

24 次运行共完成 216 次导出、14,400 个短 chat 与 5,760 个 health 请求；各场景的训练字段摘要一致，teardown 无残留。长轨迹高并发场景的循环 P99 降低 **99.7%**，超过任务 50% 的数值门槛，不代表所有负载都会有同等收益。

## 复现

在安装项目现有 CPU 依赖的同一环境中运行。仅下载 tokenizer/config，不下载模型权重。

```bash
git worktree add --detach /tmp/relax-export-baseline 46701c8f2072810b4d908ea1d1ed42a237f2514f
hf download Qwen/Qwen3-0.6B tokenizer.json tokenizer_config.json vocab.json merges.txt config.json \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir /tmp/qwen3-tokenizer

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 RAY_ENABLE_AUTO_CONNECT=0
RUNS=$(mktemp -d)
for turns in 8 96; do
  for concurrency in 1 8; do
    taskset -c 0-7 python scripts/benchmarks/benchmark_agentic_session_export.py compare \
      --baseline-src /tmp/relax-export-baseline --candidate-src "$PWD" \
      --tokenizer /tmp/qwen3-tokenizer --preset long --set groups=2 \
      --set turns="$turns" --set sessions_per_group="$concurrency" \
      --set window_s=12 --set wave_interval_s=4 --repeats 3 \
      --out-dir "$RUNS/t${turns}-c${concurrency}"
  done
done
```

每组生成各次原始 JSON 和带范围的 `summary.md`，不把逐次文件提交到仓库。

## 其他验证

导出测试覆盖隐式与显式结果、快照隔离、R3、冻结、取消和关闭等待。使用镜像固定的 Transformers 5.16.0 / Tokenizers 0.23.2 复跑导出测试，另对真实 Qwen tokenizer 检查长 token 序列及多语言、特殊 token 的解码一致性。独立本机 Ray / HTTP 冒烟验证了导出 worker 卡住时另一个会话仍能 chat、health，且等待导出的组被取消后可释放 Forest。没有运行真实模型推理、训练、GPU/多节点套件。
