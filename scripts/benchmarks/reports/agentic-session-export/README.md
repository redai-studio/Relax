# Agentic Session 导出 CPU 性能对照

此报告对应 [贡献者计划 No.14](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-14)。测量是 CPU 导出和请求调度，不是模型训练吞吐。

## 实现

- Session 进入 FINALIZING 后禁止其他路径写 Forest；每个 Shard 只运行一个导出 worker。拿到异步名额才提交到线程池，尚未提交的任务可取消并释放 Forest；开始执行的工作仍要收尾。结果 ID 和轨迹记录继续在 Shard 循环发布。
- 普通 Sample 打包保留原有快照隔离。Shard 专用路径明确交付刚构建的 Sample，不重复深拷贝交付字段；训练数组仍在打包时转换。
- 标准 HF fast tokenizer 无自定义解码或清理空格时，使用同一个 Rust 解码器的单项批解码，释放 GIL 并遵循 tokenizer 库已有的并行设置。自定义 tokenizer 和空格清理走原路径，保留 special token；不修改环境变量或 tokenizer 配置。
- [Miles #3736](https://github.com/radixark/miles/pull/3736) 提供了有界后台组装的参考；[Miles #4044](https://github.com/radixark/miles/pull/4044) 提供了 CPU 压测参考。参考了 [Relax #519](https://github.com/redai-studio/Relax/pull/519) 的提交前准入和 [#601](https://github.com/redai-studio/Relax/pull/601) 的解码观察。公开 HF `batch_decode` 在 [Transformers 5.16.0](https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/tokenization_utils_base.py#L2590-L2667) 仍逐条调用解码；释放 GIL 的是 [Tokenizers Rust 绑定](https://github.com/huggingface/tokenizers/blob/v0.23.2/bindings/python/src/tokenizer.rs#L1456-L1495)。

## 当前版：默认配置同机对照

环境同下，三版均使用 `TOKENIZERS_PARALLELISM=true`，固定上游 46701c8、同一个 Qwen3-0.6B tokenizer。对照对象为上一版 7340200、当前版和 [#601](https://github.com/redai-studio/Relax/pull/601) 在 32464ba 的生产补丁（原样应用到同一基线）。使用同一个精简工具、独立进程、交替运行顺序。单波 96 轮 × 并发 8，每条 161,039 token，窗口 6 秒；每次必须完成 8 次导出、300 个短 chat、120 个 health。以下为每版 3 次运行的中位数 \[最小, 最大\]。

| 指标                     |            上一版 7340200 |                    当前版 |                      #601 |
| ------------------------ | ------------------------: | ------------------------: | ------------------------: |
| 事件循环 P99 / ms        |    53.08 \[51.28, 54.74\] |       6.09 \[5.16, 6.24\] |       6.60 \[6.50, 7.75\] |
| 短 chat P99 / ms         | 168.29 \[123.67, 219.04\] |    17.91 \[17.66, 22.41\] |    24.72 \[21.36, 24.79\] |
| health P99 / ms          |   88.82 \[62.74, 114.78\] |    14.68 \[13.98, 14.78\] |    14.92 \[13.92, 23.57\] |
| 导出 P99 / ms（含排队）  | 588.61 \[579.41, 594.27\] | 502.00 \[485.71, 518.54\] | 532.41 \[501.23, 534.54\] |
| 整批导出吞吐 / exports/s |    13.59 \[13.46, 13.81\] |    15.80 \[15.32, 16.35\] |    14.92 \[14.85, 15.85\] |
| 进程 CPU / s             |       1.11 \[1.07, 1.15\] |       1.11 \[1.10, 1.14\] |       1.20 \[1.13, 1.21\] |
| 峰值 RSS / MiB           | 767.21 \[767.03, 780.31\] | 738.00 \[737.93, 738.05\] | 738.30 \[737.15, 738.39\] |

当前版相较上一版，事件循环 P99 从 **53.08 降到 6.09 ms（−88.5%）**；#601 在同一口径为 6.60 ms。本组是 CPU 导出与模拟请求对照，不外推为真实网络或训练吞吐。另测默认设置的 8 轮 / 并发 1（各 2 次，循环 P99 1.20 → 1.20 ms），以及显式关闭并行的 96 轮 / 并发 8（各 1 次，6.28 → 5.07 ms，仅作回归检查）。15 次运行都完成了计划负载；同负载下校验字段摘要一致，清理检查通过。

单独使用真实 SGLangMessageCompiler 和 Qwen3 tokenizer 探查线程数：在 CPU 0–7 上，默认设置的首次 prompt 编码使进程线程数由 2 增至 10，随后单项批解码仍为 10；跳过真实编译直接冷调用则会从 2 增至 10。显式关闭并行时这组探查一直是 2 个线程。线程池行为是库的原有设置，不是导出线程数限制；结果只覆盖实测的 tokenizer 路径。

## 上一版：原压测工具的同机对照

用 [原工具](https://github.com/3516027002att-ui/Relax/blob/9a57276c7dfbc18b88686b2d5f824a40f43bd73c/scripts/benchmarks/benchmark_agentic_session_export.py) 保持负载和测量不变，对比原 PR 提交 9a57276 和上一版 7340200；原工具模拟 protocol-5 序列化与缓冲区复制，不启动 Ray 服务。96 轮 / 每波并发 8，两波，固定 12 秒；每次完成 16 次导出、600 个短 chat、240 个 health。每版 3 次，交替顺序；下表为中位数 \[最小值, 最大值\]，没有跨次混合 P99。使用 2026-11-09 本机 i7-13650HX / WSL2 / CPU 0–7、Python 3.12.3、PyTorch 2.6.0、NumPy 1.26.4、Ray 2.60.0、Transformers 5.22.1 / Tokenizers 0.23.3，两侧均设 `TOKENIZERS_PARALLELISM=false`。

| 指标                    |                     原 PR |                    上一版 |
| ----------------------- | ------------------------: | ------------------------: |
| 事件循环 P99 / ms       |    66.34 \[54.80, 67.83\] |       4.89 \[4.66, 5.63\] |
| 短 chat P99 / ms        | 256.76 \[190.31, 287.30\] |    18.66 \[17.22, 19.07\] |
| 导出 P99 / ms（含排队） | 643.65 \[625.68, 648.10\] | 482.67 \[476.14, 492.65\] |
| 整批吞吐 / exports/s    |       3.44 \[3.44, 3.48\] |       3.57 \[3.57, 3.57\] |
| 进程 CPU / s            |       2.29 \[2.28, 2.37\] |       2.21 \[2.21, 2.24\] |
| 峰值 RSS / MiB          | 829.41 \[826.26, 845.19\] | 802.83 \[802.57, 803.49\] |

六次均有效，校验字段摘要一致，后台资源无残留。历史 [24 次原始数据](https://github.com/3516027002att-ui/Relax/blob/3a3461174f9e14e4435bb88cbc7ad26c5c521683/scripts/benchmarks/reports/agentic-session-export/results.json) 与 [CPU 剖析](https://github.com/3516027002att-ui/Relax/blob/3a3461174f9e14e4435bb88cbc7ad26c5c521683/scripts/benchmarks/reports/agentic-session-export/profile.json) 留在固定提交，不与本轮混用。

同一负载下不设置 `TOKENIZERS_PARALLELISM`（单项 Rust 解码），另做 2 对：事件循环 P99 **61.56 → 44.24 ms**，短 chat P99 **221.04 → 122.47 ms**，导出 P99 **617.00 → 539.18 ms**，吞吐 **3.47 → 3.53 exports/s**。进程 CPU 2.29 → 2.17 s，峰值 RSS 831.90 → 832.72 MiB（两侧范围重叠）。四次均有效、校验字段一致。明确关闭 tokenizer 并行时的降幅不能直接用于默认配置。

## 上一版：精简工具的完整矩阵

基线固定为 [46701c8](https://github.com/redai-studio/Relax/commit/46701c8f2072810b4d908ea1d1ed42a237f2514f)。同一机器、软件环境和 tokenizer，两侧均设 `TOKENIZERS_PARALLELISM=false`。每个场景各跑 3 次，基线与上一版 7340200 交替先跑；每次用新进程。以下结果与原工具对照独立，不跨表计算增幅。

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

| 指标                     |                     上游基线 |                    上一版 |
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
git worktree add --detach /tmp/relax-export-previous 7340200098adf36ce62d68879c569f16a8b2b934
hf download Qwen/Qwen3-0.6B tokenizer.json tokenizer_config.json vocab.json merges.txt config.json \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir /tmp/qwen3-tokenizer

export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 RAY_ENABLE_AUTO_CONNECT=0
RUNS=$(mktemp -d)
for turns in 8 96; do
  for concurrency in 1 8; do
    taskset -c 0-7 python scripts/benchmarks/benchmark_agentic_session_export.py compare \
      --baseline-src /tmp/relax-export-baseline --candidate-src /tmp/relax-export-previous \
      --tokenizer /tmp/qwen3-tokenizer --preset long --set groups=2 \
      --set turns="$turns" --set sessions_per_group="$concurrency" \
      --set window_s=12 --set wave_interval_s=4 --repeats 3 \
      --out-dir "$RUNS/t${turns}-c${concurrency}"
  done
done
```

复跑当前版与上一版的默认配置对照时，用 `TOKENIZERS_PARALLELISM=true`，基线指向 `/tmp/relax-export-previous`，候选指向当前 checkout，指定 `groups=1`、`turns=96`、`sessions_per_group=8`、`window_s=6` 和新的输出目录。

每组生成各次原始 JSON 和带范围的 `summary.md`，不把逐次文件提交到仓库。

## 其他验证

导出测试覆盖隐式与显式结果、快照隔离、R3、冻结、取消和关闭等待。使用镜像固定的 Transformers 5.16.0 / Tokenizers 0.23.2 复跑导出测试，真实 Qwen tokenizer 的长序列与特殊 token 解码逐字一致；两组依赖都在同一个 tokenizer 上并发执行消息编译和长序列解码，预热后线程数未再增加。独立本机 Ray / HTTP 冒烟验证了导出 worker 卡住时另一个会话仍能 chat、health，且等待导出的组被取消后可释放 Forest。没有运行真实模型推理、训练、GPU/多节点套件。
