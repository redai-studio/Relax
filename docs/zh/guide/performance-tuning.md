# 性能调优

Relax 训练吞吐量最大化实践指南。本文提到的所有参数均可在[配置参考手册](./configuration.md)中查阅。

---

## 性能分析

调优前先定位瓶颈。Relax 内置三套互补的 profiling 工具，覆盖 **推理引擎**、**训练后端** 和 **GPU 内存** 三个维度。所有 trace 文件默认保存在 `traces/<tb_experiment_name>/` 目录下，按子目录区分：

| 工具 | 目标 | 默认输出目录 | 查看方式 |
|---|---|---|---|
| SGLang Profiling | Rollout 推理的 CUDA kernel / 算子分析 | `traces/<tb_experiment_name>/sglang_trace/` | TensorBoard or `https://ui.perfetto.dev/` |
| Training Profiling | Actor 训练 / log-probs 计算的算子分析 | `traces/<tb_experiment_name>/train_trace/` | TensorBoard or `https://ui.perfetto.dev/` |
| Memory Profiling | GPU 显存分配历史，OOM 诊断 | `traces/<tb_experiment_name>/memory_snapshot/` | [PyTorch Memory Viz](https://pytorch.org/memory_viz) |

### Trace 文件命名规则

- **训练 trace** 文件名包含 `rank{global}_dp{dp}_tp{tp}_pp{pp}` 标识，例如 `train_overall_rank0_dp0_tp0_pp0.1713780123.pt.trace.json.gz`
- **内存快照** 文件名同样包含 rank 标识，例如 `memory_snapshot_time1713780123_rank0_dp0_tp0_pp0_snapshot.pickle`
- **SGLang trace** 文件以 `engine{i}` 为前缀区分不同引擎实例，例如 `engine0-1713780123-TP-0.trace.json.gz`

### 1. SGLang 推理 Profiling

对 Rollout 阶段所有 SGLang 引擎进行 `torch.profiler` 采集。通过 HTTP API `/start_profile` 和 `/stop_profile` 触发，不影响训练侧的 profiler。

**示例用法** — 每个 rollout step 都采集：

```bash
python3 relax/entrypoints/train.py \
    --sglang-profile \
    --tb-experiment-name my-experiment \
    # ... 其他参数
```

**指定 rollout step 范围** — 仅在 step 2、3、4 采集（start/end 均 inclusive，推荐用法，避免大量 trace 文件）：

```bash
python3 relax/entrypoints/train.py \
    --sglang-profile \
    --sglang-profile-step-start 2 \
    --sglang-profile-step-end 4 \
    --tb-experiment-name my-experiment \
    # ... 其他参数
```

也可以用 `--sglang-profile-steps` 指定不连续的 step 列表（优先级高于 start/end）：

```bash
--sglang-profile-steps 2 5 10
```

所有 step 参数均使用 **绝对 rollout ID**（0-indexed），即第 0 轮、第 1 轮 ... 与 `--start-rollout-id` 无关。

**进阶参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sglang-profile-step-start` | None | profiling 起始 rollout step（**inclusive**，0-indexed） |
| `--sglang-profile-step-end` | None | profiling 结束 rollout step（**inclusive**，0-indexed）。例如 start=2, end=4 采集 step 2, 3, 4 |
| `--sglang-profile-steps` | None | 不连续 step 列表，优先级高于 start/end |
| `--sglang-profile-num-steps` | 3 | 每轮 Rollout 中采集的 SGLang 前向步数。-1 表示整轮采集 |
| `--sglang-profile-activities` | CPU GPU | 要采集的活动类型 |
| `--sglang-profile-by-stage` | False | 按 prefill / decode 阶段分别采集 |
| `--sglang-profile-with-stack` | False | 记录 Python 调用栈 |
| `--sglang-profile-record-shapes` | False | 记录张量形状信息 |
| `--sglang-profile-output-dir` | None | 自定义输出目录。默认 `traces/<tb_experiment_name>/sglang_trace` |

### 2. 训练 Profiling（PyTorch Profiler）

对 Actor 训练步骤进行 `torch.profiler` 采集，生成兼容 TensorBoard 的 trace 文件。

**示例用法** — 采集第 2、3、4 步（start/end 均 inclusive）：

```bash
python3 relax/entrypoints/train.py \
    --use-pytorch-profiler \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --tb-experiment-name my-experiment \
    # ... 其他参数
```

::: tip
`--profile-step-start` 和 `--profile-step-end` 均为 **inclusive**，是从本次训练启动后的 **步数偏移**，不是绝对 rollout ID。断点续训时计数从 0 重新开始。例如 start=2, end=4 采集 step 2, 3, 4（共 3 步）。

语义与 `--sglang-profile-step-start/end` 相同（两端均 inclusive）。
:::

**详细信息标志**：

| 标志 | 作用 |
|---|---|
| `--profile-with-stack` | 在每个 trace 事件中记录 Python 调用栈。用于定位触发高开销操作的代码路径 |
| `--profile-with-memory` | 在 trace 中跟踪 CUDA 显存分配/释放。用于发现显存尖峰 |
| `--profile-with-flops` | 估算每个算子的 FLOPs。用于计算硬件利用率 (MFU) |

**完整示例**：

```bash
python3 relax/entrypoints/train.py \
    --use-pytorch-profiler \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --profile-with-stack \
    --profile-with-memory \
    --profile-with-flops \
    --tb-experiment-name my-experiment \
    # ... 其他参数
```

::: warning
启用 `--profile-with-stack` 和 `--profile-with-memory` 会增加额外开销。建议仅在诊断时使用，不用于生产训练。
:::

### 3. GPU 内存 Profiling

记录 CUDA 显存分配/释放历史，用于诊断显存泄漏和 OOM 问题。在发生 OOM 时会自动 dump 内存快照。

**最小用法** — 开启记录 + 在第 2 步后主动 dump：

```bash
python3 relax/entrypoints/train.py \
    --record-memory-history \
    --memory-snapshot-num-steps 2 \
    --tb-experiment-name my-experiment \
    # ... 其他参数
```

**进阶参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--memory-snapshot-path` | snapshot.pickle | 快照文件名后缀 |
| `--memory-snapshot-dir` | None | 自定义输出目录。默认 `traces/<tb_experiment_name>/memory_snapshot` |
| `--memory-snapshot-num-steps` | None | 在指定步数后主动 dump 快照（0-indexed，设 3 表示第 2 步后 dump） |
| `--memory-recorder` | torch | 后端选择：`torch`（PyTorch 内置）、`memray`（需 `pip install memray`） |

查看快照：访问 [PyTorch Memory Viz](https://pytorch.org/memory_viz)，拖入生成的 `.pickle` 文件。

### 三种 Profiling 联合使用

实际诊断中，可同时开启三种 profiling 以获得全面视图。以下是一个完整的联合使用示例：

```bash
python3 relax/entrypoints/train.py \
    # --- SGLang 推理 Profiling ---
    --sglang-profile \
    --sglang-profile-step-start 2 \
    --sglang-profile-step-end 4 \
    # --- 训练 Profiling ---
    --use-pytorch-profiler \
    --profile-step-start 2 \
    --profile-step-end 4 \
    # --- 内存 Profiling ---
    --record-memory-history \
    --memory-snapshot-num-steps 2 \
    # --- 实验名（决定 trace 输出目录）---
    --tb-experiment-name my-profiling-run \
    # ... 其他训练参数
```

上述配置会产出如下目录结构：

```
traces/my-profiling-run/
├── sglang_trace/                          # SGLang 引擎 trace（按 rollout step 分目录）
│   ├── rollout_2/
│   │   ├── engine0-...-TP-0.trace.json.gz
│   │   ├── engine0-...-TP-1.trace.json.gz
│   │   ├── engine1-...-TP-0.trace.json.gz
│   │   └── ...
│   ├── rollout_3/
│   │   └── ...
│   └── rollout_4/
│       ├── engine0-...-TP-0.trace.json.gz
│       └── ...
├── train_trace/                           # 训练 trace
│   ├── train_overall_rank0_dp0_tp0_pp0.....pt.trace.json.gz
│   ├── train_overall_rank1_dp0_tp1_pp0.....pt.trace.json.gz
│   └── ...
└── memory_snapshot/                       # 内存快照
    ├── memory_snapshot_time..._rank0_dp0_tp0_pp0_snapshot.pickle
    ├── memory_snapshot_time..._rank1_dp0_tp1_pp0_snapshot.pickle
    └── ...
```

---

## 动态批处理

动态批处理根据样本长度动态打包，使每个 micro-batch 的总 Token 数接近目标值，相比固定大小的 micro-batch 能更好地利用 GPU。它同时也是一种有效的 OOM 防护机制——使用固定的 `--micro-batch-size` 时，一批异常长的序列可能超出 GPU 显存，而动态批处理通过 `--max-tokens-per-gpu` 限制每个 micro-batch 的总 Token 数，使显存使用可预测。

```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu 9216
```

使用 Context Parallelism (CP) 时，`--max-tokens-per-gpu` 应设为约 `max_response_len / cp_size`。

如果计算 log probs 是单独的瓶颈，可以为该阶段设置不同的 Token 预算：

```bash
--log-probs-max-tokens-per-gpu 12288
```

::: tip
如果训练中遇到 OOM，将固定的 `--micro-batch-size` 切换为 `--use-dynamic-batch-size` 并设置保守的 `--max-tokens-per-gpu` 通常是第一步。更多详情请参阅 [OOM 排查指南](./oom-troubleshooting.md)。
:::

---

## 并行配置

### 张量并行与序列并行

对于可放入单节点的模型，张量并行 (TP) + 序列并行 (SP) 是最常见的配置：

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel
```

较大模型（30B+）通常使用 TP=2 或 TP=4 并启用 SP。

### MoE 专家并行

对于 MoE 模型（如 Qwen3-30B-A3B），将专家分布到多个 GPU：

```bash
--expert-model-parallel-size 2 \
--expert-tensor-parallel-size 1
```

### 上下文并行

长上下文训练时，将序列拆分到多个 GPU：

```bash
--context-parallel-size 2
```

---

## 激活重计算

重计算以计算换内存。对大多数 RL 训练场景，建议启用重计算：

```bash
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1
```

该配置在所有层上均匀重计算激活。根据显存/计算的权衡调整 `--recompute-num-layers`。关于 `selective` 粒度和 `block` 方法的更多细节，请参考 [Megatron-LM 文档](https://github.com/NVIDIA/Megatron-LM)。

---

## 多模态处理并行化

训练多模态模型（如 Qwen3-VL）时，HuggingFace 的图像/视频数据处理器可能因 Python GIL 成为 CPU 瓶颈。`--mm-processor-pool-size` 参数创建 `ProcessPoolExecutor` 来绕过 GIL 竞争：

```bash
--mm-processor-pool-size 4
```

| 值 | 行为 |
|---|---|
| `0`（默认） | 使用 `ThreadPoolExecutor` — 受 GIL 竞争影响 |
| `> 0` | 创建指定数量 worker 的 `ProcessPoolExecutor`，实现真正的 CPU 并行 |

::: tip
建议从每 GPU 可用的 CPU 核数开始设置。例如，在 64 CPU、8 GPU 的节点上，可尝试 `--mm-processor-pool-size 8`。
:::

---

## SGLang 推理引擎调优

### 内存分配

控制 SGLang 为 KV Cache 预留的 GPU 显存比例：

```bash
--sglang-mem-fraction-static 0.8
```

通常取值在 0.6 到 0.85 之间。Colocate 模式下，较高的值（0.8）会减少训练可用的显存但提高推理吞吐。全异步模式下使用独立 GPU 时，可以设得更高。

### 推理 TP 大小

设置每个推理引擎实例使用的 GPU 数：

```bash
--rollout-num-gpus-per-engine 1
```

对于大模型，需要增大此值以满足模型最低 TP 需求。在可能的情况下使用 TP=1 可获得最佳的单请求吞吐。

---

## Partial Rollout（部分生成）

在长响应场景（如代码生成、思维链推理）中，等待所有样本完全生成完毕会导致训练 GPU 长时间空闲。Partial Rollout 通过允许未完成的样本被中断并回收到数据缓冲区来避免这个问题，使训练可以先使用已完成的样本继续进行：

```bash
--partial-rollout
```

### 控制中断频率

默认情况下，一个样本可以被无限次中断。设置 `--partial-rollout-max-aborted-count` 来保证样本在被中断一定次数后最终完成生成：

```bash
--partial-rollout \
--partial-rollout-max-aborted-count 3
```

### On-Policy 掩码

当一个样本被回收并在后续 rollout 中继续生成时，其早期 Token 是在之前的策略版本下生成的。使用 `--mask-offpolicy-in-partial-rollout` 来掩码这些 off-policy Token，使只有 on-policy 生成的 Token 参与训练：

```bash
--partial-rollout \
--mask-offpolicy-in-partial-rollout
```

::: tip
Partial Rollout 最适合 `max_response_len` 较大（如 8K+）且响应长度差异较大的工作负载。对于短且长度均匀的响应，回收的开销可能不值得。
:::

---

## 数据加载优化

### 流式数据集

对于内存放不下的超大数据集，使用流式模式：

```bash
--use-streaming-dataset \
--streaming-buffer-size 10000
```

### 数据均衡

在数据并行 rank 间均匀分配 Token 数量以减少空闲等待：

```bash
--balance-data
```

在静态/序列长度均衡路径上，`--balance-data` 会启用 `SeqlenBalancedSampler`，并使用 `karmarkar_karp` 在数据并行 rank 间均衡 Token。

全异步结合 `--use-dynamic-batch-size` 时，动态 batch 路径会通过 `StreamingTokenBudgetSampler` 自动做 DP token 均衡。此时传入 `--balance-data` 是允许的，但不会额外增加一次均衡。

::: warning
使用 `--balance-data` 时，同一 Prompt 的不同响应可能被分到不同训练步。
:::

---

## 权重更新流水线

对于参数量大的 MoE 模型，分块进行权重更新以避免显存尖峰：

```bash
--update-weight-buffer-size 536870912  # 512 MB
```

如果在权重同步期间观察到显存压力，可减小该值。

---

## 全异步训练

如需最大化吞吐，使用全异步训练流水线，为训练和推理分配独立的 GPU 集群：

```bash
--fully-async \
--max-staleness 1 \
--num-data-storage-units 1
```

### 陈旧度调优

`--max-staleness` 参数控制 rollout 数据相对于当前训练模型可以落后多少个版本。它直接影响吞吐量与数据新鲜度之间的权衡：

| 值 | 行为 |
|---|---|
| `1`（默认） | 训练仅消费当前或前一个版本的数据。吞吐较低但数据更新鲜 |
| `2-3` | 允许适度陈旧的数据。对大多数工作负载是较好的平衡 |
| 更高 | 减少训练等待时间从而提高吞吐，但数据可能由较旧的策略生成 |

::: tip
建议从 `--max-staleness 1` 开始，如果观察到训练进程频繁等待新鲜的 rollout 数据，再逐步增大。关注训练 loss 的稳定性——如果更高的陈旧度导致 loss 不稳定，则降低该值。
:::

完整配置请参阅[全异步训练流水线](./fully-async-training.md)。

---

## 推荐配置

### Qwen3-4B 8 GPU（Colocate）

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1 \
--use-dynamic-batch-size \
--max-tokens-per-gpu 9216 \
--sglang-mem-fraction-static 0.8 \
--colocate
```

### Qwen3-30B-A3B 16 GPU（全异步）

```bash
--tensor-model-parallel-size 2 \
--sequence-parallel \
--expert-model-parallel-size 2 \
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1 \
--optimizer-cpu-offload \
--sglang-mem-fraction-static 0.6 \
--fully-async
```

---

## 调优后 GPU 利用率

下表汇总了各 launch 脚本经性能调优后的 GPU 利用率实测结果，可作为对应配置的健康度参考基线。

| 任务 | GPU UTIL (%)（治理后） | GPU SM (%)（治理后） |
|---|---|---|
| `run-qwen3-4B-8xgpu` | 87 | 61 |
| `run-qwen3-30B-A3B-int4-8xgpu` | 71 | 47 |
| `run-qwen3-vl-4B-8xgpu` | 70 | 32 |
| `run_deepeyes` | 66 | 32 |
| `run-qwen35-9B-8xgpu-openr1mm-hybrid-async` | 69 | 42 |
| `run-qwen35-35B-A3B-16xgpu` | 64 | 35 |
| `run-qwen35-9B-8xgpu` | 78 | 64 |
| `run-qwen36-35B-A3B-8xgpu-image` | 63 | 35 |

---

## 下一步

- [OOM 排查指南](./oom-troubleshooting.md) — 调优导致内存问题时的处理方法
- [配置参考手册](./configuration.md) — 完整参数列表
- [调试指南](./debugging.md) — 隔离排查训练和推理问题
