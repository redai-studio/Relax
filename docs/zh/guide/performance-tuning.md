# 性能调优

Relax 训练吞吐量最大化实践指南。本文提到的所有参数均可在[配置参考手册](./configuration.md)中查阅。

---

## 按 FLOPs 平衡动态微批

`--balance-by-flops` 根据样本训练 FLOPs 的估算值构造全局微批，并将整微批分配给数据并行（DP）rank，以均衡计算负载。开关默认 `False`，开启后同时启用微批构造和 DP 分配。

### 配置方式

Qwen3.5 文本训练可在训练配置中加入以下参数：

```bash
--train-backend megatron \
--qkv-format thd \
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192 \
--sft-packing-strategy balanced \
--polling-mode True \
--balance-by-flops
```

请根据模型和可用显存设置 token 预算，示例值为 8192。自定义 YAML 使用布尔字段 `balance_by_flops: true`。SFT 专用设置见 [SFT 配置](./sft-training.md#flops-感知调度)。

### 调度过程

Relax 为每个完整全局 batch 执行以下流程，参考 [Slime PR #2017](https://github.com/THUDM/slime/pull/2017)：

1. 计算初始全局微批数 `K = ceil(总长度 / (max_tokens_per_gpu × CP))`，最多为样本数。Allgather CP 会先将每卡预算向下取整到 `TP × data_pad_size_multiplier` 的倍数。
2. 以样本 FLOPs 为权重，用 Karmarkar–Karp（KK）分区，允许各微批包含不同数量的样本。
3. 拆分 token 总量最大的可拆分 bin，直到全局微批数能被 DP 整除。每次拆分按长度从大到小，将样本放入当前 token 总量较小的半边。
4. 以整微批的总 FLOPs 为权重，用 KK 分配给 DP，保证每个 rank 的微批数相同。
5. PP=1 且 DP>1 时，各 rank 按微批 FLOPs 升序执行，让计算量相近的微批彼此配对。

TransferQueue 将微批成员、顺序和数量传给普通 iterator 与 SFT CPU prepack。各 DP rank 收到的样本数可以不同，训练端保留全局 batch 的优化器更新边界和 loss 归一化方式。Eval 和 forward-only 调用使用各自的 token 预算。

::: warning Token 预算与显存
`--max-tokens-per-gpu` 用于计算初始微批数，最终 pack 的 token 数可能超过该预算。请根据实际 pack 大小预留显存余量。
:::

Producer 在 CPU 上估算样本训练成本，写入队列元数据，作为微批构造和 DP 分配的权重。Qwen3.5 估算器覆盖 full attention、Gated DeltaNet（GDN）、Dense MLP、激活的 MoE 专家和共享专家，以及词表输出头。一次乘加计为两个 FLOPs，训练成本近似为前向的三倍。

### 支持范围

| 配置项 | 要求 |
| --- | --- |
| 模型 | HF `model_type` 为 `qwen3_5`、`qwen3_5_text`、`qwen3_5_moe` 或 `qwen3_5_moe_text`，使用标准 Dense 或所有层均为 MoE 的布局。 |
| 训练 | 全参数、纯文本、同步 SFT 或 RL 训练。 |
| 后端与打包 | Megatron、THD、balanced 动态打包和 TransferQueue polling 模式。MTP、wholepack 和 `--debug-train-only` 保持关闭。 |
| 并行 | VPP=1、静态 CP；GDN CP>1 配置中保持 `allgather_cp: false`。SFT 异步 CPU prepack 使用 CUDA、PP=1、CP=1。 |
| MoE | `--moe-aux-loss-coeff 0`，`moe_expert_capacity_factor: null`。Router 均衡模式为 `none`、`aux_loss`、`seq_aux_loss` 或 `global_aux_loss`。 |
| SFT | 一个 TransferQueue shard，`--n-samples-per-prompt 1`。 |

SFT 使用 `docker/patch/transferqueue/9784ad0-assignment-transform.patch` 中的 TransferQueue `assignment_transform` 补丁。仓库的 `docker/Dockerfile` 和 `docker/Dockerfile.cu13` 会在构建镜像时应用该补丁。

### 恢复与排查

队列计划随 sampler checkpoint 保存。恢复时，请保持 DP 数、FLOPs 开关、token 预算和 CP/PP 配置一致。

`Cannot split ... into ... aligned microbatches` 表示物理样本行数少于 DP 对齐所需的微批数，请调整 batch 大小或 DP 数，确保有足够的样本行。

评估吞吐时，使用相同的模型、数据和训练配置，对比 FLOPs 调度与长度均衡基线的有效训练 token/s 和等待时间。

## Kimi K3 TFLOPS 与 MFU

`perf/actor_train_tflops` 估算每卡每秒的有效模型 FLOPs。`perf/mfu/actor_train` 再除以设备 BF16 dense 峰值（280G Blackwell 配置为 2250 TFLOPS），记录的是比例，0.4 表示 40%。时间分母是 actor 训练耗时，不是包含 rollout/save 的整轮耗时；训练内部的 optimizer 和 offload 耗时会降低该指标。

K3 分别计算 KDA 投影、卷积和状态递推；MLA 低秩/输出门投影及因果注意力；latent MoE 激活专家、router 和共享专家；词表输出；以及 MoonViT-V2 和 PatchMergerMLPV2。使用每条样本的实际长度和原始图片 `(T,H,W)` 网格：视觉编码器先对所有帧做注意力，再进行时间池化。沿不包含 CP 副本的 DP 组汇总样本元数据后，只除一次训练 GPU 数量。

一次乘加计为两个 FLOPs。可训练文本部分使用常规的三倍前向近似；视觉编码器和 projector 根据 `--freeze-vision-model`、`--freeze-vision-projection` 及输入梯度需求分别计算反向开销。`--freeze-language-model` 下仍可向可训练的视觉输入传播梯度。Actor/reference log-prob 前向 FLOPs 单独估算，不再把含冻结模块的训练 FLOPs 简单除以三。

这是主要矩阵计算的模型估算，不是实际 CUDA 指令计数。不计入 padding、重计算、通信、embedding 查表、AttnRes 归约、归一化及逐元素运算。KDA 每 token 计算三次状态 contraction（前向 FLOPs 为 6 × heads × head_dim²），不计入具体实现的 chunk/WY 开销。硬件 profiler 的统计口径与此不同。

K3 使用 LoRA 或自定义 `--freeze-params-name-list` / `--only-train-params-name-list` 时，HF 配置无法独立确定 adapter 与参数的实际可训练状态，因此暂不报告 FLOPs/MFU，并打印一次说明；耗时和 token 吞吐继续保留。其他模型的估算口径不变。

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

动态批处理将变长样本打包，使每个 micro-batch 的 Token 数接近目标值，以提高 GPU 利用率。默认打包路径按 token 预算检查 pack 容量。`--balance-by-flops` 使用该预算推算初始微批数，详见[按 FLOPs 平衡动态微批](#按-flops-平衡动态微批)。

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

`--balance-by-flops` 同时启用 FLOPs 微批构造和 DP 分配，详见[配置说明](#按-flops-平衡动态微批)。

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
