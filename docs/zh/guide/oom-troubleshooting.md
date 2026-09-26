# OOM 排查指南

Relax RL 训练中 CUDA 显存不足 (OOM) 错误的诊断与解决实践指南。本文提到的所有参数均可在[配置参考手册](./configuration.md)中查阅。

---

## 诊断 OOM

### 步骤 1：确定 OOM 发生阶段

Relax 中的 OOM 错误通常发生在以下阶段：

| 阶段 | 表现 | 常见原因 |
|---|---|---|
| 模型初始化 | 启动时 OOM | 模型对于可用 GPU 来说太大 |
| 前向传播（训练） | `train_actor` 期间 OOM | `--max-tokens-per-gpu` 过高或未启用重计算 |
| Log probs 计算 | `train_log_probs` 期间 OOM | 长序列消耗过多激活显存 |
| 权重同步 | `update_weights` 期间 OOM | 权重缓冲区对于剩余 GPU 显存来说太大 |
| NCCL 通信 | all-reduce/all-gather 内部 OOM | 通信缓冲区显存不足 |

### 步骤 2：捕获内存快照

Relax 提供内置的内存分析工具来捕获详细的分配信息。

#### PyTorch 内存快照

记录内存分配历史并自动导出快照：

```bash
python3 relax/entrypoints/train.py \
    --memory-snapshot-dir /path/to/snapshots \
    --memory-snapshot-num-steps 3 \
    --memory-recorder torch \
    # ... 其他参数
```

这会从一开始就记录内存分配历史，并在指定步数后导出快照。如果发生 OOM，会在故障点自动导出快照。

使用 [PyTorch Memory Visualizer](https://pytorch.org/memory_viz) 可视化快照：

```bash
python -m torch.utils.viz._memory_viz trace_plot /path/to/snapshots/*.pickle -o memory_trace.html
```

#### Memray 分析器

如需 CPU+GPU 内存分析，使用 memray 记录器：

```bash
python3 relax/entrypoints/train.py \
    --memory-snapshot-dir /path/to/snapshots \
    --memory-snapshot-num-steps 3 \
    --memory-recorder memray \
    # ... 其他参数
```

::: warning
使用 memray 时必须设置 `--memory-snapshot-num-steps`。
:::

### 步骤 3：启用 NCCL 通信内存检查

当 OOM 发生在 NCCL 集合通信操作内部时，标准调用栈可能无法显示可用显存量。`--enable-cuda-memory-check` 标志在每个底层 NCCL 调用周围添加内存监控：

```bash
python3 relax/entrypoints/train.py \
    --enable-cuda-memory-check \
    # ... 其他参数
```

启用后：

- 每次 NCCL 调用（all-reduce、all-gather、broadcast 等）前检查可用 GPU 显存。
- 如果可用显存低于 5 GB，自动调用 `torch.cuda.empty_cache()` 回收碎片化显存。
- 如果 NCCL 调用失败，内存信息会附加到异常中用于诊断。

::: tip
`--enable-cuda-memory-check` 会导致约 **20% 的训练性能劣化**。建议仅在调试时使用，不用于生产训练。
:::

### 步骤 4：使用 Profiler 内存追踪

进行更细粒度的分析时，在 PyTorch Profiler 中启用内存追踪：

```bash
python3 relax/entrypoints/train.py \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --profile-with-memory \
    --use-tensorboard \
    --tb-project-name /path/to/tb_logs \
    # ... 其他参数
```

`--profile-with-memory` 标志在 profiler trace 中记录 CUDA 显存分配/释放，可在 TensorBoard 的 Memory 视图中查看。

---

## 常见解决方案

### 1. 启用激活重计算

减少训练显存最有效的方法。以计算时间换取显存：

```bash
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 1
```

所有标准的 Relax 训练脚本都使用此配置，强烈推荐启用。

### 2. 降低 Max Tokens Per GPU

降低 `--max-tokens-per-gpu` 以减少每个 micro-batch 中打包的 Token 数量：

```bash
# 之前（OOM）
--max-tokens-per-gpu 12288

# 调整后（降低）
--max-tokens-per-gpu 8192
```

如需单独控制 log probs 计算的内存预算：

```bash
--log-probs-max-tokens-per-gpu 8192
```

### 3. 启用动态批处理防止 OOM

使用固定的 `--micro-batch-size` 时，一批异常长的序列可能超出 GPU 显存。动态批处理通过限制每个 micro-batch 的总 Token 数，使显存使用可预测，从而防止因变长输入导致的 OOM：

```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192
```

从保守的 `--max-tokens-per-gpu` 值开始，逐步增大。该参数替代 `--micro-batch-size`——启用动态批处理后，micro-batch 大小会根据 Token 预算自动确定。

### 4. 使用固定 Micro-Batch Size

如果已经在使用动态批处理仍然遇到 OOM，可能是 `--max-tokens-per-gpu` 设置过高。也可以切换到固定的 micro-batch size 并使用较短的序列：

```bash
# 去掉 --use-dynamic-batch-size 并显式设置
--micro-batch-size 1
```

### 5. 启用优化器 CPU Offload

将优化器状态（Adam 动量）移到 CPU 内存。对大模型（30B+）至关重要：

```bash
--optimizer-cpu-offload
```

如需更好的 CPU offload 性能，重叠数据传输：

```bash
--optimizer-cpu-offload \
--overlap-cpu-optimizer-d2h-h2d
```

### 6. 重计算损失函数

通过重计算损失函数而非缓存中间结果来节省显存：

```bash
--recompute-loss-function
```

### 7. 分块计算 Log Probs

将 log probs 计算拆分为更小的块以降低峰值显存：

```bash
--log-probs-chunk-size 4
```

值为 `-1`（默认）时一次性全部计算。更小的值使用更少的显存但耗时更长。

### 8. 减小权重更新缓冲区

对于参数量大的 MoE 模型，权重更新缓冲区可能消耗大量显存。减小缓冲区大小：

```bash
# 默认是 512 MB
--update-weight-buffer-size 268435456  # 256 MB
```

### 9. 禁用权重备份

权重备份器在主机内存中保留一份模型权重的副本用于恢复。禁用它可以节省主机内存：

```bash
--disable-weights-backuper
```

::: warning
禁用权重备份器意味着训练失败时无法自动恢复权重。
:::

### 10. 调整训练内存预留

Relax 预留内存以防止碎片化。调整预留量：

```bash
# 默认是 1 GB (1073741824 字节)
--train-memory-margin-bytes 536870912  # 512 MB
```

### 11. 调整 SGLang 显存比例

在 colocate 模式下，SGLang 和训练共享 GPU 显存。降低 SGLang 的分配比例为训练留出更多空间：

```bash
# 默认因场景而异，典型值
--sglang-mem-fraction-static 0.7  # 从 0.8 下调
```

### 12. 平衡 Pipeline Parallel 各 stage 的层数

启用 PP 后如果发现各 stage 显存占用明显不均，通常是最后一个 stage 占用更多（需要承担 loss 计算、输出 embedding 等额外开销）。可以通过把更多层放到第一个 stage 来减轻最后一个 stage 的负担：

```bash
# 例：Qwen3.5 35B 共 40 层，PP=2
# 默认均分时每 stage 20 层，最后一个 stage 容易 OOM
--decoder-first-pipeline-num-layers 30
```

类似地可以使用 `--decoder-last-pipeline-num-layers` 单独减少最后一个 stage 的层数。调整时建议小步迭代，每次只移动 1~2 层并观察各 stage 的显存水位。

### 13. 启用 `expandable_segments`（需要 `--selective-offload`）

`expandable_segments:True` 可以缓解碎片化，但它**与 `torch_memory_saver` 不兼容**——后者是 `--offload-train` 默认使用的机制。TMS 无法追踪基于 VMM 的 expandable segment，而且它的 hook 由预加载 `.so` 内部的 `TMS_INIT_ENABLE` 在任何 Python 代码执行之前武装，所以它自带的检查根本不会运行。你拿到的是一行裸的驱动错误，而不是可读的提示：

```
[torch_memory_saver.cpp] CUresult error: 1 (invalid argument) file=csrc/utils.h func=cu_mem_create line=169
```

满足两个条件即可使用：

1. **使用 `--selective-offload`** —— 应用层 offload，完全不加载 `torch_memory_saver` hook，冲突随之消失。
2. **用 `--train-env-vars` 把变量限定到训练 actor**，而不是写进 `RUNTIME_ENV_JSON`：

```bash
--selective-offload \
--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' \
```

`--train-env-vars` 只会合并进训练 actor 的环境，不会进入 rollout 的环境，因此 SGLang 那边仍然正常使用 `torch_memory_saver`。

::: danger
**不要**把 `expandable_segments:True` 放进 `RUNTIME_ENV_JSON`——那份环境是全局的，rollout worker 也会拿到。SGLang 的 `release_memory_occupation` / `resume_memory_occupation`（由 `--offload-rollout` 启用）依赖 `torch_memory_saver` 且**没有退路**，推理侧的 offload 会直接失效。
:::

::: tip
torch 2.9 已将 `PYTORCH_CUDA_ALLOC_CONF` 标记为废弃，建议改用 `PYTORCH_ALLOC_CONF`。注意 `torch_memory_saver` 自带的检查只识别旧名字，所以用新名字时它完全不会报警。
:::

colocate 的权重同步通过 CUDA IPC 与 SGLang 共享 tensor，expandable segment 的 tensor 通过 `pidfd_open`/`pidfd_getfd` 传递句柄——在常规内核和容器下可以正常工作。只有 ptrace 策略被加固时才会失败，报错形式为 `Tensors allocated with expandable_segments:True cannot be shared between processes`。

---

## 特定阶段的 OOM

### 训练前向传播 OOM

**症状**：`train_actor` 步骤期间 OOM。

**排查清单**：
1. 启用重计算：`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`
2. 启用 `--use-dynamic-batch-size` 并设置保守的 `--max-tokens-per-gpu`
3. 降低 `--max-tokens-per-gpu`
4. 启用 `--recompute-loss-function`
5. 尝试 `--optimizer-cpu-offload`
6. 如果开了 PP 且最后一个 stage 显存明显更高，用 `--decoder-first-pipeline-num-layers` 把更多层放到第一个 stage

### Log Probs 计算 OOM

**症状**：`train_log_probs` 步骤期间 OOM。

**排查清单**：
1. 设置更低的 `--log-probs-max-tokens-per-gpu`（与训练分开控制）
2. 使用 `--log-probs-chunk-size 4` 分块计算
3. 降低 `--max-tokens-per-gpu`

### 权重同步 OOM

**症状**：`update_weights`（训练到推理的权重传输）期间 OOM。

**排查清单**：
1. 减小 `--update-weight-buffer-size`
2. 在 colocate 模式下降低 `--sglang-mem-fraction-static`
3. 尝试 `--disable-weights-backuper`

### NCCL 通信 OOM

**症状**：NCCL 调用内部 OOM，调用栈不透明。

**排查清单**：
1. 启用 `--enable-cuda-memory-check` 获取故障点的详细内存信息
2. 使用 `--memory-snapshot-dir` 和 `--memory-recorder torch` 捕获内存快照
3. 如果内存预留过多，减小 `--train-memory-margin-bytes`
4. 如果碎片化是问题所在，增大 `--train-memory-margin-bytes`

---

## 快速参考

| 目标 | 参数 |
|---|---|
| 减少激活显存 | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` |
| 限制每批次 Token 数（防止 OOM） | `--use-dynamic-batch-size --max-tokens-per-gpu <值>` |
| 减少每批次显存 | `--max-tokens-per-gpu <更小的值>` |
| 将优化器移到 CPU | `--optimizer-cpu-offload` |
| 重计算损失函数 | `--recompute-loss-function` |
| 分块计算 log probs | `--log-probs-chunk-size 4` |
| 减少权重同步显存 | `--update-weight-buffer-size <更小的值>` |
| 节省主机内存 | `--disable-weights-backuper` |
| 调试 NCCL OOM | `--enable-cuda-memory-check` |
| 捕获内存快照 | `--memory-snapshot-dir <路径> --memory-recorder torch` |
| 分析显存使用 | `--profile-with-memory` |
| 调整内存预留 | `--train-memory-margin-bytes <字节数>` |
| SGLang 显存（colocate） | `--sglang-mem-fraction-static <比例>` |
| 平衡 PP stage 层数 | `--decoder-first-pipeline-num-layers <层数>` |
| 缓解显存碎片 | `--selective-offload` + `--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'` |

---

## 下一步

- [性能调优](./performance-tuning.md) — 解决 OOM 后最大化吞吐量
- [配置参考手册](./configuration.md) — 完整参数列表
- [调试指南](./debugging.md) — 隔离排查训练和推理问题
