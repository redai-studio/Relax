# 编译缓存复用

Relax 支持在多次训练启动之间复用节点本地的 TorchInductor 与 Triton 编译产物。共享文件系统保存不可变的缓存增量，各 GPU 节点在训练 Actor 启动前将其恢复到本地可写目录。

## 概述

编译缓存复用是可选功能。将 `RELAX_KERNEL_CACHE_DIR` 设置为所有节点均可见的绝对路径，然后通过标准入口启动训练。Relax 会在任务启动前恢复兼容的缓存文件，并在运行期间周期性回传新文件，在训练 Driver 退出时执行最终回传。

只有 Megatron 训练 Actor 会收到 `TORCHINDUCTOR_CACHE_DIR` 与 `TRITON_CACHE_DIR`。Ray Serve、Transfer Queue 和 SGLang 进程不会写入该缓存。

缓存内容包括 TorchInductor 与 Triton 生成的 Python wrapper、共享库、PTX、cubin、中间表示、autotuning 结果和元数据。它不包含模型权重、optimizer 状态、数据、NCCL communicator、DeepEP buffer 或进程内 CUDA 状态。

## 架构

```text
┌────────────────────┐     restore      ┌────────────────────┐
│ Shared Delta Store │ ───────────────> │ Node-local Cache   │
└─────────▲──────────┘                  └─────────┬──────────┘
          │ periodic/final publish               │
          │                                      ▼
┌─────────┴──────────┐                  ┌────────────────────┐
│ Detached Cache     │ <── heartbeat ── │ Training Driver    │
│ Agent per GPU Node │                  └────────────────────┘
└────────────────────┘
```

使用 `ray-job.sh` 时，Head 进程会验证所有存活 GPU 节点的 build fingerprint 完全一致，然后在每个 GPU 节点启动一个 detached cache agent。使用 local 或 SPMD 入口时，每个节点会在加入 Ray 前恢复本地缓存，之后由 Head attach cache agent。

## 快速开始

指定共享目录，然后使用常规训练入口：

```bash
export RELAX_KERNEL_CACHE_DIR="$(pwd)/.cache/relax-kernels"

bash scripts/entrypoint/ray-job.sh \
    scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh
```

Shell 展开后的路径必须是绝对路径，并且所有 GPU 节点都能访问。未设置 `RELAX_KERNEL_CACHE_DIR` 时，该功能关闭。

其他标准入口使用相同的启用变量：

```bash
# 本地启动；训练脚本仍按常规方式 source local.sh。
export RELAX_KERNEL_CACHE_DIR="$(pwd)/.cache/relax-kernels"
bash scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh

# SPMD 启动；每个 Pod 调用入口前都需要导出该变量。
export RELAX_KERNEL_CACHE_DIR="/shared/relax-kernels"
bash scripts/entrypoint/spmd-multinode.sh \
    scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `RELAX_KERNEL_CACHE_DIR` | 未设置 | 共享目录的绝对路径；设置后启用该功能。 |
| `RELAX_KERNEL_CACHE_KEY` | 自动生成 | Graph 与拓扑 profile；自动值对训练脚本、额外参数及部分模型/并行覆盖项计算哈希。 |
| `RELAX_KERNEL_CACHE_LOCAL_DIR` | 自动生成 | 节点本地可写根目录；Relax 默认在 `/tmp/relax-kernel-cache/` 下派生稳定路径。 |
| `RELAX_KERNEL_CACHE_BUILD_KEY` | 空 | 可选的人工 build 区分值，用于隔离自动 fingerprint 未覆盖的变化。 |
| `RELAX_KERNEL_CACHE_COMPRESSION` | `none` | 增量归档压缩方式：`none` 或 `gzip`。 |
| `RELAX_KERNEL_CACHE_SYNC_INTERVAL_SEC` | `900` | 周期快照间隔。 |
| `RELAX_KERNEL_CACHE_LEASE_TIMEOUT_SEC` | `600` | attach 后的 agent 最长 heartbeat 丢失时间，超过后执行 finalize。 |
| `RELAX_KERNEL_CACHE_STARTUP_TIMEOUT_SEC` | `7200` | 尚未被 Driver claim 的启动期 agent 最长存活时间。 |
| `RELAX_KERNEL_CACHE_HEARTBEAT_INTERVAL_SEC` | `30` | Driver heartbeat 间隔。 |
| `RELAX_KERNEL_CACHE_EXIT_TIMEOUT_SEC` | `600` | Driver 等待最终回传的最长时间。 |

通常只需设置 `RELAX_KERNEL_CACHE_DIR`。只有在两个任务明确使用相同 graph 与拓扑时，才使用稳定的显式 key：

```bash
export RELAX_KERNEL_CACHE_DIR="/shared/relax-kernels"
export RELAX_KERNEL_CACHE_KEY=example

bash scripts/entrypoint/ray-job.sh \
    scripts/training/sft/run-qwen3.5-397B-A17B-vl-lora-mtp-sft-128k-128xgpu.sh
```

::: warning
修改模型 graph、LoRA target、TP/PP/CP/EP/VPP 布局、attention backend、重计算策略、MTP 设置或 token/image shape 策略后，不要继续复用原显式 key。应移除该覆盖项或使用新的 key。
:::

## 兼容性与生命周期

Relax 使用两个标识隔离缓存数据：

- Profile key 覆盖训练脚本、命令行覆盖项，以及部分模型、并行、LoRA、attention、重计算和 shape 设置。
- Build fingerprint 覆盖 Python 与编译器相关包版本、GPU 型号与 compute capability、镜像/源码版本，以及部分 Megatron 源文件。

在 `ray-job` 模式下，所有存活 GPU 节点必须返回相同的 build fingerprint。CPU-only Ray Head 不会被用作编译兼容性的参考。发现不一致时，Relax 会在恢复缓存或提交训练之前失败。

每个节点会把兼容的 ready 增量恢复到本地可写缓存。训练期间，detached agent 周期性发布已完成的文件。正常结束、已传播到 Driver 的训练失败，或执行 `ray job stop` 时，Driver 会请求最终快照，并最多等待 `RELAX_KERNEL_CACHE_EXIT_TIMEOUT_SEC`。

::: warning
`SIGKILL`、节点丢失、Pod 驱逐或 Ray 集群终止无法执行 final hook。此时只能恢复此前已经发布的周期增量。抢占风险较高时可缩短同步间隔，但需要权衡共享存储 I/O。
:::

## 最佳实践

1. 跨任务使用不可变镜像，并保持 Torch、Triton、Transformer Engine、FLA、DeepEP、Megatron 与 Relax 版本一致。
2. 共享存储只作为增量仓库，活跃编译目录保持在节点本地且可写。不要把 `TORCHINDUCTOR_CACHE_DIR` 或 `TRITON_CACHE_DIR` 直接指向共享存储。
3. 常规启动优先使用自动 key。若环境或 patch 变化未被自动 fingerprint 覆盖，使用 `RELAX_KERNEL_CACHE_BUILD_KEY` 主动隔离。
4. 覆盖有代表性的 pipeline stage 和动态 shape bucket 后，再认为缓存已经充分预热。一个 shape 命中不代表未见过的 token 或图片 shape 不再编译。
5. 在大规模集群使用较短同步间隔前，先监控共享盘容量和快照 I/O。

## 故障排除

### 已恢复缓存但训练仍在编译

遇到未见过的 shape 时仍会编译，这是正常现象。同时检查训练 Actor 日志中是否包含已解析的节点本地目录。编译器专用环境变量只会注入训练 Actor。

### 没有恢复任何缓存

检查以下项目：

1. `RELAX_KERNEL_CACHE_DIR` 是绝对路径，并且所有节点均可访问。
2. Profile key 与 build fingerprint 能匹配此前发布的 ready 增量。
3. 上一个任务运行时间足以产生周期快照或最终快照。
4. 共享目录中的 archive 和 manifest 不是 partial 状态。

### GPU 节点 fingerprint 不一致

检查容器镜像、GPU 架构、已安装的编译相关包、Relax checkout 和节点本地 Megatron 文件。不要强制让不兼容节点共用 fingerprint；应修复环境或使用独立缓存命名空间。

## 下一步

- [性能调优](./performance-tuning.md) — 衡量冷启动与稳态吞吐。
- [调试指南](./debugging.md) — 分布式启动失败时收集诊断证据。
- [配置说明](./configuration.md) — 配置 Relax 训练任务。
