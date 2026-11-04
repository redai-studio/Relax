# ChunkFlow

ChunkFlow 在 Megatron 训练中把长序列拆成 chunk，按依赖安排前后向，并保留跨 chunk 的 K/V 状态及其梯度。它可以调整流水线中的工作粒度和激活保留方式，默认关闭，通过 `--use-chunkflow` 启用。

MoE 模型继续使用 Megatron 原生路由、dispatch/combine 和专家计算。切块也会增加调用开销，实际速度和显存收益需要在相同配置下与不切块训练比较。

## 启用方式

在现有 Megatron 训练脚本中加入以下参数片段，模型、数据和资源配置沿用原脚本。SFT 数据准备和启动流程见 [SFT 训练](./sft-training.md)。

```text
--use-chunkflow
--chunkflow-chunk-size 4096
--context-parallel-size 1
--calculate-per-token-loss
--no-rope-fusion
--attention-dropout 0.0
--hidden-dropout 0.0
```

`4096` 是示例 chunk 大小，可按任务调整。运行不切块对照时，移除 `--use-chunkflow`，保留其他训练配置。

ChunkFlow 默认使用 FA2。使用 FA3 时，训练环境须安装对应接口，并设置 `RELAX_CHUNKFLOW_FLASH_BACKEND=fa3`。

## 配置

相关参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--use-chunkflow` | 关闭 | 安装分块注意力并选择 ChunkFlow 调度。 |
| `--chunkflow-chunk-size` | `0` | 每个 chunk 的 token 数，启用时必须大于零。 |
| `--chunkflow-pack-policy` | `staggered` | 包含多个依赖段的 `lockstep` 单元不支持 RoPE 偏移。 |
| `--chunkflow-keep-last-k` | `0` | `0` 保留全部 chunk 的激活；正数只保留每组最后 K 个 chunk 的激活，更早的 chunk 在反向前重计算。 |
| `--chunkflow-pipeline` | `0` | PP 大于 1 时，须设为与 `--pipeline-model-parallel-size` 相同的值。 |

PP 大于 1 使用实验性的非交错流水线执行器，还须设置 `RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1`。通过 Ray 提交时，将所需的 ChunkFlow 环境变量加入 `runtime_env.env_vars`，确保训练 worker 能读取。

## 兼容限制

- 当前路径要求 CP=1、packed THD、因果注意力、per-token loss 和单个 model chunk，不支持 VPP。
- 使用 dropout=0 并关闭 RoPE fusion。显式 MRoPE 位置路径目前仅支持纯文本 batch。
- GDN 需要支持初始/末状态梯度的算子及 Megatron `_forward_compute` 接口；须关闭预处理融合，不支持 deterministic 模式或 full-layer recompute。
- MoE 辅助损失、容量限制和丢弃 token 的配置需单独验证，不能假设切块前后等价。
- PP=1 暂不更新上游 PackingMetrics，性能对比应使用实际 step 耗时。

比较 loss、耗时和显存时，保持数据、随机种子、并行配置、精度及重计算配置一致。诊断可使用框架原有 profiler，参见[性能调优](./performance-tuning.md)。
