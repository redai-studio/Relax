# ChunkFlow

ChunkFlow 在 Megatron 训练中把长序列拆成 chunk，按依赖安排前后向，并保留跨 chunk 的 K/V 状态及其梯度。它可以调整流水线中的工作粒度和激活保留方式，默认关闭，通过 `--use-chunkflow` 启用。

MoE 模型继续使用 Megatron 原生路由、dispatch/combine 和专家计算。切块也会增加调用开销，实际速度和显存收益需要在相同配置下与不切块训练比较。

## Qwen3-4B 示例

在已配置 Megatron、Megatron Bridge、Flash Attention 和 Ray 集群的 8 卡环境中，从仓库根目录运行以下命令。模型须位于 `${MODEL_DIR}/Qwen3-4B`，数据路径替换为已有的 SFT parquet 文件；示例启用 ClearML，需要配置相应访问凭据。

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/train.parquet
export CHUNK_SIZE=4096 KEEP_LAST_K=0
bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
```

数据列默认是 `question`、`answer`，可用 `INPUT_KEY`、`LABEL_KEY` 覆盖。数据格式见 [SFT 训练](./sft-training.md)。将命令最后的 `chunkflow` 换成 `baseline` 可运行不切块对照。

示例默认使用 TP2/PP4、CP1、GBS8、32K 长度上限、100 步和 FA3。这些是示例脚本的默认值；FA3 不可用时，设置 `RELAX_CHUNKFLOW_FLASH_BACKEND=fa2`。脚本会打印 `LOG_FILE`，并将日志同时输出到终端和该文件。

## 配置

在其他训练脚本中启用时，使用以下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--use-chunkflow` | 关闭 | 安装分块注意力并选择 ChunkFlow 调度。 |
| `--chunkflow-chunk-size` | `0` | 每个 chunk 的 token 数，启用时必须大于零。 |
| `--chunkflow-pack-policy` | `staggered` | 示例使用此策略；包含多个依赖段的 `lockstep` 单元不支持 RoPE 偏移。 |
| `--chunkflow-keep-last-k` | `0` | `0` 保留全部 chunk 的激活；正数只保留每组最后 K 个 chunk 的激活，更早的 chunk 在反向前重计算。示例中对应 `KEEP_LAST_K`。 |
| `--chunkflow-pipeline` | `0` | PP 大于 1 时，须设为与 `--pipeline-model-parallel-size` 相同的值。 |

PP 大于 1 使用实验性的非交错流水线执行器，还须设置 `RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1`。4B 示例默认设置该环境变量，并传入对应的 PP 参数。

## 兼容限制

- 当前路径要求 CP=1、packed THD、因果注意力、per-token loss 和单个 model chunk，不支持 VPP。
- 示例使用 dropout=0 并关闭 RoPE fusion。显式 MRoPE 位置路径目前仅支持纯文本 batch。
- GDN 需要支持初始/末状态梯度的算子及 Megatron `_forward_compute` 接口；须关闭预处理融合，不支持 deterministic 模式或 full-layer recompute。
- MoE 辅助损失、容量限制和丢弃 token 的配置需单独验证，不能假设切块前后等价。
- PP=1 暂不更新上游 PackingMetrics，性能对比应使用实际 step 耗时。

比较 loss、耗时和显存时，保持数据、随机种子、并行配置、精度及重计算配置一致。诊断可使用框架原有 profiler，参见[性能调优](./performance-tuning.md)。
