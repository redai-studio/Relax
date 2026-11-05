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

ChunkFlow 调用模型原有的 Megatron/Transformer Engine Attention，由原实现选择底层 kernel。旧脚本中的 `RELAX_CHUNKFLOW_FLASH_BACKEND` 需要移除，继续设置会报错。

连接层负责补上历史 K/V，并使用右下对齐的因果 mask，让当前 query 正确访问前文。Attention 使用新建的实际 packed 序列边界，RoPE 的位置偏移单独保留。模块对象、参数及 checkpoint 键保持不变。原生 autograd 可能为每个 chunk 保存拼接后的历史 K/V，因此从旧的自定义 Attention 反向路径迁移后，需要重新测量显存。

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

- 内置 Dense 适配器只接受 Megatron `SelfAttention` 与 `TEDotProductAttention` 的精确类型组合。其他类型及子类需要显式注册适配器。非 TE 的 `DotProductAttention` 不支持所需的 packed THD 调用，框架不会用自定义 kernel 替代它。
- 当前路径要求 CP=1、packed THD、因果注意力、per-token loss 和单个 model chunk，不支持 VPP。
- 使用 dropout=0 并关闭 RoPE fusion。显式 MRoPE 位置路径目前仅支持纯文本 batch。
- Attention 重计算使用 `--chunkflow-keep-last-k`。Megatron full-layer recompute 和 selective `core_attn` 重计算会在没有 ChunkFlow 重放标记的情况下重复写入缓存，因此会被拒绝。
- 滑动窗口、非 vanilla softmax、QK clipping/logit 统计、外部 Attention mask/bias 和 Qwen3.5 optimized recipe 需要单独适配。QSA 和带 PLE 的模型会被拒绝：保留原模块并不能自动补齐稀疏选择、n-gram 或卷积所需的历史信息。
- GDN 需要支持初始/末状态梯度的算子及 Megatron `_forward_compute` 接口；须关闭预处理融合，不支持 deterministic 模式或 full-layer recompute。
- MoE 辅助损失、容量限制和丢弃 token 的配置需单独验证，不能假设切块前后等价。
- PP=1 暂不更新上游 PackingMetrics，性能对比应使用实际 step 耗时。

比较 loss、耗时和显存时，保持数据、随机种子、并行配置、精度及重计算配置一致。诊断可使用框架原有 profiler，参见[性能调优](./performance-tuning.md)。

## 接入新的 Attention

在每个训练 worker 安装 ChunkFlow 前，通过 `relax.backends.megatron.chunkflow.adapters` 中的 `register_attention_adapter(attention_type, core_type, adapter)` 注册 `AttentionAdapter`，例如在自定义 model provider 中注册。外层和核心类都必须精确匹配，重复注册会报错。Attention 安装前先执行兼容性检查；安装器返回卸载回调，如果自身安装失败，需要撤销已经做出的修改。

如果算子的历史信息就是普通 K/V，可以用 `dense_attention_adapter(name=..., execute=..., validate=...)` 复用 ChunkFlow 的状态和梯度连接。执行函数接收保存的原生 `forward` 和 `DenseAttentionCall`，后者包含 THD query、K/V 片段、实际 Q/KV 边界、CPU 长度和最大长度。执行函数负责以右下对齐的因果关系调用原生算子，并保留原生 autograd，不能把 RoPE 偏移当作实际 padding。其他历史状态可以通过整层适配器管理；将稀疏算子注册成 Dense，并不等于实现了其 indexer 状态传递。

GDN 继续通过现有状态适配层调用 Megatron 使用的卷积和递推 kernel。ChunkFlow 传递初始/末状态并连接状态梯度，不实现 GDN 计算。新增适配器用于训练前，应比较不切块和切块执行的输出、输入及参数梯度，并验证 packed 序列边界和 ChunkFlow 重计算。
