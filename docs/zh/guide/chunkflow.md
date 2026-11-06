# ChunkFlow

ChunkFlow 是 `examples/chunkflow/` 下的独立示例，在 Megatron 训练中把长序列拆成 chunk，按依赖安排前后向，并保留跨 chunk 的状态及其梯度。实现、参数和测试都由 example 管理；Relax 核心通过示例附带的接入补丁启用它。

MoE 模型继续使用 Megatron 原生路由、dispatch/combine 和专家计算。切块也会增加调用开销，实际速度和显存收益需要在相同配置下与不切块训练比较。

## 固定版本与安装

`examples/chunkflow/relax.version` 固定 Relax 源码基线为 `d548769b77f12e445e464f99d4332290a3e5bd3d`，即引入 ChunkFlow 前的版本。使用该版本对应的 Megatron/Bridge、Transformer Engine 和 FLA 训练环境，不额外安装另一套 Attention 后端。

在该版本的 Relax checkout 中放入完整的 `examples/chunkflow/`，从仓库根目录执行：

```bash
python examples/chunkflow/setup.py
python examples/chunkflow/setup.py --check
```

安装器先对照固定 commit 检查 Relax 源码，再应用 `examples/chunkflow/relax.patch`。补丁只连接参数解析、模型安装、训练调度，以及分块位置传递。重复安装不会重复打补丁；遇到不兼容或部分修改时会拒绝覆盖。包含本示例的后续 checkout 也可以使用，但 Relax 核心必须仍与固定基线及这份补丁一致。

移除接入补丁可执行 `python examples/chunkflow/setup.py --reverse`。安装器只管理源码，不安装依赖或启动集群。

## Dense 与 MoE 示例

两个示例都使用 8 张 GPU 进行纯文本 SFT，默认 TP=2、PP=1、CP=1。Qwen3-4B 使用 EP=1，Qwen3.6-35B-A3B 使用 EP=8、ETP=1，并包含 GDN 状态适配。

先提供 `MODEL_DIR`、`DATA_DIR`、`SAVE_DIR`：模型分别位于 `${MODEL_DIR}/Qwen3-4B` 或 `${MODEL_DIR}/Qwen3.6-35B-A3B`；数据为 `${DATA_DIR}/train.jsonl`，每行包含 `messages` 对话列表。数据格式参见 [SFT 训练](./sft-training.md)。`SAVE_DIR` 必须是训练节点可访问的持久化目录。

```bash
# 预览参数，不连接 Ray，也不要求 GPU。
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh

# 配好现有集群的 RAY_ADDRESS 和 RAY_JOBS_ADDRESS 后，选择一个模型提交。
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

`RAY_JOBS_ADDRESS` 是现有集群的 Jobs HTTP 地址。脚本默认阻塞等待；`RAY_NO_WAIT=1` 可改为非阻塞。默认上传仓库工作目录，以便 driver 和 worker 能导入 example；使用自定义 `WORKING_DIR` 时也须包含打好补丁的 Relax 和完整 example。

脚本默认 `SEQ_LENGTH=8192`、`CHUNK_SIZE=2048`、`KEEP_LAST_K=1`、`NUM_STEPS=100`、`SAVE_INTERVAL=10`，均可用同名环境变量覆盖。保存与恢复路径固定为 `${SAVE_DIR}/${EXP_NAME}`，保留最近两个训练 checkpoint；恢复时保持两者不变。时间戳只用于 ClearML task 和 `examples/chunkflow/scripts/log/` 下的日志。`PROJECT_NAME` 默认 `Relax/sft/chunkflow`，`EXP_NAME` 按模型区分且可覆盖。

MoE 示例使用原生 all-to-all 分发，关闭路由辅助损失，并通过 `examples/chunkflow/configs/gdn.yaml` 关闭 GDN 预处理融合。它采用 CPU optimizer offload，需要足够的主机内存。两个示例均不启用 full-layer recompute、VPP、MTP 或多模态输入。

## Qwen3.8-Flash-Next（QSA + GDN + PLE）

实验性的 32 卡脚本使用 `${MODEL_DIR}/Qwen3.8-Flash-Next` 和相同格式的 `train.jsonl`，配置为 TP2/PP1/CP1、EP8/ETP1，并启用 sequence parallel。脚本保留 checkpoint 的 PLE 和 hyper-connection 结构，冻结 PLE 参数及硬选块的 indexer；PLE 大表使用原生 CPU offload，优化器也卸载到 CPU，主机内存需要同时容纳两者。32 卡是启动配置，不代表任意 32 卡机器都能装下模型。

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
# 设置目录和现有集群地址后提交。
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
```

`configs/qwen38.yaml` 关闭 GDN 预处理融合和 QSA Stage-2 indexer 辅助损失。脚本使用原生 `flex` attention，关闭 Qwen3.8 的融合/社区 selector 覆盖、local-query/replay 补丁及 streamed PLE。`_common.sh` 将这些模型环境变量传给 Ray worker。运行不切块对照时也应保留相同设置。

`adapters/qsa.py` 把每层完整的 hidden-state 前缀交给原生 `QwenSparseSelfAttention`，投影、池化、TopK、mask 和 attention 都由原生实现计算，适配器只取当前 chunk 的输出。选块前会恢复原 parent pack 的行数、文档数和候选宽度，再调用原生 TopK；如果只保留当前可见候选，并列分数可能选出不同的块。`adapters/ple.py` 将完整前缀的 token ID 和 hidden states 交给原生 PLE，继续使用它的 n-gram 哈希、门控和因果卷积。`adapters/prefix.py` 共用实际边界、padding、RoPE 和 SP 布局处理；PLE 与 QSA 使用不同的缓存命名空间，同一层的历史不会相互覆盖。

每次重放的参数梯度由原生 autograd 计算。历史 hidden states 的梯度累计到 `ChunkState`，在较早 chunk 执行 backward 时接回上游；`Fr` 复用这些缓存叶子。前缀重放会重复计算，也可能保留较多激活，而 selector 仍使用完整 parent 布局。这是兼容性实现，长上下文训练前需要实际测量开销。

该适配器要求固定版本的原生 QSA 类、bitset 后端（`flex`，或小规模对照用的 `dense_masked`）、PP1/CP1、packed 纯文本、零 dropout，以及由 ChunkFlow 管理重计算。不支持 `id_sparse`、Stage-2 辅助损失、推理缓存、外部 mask/bias 和上述方法覆盖优化。CPU 数值测试使用仓库补丁中的原生 selector/attention 与 PLE 方法，并用小型对象替代构造依赖；它们不能代替 CUDA kernel、多卡通信或完整 Flash-Next 训练验证。

## 接入其他训练脚本

应用补丁后，可以在其他 Megatron 训练脚本中加入以下参数，并保留原脚本的模型、数据和资源配置：

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

切块由调度器完成，Attention 内部不再二次切块。没有 `chunkflow_unit` 标记的调用直接执行原生 `forward`，保留其输入参数和行为。

## 配置

参数定义位于 `examples/chunkflow/arguments.py`。下表是解析器默认值；启动示例的 chunk 大小和激活保留数使用上文所述值。

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
- 滑动窗口、非 vanilla softmax、QK clipping/logit 统计、外部 Attention mask/bias 和 Qwen3.5 optimized recipe 需要单独适配。QSA 和 PLE 使用上文受限的原生前缀适配器，即使启用了实验流水线也必须保持 PP1。
- GDN 整层适配器精确匹配 Megatron `GatedDeltaNet`；子类需要显式注册。它需要支持初始/末状态梯度的算子及 Megatron `_forward_compute` 接口；须关闭预处理融合，不支持 deterministic 模式或 full-layer recompute。
- MoE 辅助损失、容量限制和丢弃 token 的配置需单独验证，不能假设切块前后等价。
- PP=1 暂不更新上游 PackingMetrics，性能对比应使用实际 step 耗时。

比较 loss、耗时和显存时，保持数据、随机种子、并行配置、精度及重计算配置一致。诊断可使用框架原有 profiler，参见[性能调优](./performance-tuning.md)。

## 接入新的 Attention

在每个训练 worker 安装 ChunkFlow 前，通过 `examples.chunkflow.adapters` 中的 `register_attention_adapter(attention_type, core_type, adapter)` 注册 `AttentionAdapter`，例如在自定义 model provider 中注册。普通 K/V 适配器精确匹配外层和核心类；`core_type=None` 表示整层适配，精确匹配外层类并优先使用，适用于 GDN 或带 indexer 的 Attention。整层适配器负责检查内部实现；重复注册会报错。

所有适配器共用一次安装流程：先检查全部层，再安装；安装器返回卸载回调。如果后续层安装失败，已安装的层会倒序恢复；适配器自身失败时也必须撤销自己的修改。GDN 的共享卷积包装在最后一个使用它的层卸载后恢复原函数。

如果算子的历史信息就是普通 K/V，可以用 `dense_attention_adapter(name=..., execute=..., validate=...)` 复用 ChunkFlow 的状态和梯度连接。执行函数接收保存的原生 `forward` 和 `DenseAttentionCall`，后者包含 THD query、K/V 片段、实际 Q/KV 边界、CPU 长度和最大长度。执行函数负责以右下对齐的因果关系调用原生算子，并保留原生 autograd，不能把 RoPE 偏移当作实际 padding。其他历史状态可以通过整层适配器管理；将稀疏算子注册成 Dense，并不等于实现了其 indexer 状态传递。

状态管理也共用一套接口：`runtime.history(layer_number, group_id, chunk_id, recompute=...)` 返回从第一块到当前块的 `ChunkState`，重计算复用已缓存的叶子。Dense 读取整个历史 K/V，GDN 读取上一块的卷积与递推状态。`ChunkState.link_tensor` 为参与当前输出计算的状态接回历史梯度；`link_output` 为原生算子额外返回的末状态连接梯度。只有明确处理原生 selective checkpoint 的适配器才应启用 `allow_replay=True`。

GDN 仍调用 Megatron 使用的卷积和递推 kernel；公共层只管理状态与梯度连接。原生模块如果已有可求导的分块接口，可以注册一个整层适配器来调用它，无需修改调度器。QSA/PLE 通过前缀重放补齐历史，以重复执行原生计算的代价减少适配层的算法实现。新增适配器用于训练前，应比较不切块和切块执行的输出、输入及参数梯度，并验证 packed 序列边界和 ChunkFlow 重计算。

## 运行测试

测试与实现一起放在 `examples/chunkflow/test/`，按设备分为 `cpu/` 和 `gpu/`，文件名不再带 `chunkflow` 前缀。

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest examples/chunkflow/test/cpu -q
python -m pytest examples/chunkflow/test/gpu -q
```

部分 CPU 用例仍需要 Megatron Python 依赖。GPU 用例需要真实 CUDA、Megatron/TE 或 FLA 环境；缺依赖而跳过不代表数值或分布式训练已经通过验证。
