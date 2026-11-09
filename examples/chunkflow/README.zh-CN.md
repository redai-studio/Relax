# ChunkFlow 示例

[English](README.md) · [完整指南](../../docs/zh/guide/chunkflow.md)

ChunkFlow 将长序列切块后送入 Megatron 模型，安排前后向，并连接各块之间的历史状态和梯度。Attention 与 GDN 计算调用模型原生实现。本目录包含全部实现、参数定义、接入补丁、启动脚本和测试。

## Relax 版本

固定源码基线为 **`d548769b77f12e445e464f99d4332290a3e5bd3d`**，记录在 `relax.version` 中。使用该版本对应的训练环境，包括 Megatron/Bridge、TE 和 FLA；Relax 要求 Python 3.10 及以上。

在单独检出固定版本的 Relax 中，将本目录完整复制到其 `examples/chunkflow/`。从仓库根目录执行：

```bash
python examples/chunkflow/setup.py
python examples/chunkflow/setup.py --check
```

安装器核对固定版本源码后，将 `relax.patch` 应用到三个核心文件。包含本示例的后续 checkout 也可以使用，但 Relax 核心须仍与基线一致。安装可重复执行，不兼容修改会被拒绝。执行 `python examples/chunkflow/setup.py --reverse` 可移除接入补丁；安装器不会自动安装依赖。

## Dense 与 MoE 训练

提供训练节点可访问的模型目录和数据文件路径：

| 环境变量      | 目录内容                                                   |
| ------------- | ---------------------------------------------------------- |
| `MODEL_DIR`   | Hugging Face checkpoint：`Qwen3-4B/` 或 `Qwen3.6-35B-A3B/` |
| `PROMPT_DATA` | JSONL 数据文件，每行包含 `messages` 列表                   |

数据行示例：

```json
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "2"}]}
```

使用较长对话才能覆盖多个 chunk。两个脚本均为八卡纯文本 SFT，采用零 dropout 和 ChunkFlow 激活重计算。Dense 脚本使用 TP2/PP1/CP1；MoE 脚本使用 TP2/PP4/CP1、EP2/ETP1、原生 all-to-all 分发，关闭路由辅助损失与 GDN 预处理融合，并使用 CPU optimizer offload。MoE 脚本设置 `--chunkflow-pipeline 4`，并向 worker 传递 `RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1`，启用实验性的流水线执行器。

现有验证环境中，Qwen3.6 的 PP4/EP2 配方与原生整层重计算的短程 loss 接近，但固定权重下的梯度方向存在显著差异。该配方仍属实验性质；训练跑通和 loss 接近不代表 backward 已对齐。

ChunkFlow 要求启用 `--calculate-per-token-loss`，使归一化不受切块边界影响。它会拒绝同时启用 `--fully-async` 和 `--use-dynamic-batch-size`，因为固定执行计划无法消费数量未知的流式 micro-batch。示例脚本已经使用 per-token loss 和同步训练。

设置上述目录变量后，可以不连接 Ray、不使用 GPU 地预览参数：

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/LongAlign-10k/long.jsonl
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

在独占的八卡节点上，选择一个脚本直接执行：

```bash
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

脚本自动核对固定版本的 Relax 源码并安装接入补丁，再通过 `auto` 发现现有 Ray 集群，使用关闭集群清理的标准 `ray-job.sh` 入口提交。请确保该集群空闲且由本次实验独占。没有现有集群时，标准 `local.sh` 会启动 Ray；该新节点入口会清理本机进程，应在独占训练节点使用。无需手动启动 Ray 或填写 Ray 地址。

仍可用 `RAY_ADDRESS` 或 `RAY_JOBS_ADDRESS` 显式指定集群。已配置或已发现的集群不可达时会报错退出，不会回退重启本机。请等当前实验结束后再启动下一次实验。外部入口已经设置 `RELAX_ENTRYPOINT_MODE` 时直接复用。默认阻塞等待，设置 `RAY_NO_WAIT=1` 可非阻塞提交。脚本默认上传仓库工作目录，包含本示例与已经应用的补丁。

可用环境变量覆盖 `CHUNK_SIZE`（4096）、`KEEP_LAST_K`（1）、`SEQ_LENGTH`（Dense/MoE 为 65536，QSA 为 8192）或 `NUM_STEPS`（100）。`KEEP_LAST_K=0` 保留全部块的激活计算图。每个脚本直接列出模型和训练参数，`_common.sh` 负责选择标准入口和提交任务。

Dense/MoE 同时启用原生 SFT 输出层的 `--sft-chunked-logits --sft-logits-chunk-size 1024 --recompute-loss-function`。这项优化独立于 ChunkFlow，用于降低输出层显存；baseline 对照也应保留相同参数。ChunkFlow 不保证比原生整层重计算更省显存。

三个脚本都可直接使用 `PROMPT_DATA` 指定数据文件，此时无需 `DATA_DIR`。不设置 `PROMPT_DATA` 时才使用 `${DATA_DIR}/train.jsonl`。LongAlign 可直接指向 `LongAlign-10k/long.jsonl`，并按分词后的对话长度选择 `SEQ_LENGTH`。超过上限的样本会被跳过；数据集记录的长度可能与模型 tokenizer 的统计不同。

三个脚本都从 Hugging Face 初始权重加载，不保存或恢复训练 checkpoint。无需设置 `SAVE_DIR` 或 `SAVE_INTERVAL`，此前导出的这两个变量也不会生效。`PROJECT_NAME` 默认为 `Relax/sft/chunkflow`，`EXP_NAME` 标识实验；每次运行的 ClearML task 名和 `scripts/log/` 日志带有时间戳。

## Qwen3.8-Flash-Next（QSA + GDN + PLE）

实验性的 32 卡脚本使用 `${MODEL_DIR}/Qwen3.8-Flash-Next` 和相同格式的 `train.jsonl`，配置为 TP2/PP1/CP1、EP8/ETP1，并启用 sequence parallel。脚本保留 checkpoint 的 PLE 和 hyper-connection 结构，冻结 PLE 参数及硬选块的 indexer；PLE 大表使用原生 CPU offload，优化器也卸载到 CPU，主机内存需要同时容纳两者。32 卡是启动配置，不代表任意 32 卡机器都能装下模型。

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
# 设置目录和下述节点变量后，在每个节点执行。
bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
```

新集群默认 `WORLD_SIZE=4` 个节点、每节点 `NUM_GPUS=8` 张卡，两者乘积必须为 32。多个节点共享尚未安装补丁的仓库时，先在共享目录运行一次 `python examples/chunkflow/setup.py`，再启动各节点。所有节点的 `MASTER_ADDR` 都填写主节点可解析的 **`POD_NAME`**，各自的 `POD_NAME` 填本节点名称，`HOST_IP` 填本节点可通信的 IP。标准 `spmd-multinode.sh` 入口通过 `MASTER_ADDR == POD_NAME` 的严格相等判断主节点，自动启动各节点的 Ray，等全部节点加入后仅由主节点提交一次训练。worker 持续运行；只启动主节点会一直等待其他节点。请使用独占节点：该入口会清理本机进程，并要求有权限通过 `sysctl` 设置预留网络端口。

已有 32 卡集群时，只在提交节点执行同一脚本，Ray 会自动发现；此模式不要求节点变量，也不重新组建集群。预览模式不会初始化任何集群。

`configs/qwen38.yaml` 关闭 GDN 预处理融合和 QSA Stage-2 indexer 辅助损失。脚本使用原生 `flex` attention，关闭 Qwen3.8 的融合/社区 selector 覆盖、local-query/replay 补丁及 streamed PLE。`_common.sh` 将这些模型环境变量传给 Ray worker。运行不切块对照时也应保留相同设置。

`adapters/qsa.py` 把每层完整的 hidden-state 前缀交给原生 `QwenSparseSelfAttention`，投影、池化、TopK、mask 和 attention 都由原生实现计算，适配器只取当前 chunk 的输出。选块前会恢复原 parent pack 的行数、文档数和候选宽度，再调用原生 TopK；如果只保留当前可见候选，并列分数可能选出不同的块。`adapters/ple.py` 将完整前缀的 token ID 和 hidden states 交给原生 PLE，继续使用它的 n-gram 哈希、门控和因果卷积。`adapters/prefix.py` 共用实际边界、padding、RoPE 和 SP 布局处理；PLE 与 QSA 使用不同的缓存命名空间，同一层的历史不会相互覆盖。

每次重放的参数梯度由原生 autograd 计算。历史 hidden states 的梯度累计到 `ChunkState`，在较早 chunk 执行 backward 时接回上游；`Fr` 复用这些缓存叶子。前缀重放会重复计算，也可能保留较多激活，而 selector 仍使用完整 parent 布局。这是兼容性实现，长上下文训练前需要实际测量开销。

该适配器要求固定版本的原生 QSA 类、bitset 后端（`flex`，或小规模对照用的 `dense_masked`）、PP1/CP1、packed 纯文本、零 dropout，以及由 ChunkFlow 管理重计算。不支持 `id_sparse`、Stage-2 辅助损失、推理缓存、外部 mask/bias 和上述方法覆盖优化。CPU 数值测试使用仓库补丁中的原生 selector/attention 与 PLE 方法，并用小型对象替代构造依赖；它们不能代替 CUDA kernel、多卡通信或完整 Flash-Next 训练验证。

## 代码与测试

| 文件                                                        | 职责                         |
| ----------------------------------------------------------- | ---------------------------- |
| `scheduler.py`、`units.py`                                  | 切块与执行计划               |
| `unit_batch.py`、`rope.py`、`metadata.py`                   | 输入、原始位置和 packed 边界 |
| `integration.py`、`adapters/__init__.py`、`adapters/gdn.py` | 模型安装与原生计算适配       |
| `adapters/qsa.py`、`adapters/ple.py`、`adapters/prefix.py`  | 原生 QSA/PLE 前缀重放        |
| `runtime.py`、`state_store.py`                              | 跨块状态、梯度与清理         |
| `schedules.py`、`pipeline_schedules.py`                     | 前后向执行                   |
| `arguments.py`、`setup.py`、`relax.patch`                   | 配置与固定版本接入           |

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest examples/chunkflow/test/cpu -q
python -m pytest examples/chunkflow/test/gpu -q
```

部分 CPU 用例需要 Megatron Python 依赖，GPU 用例需要 CUDA 和原生计算后端。跳过用例不代表数值验证通过。PP>1 是需要显式启用的实验路径。示例脚本不包含 CP>1、VPP、多模态输入或 full-layer recompute；适配器具体限制见完整指南。
