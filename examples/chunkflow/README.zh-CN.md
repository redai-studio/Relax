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

提供训练节点可访问的三个目录：

| 环境变量    | 目录内容                                                   |
| ----------- | ---------------------------------------------------------- |
| `MODEL_DIR` | Hugging Face checkpoint：`Qwen3-4B/` 或 `Qwen3.6-35B-A3B/` |
| `DATA_DIR`  | `train.jsonl`，每行包含 `messages` 列表                    |
| `SAVE_DIR`  | 持久化训练 checkpoint                                      |

数据行示例：

```json
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "2"}]}
```

使用较长对话才能覆盖多个 chunk。两个脚本均为八卡纯文本 SFT，采用 TP2/PP1/CP1、零 dropout 和 ChunkFlow 激活重计算。MoE 脚本采用 EP8/ETP1、原生 all-to-all 分发，关闭路由辅助损失与 GDN 预处理融合，并使用 CPU optimizer offload。

设置上述目录变量后，可以不连接 Ray、不使用 GPU 地预览参数：

```bash
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

将 `RAY_ADDRESS` 和 `RAY_JOBS_ADDRESS` 分别设为现有集群的 GCS 地址和 Jobs HTTP 地址，再选择一个脚本：

```bash
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

现有 Ray 入口会清理集群训练状态，请用于预期的独占训练集群。默认阻塞等待，设置 `RAY_NO_WAIT=1` 可非阻塞提交。脚本默认上传仓库工作目录，包含本示例与已经应用的补丁。

可用环境变量覆盖 `CHUNK_SIZE`（2048）、`KEEP_LAST_K`（1）、`SEQ_LENGTH`（8192）、`NUM_STEPS`（100）或 `SAVE_INTERVAL`（10）。`KEEP_LAST_K=0` 保留全部块的激活计算图。每个脚本直接列出模型和训练参数，`_common.sh` 只处理提交。

保存和恢复使用 `${SAVE_DIR}/${EXP_NAME}`，保留最近两个 checkpoint。恢复时保持两者不变；新实验显式更换 `EXP_NAME`。`PROJECT_NAME` 默认为 `Relax/sft/chunkflow`。时间戳只进入 ClearML task 名和 `scripts/log/` 日志，不进入 checkpoint 路径。

## 代码与测试

| 文件                                      | 职责                         |
| ----------------------------------------- | ---------------------------- |
| `scheduler.py`、`units.py`                | 切块与执行计划               |
| `unit_batch.py`、`rope.py`、`metadata.py` | 输入、原始位置和 packed 边界 |
| `integration.py`、`adapters.py`、`gdn.py` | 模型安装与原生计算适配       |
| `runtime.py`、`state_store.py`            | 跨块状态、梯度与清理         |
| `schedules.py`、`pipeline_schedules.py`   | 前后向执行                   |
| `arguments.py`、`setup.py`、`relax.patch` | 配置与固定版本接入           |

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest examples/chunkflow/test/cpu -q
python -m pytest examples/chunkflow/test/gpu -q
```

部分 CPU 用例需要 Megatron Python 依赖，GPU 用例需要 CUDA 和原生计算后端。跳过用例不代表数值验证通过。PP>1 是需要显式启用的实验路径。示例脚本不包含 QSA、PLE、CP>1、VPP、多模态输入或 full-layer recompute；适配器具体限制见完整指南。
