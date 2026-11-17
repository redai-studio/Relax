# Qwen3.8 512K / 1M 训练

提供两套 32 卡 recipe，覆盖原始长度、样本翻倍和满长度容量测试。依赖已安装 HybridEP、FlashQLA 小 head 修复及 PLE 局部计算补丁的 CUDA 13 Blackwell 镜像；旧 checkout 不能仅靠脚本启用这些功能。

## 配置与数据

共同配置：4 节点 × 8 GPU，TP1/PP2/EP8/ETP1，48 层按 24+24 划分，MTP1，GBS32/MBS1，seed9714。冻结 vision、projection、PLE 和 indexer；full uniform recompute2，优化器状态全量 offload。

| 场景 | 数据处理 | CP / DP | 序列上限 | 每卡普通 packing 预算 | PLE 通信 |
| --- | --- | --- | --- | --- | --- |
| `real512` | 保留长度 ≤512K 的原始训练样本 | 8 / 2 | 512K | 64K | `full` |
| `filled512` | 同源样本及 loss mask 重复至 512K | 8 / 2 | 512K | 64K | `full` |
| `real1m` | 同源样本及 loss mask 各重复两次 | 16 / 1 | 1M | 32K | `adaptive` |
| `filled1m` | 同源样本及 loss mask 重复至 1M | 16 / 1 | 1M | 32K | `adaptive` |

K=1024，M=1024²。数据类在原生 SFT 渲染后处理 token 和未移位的 loss mask；过滤原始长度超过 512K 的样本，不生成额外 labels。仅支持文本 causal-LM SFT，重复样本的首 token 必须不计入 loss。1M 样本是人工扩展，满长度输入用于容量和性能测试，不代表天然长文质量。

1M recipe 的 32K 是普通 packing 预算，**不是每卡输入硬上限**。过长样本独占一个微批，1M 单样本在 CP16 下每卡仍有 64K token。`full` 仍采用 PLE 局部计算，参见 [PLE 使用说明](./qwen38-qsa.md#冻结-ple-的局部计算)。

## 运行

先准备独占的 32 卡 Ray 集群和所有节点可访问的模型、数据、输出路径。入口不清理已有任务，也不保证能与其他训练共享 GPU。使用新构建镜像中的 Megatron 和 HybridEP；`RELAX_HYBRID_EP_COMMIT` 由镜像提供。

```bash
export MODEL_DIR=/shared/models/Qwen3.8-Flash-Next
export DATA_DIR=/shared/data/train.jsonl
export SAVE_DIR=/shared/output/qwen38
export MEGATRON=/root/Megatron-LM
# Ray Jobs 地址可选；不设置时由 Ray 解析现有集群。
export RAY_JOBS_ADDRESS=http://127.0.0.1:8265

# 先检查实际 argv 和 worker 环境；不会提交或创建输出目录。
DRY_RUN=1 bash scripts/training/text/run-qwen38-512k-cp.sh

# 默认上限 200 步；四场景分别运行，不要同时提交到同一组 GPU。
bash scripts/training/text/run-qwen38-512k-cp.sh
DATA_MODE=filled bash scripts/training/text/run-qwen38-512k-cp.sh
bash scripts/training/text/run-qwen38-1m-cp.sh
DATA_MODE=filled bash scripts/training/text/run-qwen38-1m-cp.sh
```

也可通过共享入口 `SCENARIO=real512 bash scripts/training/text/run-qwen38-long-context.sh` 选择场景（`real512`、`filled512`、`real1m`、`filled1m`）。

默认异步提交。各场景的默认 `EXP_NAME` 独立，`SUBMISSION_ID` 每次唯一；提交日志及实际参数保存在 `$SAVE_DIR/$EXP_NAME/logs/`。常用变量：`NUM_STEPS`、`NUM_EPOCH`、`EXP_NAME`、`SAVE_INTERVAL`、`MAX_TOKENS_PER_GPU`、`QWEN38_PLE_INPUT_HALO_TRANSPORT`。`USE_CLEARML=1` 可启用 ClearML，需自行配置凭据；checkpoint 路径统一由 `SAVE_DIR`、`EXP_NAME` 和 `SAVE_INTERVAL` 设置，不接受附加 `--load` / `--save`。其他训练参数可追加在命令末尾，改变配置后需重新验证。

默认不保存 checkpoint。至少准备 6400 条有效样本供 200 步使用，或配置足够的数据轮次；不要把不足 GBS32 的尾批当成满批验证。

## 保存与续训

```bash
# 完成第 100/200 次更新时保存 iter_0000099 / iter_0000199。
EXP_NAME=real512-save SAVE_INTERVAL=100 \
  bash scripts/training/text/run-qwen38-512k-cp.sh

# 从第 100 次更新后的保存点恢复，训练总上限仍为 200。
EXP_NAME=real512-save RESUME=1 SAVE_INTERVAL=100 \
  bash scripts/training/text/run-qwen38-512k-cp.sh --ckpt-step 99
```

新训练发现已有 checkpoint 时会拒绝自动恢复。续训要求显式 `RESUME=1`、原 `EXP_NAME`、checkpoint tracker 和入口生成的 `recipe-config.json`。入口检查场景、数据/模型路径、并行配置、packing 预算、GBS、seed、学习率及附加参数，防止误加载其他场景；不要修改该记录绕过检查。

续训自动添加 `--use-checkpoint-opt_param-scheduler`，恢复保存的 LR/WD 计划；`NUM_STEPS` 只控制训练停止位置，不重新定义恢复后的计划。模型、优化器和数据位置由框架原生 checkpoint 逻辑恢复。未设置 `SAVE_INTERVAL` 时不另存数据位置；本 recipe 不支持独立的数据位置目录。

## 验证范围

以下是整理前相同模型/数据设计的 32 卡历史验证，指标取训练阶段，非全任务墙钟吞吐。两个满长度任务在完整 40 步后计划停止，状态为 STOPPED，未完成 200 步。

| 场景 | 已验证步数 | 观测 TPS/card | 峰值 reserved | 最低采样空闲显存 |
| --- | --- | --- | --- | --- |
| real512 | 200 + 恢复后 10 步 | 2572.16（无复制窗口） | 221.05 GiB | 21.81 GiB |
| filled512 | 40 步完整前缀 | 3316.92 | 231.19 GiB | 12.35 GiB |
| real1m | 100 | 2501.36 | 195.26 GiB | 46.21 GiB |
| filled1m | 40 步完整前缀 | 2696.45 | 216.39 GiB | 25.85 GiB |

各完整步的 loss/MTP loss/grad norm 有限；real512 保存恢复后的 10 步与连续训练的 loss、MTP loss、grad norm、LR 及数据位置逐值一致。其他场景未分别验证保存恢复。

**本分支未重新运行上述 GPU 训练或构建镜像。** 历史运行包含“仅首步释放缓存”和独立数据位置记录，本 PR 未提交这两项；发布配置使用主线默认缓存释放逻辑，不保证复现历史吞吐或 reserved 峰值。CPU 检查覆盖四场景真实 parser、数据与监督 mask、场景隔离和续训参数；GPU 容量仍需在目标环境验收。
