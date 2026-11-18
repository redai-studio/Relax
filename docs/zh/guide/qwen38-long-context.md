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

默认不主动清理 CUDA 缓存（`--empty-unused-memory-level 0`），首步和后续步骤一致。默认不保存 checkpoint。至少准备 6400 条有效样本供 200 步使用，或配置足够的数据轮次；不要把不足 GBS32 的尾批当成满批验证。

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

本分支在 4 节点 × 8 张 Blackwell GPU（每卡可见显存约 269 GiB） 上完成四场景各 20 步复测，首步及后续均不主动清缓存。以下 TPS/card 和 MFU 取 step5–19 的合计训练时间；显存取完整 20 步，空闲显存为 NVML 采样值。

| 场景 | TPS/card | MFU | 峰值 reserved | 最低采样空闲显存 |
| --- | ---: | ---: | ---: | ---: |
| real512 | 2624 | 5.52% | 219.33 GiB | 24.72 GiB |
| filled512 | 3314 | 7.07% | 229.51 GiB | 14.14 GiB |
| real1m | 2496 | 5.28% | 176.11 GiB | 66.93 GiB |
| filled1m | 2701 | 5.87% | 214.85 GiB | 25.29 GiB |

四场景均为完整 GBS32，loss/MTP loss/grad norm 有限，无报告的跳步。逐步对照历史运行的 loss、MTP loss、grad norm、LR 及 token/packing 指标。

real512、real1m 全部逐值一致；filled512 前 20 步最大 loss 绝对差 3.59e-6、grad norm 相对差 0.41%，filled1m 分别为 3.30e-6、0.65%。

filled512 首轮延至 25 步时 grad norm 最大相对差 2.02%；另一次同配置 20 步复测与历史逐值一致。重复运行不保证逐值相同，差异原因尚未定位。

同代码 real512 的 10 步清缓存对照中，loss/MTP loss/grad norm/LR 逐值一致；step5–9 不主动清缓存为 2884 TPS/card，每步清缓存（level1）为 1444 TPS/card。

此前长跑覆盖 real512 200 步及恢复后 10 步、real1m 100 步、两个满长度场景各 40 步。real512 恢复后的 loss/MTP loss/grad norm/LR/数据位置与连续训练逐值一致；其他场景未分别验证保存恢复。此次 20 步复测未重新验证 GPU checkpoint 恢复，也未重新构建镜像。CPU 相关回归 65 项通过，无跳过；覆盖实际 parser、数据与监督 mask、场景隔离和续训参数。
