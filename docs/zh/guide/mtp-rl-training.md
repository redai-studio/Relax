# MTP 训练

本指南介绍如何在 Relax 中训练 Multi-Token Prediction（MTP）头，包括 SFT 阶段的联合训练、两阶段分离训练，以及 RL 后训练阶段的联合训练。

阅读前请先完成[安装](./installation.md)。SFT 的数据格式和通用参数见 [SFT 训练](./sft-training.md)；RL 部分还需要先跑通一次基线 GRPO 任务（参考[快速上手](./quick-start.md)）。

## MTP SFT：联合训练与两阶段训练

下面以 Qwen3.5-35B-A3B 和 `pokemon-gpt4o-captions` 为例。训练同时使用中英文 parquet，每个文件 833 条，共 1666 条。样本以 `conversations` 保存对话，以 `images` 保存图片。

| 方式 | 脚本 | 训练内容 |
| --- | --- | --- |
| 联合训练 | [`run-qwen3.5-35B-A3B-pokemon-mtp-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-8xgpu.sh) | 从原始 HF checkpoint 初始化，同时更新主模型和一层 MTP。 |
| Stage 1 | [`run-qwen3.5-35B-A3B-pokemon-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-8xgpu.sh) | 只做普通 SFT，并导出供 Stage 2 使用的 HF checkpoint。 |
| Stage 2 | [`run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh) | 加载 Stage 1，冻结非 MTP 参数，只更新 MTP。 |

### 联合训练

联合训练使用以下关键参数：

```text
--enable-mtp-training
--mtp-num-layers 1
--mtp-loss-scaling-factor 0.2
--mtp-detach-paths none
```

这种方式只需要一次训练，但主模型 target 和 MTP draft 会同时变化。当前示例脚本使用
`--mtp-detach-paths none`，让 MTP 辅助 loss 的梯度同时更新主模型和 MTP 参数。该参数可以独立控制
`embedding`、`backbone` 和 `lm-head` 三条梯度路径；默认切断全部三条路径，此时主模型只接收标准 SFT loss 的梯度。

### 两阶段分离训练

Stage 1 不传任何 MTP 训练参数，只优化普通 SFT loss。它通过 `--save-hf` 将训练后的主模型导出到 `${SAVE_DIR}/sft/qwen3.5-35B-A3B-sft-pokemon-gpu8-hf`，并保留原始 checkpoint 中未训练的 MTP 权重。

Stage 2 默认从该 HF 目录初始化，使用以下参数冻结主模型并只训练 MTP：

```text
--mtp-only-training
--mtp-num-layers 1
--mtp-loss-scaling-factor 0.2
```

两阶段共用同一个外部 `SAVE_DIR` 时可以直接依次启动：

```bash
SAVE_DIR=/path/to/checkpoints \
  bash scripts/entrypoint/ray-job.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-8xgpu.sh

SAVE_DIR=/path/to/checkpoints \
  bash scripts/entrypoint/ray-job.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh
```

如果两个阶段使用不同的 checkpoint 根目录，需要为 Stage 2 显式设置 `INIT_HF_DIR`。

### Pokémon 实测结果

评测从中文 parquet 固定抽取 100 条样本，使用 `seed=20260811`、`temperature=0`，并根据原生 `/generate` 返回的逐请求 speculative counters 计算：

- Micro accept rate = `sum(accepted) / sum(drafted)`
- Micro accept length = `sum(completion_tokens) / sum(spec_verify_ct)`

| Checkpoint | MTP 梯度模式 | Micro accept rate | Micro accept length |
| --- | --- | ---: | ---: |
| 原始 base（联合训练对照） | — | 65.7761% | 2.3184 |
| 联合训练（detach 全部主模型路径） | `--mtp-detach-paths embedding backbone lm-head` | 64.8330% | 2.2999 |
| 联合训练（完全联合梯度） | `--mtp-detach-paths none` | 70.8709% | 2.4250 |
| 两阶段 Stage 1 | 未训练 MTP | 43.5425% | 1.8726 |
| 两阶段 Stage 2 | `--mtp-only-training` | 68.5460% | 2.3725 |

在相同的 128-token 生成设置下，detach 联合训练相对 base 下降 0.9431 个百分点，接受长度下降 0.0185；no-detach 联合训练相对 base 提高 5.0948 个百分点，接受长度提高 0.1066，相对 detach 联合训练则提高 6.0379 个百分点和 0.1251 接受长度。Stage 1 是 target 已完成普通 SFT、MTP 仍为原始 base 权重的未对齐中间态；Stage 2 只训练 MTP 后，接受率提高 25.0035 个百分点，接受长度提高 0.4999。Stage 1 和 Stage 2 的实际输出长度完全相同：总计 11846 tokens，均值 118.46，中位数 117，范围 42–152，100 条都以 `stop` 结束。

早期实验中 base 的接受率看起来明显偏高，主要是输出长度造成的 micro 统计偏差：当生成上限为 512 时，base 有 96/100 条生成到长度上限，并包含较长的重复尾部。Micro accept rate 汇总所有 draft token，因此长序列权重更高，这些尾部会放大 base 的结果。表中的 base、detach 联合训练和 no-detach 联合训练均使用 128-token 生成上限。

::: warning 比较范围
base、detach 联合训练和 no-detach 联合训练可以直接比较。两阶段实验使用了另一组生成上限，不能与前三项直接比较；Stage 1 与 Stage 2 使用同一批样本和相同生成参数，可以互相比较。
:::

## MTP RL 联合训练

### 何时启用 MTP RL 训练

当你希望在 RL 阶段保持 MTP 头与策略同步演化,使得训练完成后的同一个 checkpoint 直接可用于推理加速(SGLang/vLLM 的 EAGLE/NEXTN speculative decoding),无需再单独做一次蒸馏时,启用此功能。

如果只关注主策略且下游没有 speculative decoding 计划,不建议启用——辅助 loss 会带来少量额外计算和显存开销。

### 前置条件

- HF checkpoint 必须已经包含 MTP 权重,即 `config.json` 中 `num_nextn_predict_layers >= 1`。目前覆盖 **Qwen3.5**、**Qwen3-next**、**MiMo-7B-RL**、**DeepSeek-V3 / V3.1**、**GLM-4.7-MoE**。
- 使用 Megatron 后端(也是 Relax 唯一的训练后端)。MTP 依赖 [`docker/patch/megatron/20260506-85bced0ae.patch`](../../../docker/patch/megatron/20260506-85bced0ae.patch) 中的 Megatron MTP 补丁;Relax 官方镜像已自动应用。
- 不能开启 combined 1F1B 流水线调度——与 MTP 不兼容,在 [`relax/backends/megatron/model.py:493`](../../../relax/backends/megatron/model.py) 处会被断言挡掉。

### 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--enable-mtp-training` | 关闭 | 启用 MTP forward 注入和 MTP 辅助 loss |
| `--mtp-num-layers` | 启用时必填 | 模型中 MTP 层数,必须与 HF checkpoint 一致 |
| `--mtp-loss-scaling-factor` | `0.1`(RL 推荐) | MTP loss 加到主 loss 前的缩放系数 |
| `--mtp-detach-paths` | `embedding backbone lm-head` | 要从 MTP loss 切断的梯度路径。可任意组合 `embedding`、`backbone` 和 `lm-head`，或者单独使用 `none` 启用完全联合梯度。 |

Relax 通过 Megatron 参数 provider 注册这些选项。启用 MTP 训练时，还必须设置 `--mtp-num-layers`。

### 启动脚本

| 脚本 | 模型 | 资源 | 备注 |
| --- | --- | --- | --- |
| [`run-qwen35-9B-mtp-8xgpu.sh`](../../../scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh) | `Qwen3.5-9B` | 8 GPU colocate | dense 模型,扩规模前的冒烟验证 |
| [`run-qwen35-35B-A3B-mtp-16xgpu.sh`](../../../scripts/training/text/run-qwen35-35B-A3B-mtp-16xgpu.sh) | `Qwen3.5-35B-A3B` | 16 GPU(2 节点)colocate | 生产目标,镜像基线 GRPO 脚本 + `MTP_ARGS` |

两个脚本都支持环境变量覆盖:

```bash
MTP_NUM_LAYERS=1 MTP_LOSS_SCALING_FACTOR=0.1 \
  bash scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh
```


### scaling factor 调优

默认 `0.1` 对 RL 而言比较保守,因为主 GRPO loss 自带来自 advantage 估计的梯度噪声。若观察到:

- `train/mtp_loss` 早早 plateau 而 `train/loss` 正常 → 提到 `0.2`–`0.3`
- 启用 MTP 后 `train/loss` 不稳定 → 降到 `0.05`
- MTP 梯度主导(对比基线观察 `train/grad_norm`)→ 降低缩放

作为参考,SFT MTP 脚本使用 `0.2`,因为 SFT 梯度更干净。slime 默认值也是 `0.2`;RL 调低是 Relax 的专门建议。

### 日志观察要点

健康的训练会在每个 train step 打出:

```text
train/loss              # 主 GRPO loss,与基线同量级
train/grad_norm         # 与基线同量级(≤2×)
train/mtp_loss          # 有界,逐步下降
```

若 `train/mtp_loss` 从未出现,说明 MTP block 没构建——通常是 checkpoint 不匹配(HF ckpt 没有 MTP 权重)。可通过下面命令验证:

```bash
python -c "from transformers import AutoConfig; \
  print(AutoConfig.from_pretrained('/path/to/ckpt').num_nextn_predict_layers)"
```


## 相关文档

- [SFT 训练](./sft-training.md) — SFT 的数据格式、checkpoint 和通用参数。
- [权重更新流水线优化](./update-weights-pipeline.md) — Megatron→SGLang 权重同步如何处理 MTP 层。
