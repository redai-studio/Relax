<div align="center">

## Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale

<img src="./assets/Relax.jpg" width="800" alt="Relax">

<p>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  </a>
  <a href="https://arxiv.org/abs/2604.11554">
    <img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red" alt="arXiv">
  </a>
  <a href="https://redai-studio.github.io/Relax">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation">
  </a>
  <a href="https://github.com/redai-studio/Relax/discussions/48" target="_blank">
    <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
  </a>
  <a href="https://github.com/redai-studio/Relax/discussions/30" target="_blank">
    <img src="https://img.shields.io/badge/Docker-Image-blue?logo=docker" alt="Docker Image">
  </a>
</p>

<p>
  <a href="./README.md">📖 English</a> | <a href="./README_zh.md">📖 中文</a>
</p>
</div>

**Relax** 是小红书 AI 平台开源的强化学习后训练框架，支持使用文本、图像、视频和音频训练大模型，也支持训练使用工具、与环境交互的智能体。

Relax 使用 Megatron-LM 训练模型、SGLang 生成样本，通过 Ray Serve 管理服务。[TransferQueue](https://github.com/redai-studio/TransferQueue) 在训练与推理之间传递数据，让两者可以独立运行。

## ✨ 核心能力

- **全模态训练** — 在同一框架中训练文本、视觉语言和音视频模型，支持 Qwen3-Omni 等全模态模型。
- **异步训练** — 生成样本与模型训练并行进行，可配置数据新鲜度。根据 GPU 资源和对 on-policy 的要求选择训练模式。
- **Agentic RL** — 支持工具调用、环境反馈、多轮交互和多智能体训练。Loss masking 只让模型输出参与训练损失，排除环境观察；多模态上下文可跨轮保留。参阅 [Agentic 指南](docs/zh/guide/agentic-rollout.md)。
- **Dense 与 MoE 模型** — 支持张量、流水线、上下文和专家并行（TP/PP/CP/EP），也支持 Dense 与 MoE 模型的 [LoRA 训练](docs/zh/guide/low-rank-adaptation-training.md)。
- **算法与奖励** — 支持多种 RL 算法和 On-Policy Distillation，内置数学、问答和指令遵循奖励。可以添加自定义奖励，也可以通过 [GenRM](docs/zh/examples/generative-reward-model.md) 让模型为回答打分。
- **Rollout 弹性扩缩容** — 在训练过程中添加推理引擎，也可接入其他集群的引擎。缩容时，等待待处理请求完成后再移除新增引擎。参阅[扩缩容指南](docs/zh/guide/elastic-rollout.md)。
- **监控与恢复** — 故障后自动重启服务，必要时重启整个训练任务。通过 TensorBoard、WandB 或 ClearML 查看指标，通过 Apprise 接收训练通知。

## 📦 安装

推荐使用官方 Docker 镜像，其中已预装 CUDA、PyTorch、Megatron-LM、SGLang 和 Ray。将 `/path/to/your/workspace` 替换为宿主机目录，用于保存代码、模型和数据。

```bash
# 拉取官方镜像
docker pull ghcr.io/redai-studio/relaxrl:latest

# 启动容器，挂载 GPU、共享内存与工作目录
docker run -it --gpus all --ipc=host --network=host \
  -v /path/to/your/workspace:/workspace \
  ghcr.io/redai-studio/relaxrl:latest bash

# 容器内克隆仓库并安装
git clone https://github.com/redai-studio/Relax.git /workspace/Relax
cd /workspace/Relax && pip install -e .
```

> 📖 关于 GPU 驱动要求、多节点部署与持久化存储挂载，请参阅 [安装指南](docs/zh/guide/installation.md)。

## 🚀 快速开始

选择一个**文本**、**视觉语言**或**音视频**任务，在训练容器的 `/workspace/Relax` 目录中运行下面的命令。设置 `EXP_DIR=/workspace` 后，脚本即可找到下载的模型与数据。本例将 ClearML 设为离线模式。

### 任务一 — DAPO Math（文本，8 卡）

使用 GRPO 在 [`dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) 上训练 Qwen3-4B，并用 AIME 2024 评估。奖励通过规则抽取与符号数学校验判断答案是否正确。

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /workspace/dapo-math-17k
hf download Qwen/Qwen3-4B --local-dir /workspace/Qwen3-4B
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /workspace/aime-2024
python scripts/tools/process_aime.py --input /workspace/aime-2024/aime-2024.jsonl

cd /workspace/Relax
export EXP_DIR=/workspace
export CLEARML_OFFLINE_MODE=1
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

<details>
<summary>Open-R1：视觉语言训练，8 卡</summary>

在 [`multimodal-open-r1-8k-verified`](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 上使用 GRPO 训练 Qwen3-VL-4B，奖励使用 `openr1mm`。

```bash
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir /workspace/multimodal-open-r1-8k-verified
python scripts/tools/process_openr1.py \
  --input-dir /workspace/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
  --output-dir /workspace/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /workspace/Qwen3-VL-4B-Instruct

cd /workspace/Relax
export EXP_DIR=/workspace
export CLEARML_OFFLINE_MODE=1
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

</details>

<details>
<summary>AVQA：图像与音频训练，16 卡 / 2 节点</summary>

在 [`AVQA-R1-6K`](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) 上使用 GRPO 训练 Qwen3-Omni-30B-A3B，奖励采用多选题匹配。

启动双节点训练前，按[多节点训练说明](docs/zh/guide/customize-training.md)配置 `MASTER_ADDR`、`POD_NAME`、`HOST_IP` 和 `WORLD_SIZE`。两个节点需要使用相同的模型与数据路径。下面的命令会准备数据，并加载 Qwen3-Omni 随附的 chat template。

```bash
hf download --repo-type dataset harryhsing/AVQA-R1-6K --local-dir /workspace/AVQA
python scripts/tools/process_avqa.py \
  --input-dir /workspace/AVQA/AVQA_R1/train/omni_rl_format_train.json \
  --output-dir /workspace/AVQA/AVQA_R1/train/omni_rl_format_train_convert.jsonl \
  --md-dir /workspace/AVQA/AVQA_R1/train
python scripts/tools/process_avqa.py \
  --input-dir /workspace/AVQA/AVQA_R1/valid/omni_rl_format_valid.json \
  --output-dir /workspace/AVQA/AVQA_R1/valid/small_valid.jsonl \
  --md-dir /workspace/AVQA/AVQA_R1/valid
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /workspace/Qwen3-Omni-30B-A3B-Instruct

python - <<'PY'
import json
from pathlib import Path

model = Path('/workspace/Qwen3-Omni-30B-A3B-Instruct')
config_path = model / 'tokenizer_config.json'
config = json.loads(config_path.read_text())
if 'chat_template' not in config:
    config['chat_template'] = json.loads((model / 'chat_template.json').read_text())['chat_template']
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
PY

cd /workspace/Relax
export EXP_DIR=/workspace
export CLEARML_OFFLINE_MODE=1
bash -x scripts/entrypoint/spmd-multinode.sh \
  scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

</details>

启动后若看到如下日志，说明训练已经正常运行：

```text
Finish rollout 0/200
training step 0/200
```

> 📖 完整教程：[快速上手指南](docs/zh/guide/quick-start.md) · [自定义训练](docs/zh/guide/customize-training.md) · [配置指南](docs/zh/guide/configuration.md)

## 🏗️ 训练模式

根据训练与样本生成（Rollout）如何使用 GPU，选择执行模式：

| 模式                      | GPU 分配                                                            | 训练方式                                                               |
| :------------------------ | :------------------------------------------------------------------ | :--------------------------------------------------------------------- |
| **Colocate（同步）**      | 训练与 Rollout 共享同一组 GPU。                                     | 生成一批样本后再训练，使用严格 on-policy 数据。                        |
| **Fully Async（全异步）** | 训练与 Rollout 使用不同的 GPU，辅助服务独立运行。                   | 生成样本与训练并行进行，可配置 staleness，在数据新鲜度与吞吐之间取舍。 |
| **Hybrid（混合）**        | 训练与 Rollout 使用不同的 GPU，参考模型推理和辅助计算复用训练 GPU。 | 保留流式训练流程与可配置 staleness，同时复用训练 GPU 完成辅助工作。    |

<div align="center">
  <img src="./assets/arch.png" width="80%" alt="Relax 架构图">
</div>

> 📖 配置与取舍：[架构指南](docs/zh/guide/architecture.md) · [全异步训练](docs/zh/guide/fully-async-training.md) · [Hybrid 训练](docs/zh/guide/hybrid-training.md)

## 🧠 支持的算法

| 算法                       | 描述                                              |
| :------------------------- | :------------------------------------------------ |
| **PPO**                    | Proximal Policy Optimization                      |
| **GRPO**                   | Group Relative Policy Optimization                |
| **M2PO**                   | Second-Moment Trust Policy Optimization           |
| **RLOO**                   | REINFORCE Leave-One-Out                           |
| **REINFORCE++**            | token KL-to-go 与全局归一化                       |
| **REINFORCE++-baseline**   | 使用组基线的 REINFORCE++ 变体                     |
| **GSPO**                   | Group-wise Sequence-level Policy Optimization     |
| **SAPO**                   | Soft Adaptive Policy Optimization                 |
| **CISPO**                  | Clipped Importance-ratio Soft Policy Optimization |
| **On-Policy Distillation** | 基于 KL 惩罚的师生蒸馏                            |

> 📖 各算法的目标函数、训练脚本与执行模式限制请参阅[算法参考](docs/zh/examples/algorithms.md)。

## 🤖 支持的模型

下表中的模型使用 Megatron 后端。可用配置见[模型配置](scripts/models/)与[训练脚本](scripts/training/)。

| 模型系列                                                        | 示例规模                                   | 模态               |
| :-------------------------------------------------------------- | :----------------------------------------- | :----------------- |
| **Qwen3**                                                       | 4B, 30B-A3B (MoE)                          | 文本               |
| **Qwen3-VL**                                                    | 4B, 30B-A3B                                | 视觉 + 语言        |
| **Qwen3.5**                                                     | 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B | 文本 + 视觉        |
| **Qwen3-Omni**                                                  | 30B-A3B                                    | 文本 + 视觉 + 音频 |
| **Qwen3.6**                                                     | 27B, 35B-A3B                               | 文本 + 视觉        |
| **Qwen3.8**                                                     | 27B                                        | 文本 + 视觉        |
| **GLM5**                                                        | 744B-A40B (MoE)                            | 文本               |
| **Kimi K2.6**                                                   | ~1T-A32B (MoE)                             | 视觉 + 语言        |
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B                                         | 视觉 + 语言        |

Kimi K2.6 示例包含 INT4 QAT 训练。dots.mocr 示例面向 OCR 与文档理解任务。

> 📖 接入其他模型架构，请参阅[外部模型接入指南](docs/zh/guide/external-model-integration.md)。

## 📚 文档

访问[完整文档](https://redai-studio.github.io/Relax/zh/)，或按任务查找：

- [自定义训练](docs/zh/guide/customize-training.md)：更换模型、数据集、奖励函数或启动配置。
- [配置指南](docs/zh/guide/configuration.md)：查阅训练参数。
- [监控训练](docs/zh/guide/metrics-service-detailed.md)：记录指标、查看训练进度。
- [导出模型](docs/zh/guide/model-conversion.md)：将 checkpoint 转换为 Hugging Face 权重。

## 🧪 示例

| 示例                                                         | 描述                                    |
| :----------------------------------------------------------- | :-------------------------------------- |
| [算法配方](./examples/algorithms/)                           | RLOO、REINFORCE++、CISPO 等策略优化算法 |
| [DeepEyes](./examples/deepeyes/)                             | 基于 Qwen3-VL 的多模态视觉语言 RL       |
| [Search-R1](./examples/search_r1/)                           | 搜索增强的单智能体与 multi-agent RL     |
| [Mini-SWE-Agent](./examples/mini_swe_agent/)                 | 面向软件工程任务的 Agentic RL           |
| [NeMo Gym Agentic](./examples/nemo_gym_agentic/)             | Agentic 环境集成与可运行配方            |
| [On-Policy Distillation](./examples/on_policy_distillation/) | 基于 KL 惩罚的师生知识蒸馏              |

## 🧩 基于 Relax 构建的项目

| 项目                                                     | 描述                                                                                                |
| :------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| [HyperEyes](https://github.com/DeepExperience/HyperEyes) | 使用 Relax 训练的多模态搜索智能体，结合视觉定位与检索，并行搜索多个实体。                           |
| [Iris](https://github.com/AllSpark-Research/Iris)        | 使用 Relax、基于 Qwen3.5/3.6 训练的开放权重搜索智能体，通过多步搜索收集证据，并为长任务管理上下文。 |

## 🤝 参与贡献

欢迎各种形式的贡献！请阅读 [贡献指南](docs/zh/guide/how-to-contribute.md) 了解详情。

## 📢 最新动态

<details>
<summary>展开更新记录</summary>

| 📣 更新                                                                                                                                                                            |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **\[08/20/2026\]** 🧠 新增 **M2PO**、**RLOO** 与两种 **REINFORCE++** 算法；提供 RLOO 与 REINFORCE++ 可运行配方，并在[算法参考](docs/zh/examples/algorithms.md)中涵盖以上四种算法。 |
| **\[08/19/2026\]** 🤖 新增支持多上下文轨迹导出的 multi-agent 训练，并提供 [Search-R1 参考配方](examples/search_r1/)。                                                              |
| **\[08/18/2026\]** 🪶 LoRA RL 现已支持 MoE 模型的 adapter 与 merge 两种工作流，详见 [LoRA 训练指南](docs/zh/guide/low-rank-adaptation-training.md)。                               |
| **\[05/26/2026\]** 🔁 新增 **Hybrid** 模式：训练与采样并行，辅助计算复用训练 GPU。详见 [Hybrid 训练指南](docs/zh/guide/hybrid-training.md)。                                       |
| **\[05/11/2026\]** 🚀 支持 Qwen3.6 系列模型（纯文本+多模）！                                                                                                                       |
| **\[04/15/2026\]** 🎉 Relax 正式开源！                                                                                                                                             |

</details>

## 📝 引用

如果 Relax 对您的研究有帮助，请引用：

```bibtex
@software{relax2026,
  title  = {Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale},
  author = {Relax Contributors},
  url    = {https://arxiv.org/abs/2604.11554},
  year   = {2026}
}
```

## 📜 许可证

本项目基于 [Apache License 2.0](./LICENSE) 开源。

## 🙏 致谢

Relax 的构建离不开以下优秀的开源项目：

- [Slime](https://github.com/THUDM/slime) — 可扩展的强化学习训练与推理框架
- [SGLang](https://github.com/sgl-project/sglang) — 高性能大语言模型推理框架
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 与 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — 大规模分布式训练框架及 HF ↔ Megatron 权重转换桥接库，衷心感谢整个 **NVIDIA** 团队
- [TransferQueue](https://github.com/Ascend/TransferQueue) — 高性能分布式数据传输队列
- [Ray](https://github.com/ray-project/ray) — 分布式计算框架
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 最先进的模型中心

衷心感谢所有贡献者和开源社区的支持！
