<div align="center">

## Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale

**Towards Async, Omni-Modal RL at Scale, Just Relax.**

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

______________________________________________________________________

**Relax**（**R**einforcement **E**ngine **L**everaging **A**gentic **X**-modality）是小红书 AI 平台开源的、面向多模态大模型的高性能强化学习后训练框架。Relax 基于 Ray Serve 构建面向服务的架构，以 Megatron-LM 为训练后端、SGLang 为推理引擎，通过 [TransferQueue](https://github.com/redai-studio/TransferQueue) 数据传输系统实现训练与推理的完全解耦，支持从文本到图像、视频、音频的全模态强化学习训练。

______________________________________________________________________

## ✨ 亮点

- 🌐 **全模态统一训练** — 单一框架覆盖文本、视觉、音频强化学习，业界少数能够在统一架构下完成 Omni 模型（Qwen3-Omni）后训练的系统
- ⚙️ **面向服务的六层架构** — 所有角色均作为独立 Ray Serve 服务部署，原生支持服务级别的弹性调度与故障恢复
- ⚡ **基于 TransferQueue 的全异步训练** — Rollout、Actor、ActorFwd、Reference、Advantages 运行在独立 GPU 集群，流式数据交换，可配置 staleness
- 🔁 **Hybrid 混合模式** — Actor 与 Rollout 独立 Placement Group + TransferQueue 流式数据，ref / actor_fwd / advantages 在 Actor 本机进程内完成；配合 `--balance-data` 与子批 forward，避免独立 ref/actor_fwd 服务的 GPU 浪费
- 🤖 **Agentic RL** — 多轮交互、loss masking、灵活的终止条件以及 VLM 多模态上下文累积，构建"执行 → 观察 → 决策"闭环训练
- 🔀 **Rollout 弹性扩缩容** — 通过 HTTP REST API 在训练过程中动态增减推理引擎，支持同集群（`ray_native`）和跨集群联邦（`external`）两种模式
- 🧠 **丰富的算法矩阵** — 开箱即用的 PPO、GRPO、M2PO、RLOO、REINFORCE++ 系列、GSPO、SAPO、CISPO 与 On-Policy Distillation，配合可插拔奖励函数和内置 **GenRM**（LLM-as-judge）模式
- 🚀 **Megatron + SGLang 后端** — Megatron-LM（TP/PP/CP/EP）训练 MoE 和深层模型，SGLang 提供高吞吐推理，DCS 基于 NCCL 广播同步权重
- 📦 **生产级运维** — HealthManager 自动恢复、中心化 Metrics Service（WandB / TensorBoard / ClearML）、Apprise 实时告警

______________________________________________________________________

## 📢 最新动态

| 📣 更新                                                                                                                                                                            |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **\[08/20/2026\]** 🧠 新增 **M2PO**、**RLOO** 与两种 **REINFORCE++** 算法；提供 RLOO 与 REINFORCE++ 可运行配方，并在[算法参考](docs/zh/examples/algorithms.md)中涵盖以上四种算法。 |
| **\[08/19/2026\]** 🤖 新增支持多上下文轨迹导出的 multi-agent 训练，并提供 [Search-R1 参考配方](examples/search_r1/)。                                                              |
| **\[08/18/2026\]** 🪶 LoRA RL 现已支持 MoE 模型的 adapter 与 merge 两种工作流，详见 [LoRA 训练指南](docs/zh/guide/low-rank-adaptation-training.md)。                               |
| **\[05/26/2026\]** 🔁 新增 **Hybrid** 执行模式 —— 流式数据 + 进程内 ref/actor_fwd，支持 `--balance-data`，详见 [Hybrid 训练指南](docs/zh/guide/hybrid-training.md)。               |
| **\[05/11/2026\]** 🚀 支持 Qwen3.6 系列模型（纯文本+多模）！                                                                                                                       |
| **\[04/15/2026\]** 🎉 Relax 正式开源！                                                                                                                                             |

______________________________________________________________________

## 🏗️ 系统架构

<div align="center">
  <img src="./assets/arch.png" width="80%" alt="Relax 架构图">
</div>

Relax 采用**面向服务的六层架构**，每个角色均作为独立的 [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) 部署，将编排、组件、引擎、后端与分布式能力彻底解耦：

| 层级                        | 职责                                                                                                           |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Entrypoints（入口层）**   | `train.py` — 信号处理、CLI 解析、Ray 集群连接、Controller 启动                                                 |
| **Orchestration（编排层）** | `Controller`（训练循环、全局重启）、`Service`（Placement Group、生命周期管理）、`Registry`（角色与算法注册）   |
| **Components（组件层）**    | Ray Serve 部署：**Actor**、**Rollout**、**Critic**、**ActorFwd**、**Advantages**、**GenRM**                    |
| **Engine（引擎层）**        | SGLang Rollout 引擎、可插拔奖励函数、请求路由、数据过滤                                                        |
| **Backends（后端层）**      | **Megatron-LM** 训练后端（TP/PP/CP/EP）与 **SGLang** 推理引擎                                                  |
| **Distributed（分布式层）** | Ray Actor Groups（RolloutManager / GenRMManager）与 **DCS**（分布式 Checkpoint 服务，支持 NCCL/GLOO 权重同步） |

支持**三种执行模式**：

- **Colocate（同步模式）** — Actor 与 Rollout 共享同一组 GPU，Rollout 将整批数据写入 TransferQueue 后释放 GPU 供训练使用；显存友好，严格 on-policy（`max_staleness=0`）。
- **Fully Async（全异步模式）** — Actor、Rollout、ActorFwd、Reference、Advantages 运行在**独立 GPU 集群**上完全并行，通过 TransferQueue 交换数据，通过 DCS 异步同步权重，在可配置 staleness 下实现最大吞吐。
- **Hybrid（混合模式）** — Actor 与 Rollout 使用**独立的 Placement Group**（与全异步一致），通过 TransferQueue 流式交换数据并支持可配置 staleness；但 ref / actor_fwd / advantages 通过 `TensorBackuper` + `_switch_model` 在 Actor 自身 GPU 上**进程内复用权重**（与 Colocate 一致）。在不为独立 ref/actor_fwd 服务付出额外 GPU 的前提下，同时获得流式数据管线与 `--balance-data` 支持。

> 📖 了解更多：[架构指南](docs/zh/guide/architecture.md) · [全异步训练](docs/zh/guide/fully-async-training.md) · [Hybrid 训练](docs/zh/guide/hybrid-training.md) · [Rollout 弹性扩缩容](docs/zh/guide/elastic-rollout.md)

______________________________________________________________________

## 🧠 支持的算法

| 算法                       | 类型         | 描述                                              |
| :------------------------- | :----------- | :------------------------------------------------ |
| **PPO**                    | Actor-Critic | Proximal Policy Optimization                      |
| **GRPO**                   | 策略优化     | Group Relative Policy Optimization                |
| **M2PO**                   | 策略优化     | Second-Moment Trust Policy Optimization           |
| **RLOO**                   | 策略优化     | REINFORCE Leave-One-Out                           |
| **REINFORCE++**            | 策略优化     | token KL-to-go 与全局归一化                       |
| **REINFORCE++-baseline**   | 策略优化     | 使用组基线的 REINFORCE++ 变体                     |
| **GSPO**                   | 策略优化     | Group-wise Sequence-level Policy Optimization     |
| **SAPO**                   | 策略优化     | Soft Adaptive Policy Optimization                 |
| **CISPO**                  | 策略优化     | Clipped Importance-ratio Soft Policy Optimization |
| **On-Policy Distillation** | 知识迁移     | 基于 KL 惩罚的师生蒸馏                            |

> 📖 各算法的目标函数、训练脚本与执行模式限制请参阅[算法参考](docs/zh/examples/algorithms.md)。

______________________________________________________________________

## 🤖 支持的模型

Relax 专为**全模态强化学习训练**设计 —— 文本、视觉、音频统一框架。通过 `--multimodal-keys` 参数灵活配置多模态数据，框架内置了完整的图像、视频、音频处理管线（`relax/utils/multimodal/`），支持图像 token 数量控制、视频帧率采样、音频采样率等精细调节。

| 模型系列                                                        | 规模                                       | 模态               | 典型任务                                 | 后端     |
| :-------------------------------------------------------------- | :----------------------------------------- | :----------------- | :--------------------------------------- | :------- |
| **Qwen3**                                                       | 4B, 30B-A3B (MoE)                          | 文本               | 数学推理、代码生成、多轮对话、工具调用   | Megatron |
| **Qwen3-VL**                                                    | 4B, 30B-A3B                                | 视觉 + 语言        | 视觉问答、图像理解、多模态推理           | Megatron |
| **Qwen3.5**                                                     | 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B | 文本 + 视觉        | 推理、SFT、视觉问答、多模态推理          | Megatron |
| **Qwen3-Omni**                                                  | 30B-A3B                                    | 文本 + 视觉 + 音频 | 图文音频联合问答、全模态理解             | Megatron |
| **Qwen3.6**                                                     | 27B, 35B-A3B                               | 文本 + 视觉        | 推理、视觉问答、多模态推理               | Megatron |
| **Qwen3.8**                                                     | 27B                                        | 文本 + 视觉        | 推理、SFT、视觉问答、多模态推理          | Megatron |
| **GLM5**                                                        | 744B-A40B (MoE)                            | 文本               | 数学推理、代码生成、多轮对话             | Megatron |
| **Kimi K2.6**                                                   | ~1T-A32B (MoE)                             | 视觉 + 语言        | 视觉问答、多模态推理；支持 INT4 QAT 训练 | Megatron |
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B                                         | 视觉 + 语言        | OCR、文档理解、多模态推理                | Megatron |

> 📖 新模型架构通过 Megatron Bridge 接入；Rollout、训练与权重转换所需的适配点请参阅[外部模型接入指南](docs/zh/guide/external-model-integration.md)。

______________________________________________________________________

## 📦 安装

推荐使用官方 Docker 镜像运行 Relax，镜像中已预装并版本对齐 CUDA、PyTorch、Megatron-LM、SGLang、Ray 等全部依赖。

```bash
# 拉取官方镜像
docker pull ghcr.io/redai-studio/relaxrl:latest

# 启动容器，挂载 GPU、共享内存与工作目录
docker run -it --gpus all --ipc=host --network=host \
  -v /path/to/your/workspace:/root \
  ghcr.io/redai-studio/relaxrl:latest bash

# 容器内克隆仓库并安装
git clone https://github.com/redai-studio/Relax.git /root/Relax
cd /root/Relax && pip install -e .
```

> 📖 关于 GPU 驱动要求、多节点部署与持久化存储挂载，请参阅 [安装指南](docs/zh/guide/installation.md)。

______________________________________________________________________

## 🚀 快速开始

三个端到端任务覆盖**文本**、**视觉-语言**、**全模态**训练。每个任务直接从 HuggingFace 下载数据集与模型，并通过单条脚本启动。只需将 `EXP_DIR` 指向模型与数据集所在的根目录（如 `/root`），脚本会自动定位。

### 任务一 — DAPO Math（文本，8 卡）

在 [`dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) 数学推理数据集上使用 GRPO 训练 Qwen3-4B，奖励采用规则抽取 + 符号数学校验。

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

cd /root/Relax && export EXP_DIR=/root
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

### 任务二 — Open-R1（视觉-语言，8 卡）

在 [`multimodal-open-r1-8k-verified`](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 上使用 GRPO 训练 Qwen3-VL-4B，奖励使用 `openr1mm`。

```bash
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir /root/multimodal-open-r1-8k-verified
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct

cd /root/Relax && export EXP_DIR=/root
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

### 任务三 — AVQA（全模态：图像 + 音频，16 卡 / 2 节点）

在 [`AVQA-R1-6K`](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) 上使用 GRPO 训练 Qwen3-Omni-30B-A3B，奖励采用多选题匹配。

```bash
hf download --repo-type dataset harryhsing/AVQA-R1-6K --local-dir /root/AVQA-R1-6K
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /root/Qwen3-Omni-30B-A3B-Instruct

cd /root/Relax && export EXP_DIR=/root
bash -x scripts/entrypoint/spmd-multinode.sh \
  scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

启动后若看到如下日志，说明训练已经正常运行：

```text
Finish rollout 0/200
training step 0/200
```

Relax 保存的 checkpoint 为 Megatron DCP 格式，可通过 `scripts/tools/convert_torch_dist_to_hf_bridge.py` 转换为 HuggingFace 权重。

> 📖 完整教程：[快速上手指南](docs/zh/guide/quick-start.md) · [自定义训练](docs/zh/guide/customize-training.md) · [配置指南](docs/zh/guide/configuration.md)

______________________________________________________________________

## ⚡ 核心特性

### 基于 TransferQueue 的全异步训练

在全异步模式下，Rollout、Actor、ActorFwd、Reference、Advantages 运行在**独立 GPU 集群**上完全并行。三大机制共同保证高吞吐：

- **StreamingDataLoader** — Actor 在 Rollout 增量写入 TransferQueue 的同时即可开始消费样本，消除阶段之间的 GPU 空闲。
- **可配置 staleness** — `--max-staleness` 精确控制数据新鲜度，在 on-policy 准确性与训练吞吐之间灵活权衡。
- **DCS 权重同步** — 每一步训练结束后，权重通过分布式 Checkpoint 服务（DCS）以 NCCL 广播方式从 Actor 分发至 Rollout/ActorFwd/Reference，与下一步训练计算重叠。

### Agentic RL

Relax 为多轮闭环"执行 → 观察 → 决策"训练提供一等公民支持：

- **多轮采样 + loss masking** — 模型输出（mask=1）与环境观察（mask=0）清晰区分，只有模型动作参与训练。
- **环境与 Rollout 解耦** — 标准 `BaseInteractionEnv` 接口（`reset`、`step`、`format_observation`）让环境独立于采样器演进。
- **VLM 多模态上下文累积** — Rollout 侧的 `image_data` 与训练侧的 `multimodal_train_inputs` 每轮增量合并，确保多轮视觉观察正确拼接。
- **灵活的终止条件** — 组合 `max_turns`、token 预算耗尽与环境 `done` 信号。DeepEyes 示例展示了 Qwen3-VL-30B-A3B 的 Agentic 多轮 GRPO 训练。

### Rollout 弹性扩缩容

由于 RL 训练中 60–70% 的时间花在 Rollout 阶段，Relax 通过 **HTTP REST API** 在训练过程中动态增减推理引擎，无需中断训练循环：

- **`ray_native`** 模式 — 指定目标引擎数量，Relax 自动在当前 Ray 集群内分配资源并启动新的 SGLang 引擎。
- **`external`** 模式 — 注册部署在其他集群的 SGLang 引擎，面向跨集群联邦推理，适合抢占式或闲置资源。

扩缩容操作异步、幂等、互斥，支持优雅缩容（等待在途请求完成）与取消回滚。启动参数定义的初始引擎受保护，只有动态添加的引擎可被缩容。

### Megatron 训练后端与 SGLang 推理

训练后端采用 **Megatron-LM**，完整支持 Tensor / Pipeline / Context / Expert 并行，适配 MoE 与超深模型。推理后端采用 **SGLang** 并统一管理进程生命周期。新模型架构通过 Megatron Bridge 接入，HF ↔ Megatron 权重自动转换。

### 可插拔奖励中心

内置数学（DeepScaler、DAPO）、GPQA、F1、IFBench、多选题、多模态 Open-R1，以及 **GenRM**（生成式 LLM-as-judge）等奖励函数。自定义奖励只需在 `relax/engine/rewards/` 中添加一个文件。

### 生产级运维

- **HealthManager** — 心跳监控 + 两级自动恢复（优先就地恢复，失败后全局重启）。
- **Metrics Service** — 集中式 Ray Serve 部署，向 TensorBoard、WandB、ClearML 分发指标。
- **Notifications** — 通过 Apprise 发送实时训练告警（Slack、微信、邮件等）。

______________________________________________________________________

## 📚 文档

完整的双语文档请访问 **[redai-studio.github.io/Relax](https://redai-studio.github.io/Relax)**。

______________________________________________________________________

## 🧪 示例

| 示例                                                         | 描述                                    |
| :----------------------------------------------------------- | :-------------------------------------- |
| [算法配方](./examples/algorithms/)                           | RLOO、REINFORCE++、CISPO 等策略优化算法 |
| [DeepEyes](./examples/deepeyes/)                             | 基于 Qwen3-VL 的多模态视觉语言 RL       |
| [Search-R1](./examples/search_r1/)                           | 搜索增强的单智能体与 multi-agent RL     |
| [Mini-SWE-Agent](./examples/mini_swe_agent/)                 | 面向软件工程任务的 Agentic RL           |
| [NeMo Gym Agentic](./examples/nemo_gym_agentic/)             | Agentic 环境集成与可运行配方            |
| [On-Policy Distillation](./examples/on_policy_distillation/) | 基于 KL 惩罚的师生知识蒸馏              |

______________________________________________________________________

## 🧩 基于 Relax 构建的项目

| 项目                                                     | 描述                                                                                                    |
| :------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| [HyperEyes](https://github.com/DeepExperience/HyperEyes) | 一个并行多模态搜索智能体，使用 Relax 进行高效的 RL 训练，结合视觉定位与检索能力并发搜索多个实体。       |
| [Iris](https://github.com/AllSpark-Research/Iris)        | 基于 Relax、从 Qwen3.5/3.6 后训练得到的开放权重搜索智能体系列，面向长程搜索、迭代证据收集与上下文管理。 |

______________________________________________________________________

## 🤝 参与贡献

欢迎各种形式的贡献！请阅读 [贡献指南](docs/zh/guide/how-to-contribute.md) 了解详情。

______________________________________________________________________

## 🛠️ AI 编程技能（Skills）

Relax 在 `skills/` 目录下内置了一套 [Claude Code](https://claude.ai/code) 斜杠命令技能，用于加速开发和运维工作。在 Claude Code 中以 `/技能名` 方式调用。

| 技能                 | 描述                                                           |
| :------------------- | :------------------------------------------------------------- |
| `/code-review`       | 专业代码审查 —— 检测 SOLID 违规、安全风险、ML/分布式训练问题   |
| `/debug-hang`        | 自动排查 Ray 分布式训练 hang 问题，收集调用栈与 Actor 状态     |
| `/dev`               | 开发调试 Relax 代码；向远程 Ray 集群提交并监控训练任务         |
| `/doc-writer`        | 编写和维护中英双语 VitePress 文档                              |
| `/git-commit`        | 生成 Conventional Commits 格式提交，自动运行 pre-commit 钩子   |
| `/model-integration` | 新模型架构接入训练管线的分步指南                               |
| `/perf-doctor`       | 审查训练启动脚本中的性能与显存配置问题                         |
| `/ssh-ray-cluster`   | SSH 连接远程 Ray 集群 Head 节点，检查状态、日志和调试任务      |
| `/verl-to-relax`     | 将 RL 配方从 verl 迁移到 Relax（奖励函数、工具环境、启动脚本） |
| `/creating-skills`   | 按 Anthropic 最佳实践编写新 Claude Code 技能的指南             |

______________________________________________________________________

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

______________________________________________________________________

## 📜 许可证

本项目基于 [Apache License 2.0](./LICENSE) 开源。

______________________________________________________________________

## 🙏 致谢

Relax 的构建离不开以下优秀的开源项目：

- [Slime](https://github.com/THUDM/slime) — 可扩展的强化学习训练与推理框架
- [SGLang](https://github.com/sgl-project/sglang) — 高性能大语言模型推理框架
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 与 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — 大规模分布式训练框架及 HF ↔ Megatron 权重转换桥接库，衷心感谢整个 **NVIDIA** 团队
- [TransferQueue](https://github.com/Ascend/TransferQueue) — 高性能分布式数据传输队列
- [Ray](https://github.com/ray-project/ray) — 分布式计算框架
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 最先进的模型中心

衷心感谢所有贡献者和开源社区的支持！
