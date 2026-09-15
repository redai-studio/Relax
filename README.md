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

**Relax** (**R**einforcement **E**ngine **L**everaging **A**gentic **X**-modality) is a high-performance reinforcement learning post-training framework open-sourced by the Xiaohongshu AI Infra Team for multimodal large language models. Built on Ray Serve with a service-oriented architecture, Relax uses Megatron-LM as the training backend and SGLang as the inference engine. Through the [TransferQueue](https://github.com/redai-studio/TransferQueue) data transfer system, it achieves complete decoupling of training and inference, supporting end-to-end multimodal RL training from text to images, videos, and audio.

______________________________________________________________________

## ✨ Highlights

- 🌐 **Full Omni-Modal Training** — One unified framework for text, vision, and audio RL — one of the few systems capable of end-to-end Omni model (Qwen3-Omni) post-training
- ⚙️ **Service-Oriented Six-Layer Architecture** — Every role is an independent Ray Serve deployment, with native service-level elastic scheduling and fault recovery
- ⚡ **Fully Async via TransferQueue** — Rollout, Actor, ActorFwd, Reference, and Advantages run on independent GPU clusters with streaming data exchange and configurable staleness
- 🔁 **Hybrid Mode** — Separate Actor/Rollout placement groups with TransferQueue streaming, while ref / actor_fwd / advantages run in-process on the actor — pairs `--balance-data` with sub-batched forward to minimize GPU waste
- 🤖 **Agentic RL** — Multi-turn interaction, loss masking, flexible termination, and VLM multimodal context carry-over for closed-loop "execute → observe → decide" training
- 🔀 **Elastic Rollout Scaling** — Dynamically grow/shrink inference engines mid-training via HTTP REST API, with same-cluster (`ray_native`) and cross-cluster (`external`) federation modes
- 🧠 **Rich Algorithm Suite** — PPO, GRPO, M2PO, RLOO, REINFORCE++ variants, GSPO, SAPO, CISPO, and On-Policy Distillation out of the box, with pluggable rewards and built-in **GenRM** (LLM-as-judge) mode
- 🚀 **Megatron + SGLang Backends** — Megatron-LM (TP/PP/CP/EP) for MoE and deep models, SGLang for high-throughput inference, DCS for NCCL-broadcast weight sync
- 📦 **Production-Ready Ops** — HealthManager auto-recovery, centralized Metrics Service (WandB / TensorBoard / ClearML), and Apprise real-time notifications

______________________________________________________________________

## 📢 News

| 📣 Updates                                                                                                                                                                                                    |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **\[08/20/2026\]** 🧠 Added **M2PO**, **RLOO**, and two **REINFORCE++** variants, with runnable RLOO and REINFORCE++ recipes plus an [Algorithm Reference](docs/en/examples/algorithms.md) covering all four. |
| **\[08/19/2026\]** 🤖 Added multi-agent training with multi-context trajectory export and a [Search-R1 reference recipe](examples/search_r1/).                                                                |
| **\[08/18/2026\]** 🪶 LoRA RL now supports MoE models in both adapter and merge workflows. See the [LoRA Training Guide](docs/en/guide/low-rank-adaptation-training.md).                                      |
| **\[05/26/2026\]** 🔁 New **Hybrid** execution mode — streaming data + in-process ref/actor_fwd, with `--balance-data` support. See the [Hybrid Training Guide](docs/en/guide/hybrid-training.md).            |
| **\[05/11/2026\]** 🚀 Support for Qwen3.6 series models (text + VLM)!                                                                                                                                         |
| **\[04/15/2026\]** 🎉 Relax is now open-source!                                                                                                                                                               |

______________________________________________________________________

## 🏗️ Architecture

<div align="center">
  <img src="./assets/arch.png" width="80%" alt="Relax Architecture">
</div>

Relax adopts a **six-layer service-oriented architecture** where every role is deployed as an independent [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) deployment, cleanly separating orchestration, components, engines, backends, and distributed capabilities:

| Layer             | Responsibility                                                                                                               |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **Entrypoints**   | `train.py` — signal handling, CLI parsing, Ray cluster connection, Controller launch                                         |
| **Orchestration** | `Controller` (training loop, global restart), `Service` (placement groups, lifecycle), `Registry` (role & algorithm mapping) |
| **Components**    | Ray Serve deployments: **Actor**, **Rollout**, **Critic**, **ActorFwd**, **Advantages**, **GenRM**                           |
| **Engine**        | SGLang rollout engine, pluggable reward functions, request router, data filters                                              |
| **Backends**      | **Megatron-LM** training backend (TP/PP/CP/EP) and **SGLang** inference engine                                               |
| **Distributed**   | Ray Actor groups (RolloutManager / GenRMManager) and **DCS** (Distributed Checkpoint Service) for NCCL/GLOO weight sync      |

**Three execution modes** are supported:

- **Colocate (Sync)** — Actor and Rollout time-share the same GPUs; Rollout writes a full batch to TransferQueue, then yields GPUs for training. Memory-efficient for constrained hardware and strict on-policy (`max_staleness=0`).
- **Fully Async** — Actor, Rollout, ActorFwd, Reference, and Advantages run on **independent GPU clusters** in parallel, exchanging data through TransferQueue and syncing weights asynchronously through DCS for maximum throughput with configurable staleness.
- **Hybrid** — Actor and Rollout sit on **separate GPU placement groups** (like Fully Async) and exchange data via TransferQueue with configurable staleness, but ref / actor_fwd / advantages run **in-process on the actor's own GPUs** via `TensorBackuper` + `_switch_model` (like Colocate). Enables streaming pipelines plus `--balance-data` without paying for standalone ref/actor_fwd services.

> 📖 Learn more: [Architecture Guide](docs/en/guide/architecture.md) · [Fully Async Training](docs/en/guide/fully-async-training.md) · [Hybrid Training](docs/en/guide/hybrid-training.md) · [Elastic Rollout Scaling](docs/en/guide/elastic-rollout.md)

______________________________________________________________________

## 🧠 Supported Algorithms

| Algorithm                  | Type                | Description                                       |
| :------------------------- | :------------------ | :------------------------------------------------ |
| **PPO**                    | Actor-Critic        | Proximal Policy Optimization                      |
| **GRPO**                   | Policy Optimization | Group Relative Policy Optimization                |
| **M2PO**                   | Policy Optimization | Second-Moment Trust Policy Optimization           |
| **RLOO**                   | Policy Optimization | REINFORCE Leave-One-Out                           |
| **REINFORCE++**            | Policy Optimization | Token KL-to-go with global normalization          |
| **REINFORCE++-baseline**   | Policy Optimization | Group-baseline REINFORCE++ variant                |
| **GSPO**                   | Policy Optimization | Group-wise Sequence-level Policy Optimization     |
| **SAPO**                   | Policy Optimization | Soft Adaptive Policy Optimization                 |
| **CISPO**                  | Policy Optimization | Clipped Importance-ratio Soft Policy Optimization |
| **On-Policy Distillation** | Knowledge Transfer  | Teacher-student KL penalty distillation           |

> 📖 See the [Algorithm Reference](docs/en/examples/algorithms.md) for objectives, recipes, and execution-mode constraints.

______________________________________________________________________

## 🤖 Supported Models

Relax is designed for **omni-modal RL training** — text, vision, and audio in one unified framework. Multimodal data is configured via the `--multimodal-keys` flag, with complete image/video/audio processing pipelines under `relax/utils/multimodal/` for fine-grained control over image token counts, video frame sampling, and audio sample rates.

| Model Family                                                    | Sizes                                      | Modality              | Typical Tasks                                        | Backend  |
| :-------------------------------------------------------------- | :----------------------------------------- | :-------------------- | :--------------------------------------------------- | :------- |
| **Qwen3**                                                       | 4B, 30B-A3B (MoE)                          | Text                  | Math reasoning, code, multi-turn dialogue, tool use  | Megatron |
| **Qwen3-VL**                                                    | 4B, 30B-A3B                                | Vision + Language     | Visual QA, image understanding, multimodal reasoning | Megatron |
| **Qwen3.5**                                                     | 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B | Text + Vision         | Reasoning, SFT, visual QA, multimodal reasoning      | Megatron |
| **Qwen3-Omni**                                                  | 30B-A3B                                    | Text + Vision + Audio | Audio-visual QA, omni-modal understanding            | Megatron |
| **Qwen3.6**                                                     | 27B, 35B-A3B                               | Text + Vision         | Reasoning, visual QA, multimodal reasoning           | Megatron |
| **Qwen3.8**                                                     | 27B                                        | Text + Vision         | Reasoning, SFT, visual QA, multimodal reasoning      | Megatron |
| **GLM5**                                                        | 744B-A40B (MoE)                            | Text                  | Math reasoning, code, multi-turn dialogue            | Megatron |
| **Kimi K2.6**                                                   | ~1T-A32B (MoE)                             | Vision + Language     | Visual QA, multimodal reasoning; INT4 QAT training   | Megatron |
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B                                         | Vision + Language     | OCR, document understanding, multimodal reasoning    | Megatron |

> 📖 New architectures are integrated through Megatron Bridge; see the [External Model Integration Guide](docs/en/guide/external-model-integration.md) for the rollout, training, and weight-conversion hooks.

______________________________________________________________________

## 📦 Installation

The recommended way to run Relax is via the official Docker image, which ships with all CUDA, PyTorch, Megatron-LM, SGLang, and Ray dependencies pre-installed and version-matched.

```bash
# Pull the official image
docker pull ghcr.io/redai-studio/relaxrl:latest

# Launch a container with GPUs, shared memory, and your workspace mounted
docker run -it --gpus all --ipc=host --network=host \
  -v /path/to/your/workspace:/root \
  ghcr.io/redai-studio/relaxrl:latest bash

# Inside the container
git clone https://github.com/redai-studio/Relax.git /root/Relax
cd /root/Relax && pip install -e .
```

> 📖 For GPU driver requirements, multi-node setup, and persistent storage mounts, see the [Installation Guide](docs/en/guide/installation.md).

______________________________________________________________________

## 🚀 Quick Start

Three end-to-end tasks cover **text**, **vision-language**, and **omni-modal** training. Each task downloads a public HuggingFace dataset and model, then launches training with a single script. Set `EXP_DIR=/root` (or wherever your models and datasets live) and the scripts will locate them automatically.

### Task 1 — DAPO Math (Text, 8 GPUs)

Train Qwen3-4B on [`dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) with GRPO. Reward is rule-based answer extraction plus symbolic math verification.

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

cd /root/Relax && export EXP_DIR=/root
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

### Task 2 — Open-R1 (Vision-Language, 8 GPUs)

Train Qwen3-VL-4B on [`multimodal-open-r1-8k-verified`](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) with GRPO using the `openr1mm` reward.

```bash
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir /root/multimodal-open-r1-8k-verified
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct

cd /root/Relax && export EXP_DIR=/root
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

### Task 3 — AVQA (Omni-Modal: Image + Audio, 16 GPUs / 2 nodes)

Train Qwen3-Omni-30B-A3B on [`AVQA-R1-6K`](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) with GRPO and a multiple-choice reward.

```bash
hf download --repo-type dataset harryhsing/AVQA-R1-6K --local-dir /root/AVQA-R1-6K
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /root/Qwen3-Omni-30B-A3B-Instruct

cd /root/Relax && export EXP_DIR=/root
bash -x scripts/entrypoint/spmd-multinode.sh \
  scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

Once running, you should see logs like:

```text
Finish rollout 0/200
training step 0/200
```

Checkpoints are saved in Megatron DCP format; convert them to HuggingFace weights with `scripts/tools/convert_torch_dist_to_hf_bridge.py`.

> 📖 Full walkthrough: [Quick Start Guide](docs/en/guide/quick-start.md) · [Customize Training](docs/en/guide/customize-training.md) · [Configuration Guide](docs/en/guide/configuration.md)

______________________________________________________________________

## ⚡ Key Features

### Fully Async Training via TransferQueue

In fully-async mode, Rollout, Actor, ActorFwd, Reference, and Advantages run on **independent GPU clusters** in parallel. Three mechanisms make this efficient:

- **StreamingDataLoader** — Actor begins consuming samples as Rollout incrementally writes them to TransferQueue, eliminating GPU idle time between phases.
- **Configurable staleness** — `--max-staleness` precisely controls how off-policy training data can drift, flexibly balancing on-policy accuracy and throughput.
- **DCS weight sync** — After each training step, weights are NCCL-broadcast from Actor to Rollout/ActorFwd/Reference via the Distributed Checkpoint Service, overlapped with the next training computation.

### Agentic RL

Relax provides first-class support for multi-turn, closed-loop "execute → observe → decide" training:

- **Multi-turn sampling with loss masking** — model outputs (mask=1) are cleanly separated from environment observations (mask=0) so only model actions participate in training.
- **Environment / Rollout decoupling** — a standard `BaseInteractionEnv` interface (`reset`, `step`, `format_observation`) lets environments evolve independently of the sampler.
- **VLM multimodal context carry-over** — `image_data` on the Rollout side and `multimodal_train_inputs` on the training side are incrementally merged each turn so visual observations concatenate correctly.
- **Flexible termination** — combine `max_turns`, token-budget exhaustion, and env-signalled `done`. The DeepEyes example demonstrates Agentic multi-turn GRPO with Qwen3-VL-30B-A3B.

### Elastic Rollout Scaling

Since 60–70% of RL training time is spent in the Rollout phase, Relax exposes **HTTP REST APIs** to dynamically add or remove inference engines mid-training without interrupting the training loop:

- **`ray_native`** mode — specify a target engine count; Relax allocates resources and launches new SGLang engines inside the current Ray cluster.
- **`external`** mode — register SGLang engines already deployed in other clusters for cross-cluster federated inference on preemptible or idle resources.

Scaling is asynchronous, idempotent, mutually exclusive, and supports graceful drain-and-remove plus cancellation with rollback. Engines from startup parameters are protected; only dynamically added engines can be scaled in.

### Megatron Training Backend & SGLang Inference

Training uses **Megatron-LM** with full Tensor / Pipeline / Context / Expert parallelism for MoE and ultra-deep models. Inference uses **SGLang** with process-lifecycle management. New model architectures plug in through Megatron Bridge for automatic HF ↔ Megatron weight conversion.

### Pluggable Reward Hub

Built-in rewards for math (DeepScaler, DAPO), GPQA, F1, IFBench, multiple-choice, multimodal Open-R1, and **GenRM** (generative LLM-as-judge). Add a custom reward by dropping a single file into `relax/engine/rewards/`.

### Production Operations

- **HealthManager** — heartbeat monitoring with two-tier auto-recovery (in-place restart first, global restart as fallback).
- **Metrics Service** — centralized Ray Serve deployment that fans out to TensorBoard, WandB, and ClearML.
- **Notifications** — real-time training alerts via Apprise (Slack, WeChat, email, and more).

______________________________________________________________________

## 📚 Documentation

Full bilingual documentation is available at **[redai-studio.github.io/Relax](https://redai-studio.github.io/Relax)**.

______________________________________________________________________

## 🧪 Examples

| Example                                                      | Description                                             |
| :----------------------------------------------------------- | :------------------------------------------------------ |
| [Algorithm Recipes](./examples/algorithms/)                  | RLOO, REINFORCE++, CISPO, and related policy optimizers |
| [DeepEyes](./examples/deepeyes/)                             | Multi-modal vision-language RL with Qwen3-VL            |
| [Search-R1](./examples/search_r1/)                           | Search-augmented single-agent and multi-agent RL        |
| [Mini-SWE-Agent](./examples/mini_swe_agent/)                 | Agentic RL for software-engineering tasks               |
| [NeMo Gym Agentic](./examples/nemo_gym_agentic/)             | Agentic environment integrations and runnable recipes   |
| [On-Policy Distillation](./examples/on_policy_distillation/) | Teacher-student knowledge distillation via KL penalty   |

______________________________________________________________________

## 🧩 Projects Built upon Relax

| Project                                                  | Description                                                                                                                                                             |
| :------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [HyperEyes](https://github.com/DeepExperience/HyperEyes) | A parallel multimodal search agent that uses Relax for efficient RL training, combining visual grounding and retrieval to search across multiple entities concurrently. |
| [Iris](https://github.com/AllSpark-Research/Iris)        | An open-weight search agent family post-trained with Relax from Qwen3.5/3.6, designed for long-horizon search, iterative evidence gathering, and context management.    |

______________________________________________________________________

## 🤝 Contributing

We welcome contributions of all kinds! Please read our [Contributing Guide](docs/en/guide/how-to-contribute.md) to get started.

______________________________________________________________________

## 🛠️ AI Coding Skills

Relax ships a set of [Claude Code](https://claude.ai/code) slash-command skills under `skills/` to accelerate development and operations. Invoke them in Claude Code with `/skill-name`.

| Skill                | Description                                                                                     |
| :------------------- | :---------------------------------------------------------------------------------------------- |
| `/code-review`       | Expert review of git changes — SOLID violations, security risks, ML/distributed training issues |
| `/debug-hang`        | Automatically diagnose Ray distributed training hangs — collects call stacks and actor states   |
| `/dev`               | Develop and debug Relax code; submit and monitor jobs on a remote Ray cluster                   |
| `/doc-writer`        | Write and maintain bilingual (English + Chinese) VitePress documentation                        |
| `/git-commit`        | Create Conventional Commits with rich markdown body and auto-run pre-commit hooks               |
| `/model-integration` | Step-by-step guide for integrating new model architectures into the training pipeline           |
| `/perf-doctor`       | Audit training launch scripts for performance and GPU memory misconfiguration                   |
| `/ssh-ray-cluster`   | SSH into a remote Ray cluster head node to inspect status, logs, and debug jobs                 |
| `/verl-to-relax`     | Migrate RL recipes from verl to Relax (rewards, tool envs, launch scripts)                      |
| `/creating-skills`   | Guide for authoring new Claude Code skills following Anthropic best practices                   |

______________________________________________________________________

## 📝 Citation

If you find Relax useful in your research, please cite:

```bibtex
@software{relax2026,
  title  = {Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale},
  author = {Relax Contributors},
  url    = {https://arxiv.org/abs/2604.11554},
  year   = {2026}
}
```

______________________________________________________________________

## 📜 License

This project is licensed under the [Apache License 2.0](./LICENSE).

______________________________________________________________________

## 🙏 Acknowledgements

Relax is built upon the shoulders of excellent open-source projects:

- [Slime](https://github.com/THUDM/slime) — Scalable training and inference framework for reinforcement learning
- [SGLang](https://github.com/sgl-project/sglang) — Fast serving framework for large language models
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) & [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — Large-scale distributed training framework and HF ↔ Megatron weight conversion bridge, with sincere thanks to the entire **NVIDIA** team
- [TransferQueue](https://github.com/Ascend/TransferQueue) — High-performance distributed data transfer queue
- [Ray](https://github.com/ray-project/ray) — Distributed computing framework
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — State-of-the-art model hub

We sincerely thank all contributors and the open-source community for making this project possible.
