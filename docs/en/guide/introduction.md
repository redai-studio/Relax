# Relax Introduction

## What is Relax?

**Relax** (**R**einforcement **E**ngine **L**everaging **A**gentic **X**-modality) is a high-performance reinforcement learning post-training framework for multimodal large language models. Built on Ray Serve with a service-oriented architecture, Relax uses Megatron-LM as the training backend and SGLang as the inference engine. Through the [TransferQueue](https://github.com/redai-studio/TransferQueue) data transfer system, it achieves complete decoupling of training and inference, supporting end-to-end multimodal RL training from text to images, videos, and audio.

---

## Core Features

### 🌐 Full Multimodal Training Support

Relax natively supports multimodal RL training across text, images, videos, and audio—one of the few systems in the industry capable of completing Omni model RL training within a unified framework.

| Modality | Capabilities | Representative Models |
|----------|--------------|----------------------|
| **Text** | Math reasoning, code generation, multi-turn dialogue, tool use | Qwen3, GLM5 |
| **Vision** | Visual QA, image understanding, multimodal reasoning | Qwen3-VL, Qwen3.5, Qwen3.6, Kimi K2.6 |
| **Omni** | Joint image, text, and audio understanding | Qwen3-Omni |

Multimodal data is flexibly configured via the `--multimodal-keys` parameter. The framework includes complete image, video, and audio processing pipelines (`relax/utils/multimodal/`), supporting fine-grained control over image token counts, video frame sampling, audio sample rates, and more.

### ⚙️ Service-Oriented Six-Layer Architecture

Relax adopts a service-oriented six-layer architecture where all components are deployed as independent Ray Serve services, natively supporting service-level elastic scheduling and fault recovery. See [Architecture Design](./architecture.md) for details.

### ⚡ Fully Asynchronous Training via TransferQueue

Open source at [TransferQueue](https://github.com/redai-studio/TransferQueue). See [Fully Asynchronous Training](./fully-async-training.md) for details.

In fully async mode, five roles—Rollout (inference), Actor (training), ActorFwd (forward pass), Reference (reference model), and Advantages (advantage computation)—run on **independent GPU clusters** and exchange data via TransferQueue, with weights synchronized asynchronously through DCS (Distributed Checkpoint Service).

**Core Mechanisms**:

- **StreamingDataLoader**: Actor uses a streaming data loader that begins consuming training data as Rollout incrementally writes it, eliminating wait time
- **Configurable Staleness**: The `--max-staleness` parameter precisely controls data freshness, flexibly balancing on-policy accuracy and training throughput
- **DCS Weight Synchronization**: After training, weights are distributed to Rollout/ActorFwd/Reference via NCCL broadcast, overlapped with training computation

### 🔀 Elastic Rollout Scaling

See [Elastic Scaling](./elastic-rollout.md) for details.

Relax supports **dynamically adjusting the number of Rollout engines** during training via HTTP REST API without interrupting the training process. Since 60–70% of RL training time is spent in the Rollout phase, elastic scaling allows flexible adjustment of inference resources based on actual load.

**Two Scaling Modes**:

| Mode | Scenario | Description |
|------|----------|-------------|
| **ray_native** | Same-cluster scaling | Specify target engine count; automatically create new engines within the Ray cluster |
| **external** | Cross-cluster federated inference | Integrate externally deployed SGLang engines for cross-cluster elastic compute utilization |

### 🎯 Agentic RL

Unlike traditional VLM single-turn QA, **Agentic VLM** achieves closed-loop iteration through continuous interaction—"execute → observe → decide". Relax includes complete support for multi-turn Agentic training:

- **Multi-turn sampling and loss masking**: Through custom generate functions, the model generates action instructions each turn based on current context, the environment returns observations and incrementally injects them into context until task completion or termination. After obtaining the full trajectory, loss masking precisely distinguishes model outputs (mask=1) from environment feedback (mask=0), ensuring only model actions participate in training
- **Environment and Rollout decoupling**: Relax defines a standard interface for environments (`BaseInteractionEnv`), including `reset()`, `step(response) → (observation, done, info)`, and `format_observation()`. How the environment parses actions is completely independent of sampling and training, facilitating reuse and extension
- **VLM multimodal context maintenance**: In visual multi-turn interactions, each observation may carry new images/videos. Relax simultaneously maintains `image_data` on the Rollout side (incrementally appending encoded images per turn) and `multimodal_train_inputs` on the training side (incrementally merging processor-generated tensors per turn), enabling correct concatenation of multi-turn multimodal data
- **Flexible termination conditions**: Supports combinations of three termination mechanisms: `max_turns` (maximum interaction turns), token budget (truncate when available token budget is exhausted), and `env done` (environment signals task completion)
- **Typical Application**: DeepEyes task—Agentic Multi-Turn GRPO Training using Qwen3-VL-30B-A3B

## Model Configurations and Examples

The following models have configurations or integration code in this repository. Training modes and constraints are described by the linked scripts and guides; model configurations are collected in [`scripts/models/`](../../../scripts/models/).

| Model | Scope | Configuration / guide |
|---|---|---|
| Qwen2.5 | Text | [7B](../../../scripts/models/qwen25-7B.sh) |
| Qwen3 | Text | [0.6B](../../../scripts/models/qwen3-0.6B.sh), [1.7B](../../../scripts/models/qwen3-1.7B.sh), [4B](../../../scripts/models/qwen3-4B.sh), [8B](../../../scripts/models/qwen3-8B.sh), [32B](../../../scripts/models/qwen3-32B.sh), [30B-A3B](../../../scripts/models/qwen3-30B-A3B.sh) |
| GLM4.7 | Text | [30B-A3B](../../../scripts/models/glm4.7-30B-A3B.sh) |
| GLM5 | Text | [744B-A40B](../../../scripts/models/glm5-744B-A40B.sh) |
| DeepSeek-V4-Flash | Text | [Flash](../../../scripts/models/deepseek-v4-flash.sh) |
| Gemma4 | Text recipes | [26B-A4B](../../../scripts/models/gemma4-26B.sh), [31B](../../../scripts/models/gemma4-31B.sh) |
| Qwen3-VL | Vision + language | [2B](../../../scripts/models/qwen3-vl-2B.sh), [4B](../../../scripts/models/qwen3-vl-4B.sh), [8B](../../../scripts/models/qwen3-vl-8B.sh), [30B-A3B](../../../scripts/models/qwen3-vl-30B-A3B.sh) |
| Qwen3.5 | Vision + language | [4B](../../../scripts/models/qwen35-4B.sh), [9B](../../../scripts/models/qwen35-9B.sh), [27B](../../../scripts/models/qwen35-27B.sh), [35B-A3B](../../../scripts/models/qwen35-35B-A3B.sh), [122B-A10B](../../../scripts/models/qwen35-122B-A10B.sh), [397B-A17B](../../../scripts/models/qwen35-397B-A17B.sh) |
| Qwen3.6 | Vision + language | [27B](../../../scripts/models/qwen36-27B.sh), [35B-A3B](../../../scripts/models/qwen36-35B-A3B.sh) |
| Qwen3.8 | Vision + language | [27B](../../../scripts/models/qwen38-27B.sh) |
| Kimi K2.6 | Vision + language | [K2.6](../../../scripts/models/kimi-k2.6.sh) |
| Kimi K3 | Multimodal full-parameter / LoRA SFT and export | [K3](../../../scripts/models/kimi-k3.sh) · [Training and export guide](../../../examples/models/kimi-k3/README.md) |
| dots.mocr | OCR / document understanding | [3B](../../../scripts/models/dotsocr2.sh) |
| Qwen3-Omni | Text + vision + audio | [30B-A3B](../../../scripts/models/qwen3-omni-30B-A3B.sh) |
| Qwen-Image | Text-to-image diffusion RL | [Training guide](./diffusion-generative-rl.md) |

## Project Structure

```
relax/                   Core framework: six-layer architecture
examples/                examples (DeepEyes, etc.)
scripts/                 Training launch scripts for various models & scales
configs/                 Runtime environment configuration
```

---

## Next Steps

- [Installation Guide](./installation.md) — Environment setup and dependency installation
- [Quick Start](./quick-start.md) — Run your first RL training task
- [Configuration Reference](./configuration.md) — Complete configuration parameter manual
- [Architecture Design](./architecture.md) — Deep dive into Relax's six-layer architecture
- [DeepEyes Example](../examples/deepeyes.md) — Multimodal vision RL training in practice
