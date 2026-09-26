# Generative Reward Model (GenRM) Examples

Training examples that use a **Generative Reward Model** (GenRM) — an LLM-as-judge approach — to score rollout responses, replacing traditional trained reward models.

## Overview

GenRM leverages a pre-trained LLM (e.g. Qwen3-VL-30B-A3B-Instruct) to evaluate whether model responses are consistent with ground-truth labels. Instead of training a separate reward model, GenRM performs inference-time evaluation via an SGLang engine deployed as an independent Ray Serve service.

Key benefits:

- **Zero reward-model training** — uses an off-the-shelf LLM directly
- **Strong generalization** — leverages LLM reasoning for evaluation on unseen tasks
- **Flexible criteria** — evaluation behavior is controlled via prompt templates

Both scripts in this directory train **Qwen3-4B** with **GRPO** on the `dapo-math-17k` dataset, using GenRM (`--rm-type dapo-genrm`) for reward scoring and AIME-2024 for evaluation.

## Scripts

| Script                                          | Mode                  | Description                                                                                                                                                                                                          |
| :---------------------------------------------- | :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run-qwen3-4B-8xgpu-colocated.sh`               | Colocate (sync)       | Actor & Rollout share GPUs; GenRM on separate GPUs                                                                                                                                                                   |
| `run-qwen3-4B-8xgpu-async.sh`                   | Fully Async           | Independent GPU pools per role; maximum throughput                                                                                                                                                                   |
| `run-qwen35-35B-A3B-16xgpu-genrm-397B-split.sh` | Split-bundle colocate | 35B-A3B policy + 397B FP8 GenRM on 16 GPU (2 nodes), rollout / GenRM on separate 8-GPU shards, inline reward                                                                                                         |
| `run-qwen35-35B-A3B-16xgpu-genrm-397B-defer.sh` | Defer / swap colocate | 35B-A3B policy + 397B FP8 GenRM on 16 GPU (2 nodes), shared bundles, two-phase sleep-wake swap, batch reward                                                                                                         |
| `run-qwen3-4B-8xgpu-dual-genrm-split.sh`        | Split-bundle colocate | Qwen3-4B policy on 4 GPU + **two** GenRM instances (`--genrm-instances`) on the other 4 GPU, routed by `route_key`; see [Multi-Instance GenRM](#multi-instance-genrm-multiple-judge-models-behind-one-service) below |

`reward_dual_genrm_quality_safety.py` is the custom reward function (`--custom-rm-path`) used by the dual-GenRM script above. It calls two independently-deployed judge models — one scoring answer correctness (`route_key="quality"`), one scoring response safety (`route_key="safety"`) — and combines both into a single training reward (`score = quality_score * safety_score`). Use it as a template for routing different reward signals — or, in agentic settings, different agent modules — to different GenRM instances.

## 大模型 GenRM 两种部署模式选择 (35B-A3B + 397B FP8, 16GPU)

同样是 policy + 大 GenRM 共 16 卡的场景，`split` 与 `defer` 两个脚本对应两种正交的资源编排思路，收益模型不一样，**先看你的 rollout 长尾情况再选**。

### `run-qwen35-35B-A3B-16xgpu-genrm-397B-split.sh` — 拆分并行

```
GPU 0..7  : rollout (35B-A3B, TP=8)
GPU 8..15 : GenRM   (397B FP8, TP=DP=EP=8)
```

- **触发条件**: `rollout_num_gpus + genrm_num_gpus == actor_total_gpus` 命中 `relax/utils/arguments.py:2887` 的 split-bundles 分支。
- **打分方式**: `--rm-type dapo-genrm`，每个 sample 生成完立即通过 HTTP 打给 GenRM，与后续 rollout **重叠**。
- **优点**: rollout 后段的长尾（tail latency）时间被 GenRM 复用，两个 8 卡分片全程并行不空转。
- **缺点**: rollout 只拿到 8 卡；GenRM 也只拿到 8 卡；两个都受限于一半的算力。

**适用场景**:

- **Agentic / 多轮工具调用** 场景，response 长度方差大，长尾明显（一部分 request 可能要跑几十秒才结束）——此时 GenRM 的 8 卡如果闲着就是纯浪费，让它并行打分能把闲置算力拉起来。
- 打分本身不是关键路径的场景。

### `run-qwen35-35B-A3B-16xgpu-genrm-397B-defer.sh` — 两阶段串行

```
Phase A (rollout):  16 GPU 全给 rollout (2× TP=8 engines), GenRM 睡眠
Phase B (score):    16 GPU 全给 GenRM   (2× TP=8 engines), rollout 睡眠
Phase C (train):    actor 训练 + weight update, GenRM 保持睡眠
```

- **触发条件**: `rollout_num_gpus == genrm_num_gpus == actor_total_gpus` 命中 shared-bundles 分支 + `--defer-reward-to-post-process` + `--rm-type dummy` + `--custom-reward-post-process-path examples.generate_reward_model.post_process_genrm_swap.custom_reward_post_process`。
- **打分方式**: inline reward 是 no-op（rm-type=dummy 返回 0），全部 rollout 完成后 `post_process_genrm_swap.py` 编排 offload rollout → onload GenRM → 批量打分 → offload GenRM 的顺序切换。
- **优点**: rollout 和 GenRM 各自都能吃满 16 卡；相比 split 版 GenRM 吞吐 ~2×、rollout 也可能 ~1.8×（视模型/batch）。SGLang GenRM engine 一次收到全 batch 请求，prefill 能凑成大 batch，调度效率更高。
- **缺点**: 少了 rollout ↔ GenRM 的重叠；多了两次 sleep/wake（release/resume_memory_occupation）的秒级开销。

**适用场景**:

- **RLVR / rule-based verifier** 或结构化 math 数据集，response 长度分布集中，rollout 长尾**不严重**——此时 split 模式那 8 卡 GenRM 大部分时间在等 rollout tail，浪费一半算力。
- Rollout kernel 本身很重（长 context / MoE 大模型），扩到 16 卡的加速比接近线性。
- 打分 model 显著大于 policy（比如 397B GenRM vs 35B policy），单批攒足够大再打分能显著提升 GenRM 侧的 GPU 利用率。

### 选择的经验法则

| 观察                                                  | 推荐                                                     |
| :---------------------------------------------------- | :------------------------------------------------------- |
| Rollout p50/p99 时长比 > 2×，或多轮 agentic           | **split**                                                |
| Rollout 长度方差小、rule-based 或短 response verifier | **defer**                                                |
| GenRM 明显比 policy 大（4×+）                         | **defer**（GenRM 拿满卡收益最大）                        |
| GenRM 比 policy 小或相当                              | **split**（GenRM 8 卡也够用，overlap 更划算）            |
| 不确定                                                | 先跑 **defer** 拿到 baseline wall-time，再对比 **split** |

## Multi-Instance GenRM (multiple judge models behind one service)

Everything above assumes a single judge model. `--genrm-instances` lets one GenRM Serve deployment host **several independent judge models** at once (different sizes, different checkpoints, different scoring criteria), each addressed by a `route_key` string the caller supplies per request. This is the mechanism behind `run-qwen3-4B-8xgpu-dual-genrm-split.sh` / `reward_dual_genrm_quality_safety.py` above, and it is the natural fit for **agentic pipelines** where different modules (planner, tool-caller, final-answer judge, ...) want different judges.

See the [**Multi-Instance GenRM** section](../../docs/en/examples/generative-reward-model.md#multi-instance-genrm-multiple-judge-models-behind-one-service) of the full GenRM guide (or its [Chinese version](../../docs/zh/examples/generative-reward-model.md#多实例-genrm一个服务托管多个评判模型)) for the complete `--genrm-instances` reference, the `route_key` request/response format, and an agentic multi-module example. Quick summary:

- Configure `--genrm-instances '{"<route_key>": {"model_path": ..., "num_gpus": ...}, ...}'` instead of `--genrm-model-path` — each entry is a fully independent judge with its own GPU budget (`num_gpus` is required per instance; there is no automatic even split).
- From a reward function, call `await genrm_client.generate(messages, route_key="<route_key>")` to pick which instance answers the call. Omitting `route_key` (or using the legacy `--genrm-model-path` single-model config) falls back to the sole `"__default__"` instance.
- `--resource`'s `"genrm"` entry must equal the **sum** of every instance's `num_gpus` (Split/Shared bundle math is otherwise unchanged from the single-model case).

### Resource Layout

**Colocate mode** (`--colocate`):

```
Actor (training):  8 GPU (colocated with rollout)
Rollout:           4 GPU (time-shared with actor)
GenRM:             4 GPU
```

**Async mode** (`--fully-async`):

```
Actor (training):  2 GPU
Rollout:           3 GPU
Reference:         1 GPU
Actor Forward:     1 GPU
GenRM:             1 GPU
```

## Quick Start

### Prerequisites

1. **Model weights** — Download Qwen3-4B (policy) and Qwen3-VL-30B-A3B-Instruct (GenRM judge):

   ```bash
   # Place under exps/ (or set EXP_DIR / MODEL_DIR)
   exps/Qwen3-4B/
   exps/Qwen3-VL-30B-A3B-Instruct/
   ```

2. **Dataset** — `dapo-math-17k` and `aime-2024` for evaluation:

   ```bash
   exps/dapo-math-17k/dapo-math-17k.jsonl
   exps/aime-2024/aime-2024.jsonl
   ```

3. **Ray cluster** — A running Ray cluster reachable at `http://127.0.0.1:8265`.

### Run Training

```bash
# Colocate mode (memory-efficient, 8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-colocated.sh

# Fully async mode (higher throughput, 8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-async.sh
```

## Key Parameters

| Parameter                     | Default                      | Description                                                                                                                                                                                                  |
| :---------------------------- | :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--rm-type dapo-genrm`        | —                            | Use DAPO-GenRM reward function                                                                                                                                                                               |
| `--genrm-model-path`          | —                            | Path to the GenRM judge model (single-instance config)                                                                                                                                                       |
| `--genrm-instances`           | —                            | JSON dict deploying multiple named judge models, routed by `route_key`; takes priority over `--genrm-model-path`. See [Multi-Instance GenRM](#multi-instance-genrm-multiple-judge-models-behind-one-service) |
| `--genrm-num-gpus-per-engine` | 1 or 4                       | GPUs allocated per GenRM SGLang engine                                                                                                                                                                       |
| `--genrm-engine-config`       | `{"max_context_len": 10240}` | SGLang engine configuration                                                                                                                                                                                  |
| `--genrm-sampling-config`     | `{"temperature": 0.1, ...}`  | Sampling params for the judge                                                                                                                                                                                |
| `--max-staleness`             | 0 (coloc) / 2 (async)        | Max data staleness for async training                                                                                                                                                                        |

## File Structure

```
examples/generate_reward_model/
├── README.md                                              # This document
├── run-qwen3-4B-8xgpu-colocated.sh                        # 4B colocate mode
├── run-qwen3-4B-8xgpu-async.sh                            # 4B fully async mode
├── run-qwen35-35B-A3B-16xgpu-genrm-397B-split.sh          # 35B + 397B GenRM, split-bundle (inline reward)
├── run-qwen35-35B-A3B-16xgpu-genrm-397B-defer.sh          # 35B + 397B GenRM, shared-bundle two-phase swap
├── run-qwen3-4B-8xgpu-dual-genrm-split.sh                 # 4B policy + two GenRM instances (--genrm-instances), split-bundle
├── reward_dual_genrm_quality_safety.py                    # Custom reward routing to two GenRM instances via route_key
└── post_process_genrm_swap.py                             # Custom post-process function used by the defer script
```

## Further Reading

- [GenRM Example](../../docs/en/examples/generative-reward-model.md) — Full GenRM architecture, configuration, and script walkthrough
- [Fully Async Training](../../docs/en/guide/fully-async-training.md) — How async mode works in Relax
- [Configuration Guide](../../docs/en/guide/configuration.md) — Complete parameter reference
