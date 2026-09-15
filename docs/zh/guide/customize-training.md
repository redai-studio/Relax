# 自定义训练

## 前置条件

确保您已完成[安装](./installation.md)步骤。

## 模型准备

### 下载

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。以下是使用 `huggingface_hub` 下载示例资源的命令：

```bash
# 下载模型权重 (Qwen3-VL-4B)
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

### Megatron 权重转 HF 权重

::: tip 使用 Megatron Bridge 无需手动转换
Relax 默认使用 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 作为训练后端的权重桥接层，在训练过程中自动完成 HF ↔ Megatron 权重的双向转换，**无需任何手动转换步骤**。只需在启动脚本中指定以下选项即可：
:::

```bash
--megatron-to-hf-mode bridge
```

### HF 权重转 Megatron 权重

详见[快速上手 — 导出模型](./quick-start.md#导出模型)。

### 新增模型

新增模型需要完成以下两部分工作：

#### 1. 模型配置脚本

模型配置文件位于 `scripts/models/`，从 HF config 中提取对应的 Megatron 架构参数。例如 `scripts/models/qwen3-4B.sh`：

```bash
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
```

添加后在训练启动脚本中 `source` 对应模型的配置文件即可。

#### 2. Megatron Bridge 模型适配

Relax 通过 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 实现 HF ↔ Megatron 的自动权重转换。若您的模型尚未被 Megatron Bridge 支持，需要先在 Megatron Bridge 侧完成适配，详见其项目文档。

::: tip AI 辅助接入
本项目提供了 Codewiz skill `model-integration`（位于 `.codewiz/skills/model-integration/`），涵盖 Bridge / Raw / FSDP 三种后端的完整接入流程、权重转换器编写规范、TP 分片逻辑及常见陷阱，可在 Codewiz 中通过 `invoke skill model-integration` 调用以获得逐步指导。
:::

## 数据准备

Relax 支持加载 `.jsonl` 和 `.parquet` 格式文件。以 `.jsonl` 为例，文件的每一行都是一个 JSON 对象：

```json
{
  "prompt": [
    {
      "content": "<image><audio><video>What happened in the video?\nOptions:\nA. a sunny day\nB. It's Hailing\nC. a furious storm\nD. Flood",
      "role": "user"
    }
  ],
  "image_key": ["path to your image"],
  "audio_key": ["path to your audio"],
  "video_key": ["path to your video"],
  "label": "<answer>B</answer>"
}
```

对于多模态数据，每种模态应有对应占位符位于 content 部分，如上 `<image><audio><video>`，用于正确处理 message 格式。多模态数据支持传输本地文件路径、URL、二进制文件。

上例在训练脚本中对应的配置为：

```bash
--input-key prompt
--label-key label
--apply-chat-template
# 每一种多模态数据需明确给定配置才会读取
--multimodal-keys '{"image":"image_key","audio":"audio_key","video":"video_key"}'
```

我们提供了 OpenR1 和 AVQA 数据集的转换脚本，位于 `scripts/tools/`：

```bash
python scripts/tools/process_openr1.py \
  --input-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
  --output-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001-test.parquet

# --md-dir 指向 image 和 audio 文件目录所在路径，
# 用于将相对路径拼接为绝对路径，若不传则使用相对路径。
python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train.json \
  --output-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train_test.jsonl \
  --md-dir /root/AVQA-R1-6K/AVQA_R1/train
```

## 自定义 Reward 方法

在自己的 `.py` 文件中定义单样本 Reward
`reward_func(args, sample: Sample, **kwargs) -> int | float | dict`，或 batch/group Reward
`reward_func(args, samples: list[Sample], **kwargs) -> list[int | float | dict]`，然后在启动脚本中配置其导入路径。
具体使用可参考 [DeepEyes](../examples/deepeyes.md)。

```bash
--custom-rm-path examples.deepeyes.reward_deepeyes.reward_func
# 自定义 reward_func 允许返回 dict，但若如此您需要明确哪个 key 对应于实际的 reward 得分
--reward-key score
```

同步函数在进程隔离的 Ray `RewardWorker` Actor 中运行。使用 `async def` 声明的函数在调用方进程中直接
`await`，不使用 worker 池。`--reward-num-workers` 设置同步 worker 数量，`--reward-max-concurrency`
限制每个调用方进程内同时执行的 Reward 调用数。

batch/group 自定义 Reward 每次接收完整的样本列表，一次调用只占用一个并发名额。自定义函数按需加载并缓存，
直到显式触发 `custom_rm` reload。

同步自定义 Reward 的输入和返回值必须可由 Ray 序列化。

> **警告：外部 I/O 必须设置超时**
>
> 取消调用方或对应的 Ray task 只是 best-effort，**不能保证终止**已经在 `RewardWorker` 中执行的 Python
> 函数。HTTP、RPC、数据库调用等可能阻塞的外部 I/O 操作必须显式设置 timeout。否则，该 worker 可能一直被占用，
> 直到函数返回，后续分配到此 worker 的 Reward 调用也必须继续等待。

### 格式感知的 Reward 路由

内置 reward 统一注册在 `relax/engine/rewards/registry.py` 中，新增一个 reward 只需一行 `register_reward` 调用，无需修改任何分发分支：

```python
register_reward("my_reward", "my_pkg.my_module:my_reward_fn")  # def my_reward_fn(response, label) -> float
```

每个样本按以下优先级解析 reward 类型：样本 `metadata["rm_type"]` > `--rm-type` > label 推断（需开启）> fallback（需开启）。因此一个 batch 可以混合多种任务格式（如 math + multiple-choice），只需每条数据携带 `"metadata": {"rm_type": "..."}`（列名由 `--metadata-key` 指定，默认 `metadata`）。

两个可选参数控制降级样本的行为，默认值下与现有行为完全一致：

- `--rm-type-fallback`：设为 `zero` 时，未知/缺失类型的样本记 `0.0` 分（感知 reward-key）并告警；设为已注册的 reward 名则路由到该 reward；不设置保持现有报错行为。
- `--rm-type-infer`：无显式类型时按注册的保守 matcher 从 label 推断类型（严格数字答案 → `math`，`<answer>X</answer>` 单字母 → `multiple_choice`）。当 label 与显式类型冲突时记录告警，并以显式类型优先。

降级告警包含样本 index、异常值与原因；每种 `(原因, 值)` 组合只打印一次，但计数器精确累计。注意：`--custom-rm-path` 仍然完全绕过路由；返回 dict 的 reward（如 `dapo`）不能与标量 reward 混在同一 batch（`--reward-key` 是全局的）；driver 侧运行时调用 `register_reward` 注册的 sync reward 对已启动的 Ray worker 不可见——请像 `registry.py` 一样在 import 期注册，或使用 `--custom-rm-path`。

## 自定义 Generate 函数

对于多轮对话、工具调用、Agent 交互等场景，可自定义 `generate` 函数替换默认的单轮生成逻辑。函数签名如下：

```python
from relax.utils.types import Sample
# 必需签名
async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample: ...
# 可选：添加 evaluation 参数 — 框架在评估时自动传入 True
async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample: ...
```

函数返回前必须填充以下 `sample` 字段：`tokens`（完整 prompt+response token ID）、`response`（解码后字符串）、`response_length`、`loss_mask`（逐 token：`1`=参与训练，`0`=跳过）、`rollout_log_probs` 以及 `status`（`Sample.Status.COMPLETED` / `TRUNCATED` 等）。

**示例** — 简化自 [`examples/deepeyes/rollout.py`](../examples/deepeyes.md)（多轮工具调用 rollout）：

```python
from relax.engine.rollout.sglang_rollout import GenerateState

async def generate(args, sample: Sample, sampling_params) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env = build_env(sample=sample, args=args); env.reset()
    prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    sample.tokens, sample.loss_mask, sample.rollout_log_probs, response_tokens = list(prompt_ids), [], [], []
    for turn in range(args.max_turns):
        # 每轮各取一次 permit：post_generate 内部获取/释放，环境与工具执行期间不占用
        output = await state.post_generate(url, {"input_ids": sample.tokens, "sampling_params": sampling_params, "return_logprob": True})
        new_tokens = [t[1] for t in output["meta_info"]["output_token_logprobs"]]
        new_probs = [t[0] for t in output["meta_info"]["output_token_logprobs"]]
        sample.tokens.extend(new_tokens); response_tokens.extend(new_tokens)                 # 模型输出
        sample.loss_mask.extend([1] * len(new_tokens)); sample.rollout_log_probs.extend(new_probs)
        observation, done, info = env.step(output["text"])
        if done: break
        obs_ids = state.tokenizer.encode(observation, add_special_tokens=False)
        sample.tokens.extend(obs_ids); response_tokens.extend(obs_ids)                       # 环境观测
        sample.loss_mask.extend([0] * len(obs_ids)); sample.rollout_log_probs.extend([0.0] * len(obs_ids))
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.status = Sample.Status.COMPLETED
    return sample


# 声明本函数按“单次请求”粒度自行管理 permit（不声明则退回会话级锁，行为同旧版）
generate.manages_inference_permit = True
```

通过启动脚本指定（`--custom-generate-function-path examples.deepeyes.rollout.generate`），或在评估数据集配置中通过 `custom_generate_function_path` 按数据集设置。

### 多轮 Rollout 的请求级并发调度

默认情况下，`generate_and_rm` 会为整个自定义 `generate` 调用持有一把会话级并发锁（`GenerateState.semaphore`）——多轮 rollout 在环境/工具执行期间也一直占用名额，降低推理引擎利用率。

要把并发控制细化到「单次模型请求」，在自定义函数上声明 opt-in 标志，并用 `state.post_generate` 发起每轮请求：

- `generate.manages_inference_permit = True`：声明后本函数**不再**被会话级锁包裹，改由函数自身按请求获取 permit；
- `await state.post_generate(url, payload)`：获取一个 permit → 发请求 → 返回后立即释放。一个 permit == 一次在飞请求（含内部最多 6 次重试）；环境交互、观测编码等应放在 `post_generate` 之外，不占用 permit；
- **abort**：若获取 permit 时 rollout 已被 abort，`post_generate` 抛出 `GenerationAborted`。可捕获它以写入精细的续跑元数据；**不捕获也不会导致整步崩溃**——框架会兜底把该样本标记为 `ABORTED`。

**使用契约**：只有声明了 `manages_inference_permit = True` 的函数才能调用 `state.post_generate` / `state.inference_permit()`。未声明的函数仍在会话级锁内运行，若再获取 permit 会在**同一把不可重入信号量**上嵌套获取——并发上限为 1 时直接死锁（框架已加运行时护栏，检测到误用会抛出清晰的 `RuntimeError` 而非挂死）。

**注意**：解除会话级锁后，同时“活跃”的会话数不再受该信号量限制（仅在飞请求受限）；如需限制并发环境/工具的资源占用，请在自定义函数内自行控制。

## 训练脚本与关键参数概览

完整的参数可参照 [配置说明](./configuration.md)。

完成准备工作后即可运行训练脚本，以 Qwen3 VL 4B 为例：

```bash
cd /root/Relax && \
export MODEL_CONFIG_DIR=$(pwd)/scripts/models && \
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

### 模型配置参数

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"
```

此部分命令为 Megatron 提供所需的超参数。由于 Megatron 无法直接从检查点（checkpoint）中读取模型配置，因此需要手动指定。我们在 `scripts/models/` 目录下提供了一些常用模型的配置示例。若您需要新增模型，请在该目录下添加对应的配置文件，并在任务启动脚本中 source 对应模型的配置文件。

### 检查点与路径参数

```bash
CKPT_ARGS=(
  # 用于加载 tokenizer 等其他信息；实际上不会使用 HF 路径中的模型权重参数
  --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
  # 参考模型的检查点
  # 当 --load 未设置时，将使用此作为训练的初始检查点
  --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
  # 启用 megatron bridge 自动权重转换
  --megatron-to-hf-mode bridge
  # Actor 模型的加载路径。若为空或不存在有效的 checkpoint，则从 --ref-load 加载
  # 当需要使用断点续训时，请使用此选项指向 checkpoint 路径
  --load /path/checkpoint/
  # 训练过程中模型的保存路径
  --save /path/checkpoint/
  # 模型保存间隔（步数）
  --save-interval 20
)
```

::: tip 路径变量约定
脚本顶部统一定义了三个路径变量：

- `MODEL_DIR` —— HF 权重 / `--ref-load` 等模型相关路径
- `DATA_DIR` —— `PROMPT_SET`、`--eval-prompt-data` 等数据集路径
- `EXP_DIR` —— `--load` / `--save` 等训练输出路径

三者均可单独通过环境变量覆盖：`MODEL_DIR` / `DATA_DIR` 未设置时回落到 `EXP_DIR`，因此只设置 `export EXP_DIR=/root` 即可同时控制三个路径。
:::

### 数据生成与训练参数

```bash
# 数据集路径
--prompt-data ${PROMPT_SET}
# 定义每轮采样的 Prompt 数量
--rollout-batch-size 32
# 定义每个 Prompt 生成的回复数量
# 与 --rollout-batch-size 相乘决定了单轮采样产生的总样本数
--n-samples-per-prompt 8
# 定义执行一次参数更新（optimizer.step）所需的样本量
--global-batch-size 256
# 控制整个"采样→训练"循环的总执行轮次
--num-rollout ${NUM_ROLLOUT}
```

### Message 处理参数

```bash
# 数据集输入标识 key
--input-key prompt
# 数据集 label 标识 key
--label-key label
# 若 Prompt 的 input_key 是 OpenAI message 格式，则应用 Chat Template
--apply-chat-template
# 所采用的 reward 计算方法，此选项仅支持内置 reward 方法
# 若需要自定义 reward，请使用 --custom-rm-path
--rm-type openr1mm
# 多模态数据提取标识
--multimodal-keys '{"image":"image"}'
# 自定义 SYSTEM_PROMPT 添加；会在 prompt 头部插入一条新内容
--system-prompt ${SYSTEM_PROMPT}
```

### 评估参数

您可添加 eval 数据集用于评估，请注意每次调用 eval 时都会把整个数据集过一遍，建议 eval 数据集不要太大。

```bash
VAL_ARGS=(
  # 评估间隔（Rollout 数）
  --eval-interval 5
  # 评估用的 Prompt 数据集
  --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
  # 每个评估 Prompt 的采样数量
  --n-samples-per-eval-prompt 16
  # 评估时最大响应长度
  --eval-max-response-len 16384
  # 评估时采样参数
  --eval-top-p 0.7
)
```

### 监控与 Dump

```bash
# 启用 ClearML
--use-clearml
# 启用 TensorBoard
--use-tensorboard
# 启用集中化指标收集和报告服务
--use-metrics-service
# TensorBoard/ClearML 存储路径
--tb-project-name ${PROJECT_NAME}
# TensorBoard/ClearML 存储名称
--tb-experiment-name name
# 下载每步 rollout 和训练等细节到指定目录
--dump-details /path
```

### 并行与性能调优参数

```bash
# 训练并行
--tensor-model-parallel-size 2
--sequence-parallel
--pipeline-model-parallel-size 1
--expert-model-parallel-size 8
# 重计算
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
# CPU offload optimizer
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
# 推理
--rollout-num-gpus-per-engine 2 # sglang tp
--sglang-mem-fraction-static 0.8
# 启用动态批处理。此时会忽略 --micro-batch-size
--use-dynamic-batch-size
# 每张 GPU 处理的最大 Token 数。
# 启用动态批处理（use_dynamic_batch_size）后，系统会智能地将长短不一的样本打包，使每个 micro-batch 的总 Token 数接近此限制，从而提升训练效率。
# 如果单个样本长度超过该值，它将独立形成一个 batch。
--max-tokens-per-gpu 9216
```

### Ray 启动命令

```bash
ray job submit --address="http://127.0.0.1:8265" \
  -- python3 relax/entrypoints/train.py \
  # [1, 8] 分别表示副本数和总占用的卡数，副本数给 1 就行
  # 通过 _derive_cluster_args_from_resource 划分资源
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --max-staleness 0 \
  --num-data-storage-units 1 \
  --colocate \
  # 其他参数在下方展开
```

### 多机任务启动

Relax 提供两种多机启动方式：**SPMD 多机模式**（自建 Ray 集群）和 **Ray Job 模式**（已有 Ray 集群）。

#### 方式一：SPMD 多机模式

适用于在裸机或容器环境中从零启动 Ray 集群并运行训练。脚本会自动区分 Head 节点和 Worker 节点，组建集群后在 Head 节点上提交训练任务。

**前置环境变量**（需在每台机器上设置）：

| 变量 | 说明 | 示例 |
|---|---|---|
| `MASTER_ADDR` | Head 节点的 hostname | `node-0` |
| `POD_NAME` | 当前节点的 hostname | `node-0` / `node-1` |
| `HOST_IP` | 当前节点的 IP 地址 | `<node-ip>` |
| `WORLD_SIZE` | 节点总数（默认 2） | `2` |
| `NUM_GPUS` | 每节点 GPU 数（默认 8） | `8` |

**在每台机器上执行相同的命令**：

```bash
bash scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

脚本根据 `MASTER_ADDR == POD_NAME` 自动判断角色：
- **Head 节点**：启动 Ray Head → 等待所有 Worker 加入 → 执行训练脚本
- **Worker 节点**：加入 Ray 集群 → 阻塞等待训练结束

#### 方式二：Ray Job 模式

适用于 Ray 集群已经由外部管理平台（如 KubeRay）启动并运行的场景。脚本不会启动或停止 Ray，仅清理残留进程后直接执行训练。

**前置条件**：
- Ray 集群已在运行，且当前节点可通过 `ray status` 连接
- 脚本会自动从 Ray 集群中获取 `MASTER_ADDR`

```bash
bash scripts/entrypoint/ray-job.sh scripts/training/multimodal/run-qwen35-9B-8xgpu-async.sh
```

#### 两种模式对比

| | SPMD 多机 | Ray Job |
|---|---|---|
| Ray 集群管理 | 脚本自建（Head + Worker） | 外部管理（KubeRay 等） |
| 需要每台机器执行 | 是 | 否（仅提交节点） |
| 适用场景 | 裸机 / 容器 SPMD 调度 | 已有 Ray 集群 |
| 入口脚本 | `scripts/entrypoint/spmd-multinode.sh` | `scripts/entrypoint/ray-job.sh` |

## 下一步

### 自定义实验

1. **修改启动脚本**：编辑 `examples/` 或 `scripts/` 中的 shell 脚本
2. **更换模型**：更新 `--hf-checkpoint` 参数指向您的模型
3. **调整训练**：修改优化器、学习率和批次大小参数
4. **自定义奖励**：实现自定义奖励函数（参见 DeepEyes 示例）

### 探索示例

- [DeepEyes](../examples/deepeyes.md) — 多模态视觉-语言强化学习
- [在线策略蒸馏](../examples/on-policy-distillation.md) — 知识蒸馏

### 学习核心概念

- [架构设计](./architecture.md) — 理解系统设计
- [数据集设计](./dataset-design.md) — 了解数据加载
- [分布式检查点](./distributed-checkpoint.md) — 检查点管理

## 故障排除

### 常见问题

#### Ray 连接失败

```bash
# 检查 Ray 状态
ray status

# 重启 Ray
ray stop
ray start --head
```

#### CUDA 内存不足

在启动脚本中减少批次大小：

```bash
# 在 .sh 文件中修改：
--global-batch-size 128  # 从 256 减少
--rollout-batch-size 16  # 从 32 减少
```

#### 服务部署失败

检查服务日志：

```bash
# 查看 Ray 日志
tail -f /tmp/ray/session_latest/logs/serve/*.log
```

## 获取帮助

- [GitHub Issues](https://github.com/redai-studio/Relax/issues)
- [Discussions](https://github.com/redai-studio/Relax/discussions)
- [介绍](../guide/introduction.md)
