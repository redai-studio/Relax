# DeepEyes 示例

DeepEyes 示例演示了使用 Relax 进行 **Agent 式多轮视觉语言强化学习**。模型学习通过工具调用（缩放、旋转）与图像交互，在回答视觉问题前先观察感兴趣的区域，使用 GRPO 算法训练。

## 概述

[DeepEyes](https://huggingface.co/papers/2505.14362) 是一个多轮交互式视觉问答环境，模型可以调用图像操作工具（`image_zoom_in_tool`、`image_rotate_tool`）来检查感兴趣的区域，然后产生最终的 `<answer>...</answer>`。这是 Relax **Agentic RL** 能力的典型示例：

- **多轮采样与 loss mask** — 模型输出（mask=1）与环境观测（mask=0）分离，只有模型动作参与训练。
- **工具使用环境** — `DeepeyesEnv` 解析 `<tool_call>` 块，执行图像变换，返回更新后的图像作为 `<tool_response>`。
- **VLM 多模态上下文累积** — 每轮工具交互产生的图像被逐步合并到跨轮的视觉上下文中。
- **Judge 奖励** — LLM judge 对答案准确性打分，结合格式和工具使用奖励。

## 数据准备

DeepEyes 使用 HuggingFace 上的 [ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k) 数据集。数据集包含视觉问答样本，图像以 HF Image 格式内嵌存储。

### 下载数据集

```bash
# 使用 huggingface-cli
hf download --repo-type dataset ChenShawn/DeepEyes-Datasets-47k \
  --local-dir /root/deepeyes-v1

# 或使用 Python
from datasets import load_dataset
ds = load_dataset("ChenShawn/DeepEyes-Datasets-47k")
ds.save_to_disk("/root/deepeyes-v1")
```

下载后会产生 parquet 文件。训练使用 `data_0.1.2_visual_toolbox_v2.parquet`（22,362 个样本，来自 V*-Bench 数据源）。

### 数据集格式

每个样本包含以下字段：

| 列名 | 类型 | 描述 |
|------|------|------|
| `prompt` | list[dict] | 聊天消息（system + user），user 内容中包含 `<image>` 占位符 |
| `images` | list[dict] | HF Image 格式：`{"bytes": b"...", "path": "..."}` |
| `reward_model` | dict | `{"ground_truth": "...", "style": "model"}` |
| `extra_info` | dict | `{"answer": "...", "question": "...", "index": "...", "split": "train"}` |
| `data_source` | str | 数据来源标识（如 `"vstar"`） |
| `ability` | str | 能力分类 |
| `env_name` | str | 环境名称 |

HF Image dict 格式（`{"bytes": ...}`）被 Relax 的图像加载管线原生支持，无需转换。

### 下载模型

```bash
hf download Qwen/Qwen3-VL-30B-A3B-Thinking \
  --local-dir /root/Qwen3-VL-30B-A3B-Thinking
```

## 快速开始

### 30B-A3B 模型（8 GPU，MoE）

完整配置需要 judge 模型进行奖励评分：

```bash
export MODEL_DIR=/root
export DATA_DIR=/root
export SAVE_DIR=/root/save

# 设置 judge 模型 API（完整奖励评分所需）
# 或执行 sglang_judge_service.sh 实现本地部署
export DEEPEYES_JUDGE_API_KEY=your-api-key
export DEEPEYES_JUDGE_BASE_URL=http://your-judge-endpoint/v1

cd /root/Relax
bash examples/deepeyes/run_deepeyes.sh
```

在单机 8×H800（80GB）上运行 `run_deepeyes.sh` 时的 GPU 使用率监控（平均约 66%）：

![单机 8×H800（80GB）运行 run_deepeyes.sh 的 GPU 使用率](/deepeyes-h800.png)

### 远程集群（通过 Ray Job）

```bash
WORKING_DIR="./" RAY_ADDRESS=<RAY_HEAD_IP>:6379 \
  MODEL_DIR=/root DATA_DIR=/root SAVE_DIR=/root/save \
  bash -x scripts/entrypoint/ray-job.sh examples/deepeyes/run_deepeyes.sh
```

## 架构

### 文件结构

```
examples/deepeyes/
├── run_deepeyes.sh            # 启动脚本（Qwen3-VL-30B-A3B，完整配置）
├── deepeyes_config.yaml       # 任务配置（max_turns、环境路径）
├── rollout.py                 # 多轮 rollout 逻辑
├── env_deepeyes.py            # DeepEyes 工具使用环境
├── base_env.py                # BaseInteractionEnv 接口
├── reward_deepeyes.py         # 基于 Judge 的奖励函数
└── sglang_judge_service.sh    # Judge 模型的 SGLang 服务
```

### 多轮 Rollout

自定义 rollout 函数（`rollout.py:generate`）实现多轮交互循环：

1. **初始化**：加载环境，编码初始图像，准备采样参数
2. **推理步骤**：将 token + 图像发送到 SGLang 引擎，获取模型响应
3. **环境步骤**：从响应中解析工具调用，执行图像变换，返回观测
4. **追加与掩码**：模型输出 token 设置 `loss_mask=1`，观测 token 设置 `loss_mask=0`
5. **重复**直到出现 `<answer>` 标签、达到最大轮数或 token 预算耗尽

每轮的状态（图像数据、多模态输入）被累积以实现正确的 VLM 上下文传递。

### 环境

DeepEyes 环境定义在 [`env_deepeyes.py`](../../../examples/deepeyes/env_deepeyes.py) 中：

```python
class DeepeyesEnv(BaseInteractionEnv):
    """Environment for Deepeyes with zoom in and rotate tools."""

    MIN_DIMENSION = 28

    def __init__(self, *, max_turns: int | None = None, image=None):
        self.max_turns = max_turns
        self.turn = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.current_image = image
        self.origin_image = image

    def reset(self):
        self.turn = 0
        self.tool_calls.clear()
        observation: dict[str, Any] = {}
        reset_info = {"has_image": self.current_image is not None}
        return observation, reset_info

    def step(self, response_text: str):
        """Process agent response and return observation, done flag, and info."""
        self.turn += 1
        # Check if answer is provided
        if ANSWER_RE.search(response_text):
            return self._build_obs_text(text="Answer received."), True, {"final_answer": True}

        # Extract and execute tool call
        tool_call = self._extract_tool_call(response_text)
        if not tool_call:
            obs = self._build_obs_text(text="No tool call detected; ending the episode.")
            return obs, True, {"tool_executed": False}

        obs, done, info = self._apply_tool(tool_call)
        # Check if max turns reached
        if self.max_turns is not None and self.turn >= self.max_turns:
            done = True
        return obs, done, info
```

核心行为：

- **`reset()`** — 初始化轮次计数器和工具调用历史
- **`step(response_text)`** — 从模型输出中解析 `<tool_call>` / `<answer>`：
  - 如果找到 `<answer>` → 回合结束
  - 如果 `<tool_call>` 包含 `image_zoom_in_tool` → 裁剪图像到边界框
  - 如果 `<tool_call>` 包含 `image_rotate_tool` → 按角度旋转图像
  - 如果没有识别到标签 → 回合结束（未检测到工具）
- **`_build_obs_text()`** — 构建包含文本和可选图像的观测字典，用于 `<tool_response>`

环境通过 `build_env(sample, args)` 实例化，从 `sample.multimodal_inputs` 中提取初始图像。

### 奖励函数

奖励函数定义在 [`reward_deepeyes.py`](../../../examples/deepeyes/reward_deepeyes.py) 中：

```python
def compute_score(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    # 1. 格式检查：验证 <think>、</think>、<answer>、</answer> 标签
    # 2. 如果格式错误 → acc_reward=0，跳过 judge
    # 3. 如果格式正确 → 通过 OpenAI 兼容 API 调用 LLM judge
    # 4. 综合各信号：
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    return {"score": final_score, "acc": acc_reward, "format": format_reward, "tool": tool_reward, ...}


async def reward_func(args, sample: Sample, **kwargs):
    question = sample.metadata.get("question")
    ground_truth = sample.metadata.get("answer")
    return await asyncio.to_thread(compute_score, sample.response, ground_truth, {"question": question})
```

奖励综合三个信号：

```
final_score = 0.8 × acc_reward + 0.2 × format_reward + 1.2 × tool_reward
```

| 组成部分 | 取值 | 条件 |
|---------|------|------|
| `acc_reward` | 0.0 或 1.0 | LLM judge 将提取的 `<answer>` 与 ground truth 比较 |
| `format_reward` | -1.0 或 0.0 | 如果 `<think>`/`</think>`/`<answer>`/`</answer>` 标签格式错误则为 -1.0 |
| `tool_reward` | 0.0 或 1.0 | 如果模型使用了工具且答案正确则为 1.0 |

**没有 judge 模型时**：如果 `DEEPEYES_JUDGE_API_KEY` 未配置（或指向不可达的端点），所有格式错误的样本获得 `acc_reward=0`，格式正确的样本会尝试调用 judge（调用失败后默认 `acc_reward=0`）。训练仍会继续，但奖励信号退化——仅格式正确性和工具使用会被奖励。

## 配置

### 任务配置（`deepeyes_config.yaml`）

```yaml
max_turns: 10                                          # 每个回合的最大交互轮数
rollout_interaction_env_path: examples.deepeyes.env_deepeyes  # 环境模块路径
```

### 关键启动脚本参数

| 参数 | 4B 脚本 | 30B 脚本 | 描述 |
|------|---------|----------|------|
| 模型 | `qwen3-vl-4B.sh` | `qwen3-vl-30B-A3B.sh` | 模型架构配置 |
| `--tensor-model-parallel-size` | 4 | 4 | 张量并行度 |
| `--expert-model-parallel-size` | 1 | 8 | 专家并行度（MoE） |
| `--rollout-batch-size` | 16 | 32 | Rollout 批大小 |
| `--n-samples-per-prompt` | 4 | 8 | GRPO 每个 prompt 的采样数 |
| `--global-batch-size` | 128 | 256 | 训练批大小 |
| `--rollout-max-response-len` | 1024 | 2048 | 每轮最大响应 token 数 |
| `--max-tokens-per-gpu` | 8192 | 8192 | 动态 batching 每 GPU 最大 token 数 |
| `--colocate` | ✓ | ✓ | Actor 和 Rollout 共享 GPU |
| `--max-staleness` | 0 | 0 | 严格 on-policy 训练 |

### Judge 模型配置

奖励函数使用 OpenAI 兼容的 API 进行 LLM-as-judge 评分：

| 环境变量 | 默认值 | 描述 |
|---------|--------|------|
| `DEEPEYES_JUDGE_API_KEY` | （必需） | Judge 端点的 API 密钥 |
| `DEEPEYES_JUDGE_BASE_URL` | （可选） | Judge API 的 Base URL |
| `DEEPEYES_JUDGE_MODELS` | `gpt-4o` | 逗号分隔的 judge 模型名称列表 |
| `DEEPEYES_JUDGE_TIMEOUT` | `120` | Judge API 调用超时（秒） |

可以使用本地 SGLang 服务器作为 judge：

```bash
# 启动 judge 模型服务
bash examples/deepeyes/sglang_judge_service.sh
```

## 故障排除

### 内存不足

减小启动脚本中的批大小：

```bash
--global-batch-size 64       # 从 128 减小
--rollout-batch-size 8       # 从 16 减小
--max-tokens-per-gpu 4096    # 从 8192 减小
```

### Judge API 错误

如果 judge 模型不可达，奖励函数会重试 3 次然后将 `acc_reward` 设为 0。训练会继续，但只有格式奖励。请检查：

1. `DEEPEYES_JUDGE_API_KEY` 和 `DEEPEYES_JUDGE_BASE_URL` 设置正确
2. Judge 端点可从所有 Ray worker 节点访问
3. `DEEPEYES_JUDGE_MODELS` 中的模型名称与部署的模型匹配

### 响应中没有工具调用

如果模型始终不产生 `<tool_call>` 标签，回合会立即终止。这在训练早期是正常的——模型通过 RL 学习工具使用。请确保：

1. 数据中的 system prompt 包含工具使用说明
2. `deepeyes_config.yaml` 中 `max_turns > 1`
3. Rollout 温度足够高以支持探索（默认：1.0）
