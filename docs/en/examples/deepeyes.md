# DeepEyes Example

The DeepEyes example demonstrates **agentic multi-turn vision-language RL** using Relax. The model learns to interact with images through tool calls (zoom, rotate) before answering visual questions, trained with GRPO.

## Overview

[DeepEyes](https://huggingface.co/papers/2505.14362) is a multi-turn interactive visual QA environment where the model can invoke image manipulation tools (`image_zoom_in_tool`, `image_rotate_tool`) to inspect regions of interest before producing a final `<answer>...</answer>`. This is a canonical example of Relax's **Agentic RL** capabilities:

- **Multi-turn sampling with loss masking** — model outputs (mask=1) are separated from environment observations (mask=0) so only model actions participate in training.
- **Tool-use environment** — the `DeepeyesEnv` parses `<tool_call>` blocks, applies image transformations, and returns updated images as `<tool_response>`.
- **VLM multimodal context carry-over** — images from each tool interaction are incrementally merged into the visual context across turns.
- **Judge-based reward** — an LLM judge scores answer accuracy, combined with format and tool-use bonuses.

## Data Preparation

DeepEyes uses the [ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k) dataset from HuggingFace. The dataset contains visual QA samples with images stored inline (HF Image format).

### Download the Dataset

```bash
# 使用 huggingface-cli
hf download --repo-type dataset ChenShawn/DeepEyes-Datasets-47k \
  --local-dir /root/deepeyes-v1

# 或使用 Python
from datasets import load_dataset
ds = load_dataset("ChenShawn/DeepEyes-Datasets-47k")
ds.save_to_disk("/root/deepeyes-v1")
```

The download produces parquet files. For training, use `data_0.1.2_visual_toolbox_v2.parquet` (22,362 samples from the V*-Bench data source).

### Dataset Format

Each sample contains:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | list[dict] | Chat messages (system + user) with `<image>` placeholders |
| `images` | list[dict] | HF Image format: `{"bytes": b"...", "path": "..."}` |
| `reward_model` | dict | `{"ground_truth": "...", "style": "model"}` |
| `extra_info` | dict | `{"answer": "...", "question": "...", "index": "...", "split": "train"}` |
| `data_source` | str | Dataset source identifier (e.g., `"vstar"`) |
| `ability` | str | Ability category |
| `env_name` | str | Environment name |

The HF Image dict format (`{"bytes": ...}`) is natively supported by Relax's image loading pipeline — no conversion needed.

### Download the Model

```bash
hf download Qwen/Qwen3-VL-30B-A3B-Thinking \
  --local-dir /root/Qwen3-VL-30B-A3B-Thinking
```

## Quick Start

### 30B-A3B Model (8 GPUs, MoE)

The full-scale configuration requires a judge model for reward scoring:

```bash
export MODEL_DIR=/root
export DATA_DIR=/root
export SAVE_DIR=/root/save

# Set judge model API (required for full reward scoring)
# Or run sglang_judge_service.sh for local deployment
export DEEPEYES_JUDGE_API_KEY=your-api-key
export DEEPEYES_JUDGE_BASE_URL=http://your-judge-endpoint/v1

cd /root/Relax
bash examples/deepeyes/run_deepeyes.sh
```

GPU utilization while running `run_deepeyes.sh` on a single node with 8×H800 (80GB) (~66% on average):

![GPU utilization running run_deepeyes.sh on a single node with 8×H800 (80GB)](/deepeyes-h800.png)

### Remote Cluster (via Ray Job)

```bash
WORKING_DIR="./" RAY_ADDRESS=<RAY_HEAD_IP>:6379 \
  MODEL_DIR=/root DATA_DIR=/root SAVE_DIR=/root/save \
  bash -x scripts/entrypoint/ray-job.sh examples/deepeyes/run_deepeyes.sh
```

## Architecture

### File Structure

```
examples/deepeyes/
├── run_deepeyes.sh            # Launch script (Qwen3-VL-30B-A3B, full config)
├── deepeyes_config.yaml       # Task config (max_turns, env path)
├── rollout.py                 # Multi-turn rollout logic
├── env_deepeyes.py            # DeepEyes tool-use environment
├── base_env.py                # BaseInteractionEnv interface
├── reward_deepeyes.py         # Judge-based reward function
└── sglang_judge_service.sh    # SGLang server for judge model
```

### Multi-Turn Rollout

The custom rollout function (`rollout.py:generate`) implements the multi-turn interaction loop:

1. **Initialize**: Load environment, encode initial image, prepare sampling params
2. **Inference step**: Send tokens + images to SGLang engine, get model response
3. **Environment step**: Parse tool calls from response, apply image transformation, return observation
4. **Append & mask**: Model output tokens get `loss_mask=1`, observation tokens get `loss_mask=0`
5. **Repeat** until `<answer>` tag, max turns, or token budget exhausted

Each turn's state (image data, multimodal inputs) is accumulated for proper VLM context carry-over.

### Environment

The DeepEyes environment is defined in [`env_deepeyes.py`](../../../examples/deepeyes/env_deepeyes.py):

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

Key behaviors:

- **`reset()`** — Initialize turn counter and tool call history
- **`step(response_text)`** — Parse `<tool_call>` / `<answer>` from model output:
  - If `<answer>` found → episode done
  - If `<tool_call>` with `image_zoom_in_tool` → crop image to bounding box
  - If `<tool_call>` with `image_rotate_tool` → rotate image by angle
  - If no recognizable tag → episode done (no tool detected)
- **`_build_obs_text()`** — Build observation dict with text and optional image for `<tool_response>`

The environment is instantiated via `build_env(sample, args)`, which extracts the initial image from `sample.multimodal_inputs`.

### Reward Function

The reward function is defined in [`reward_deepeyes.py`](../../../examples/deepeyes/reward_deepeyes.py):

```python
def compute_score(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    # 1. Format check: validate <think>, </think>, <answer>, </answer> tags
    # 2. If format error → acc_reward=0, skip judge
    # 3. If format OK → call LLM judge via OpenAI-compatible API
    # 4. Combine signals:
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    return {"score": final_score, "acc": acc_reward, "format": format_reward, "tool": tool_reward, ...}


async def reward_func(args, sample: Sample, **kwargs):
    question = sample.metadata.get("question")
    ground_truth = sample.metadata.get("answer")
    return await asyncio.to_thread(compute_score, sample.response, ground_truth, {"question": question})
```

The reward combines three signals:

```
final_score = 0.8 × acc_reward + 0.2 × format_reward + 1.2 × tool_reward
```

| Component | Value | Condition |
|-----------|-------|-----------|
| `acc_reward` | 0.0 or 1.0 | LLM judge compares extracted `<answer>` against ground truth |
| `format_reward` | -1.0 or 0.0 | -1.0 if `<think>`/`</think>`/`<answer>`/`</answer>` tags are malformed |
| `tool_reward` | 0.0 or 1.0 | 1.0 if model used tools AND answer was correct |

**Without a judge model**: If `DEEPEYES_JUDGE_API_KEY` is not configured (or points to an unreachable endpoint), all format-incorrect samples receive `acc_reward=0`, and format-correct samples will attempt the judge call (which will fail and default to `acc_reward=0`). The training will still run but with degraded reward signal — only format correctness and tool usage will be rewarded.

## Configuration

### Task Config (`deepeyes_config.yaml`)

```yaml
max_turns: 10                                          # Max interaction turns per episode
rollout_interaction_env_path: examples.deepeyes.env_deepeyes  # Environment module path
```

### Key Launch Script Parameters

| Parameter | 4B Script | 30B Script | Description |
|-----------|-----------|------------|-------------|
| Model | `qwen3-vl-4B.sh` | `qwen3-vl-30B-A3B.sh` | Model architecture config |
| `--tensor-model-parallel-size` | 4 | 4 | Tensor parallelism |
| `--expert-model-parallel-size` | 1 | 8 | Expert parallelism (MoE) |
| `--rollout-batch-size` | 16 | 32 | Rollout batch size |
| `--n-samples-per-prompt` | 4 | 8 | Samples per prompt for GRPO |
| `--global-batch-size` | 128 | 256 | Training batch size |
| `--rollout-max-response-len` | 1024 | 2048 | Max response tokens per turn |
| `--max-tokens-per-gpu` | 8192 | 8192 | Max tokens per GPU for dynamic batching |
| `--colocate` | ✓ | ✓ | Actor and Rollout share GPUs |
| `--max-staleness` | 0 | 0 | Strict on-policy training |

### Judge Model Configuration

The reward function uses an OpenAI-compatible API for LLM-as-judge scoring:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DEEPEYES_JUDGE_API_KEY` | (required) | API key for the judge endpoint |
| `DEEPEYES_JUDGE_BASE_URL` | (optional) | Base URL for the judge API |
| `DEEPEYES_JUDGE_MODELS` | `gpt-4o` | Comma-separated list of judge model names |
| `DEEPEYES_JUDGE_TIMEOUT` | `120` | Timeout in seconds for judge API calls |

You can use a local SGLang server as the judge:

```bash
# Start judge model server
bash examples/deepeyes/sglang_judge_service.sh
```

## Troubleshooting

### Out of Memory

Reduce batch sizes in the launch script:

```bash
--global-batch-size 64       # Reduce from 128
--rollout-batch-size 8       # Reduce from 16
--max-tokens-per-gpu 4096    # Reduce from 8192
```

### Judge API Errors

If the judge model is unreachable, the reward function will retry 3 times and then assign `acc_reward=0`. Training continues but with format-only rewards. Check:

1. `DEEPEYES_JUDGE_API_KEY` and `DEEPEYES_JUDGE_BASE_URL` are correctly set
2. The judge endpoint is accessible from all Ray worker nodes
3. The judge model name in `DEEPEYES_JUDGE_MODELS` matches the deployed model

### No Tool Calls in Responses

If the model never produces `<tool_call>` tags, episodes terminate immediately. This is expected early in training — the model learns tool use through RL. Ensure:

1. The system prompt in your data includes tool-use instructions
2. `max_turns > 1` in `deepeyes_config.yaml`
3. Rollout temperature is high enough for exploration (default: 1.0)
