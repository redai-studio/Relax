# Agentic Rollout

Agentic rollout 将已有 agent app (harness) 接入 Relax 训练。Relax 为每个 Session 启动并管理一个 agent process，
由该 process 运行原有 harness。Relax 记录已提交的 conversation，并把选中的 context 转换为训练 sample。

**当您已经有一个通过 OpenAI-compatible Chat Completions API 调用模型的 agent application (harness) 时，
Agentic rollout 尤其适合将其接入 Relax。使用 OpenAI Responses 或 Anthropic Messages 的 agent 也可以接入。
Agent 既可以 standalone 运行，也可以由集中式平台统一执行。**

::: tip 推荐工作流
评估和接入 agent app (harness)、检查启动配置或开展实验时，建议使用仓库 `skills/agentic-rollout/` 下的
`agentic-rollout` skill。该 skill 会检查当前 checkout，并按阶段检查 context topology、parser、export 与 advantage、
timeout、并发容量和 runtime 证据。实验仍需用户明确授权；本文继续说明 model API 请求与响应格式、API 和 export
契约。

手动阅读时：

- 从已有 agent 开始：先读[准备 Agent](#准备-agent)，再读[用户接入](#用户接入)。
- 多 Agent 训练、导出多个上下文或为不同上下文分配 advantage：
  [选择训练 Context 与分数](#选择训练-context-与分数)。
- 调整并发或跨 step 执行：[配置 Runtime 行为](#配置-runtime-行为)。
- 理解 SessionForest 与调度原理：[理解 Agentic Rollout 原理](#理解-agentic-rollout-原理)。
:::

![Agent integration](/agentic/agent_app.svg)

## 核心能力

1. **用已有 agent 做 Agentic RL**
   通过 Chat Completions、Responses 或 Messages endpoint 接入已有 agent app (harness)，并连接到 Relax 训练。

2. **Agent process warmup**
   在 rollout 执行前提前启动 agent process，隐藏进程启动、tool setup 和环境初始化耗时。

3. **Request-level partial rollout**
   无需修改 agent，即可跨 rollout step 中断并恢复 model generation。

## 准备 Agent

先让 agent 在 Relax 外独立运行，使用它原有的 task input 和 model endpoint。继续接入前，确认 agent 能够：

- 通过原有输入接口接收一个真实任务；
- 调用一个受支持的 model endpoint；
- 完成一次完整的 harness 运行，需要时包含多个 turn；
- 写出最终结果；
- 正常退出，不产生错误。

Task input、model endpoint、API credential 和 result output 都应支持配置。接入后 harness 的行为保持不变；
Relax 在其外围提供新的 input、endpoint 和 output boundary。

Relax 为每个 Session 启动一个进程作为 agent 的入口。这个进程可以直接运行 agent、启动子进程，或把任务提交到
其他机器或集中式平台。Agent 在哪里运行没有影响，只要请求能到达 `RELAX_BASE_URL`。这个进程持续运行到任务
结束，然后退出。

::: warning 远程集中式 Agent 平台
如果您的 agent 并非在本地直接运行，而是提交到远程平台集中运行，并且远程平台带有 agent 并发限制，则启动前
必须阅读[配置 Runtime 行为](#配置-runtime-行为)。
:::

## 用户接入

### Dataset 与 Session Input

Relax 把每个任务写入 `RELAX_INPUT_JSON` 指向的文件。文件可以包含 `messages`、`metadata`，或同时包含两者。
下面的例子直接提供可用的 messages：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": "Which city hosted the event?"}
  ],
  "metadata": {
    "task_id": "example-001"
  }
}
```

#### Text 与 Message Input

标准 dataset 路径把 `--input-key` 映射到 `messages`。String 会转换成一条 user message；message list 保持 OpenAI
message 结构。

#### Metadata-Only Task

`--metadata-key` 把 dataset object 映射到 `metadata`。Harness 可以在运行时读取 metadata，并现场组装 messages。

#### Multimodal Input

::: tip 继续使用 Relax 原有 Dataset 格式
图片数据继续按照 Relax 其他 multimodal training 的方式准备：在 prompt 或 message content 中放置 `<image>`
占位符，把 image path、URL 或 binary value 保存在独立 dataset field 中，再用 `--multimodal-keys` 建立映射。
Agentic rollout 会在 agent process 读取前，自动把该输入转换成 OpenAI `image_url` 格式。
:::

例如，dataset row 可以继续使用普通 Relax 格式：

```json
{
  "input": [{"role": "user", "content": "<image>Describe this image."}],
  "images": ["/path/to/image.png"]
}
```

把 image modality 映射到 dataset field：

```bash
--multimodal-keys '{"image":"images"}'
```

`images` 中的每张图片都需要一个对应的 `<image>` 占位符。标准 Relax data path 会把占位符与 image value 结合，
并生成内部 image item。Agentic rollout 随后增加 process-boundary 转换，将其变成 OpenAI `image_url`。Dataset
本身无需保存 `image_url` object。Placeholder 与 field mapping 路径仅用于 dataset 生成的初始输入。

| 阶段 | 图片处理 |
| --- | --- |
| 标准 Relax dataset 路径 | `--multimodal-keys` 把 dataset image 插入 prompt，并提取 model media input |
| Agentic Session Input | Process 启动前，Relax 把内部 image item 转换为 OpenAI `image_url` content |
| Agent model request | Relax 读取 `image_url`，为 SGLang 准备 backend media，并构造 processor-expanded training input |

在 process boundary，已有的 `data:image/...`、`http://` 或 `https://` URL 会保持原样。Local path、byte payload 或
内存中的 image 会被加载、转换成 RGB PNG，并编码为 data URI。

`messages` 中的图片使用 OpenAI `image_url` content 格式：

```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64-data>"}},
    {"type": "text", "text": "Describe this image."}
  ]
}
```

初始 dataset image 会自动转换成该格式。Agent 在后续 turn 中新增图片时，直接在 `messages` 中追加相同的
`image_url` content；后续 observation 不经过 dataset placeholder 或 `--multimodal-keys`。Tool observation 同样
可以包含 `image_url` item。完整的 multi-turn 实现可参考
`examples/deepeyes_agentic/app/env_deepeyes.py`；inference 与 training token 视图参见
[Token-in, Token-out](#token-in-token-out)。

### 最小 Agent Application

下面的 application 接收可直接使用的 messages，调用一次 model，并导出最终 conversation。基于 metadata 的
harness 可以先读取 metadata，再组装 `messages`。实际 agent 可以保留原有 tool loop，并用同一个 client 发起更多请求。

```python
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI


async def main() -> None:
    session_input = json.loads(Path(os.environ["RELAX_INPUT_JSON"]).read_text(encoding="utf-8"))
    messages = session_input["messages"]

    client = AsyncOpenAI(
        base_url=os.environ["RELAX_BASE_URL"],
        api_key=os.environ["RELAX_SESSION_ID"],
        timeout=9999,
    )
    response = await client.chat.completions.create(
        model="model",
        messages=messages,
    )
    messages.append(response.choices[0].message.model_dump())

    output = {
        "metadata": {"task_success": 1.0},
        "reward": 1.0,
    }
    Path(os.environ["RELAX_OUTPUT_JSON"]).write_text(
        json.dumps(output, ensure_ascii=False),
        encoding="utf-8",
    )


asyncio.run(main())
```

上面的 `timeout=9999` 是 agent 向 `RELAX_BASE_URL` 发起单次 model request 的 wall-clock timeout。Prelaunch、
partial rollout 的 abort/resume 或 fully async 执行可能会 hold 住同一个 request，因此 client timeout 需要覆盖
最长的等待时间。

使用 `model_dump()` 保存完整 assistant message。Reasoning content 与 tool call 才能参与后续 turn 和 SessionForest
matching。

### Model API

Relax 接收 Chat Completions request，以及下文列出的 Responses 和 Messages request 结构。三者使用相同的
generation 与 SessionForest core。每个 endpoint 都使用 `Authorization: Bearer <RELAX_SESSION_ID>` 鉴权。

| Interface | Endpoint | 完整 history 字段 | Turn limit |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `/v1/chat/completions` | `messages` | `max_completion_tokens`，或旧版 `max_tokens` |
| OpenAI Responses | `/v1/responses` | typed `input` Items | `max_output_tokens` |
| Anthropic Messages | `/v1/messages` | `system` 与 `messages` | 必填 `max_tokens` |

::: tip 不使用标准 client 时
标准 client 应原样接收 `RELAX_BASE_URL`，由 client 自动追加 resource path。自行发送 HTTP request 时，使用
`RELAX_BASE_URL.rstrip("/") + endpoint`，其中 endpoint 取自上表。拼接时需要保留已有 service path；部分 URL
helper 会把前导 `/` 解析到 origin 根路径，从而丢弃 `/agentic_api`。
:::

::: warning 当前 API 限制
Model request 必须通过 HTTP 发送。省略 `stream` 或设为 `false` 时返回 JSON；`stream=true` 使用 Buffered SSE，
并在完整 generation 结束后发送 model-content events。当前不支持 WebSocket 和 token-level streaming。每条
request 都必须重放所属 conversation branch 的完整 history。Responses request 必须包含完整 `input`；
Relax 不使用 `previous_response_id`。
:::

Relax 把三种 interface 的 request 规范化为 canonical `messages`、function `tools` 和 `chat_template_kwargs`，再用
该 state 完成 SessionForest matching 与 generation。API response 按当前 request 使用的 interface 返回。`model`
字段会回显到 response，但不选择 Relax inference backend。下表之外的 request field 不会改变 generation。

#### Chat Completions

训练配置提供 `temperature` 和 `top_p`。Request 中的这两个字段会被忽略。

| 字段 | 行为 |
| --- | --- |
| `messages` | 必需；一个 conversation branch 的完整 history |
| `tools` | 该 branch 使用的 tool definitions |
| `chat_template_kwargs` | 该 branch 使用的 template arguments |
| `max_completion_tokens` | 当前 turn 的最大 generated token 数 |
| `max_tokens` | `max_completion_tokens` 的旧版别名；同时设置时优先使用新字段 |
| `stop` | 当前 turn 的 stop string 或 list |
| `seed` | 当前 turn 的 sampling seed |
| `logprobs` | 在 response 中返回 generated-token logprobs |

::: warning Chat Completions message 契约
Message role 使用 `user`、`assistant`、`tool` 或 `system`。User、system 与 tool message 需要非空 content；tool
返回 `None` 或 `""` 时，使用稳定的非空表示。Assistant 包含 tool call 或 reasoning 时可以省略文本 content。Harness
发送 Chat `developer` message 时，需要在 harness 侧转换为 `system`。`add_generation_prompt`、`tokenize` 与
`tools` 由 Relax 管理，请勿在 request `chat_template_kwargs` 中设置。
:::

使用了 `tools` 或 `chat_template_kwargs` 的 request 必须持续传入这些字段。模型和 chat template 需要时，配置
`--agentic-reasoning-parser` 和 `--agentic-tool-call-parser`。

#### Responses

Relax 投影以下 typed `input` Items：

| Item | 投影结果 |
| --- | --- |
| Top-level `instructions` string | 放在 `input` 之前的 canonical system message |
| 带 `user`、`assistant`、`system` 或 `developer` role 的 `message` | Canonical message；`developer` 转换为 `system` |
| `input_text` / `output_text` | Message text |
| 带 `image_url` string 的 user `input_image` | Canonical `image_url` content |
| `summary` 或 `content` 中带可读文本的 `reasoning` | Assistant `reasoning_content` |
| `function_call` | 保持同一 `call_id` 的 assistant function tool call |
| 带 text output 的 `function_call_output` | 保持同一 `call_id` 的 tool message |

Responses function tool 使用扁平的 `type`、`name`、`parameters` 和可选 `description` 结构。输出 reasoning、
assistant text 与 function call 会变成相应 Responses Item。`function_call_output` 接受 string 或 `input_text`
block。因长度限制结束的结果使用 `status: "incomplete"`；其他成功结果使用 `status: "completed"`。

#### Anthropic Messages

Relax 投影以下 Messages block：

| Block | 投影结果 |
| --- | --- |
| `system` string 或 text block | Canonical system message |
| User 或 assistant `text` | Canonical message text |
| 带 URL 或 base64 source 的 user `image` | Canonical `image_url` content |
| Assistant `thinking` | Assistant `reasoning_content` |
| Assistant `tool_use` | 保持同一 ID 的 assistant function tool call |
| 带 text content 的 user `tool_result` | 保持同一 tool-use ID 的 tool message |

Messages tool 使用 `name`、`input_schema` 和可选 `description`；省略 tool `type` 或设置 `type: "custom"` 时，
会投影为 canonical function tool。Anthropic response 使用 text、thinking 和 tool-use block，并提供 Anthropic stop reason。
`stop_sequences` 会作为当前 turn 的 stop string 传给 generation。

#### Buffered SSE

Buffered SSE 会在完整 generation 执行期间保持 HTTP request 打开。Relax 先发送 connection frame；generation
等待期间每 15 秒发送一次 heartbeat。完成后，Relax 把当前协议的完整 terminal event sequence 一起发出，
不会暴露 token-by-token delta。

| 协议 | Terminal sequence |
| --- | --- |
| Chat Completions | 一个带 `usage` 的完整 `chat.completion.chunk`，随后是 `[DONE]`；失败时发送 OpenAI error data 和 `[DONE]` |
| Responses | `response.created`、每个 output Item 对应一个 `response.output_item.done`，随后是 `response.completed`、`response.incomplete` 或 `response.failed` |
| Messages | `message_start`、完整 content-block events、`message_delta`，随后是 `message_stop`；失败时使用 `event: error` |

Chat 与 Responses 使用 SSE comment 作为 connection 和 heartbeat frame。Messages 使用 `event: ping`。JSON 与
Buffered SSE 共享 normalized request、generation、SessionForest commit、usage accounting 和 finish reason。每个
成功的 Chat terminal chunk 都包含完整 usage，不依赖 `stream_options.include_usage`；Relax 不会额外发送 usage chunk。

### Agent Process Contract

Relax 向每个 agent process 注入：

| 变量 | 含义 |
| --- | --- |
| `RELAX_INPUT_JSON` | Session input JSON path |
| `RELAX_OUTPUT_JSON` | Session output path |
| `RELAX_SESSION_IO_DIR` | Per-session 临时目录 |
| `RELAX_BASE_URL` | Agentic model API base URL |
| `RELAX_SESSION_ID` | Session ID 与 API credential |
| `RELAX_ROLLOUT_MODE` | `train` 或 `eval` |
| `RELAX_GROUP_ID` | Runtime Group ID |

`RELAX_` 前缀由 Relax 保留。Shell launcher 可以把这些值映射到 application 已有接口：

```bash
#!/usr/bin/env bash

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

python -m my_agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
```

Relax 会管理 launcher 创建的 process group。Shell wrapper 可以使用 `exec` 简化 process tree，但 `exec` 不是
必须条件。Application 配置通过 `--agent-env` 传入：

```bash
--agent-env FOO=bar BAZ=qux
```

### 启动训练

在可运行的 Relax 训练命令中加入：

```bash
--use-agentic-rollout \
--agent-cwd /path/to/agent_repo \
--agent-command "bash run_agent_app.sh"
```

Model、dataset、parallelism 和 algorithm 配置可参考 `examples/` 下的完整 recipe。

### 验证第一个 Rollout

第一个 rollout step 完成时，日志中会出现 `accounting_end`：

```text
AGENTIC ROLLOUT event=accounting_end rollout=0 ...
```

进度条中的 `scored` 表示 sample 的打分已经完成。下一节说明不同场景下如何生成 reward、分配 advantage。

## 选择训练 Context 与分数

Agent 完成任务后，需要确定哪些 context 进入训练，以及 reward/advantage 如何产生。最常见的情况是，一个
sample 对应一个 reward/advantage，其数值来自环境或奖励函数。

![如何选择 Agentic reward 与 advantage](/agentic/agentic-reward-advantage-zh.svg)

| Example | 任务分数 | 训练 Context | Advantage |
| --- | --- | --- | --- |
| [`mini_swe_agent`](../../../examples/mini_swe_agent/README.md) | 分数来自环境自带的测试脚本，并保存在 `reward` 中 | 一个 context | Reward 后处理方式由 `--advantage-estimator` 决定 |
| [DeepEyes Agentic](../../../examples/deepeyes_agentic/run_deepeyes_agentic.sh) | 通过 `--custom-rm-path` 调用 [`reward_deepeyes.reward_func`](../../../examples/deepeyes_agentic/reward_deepeyes.py) | 一个 context | Reward 后处理方式由 `--advantage-estimator` 决定 |
| [Search-R1 multiagent](../../../examples/search_r1/app/multiagent.py) | Agent 计算 exact-match 分数并写入 export metadata | `main + searcher_*` | [Custom advantage](../../../examples/search_r1/advantage_search_r1.py) 为每个 export 分配 scalar advantage |

### 一个训练 Context

一个 Session 只导出最终 context 时，Relax 会直接使用已记录的 conversation。环境负责打分时，把结果写入
`RELAX_OUTPUT_JSON` 的 `reward` 字段；使用奖励函数时，配置 `--custom-rm-path`。需要记录其他信息时，把它们
写入同一 JSON object 的 `metadata` 字段。Relax 会在 rollout dump 中保留完整 metadata，并把其中的顶层数值字段
作为 metrics 上报到 ClearML、TensorBoard 等已启用的观测平台。

```json
{
  "metadata": {"task_success": 1.0},
  "reward": 1.0
}
```

`reward` 只有一个值时，直接写 number。需要同时记录多个打分结果时，写成 JSON object，并通过下面的参数指定
其中哪个字段作为 reward 参与训练；其他顶层数值字段会作为 reward metrics 上报：

```bash
--reward-key <primary-key>
```

隐式导出仅用于经过审查的严格线性历史。任何非线性历史都必须显式导出，即使当前只有一个 exportable leaf。

### 显式导出一个或多个 Context

当一个 Session 的对话上下文不再是严格线性拼接的，例如 multi-agent 协作或历史压缩，其中可能存在多个线性
context。显式导出允许用户选择其中哪些 context 用于训练，并通过
`--agentic-custom-advantage-path` 为每个 export 分配 advantage。

每个选中的 context 各占一条 JSONL record，全部写入 `RELAX_OUTPUT_JSON` 指向的文件。例如，一个 Session
选择 `main` 和两个 `searcher` context 时，写成：

```jsonl
{"name":"main","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"outcome":1.0}}
{"name":"searcher_0","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"usefulness":0.6}}
{"name":"searcher_1","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"usefulness":0.4}}
```

| 字段 | 是否必需 | 含义 |
| --- | --- | --- |
| `name` | 是 | Session 内唯一的非空名称，同时作为 custom advantage key |
| `messages` | 是 | Generation 实际使用的完整 message history |
| `tools` | 使用时必需 | 该 context 使用的准确 tools |
| `chat_template_kwargs` | 使用时必需 | 该 context 使用的准确 template arguments |
| `metadata` | 否 | Per-context metric 与 custom advantage 输入 |
| `reward` | 否 | Per-context 任务 outcome，可以是 number、object 或 `null` |

Relax 训练 JSONL 中出现的 record；未写入的 context 不参与训练。需要显式导出时，建议 agent 全程使用 Chat
Completions，并在 export record 中复用 generation 的完整 `messages`、`tools` 和 `chat_template_kwargs`，以匹配
Relax 记录的 SessionForest state。

::: warning 多 Context 训练参数
一个 Session 导出多个 context 时，训练命令还需要：

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu <token-budget>
```

这是多个物理训练行的 batching 约束，与 reward 来源或 advantage 粒度无关。
:::

::: warning Eval 只导出一个 Context
默认 Eval 会把每个 export 当作一个 sample 统计。一个 Session 导出多个 context 会改变平均 reward 的分母，也会
打乱 `--log-passrate` 按 `n_samples_per_eval_prompt` 进行的分组。因此 Eval 应只导出一个代表 context，通常是
`main`。Search-R1 multiagent 已按此方式实现：training 导出 `main + searcher_*`，Eval 只导出 `main`。只有自定义
Eval logger 明确按 Session 重新聚合时，才应在 Eval 中导出多个 context。
:::

### Custom Advantage

Search-R1 multiagent 展示了 scalar advantage：它导出 `main` 和 `searcher_*`，在 sampled Group 内归一化 main
分数，并为每个 export 分配一个 scalar。

Custom advantage 使用具名的显式 export record，即使一个 Session 只导出一个 context。

配置方式：

```bash
--agentic-custom-advantage-path my_package.advantage.advantage_func
```

`advantage_func` 会收到当前 Group 中所有 Session 的导出信息。前面每条 JSONL record 里的 `name` 和
`metadata`，在这里会变成 `{name: metadata}`。下面的最小例子读取前面 JSONL 中的 `outcome` 和 `usefulness`，
先在当前 Group 内对 main outcome 做中心化，再为每个 export 生成 scalar advantage：

```python
import statistics
from typing import Any


def advantage_func(
    metadata_by_slot: list[dict[str, dict[str, Any]]],
) -> list[dict[str, float | list[float]]] | None:
    main_scores = [float(exports["main"]["outcome"]) for exports in metadata_by_slot]
    mean = statistics.fmean(main_scores)
    return [
        {
            name: (score - mean) * float(metadata.get("usefulness", 1.0))
            for name, metadata in exports.items()
        }
        for exports, score in zip(metadata_by_slot, main_scores)
    ]
```

`advantage_func` 返回 `None`，或返回一个与输入 Session 一一对应的 list。List 中第 i 项属于第 i 个 Session，
并沿用显式导出时填写的 export name。

每个 export 对应的 advantage 可以是：

- 一个 scalar：所有 assistant 回合共用这个分数；
- 一个 list：每个 assistant 回合对应一个分数，list 长度必须等于 assistant 回合数。

Relax 会把每个回合的分数应用到该回合的 token，observation token 的分数为零。如果整个函数返回 `None`，Relax
会丢弃并补采整个 Group。Eval 不调用 `advantage_func`。

::: warning
Custom advantage 会跳过 GRPO Group normalization 等标准 reward 后处理。需要在 Group 内比较或归一化时，
应在 `advantage_func` 中完成。

`--normalize-advantages` 与 GRPO normalization 无关。该开关在 advantage 生成后，对 data-parallel group 中所有
有效 token 的 advantage 做 masked whitening。GRPO normalization 属于更早的 reward 后处理，按同一 prompt 的
sampled Group 处理 reward；它是否执行不由 `--normalize-advantages` 控制。
:::

::: warning
训练使用 custom advantage 返回的值。函数需要的值都要写进 export metadata。Eval 或 reward 指标还需要
`reward` 时，在 `RELAX_OUTPUT_JSON` 中保留相应的 `reward` 字段。
:::

### Metrics 与 Passrate

- `reward` 为 `rollout/raw_reward` 和 `--log-passrate` 提供任务 outcome。
- Reward object 中的 numeric helper field 使用 `rollout/<field>/mean|median|max|min`。
- Output metadata 的顶层 numeric field 使用 `<field>/mean|median|max|min`，不带 `rollout/` 前缀。
- Rollout dump 保留完整 metadata。

启用 `--log-passrate` 时，multi-context Session 使用显式导出，并且由一个代表 context 携带 reward，通常是
`main`。选中的 primary reward value 成功时设为 `1`，其他情况设为 `0`；reward object 通过 `--reward-key` 选择该
值。Sibling context 不设置 reward。Custom advantage 需要的 outcome 可以保存在各 context 的 metadata 中。
Multi-context training 中，reward 用于上报任务结果，custom advantage 决定每个 export 的 advantage。

## 配置 Runtime 行为

::: danger 必须满足的 external agent 容量
Agent 在带有硬并发限制的远程集中式平台运行时，先计算 train 上限和每个 Eval dataset 的上限：

```text
T = agentic_concurrency * n_samples_per_prompt
G_d = dataset d 的 n_samples_per_eval_prompt
C_d = 显式配置的 agentic_eval_concurrency，未配置时为 ceil(T / G_d)
E_d = C_d * G_d
E_peak = 所有 Eval dataset 的 E_d 最大值
```

多个 Eval dataset 串行执行，因此组合峰值是 `E_peak`，无需把所有 `E_d` 相加。

启用 `--agentic-prelaunch`、`--partial-rollout` 或 `--fully-async` 时，表格中的
**Eval 期间是否保留 train sessions** 应选择“是”。Prelaunch 与当前 step 共享 training resident capacity，
因此 train 上限仍是 `T`；它会让 `T` 与 `E_peak` 同时存在。

Train 与 Eval 共用一个 executor 时，再选择对应场景：

| 是否启用 Eval | Eval 期间是否保留 train sessions | 所需 slots |
| --- | --- | --- |
| 否 | — | `external_slots >= T` |
| 是 | 否 | `external_slots >= max(T, E_peak)` |
| 是 | 是 | `external_slots >= T + E_peak` |

Slots 不足时，Group 启动可能卡在 all-session first-request barrier。启动任务前必须检查该表格。
:::

| 目标 | 参数 | 行为 |
| --- | --- | --- |
| 限制 resident training Group | `--agentic-concurrency` | Prepare 与 Runtime 共享的 capacity |
| 设置 resident Eval Group | `--agentic-eval-concurrency` | 逻辑 Eval prompt Group capacity，未配置时根据 training capacity 推导 |
| 提前启动 agent | `--agentic-prelaunch` | Resident capacity 空闲时启动 process |
| 复用未完成 sample | `--partial-rollout` | 跨 step abort 并恢复 backend attempt |
| 异步执行 rollout 与 training | `--fully-async` | Partition 前进时保留未完成 session |
| 限制 partial abort 次数 | `--partial-rollout-max-aborted-count` | 保护多次被 abort 的 attempt |
| 终止运行过久的 agent | `--agent-timeout` | Runtime active-time budget 用尽后终止 agent process |

`--agent-timeout` 在 Session 进入 Runtime 后开始计时。Agent 的 loop、tool call 等持续运行过久时，Relax 会终止
该进程。Prelaunch Session 等待 Runtime 时不会消耗这段时间；partial rollout 或 fully async 让普通 Session 跨
step 暂停时，计时也会暂停。

`--agentic-concurrency` 默认使用 `--over-sampling-batch-size`，后者默认使用 `--rollout-batch-size`。
`--agentic-eval-concurrency` 未配置时，分别根据 `T` 和每个 Eval dataset 的 Group size 推导。两个参数都以逻辑
prompt Group 为单位。无论使用 ordinary RM 还是 Group RM，dataset `d` 都持有 `E_d` 个 Sessions；ordinary RM
内部使用 singleton Runtime Group，但不会改变该总数。多个 Eval dataset 串行执行，因此 Train 与 Eval 使用独立
executor 时，分别为 `T` 和 `E_peak` 配置容量。

Retriever、environment server 等 cross-session service 应在 per-session agent command 外启动。

Session KV lifecycle 与 program-aware admission 是长时间 Agentic workload 的可选能力。参见
[Agentic KV Scheduling](./agentic-kv-scheduling.md)。

## 理解 Agentic Rollout 原理

### Session Lifecycle

一个 dataset sample 创建一个 Session。Session 拥有一个 agent process、一个 SessionForest、rollout mode
和 active-time budget。Process 可以顺序或并发调用 model API。Process 退出后，Relax 选择指定 Forest
state、计算 advantage，并把 sample 发送到训练侧。

主要 runtime 路径是：

```text
Prepare → Runtime → Reward → Transfer
```

### SessionForest

SessionForest 保存每个 committed conversation state。不同 initial history、tools 或 template arguments 会形成不同
subtree。一个 subtree 可以包含多个 turn 和多个 branch。

![SessionForest multi-turn branches](/agentic/session_forest.svg)

Observation node 保存新增的 system、user 或 tool message，loss mask 为 `0`。Response node 保存 generated token
IDs、rollout logprobs 与可训练 loss mask。沿同一 normalized history 的多次 request 仍属于一个 context；history
分叉后形成不同 branch。最终导出的 leaf 决定训练 context。

#### State Identity 与 Prefix Matching

每种协议的 request 都携带完整 history，Relax 会先把它投影为 canonical messages。Relax 查找 tools 和 template
arguments 相同的最长 committed message prefix。未匹配的 suffix 成为新的 observation。完整匹配会从已有 state
创建 branch；没有匹配时从 technical root 开始。

<details>
<summary>参考实现</summary>

```python
@staticmethod
def _match_parent_state(
    *,
    forest: SessionForest,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> tuple[MsgNode, list[dict[str, Any]]]:
    for prefix_length in range(len(messages), 0, -1):
        prefix_hash = _messages_tools_template_state_hash(
            messages[:prefix_length],
            tools,
            chat_template_kwargs,
        )
        parent = forest.nodes_by_hash.get(prefix_hash)
        if parent is not None:
            return parent, messages[prefix_length:]
    root_state_hash = forest.root_state_hash
    assert root_state_hash is not None
    return forest.nodes_by_hash[root_state_hash], messages
```

</details>

### Token 与 Sample 语义

#### Token-in, Token-out

Relax 把选中 lineage 的 inference token prefix 发给 SGLang，并把 SGLang 返回的 token IDs 原样追加到 inference
和 training response delta。Generated token、loss mask 与 rollout logprob 保持对齐，同时避免重新 tokenize
response。

Text observation 的 inference 与 training 使用相同 tokenizer IDs。Multimodal observation 为 inference 保存
tokenizer IDs 与 media，同时为 training 保存 processor-expanded IDs 与 `multimodal_train_inputs`。Context limit
使用 processor 展开后的 training length。新增 media 归属当前 observation node，并沿 exported lineage 合并。

Observation-delta 与 multi-turn multimodal design 参考
[One Rollout to Rule Them All: Seamless Multi-Turn RL for LLM and VLM](https://app.notion.com/p/One-Rollout-to-Rule-Them-All-Seamless-Multi-Turn-RL-for-LLM-and-VLM-2e1ab71c210b8055b51de78b637e39b1#2e1ab71c210b8096bcb1ce296737fd90)。

#### 从 Branch 构造 Training Sample

```text
initial observation ─ response ─ observation ─ response
       prompt          loss=1       loss=0       loss=1
```

Relax 从 exported state 回溯到 root，并拼接已记录的 delta。Initial observation 构成 prompt。后续 observation
保留在 continuation 中，loss mask 为 `0`。Model response 使用可训练 mask 和对齐的 rollout logprob。Exported
lineage 至少包含一个 committed response。

### Runtime Scheduling

#### Resident Capacity

Prepare 与 Runtime 共享 resident Group capacity。Reward work 位于该 capacity 外。Runtime Group 完成或 drop 后
会释放 slot。Group 被 filter 后会产生新的需求。

![Shared resident capacity and prelaunch](/agentic/resident_capacity.svg)

#### Prelaunch

Prelaunch 改变 agent process 的启动时间，不改变 request 进入 Runtime 的时间。启用 prelaunch 后，agent 可以提前
发送第一条 request；Relax 会持有该 request，直到 Group 获得 Runtime lease。

![Agent process prelaunch across multiple turns](/agentic/warmup.svg)

#### 跨 Step 留存

Partial rollout 与 fully async 都可以跨 rollout step 保留 Session。下图展示 partial rollout 路径：SGLang
在 abort 后返回 partial token prefix，Relax 暂停 request，后续 backend attempt 继续同一个 HTTP request。

![Request-level partial rollout](/agentic/partial_rollout.svg)

Fully async 留存不要求每个 session 都经过同一套 abort/resume 时序。Partition 与 backfill 行为参见
[Fully Async Training](./fully-async-training.md)。

#### KV Scheduling

Program-aware admission 可以根据预测的 KV 使用量延迟 backend attempt。Session lifecycle 可以在 session 结束时
释放 Session radix-cache entry。参见 [Agentic KV Scheduling](./agentic-kv-scheduling.md)。

## 运维与故障排除

### Metrics 与 Dump

配置 `--save-debug-rollout-data <path-with-{rollout_id}>`，可保存完整 metadata、SessionForest state hash、terminal
status、turn count、request timing、abort count 与 weight-version 信息。

### Agentic 可观测性 JSONL

设置 `--dump-agentic-obs <directory>` 后，Relax 在生成的 `run_*` 目录下写入
`manifest.json` 和 JSONL 分片。需要收集所有生产节点的文件，或使用共享输出目录。
每条 JSONL 都包含 `schema_version` 和 `run_id`；`sample_export_id` 使用十进制字符串，
`samples[]` 内也遵循同一规则。

| 数据流 | 记录与关联键 |
| --- | --- |
| `prompts/*.jsonl` | 原始数据行、`prompt_id`、数据集来源、原始 `input_payload`、可选引用和文本预览。 |
| `groups/*.jsonl` | Runtime Group 创建信息、`group_id`、`group_index`、`prompt_id`、预期 Session 数、模式和创建时间。 |
| `sessions/*.jsonl` | `session_id`、`group_id`、`prompt_id`、sample/replicate 索引、预期 Group 大小及合并后的采样默认值。 |
| `trajectories/*.jsonl` | 导出的 lineage、每 resp 节点的 `weight_version_spans`（token 区间 + 生成时版本，未知为 null）、`sample_export_id`、`session_id`、runtime `group_id`、`sample_index` 和 `exported_at`。同一 Sample 的多个 export 共享 sample index，各有独立 export ID。 |
| `stages/*.jsonl` | 成对的 stage 边界，包含 `stage_id`、`record_id`、`scope`、`event_seq`、`occurred_at` 和 `is_terminal`。trajectory 阶段描述 reward、filter 和 transfer；rollout/update 事件包含 `group_ids` 和 `groups_finalized`。 |
| `train/*.jsonl` | 每个真实 policy update 的 DP 分片：`update_id`、`train_generation_id`、`stage_id`、`capture_phase=training_input`、`samples[]`、`dp_rank`、`dp_world_size` 和 `input_complete`。 |

目前训练侧支持 Megatron policy actor。Update ID 为 `actor:<rollout_id>:<step_id>`，
作用域为 `(run_id, train_generation_id)`；一个 rollout 可以包含多次 update。输入在 micro-batch 被实际消费时、
packing 和 dynamic CP 切分前采集。Dummy batch 不计入，真实的全零 loss mask 样本仍保留。
每个样本包含长度、有效 token 数，以及按 mask 计算的 advantage/return 统计。
训练消费时刻的 actor 权重版本记为每样本 `consumed_weight_version`（未采集为 null）。
Train schema `4.0.0` 将 `loss_mask.ranges` 保存为 response 空间中从零开始的
左闭右开区间 `[start, end, value]`。按 `response_length` 还原完整长度，区间外填零。
例如 `[1, 1, 0, 1]` 写成 `{"ranges": [[0, 2, 1], [3, 4, 1]]}`。
Train schema `3.0.0` 的 `loss_mask` 是逐 token 数组。Static CP 下统计为 `null`，
dynamic CP 在切分前采集完整序列。这里记录的是 loss 内部 TIS/OPSM 筛选前的输入 mask。

DP 分片主键是 `(run_id, train_generation_id, update_id, dp_rank)`，平台按
`(run_id, train_generation_id, update_id)` 合并。Actor group 初始化时为 worker 参数副本
分配一个共享代际 ID；重新初始化会换新 ID，不改变训练配置，也不增加 collective。
输入分片还必须匹配对应 DP 边界的 `stage_id`，避免重放 update 时误用旧执行的输入。
代际 ID 缺失时表示未知。
每个 DP owner 向 `stages/actor_<rollout_id>_dp_<rank>.jsonl` 写成对的
`scope=update_fragment` 边界，包含 `dp_world_size`、`step_id`、`train_generation_id`、`update_sequence`
以及本地 `optimizer_successful` 结果。训练中不再执行观测专用 collective。
训练异常写 failed/cancelled，此时 optimizer 结果未知，成员集合不完整。

读取端调用 `relax.utils.agentic_obs.merge_train_updates(stage_records, train_records)`。
只有全部预期 DP 都有唯一、一致的 start/end，且 optimizer 结果均成功，才判定
`committed`。证据缺失或冲突表示 `unknown`；训练输入分片缺失则独立地令
`input_complete=false`。旧格式的 `scope=update` 记录原样保留。
`optimizer_step` 在每个 actor group 代际内从第一次成功 update 开始计数，起始为 1；跳过的 update
不递增。各 DP 序号历史出现缺口或冲突后，该计数为 null。
`groups_finalized=true` 表示成员集合封口，不表示样本成功。
Runtime `group_id` 与 reward 归一化使用的 `group_index` 含义不同。

自定义转换器应提供与输出行对齐的 CPU int64 `sample_export_ids`。身份缺失或非法时，
补齐 `-1` ID 和空 group ID，训练数据行保持不变，TQ 观测列仍会提供；下游成员集合和
输入完整性标记为 false。不会按输入顺序猜测输出身份。

Reward 完成和 transfer 回执均不是终态。提交后 TQ RPC 失败时，使用
`reason_code=transfer_outcome_unknown` 和 `is_terminal=false`，因为远端写入可能已经成功。
筛选失败和 Eval 完成是 trajectory 终态。平台按 `record_id` 去重，并在对应 scope 的身份内
按 `event_seq` 排序，不使用跨分片的文件顺序。

训练主进程在 `stages/run.jsonl` 写入生命周期，使用
`scope=run, stage=execution`。开始时写 `start`，结束时写一个 `completed`、
`failed` 或 `cancelled` 终态；两条记录共用 `stage_id`，终态的 `event_seq=2`。
终态的 `occurred_at` 是训练结束时间，`operation` 表示 `startup`、`training`
或已确认的 `checkpoint` 失败阶段。`reason_code` 为 `run_failed` 或
`run_cancelled`，`reason` 保留异常类型及错误信息，取消时记录收到的信号。
Megatron 在 Checkpoint 错误跨 Ray 传递前标注阶段，包括异步保存的收尾错误；
没有标注时保留主进程所知的较粗阶段，不根据 traceback 文本猜测阶段或配额错误。
首个终态生效，清理期间的信号不会覆盖已有的完成或失败结果。
原有 `scope=run, stage=startup` 记录描述已解析的调度计划，不能当作整个 Run 的结果。

平台消费时应分别维护三个状态：

| 展示项 | 判断依据 |
| --- | --- |
| 训练 Run | Run execution 终态；缺失时使用权威 Trial 终态兜底。 |
| 采集新鲜度 | 采集器最近一次成功采集时间及配置的新鲜度阈值。 |
| Session | 该 Session 自身的执行事件，与 Run 和采集器状态分别判断。 |

不能长期保留登记时的 `Running`。没有 Run 终态或权威 Trial 结果时，采集过期只能显示
**状态待确认**，不能据此认定成功、失败或取消。Run 已结束也不能证明各生产端文件已收齐。
`SIGTERM` 和 `SIGINT` 会在远程清理前记录取消；`SIGKILL`、节点消失或观测目录不可写
可能只留下 start。进程退出但没有记录结果时，Run 结果同样未知。参数解析发生在 Run start
之前。Checkpoint 存储可能写满时，应将观测文件放在仍有独立配额的存储位置。

Run 失败后仍保留已完成的训练进度：在 Run 内按 `(train_generation_id, update_id)` 去重统计 committed 更新，
将这些更新各 DP 输入的 `sample_export_id` 关联到 trajectory 的 `session_id`，再跨更新
去重，得到实际参与训练的 Session。不能仅凭 Group 关联推断 Session 已被消费。
DP 分片或 trajectory 未收齐时，应展示已观测到的下界或未知成员集合，不能给出精确的
Session 总数。这样页面可以同时展示 Checkpoint 失败原因、已提交更新数和已参与训练的 Session。

Dump 采用 best-effort 语义。生产端只构建快照并入队，不等待磁盘。
Daemon 队列同时限制条目数和字节数，正在写入的任务也计入容量。每次 recorder 或进程
关闭共用一秒 drain 预算，包含 flush/status；超时或取消后丢弃尚未执行的队列尾部，
只有正在执行的写入继续持有资源。每个进程最多运行 16 个观测 worker，防止磁盘持续
阻塞时反复创建 recorder 导致线程无限增长。观测持久化不再屏蔽取消，也不新增训练 collective。

磁盘可写时，生产端 `*.status.json` 提供写入和丢弃计数。输入队列饱和时，独立的有界队列
仍可保存小型本地 optimizer 事实；整体写盘失败或进程突然退出仍可能丢记录。
只有收齐全部 DP 输入分片且 `input_complete=true`，才能确认输入集合完整。
记录缺失、日志文件缺失、export 交付未确认或 JSONL 末尾只有半行均表示**未知**，
不能据此断言训练仍在等待或已被拒绝。
Managed 日志在临时目录清理前提取本地最后 1 MiB 快照；`log_ref` 表示后台写入已接纳，
读取端仍需检查文件是否存在。
Manifest 使用 `record_format=agentic-observability-full-v3`、schema `3.2.0`。
`train_schema_version` 标明 train 数据流的版本。读取端通过合并 DP 事实判断 update 结果，
不能继续等待生产端写出的全局 commit 事件。

Eager 和 streaming reader 在自定义 prompt 处理、模板渲染、过滤、打乱和复制之前保存来源身份。
`source_row_id` 是从零开始的 JSONL 物理行号或 Parquet 行号，切片和多文件数据集也保留原行号。
`dataset_id` 默认是解析后的源文件路径；metadata 可以提供 `dataset_id`、`dataset_version`
和 `split`，未知版本保留 `null`。Prompt ID 对来源和原始内容计算摘要：同一行重采样保持相同
身份，内容相同但来源行不同、或未声明版本的内容发生变化时会产生不同 ID。
Train 和 Eval split 使用不同身份。

自定义数据源可以通过 `Sample.agentic_prompt_source` 提供这些来源和 payload 字段。
否则使用 `identity_kind=content_fallback`，按内容计算摘要，dataset/row 字段为 null；无法将
完全相同的自定义输入区分为不同原始行。平台在 run 内按 ID 对 prompt 和 Group 清单去重；
每个 driver 实例写入独立的生产端分片。

Group 在取出数据时记录，包括仍在等待准备的 Group。Session 在确认启动后、注册到 Shard 前
写入清单；真实准备失败也保留对应清单。Group 延后到下一 step 不会产生失败 Session 或额外
副本。Eval 未启用 group reward 时，每个单 Session runtime Group 的 `expected_group_size=1`，
算法层 `group_index` 仍可能相同。`sampling_config` 保存合并后的 Session 默认值与覆盖配置；
终态的 `llm_calls` 保存实际请求的覆盖参数及 context 上限裁剪结果。

每次执行写一个 `start` 和一个具体结果，两行使用相同
`stage_id`，`occurred_at` 分别是起止时间。`record_id` 用于事件去重；`event_seq` 在同一
scope 对象内跨阶段递增。`is_terminal` 表示对象生命周期结束，与某个 stage 闭合分别判断。

| scope / stage | start 边界 | 结束结果 |
| --- | --- | --- |
| run / execution | 训练主进程开始启动 | 整个训练 Run 的 completed / failed / cancelled |
| rollout / execution | train 或 Eval 调度批次开始 | 批次收尾后的 completed / failed / cancelled |
| session / rollout | Driver 开始准备一个已接受的 Session | Shard 清理后的 completed / failed / cancelled |
| trajectory / reward | 开始 RM 或接收 agent 提供的 reward | completed / failed / cancelled |
| trajectory / filter | 开始筛选或 custom advantage 决策 | passed / filtered / failed / cancelled |
| trajectory / transfer | 开始转换并提交训练队列 | queued_for_train / failed / cancelled |
| update_fragment / train | 当前 DP owner 开始 update | completed / failed / cancelled；读取端推导全局 committed/failed/unknown |

成功入队使用 `status=queued_for_train`。旧版 dump 使用 `completed` 表示同一队列回执，
读取器可在 `stage=transfer` 时兼容该旧值。两种状态都不能证明 optimizer update 已提交。

`filter.operation` 区分 `dynamic_filter`、`custom_advantage`、`oversampling`、`shutdown`、
`execution`（group 丢弃决策）、`export_delivery` 和 `eval`（接受评估结果）。这些表示实际的
筛选决策，瞬时决策允许耗时为零。已提供的 reward 使用 `reward.operation=provided_reward`；
不含 reward 的 custom advantage 路径不会制造 RM 区间或零分。一次批量入队的各 export
共享 `transfer_batch_id`。队列回执未知仍为非终结状态，观察器不会自动重试。
`retry_of_stage_id` 留给显式业务重试；观察器本身不新增重试行为。

**Session 只写 start 和最终 rollout 结果：不包含 progress 字段、phase 字段、请求事件或
暂停恢复事件。** Driver 唯一写 start；尝试注册到 Shard 后由 Shard 唯一写 end。
可证明的本地准备失败写配对终态和空调用列表。注册结果未知仅记日志，保留开放的 start，
等待 Shard 的结果；Driver 不制造 failed 终态。

| 终态字段 | 口径 |
| --- | --- |
| `wall_ms` / `active_ms` | Driver 准备开始的墙钟耗时 / 排除暂停的 managed process 活动预算耗时。跨生产端需要时钟同步。 |
| `llm_calls` / `llm_calls_complete` | 所有已接纳的逻辑请求，包括拒绝、失败、取消和未导出调用；完整性描述 call/attempt 边界是否采齐。 |
| `request_count` / `attempt_count` | 不同逻辑请求数 / 进入 backend.generate 的次数。内部 HTTP 重试不单独计为 Runtime attempt。 |
| `generated_tokens` / `token_count_complete` | 返回的 token delta 总和，包括 aborted prefix 和清理期间的结果。生成结果或内部重试无法观察时 token 为 null。 |
| `export_ids` | 已发布的 export ID，十进制字符串；无 export 时为空。 |
| `diagnostics` / `log_ref` | 软性清理诊断 / 已入队 agent 日志的相对路径，可为 null，需检查文件是否存在。 |

每个 call 包含 `request_id`、从零开始的 `call_index`、`started_at`、`finished_at`、
`status`、`parent_state_hash`、`resp_state_hash` 和 `attempts`。每个 Runtime attempt 包含
从零开始的 `attempt_index`、`rollout_id`、起止时间和状态。partial 中止与恢复共用逻辑请求
ID，可以跨 rollout；agent 重新提交的请求使用新 ID，不能从内容相似推断重试。
生成前拒绝的请求使用 `attempts=[]`。达到长度上限是 `finish_reason=length` 的 completed
call，不是 failed call。

请求失败后 agent 可以恢复，不必导致 Session 失败。终态原因包括 `agent_timeout`、
`agent_exit_nonzero`、`agent_output_invalid`、`export_validation_failed`、
`session_prepare_failed`、`session_infra_failed` 和 `session_cleanup_failed`。
连带停止的未完成 sibling 及显式丢弃使用 cancelled 和 `category=selection`；Session 已完成
后不会被后续过滤重写。

平台获取 Session timeline 的步骤：

1. 在 `run_id` 内按身份加载 Session、Group、Prompt，收集该 Session 的全部 exports。
2. 展开所有 DP 分片的 `train.samples`，按 export ID 查实际消费 update；不能仅凭 group
   关联推断训练消费。
3. 收集 Session、exports 和实际消费 updates 的事件；按 `record_id` 去重，再按 `stage_id`
   配对并核对 scope、对象和 stage。内容冲突或多个终态属于数据错误，文件顺序不代表事件顺序。
4. 只从唯一 Session 终态读取 calls；没有终态时调用列表未知。每个逻辑 call 只画一次，内部
   绘制 attempts，以 `(session_id, request_id)` 关联 export 的 `agentic_trace.turns`，再按
   `resp_state_hash` 关联 lineage；未导出调用也保留。
5. 每条 export 绘制 reward/filter/transfer，每个实际消费 update 只画一次。按
   `transfer_batch_id` 聚合批量入队，以 attempt 的 rollout ID 关联批次背景；关联时带上
   Session 的 `rollout_mode`，区分 train 和 Eval。
6. 只有边界完整且先后正常才计算耗时。缺 end 保留开放区间，快照时间只用于展示，不是实际
   结束时间；时钟倒置应标记。LLM 调用间隔不能当作工具耗时。只有 committed update 才表示
   成功训练；只有收齐全部 DP 分片才可声明输入完整。

`--dump-agentic-obs-media` 还会保存按内容寻址的媒体字节。关闭时，prompt/trajectory 中的
data URI 和原始二进制 payload 替换为摘要/大小引用。日志在 agent 临时目录被移除前复制；
Shard 突然退出仍可能使最终复制无法完成。`tokens.jsonl` 按当前设计继续留到后续。

#### 权重版本滞后审计

对已产出的观测目录离线运行：

```bash
python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only] [--jsonl]
```

输出 run 级汇总（各滞后均值/中位数/最大值、三口径缺失率）与样本级明细（session → turn → attempt → token 区间 → 来源版本 → 消费时年龄）。指标口径定义见 `docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md` §6。

### 常见问题

- **Agent 没有启动或异常退出：** 检查 `--agent-cwd` 和 `--agent-command`，然后查看 `run.log`。Relax 会把 agent
  stdout 与 stderr 的末尾内容附加到 `AgentExecutionError`。
- **第一条 request 长时间等待：** Group 获得 Runtime lease 前，request 会保持等待；client 应配置长 timeout。
- **显式导出无法匹配：** 保留 generation 实际使用的完整 normalized `messages`、`tools` 和
  `chat_template_kwargs`。
- **隐式导出无法唯一确定 context：** 为需要训练的 context 写入具名显式 record。
- **多轮后 context length 失败：** 减少 observation size、completion length 或 turn 数；limit 使用完整的
  processor-expanded training lineage。

## 示例与下一步

- `examples/search_r1/`：text-agent 与 multi-agent training。
- `examples/deepeyes_agentic/`：multimodal tool-use training。
- `examples/mini_swe_agent/`：external agent server 与 sandboxed coding task。
- [Fully Async Training](./fully-async-training.md)：异步 rollout 与 training。
- [Dataset Design](./dataset-design.md)：dataset input 与 metadata。
