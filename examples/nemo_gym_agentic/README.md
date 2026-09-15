# Relax × NeMo Gym Agentic 集成

本目录把 [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) 的环境、agent harness 和 verifier
接入 Relax 的 agentic rollout 与强化学习训练链路。这里的目标不是在 Relax 内重写一套工具环境，
而是让 Relax 负责模型推理、采样和参数更新，让 NeMo Gym 负责任务状态、工具或 sandbox、agent
循环与可执行 reward。

当前镜像固定 NeMo Gym commit：

```text
a85670eb167ba9b48cc53a36a070eed815e6c40d
```

NeMo Gym 仍处于快速演进阶段。本目录的命令、patch 和数据格式都以这个 commit 为准，不能直接套用
NeMo Gym `main` 分支的新 CLI 或配置。

## 为什么 Relax 要集成 NeMo Gym

只靠“prompt -> completion -> 字符串 reward”的训练脚本，无法可靠覆盖以下 agentic 场景：

- 多轮工具调用及工具结果回灌；
- 每条轨迹独立的有状态数据库；
- 代码仓库、shell、文件系统和可执行测试；
- sandbox 生命周期、超时、清理和并发隔离；
- 用最终环境状态或测试结果产生可验证 reward。

NeMo Gym 将一个 environment 定义为任务数据、agent harness、verifier 和每任务状态的组合。Relax
已经具备分布式模型训练、SGLang rollout、Megatron actor 和 GRPO/PPO 等能力，因此二者的职责边界是：

| 组件                           | 负责内容                                                          |
| ------------------------------ | ----------------------------------------------------------------- |
| Relax                          | 模型服务、采样、token/trajectory dump、优势计算和参数更新         |
| 本目录的 thin client + Gateway | session 绑定、trial 生命周期、callback 转发、deadline/lease/abort |
| NeMo Gym                       | agent harness、工具或 sandbox、环境状态和 verifier                |

对于 GSM8K 这类无状态任务，单独写 reward 函数更简单；集成的主要价值体现在 Workplace Assistant、
R2E-Gym 这类多轮、有状态或带 sandbox 的任务。

## 使用 Skill 接入新环境

新增或调试 `examples/nemo_gym_agentic/recipes/` 下的 recipe 时，请使用仓库内的
[`nemo-gym-recipe-integration`](../../skills/nemo-gym-recipe-integration/SKILL.md) skill：

```text
/nemo-gym-recipe-integration <nemo-gym-environment>
```

该 skill 会按“准备数据、启动本地 NeMo Gym 服务、启动远端 Relax 训练”三个步骤指导接入，并覆盖
verifier 验证、callback 网络、资源清理和失败排查。

## Recipe 索引

| Recipe              | 用途                                                  | Sandbox                            | 当前参考配置             | 文档                                                                                                |
| ------------------- | ----------------------------------------------------- | ---------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------- |
| Calendar            | 长对话历史、日程约束和可验证二值 reward               | 无                                 | Qwen3-4B，8K             | [README](recipes/calendar/README.md) · [PITFAIL](recipes/calendar/PITFAIL.md)                       |
| GSM8K               | 最快验证 trial、callback 和数值 reward；无工具        | 无                                 | Qwen3-4B，8K             | [README](recipes/gsm8k/README.md) · [PITFAIL](recipes/gsm8k/PITFAIL.md)                             |
| Workplace Assistant | 多轮工具调用、五组有状态数据库、最终状态 verifier     | 进程内会话状态，不依赖 OCI sandbox | Qwen3-4B，8K             | [README](recipes/workplace-assistant/README.md) · [PITFAIL](recipes/workplace-assistant/PITFAIL.md) |
| Math + Workplace    | 同一训练任务按 row 混合无状态 math 与有状态 tools     | 按环境分别执行 cleanup             | Qwen3-4B，24K            | [README](recipes/multienv-math-workplace/README.md)                                                 |
| reasoning-gym-cc    | Claude Code、Bash 工具和 reasoning-gym scorer         | Gym 容器内进程组                   | Qwen3-4B，16K            | [README](recipes/reasoning-gym-cc/README.md)                                                        |
| R2E-Gym             | 代码仓库修改、OpenHands、Apptainer、可执行测试 reward | 每题独立 SIF，必须有 Apptainer     | Qwen3-4B 集成 smoke，32K | [README](recipes/r2e-gym/README.md) · [PITFAIL](recipes/r2e-gym/PITFAIL.md)                         |

选择建议：

1. 验证长历史和约束 verifier 用 Calendar。
2. 首次检查协议用 GSM8K。
3. 验证真正的 tool call/session/reward 用 Workplace Assistant。
4. 验证逐行多环境路由用 Math + Workplace。
5. 验证 Claude Code 原生 Messages 和 Bash 工具循环用 reasoning-gym-cc。
6. 验证长程代码 agent、sandbox 和执行式 reward 用 R2E-Gym。

## 总体架构

推荐始终使用两套独立 Ray cluster。它们只通过 HTTP 通信：

```text
Relax 训练容器 / Relax Ray (:6379, dashboard :8265)
  ├─ Actor / Megatron
  ├─ Rollout / SGLang
  ├─ Agentic model APIs (Ray Serve :8000)
  └─ 每条 session 启动一个 NeMo Gym thin client
       │
       │ POST /v1/trials + GET/renew/abort
       ▼
NeMo Gym 容器 / Gym 私有 Ray (:6381)
  ├─ Relax Gateway model (:28100)
  ├─ NeMo Gym agent (:28101)
  ├─ resource/verifier (:28102，可选)
  └─ Gym head server (:28103)
       │
       │ /ng-rollout/<opaque-id>/v1/chat/completions、/v1/responses 或 /v1/messages
       ▼
Gateway callback bridge
       │
       │ request-scoped RELAX_BASE_URL + Bearer session token
       └──────────────────────────────► Relax Agentic model API (:8000)
```

R2E-Gym 在 agent 后面还会为每条题目启动 OpenHands 和 evaluator Apptainer：

```text
swe_agents (:28101)
  ├─ OpenHands Apptainer
  └─ R2E evaluator Apptainer
       └─ 在该题 SIF 内执行测试并产生 reward
```

### 同一条 session 是怎样关联的

Relax 的 managed-agent runtime 为每条 session 提供：

- `RELAX_SESSION_ID`；
- `RELAX_GROUP_ID`；
- `RELAX_BASE_URL`，指向该 session 的 Agentic model API；
- 输入和输出 JSON 文件。

thin client 用 session ID 和 attempt 生成稳定但不泄露原 ID 的 `request_id`，然后向 Gateway
发送 `POST /v1/trials`。Gateway 为 trial 生成新的 opaque `rollout_id`，并在内存中保存：

```text
rollout_id -> Relax callback URL + session bearer token + model name
```

NeMo Gym agent 的每次模型请求都经过
`/ng-rollout/<rollout_id>/v1/chat/completions`、`/v1/responses` 或 `/v1/messages`。Gateway 因而能把每个 turn
准确转发到原 Relax session。trial 进入终态后，callback capability 会被删除。

### 为什么 Relax 调用 `/v1/trials`

`/v1/trials` 是本集成增加的长期服务协议，不是 Relax 随意猜出的上游路由。NeMo Gym 原始 agent
通常暴露一次性的 `/run`；Relax 的 managed-agent 范式还需要：

- create 幂等；
- admission queue 和并发限制；
- 长任务的 deadline 与 lease heartbeat；
- poll 终态；
- 取消、清理确认和终态竞争；
- request-scoped callback 与 session 隔离。

因此 Gateway 对 Relax 暴露 trial API，再在内部调用 NeMo Gym agent `/run`。

## 端口和网络

| 端口  | 所属进程                 | 要求                         |
| ----- | ------------------------ | ---------------------------- |
| 6379  | Relax Ray GCS            | 只给 Relax cluster 使用      |
| 6381  | Gym 私有 Ray GCS         | 只给 Gym graph 使用          |
| 8000  | Relax Ray Serve          | Gym host 必须能访问          |
| 8265  | Relax Ray dashboard/jobs | 提交和查看 Relax job         |
| 28100 | Gateway                  | 所有 Relax worker 必须能访问 |
| 28101 | NeMo Gym agent           | Gym 内部及验证脚本访问       |
| 28102 | resource/verifier        | GSM8K、Workplace 使用        |
| 28103 | Gym head server          | Gym graph 管理               |

`GYM_HOST` 是其他机器能够访问的 Gym host/IP，不是 Ray dashboard URL。`NEMO_GYM_CALLBACK_ALLOWED_NETWORKS` 默认为 `10.0.0.0/8`，必须覆盖
Relax 实际 callback URL 中的 IP，使用逗号分隔的严格 CIDR（单地址用 `/32` 或 `/128`，禁止 `/0`）。代理配置必须把上述内网 host 加入 `NO_PROXY`，否则本地
callback 可能被错误发送到代理。

## 镜像关系

需要两个运行角色：

1. `RELAX_IMAGE`：公开的标准 Relax 训练镜像 `ghcr.io/redai-studio/relaxrl:latest`，运行 GPU Ray
   cluster 和训练任务；
2. `NEMO_GYM_IMAGE`：由本目录 Dockerfile 基于 `RELAX_IMAGE` 构建，运行 NeMo Gym 服务。

Gym 镜像继承完整 Relax 镜像，所以体积较大。这样做的原因是复用 Python 3.12、Ray、Apptainer
和系统依赖，并不表示 Gym 服务要占用 GPU。Dockerfile 保留 Relax 系统 Python/`ray` 为默认
环境；NeMo Gym launcher 使用 `/opt/nemo-gym/.venv/bin/...` 的绝对路径，避免训练入口误用 Gym
venv。

### 构建 NeMo Gym 镜像

```bash
export RELAX_IMAGE="ghcr.io/redai-studio/relaxrl:latest"
export NEMO_GYM_IMAGE="relax-nemo-gym:a85670e"
export http_proxy="http://proxy.example.com:3128"   # 无代理时留空
export https_proxy="${http_proxy}"
export no_proxy="127.0.0.1,localhost"

DOCKER_BUILDKIT=1 docker build \
  --network host \
  -f examples/nemo_gym_agentic/service/Dockerfile \
  --build-arg HTTP_PROXY="${http_proxy}" \
  --build-arg HTTPS_PROXY="${https_proxy}" \
  --build-arg NO_PROXY="${no_proxy}" \
  -t "${NEMO_GYM_IMAGE}" \
  .
```

Dockerfile 默认基于 `${RELAX_IMAGE}`。如果要使用其他已有的 Relax tag，再显式增加
`--build-arg RELAX_IMAGE="<image>"`，不需要从本仓库重新构建 Relax 镜像。

构建上下文由 `.dockerignore` 排除所有 `env.sh`。不要把 API key、模型服务 header 或代理凭据写入
Dockerfile、README 或提交到仓库。

构建后做双用途 preflight：

```bash
docker run --rm "${NEMO_GYM_IMAGE}" bash -lc '
  test "${NEMO_GYM_COMMIT}" = "a85670eb167ba9b48cc53a36a070eed815e6c40d"
  /opt/nemo-gym/.venv/bin/gym --help >/dev/null
  /usr/bin/python3 -c "import loguru, ray"
  ray serve --help >/dev/null
  apptainer version
'
```

最后三项保证这个镜像即使被平台复用于训练，也不会再次出现 Gym venv 抢占 PATH 后
`loguru` 缺失或 `ray serve` 不存在的问题。

## 模型准备

当前 Qwen3-4B 训练脚本要求：

```text
${MODEL_DIR}/Qwen3-4B/
```

`MODEL_DIR` 是父目录。在当前机器直接使用 `hf` CLI 下载：

```bash
export MODEL_DIR="/绝对路径/models"
mkdir -p "${MODEL_DIR}"

hf download Qwen/Qwen3-4B \
  --local-dir "${MODEL_DIR}/Qwen3-4B"
```

检查：

```bash
test -s "${MODEL_DIR}/Qwen3-4B/config.json"
test -s "${MODEL_DIR}/Qwen3-4B/tokenizer.json"
```

Qwen3-4B 原生上下文为 32K。GSM8K/Workplace reference recipe 限制为 8K；R2E reference
recipe 使用 32K。4B 适合验证集成，尤其 R2E 的真实解题率不应以 4B smoke 作为能力预期。

## 正确的启动顺序

每个 recipe 的完整命令见独立 README；所有 recipe 都遵循：

1. 使用公开 Relax 镜像并构建 NeMo Gym 镜像。
2. 下载模型。
3. 准备原始任务数据和 Relax 格式预览；R2E 还要构建 SIF。
4. 启动 NeMo Gym 私有 Ray 和对应 graph。
5. 等待 `http://${GYM_HOST}:28100/readyz` 返回 `"ready": true`。
6. 运行 deterministic verifier/trial 检查。
7. 启动或复用独立的 Relax Ray cluster。
8. 从 Relax Ray head 提交 recipe 训练脚本。
9. 同时检查 Relax JSONL dump、Actor 日志和 Gym reward；不能只看 Ray Job 状态。

Relax 标准 launcher 会清理 Ray Serve 和残留训练进程，所以不要把 Gym graph 和 Relax 训练放在
同一套 Ray cluster。独立 Gym 可以先启动并长期复用。

## 怎样判断“跑通”

“跑通”分四层，必须明确说的是哪一层：

| 层级              | 验收信号                                                                             |
| ----------------- | ------------------------------------------------------------------------------------ |
| 服务 ready        | `/readyz` 的 `ready == true`，只证明 graph 存活                                      |
| verifier contract | 正确动作/patch reward=1，错误动作 reward=0                                           |
| model rollout     | `rollout_result/train/*.jsonl` 存在，status 终态，工具环境有真实多轮交互             |
| training step     | Actor 日志没有 OOM/异常，并出现 optimizer/metrics step；不能只信 Ray Job `SUCCEEDED` |

查看 rollout：

```bash
export RESULT_DIR="/绝对路径/实验目录/Qwen3-4B_mcore_8xgpu/rollout_result/train"

find "${RESULT_DIR}" -maxdepth 1 -name '*.jsonl' -type f -print
jq -c '{
  rollout_id,
  sample_index,
  status,
  reward,
  agent_turns,
  prompt_token_count,
  response_token_count,
  total_token_count
}' "${RESULT_DIR}"/*.jsonl
```

当前 Relax dump 的 `.response` 是序列化文本，tool call 不一定是独立 JSON object。检查 Qwen
tool-call 标签：

```bash
jq -r '.response' "${RESULT_DIR}"/*.jsonl | rg -c '<tool_call>'
```

也可以直接启动内置 viewer：

```bash
python -m relax.utils.visualize "/绝对路径/实验目录/Qwen3-4B_mcore_8xgpu/rollout_result"
```

### 当前验证边界

截至 2026-07-28：

- Gateway 协议、session capability 和 adapter 有本地自动化测试；
- R2E 一条真实模型 rollout 已经经过 OpenHands、12 个模型 turn、Apptainer evaluator 并写出 Relax
  JSONL；模型没有生成有效 patch，因此 reward=0，这是能力失败，不是链路失败；
- 该 R2E 2-GPU 任务随后进入 Actor 训练，但先后遇到长序列 log-prob 和无效 entropy 计算 OOM；
  已加入 log-prob chunk 和 `entropy_coef=0` 时跳过 entropy 的修正，最后一次 optimizer step 尚未在
  用户停止任务后复验；
- 因此本目录不能宣称“R2E 2-GPU 训练端到端已通过”。正式验收仍应使用 8-GPU reference，并按
  R2E README 重新验证 optimizer step。

## 目录结构

```text
examples/nemo_gym_agentic/
├── README.md                         # 本文：中文总览、架构、recipe 索引
├── RUNBOOK.md / RUNBOOK_zh.md        # 兼容入口，按 recipe 分流
├── app/
│   ├── client.py                     # 每 session thin client
│   ├── protocol.py                   # relax-nemo-gym/v1 wire types
│   └── result.py                     # Gym terminal result -> Relax output
├── recipes/
│   ├── calendar/                     # 三步 Calendar recipe
│   │   ├── README.md / PITFAIL.md
│   │   └── prepare / start / run 脚本
│   ├── gsm8k/
│   │   ├── README.md / PITFAIL.md
│   │   ├── prepare_gsm8k.sh / start_gsm8k_gym.sh
│   │   ├── run-qwen3-4B-8xgpu-nemo-gym.sh
│   │   └── verify_gsm8k.py
│   ├── workplace-assistant/
│   │   ├── README.md / PITFAIL.md
│   │   ├── prepare_workplace_assistant.sh / start_workplace_assistant_gym.sh
│   │   ├── run-qwen3-4B-8xgpu-nemo-gym-workplace.sh
│   │   └── verify_workplace_assistant*.py
│   ├── multienv-math-workplace/      # math + workplace 逐行混合
│   │   ├── README.md
│   │   └── prepare / start / run 脚本
│   ├── reasoning-gym-cc/              # Claude Code + reasoning-gym
│   │   ├── README.md / configs/
│   │   └── prepare / start / verify / run 脚本
│   └── r2e-gym/
│       ├── README.md / PITFAIL.md
│       ├── prepare_r2e_gym.py / prepare_r2e_gym.sh
│       ├── start_r2e_gym_remote.sh   # Gym 与训练分离部署
│       ├── start_r2e_gym_local.sh    # Gym 与训练共用 Ray
│       ├── submit_r2e_gym.sh
│       ├── run-qwen3-4B-*xgpu-nemo-gym-r2e.sh
│       └── verify_r2e_gym_trial.py
├── scripts/
│   ├── convert_dataset.py            # 所有 recipe 共用的数据转换
│   ├── run_agent_app.sh              # Relax managed-command 入口
│   ├── run_gateway.sh                # 独立 Gateway 入口
│   └── run_training.sh               # Ray Job 内训练入口
├── service/
│   ├── app.py                        # Gateway HTTP API
│   ├── registry.py                   # admission、lease、deadline、终态
│   ├── callback_provider.py          # Gym -> Relax callback bridge
│   ├── run_adapter.py                # Gateway -> Gym /run adapter
│   ├── Dockerfile                    # 固定 Gym commit 和预建 venv
│   ├── nemo_gym_gateway_model/       # NeMo Gym model-server plugin
│   └── patches/                      # Workplace/R2E 固定 commit 兼容补丁
└── test/                             # 单元和协议测试
```

## 限制

- Gateway registry 当前是单进程内存状态，不能横向启动多个 Gateway worker，也不能在进程重启后恢复
  运行中 trial。
- 上游 environment 的通用 cancellation/cleanup contract 不完整；没有 cleanup probe 的环境会在
  中断时保守报告 `cleanup_unverified`。
- 单环境 recipe 通过进程环境选择 environment/config；multienv recipe 通过每行 metadata 动态路由。
- 数据下载脚本未固定 Hugging Face dataset revision；正式实验应记录数据文件 hash 或自行固定 revision。

## 致谢与引用

感谢 NVIDIA NeMo Gym 团队开源环境抽象、agent harness、数据、verifier 和大量可复用的 server
实现。本集成固定并适配其代码，但 NeMo Gym、Workplace Assistant、OpenHands 和 R2E-Gym 的原始
工作归各自项目与贡献者所有。

- [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym 文档](https://docs.nvidia.com/nemo/gym/)
- [Workplace Assistant 数据集](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant)
- [OpenAI GSM8K](https://github.com/openai/grade-school-math)
- [R2E-Gym 项目](https://github.com/R2E-Gym/R2E-Gym)
- [R2E-Gym-Lite 数据集](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Lite)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)

若在论文或公开报告中使用这些环境，请同时遵守各项目 license，并引用其官方论文或仓库中的
BibTeX。
