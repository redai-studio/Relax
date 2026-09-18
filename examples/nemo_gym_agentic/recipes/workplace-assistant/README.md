# Workplace Assistant Recipe：有状态多轮工具训练

Workplace Assistant 是本目录首选的 tool-use correctness recipe。它覆盖工具 schema 透传、多轮模型
callback、工具结果回灌、每 session 独立状态，以及根据最终数据库状态判分。

接入踩坑见 [PITFAIL.md](PITFAIL.md)，总体协议见[顶层文档](../../README.md)。

## 任务概况

任务和 ground truth 来自 NVIDIA 发布的
[`nvidia/Nemotron-RL-agent-workplace_assistant`](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant)。
任务模拟邮件、日历、数据分析、项目管理和 CRM 等办公操作。

固定 NeMo Gym commit 的上游说明为：

- 5 组有状态数据库；
- 上游 README/dataset card 描述为 26 个工具；固定 commit 的内置 example 实际暴露 27 个 tool
  schema，因此运行时以每条任务的 `responses_create_params.tools` 为准；
- 690 个底层任务模板；
- simple agent 最多 6 个 step；
- 数据集包含 train 与 validation。Hugging Face 当前可见约 1255 条 train、545 条 validation；
  下载脚本没有固定 dataset revision，实际条数应以本次文件为准。

| 项目                 | 本 recipe 的值                                               |
| -------------------- | ------------------------------------------------------------ |
| NeMo Gym environment | `workplace_assistant`                                        |
| Gateway config       | `workplace-assistant-v1`                                     |
| Agent                | `workplace_assistant_simple_agent`                           |
| Verifier             | 在新环境执行模型动作和 ground truth，比较最终数据库状态      |
| Reward               | 状态等价为 1，不等价为 0                                     |
| 参考模型             | Qwen3-4B                                                     |
| 配置上下文           | 8K total context，2K response                                |
| 推荐模型规模         | 4B 用于链路 smoke；有意义的工具成功率建议从 7B/8B 级以上评估 |
| Sandbox              | 无 OCI/Apptainer；状态由 resource server 按 session 隔离     |

模型不必严格复现 ground-truth action trace。只读搜索后再执行正确写操作，只要最终状态等价，reward
仍应为 1。

## 组件和状态生命周期

```text
Relax managed session
  -> thin client POST /v1/trials
  -> Gateway :29000
  -> simple_agent :29001
       -> policy callback -> Relax Agentic Chat API :8000
       -> tool request -> workplace resource :29002
       -> tool result -> 下一轮 policy callback
       -> 最终 response -> verifier
  -> reward + metrics
  -> Relax rollout JSONL / optimizer
```

resource server 为每条请求创建独立数据库会话。集成 patch 维护
`rollout_id <-> resource session_id` 映射，并提供 `/cleanup/{rollout_id}`，让 Gateway 在完成、
取消或超时时确认状态已经释放。这个环境不需要 Docker-in-Docker。

## 0. 准备变量

在 Relax 仓库根目录执行：

Callback 白名单只配置 `NEMO_GYM_CALLBACK_ALLOWED_NETWORKS`，默认为 `10.0.0.0/8`，可覆盖为逗号分隔的 CIDR；
填写实际覆盖 Relax callback IP 的网段，单个 IPv4/IPv6 地址使用 `/32` 或 `/128`。
远程启动脚本也可重复传入 `--callback-network`。CIDR 按 URL 中的 IP 匹配，不解析域名，且不接受 `/0`。

```bash
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"
export REPO_ROOT="$(pwd)"
export RELAX_IMAGE="ghcr.io/redai-studio/relaxrl:latest"
export NEMO_GYM_IMAGE="relax-nemo-gym:a85670e"
export DATA_ROOT="/绝对路径/relax-nemo-data"
export MODEL_DIR="/绝对路径/models"
export GYM_HOST="<Gym 主机可路由 IP>"
export RELAX_HOST="<Relax Ray head 可路由 IP>"

export http_proxy="http://proxy.example.com:3128"   # 无代理时留空
export https_proxy="${http_proxy}"
export no_proxy="127.0.0.1,localhost,${GYM_HOST},${RELAX_HOST}"

mkdir -p "${DATA_ROOT}/nemo-gym" "${DATA_ROOT}/experiments" "${MODEL_DIR}"
```

## 1. 构建镜像

Relax 训练直接使用 `${RELAX_IMAGE}`。只需构建 Gym 镜像：

```bash
DOCKER_BUILDKIT=1 docker build \
  --network host \
  -f examples/nemo_gym_agentic/service/Dockerfile \
  --build-arg HTTP_PROXY="${http_proxy}" \
  --build-arg HTTPS_PROXY="${https_proxy}" \
  --build-arg NO_PROXY="${no_proxy}" \
  -t "${NEMO_GYM_IMAGE}" \
  .
```

Dockerfile 默认基于 `ghcr.io/redai-studio/relaxrl:latest`。使用其他已有 Relax tag 时，给上述命令
增加 `--build-arg RELAX_IMAGE="<image>"`；不需要构建 Relax 镜像。

镜像在构建阶段预建 Gateway、simple agent 和 Workplace resource venv，并应用 session cleanup
patch。运行时不应再安装依赖：

```bash
docker run --rm "${NEMO_GYM_IMAGE}" bash -lc '
  test "${NEMO_GYM_COMMIT}" = "a85670eb167ba9b48cc53a36a070eed815e6c40d"
  test -x /opt/nemo-gym/resources_servers/workplace_assistant/.venv/bin/python
  /usr/bin/python3 -c "import loguru, ray"
  ray serve --help >/dev/null
'
```

## 2. 下载模型

```bash
hf download Qwen/Qwen3-4B \
  --local-dir "${MODEL_DIR}/Qwen3-4B"

test -s "${MODEL_DIR}/Qwen3-4B/config.json"
test -s "${MODEL_DIR}/Qwen3-4B/tokenizer.json"
```

当前训练脚本只为 Qwen3-4B 提供 Megatron model config。换 8B/14B/32B 模型时，需要使用对应的
Relax model config 和训练脚本，不能只改目录名。

## 3. 下载和处理数据

### 3.1 先用镜像内置 example

不访问 Hugging Face，先验证 converter 和共享目录：

```bash
docker run --rm --network host \
  -v "${DATA_ROOT}:/data" \
  -e WORKPLACE_ASSISTANT_SPLIT=example \
  "${NEMO_GYM_IMAGE}" \
  bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/workplace-assistant/prepare_workplace_assistant.sh
```

生成：

```text
${DATA_ROOT}/nemo-gym/workplace_assistant_example.jsonl
${DATA_ROOT}/nemo-gym/workplace_assistant_example_relax.jsonl
```

### 3.2 下载正式 train

```bash
docker run --rm --network host \
  -v "${DATA_ROOT}:/data" \
  -e WORKPLACE_ASSISTANT_SPLIT=train \
  -e HTTP_PROXY="${http_proxy}" \
  -e HTTPS_PROXY="${https_proxy}" \
  -e NO_PROXY="${no_proxy}" \
  "${NEMO_GYM_IMAGE}" \
  bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/workplace-assistant/prepare_workplace_assistant.sh
```

生成：

```text
${DATA_ROOT}/nemo-gym/workplace_assistant_train.jsonl
${DATA_ROOT}/nemo-gym/workplace_assistant_train_relax.jsonl
```

第一份是 Gym 原始任务，也是训练脚本的 `NEMO_GYM_SOURCE_DATA`；第二份只用于检查 Relax
`input + metadata` 转换。

检查非空 JSONL 记录和关键字段：

```bash
test -s "${DATA_ROOT}/nemo-gym/workplace_assistant_train.jsonl"
test -s "${DATA_ROOT}/nemo-gym/workplace_assistant_train_relax.jsonl"

awk 'NF { count++ } END { print count, FILENAME }' \
  "${DATA_ROOT}/nemo-gym/workplace_assistant_train.jsonl"
awk 'NF { count++ } END { print count, FILENAME }' \
  "${DATA_ROOT}/nemo-gym/workplace_assistant_train_relax.jsonl"

jq -c '{
  id,
  category,
  messages:(.responses_create_params.input|length),
  tools:(.responses_create_params.tools|length),
  ground_truth:(.ground_truth|length)
}' "${DATA_ROOT}/nemo-gym/workplace_assistant_train.jsonl" | head -1

jq -c '{
  messages:(.input|length),
  tools:(.metadata.tools|length),
  category:.metadata.category,
  ground_truth:(.metadata.ground_truth|length)
}' "${DATA_ROOT}/nemo-gym/workplace_assistant_train_relax.jsonl" | head -1
```

不要删除 tools 或 ground truth。

### 3.3 下载 validation

```bash
docker run --rm --network host \
  -v "${DATA_ROOT}:/data" \
  -e WORKPLACE_ASSISTANT_SPLIT=validation \
  -e HTTP_PROXY="${http_proxy}" \
  -e HTTPS_PROXY="${https_proxy}" \
  -e NO_PROXY="${no_proxy}" \
  "${NEMO_GYM_IMAGE}" \
  bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/workplace-assistant/prepare_workplace_assistant.sh
```

把 validation 保留给 base/trained checkpoint 的同配置对比，不要混入 train。

## 4. 启动 NeMo Gym

在 Gym 主机从 Relax checkout 启动服务。`--repo-dir` 是 Docker daemon 所在主机看到的
checkout 绝对路径；普通 Docker 通常就是当前目录，Docker-in-Docker 则应传外层 Docker host
上的路径：

```bash
bash examples/nemo_gym_agentic/recipes/workplace-assistant/start_workplace_assistant_gym_remote.sh \
  --gym-host "${GYM_HOST}" \
  --callback-network "${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS}" \
  --image "${NEMO_GYM_IMAGE}" \
  --repo-dir "${REPO_ROOT}" \
  --max-concurrency 8
```

这个单一入口负责替换同名 Workplace 容器、启动并等待 ready。它不会操作 R2E-Gym 容器。默认启动
Gym 私有 Ray `:6382`、Gateway `:29000`、agent `:29001`、resource/verifier `:29002` 和
Gym head `:29003`。

另一个终端等待：

```bash
until curl --noproxy "*" -fsS "http://${GYM_HOST}:29000/readyz" |
  jq -e '.ready == true' >/dev/null; do
  sleep 2
done

curl --noproxy "*" -fsS "http://${GYM_HOST}:29000/readyz" | jq .
```

## 5. 在训练前验证环境

### 5.1 Verifier 语义

该脚本先做一次只读搜索，再做正确回复，确认最终状态等价仍为 1；同时确认错误回复为 0：

```bash
docker run --rm --network host \
  "${NEMO_GYM_IMAGE}" \
  /opt/nemo-gym/resources_servers/workplace_assistant/.venv/bin/python \
  /opt/relax-integration/examples/nemo_gym_agentic/recipes/workplace-assistant/verify_workplace_assistant.py \
  --resource-url "http://${GYM_HOST}:29002"
```

必须看到：

```text
Workplace Assistant verifier contract passed: equivalent_reward=1.0 incorrect_reward=0.0
```

### 5.2 完整 deterministic trial

这个脚本启动临时 callback，依次返回搜索工具、回复工具和最终文本，验证三次 callback 共享历史：

```bash
docker exec nemo-gym-workplace \
  /opt/nemo-gym/.venv/bin/python \
  /opt/relax-integration/examples/nemo_gym_agentic/recipes/workplace-assistant/verify_workplace_assistant_trial.py
```

必须看到：

```text
Workplace Assistant trial passed: reward=1.0 tool_calls=2 tool_outputs=2 callbacks=3
```

这一步通过才说明 Gateway、tool result 回灌、session 和 verifier 一起工作。

## 6. 启动或连接 Relax Ray

已有平台 Ray：

```bash
export RAY_ADDRESS="${RELAX_HOST}:6379"
ray status --address="${RAY_ADDRESS}"
```

不要在平台管理的 Ray 上再次 `ray start`。

若从零自建单节点 8-GPU 训练容器：

```bash
docker run -dit \
  --name relax-train \
  --network host \
  --gpus all \
  --shm-size 128g \
  -v "${REPO_ROOT}:/workspace/Relax" \
  -v "${DATA_ROOT}:/data" \
  -v "${MODEL_DIR}:/models" \
  "${RELAX_IMAGE}" bash

docker exec relax-train ray start --head \
  --node-ip-address="${RELAX_HOST}" \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=8 \
  --disable-usage-stats
```

## 7. 启动训练

### 平台或已有 Relax Ray

先跑一条 sample 的链路 smoke：

```bash
export RAY_ADDRESS="${RELAX_HOST}:6379"
export MODEL_DIR="/训练环境中包含 Qwen3-4B 的父目录"
export GYM_HOST="<Gym 主机可路由 IP>"
export NEMO_GYM_SOURCE_DATA="/训练环境可见的/workplace_assistant_train.jsonl"
export NEMO_GYM_DATA_LIMIT=1
export NEMO_GYM_NUM_ROLLOUT=1
export NEMO_GYM_N_SAMPLES_PER_PROMPT=1
export NEMO_GYM_GLOBAL_BATCH_SIZE=1
export EXP_DIR="/训练环境可写的/experiments/nemo-gym-workplace"

bash scripts/entrypoint/ray-job.sh \
  examples/nemo_gym_agentic/recipes/workplace-assistant/run-qwen3-4B-8xgpu-nemo-gym-workplace.sh
```

脚本直接使用当前 checkout 的绝对路径，因此所有 Ray 节点必须以同一个绝对路径挂载该 checkout。
不要设置 `WORKING_DIR`，否则 Ray 会打包上传整个仓库。默认 submission ID 带启动时间、进程号和
随机后缀；需要停止或查询任务时，使用提交输出中的实际 ID，不需要手工指定固定 ID。

默认从 `RAY_ADDRESS=<head>:6379` 推导 Dashboard 为 `http://<head>:8265`。如果平台 Dashboard
使用其他地址，显式设置：

```bash
export RAY_DASHBOARD_ADDRESS="http://<ray-dashboard-host>:<port>"
```

不要设置 `RAY_NO_WAIT`；命令会前台跟随训练 job。

### 自建 `relax-train` 容器

```bash
docker exec \
  -e RAY_ADDRESS="${RELAX_HOST}:6379" \
  -e MODEL_DIR="/models" \
  -e GYM_HOST="${GYM_HOST}" \
  -e NEMO_GYM_SOURCE_DATA="/data/nemo-gym/workplace_assistant_train.jsonl" \
  -e NEMO_GYM_DATA_LIMIT=1 \
  -e NEMO_GYM_NUM_ROLLOUT=1 \
  -e NEMO_GYM_N_SAMPLES_PER_PROMPT=1 \
  -e NEMO_GYM_GLOBAL_BATCH_SIZE=1 \
  -e EXP_DIR="/data/experiments/nemo-gym-workplace" \
  relax-train bash -lc '
    cd /workspace/Relax
    exec bash scripts/entrypoint/ray-job.sh \
      examples/nemo_gym_agentic/recipes/workplace-assistant/run-qwen3-4B-8xgpu-nemo-gym-workplace.sh
  '
```

链路通过后，真正进行 GRPO 训练至少设置：

```bash
export NEMO_GYM_N_SAMPLES_PER_PROMPT=4
export NEMO_GYM_GLOBAL_BATCH_SIZE=4
export NEMO_GYM_DATA_LIMIT=32
```

先保持 `WORKPLACE_ASSISTANT_MAX_CONCURRENCY=8`，再根据 CPU、内存和服务延迟逐步扩容。

## 8. 查看和验收结果

```bash
export RESULT_DIR="${EXP_DIR}/Qwen3-4B_mcore_8xgpu/rollout_result/train"

find "${RESULT_DIR}" -maxdepth 1 -name '*.jsonl' -type f -print
jq -c '{
  rollout_id,
  status,
  reward,
  agent_turns,
  prompt_token_count,
  response_token_count
}' "${RESULT_DIR}"/*.jsonl

jq -r '.response' "${RESULT_DIR}"/*.jsonl |
  rg '<tool_call>|email_|calendar_|analytics_|company_directory_|project_management_|customer_relationship_manager_'
```

一次正确的 model rollout 至少满足：

- dump 存在，status 为 `completed`，或 `truncated` 有明确预算原因；
- `agent_turns > 1`；
- response 中有真实工具调用和工具结果上下文；
- reward 是数值；reward=0 可能只是模型动作错误，不自动等于链路失败；
- Gym `active_trials` 最终归零；
- 要声称 optimizer step 跑通，还要检查 Actor 没有 OOM/traceback 并出现 step metrics。

对比模型效果时，在同一 validation 文件、采样配置、重复次数和 verifier 版本下分别运行 base 与
trained checkpoint，报告平均 reward/pass rate。只展示一条成功轨迹不能证明训练有效。

## 9. 停止

```bash
ray job list --address="http://${RELAX_HOST}:8265"
ray job stop --address="http://${RELAX_HOST}:8265" "<提交输出中的 submission_id>"

docker stop nemo-gym-workplace
```
