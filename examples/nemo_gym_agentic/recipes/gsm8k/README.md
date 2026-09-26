# GSM8K Recipe：无工具协议 Smoke

本 recipe 用 GSM8K 快速验证 Relax 与 NeMo Gym 之间的 trial、callback、reward 和训练输入转换。
它没有工具调用、sandbox 或持久状态，适合作为接入第一步，不适合证明 agentic tool-use 链路正确。

接入踩坑见 [PITFAIL.md](PITFAIL.md)，总体架构见[顶层文档](../../README.md)。

## 任务概况

GSM8K 来自 [OpenAI grade-school-math](https://github.com/openai/grade-school-math)，包含约 8.5K 道
小学数学文字题，原始划分约为 7.5K train 和 1.3K test。固定的 NeMo Gym commit 使用 test split
的 1319 题，经过其 prepare 脚本清理 calculator annotation、修正已知答案并转换字段。

| 项目                 | 本 recipe 的值                                              |
| -------------------- | ----------------------------------------------------------- |
| NeMo Gym environment | `gsm8k`                                                     |
| Gateway config       | `gsm8k-v1`                                                  |
| Agent                | `gsm8k_math_with_judge_simple_agent`                        |
| Verifier             | `math_with_judge`，比较 `\boxed{}` 中的答案                 |
| 数据量               | 1319 条 test；smoke 默认只取少量                            |
| 参考模型             | Qwen3-4B                                                    |
| 配置上下文           | 8K total context，2K response                               |
| 推荐模型规模         | 4B 足够验证链路；数学能力研究应自行选择模型和独立 train set |
| Sandbox              | 无                                                          |

重要：本 recipe 读取的是 benchmark test split。它只应用于 E2E smoke，不能当作无污染的正式训练集，
也不能用它宣称模型泛化提升。

## 组件交互

```text
Relax managed session
  -> thin client POST /v1/trials
  -> Gateway :28100
  -> simple_agent :28101
  -> math_with_judge :28102
  -> Gateway callback
  -> Relax Agentic Chat API :8000
  -> verifier reward 0/1
  -> Relax rollout JSONL / optimizer
```

因为没有工具，通常只有一次模型 callback。dump 中没有 `<tool_call>` 是正确结果。

## 0. 准备变量

以下命令都从 Relax 仓库根目录执行：

Callback 白名单只配置 `NEMO_GYM_CALLBACK_ALLOWED_NETWORKS`，默认为 `10.0.0.0/8`，可覆盖为逗号分隔的 CIDR；
填写实际覆盖 Relax callback IP 的网段，单个 IPv4/IPv6 地址使用 `/32` 或 `/128`。
CIDR 按 URL 中的 IP 匹配，不解析域名，且不接受 `/0`。

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

`GYM_HOST` 和 `RELAX_HOST` 只填裸 host/IP，不带协议或端口。

## 1. 构建镜像

Relax 训练直接使用 `${RELAX_IMAGE}`。只需构建固定 NeMo Gym commit 的服务镜像：

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

检查镜像：

```bash
docker run --rm "${NEMO_GYM_IMAGE}" bash -lc '
  test "${NEMO_GYM_COMMIT}" = "a85670eb167ba9b48cc53a36a070eed815e6c40d"
  /opt/nemo-gym/.venv/bin/gym --help >/dev/null
  /usr/bin/python3 -c "import loguru, ray"
  ray serve --help >/dev/null
'
```

## 2. 下载模型

训练脚本要求 `${MODEL_DIR}/Qwen3-4B/`：

```bash
hf download Qwen/Qwen3-4B \
  --local-dir "${MODEL_DIR}/Qwen3-4B"

test -s "${MODEL_DIR}/Qwen3-4B/config.json"
test -s "${MODEL_DIR}/Qwen3-4B/tokenizer.json"
```

如果模型已由内部模型仓库准备，只需保证最终目录名和层级相同。

## 3. 下载并处理数据

prepare 脚本会调用 pinned Gym 的 benchmark prepare，然后生成：

```text
${DATA_ROOT}/nemo-gym/gsm8k_benchmark.jsonl
${DATA_ROOT}/nemo-gym/gsm8k_relax.jsonl
```

执行：

```bash
docker run --rm --network host \
  -v "${DATA_ROOT}:/data" \
  -e HTTP_PROXY="${http_proxy}" \
  -e HTTPS_PROXY="${https_proxy}" \
  -e NO_PROXY="${no_proxy}" \
  "${NEMO_GYM_IMAGE}" \
  bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/gsm8k/prepare_gsm8k.sh
```

检查非空记录数和 schema：

```bash
test -s "${DATA_ROOT}/nemo-gym/gsm8k_benchmark.jsonl"
test -s "${DATA_ROOT}/nemo-gym/gsm8k_relax.jsonl"

awk 'NF { count++ } END { print count, FILENAME }' \
  "${DATA_ROOT}/nemo-gym/gsm8k_benchmark.jsonl"
awk 'NF { count++ } END { print count, FILENAME }' \
  "${DATA_ROOT}/nemo-gym/gsm8k_relax.jsonl"

jq -c '{question,expected_answer,reference_solution}' \
  "${DATA_ROOT}/nemo-gym/gsm8k_benchmark.jsonl" | head -1
jq -c '{messages:(.input|length),expected_answer:.metadata.expected_answer}' \
  "${DATA_ROOT}/nemo-gym/gsm8k_relax.jsonl" | head -1
```

训练时传入原始 `gsm8k_benchmark.jsonl`，不是 `_relax.jsonl`。`run_training.sh` 会按本次
`NEMO_GYM_DATA_LIMIT` 重新生成实验输入。

## 4. 启动 NeMo Gym

在 Gym 主机启动前台容器：

```bash
docker run --rm \
  --name nemo-gym-gsm8k \
  --network host \
  --shm-size 12g \
  -e GYM_HOST="${GYM_HOST}" \
  -e NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS}" \
  "${NEMO_GYM_IMAGE}" \
  bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/gsm8k/start_gsm8k_gym.sh
```

脚本会启动独立 Gym Ray `:6381` 和 `28100`—`28103` 服务。不要给它传 Relax 的
`RAY_ADDRESS`。

另一个终端等待完整 graph：

```bash
until curl --noproxy "*" -fsS "http://${GYM_HOST}:28100/readyz" |
  jq -e '.ready == true' >/dev/null; do
  sleep 2
done

curl --noproxy "*" -fsS "http://${GYM_HOST}:28100/readyz" | jq .
```

## 5. 先验证 verifier

直接向 resource server 发送正确和错误答案：

```bash
docker run --rm --network host \
  "${NEMO_GYM_IMAGE}" \
  /opt/nemo-gym/resources_servers/math_with_judge/.venv/bin/python \
  /opt/relax-integration/examples/nemo_gym_agentic/recipes/gsm8k/verify_gsm8k.py \
  --resource-url "http://${GYM_HOST}:28102"
```

必须看到：

```text
GSM8K verifier contract passed: correct_reward=1.0 incorrect_reward=0.0
```

`/readyz` 通过但 verifier 不通过时不要启动训练。

## 6. 启动或连接 Relax Ray

平台已经提供 Ray Job/cluster 时，只检查：

```bash
export RAY_ADDRESS="${RELAX_HOST}:6379"
ray status --address="${RAY_ADDRESS}"
```

不要在平台管理的集群上再次执行 `ray start`。

若从零自建单节点训练容器：

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

Gym 和 Relax 必须是两套 Ray。所有 Relax worker 需要访问 `${GYM_HOST}:28100`，Gym 主机需要访问
`${RELAX_HOST}:8000`。

## 7. 启动训练

### 平台或已有 Relax Ray

在 Relax Ray head 的仓库根目录执行。下面是可快速完成的一条 sample smoke：

```bash
export RAY_ADDRESS="${RELAX_HOST}:6379"
export MODEL_DIR="/训练环境中包含 Qwen3-4B 的父目录"
export GYM_HOST="<Gym 主机可路由 IP>"
export NEMO_GYM_SOURCE_DATA="/训练环境可见的/gsm8k_benchmark.jsonl"
export NEMO_GYM_DATA_LIMIT=1
export NEMO_GYM_NUM_ROLLOUT=1
export NEMO_GYM_N_SAMPLES_PER_PROMPT=1
export NEMO_GYM_GLOBAL_BATCH_SIZE=1
export EXP_DIR="/训练环境可写的/experiments/nemo-gym-gsm8k"

bash scripts/entrypoint/ray-job.sh \
  examples/nemo_gym_agentic/recipes/gsm8k/run-qwen3-4B-8xgpu-nemo-gym.sh
```

所有 Ray 节点必须以同一个绝对路径看到当前 Relax checkout；不要设置 `WORKING_DIR`。默认
submission ID 唯一，查询和停止任务时使用提交输出中的实际 ID。

不要设置 `RAY_NO_WAIT`，命令会前台跟随内层训练 job 日志。

### 自建 `relax-train` 容器

```bash
docker exec \
  -e RAY_ADDRESS="${RELAX_HOST}:6379" \
  -e MODEL_DIR="/models" \
  -e GYM_HOST="${GYM_HOST}" \
  -e NEMO_GYM_SOURCE_DATA="/data/nemo-gym/gsm8k_benchmark.jsonl" \
  -e NEMO_GYM_DATA_LIMIT=1 \
  -e NEMO_GYM_NUM_ROLLOUT=1 \
  -e NEMO_GYM_N_SAMPLES_PER_PROMPT=1 \
  -e NEMO_GYM_GLOBAL_BATCH_SIZE=1 \
  -e EXP_DIR="/data/experiments/nemo-gym-gsm8k" \
  relax-train bash -lc '
    cd /workspace/Relax
    exec bash scripts/entrypoint/ray-job.sh \
      examples/nemo_gym_agentic/recipes/gsm8k/run-qwen3-4B-8xgpu-nemo-gym.sh
  '
```

一条 sample 的 GRPO advantage 为 0，只验证链路。真正做 GRPO 更新至少设置：

```bash
export NEMO_GYM_N_SAMPLES_PER_PROMPT=4
export NEMO_GYM_GLOBAL_BATCH_SIZE=4
```

并使用独立 train data；不要扩大 GSM8K test 的训练范围。

## 8. 查看和验收结果

训练输出目录由脚本拼接为：

```text
${EXP_DIR}/Qwen3-4B_mcore_8xgpu/rollout_result/train/*.jsonl
```

检查：

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
```

预期：

- verifier contract 的正确/错误 reward 分别为 1/0；
- rollout dump 存在且 reward 是数值；
- GSM8K 没有 tool call；
- Gateway 最终 `active_trials == 0`；
- 若要声称“训练 step 跑通”，Actor 日志还必须没有 OOM/traceback，并出现 step metrics。

只看 Ray Job `SUCCEEDED` 不够；Actor 异常可能先写入子进程日志，而 driver 仍正常退出。

## 9. 停止

训练 job 使用其 submission ID 停止：

```bash
ray job list --address="http://${RELAX_HOST}:8265"
ray job stop --address="http://${RELAX_HOST}:8265" "<提交输出中的 submission_id>"
```

Gym 容器前台运行时按 `Ctrl-C`，或另一个终端执行：

```bash
docker stop nemo-gym-gsm8k
```
