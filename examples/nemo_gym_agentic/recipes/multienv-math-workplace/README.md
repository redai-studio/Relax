# Math + Workplace Assistant 多环境训练

这个 recipe 使用 NeMo Gym 的两个 `simple_agent`，在同一个 Relax 训练任务中混合数学推理和
Workplace Assistant 工具调用。两个 agent 通过 OpenAI Responses API 向 Relax 请求 Qwen3-4B，
分别由 `math_with_judge` 和 `workplace_assistant` resource 计算 reward。

| 环境                  | Gateway config           | 交互方式                                   |
| --------------------- | ------------------------ | ------------------------------------------ |
| `math_with_judge`     | `math-with-judge-v1`     | 数学推理与答案验证                         |
| `workplace_assistant` | `workplace-assistant-v1` | 多轮 function tools，最多 6 个 agent steps |

以下步骤覆盖从零运行，命令均在 Relax 仓库根目录执行。`${GYM_HOST}` 是 Gym 主机 IP，
`${RELAX_HOST}` 是训练 Ray head IP；Gym 与训练 worker 之间需要双向可达。

```text
Prepared raw row: environment/config
  -> convert_dataset.py: input + metadata.environment/config
  -> Relax managed app/client.py
  -> NeMo Gym trial selects math or workplace
  -> simple_agent
  -> POST /ng-rollout/<rollout-id>/v1/responses
  -> Gateway forwards to Relax Responses API
  -> canonical messages/tools/chat_template_kwargs
  -> SessionForest
```

每次模型请求携带完整 `input` 历史。Qwen3-4B 使用 `qwen3` reasoning parser 和 `qwen` tool-call
parser。Workplace 还配置了 abort、force-cleanup 和 cleanup probe，用于清理远端环境状态。

## 构建 Gym 镜像

Callback 白名单配置 `NEMO_GYM_CALLBACK_ALLOWED_NETWORKS`，默认 `10.0.0.0/8`；填写实际覆盖
Relax callback IP 的网段，单个 IPv4/IPv6 地址使用 `/32` 或 `/128`。可传逗号分隔的 CIDR，或重复使用
启动参数 `--callback-network`。CIDR 按 callback URL 中的 IP 匹配，不解析域名，不接受 `/0`。

```bash
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"
export NEMO_GYM_IMAGE=relax-nemo-gym:multienv-dev

docker build \
  -f examples/nemo_gym_agentic/service/Dockerfile \
  -t "${NEMO_GYM_IMAGE}" \
  .
```

镜像固定 NeMo Gym commit `a85670eb167ba9b48cc53a36a070eed815e6c40d`，预装 math、workplace、
`simple_agent` 和 Gateway 依赖，并应用 Workplace cleanup patch。已有按此 Dockerfile 构建的兼容镜像时，
将 `NEMO_GYM_IMAGE` 设置为实际标签，可跳过构建；启动脚本不会自动构建或拉取镜像。

## 准备数据

```bash
export DATA_DIR=/共享目录/nemo-gym/multienv
export NEMO_GYM_SOURCE_DATA="${DATA_DIR}/multienv_train_raw.jsonl"
export RELAX_REPO_ROOT="$(pwd)"

bash examples/nemo_gym_agentic/recipes/multienv-math-workplace/prepare_multienv.sh
```

准备脚本读取上一节导出的 `NEMO_GYM_IMAGE`，通过 Docker 挂载仓库和数据目录。`RELAX_REPO_ROOT`
必须是 Docker daemon 可见的仓库绝对路径；开发容器内的路径与宿主机不同时，需要改为宿主机上的路径。
数据目录也必须同时对调用脚本的进程和 Docker daemon 可见。

数据来自以下两个 Hugging Face 数据集的 `train.jsonl`：

- `nvidia/Nemotron-RL-math-OpenMathReasoning`
- `nvidia/Nemotron-RL-agent-workplace_assistant`

准备脚本使用这些文件：

| 文件                                          | 用途                                                |
| --------------------------------------------- | --------------------------------------------------- |
| `${DATA_DIR}/math_with_judge_train.jsonl`     | math 原始数据；非空时复用，否则下载                 |
| `${DATA_DIR}/workplace_assistant_train.jsonl` | workplace 原始数据；非空时复用，否则下载            |
| `${DATA_DIR}/multienv_train_raw.jsonl`        | 添加环境路由并打乱后的混合数据，供训练脚本读取      |
| `${DATA_DIR}/multienv_train.jsonl`            | 训练启动时由 `convert_dataset.py` 生成的 Relax 数据 |

已有原始数据时，将文件放到上述对应位置后再运行准备脚本。默认选取前 2400 条 math 数据，以及所有
`ground_truth` 非空的 workplace 数据；两类数据都必须至少有一条。可通过
`MULTIENV_MATH_TARGET_LINES` 和 `MULTIENV_WORKPLACE_TARGET_LINES` 设置非负条数，`0` 表示全部。
脚本先筛选和截取，再以固定种子 `20260829` 打乱合并结果；数据总量取决于输入文件。

准备产物的 `environment`、`config`、`data_source` 位于每行顶层，训练转换后进入 `metadata`。
`NEMO_GYM_SOURCE_DATA` 应指向 `multienv_train_raw.jsonl`；直接传未标记的 workplace/math 原文件会
缺少环境路由，传已经转换的 `multienv_train.jsonl` 则会重复转换。准备脚本要求输出路径为绝对路径，
且以 `/multienv_train_raw.jsonl` 结尾。

## 启动 Gym

```bash
bash examples/nemo_gym_agentic/recipes/multienv-math-workplace/start_multienv_gym_remote.sh \
  --gym-host "${GYM_HOST}" \
  --callback-network "${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS}" \
  --port-base 29300 \
  --image "${NEMO_GYM_IMAGE}" \
  --repo-dir "${RELAX_REPO_ROOT}" \
  --max-concurrency 256
```

`--repo-dir` 用于挂载仓库，使 recipe 修改无需重建镜像即可生效。省略此参数时仍会读取
`RELAX_REPO_ROOT`；两者都未设置时使用镜像内的 recipe 文件。Gym 服务不读取训练 JSONL，
启动脚本无需也不接受 `--data-jsonl`。

`--max-concurrency` 同时设置 math 和 workplace **各自**的并发上限，默认各 256；不传此参数时，
可以分别通过 `MATH_MAX_CONCURRENCY`、`WORKPLACE_MAX_CONCURRENCY` 配置。

| 服务                        | 相对 `--port-base` | 默认端口 |
| --------------------------- | ------------------ | -------- |
| Relax Gateway model         | +0                 | 29300    |
| math simple agent           | +1                 | 29301    |
| math resource/verifier      | +2                 | 29302    |
| workplace simple agent      | +3                 | 29303    |
| workplace resource/verifier | +4                 | 29304    |
| NeMo Gym head server        | +5                 | 29305    |

`--port-base` 范围为 `1–65530`。Gym 使用独立私有 Ray，默认 GCS 端口为 `6384`，预留的 dashboard
端口为 `30365`（dashboard 默认关闭），CPU 数为 16；对应环境变量为 `GYM_RAY_PORT`、
`GYM_DASHBOARD_PORT`、`GYM_RAY_NUM_CPUS`，不随 `--port-base` 改变。

启动脚本等待 `/readyz` 返回 `ready=true`、匹配的 Gym commit 和 `active_trials=0`；`ready` 包含
janitor 与环境服务的就绪状态，并不要求历史 trial 总数为零。

启动后分别验证两个 resource 的评分行为；使用自定义基准端口时，将下方端口改为基准值加 2、加 4：

```bash
python examples/nemo_gym_agentic/recipes/gsm8k/verify_gsm8k.py \
  --resource-url "http://${GYM_HOST}:29302"
python examples/nemo_gym_agentic/recipes/workplace-assistant/verify_workplace_assistant.py \
  --resource-url "http://${GYM_HOST}:29304"
```

这两个检查直接调用 resource verifier，分别验证正确/错误数学答案、等价/错误 Workplace 最终状态，
不覆盖模型 callback、多轮 agent 或训练更新链路。

## 启动训练

在已有训练 Ray 集群的训练容器中执行，确保所有 worker 都能访问模型、数据和同一份 Relax 代码。
在该容器中重新设置数据目录：

```bash
export DATA_DIR=/共享目录/nemo-gym/multienv

RAY_ADDRESS="${RELAX_HOST}:6379" \
MODEL_DIR=/共享模型目录 \
DATA_DIR="${DATA_DIR}" \
NEMO_GYM_GATEWAY_PORT=29300 \
NEMO_GYM_SOURCE_DATA="${DATA_DIR}/multienv_train_raw.jsonl" \
GYM_HOST="${GYM_HOST}" \
RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
    examples/nemo_gym_agentic/recipes/multienv-math-workplace/run-qwen3-4B-8xgpu-nemo-gym-multienv.sh
```

`MODEL_DIR` 下必须包含 `Qwen3-4B/`。`DATA_DIR` 必填：若路径不以 `/nemo-gym/multienv` 结尾，
训练脚本会自动追加这一后缀，并将转换结果写到该目录的 `multienv_train.jsonl`。即使显式传入
`NEMO_GYM_SOURCE_DATA`，也仍需设置 `DATA_DIR`。

训练端的 `NEMO_GYM_GATEWAY_PORT` 必须等于 Gym 启动时的 `--port-base`。例如 Gym 使用
`--port-base 9500`，训练就设置 `NEMO_GYM_GATEWAY_PORT=9500`。

当前配置使用 Qwen3-4B、8 GPU colocate、Actor TP=4 和每个 rollout engine 2 GPU。每批读取 32 个
prompt，每个 prompt 采样 8 条，共 256 个 Session；每个 prompt 的 Group 内使用同一个环境。
共配置 200 个 rollout step。缩小实验时直接修改训练脚本中的 `--num-rollout`、`--rollout-batch-size`、
`--n-samples-per-prompt` 和 `--global-batch-size`，并保持 batch 配置匹配。

当前 multienv 训练脚本未配置 checkpoint 保存或 `--dump-details`，也未使用 `SAVE_DIR`。
提交日志写到 `log/qwen3-4B-8xgpu-nemo-gym-multienv-<时间>.log`；`RAY_NO_WAIT=1` 使提交命令返回后
不继续跟随训练日志。将 `JOB_ID` 设置为提交输出中的实际 job ID 后查看运行日志：

```bash
ray job logs --address="http://${RELAX_HOST}:8265" "${JOB_ID}"
```

正常线性历史使用 implicit export。若 agent 改写历史产生多个 committed leaf，无法唯一完成导出的
Session 会被丢弃并补采。出现 `NEMO_GYM_ENVIRONMENT must be set to a non-empty value` 时，先检查
训练输入是否为准备脚本生成的混合原始数据，而不是未标记的单环境原文件。
