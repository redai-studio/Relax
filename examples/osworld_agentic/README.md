# OSWorld agentic RL

使用 OSWorld 桌面任务训练 Qwen3-VL-8B-Thinking，当前提供 8 GPU 的 GRPO outcome 配方。
以下命令均在 Relax 仓库根目录执行。

## 0. 下载代码和准备目录

如果还没有代码，先准备 OSWorld Nex fork 和包含本配方的 Relax 分支。MCP
源码已经随 OSWorld fork 提供，不需要再下载 `toolcua_mcp_swap`。
将 `<RELAX_REPO_URL>` 替换为有权限访问的 Relax Git 地址：

```bash
mkdir -p /work
cd /work
git clone https://github.com/fxsc03/OSWorld-main-nex.git OSWorld-main-nex
git clone <RELAX_REPO_URL> Relax
cd Relax
git checkout feat/fengxiaoshi/osworld
cd /work
```

目录建议保持如下布局：

```text
/work/
├── OSWorld-main-nex/
│   └── mcp/
└── Relax/
```

先按 [Relax 安装说明](../../docs/zh/guide/installation.md) 准备含 PyTorch、
Megatron 和 SGLang 的训练环境。下面的 pip 命令补充应用依赖，不会构建完整的
GPU 训练环境。在所有 Ray 节点使用相同的训练环境安装：

```bash
python -m pip install -r /work/Relax/requirements.txt
python -m pip install -r /work/OSWorld-main-nex/requirements.txt qwen-agent
```

先确认 HybridAgentLocal 可以导入：

```bash
cd /work/OSWorld-main-nex
python -c "from mm_agents.hybrid_agent_local import HybridAgentLocal; print('HybridAgentLocal OK')"
cd /work/Relax
```

## 1. 环境配置

先准备好 Relax 训练环境、Megatron 和可用的 Ray 集群。OSWorld 侧需要：

- **配套的外部 OSWorld 版本**：包含 Nex provider、MCP 支持和 `HybridAgentLocal`。
  标准上游 OSWorld checkout 不能直接用于本配方。
- OSWorld fork 内置的 MCP 源码，以及可用的 Nex workspace、桌面模板和 API key。
- 按该外部版本的安装说明安装 OSWorld 依赖，并额外安装
  `qwen-agent`（`HybridAgentLocal` 的 GUI 工具定义依赖它）。当前
  [run_agent_app.sh](run_agent_app.sh) 直接使用运行环境中的 `python`，不会自动切换 conda 环境。

外部文件按以下布局准备，`OSWORLD_REPO` 指向其中的 `OSWorld-main-nex/`。
`agents/`、`prompts/` 和 `tools/` 已经随 Nex fork 提供，不需要再从其他
agent 仓库复制：

```text
external_root/
├── OSWorld-main-nex/
│   ├── desktop_env/
│   ├── mm_agents/hybrid_agent_local.py
│   ├── agents/tool_retriever.py
│   ├── prompts/policy_hybrid.py
│   ├── tools/tools_registry.json
│   ├── mcp/
│   └── evaluation_examples/
└── Relax/
```

复制环境配置模板，已有 `env.local.sh` 时保留原文件：

```bash
printf '%s\n' '/examples/osworld_agentic/env.local.sh' >> "$(git rev-parse --git-path info/exclude)"
cp -n examples/osworld_agentic/env.sh examples/osworld_agentic/env.local.sh
chmod 600 examples/osworld_agentic/env.local.sh
```

编辑 `examples/osworld_agentic/env.local.sh`，将占位值替换为实际配置：

- `MEGATRON`：Megatron-LM 源码目录。
- `MODEL_DIR`：模型根目录，下面应有 `Qwen3-VL-8B-Thinking/`。
- `DATA_DIR`：准备好的 parquet 的根目录，训练读取其下的 `osworld/train.parquet`。
- `SAVE_DIR`：本次实验输出目录。从 base 模型开始训练时使用新目录。
- `RAY_JOB_ADDRESS`：已有 Ray 集群的 Jobs 地址。
- OSWorld 路径、Nex 连接信息：见第 3 节，准备数据前先填好。

启动脚本先读取 `env.local.sh`，再用 [env.sh](env.sh) 补齐默认值。
所有 Ray 节点需能访问相同路径下的 Relax、Megatron、OSWorld（含 `mcp/`）、模型、数据及本地配置。

`MCP_SRC_ROOT` 默认等于 `OSWORLD_REPO`，代码会读取 `${MCP_SRC_ROOT}/mcp`。
已有 `env.local.sh` 如果还指定旧的 MCP 路径，请在 `OSWORLD_REPO` 设置之后改为：

```bash
export MCP_SRC_ROOT="${OSWORLD_REPO}"
```

这里填仓库根目录，不能填 `${OSWORLD_REPO}/mcp`，否则会多拼接一级 `mcp/`。
沙箱镜像仍需预装 FastMCP、LibreOffice/UNO、桌面应用和工具所需依赖；源码注入不会
自动安装这些软件。`pip install qwen-agent` 安装在 agent 所用 Python 环境中，
也不会给远端沙箱安装 MCP server。完整目录和 guest 依赖见 OSWorld 的 `mcp/README.md`。

## 2. 下载 / 准备数据集

任务数据来自配套 OSWorld 版本中的 `evaluation_examples/`。获取外部 OSWorld 时，
一并准备好这个目录中的任务 JSON。训练任务列表在
[scripts/train_tasks.txt](scripts/train_tasks.txt)，每行一个任务 ID，当前共 12 个任务。

填好 `env.local.sh` 后，将任务 JSON 转为训练用 parquet：

```bash
source examples/osworld_agentic/env.local.sh
source examples/osworld_agentic/env.sh

python3 examples/osworld_agentic/scripts/prepare_data.py \
    --input-dir "${OSWORLD_REPO}/evaluation_examples" \
    --output-dir "${DATA_DIR}/osworld" \
    --train-manifest examples/osworld_agentic/scripts/train_tasks.txt \
    --eval-manifest examples/osworld_agentic/scripts/train_tasks.txt
```

产出 `${DATA_DIR}/osworld/train.parquet` 和 `${DATA_DIR}/osworld/eval.parquet`。
当前启动脚本只读取 `train.parquet`，不运行独立 eval；任务执行结束后仍会由
OSWorld evaluator 评分。上述命令的两个 parquet 使用同一份任务列表。

更换任务时，修改 `scripts/train_tasks.txt`，然后重新执行上述命令覆盖 parquet。
修改源任务 JSON 后也要重新生成。启动脚本仅在 `train.parquet` 不存在或为空时自动准备数据，
不会刷新已有文件。检查生成日志中的任务条数；出现 `unknown task_id` 时，说明列表中的
ID 没有在输入目录找到，对应任务会被跳过。

## 3. OSWorld 参数在哪里填

### 路径与 Nex 连接：`env.local.sh`

编辑 `examples/osworld_agentic/env.local.sh`：

| 参数                | 填写内容                                                                |
| ------------------- | ----------------------------------------------------------------------- |
| `OSWORLD_REPO`      | 配套 OSWorld 源码目录，内含 `desktop_env/` 和 `evaluation_examples/`    |
| `OSWORLD_CACHE_DIR` | OSWorld 缓存目录；默认 `${OSWORLD_REPO}/cache`，需可写                  |
| `MCP_SRC_ROOT`      | MCP 根目录；默认填 `${OSWORLD_REPO}`（其下有 `mcp/`），可覆盖为兼容实现 |
| `NEX_API_BASE`      | Nex 管理 API 根地址，包含 `/api/v1`，如 `https://<nex-host>/api/v1`     |
| `NEX_WORKSPACE_ID`  | 可用的 Nex workspace ID                                                 |
| `NEX_TEMPLATE`      | 该 workspace 可用的 OSWorld 桌面模板名称                                |
| `NEX_API_KEY`       | Nex API key，需自行在本地配置中添加，或从私有凭据文件加载               |
| `download_proxy`    | 可选，仅供 evaluator 下载资源使用的代理                                 |

`run_agent_app.sh` 使用 Ray worker 环境中的 `python`。只在提交端激活 venv
不会改变已启动的远端 Ray worker 解释器；第 0 节的依赖检查须在 worker 环境也通过。

例如，在本地配置中补上模板未提供的 key：

```bash
export NEX_API_KEY="<填写你的 Nex API key>"
```

访问 Nex guest 的请求需要直连，不要让 `http_proxy` / `https_proxy` 等全局代理接管。
密钥保存在已忽略的 `env.local.sh` 或私有凭据文件中。

`NEX_API_BASE` 用于 workspace、模板和沙箱生命周期管理，请向平台管理员取得地址。
客户端以 `x-api-key` 请求头传递 `NEX_API_KEY`。它不是浏览器桌面地址，也不是
LLM 的 `/v1/chat/completions` 地址；模型端点由 Relax 为每个 session 注入。
如果平台提供模板 ID，也可以设置 `NEX_TEMPLATE_ID`，优先于 `NEX_TEMPLATE`。
下面两个现有 smoke 脚本仍要求设置 `NEX_TEMPLATE`。

开始训练前，在可访问 Nex 的 worker 环境中检查桌面和 MCP。后两个命令会创建并销毁
真实沙箱，使用 workspace 配额；它们不需要加载 LLM：

```bash
source examples/osworld_agentic/env.local.sh
source examples/osworld_agentic/env.sh
cd "${OSWORLD_REPO}"
python nex_env_check.py
python nex_task_smoke.py
python nex_mcp_smoke.py
cd /work/Relax
```

依次确认导入依赖通过、任务 setup/截图/evaluator 可用、MCP 工具列表非空且调用成功。
桌面 smoke 没有让 agent 完成任务，所以 evaluator 得分为 `0` 不等于链路失败。
MCP 启动失败时检查沙箱内 `/tmp/mcp_server.log` 和镜像依赖。

### 任务执行配置：`app/osworld_config.yaml`

直接编辑 [app/osworld_config.yaml](app/osworld_config.yaml)，后续启动的 session 会读取它：

| 参数                                              | 当前值               | 用途                             |
| ------------------------------------------------- | -------------------- | -------------------------------- |
| `max_turns`                                       | `30`                 | 每个任务最多执行的回合数         |
| `max_image_history_length`                        | `2`                  | 保留的历史截图数量               |
| `context_policy`                                  | `baseline`           | 对话上下文策略                   |
| `temperature` / `top_p`                           | `1.0` / `1.0`        | agent 请求的采样参数             |
| `max_tokens`                                      | `8192`               | 单次回复的 token 上限            |
| `llm_call_timeout_s`                              | `600`                | 单次模型请求的超时秒数           |
| `max_llm_call_retries` / `llm_call_backoff_s`     | `3` / `5`            | 请求尝试次数上限与重试等待秒数   |
| `provider_name` / `snapshot_name`                 | `nex` / `init_state` | 桌面环境 provider 与默认快照名称 |
| `screen_width` / `screen_height`                  | `1920` / `1080`      | 桌面分辨率                       |
| `headless` / `require_a11y_tree`                  | `true` / `false`     | 无头模式与是否请求无障碍树       |
| `tool_retriever_backend` / `tool_retriever_top_k` | `bm25` / `18`        | 工具检索方式与数量               |

调整 `max_turns` 时，同步修改训练脚本中的 `OSWORLD_MAX_STEPS_REF`，保持奖励计算的
步数基准一致；调整 `max_tokens` 时，同步检查脚本中的 `--rollout-max-response-len`。

## 4. 原本参数在哪里修改

训练参数仍集中在 [run_qwen3vl_8B_osworld_outcome.sh](run_qwen3vl_8B_osworld_outcome.sh)。
路径、Nex 凭据和运行时环境放在 `env.local.sh`；单个 session 的桌面和模型请求
参数放在 [app/osworld_config.yaml](app/osworld_config.yaml)；训练任务列表放在
[scripts/train_tasks.txt](scripts/train_tasks.txt)。

### 训练脚本参数

| 参数                          |         默认值 | 修改位置和作用                                                                           |
| ----------------------------- | -------------: | ---------------------------------------------------------------------------------------- |
| `ROLLOUT_BATCH_SIZE`          |           `12` | 直接修改 `run_qwen3vl_8B_osworld_outcome.sh`；每个 rollout 的任务组数。                  |
| `N_SAMPLES_PER_PROMPT`        |            `8` | 直接修改启动脚本；每个任务组生成的 session 数。默认每轮为 `12 x 8 = 96` 个 session。     |
| `OSWORLD_UPDATES_PER_ROLLOUT` |            `3` | 环境变量或启动脚本；每轮 rollout 切分的 optimizer update 数，只允许 `1`、`2`、`3`、`6`。 |
| `OSWORLD_LEARNING_RATE`       |         `1e-6` | 环境变量或启动脚本；Adam 学习率。                                                        |
| `NUM_ROLLOUT`                 |           `10` | 环境变量或启动脚本；rollout 总轮数。                                                     |
| `OSWORLD_LR_DECAY_ITERS`      | `45 x updates` | 环境变量或启动脚本；cosine 学习率衰减的 optimizer update 总数。                          |

`ROLLOUT_BATCH_SIZE` 和 `N_SAMPLES_PER_PROMPT` 当前是脚本中的固定值；其余四项
支持环境变量覆盖。例如：

```bash
OSWORLD_UPDATES_PER_ROLLOUT=1 \
OSWORLD_LEARNING_RATE=5e-7 \
NUM_ROLLOUT=20 \
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_outcome.sh
```

默认配置下，`96` 个 session 分成 `3` 个 update，框架计算得到的全局 batch size
为 `32`。`NUM_ROLLOUT` 增大不会自动改变默认的学习率衰减周期；要改变衰减周期，
显式设置 `OSWORLD_LR_DECAY_ITERS`。修改 batch、采样数或 update 数时，要同时
检查 Nex workspace 配额、`OSWORLD_MAX_CONCURRENT_GROUPS` 和 Ray GPU 资源。

### 固定的训练和并行参数

以下参数也在 `run_qwen3vl_8B_osworld_outcome.sh` 中，当前配方固定为 8 GPU：

| 参数                             |                       当前值 | 说明                                       |
| -------------------------------- | ---------------------------: | ------------------------------------------ |
| `--resource`                     | `actor=[1,8], rollout=[1,8]` | Ray actor 和 rollout 资源。                |
| `--tensor-model-parallel-size`   |                          `4` | Megatron TP。                              |
| `--pipeline-model-parallel-size` |                          `1` | Megatron PP。                              |
| `--rollout-num-gpus-per-engine`  |                          `2` | 每个 SGLang rollout engine 使用的 GPU 数。 |
| `--rollout-max-context-len`      |                      `32768` | rollout 上下文长度。                       |
| `--rollout-max-response-len`     |                       `8192` | 单次模型回复上限。                         |
| `--rollout-max-prompt-len`       |                      `16384` | prompt 长度上限。                          |
| `--micro-batch-size`             |                          `1` | 固定单样本 micro batch，控制训练显存峰值。 |
| `--recompute-granularity`        |                       `full` | 全层重计算，降低训练激活显存。             |
| `--sglang-mem-fraction-static`   |                        `0.7` | SGLang 静态显存比例。                      |
| `--sglang-router-policy`         |         `consistent_hashing` | rollout 请求路由策略。                     |

这些值涉及模型显存和 Ray 资源分配，修改后需要重新检查 TP、rollout engine
数量以及每个节点的 GPU 数量。

### Session 和桌面参数

编辑 [app/osworld_config.yaml](app/osworld_config.yaml)：

| 参数                                              |               当前值 | 作用                                                            |
| ------------------------------------------------- | -------------------: | --------------------------------------------------------------- |
| `max_turns`                                       |                 `30` | 每个任务最多执行的 agent 回合数。                               |
| `max_image_history_length`                        |                  `2` | 模型请求保留的历史截图数量。                                    |
| `context_policy`                                  |           `baseline` | 对话上下文策略。                                                |
| `temperature` / `top_p`                           |        `1.0` / `1.0` | 模型采样参数。                                                  |
| `max_tokens`                                      |               `8192` | 单次模型回复 token 上限。                                       |
| `llm_call_timeout_s`                              |                `600` | 单次 Relax 模型请求超时。                                       |
| `max_llm_call_retries` / `llm_call_backoff_s`     |            `3` / `5` | 请求重试次数和退避时间。                                        |
| `provider_name` / `snapshot_name`                 | `nex` / `init_state` | OSWorld provider 和快照名；Nex 实际用 `NEX_TEMPLATE` 选择模板。 |
| `screen_width` / `screen_height`                  |      `1920` / `1080` | 桌面分辨率。                                                    |
| `tool_retriever_backend` / `tool_retriever_top_k` |        `bm25` / `18` | 本地工具检索方式和数量。                                        |

调整 `max_turns` 时，需要同步检查启动脚本中的 `OSWORLD_MAX_STEPS_REF`；调整
`max_tokens` 时，需要同步检查 `--rollout-max-response-len`。奖励和轨迹调试参数
如 `OSWORLD_SHOT_DUMP_DIR`、`OSWORLD_SHOT_EVERY_N_TASKS` 可放在 `env.local.sh`。

### 任务列表和数据

编辑 [scripts/train_tasks.txt](scripts/train_tasks.txt) 可以更换训练任务。修改后
重新运行 `scripts/prepare_data.py` 生成新的 `${DATA_DIR}/osworld/train.parquet`；
启动脚本只会在 parquet 不存在或为空时自动生成，不会覆盖已有数据。

## 5. 启动训练

将 Qwen3-VL-8B-Thinking 模型文件准备到 `${MODEL_DIR}/Qwen3-VL-8B-Thinking/`，
确认第 1～4 节的环境、数据和配置就绪后执行：

```bash
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_outcome.sh
```

学习率、batch size、采样数、训练轮数、GPU 资源和并行配置等训练参数，直接在
[run_qwen3vl_8B_osworld_outcome.sh](run_qwen3vl_8B_osworld_outcome.sh) 中修改。

## 6. 运行产物与排查

| 内容               | 路径                                                            |
| ------------------ | --------------------------------------------------------------- |
| 训练日志           | `${SAVE_DIR}/osworld-outcome-<时间戳>.log`                      |
| Checkpoint         | `${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome/`                      |
| Session 日志       | `examples/osworld_agentic/log/agent/<run_tag>/<session_id>.log` |
| 奖励明细           | `${SAVE_DIR}/reward_dump.jsonl`                                 |
| 截图与动作（可选） | `${OSWORLD_SHOT_DUMP_DIR}/<task_id>/<时间戳>_<session标识>/`    |

需要查看桌面截图和动作时，在 `env.local.sh` 中添加：

```bash
export OSWORLD_SHOT_DUMP_DIR="${SAVE_DIR}/screenshots"
export OSWORLD_SHOT_EVERY_N_TASKS=1
```

将这两行放在 `SAVE_DIR` 设置之后。`OSWORLD_SHOT_EVERY_N_TASKS=1` 保存全部任务；
默认值 `37` 按任务 ID 哈希抽样。每个被选中的 session 目录包含逐回合 PNG、
`traj.jsonl` 和 `meta.json`。

排查时优先看对应的 session 日志：

- **导入失败或 session 提前退出**：检查 OSWorld / MCP 路径、外部目录布局和当前 Python 环境的依赖。
- **沙箱创建失败**：检查 Nex 地址、workspace、API key、模板及配额，结合 session 日志中的错误定位。
- **任务 setup 上传失败或返回 503**：检查全局 HTTP 代理是否影响了 Nex guest 的直连访问。
- **奖励异常或任务总是到达回合上限**：查看奖励明细中的 `agent_turns`、`stop_reason`、
  `advantage`、`group_std` 和 `group_kept`，再结合截图与动作记录定位。

汇总奖励和成功率：

```bash
source examples/osworld_agentic/env.local.sh
source examples/osworld_agentic/env.sh
python3 examples/osworld_agentic/scripts/analyze_run.py \
    --dump "${SAVE_DIR}/reward_dump.jsonl" \
    --groups-per-step 12 \
    --md
```

`--groups-per-step` 与训练脚本中的 `ROLLOUT_BATCH_SIZE` 保持一致。
奖励明细里的 `step` 是后处理调用序号，分批传输时不等于 rollout 编号。
统计脚本的 `cap` / “撞顶”列仍使用旧的 50-turn 阈值；当前 30-turn 配置请通过
`stop_reason=max_turns` 判断是否达到上限。
