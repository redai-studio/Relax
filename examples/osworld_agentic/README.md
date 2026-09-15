# OSWorld Agentic RL

Qwen3-VL-8B-Thinking 在 OSWorld 上的 agentic RL recipe。沙箱由 Nex 平台提供，每个
rollout session 独占一台虚拟桌面，agent 靠截图 + pyautogui / MCP 操作，按任务最终
是否完成给 outcome reward（GRPO，无 critic）。

|              |                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------- |
| 训练入口     | `launch_fixed12_minibatch3.sh`（12 任务，96 条 session，3 次更新）                        |
| agent 子进程 | `app/agent.py`，由 Relax 以 `python -m app.agent --input-json ... --output-json ...` 拉起 |
| session 配置 | `app/osworld_config_t30i2.yaml`（可被 `OSWORLD_CONFIG_PATH` 覆盖）                        |
| 参考硬件     | 8 x H20（97871 MiB），2 NUMA node                                                         |

______________________________________________________________________

## 1. 环境准备

复制 `env.sh.example` 为本目录的 `env.sh`，填写 `NEX_API_KEY`、
`NEX_WORKSPACE_ID`、`MODEL_DIR`、`DATA_DIR`、`SAVE_DIR`、`MEGATRON` 和
`OSWORLD_REPO`。`env.sh` 已被 Git 忽略；密钥不要写进仓库或日志。
`MODEL_DIR` 下应有 `Qwen3-VL-8B-Thinking`，`SAVE_DIR` 使用新的输出目录。
外部 OSWorld checkout 提供 `desktop_env` 和 `mm_agents`，依赖边界见 [VENDOR.md](VENDOR.md)。

```bash
cp examples/osworld_agentic/env.sh.example examples/osworld_agentic/env.sh
chmod 600 examples/osworld_agentic/env.sh
```

______________________________________________________________________

## 2. 安装依赖

OSWorld 的 `requirements.txt` 直接装会跟训练栈打架（它要 `protobuf<5`、`Pillow~=11.0`，
会把 Ray / torchvision 拖坏），resolver 还会 ResolutionImpossible。所以核心包一律
`--no-deps`，缺的传递依赖单独补：

```bash
# 2.1 核心包 —— 必须 --no-deps
pip install --break-system-packages --no-deps \
  gymnasium easyocr pyautogui python-xlib fuzzywuzzy odfpy openpyxl pypdf python-docx \
  python-pptx mss beautifulsoup4 xmltodict rapidfuzz func_timeout filetype borb pdfplumber \
  PyMuPDF wrapt_timeout_decorator IMAPClient pyacoustid pygame pytesseract translate send2trash \
  pyperclip pynput python-Levenshtein playwright pydrive oauth2client requests_toolbelt formulas \
  tldextract requests_file imagehash backoff json5 dashscope qwen-agent

# 2.2 google 客户端 —— 这个要带依赖装（需要 httplib2 / uritemplate）
pip install --break-system-packages google-api-python-client google-auth google-auth-oauthlib

# 2.3 立刻把 protobuf 钉回来（上一步会把它降到 5.29）
pip install --break-system-packages -U 'protobuf>=6.31.1'
```

**`qwen-agent` 别漏。** `mm_agents/agent_function_call.py` 顶部就 `from qwen_agent.tools.base import BaseTool, register_tool`；缺了它 agent 子进程会以 code 1 退出，而 Relax 侧只会
刷 `Dropping group ... because an agent exited before every first IR existed`，看不出真正
原因——必须去 `examples/osworld_agentic/log/agent/<最新时间戳>/` 翻 session 日志才能看到
`ModuleNotFoundError`。

______________________________________________________________________

## 3. 环境验证

```bash
FX=/path/to/workspace   # 底下有 Relax/ 和 OSWorld-main-nex/

python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
python3 -c "import ray, sglang, megatron; from torch.utils.tensorboard import SummaryWriter; print('training stack ok')"

# 整条 agent import 链 —— 一定要 import app.agent，不要只 import 子模块
cd $FX/Relax/examples/osworld_agentic && \
  OSWORLD_REPO=$FX/OSWorld-main-nex PYTHONPATH=$FX/OSWorld-main-nex:. \
  python3 -c "import app.agent; print('agent chain ok')"

command -v numactl && numactl -H | grep -c '^node [0-9]* size'   # 应输出 2
```

第三条特别强调：`app/agent.py` 第 27 行 import `app.env_osworld`、第 28 行才 import
`app.hybrid_agent`。只测 `app.env_osworld` 会得到假绿灯——真正缺包的是第 28 行往下那条链
（`app.hybrid_agent` → `mm_agents.hybrid_agent_local` → `mm_agents.agent_function_call`
→ `qwen_agent`）。**只有 `import app.agent` 才走完整条链。**

______________________________________________________________________

## 4. 启动 12 任务训练

先启动可用的 Ray 集群，再从仓库根目录运行：

```bash
bash examples/osworld_agentic/launch_fixed12_minibatch3.sh
```

入口默认从 base 模型和新 optimizer 开始，跑 10 个 rollout step，不执行独立 eval；
每条训练轨迹仍由 OSWorld evaluator 评分。`scripts/manifests/train_fixed12.txt`
列出固定的 12 个任务 ID。首次运行从 `$OSWORLD_REPO/evaluation_examples` 自动生成
`$DATA_DIR/osworld/fixed12/train.parquet`，也可用 `OSWORLD_TASK_DIR` 指定任务 JSON 目录。
每步覆盖全部 12 个任务，每任务采样 8 条，96 条 session 分为 3 个原生 optimizer batch，
每条样本训练一次。`NUM_ROLLOUT` 可调整步数，`OSWORLD_LEARNING_RATE` 默认 `1e-6`。

`train_fixed_tasks.py` 接入固定任务数据源和整组补采；`train_fixed12_minibatch3.py`
校验 96/32 的分批设置。
公用的 `run_qwen3vl_8B_osworld_outcome.sh` 负责模型、训练参数和 Ray 提交。

### 每轮一次更新，训练 100 步

设置 `OSWORLD_NATIVE_MINIBATCH3=0` 后，96 条 session 合成一个 global batch，
每个 rollout step 只执行一次 optimizer 更新，每条样本仍只训练一次。
此模式使用 `train_fixed_tasks.py`；默认三批模式保持不变。

先在 `env.sh` 中设置新的 `SAVE_DIR`，并确保它不会覆盖下面的环境变量，再执行：

```bash
unset OSWORLD_RESUME_DIR RESUME_FROM_SAVE
OSWORLD_NATIVE_MINIBATCH3=0 \
NUM_ROLLOUT=100 \
OSWORLD_LR_DECAY_ITERS=100 \
bash examples/osworld_agentic/launch_fixed12_minibatch3.sh
```

这个入口从 base 模型和新 optimizer 开始，不执行独立 eval。
`OSWORLD_LR_DECAY_ITERS` 的单位是 optimizer 更新次数：单次更新模式默认 45，
三批模式默认 135；修改总步数时需显式设置衰减周期，100 步分别对应 100 / 300。

______________________________________________________________________

## 5. 可调开关（默认值对应 12 任务入口）

### 规模 / 并发

| 变量                            | 默认 | 说明                                               |
| ------------------------------- | ---- | -------------------------------------------------- |
| `ROLLOUT_BATCH_SIZE`            | 12   | 每步固定覆盖全部任务                               |
| `N_SAMPLES_PER_PROMPT`          | 8    | 每个 prompt 采多少条轨迹（GRPO 的组大小）          |
| `GLOBAL_BATCH_SIZE`             | 32   | 96 条 session 分成 3 次 optimizer update           |
| `OSWORLD_MAX_CONCURRENT_GROUPS` | 20   | 并发组数，实际沙箱数 = 它 × `N_SAMPLES_PER_PROMPT` |
| `PREPARE_POOL_SIZE`             | 20   | 预热池大小，一般跟上面取同值                       |

### session 行为

| 变量                    | 默认                            | 说明                                                              |
| ----------------------- | ------------------------------- | ----------------------------------------------------------------- |
| `OSWORLD_CONFIG_PATH`   | `app/osworld_config_t30i2.yaml` | session 配置文件                                                  |
| `OSWORLD_MAX_TURNS`     | 30                              | 单条轨迹最多几个 turn                                             |
| `OSWORLD_MAX_STEPS_REF` | 30                              | reward shaping 的 `T_ref`（paper Eq.3），**必须等于 `max_turns`** |
| `AGENT_TIMEOUT`         | 21600                           | 单个 session 墙钟上限（秒）                                       |

yaml 里另有 `max_image_history_length`（prompt 里保留最近几张截图，**没有环境变量能覆盖**）、
`temperature` / `top_p` / `max_tokens` / `context_policy`。

### 训练性能

| 变量                    | 默认        | 说明                                       |
| ----------------------- | ----------- | ------------------------------------------ |
| `RECOMPUTE_GRANULARITY` | `selective` | `full` / `selective` / `none`              |
| `RECOMPUTE_METHOD`      | `uniform`   | 仅 `full` 时生效                           |
| `RECOMPUTE_NUM_LAYERS`  | 1           | 仅 `full` 时生效                           |
| `TRUE_ON_POLICY`        | 0           | 公用 runner 的可选开关；本 recipe 保持关闭 |

### 轨迹落盘（调试用）

| 变量                         | 默认       | 说明                                    |
| ---------------------------- | ---------- | --------------------------------------- |
| `OSWORLD_SHOT_DUMP_DIR`      | 空（关闭） | 设了就按 task 抽样存每 turn 截图 + 动作 |
| `OSWORLD_SHOT_EVERY_N_TASKS` | 35         | 对 `task_id` 取 md5，`% N == 0` 才存    |

抽样是按 `task_id` 哈希的确定性抽样，所以：同一个 task 的**整组样本会被完整抓下来**
（可以横着比同组 8 条轨迹为什么 advantage 是 0），而且**每个 step 抓的是同一批 task**
（可以竖着看这些任务有没有学会）。

______________________________________________________________________

## 6. 产物在哪

| 内容                                        | 路径                                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| 训练主日志                                  | `osworld-*.log`（启动命令里自己重定向的）                                              |
| 每条 session 的 reward / turn 数 / 停止原因 | `$OSWORLD_REWARD_DUMP`（jsonl，一行一个 session）                                      |
| 完整 rollout 轨迹（prompt / 输出 / token）  | `$SAVE_DIR/<exp>/rollout_result/train/<step>.jsonl`                                    |
| 轨迹截图 + 动作                             | `$OSWORLD_SHOT_DUMP_DIR/<task_id>/<时间戳>_<session>/{NNN.png, traj.jsonl, meta.json}` |
| checkpoint                                  | `$SAVE_DIR/<exp>/iter_XXXXXXX/`（Megatron 分布式格式 `.distcp`，**不是 `.pt`**）       |
| agent 子进程日志                            | `examples/osworld_agentic/log/agent/<时间戳>/`                                         |

reward dump 每行的字段：

```
ts run_tag step group_index session task_id data_source
session_wallclock_s agent_turns stop_reason parse_failed_steps last_error
action_types outcome turns records shaped advantage group_std mean_succ group_kept
```

排查梯度信号时同时看 `advantage`、`group_std` 和 `group_kept`；除了组内零方差，
reward 后处理的准确率阈值也会屏蔽整组。`stop_reason` 用来核对终止原因。

______________________________________________________________________

## 7. 性能设计

- Agentic rollout 在 session 内复用截图张量，按完整任务组分批传入 TensorQueue，传输后释放源端像素，降低 CPU 内存峰值。
- `RECOMPUTE_GRANULARITY` 默认 `selective`。可按显存需求改为 `full`，结合 `RECOMPUTE_METHOD` 和 `RECOMPUTE_NUM_LAYERS` 配置重算。
