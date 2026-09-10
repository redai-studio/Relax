# OSWorld Agentic RL

Qwen3-VL-8B-Thinking 在 OSWorld 上的 agentic RL recipe。沙箱由 Nex 平台提供，每个
rollout session 独占一台虚拟桌面，agent 靠截图 + pyautogui / MCP 操作，按任务最终
是否完成给 outcome reward（GRPO，无 critic）。

| | |
|---|---|
| 训练入口 | `run_qwen3vl_8B_osworld_outcome.sh`（baseline，outcome_only_rl） |
| 变体 | `run_qwen3vl_8B_osworld_context.sh`（context_rl，只换 `OSWORLD_CONFIG_PATH`） |
| agent 子进程 | `app/agent.py`，由 Relax 以 `python -m app.agent --input-json ... --output-json ...` 拉起 |
| session 配置 | `app/osworld_config.yaml`（可被 `OSWORLD_CONFIG_PATH` 覆盖） |
| 参考硬件 | 8 x H20（97871 MiB），2 NUMA node |

---

## 1. 环境准备：Nex 沙箱 API

API key 只落在 `/data/osworld_env.sh`（mode 600），**不要写进仓库、脚本或日志**。
文件内容是四行 export：

```
export NEX_API_KEY=your_api_key
export NEX_TEMPLATE=osworld-guest
export NEX_WORKSPACE_ID=your_workspace_id
export download_proxy=http://10.140.24.177:3128
```

写完后 `chmod 600 /data/osworld_env.sh`，并且只用长度校验、不要 `cat`：

```bash
chmod 600 /data/osworld_env.sh
echo "key_len=$(. /data/osworld_env.sh; echo ${#NEX_API_KEY})"
```

后面所有启动命令都用 `. /data/osworld_env.sh` 注入。

> 训练主日志开头会 dump 一份完整环境配置。`tail` / `grep` 日志时习惯性加一道
> `| grep -vi key`，避免 key 被贴进聊天、工单或截图。

---

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

**`qwen-agent` 别漏。** `mm_agents/agent_function_call.py` 顶部就 `from qwen_agent.tools.base
import BaseTool, register_tool`；缺了它 agent 子进程会以 code 1 退出，而 Relax 侧只会
刷 `Dropping group ... because an agent exited before every first IR existed`，看不出真正
原因——必须去 `examples/osworld_agentic/log/agent/<最新时间戳>/` 翻 session 日志才能看到
`ModuleNotFoundError`。

---

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

---

## 4. 启动训练

```bash
cd $FX/Relax && \
  nohup numactl --interleave=all bash -c '. /data/osworld_env.sh; exec bash launch_osworld_example.sh' \
    > osworld-$(date +%m%d-%H%M).log 2>&1 &
disown
```

`launch_osworld_example.sh` 是本目录下的启动模板，把路径和超参都收在顶部，照着改即可。

**`numactl --interleave=all` 不能省。** 不开 NUMA interleave，rollout 高并发时内存会全压
在单个 node 上，直接 OOM。

---

## 5. 可调开关（全部走环境变量）

### 规模 / 并发

| 变量 | 默认 | 说明 |
|---|---|---|
| `ROLLOUT_BATCH_SIZE` | 8 | 每 step 采多少 prompt |
| `N_SAMPLES_PER_PROMPT` | 8 | 每个 prompt 采多少条轨迹（GRPO 的组大小） |
| `GLOBAL_BATCH_SIZE` | 64 | 建议 `= ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT`，即一次 rollout 一次 optimizer update |
| `OSWORLD_MAX_CONCURRENT_GROUPS` | — | 并发组数，实际沙箱数 = 它 × `N_SAMPLES_PER_PROMPT` |
| `PREPARE_POOL_SIZE` | — | 预热池大小，一般跟上面取同值 |

### session 行为

| 变量 | 默认 | 说明 |
|---|---|---|
| `OSWORLD_CONFIG_PATH` | `app/osworld_config.yaml` | session 配置文件 |
| `OSWORLD_MAX_TURNS` | yaml 的 `max_turns` | 单条轨迹最多几个 turn |
| `OSWORLD_MAX_STEPS_REF` | 50 | reward shaping 的 `T_ref`（paper Eq.3），**必须等于 `max_turns`** |
| `AGENT_TIMEOUT` | 1500 | 单个 session 墙钟上限（秒） |

yaml 里另有 `max_image_history_length`（prompt 里保留最近几张截图，**没有环境变量能覆盖**）、
`temperature` / `top_p` / `max_tokens` / `context_policy`。

### 训练性能

| 变量 | 默认 | 说明 |
|---|---|---|
| `RECOMPUTE_GRANULARITY` | `full` | `full` / `selective` / `none` |
| `RECOMPUTE_METHOD` | `uniform` | 仅 `full` 时生效 |
| `RECOMPUTE_NUM_LAYERS` | 1 | 仅 `full` 时生效 |
| `TRUE_ON_POLICY` | 0 | 置 1 加 `--true-on-policy-mode`，**colocate 下不省时间**，见第 7 节 |

### 轨迹落盘（调试用）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OSWORLD_SHOT_DUMP_DIR` | 空（关闭） | 设了就按 task 抽样存每 turn 截图 + 动作 |
| `OSWORLD_SHOT_EVERY_N_TASKS` | 35 | 对 `task_id` 取 md5，`% N == 0` 才存 |

抽样是按 `task_id` 哈希的确定性抽样，所以：同一个 task 的**整组样本会被完整抓下来**
（可以横着比同组 8 条轨迹为什么 advantage 是 0），而且**每个 step 抓的是同一批 task**
（可以竖着看这些任务有没有学会）。

---

## 6. 产物在哪

| 内容 | 路径 |
|---|---|
| 训练主日志 | `osworld-*.log`（启动命令里自己重定向的） |
| 每条 session 的 reward / turn 数 / 停止原因 | `$OSWORLD_REWARD_DUMP`（jsonl，一行一个 session） |
| 完整 rollout 轨迹（prompt / 输出 / token） | `$SAVE_DIR/<exp>/rollout_result/train/<step>.jsonl` |
| 轨迹截图 + 动作 | `$OSWORLD_SHOT_DUMP_DIR/<task_id>/<时间戳>_<session>/{NNN.png, traj.jsonl, meta.json}` |
| checkpoint | `$SAVE_DIR/<exp>/iter_XXXXXXX/`（Megatron 分布式格式 `.distcp`，**不是 `.pt`**） |
| agent 子进程日志 | `examples/osworld_agentic/log/agent/<时间戳>/` |

reward dump 每行的字段：

```
ts run_tag step group_index session task_id data_source
session_wallclock_s agent_turns stop_reason parse_failed_steps last_error
action_types outcome turns records shaped advantage group_std mean_succ group_kept
```

排查「为什么这一步没梯度」优先看 `group_std`（组内 reward 全一样 → advantage 全 0）
和 `stop_reason`。

---

## 7. 性能笔记

以 74 prompt × 8 sample = 592 session、`max_tokens_per_gpu=32768`、8×H20 实测：

| | 时间 |
|---|---|
| rollout 等待 | 约 2h38m |
| train | 约 5h28m |
| step 合计 | 约 8h06m |

train 内部拆开：`actor_train` 13038s + `log_probs` 3298s + `ref_log_probs` 3306s。

**`RECOMPUTE_GRANULARITY=selective`（推荐开）**：同配置 A/B 实测 `actor_train` 按 token
归一化后降 16.4%（吞吐 11719 → 14017 tok/s），显存峰值涨 4.6 GB（80.2 → 84.8 GB，离
97.9 GB 还有 13 GB）。理论上也对得上——`actor_train / log_probs` 从 4.91 降到 4.11，正好
少了 0.8 个前向（全重算多付 1 个完整前向，selective 只重算 attention）。真 OOM 就退回
`RECOMPUTE_METHOD=block` + `RECOMPUTE_NUM_LAYERS=18`。

**`TRUE_ON_POLICY=1` 在 colocate 下不省时间。** `relax/core/registry.py` 的
`process_role()` 里，把 `actor_fwd` 角色摘掉的分支只在 `fully_async` 下可达；colocate
走的是 `ROLES_COLOCATE`，`actor_fwd` 照跑，`log_probs_time` 一秒不少。它唯一的效果是
`loss.py` 里把 ratio 强制成 1，于是 `ppo_kl` 恒等于 0——等于白白丢掉一个数值健康指标。
除非改 `process_role()`，否则保持 0。

**`kl_loss_coef`（默认 0.02）是真在用的**，别看到 `kl_coef=0` 就以为 reference 前向白算了。

---