# OSWorld agentic RL

用外部 OSWorld 的 `HybridAgentLocal` 和 Nex `DesktopEnv`，通过 Relax 原生
Agentic rollout 训练 Qwen3-VL-8B-Thinking。本目录只保留 Relax 适配和一个已跑通的
outcome 配方：8 GPU、12 任务 × 8 session、每轮 3 次 optimizer update。

## 环境配置

仓库中的 [env.sh](env.sh) **是环境配置示例**，只包含占位值。
将实际路径和凭据放到 `env.local.sh`，按下面的命令将它加入本地
`.git/info/exclude`（不会随 PR 提交）。两个启动脚本都会先读取 `env.local.sh`，
再用 `env.sh` 补齐未设置的默认值。已有 `env.local.sh` 时直接使用，不要覆盖。

```bash
printf '%s\n' '/examples/osworld_agentic/env.local.sh' >> "$(git rev-parse --git-path info/exclude)"
cp -n examples/osworld_agentic/env.sh examples/osworld_agentic/env.local.sh
chmod 600 examples/osworld_agentic/env.local.sh
```

填写 `MEGATRON`、`MODEL_DIR`、`DATA_DIR`、`SAVE_DIR`、`OSWORLD_REPO`、
`OSWORLD_CACHE_DIR`、`MCP_SRC_ROOT` 和 Nex 配置。`MODEL_DIR` 下需要有
`Qwen3-VL-8B-Thinking`；每次从 base 模型训练时使用新的 `SAVE_DIR`。
`NEX_API_KEY` 在 `env.local.sh` 中设置，或从私有凭据文件读取。
如需下载代理，使用 evaluator 的 `download_proxy`；不要让访问 Nex guest 的请求走 HTTP 代理。

这里依赖**包含 Nex provider、MCP 支持和 HybridAgentLocal 的 OSWorld 版本**。
外部目录需要保持以下布局；工具注册表和 prompt 由外部 agent 维护：

```text
external_root/
├── OSWorld-main-nex/           # OSWORLD_REPO
│   ├── desktop_env/
│   ├── mm_agents/hybrid_agent_local.py
│   └── evaluation_examples/
├── agents/tool_retriever.py
├── prompts/
└── tools/tools_registry.json
```

训练环境需要预先安装 Relax 和该外部 OSWorld 的运行依赖；仅有标准上游
OSWorld checkout 不足以运行此配方。所有 Ray 节点需要能够访问相同路径的
Relax、外部 OSWorld、MCP 源码、模型和本地环境配置。

## 启动与参数

先准备可用的 Ray 集群，再从 Relax 仓库根目录执行：

```bash
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_outcome.sh
```

脚本直接提交 `relax/entrypoints/train.py`，使用框架默认数据源和 Agentic rollout。
首次运行如果 `$DATA_DIR/osworld/train.parquet` 不存在，会根据
[scripts/train_tasks.txt](scripts/train_tasks.txt) 的 12 个任务 ID，从外部
`evaluation_examples` 生成数据。修改任务列表后，需要重新运行 `scripts/prepare_data.py`
覆盖已有 parquet；生成命令见该脚本说明。配方不运行独立 eval，session 内仍由
OSWorld evaluator 给每条轨迹评分。

| 参数                          | 默认值         | 含义                                                  |
| ----------------------------- | -------------- | ----------------------------------------------------- |
| `ROLLOUT_BATCH_SIZE`          | 12             | 脚本内固定的任务组数                                  |
| `N_SAMPLES_PER_PROMPT`        | 8              | 脚本内固定的每组 session 数，共 96 条                 |
| `OSWORLD_UPDATES_PER_ROLLOUT` | 3              | 96 条 session 分成 3 个 batch，每条只训练一次         |
| `OSWORLD_LEARNING_RATE`       | `1e-6`         | 学习率                                                |
| `NUM_ROLLOUT`                 | 10             | rollout 总轮数                                        |
| `OSWORLD_LR_DECAY_ITERS`      | `45 × updates` | optimizer 更新次数；默认 135，不随 `NUM_ROLLOUT` 改变 |

后四项可用环境变量覆盖；当前 DP=2，更新次数可选 1、2、3、6，使每批任务组能够均分。
默认全局 batch size 由框架计算为 `96 / 3 = 32`，无需单独配置。例如只增加 rollout 数：

```bash
NUM_ROLLOUT=20 bash examples/osworld_agentic/run_qwen3vl_8B_osworld_outcome.sh
```

此时 LR 衰减仍为 135 次更新。若设置 `OSWORLD_UPDATES_PER_ROLLOUT=1`，
全局 batch size 为 96，默认衰减周期为 45 次更新。

数据按列表顺序循环使用。正常情况下每轮对应 12 个任务；session 失败或任务组被
丢弃时，按框架默认策略补采，**不再额外保证每轮恰好包含 12 个不同任务**。

TP=4、每个 rollout engine 使用 2 GPU、selective recompute、CPU optimizer offload、
KL/ref 更新、图像缓存和传输设置保留为启动脚本中的固定配置。
唯一的 session 配置是 [app/osworld_config.yaml](app/osworld_config.yaml)：
最多 30 turns、保留 2 张历史截图、`context_policy: baseline`、采样温度和 top-p 均为 1。

## 模块职责

| 文件                                | 职责                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------- |
| `run_qwen3vl_8B_osworld_outcome.sh` | 实验参数、数据准备、Ray 环境和训练提交                                    |
| `run_agent_app.sh`                  | 读取本地环境，设置外部导入路径和 Relax session endpoint，启动 agent       |
| `env.sh`                            | 可复制的环境配置示例                                                      |
| `scripts/train_tasks.txt`           | 12 个训练任务 ID                                                          |
| `scripts/prepare_data.py`           | 把外部任务 JSON 转为 Relax parquet                                        |
| `app/agent.py`                      | session 主循环：动作执行、评估、清理和逐 turn 导出                        |
| `app/hybrid_agent.py`               | 将外部 HybridAgentLocal 的 LLM 请求接到 Relax，保留逐 turn 对话记录       |
| `app/env_osworld.py`                | 外部 DesktopEnv 的异步适配、Nex 创建重试、并发限制和清理                  |
| `app/task_setup_compat.py`          | 任务 setup 配置兼容处理                                                   |
| `app/reward_post_process.py`        | 按任务组和 session 聚合奖励，将优势分配给各 turn                          |
| `reward_osworld.py`                 | 未携带 session reward 时的 RM 回调；正常路径使用 session evaluator 的结果 |
| `app/shotdump.py`                   | 可选的截图和动作采集                                                      |
| `scripts/analyze_run.py`            | 分析 reward dump 中的成功率和组内奖励统计                                 |

逐 turn 导出仍然需要保留：外部 agent 会重建 prompt、裁剪截图历史，连续 turn
不是严格的追加关系。`hybrid_agent.py` 记录每次实际请求的对话，`agent.py` 将每个
turn 输出成独立 JSONL 记录；reward 后处理按 session 聚合，避免把这些 turn
误当成不同的 GRPO 样本。

奖励后处理保留当前配方：成功/失败为 ±1，按轮数扣除长度惩罚，达到 30 turns
额外扣分；组内标准化后复制到各 turn，关闭按轨迹长度再归一化。
全成功或全失败组的优势为 0。`OSWORLD_MAX_STEPS_REF=30` 和
`OSWORLD_PER_TRAJ_NORM=0` 已固定在启动脚本中。

## 运行产物与排查

| 内容         | 路径                                                 |
| ------------ | ---------------------------------------------------- |
| 训练日志     | `$SAVE_DIR/osworld-outcome-<时间戳>.log`             |
| checkpoint   | `$SAVE_DIR/Qwen3-VL-8B-osworld-outcome/`             |
| session 日志 | `log/agent/<run_tag>/<session_id>.log`（相对本目录） |
| 奖励明细     | `$SAVE_DIR/reward_dump.jsonl`                        |

遇到导入错误或 session 提前退出，先查 session 日志。奖励明细中可以查看
`agent_turns`、`stop_reason`、`advantage`、`group_std` 和 `group_kept`；
其中 `step` 是 reward 后处理调用序号，分批传输时不等于 rollout 编号。
可选截图通过在 `env.local.sh` 设置 `OSWORLD_SHOT_DUMP_DIR` 开启；
`OSWORLD_SHOT_EVERY_N_TASKS` 控制按任务 ID 哈希抽样的频率，默认 37。
更多运行注意事项见 [PITFALLS.md](PITFALLS.md)。

历史统计脚本 `scripts/analyze_run.py` 用于本配方时需传 `--groups-per-step 12`；
其 `cap` 列仍采用旧的 50-turn 阈值，30-turn 到顶情况请查看奖励明细的 `stop_reason`。
