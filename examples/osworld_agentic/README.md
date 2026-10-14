# OSWorld + Nex agentic 训练

[English](README.en.md)

本 recipe 使用 Qwen3-VL-8B-Thinking、Relax resident agentic rollout 和 Nex 桌面沙箱进行 GRPO 训练。它依赖外部 Nex 服务及可用的 OSWorld 桌面模板；只克隆仓库不能直接运行。

## 目录与配置

- `env.example.sh`：公开的部署配置模板，不包含实际服务地址或密钥。
- `env.local.sh`：用户复制模板后填写的私有配置，Git 忽略，不能提交。
- `scripts/run-qwen3vl-8B-osworld-8xgpu.sh`：正式训练参数与提交入口；修改训练参数请改这里。
- `run_qwen3vl_8B_osworld_outcome.sh`：兼容旧入口，转发到正式脚本。
- `run_agent_app.sh`：Relax 为每个 session 调用的 agent 入口，不是独立训练入口。
- `app/`：桌面环境、模型调用适配、奖励与监控；`app/osworld_config.yaml` 控制交互轮数等。
- `scripts/prepare_data.py`、`train_tasks.txt`：数据转换与固定 12 任务清单。
- `scripts/analyze_run.py`、`plot_training.py`：离线分析工具；`reward_osworld.py` 保留兼容 scorer，正式脚本使用 agent metadata 和自定义 advantage 回调。

训练脚本和 agent 入口直接读取同目录的 `env.local.sh`，无需额外配置文件。启动时传入的三个根目录优先。

## 准备环境

1. 按 [Relax 安装文档](../../docs/zh/guide/installation.md)准备训练镜像、Megatron、SGLang、Ray 和模型权重。此配置需要单节点 8 张有足够显存的 GPU；显存需求随轨迹变化。脚本固定 `MEGATRON=/root/Megatron-LM`，其他镜像需修改该行。所有 worker 必须能访问相同路径、相同版本的代码与依赖。

2. 准备 OSWorld fork。下面固定到本 recipe 检查的版本：

   ```bash
   git clone https://github.com/fxsc03/OSWorld-main-nex.git /data/OSWorld-main-nex
   git -C /data/OSWorld-main-nex checkout 50b4a85778cfc13ce00e3bf63a3cd08bd260fb96
   python3 -m venv /data/osworld-venv
   /data/osworld-venv/bin/python -m pip install -r /data/OSWorld-main-nex/requirements.txt qwen-agent
   ```

   使用独立 agent Python 环境，避免外部依赖覆盖训练环境。宿主环境安装不能替代沙箱内的依赖：Nex 模板仍需提供桌面应用、LibreOffice/UNO、MCP 工具及任务所需文件。请按该 fork 的 Nex 部署说明验证模板、网络和任务 reset/evaluator。

3. 在 recipe 目录复制配置，仅首次执行，不要覆盖已有私有文件：

   ```bash
   cd examples/osworld_agentic
   cp -n env.example.sh env.local.sh
   chmod 600 env.local.sh
   ```

   填写 `OSWORLD_REPO`、`OSWORLD_AGENT_PYTHON`、`NEX_API_BASE`、`NEX_WORKSPACE_ID`、`NEX_API_KEY` 和 `NEX_TEMPLATE`（或 `NEX_TEMPLATE_ID`）。`MCP_SRC_ROOT` 是包含 `mcp/` 的仓库根目录。外部 Nex client 含部署相关默认值，因此 agent 入口要求显式提供 Nex 配置，不依赖这些默认值。

   私有文件必须在所有 agent worker 的相同路径可读。若使用 Ray 上传代码，被忽略的文件不会随代码上传，需另外安全部署；也可预先在 worker 环境配置。不要通过 runtime-env JSON 或命令行传递密钥，相关参数可能进入 Jobs 元数据与日志。

4. 正式脚本默认开启 ClearML。按自己的部署配置训练节点的 ClearML（例如 `clearml-init`）；模板不携带任何账户、密钥或内部端点。

## 启动

配置好 `env.local.sh` 后，在 Relax 根目录运行：

```bash
MODEL_DIR=/data/models \
DATA_DIR=/data/rl_data \
SAVE_DIR=/data/outputs \
bash examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh
```

- 模型应位于 `${MODEL_DIR}/Qwen3-VL-8B-Thinking`。
- 数据读取 `${DATA_DIR}/osworld/train.parquet`；不存在时从外部仓库的 `evaluation_examples/examples` 和固定清单生成。清单包含未知任务或为空会报错。已有 parquet 会复用，修改清单后请明确重新生成，避免继续读旧数据。
- 自动生成的 train/eval 使用同一清单；这不是独立验证集，默认也未配置周期评估。
- 输出位于 `${SAVE_DIR}/qwen3vl-8b-osworld-8xgpu/`。

## 当前正式参数

| 配置                           | 值                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| 初始模型                       | Qwen3-VL-8B-Thinking                                                                 |
| 算法                           | GRPO，自定义轨迹级 advantage，TIS                                                    |
| 每轮采样                       | 12 组 × 每组 8 条 = 96 条轨迹                                                        |
| 更新次数                       | 每 rollout 2 次，共 200 rollouts                                                     |
| 学习率                         | 5e-6，constant                                                                       |
| KL                             | k3，系数 0.001，ref 每 rollout 更新                                                  |
| PPO clip                       | 下/上界均 0.2                                                                        |
| 训练并行                       | TP4 / PP1 / CP1，8 卡下 DP2                                                          |
| 单卡训练 / log-prob token 预算 | 均为 49152；不是模型上下文长度                                                       |
| Rollout 上下文 / response 上限 | 16384 / 8192 tokens                                                                  |
| SGLang                         | 每引擎 2 GPU，static memory fraction 0.7，consistent hashing                         |
| 显存策略                       | 冻结 vision trunk；selective recompute `core_attn mlp`；optimizer CPU offload 未开启 |
| 沙箱                           | prelaunch 开启；active 与预启动共享 12 个 group permits，最多 96 个 session          |
| Dynamic filter                 | 关闭                                                                                 |
| Agent                          | 最多 30 轮，图像历史 2 张，session timeout 21600 秒                                  |
| Checkpoint                     | 关闭；每次启动从初始模型开始，不恢复优化器状态                                       |

Actor 与 rollout colocate，共用 8 张物理 GPU。session 并发限制不等于 Nex 服务端实例计数保证：创建重试、服务端回收延迟仍需结合 Nex 配额监控。

## 奖励、轨迹与指标

每条轨迹的 shaped reward 为：`(成功 ? 1 : -1) - 0.05 × T/30 - 0.2 × I(T>=30)`，其中成功判定为 evaluator outcome ≥ 0.5，T 为轨迹轮数。每组 8 条轨迹计算并标准化 advantage，再分配到该轨迹的训练导出记录；不是按导出记录数量组成 8 条一组。

96 条逻辑轨迹可以导出远多于 96 条训练记录。`rollout/raw_reward` 为对齐交接实现的**按导出记录数加权的 shaped reward**，不是轨迹成功率。判断任务能力应同时看成功率、轮数、异常比例；不能将 raw reward 图标成 success rate。全成功/全失败组的策略 advantage 被置零，但启用的 KL 项仍可能产生梯度。

`max_image_history_length=2` 限制单次模型上下文的图像历史，不表示一条轨迹只有两张截图。截图 dump 是额外诊断功能，磁盘图片数不能直接代表交互轮数。

## 日志与隐私

- 提交日志：`examples/osworld_agentic/log/<experiment>-<timestamp>.log`。
- Agent 日志：`examples/osworld_agentic/log/agent/<run-tag>/`。
- 训练导出：`${SAVE_DIR}/<experiment>/rollout_result/<run-tag>/train/`。
- 轨迹奖励：同一 run 目录的 `reward_dump.jsonl`。
- TensorBoard：`${SAVE_DIR}/<experiment>/tensorboard_log/<run-tag>/`；ClearML 实验名带启动时间。

```bash
python examples/osworld_agentic/scripts/analyze_run.py \
  --dump /data/outputs/qwen3vl-8b-osworld-8xgpu/rollout_result/<run-tag>/reward_dump.jsonl \
  --groups-per-rollout 12 --samples-per-group 8 --md
```

该离线分组适用于当前未开启 dynamic filter 的固定批次配置；更换采样流程后需要重新核实其分组假设。

Git 忽略私有配置、常见日志/截图/缓存目录，但忽略规则不是数据脱敏机制。桌面截图、任务内容、agent 日志及导出 JSON 可能包含敏感信息，发布前应单独审查；不要强制添加私有文件。加载私有配置时禁用 shell tracing，仍不应把整个私有环境打印到日志。

本 recipe 的审查不代表外部 OSWorld fork、沙箱镜像或历史 Git 提交已完成隐私审计。尤其外部 fork 含部署相关默认端点，公开分发该依赖前应单独审查。

## CHANGELOG

以下按实验启动时间升序记录，时区为 UTC+8。日期后的时间对应实验名中的启动时间；“更新”指优化器更新，编号从 0 开始。只记录可由保留的提交参数和日志确认的结果，脚本修改不算已验证实验。

### 2026-10-11 18:33 — 初始集成实验

- **配置**：Qwen3-VL-8B-Thinking，12 任务 × 8 条轨迹；`num_rollout=10`、每轮 3 次更新；LR `1e-6` / cosine；KL 系数 `0.02`；训练预算和 rollout context 均为 32K；full/uniform recompute，optimizer CPU offload 开启。
- **结果**：日志记录 9 次更新，随后 SGLang engine 在 flush cache 时连接拒绝，任务失败，未完成计划。这里的 10 rollouts 最多对应 30 次更新；这个短配置不满足后续长期训练需求。
- **后续**：延长至 200 rollouts，并调整 offload/recompute。Agent 按逻辑 session 管理沙箱、按模型可见上下文导出训练记录；96 条轨迹可以对应 1400 多条训练记录，不代表同等数量的沙箱。

### 2026-10-11 22:58 — 200 rollouts，关闭 offload、减少重计算

- **相对上次**：`num_rollout: 10 → 200`；关闭 optimizer CPU offload；full recompute 改为 selective；训练预算与 rollout context 暂保留 32K。
- **结果**：首次训练发生 CUDA OOM，没有完成优化器更新。
- **后续**：降低单卡 token 预算，补齐长度配置之间的约束。

### 2026-10-12 01:15 — 训练预算降至 16K

- **相对上次**：`max_tokens_per_gpu: 32768 → 16384`，rollout context 仍为 32768。
- **结果**：启动参数校验失败：`max_tokens_per_gpu × CP` 小于 rollout context；没有进入训练。

### 2026-10-12 01:16 — Rollout context 降至 16K

- **相对上次**：`rollout_max_context_len: 32768 → 16384`。
- **结果**：再次在参数校验阶段失败：prompt 上限仍为 16384，与 context 相等，未给生成 token 留空间；没有进入训练。

### 2026-10-12 01:21 — 修正 prompt/context 约束

- **相对上次**：prompt 上限改为 16383，保持 16K context 和 16K 单卡训练预算。
- **结果**：完成 1 次更新后再次 OOM，尚不能稳定运行。
- **后续**：增加 MLP 重计算以降低训练峰值显存。

### 2026-10-12 02:01 — 增加 MLP 重计算

- **相对上次**：selective recompute 显式使用 `core_attn mlp`。
- **结果**：Actor 初始化阶段遇到 `uncorrectable ECC error`，没有进入训练。这次失败不是训练 loss 或奖励导致的。

### 2026-10-12 02:07 — ECC 报错后重新提交

- **相对上次**：保留 16K 预算、16K context、`core_attn mlp` selective recompute 和关闭 optimizer CPU offload 的组合。
- **结果**：完成 57 次更新（19 rollouts），最大 grad norm 2.854678，之后主动停止以修正奖励逻辑；不是跑满 200 rollouts。
- **问题**：当时的奖励口径尚未完整对齐交接版，不能将该曲线当成交接实验的复现结果。

### 2026-10-12 15:42 — 恢复交接版奖励口径

- **相对上次**：恢复轨迹 outcome shaping、8 条轨迹组内 advantage 标准化，以及按导出记录数加权的 `rollout/raw_reward`。保留 LR `1e-6` / cosine、每轮 3 次更新、16K 训练预算。
- **结果**：完成 21 次更新（7 rollouts），最大 grad norm 4.100425；随后主动停止，继续对齐学习率和训练配置。
- **口径**：shaped reward 为 `(成功 ? 1 : -1) - 0.05 × T/30 - 0.2 × I(T>=30)`。成功率另按 session 去重统计，不能和 `rollout/raw_reward` 混用。

### 2026-10-12 21:02 — LR 5e-6 / constant，32K，冻结 vision

- **相对上次**：LR `1e-6 → 5e-6`，cosine → constant；训练 token 预算 16K → 32K；冻结 vision trunk。保留每轮 3 次更新、KL `0.02`、reference 每 rollout 更新、selective recompute；未开启 sandbox prelaunch。
- **结果**：完成 147 次更新（49 rollouts）后主动停止，进入预启动实验。出现 4 次 `grad_norm > 10`：更新 11、35、71、110，均是所在 rollout 的第 3 次更新；最大 grad norm 491186.094，最大未乘系数的 `train/kl_loss` 14710.166。
- **发现**：大梯度尖峰与重复更新后的 k3 KL 异常值相关。grad norm 是裁剪前值，`clip_grad=1.0` 不代表该指标必须小于 1。

### 2026-10-13 22:15 — 开启 sandbox prelaunch

- **相对上次**：开启 `agentic-prelaunch`，active 和预启动共享 12 个 group permits，最多 96 个 session；其余保留 32K、每轮 3 次更新、KL `0.02`。
- **结果**：停止前日志记录 14 次更新。第 3 次更新再次出现尖峰：grad norm 189.610、未乘系数的 KL loss 2.846，KL 项约占该点 total loss 的 96.5%。预启动没有消除这个训练稳定性问题。
- **后续**：将 KL 系数降至 `0.001`，每轮更新次数减至 2。没有逐 token 或分离 PG/KL 梯度证据，暂不把 KL 异常归因到具体 token；也未得到控制其他变量后的 prelaunch 独立加速比。

### 2026-10-14 00:58 — KL 0.001、每轮 2 次更新、24K，开启 dynamic filter

- **相对上次**：KL `0.02 → 0.001`；每轮更新 `3 → 2`，每次更新的逻辑轨迹数 `32 → 48`；训练预算 `32K → 24K`；开启基于轨迹 outcome 的 dynamic filter。保留 LR `5e-6`、prelaunch、冻结 vision 和每 rollout 更新 reference。
- **结果**：完成 2 次更新，最大 grad norm 1.159800，未出现大于 10 的尖峰；随后按要求主动停止并关闭 dynamic filter。窗口过短，不能据此判断长期稳定性或过滤收益。

### 2026-10-14 01:48 — 关闭 dynamic filter，开始持续训练

- **相对上次**：仅关闭 dynamic filter；保留 KL `0.001`、每轮 2 次更新、24K、LR `5e-6` / constant、prelaunch 和冻结 vision。
- **结果快照：2026-10-14 13:20:39**：完成 27 rollouts、54 次更新（0–53）；grad norm 最小/平均/最大为 **0.561670 / 0.954595 / 1.337857**，`grad_norm > 10` 为 **0 次**；最大未乘系数的 KL loss **0.003474**。该窗口内原来的大幅梯度尖峰未再出现。
- **曲线**：rollout 0 → 26，raw reward `-0.403194 → 0.144336`，session 成功率 `48.96% → 80.21%`；最新 actor training MFU 约 `74.39%`。首尾两个点不代表单调上涨或独立评测成绩，actor MFU 也不代表整个 rollout 周期的利用率。
- **结论范围**：KL 系数、更新次数和 token 预算共同变化，不能把稳定性改善单独归因于 KL 系数；实验仍在运行，尚未完成 200 rollouts。

### 2026-10-14 — 当前实验期间的代码与文档整理，未重启

- **脚本修改，未实跑**：训练/log-prob token 上限改为 48K；当前运行仍是 24K。按 `create-training-script` 整理数组、直接启动和自动入口识别；固定 LR `5e-6`、200 rollouts、每轮 2 次更新，移除间接变量 `OSWORLD_UPDATES_PER_ROLLOUT`。保留默认阻塞提交、不保存 checkpoint 的要求。
- **监控与 dump**：区分 session 与导出记录，补齐 session/task/turn/export 和解析重试标识；修正 raw reward 图的名称与纵轴；正常完成时关闭 metrics adapters。这些改动不构成一次新的对照实验。
- **时间审计，12:15:20 快照**：24 个 `perf/step_time` 合计 10小时21分18.736秒，平均 **1553.281 秒/rollout**；加计时前初始化 3分57.812秒及尚未上报的尾段 1分27.839秒，与 Job 总时长 10小时26分44.387秒对齐。该指标每 rollout 一个点，包含两次更新；rollout 1–23 的累计时间与日志跨度仅差 0.458 秒。
- **开源整理**：提供 `env.example.sh`，直接读取忽略的 `env.local.sh`，删除多余 `env.sh` 及其忽略例外；更新中英文文档，缺失/空任务清单明确报错。未公开内部端点、密钥或私有路径。
- **验证结果**：提交前全部 pre-commit 通过，相关测试 61 passed、1 个 opt-in live Ray Serve 测试 skipped。隔离环境验证过脚本提交参数；48K 尚未完成真实 GPU 训练验证。共享前缀训练、广播和 GPU→CPU 拷贝仅做分析，没有实施新的优化算法。
