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
