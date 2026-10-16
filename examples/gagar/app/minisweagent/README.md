# Mini-swe-agent GAGAR 示例

本目录包含 SWE 的 agent server、客户端、SIF 缓存、bridge、grader、任务适配器和 sandbox 工具。提示词与 grader 配置统一放在 GAGAR 根目录的 `grader.yaml` 中，不导入或启动其他示例 recipe。组内评分前，仍按原有流程计算 actor 提交的原始 0/1 测试结果。

## 环境与数据

在指定执行节点安装 Apptainer，使用 Bash 5.1 或更新版本，并为 agent 进程创建独立虚拟环境：

```bash
uv venv /local/path/mini-swe-agent-venv --python 3.12
source /local/path/mini-swe-agent-venv/bin/activate
uv pip install mini-swe-agent==2.4.1 flask pyarrow datasets swebench==4.1.0 huggingface_hub
```

在 Relax 仓库根目录配置模型、数据、缓存和运行目录：

```bash
export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export SAVE_DIR=/path/to/checkpoints
export R2E_DATA_PATH=/path/to/r2e-data
export R2E_SIF_DIR=/path/to/sif-cache
export EXP_ROOT=/path/to/new-experiment
R2E_SETUP_SKIP_SIFS=1 bash examples/gagar/app/minisweagent/sandbox/setup_r2e_data_and_sifs.sh
RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh examples/gagar/run_minisweagent.sh
```

`MODEL_DIR` 下应包含 Qwen3.5-35B-A3B 和默认 grader 模型 Qwen3.8-27B，也可以通过 `GAGAR_MODEL_PATH` 指定其他 grader 模型路径。准备脚本下载 R2E 训练与评估的 parquet 数据和 wheels，镜像在首次使用时按需转换。

提交进程只负责提交 Ray job。`sandbox/driver.sh` 管理 server 和训练进程，检查服务就绪状态、处理取消信号，并等待清理完成。

## 文件职责

- GAGAR 根目录的 `run_minisweagent.sh`：SWE 训练与 split GenRM 启动入口，不包含独立 baseline 或 PD recipe。
- `sandbox/driver.sh`：Ray job 内的进程监督。
- `sandbox/agent_server.py`、`sandbox/agent_client.sh`、`sandbox/agent_config.yaml`：actor 执行与原始判题。
- `sandbox/sif_cache.py`、`sandbox/setup_r2e_data_and_sifs.sh`、`sandbox/check_r2e_sifs.py`：数据与镜像准备。
- `group_rm.py`、`sandbox/bridge.py`：接入框架已有的 group RM hook，将工具调用路由至候选环境。
- `grader.py`、`scoring.py`、`sandbox/git_evidence.py`：grader 环境适配、权重映射与 actor 修改的证据冻结。
- GAGAR 根目录 `grader.yaml` 的 `tasks.minisweagent`：system/instance 提示词、环境与 GenRM 配置。

## 评分链路与资源布局

本 recipe 保留 mini-swe-agent 原有的二值 outcome 判题，通过 Relax 的 `--group-rm` 调用组内 grader。grader 产生的 weights 传入 `compute_group_rewards`，最终由 Relax 执行组内减均值。

完成上述环境与数据准备后，运行根目录的 `run_minisweagent.sh`。资源采用 split colocate 布局：actor 使用 8 张 GPU，rollout 和 GenRM 各使用 4 张 GPU，GenRM 的 TP 为 4；训练与推理阶段分时复用 GPU。`GAGAR_MODEL_PATH` 默认为 `${MODEL_DIR}/Qwen3.8-27B`。SWE grader 使用 65,536-token context，单次回复预算为 2,048 tokens。此入口始终启用组内评分，不使用 deferred reward hook 或自定义 advantage，也不包含独立 SWE baseline。

`run_minisweagent.sh` 默认启用按需 SIF 缓存，容量为 128 GiB，最多同时转换两个镜像。使用 `R2E_SETUP_SKIP_SIFS=1 bash examples/gagar/app/minisweagent/sandbox/setup_r2e_data_and_sifs.sh` 仅准备数据和 wheels；镜像首次使用时转换到 `R2E_SIF_DIR`。缓存临时目录与镜像仓库访问方式可通过下文环境变量配置，`AGENT_SERVER_SIF_CACHE=0` 则使用预构建镜像。

SIF 租约随保留的 sandbox 一起经过 outcome 判题和组内评分，只有实例停止后才释放。缓存容量不包含构建临时空间，正在使用的镜像不会被驱逐。

## Grader 进程边界

`GAGAR_GRADER_COMMAND` 是由 agent server 在独立进程中启动的可信命令，接收 `--input <manifest.json>` 和 `--output <result.json>`。manifest 包含任务、按顺序排列的候选、原始 outcome、测试前冻结的 patch/轨迹、原始测试输出、Relax GenRM 服务地址，以及短期有效的 bridge 访问凭证。结果格式为 `{"weights": [...], "rewards": [...]}`，顺序与候选一致。`rewards` 可省略；修正只允许将通过改为失败，不能将失败改为通过。

SWE patch 以环境准备完成、actor 开始执行前的已跟踪工作树为基准。通过临时 Git index 冻结该基准，不改变 HEAD 或真实 index。这样可排除任务镜像预先存在的未提交修改，同时保留 actor 的改动，包括对同一文件的后续修改。manifest 保留 `baseline`（HEAD、tree 和初始 diff）与 `head_patch` 供审计。与原 harness 一致，未跟踪文件只有加入暂存区后才进入 patch。证据采集失败不影响原始二值判题；只要任一通过候选的证据不完整，整组质量评分就退回原始 outcome。

默认命令为 `examples.gagar.app.runner --task minisweagent`，使用 mini-swe-agent 虚拟环境，按 YAML 选择共用 GRS 入口。`DefaultAgent` 循环携带完整消息历史调用 Relax 已有的 GenRM `/generate` 接口。GenRM 当前返回文本，recipe 从每次回复中解析一个 JSON 工具动作，不修改框架 API。可通过 `GAGAR_CONFIG` 选择另一份统一 grader YAML 配置。默认预算为 40 次调用、1500 秒。

模型生成的命令只能通过 `bash(rollout_n, command)` 执行。bridge 在当前固定候选组内解析 `rollout_n`，将命令送到保留的候选环境，不在宿主执行。缺失或非法候选 ID、未知参数、过期凭证均会被拒绝。候选 ID 从 0 开始，不暴露底层服务地址。评分任务结束后，保存的 grader 证据会移除访问凭证。

## 生命周期与失败处理

完成的训练 session 将 sandbox 移交 bridge，同时立即释放 rollout worker。评估、基础设施故障与取消的 session 沿用原清理流程。已完成的通过和失败候选都可供检查，但 grader 不能将未正式提交的轨迹修正为通过。

评分完成、失败或取消时，释放整组环境。独立租约到期机制也会回收在进入 group RM 前已被丢弃的组。清理会等待正在执行的工具命令结束；失败的清理记录会保留以便重试，只有实例停止后才移除临时目录。重复 GET 与释放调用是安全的。原 server 记录的 TTL 与保留 sandbox 的所有权分别管理。

| 环境变量                    | 默认值 | 含义                                 |
| --------------------------- | ------ | ------------------------------------ |
| `GAGAR_CONCURRENCY`         | 4      | 同时运行的 grader 进程数             |
| `GAGAR_TIMEOUT_SECONDS`     | 1800   | 包含排队时间的整组评分时限，单位为秒 |
| `GAGAR_RETENTION_SECONDS`   | 14400  | 尚未被评分任务接管的组的最长等待秒数 |
| `GAGAR_MAX_RETAINED_GROUPS` | 32     | 最多保留的候选组数                   |

保留容量不足不会阻塞 rollout worker：无法获得 sandbox 的组退回原始 outcome。应根据 partial-rollout 或暂停时长配置保留容量与租约；租约到期只会失去该组的质量评分，不会丢失原始 outcome reward。

评分不可用或格式非法时，整组回退。返回字段包括 `score`（用于训练）、`test_reward`（原始二值结果）、`outcome_reward`（修正后的结果）、`quality_weight` 和 `grader_fallback`。评估使用 `test_reward`。等价 `score` 可以超过 1，不能裁剪或用来判断测试是否通过。应保留组内减均值，并关闭 GRPO 标准差归一化。

## 质量评分（论文 Appendix A.1）

Grader 联合检查同组候选，对原始通过候选按以下五项标准打 1–5 的整数分：approach（解决思路，30%）、precision（修改准确性，25%）、minimality（最小改动，20%）、side-effect avoidance（避免副作用，15%）、consistency（与仓库惯例一致，10%）。程序根据评分计算初始顺序与分档，模型不直接指定任意权重。

- 优先判定 T3：存在未要求的重写、针对测试的绕过、approach=1，或 minimality 与 side-effect avoidance 同时不高于 2。
- T1 要求每项均不低于 4，且没有严重流程问题或尚未解决的回归；其余候选为 T2。
- T1 最高并列名次权重为 1.0，其他 T1 名次为 0.9。
- T2 按不同并列名次从 0.85 到 0.4 插值；只有一个并列名次时取 0.85。
- T3 权重为 0.2。权重按不同并列名次数计算，不按排在前面的候选数量计算。

调整排名必须给出与任务相关的解释。确认依赖泄漏答案或外部解答时，需提供独立证据，并在排名及 GAGAR 统计前将 outcome 修正为 0。仅有针对测试的绕过属于 T3 质量问题，不会自动修正 outcome。结果保留各项评分、推导出的分档与最终排名，便于 review。

`read_evidence` 分页读取测试前冻结的 patch/轨迹和原始测试输出；`bash` 检查当前测试后的 sandbox。grader 必须区分候选原始改动与测试或后续检查引入的变化。与参考 patch 的相似度不是质量标准。

## SIF 缓存配置

| 环境变量                             | 默认值                              | 用途                                   |
| ------------------------------------ | ----------------------------------- | -------------------------------------- |
| `AGENT_SERVER_SIF_CACHE`             | 本启动脚本中为 `1`                  | 按需转换；`0` 表示使用预构建 SIF       |
| `AGENT_SERVER_SIF_CACHE_GB`          | `128`                               | 镜像缓存容量；租约未释放的镜像不可驱逐 |
| `AGENT_SERVER_SIF_BUILD_CONCURRENCY` | `2`                                 | 同时转换的镜像数                       |
| `AGENT_SERVER_SIF_BUILD_TIMEOUT`     | `1800`                              | 镜像转换超时，单位为秒                 |
| `AGENT_SERVER_SIF_BUILD_TMP_DIR`     | agent-server 临时目录 / `sif-build` | 构建临时目录，不计入镜像缓存容量       |
| `AGENT_SERVER_SIF_REGISTRY`          | 未设置                              | 可选的镜像仓库镜像源                   |
| `AGENT_SERVER_SIF_PROXY`             | 未设置                              | 仅用于镜像转换的代理                   |

提交前设置这些变量，启动脚本会将其传入 Ray job。未设置 `R2E_SETUP_SKIP_SIFS` 时，`sandbox/setup_r2e_data_and_sifs.sh` 也可预构建镜像。预构建与按需转换两条路径均保留，无需引入其他训练 recipe。

统一配置位于 GAGAR 根目录的 `grader.yaml`，本任务选择 `tasks.minisweagent`。system/instance 提示词、工具超时和推理预算都在该文件中修改，原始 rollout 与判题配置保持不变。
