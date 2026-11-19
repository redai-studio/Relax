# OSWorld 决策 hidden states

[English](HIDDEN_STATES.en.md)

默认关闭。设置 `RELAX_DECISION_HIDDEN_DIR` 后，在实际 rollout 策略模型的同一次生成前向中采集每轮决策表示。没有 training forward hook、重放或额外模型前向，不改变输入、图片窗口、数值采样参数、动作解析、reward 或训练样本。思考和动作文本一起池化。

## 表示与对齐

设 prompt 为 `p`，本轮生成 `y1, y2, …, yn`。用于预测 `y1` 的向量来自 prefill 中 `p` 的最后位置；用于预测 `yi` 的向量来自输入 `yi-1` 的 decode 位置。取最后一层 **final normalization 后、LM head 前**的向量。它不是将 `yi` 输入后在 `yi` 位置得到的表示。

只按实际采样位置取行，prompt 的其它位置、历史文本和图片位置不参与池化。EOS、模型额外终止 ID、请求 stop token ID 不计入均值；思考和工具标记不会因为属于 tokenizer special tokens 就被全部过滤。字符串 stop 的普通文本 token 仍计入。以 FP32 在 GPU 累加 sum/count，请求结束时只传出一次均值及 count，不传回所有 token 向量。count 为 0 时保存零向量；分析时应排除。

## 支持范围与启用

当前环境的包元数据和 Git 标签报告 SGLang **0.5.9**（HEAD `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`），但源码有大量本地修改。仓库 Docker 默认则为 **0.5.17**。实现同时检查包版本和四个关键源码文件 SHA256，**仅接受已检查的本地源码快照**；不声称支持原版 0.5.9 或 Docker 0.5.17。其它快照必须先核对接口与对齐并更新校验值，不能仅删除检查。

现有 `return_hidden_states` 包含逐 token CPU 传输，并可能优先选 normalization 前的表示，因此不能直接使用。扩展在现有 scheduler 子进程入口安装两个小范围 hook：LogitsProcessor 保留实际采样行的 post-norm 向量，scheduler 累加并保存。使用现有 `custom_params` 传递 task/session/turn 元数据，不调整采样数值。

只支持 `Qwen3VLForConditionalGeneration`、普通自回归、eager、非 overlap、非流式 backend 请求。支持普通 TP：该模型最终 hidden dimension 在 TP rank 上完整复制，只由 TP rank 0 池化和保存。PP 必须为 1，DP 必须为 1；不支持 speculative、LoRA、DP attention、torch compile、prefill/decode 分离、grammar、`return_hidden_states`、partial rollout、fully async、agentic prelaunch。不支持时明确报错。多台独立 engine 可以各自输出分片。

1. 确认使用上述匹配源码的引擎和本分支 Relax；引擎、SessionShard、API、agent 进程均需收到同一个环境变量。目录应为一次实验独有的持久共享目录；若使用节点本地盘，离线分析前汇总所有节点分片。只给 agent 设置变量是不够的。

2. 在 `scripts/run-qwen3vl-8B-osworld-8xgpu.sh` 删除 `--agentic-prelaunch`，在 `SGLANG_ARGS` 添加 `--sglang-disable-cuda-graph --sglang-disable-overlap-schedule`。这些限制只用于开启采集的运行；关闭采集时保留原配置。

3. 用现有 Ray runtime env 传递开关，按原有方式提交训练。例如，合并到现有 `RUNTIME_ENV_JSON` 的 `env_vars`：

   ```json
   {"RELAX_DECISION_HIDDEN_DIR": "/path/to/run/decision-hidden"}
   ```

   完整环境变量示例（已有其他 runtime env 配置时应合并）：

   ```bash
   export RUNTIME_ENV_JSON='{"env_vars":{"RELAX_DECISION_HIDDEN_DIR":"/path/to/run/decision-hidden"}}'
   ```

取消设置该变量即关闭；不安装 hook、不保留额外 hidden 张量、不创建这些输出文件。Agent 的 Python 环境需要能导入本分支 `relax.agentic.decision_hidden`。

## 文件和关联

成功轨迹 archive 会清理失败或超配额轨迹，所以研究数据使用独立的 process 分片，无数据库、服务、队列或共享文件写锁：

```text
decision-hidden/
  requests-<worker_uuid>-<pid>.jsonl
  outcomes-<worker_uuid>-<pid>.jsonl
```

每个引擎 worker 写自己的 requests 文件；每个 agent 进程写自己的 outcomes 文件。请求完成时直接落盘，不依赖 HTTP 响应是否送达。以下是缩短到三维的格式示例，不是真实模型输出：

```json
{"task_id":"task-a","session_id":"session-a","trajectory_id":"session-a","turn_id":0,"backend_request_id":"req_session-a_0:0","request_id":"req_session-a_0:0:unique-generation-id","policy_version":"engine-instance:7","checkpoint":"/path/to/checkpoint","vector":[0.1,0.2,0.3],"token_count":12,"hidden_dim":3,"layer":"final_norm","semantics":"mean_final_norm_pre_lm_head_predicting_generated_tokens_v1","finish_reason":{"type":"length","length":12}}
{"task_id":"task-a","session_id":"session-a","reward":0.0,"success":false,"evaluation_error":null,"stop_reason":"env_done"}
```

`turn_id` 是 OSWorld 的实际动作轮次，同轮解析重试保持相同值。`backend_request_id` 是 Relax 稳定请求 ID；`request_id` 追加生成实例 UUID，连 HTTP 层重新执行同一请求也不会冲突。`trajectory_id` 使用 resident session ID，与成功 archive 的独立随机 trajectory ID 不同，可通过 archive 的 `session_id` 关联。

`policy_version` 是推理 engine 实例 UUID 加实际成功权重修改的单调修订号；磁盘加载、tensor/distributed/IPC 更新和权重后处理都会推进它。它不是训练 global_step，也不是权重内容哈希。分块更新可能推进多次；不同 engine 即使装载相同权重也有不同实例 ID。`checkpoint` 记录初始或最近磁盘加载路径，在线更新后不代表该路径包含最新权重。请求中途修订号变化会报错，不混合两个版本。

最终 reward 使用原 OSWorld evaluator 的返回值；`success` 表示无 evaluator 错误且 reward 恰好为 1。evaluator 错误时 success 为 null。取消、进程崩溃等未完成评价的任务可能只有请求，没有 outcome，不能把缺失当作失败。文件系统故障会报错，不静默丢弃数据；进程被强制终止时最后一行可能不完整，读取前应检查。

离线关联示例：

```bash
python examples/osworld_agentic/scripts/join_decision_hidden.py /path/to/run/decision-hidden > joined.jsonl
```

也可直接读取向量：

```python
from pathlib import Path
from examples.osworld_agentic.scripts.join_decision_hidden import joined_records

for row in joined_records(Path("/path/to/run/decision-hidden")):
    if row["outcome_available"] and row["success"] is not None and row["token_count"] > 0:
        vector, success, reward = row["vector"], row["success"], row["reward"]
```

这里仅提供采集与关联，不进行聚类、诊断器训练或任务合成。
