# ReTool：Relax Agentic 训练示例

本目录将 ReTool 的多轮 Python 工具循环接入 resident `relax/agentic`。启用 `--use-agentic-rollout` 后，每条轨迹由一个 managed agent 进程执行，通过 Session HTTP API 请求模型；Agentic 管理上下文、真实生成 token/logprob、工具反馈 mask、权重版本和训练导出。

实现独立放在本目录，复用 `examples/retool/` 的数据准备、sandbox 生命周期、代码校验、答案判分、词表训练 hook 和 checkpoint 校验器。现有原生 rollout 示例继续使用原脚本。

## 启动与恢复

在安装好 Relax 训练依赖、Megatron 和 sandbox 运行环境的独占单机 8 卡容器中运行。`MODEL_DIR` 指向具体 HF 模型（例如前序 SFT 的 `hf_49`，包含 `tokenizer.json`），`DATA_DIR` 包含原示例已准备的 `train.jsonl`，`SAVE_DIR` 是持久化 checkpoint 根目录。

```bash
export MODEL_DIR=/path/to/sft/hf_49
export DATA_DIR=/path/to/retool-data
export SAVE_DIR=/path/to/checkpoints
bash examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

脚本自动 source `scripts/entrypoint/local.sh`，随后阻塞提交 Ray Job。该入口会清理本机进程，适用于新任务的独占容器。已有共享 Ray 集群使用框架入口：

```bash
RAY_ADDRESS=http://head-host:8265 \
  bash scripts/entrypoint/ray-job.sh \
  examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

默认保存位置为 `$SAVE_DIR/qwen25-7b-retool-agentic-grpo-8xgpu`，与原示例的默认实验目录区分。每 25 步保存模型、optimizer、scheduler 和抽样 RNG 状态，HF 导出位于同目录下的 `hf_{rollout_id}`。保持 `SAVE_DIR`、`EXP_NAME`、数据内容、seed、batch/K 不变，再执行相同脚本即可从完整 checkpoint 恢复。只复制 `hf_*` 目录属于从权重开始新实验。

`PROJECT_NAME` 默认 `Relax/retool-agentic`，`EXP_NAME` 默认 `qwen25-7b-retool-agentic-grpo-8xgpu`；允许通过环境覆盖。ClearML 每次启动创建带时间戳的 task，模型恢复目录保持稳定。默认阻塞跟随日志；只有显式设置非空 `RAY_NO_WAIT` 才非阻塞。

工具服务由现有 `examples.retool.train` 在 Ray Job 生命周期内管理。自动选择 Landlock 或 Apptainer 后端，镜像缓存仍为 `$SAVE_DIR/.retool-sandbox/`；可选 `SANDBOX_IMAGE` 或 `SANDBOX_FUSION_URL` 的配置与[原示例](../retool/README.md)相同。服务 URL 会传入 Session 所在 worker，tokenizer 路径通过 `--agent-env` 传入 agent 进程。

若机器出现 NVSwitch multicast 建链失败并伴随 NCCL CUDA 999，可设置 `NCCL_NVLS_ENABLE=0` 关闭 NVLS 后重启。脚本会将显式设置传入 Ray worker，覆盖入口按 NVLink 拓扑自动生成的值；未设置时沿用入口默认值。

## 默认训练配置

| 参数               | 值                                                            |
| ------------------ | ------------------------------------------------------------- |
| 算法与训练步数     | GRPO，200 rollout，每 rollout 一次训练更新                    |
| Batch              | 64 个有放回抽样的问题 × 16 条轨迹 = 1024                      |
| Resident 并发      | 16 个 group，最多 256 个 agent 进程                           |
| 模型请求容量       | 8 个 SGLang engine × 32 请求；Session 使用 consistent hashing |
| 工具服务容量       | 复用原服务的 32 路执行并发，单次 30 秒 / 1 GiB                |
| Token 预算         | prompt 2048；回复与工具反馈合计 8192；上下文 10240            |
| Assistant 轮数上限 | 16，终止轮不会继续执行代码                                    |
| Sampling           | temperature=1，top-p=1，top-k=-1                              |
| 优化               | lr=1e-6；clip=0.2；KL/entropy=0；Actor TP=2                   |

每个问题的 16 条轨迹独立执行，不共享工具 Python 状态。数据源接受 Agentic 的分批 group 请求，抽题顺序与原生示例相同；只在完整 batch 边界保存 RNG。

本版覆盖同步训练：未接入 Eval，禁止 prelaunch、partial rollout、fully async、动态过滤与额外 oversampling，以保留完整 batch 的抽样与恢复约束。不要传入 `EVAL_CONFIG` 或 Eval 参数。分布式 GPU 训练和 256 个 resident 进程的实际吞吐仍需在新任务环境验收，本地 CPU 检查不能证明 GPU 吞吐或训练已启动。

## 上下文、mask 与奖励

模型继续生成原作者的 `<code> ... </code>` 文本，无需函数调用或 reasoning parser。每次 Chat Completions 请求使用完整历史、固定模板、`stream=false`；URL 保留 `RELAX_BASE_URL` 的路由前缀，并使用 `Bearer RELAX_SESSION_ID`。

专用模板使初始 prompt 与 Qwen2.5 模板一致。工具结果以非空的完整 `\n<interpreter>\n...\n</interpreter>\n` 内容追加为 tool message；模板将其原样拼回 assistant 流，不额外插入 ChatML 轮次边界。即使 stdout 为空，wrapper 仍完整保留。后续请求复用原 assistant 返回消息，SessionForest 只形成一条线性路径并隐式导出一份训练样本。

Assistant token 的 loss mask 为 1，实际反馈 token 为 0；logprob 来自 Agentic 的原生后端记录。SGLang sampling params 中的 `logit_bias` 屏蔽 padded token `[151665, 152064)`，训练端复用原词表 hook。原采样 seed 的按样本、按轮偏移保持一致。原生 API 的 `</code>` stop 只代表一个回复结束；若 agent 因总预算或轮数终止，reward adapter 将导出样本标为 truncated。

奖励继续采用原 verifier 的答案分数，减每次工具失败 0.1，无下限；基础设施故障不转换成工具惩罚。代码只执行一次，无 HTTP 重试或历史代码重放。Agent 进程失败遵循 Agentic 的整组丢弃机制；本例的完整 batch/权重版本校验不支持跨 batch 自动补样恢复。

模型 HTTP 请求不设置 wall timeout，以允许 Runtime 暂停请求；`--agent-timeout 1800` 是独立的进程 active-time 上限。Sandbox 请求继续使用原客户端的 70 秒 wall timeout。

## 本地验证

```bash
python -m pytest examples/retool_agentic/tests -q

# 可选：使用实际训练模型的 tokenizer 重跑同一组检查。
RETOOL_TEST_MODEL_DIR="$MODEL_DIR" \
  python -m pytest examples/retool_agentic/tests -q
```

测试通过真实 `SGLangMessageCompiler`、上下文匹配和 `SessionForest` 导出，与原 rollout 逐 token 比较输入、mask、logprob、终止状态和轮数；包含多轮、空输出、代码错误、非法调用与预算终止。同时验证真实子进程的 HTTP/JSON 交互、路由前缀、抽样恢复和启动脚本。模型响应与 sandbox 执行在 CPU 测试中由固定结果替代，不占用 GPU。
