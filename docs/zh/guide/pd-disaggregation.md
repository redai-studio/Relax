# Prefill–Decode 分离

Prefill–decode（PD）分离将 prompt 处理和逐 token 生成交给不同的 SGLang engine。编码任务中的每次工具执行都会为后续请求增加上下文；分离后，prefill engine 可以处理新上下文，decode engine 则负责生成回复。实际收益取决于 prompt 长度、并发量和 KV 状态传输开销，开启 PD 本身不保证吞吐提升。

[Mini SWE Agent PD 脚本](../../../examples/mini_swe_agent/run_mini_swe_agent_pd.sh) 使用 Qwen3.5-35B-A3B 运行 R2E-Gym 任务，配置为 **8 卡、2 个 prefill engine 加 2 个 decode engine，每个 engine 使用 TP2**。它沿用已有 Mini SWE agent、沙箱、任务数据、奖励计算和 GRPO 训练配置。[原始非 PD 脚本](../../../examples/mini_swe_agent/run_mini_swe_agent.sh) 仍保留为独立入口。

## GPU 分配和请求流程

| Rollout 阶段的角色 | Engine 数量 | 每个 engine 的 GPU 数 | GPU 总数 |
| --- | ---: | ---: | ---: |
| Prefill | 2 | 2（TP2） | 4 |
| Decode | 2 | 2（TP2） | 4 |

训练与 rollout 通过 `--colocate` 分时复用这 8 张 GPU。Megatron 沿用原 Mini SWE 配置：TP2、CP4、PP1、EP8、expert TP1。PD 分离的是 **rollout 推理内部的阶段**；本示例没有将训练与 rollout 放在不同 GPU 池，也没有开启 fully async 训练。

1. Relax 为一个任务建立包含 8 个常驻 Session 的 Group。
2. 每个 Session 启动 `agent_client.sh`，请求 `agent_server.py` 在 Apptainer 沙箱中运行 Mini SWE。
3. Mini SWE 将完整对话发给 Relax 的 agentic Chat Completions API。PD router 选择 prefill 和 decode engine，并将 prefill 状态传给 decode 完成生成。
4. Agent 执行工具并继续请求模型。正式提交结果后，server 执行任务测试，返回标量 reward；Relax 导出轨迹用于 GRPO。
5. 到达训练边界后，Relax 更新全部 prefill 和 decode engine 的权重，再开始下一轮 rollout。

脚本将通用和 prefill 路由策略设为 `consistent_hashing`，通过 Session routing key 保持 prefill 亲和性。Decode 使用 `power_of_two`，router 在两个可用的 decode worker 之间选择执行位置。这些策略不会为 Session 的整个生命周期预留某个 engine。

## 准备环境

使用**单机 8 卡、供本任务独占的 Ray 集群**，GPU 显存需要容纳 TP2 的 Qwen3.5-35B-A3B、KV 状态和共卡训练。请在该节点提交任务。仓库的 Ray 入口会清理残留的 Relax worker 和 Serve 应用，因此应先确认集群可供本任务使用。

使用仓库的训练环境及其配套 SGLang patch。当前 `docker/patch/latest/sglang.patch` 对应 SGLang v0.5.17。PD 使用 Mooncake 传输后端，需要在训练环境中准备对应依赖。Router 启动成功并不代表模型初始化和状态传输已经可用。

按 [Mini SWE 准备说明](../../../examples/mini_swe_agent/README.md#prepare-the-mini-swe-agent-venv) 准备虚拟环境、parquet 任务数据、Python wheels 和匹配的 SIF 镜像。训练读取 `R2E-Gym-Lite`；即使关闭 Eval，agent server 启动时也会加载配置中的 `SWE-Bench-Lite` 数据集。使用 Bash 5.1 或更新版本，准备 Apptainer，并确保 `jq`、`curl` 可用。Qwen3.5 checkpoint 应位于 `$MODEL_DIR/Qwen3.5-35B-A3B`。

## 启动 PD 示例

每次运行使用新的输出目录，并按实际环境替换路径、Ray head 地址和网卡名：

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs
export RUN_ID=mini-swe-pd-$(date +%Y%m%d-%H%M%S)
export SAVE_DIR=/path/to/checkpoints/${RUN_ID}
export EXP_ROOT=/path/to/experiments/${RUN_ID}
export RAY_ADDRESS=<ray-head-ip>:6379
export RAY_JOB_ADDRESS=http://<ray-head-ip>:8265
export NCCL_SOCKET_IFNAME=<network-interface>

NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_pd.sh
```

提交是异步的。Ray driver 启动 agent server、等待 HTTP 就绪，然后从当前 checkout 启动训练。训练结束或作业停止后，由 driver 关闭 server。Server 绑定到 Ray 入口检测出的节点地址，默认使用 `8765` 端口；若端口被占用，可设置 `AGENT_SERVER_PORT`。

以上命令执行 3 轮 rollout 和训练，每轮包含 1 个任务、8 个样本。这是完整真实任务的小规模运行：agent 仍最多进行 40 次模型调用，每次回复上限为 8192 tokens，上下文上限为 32768 tokens。任务执行和评分可能比模型生成本身耗时更长。

| 设置 | 脚本默认值 | 含义 |
| --- | --- | --- |
| `NUM_ROLLOUT` | `1000` | Rollout／训练轮数；以上命令设为 `3` |
| `ROLLOUT_BATCH_SIZE` | `16` | 每轮任务 Group 数；以上命令设为 `1` |
| 每个 Group 的样本数 | `8` | 与 `agent_config.yaml` 的 `r2e.datasets.train.samples` 一致 |
| `AGENT_SERVER_TRAIN_CONCURRENCY` | `ROLLOUT_BATCH_SIZE * 8` | Server 槽位数，必须覆盖全部常驻训练 Session |
| `AGENT_SERVER_REWARD_CONCURRENCY` | `24` | 同时执行任务评分的最大数量 |
| `ENABLE_EVAL` | `0` | 可选的 SWE-Bench Eval，每 100 轮执行一次 |
| `AGENT_SERVER_EVAL_CONCURRENCY` | `64` | 独立 Eval 槽位，开启 Eval 时至少为 `50` |
| `MAX_RESPONSE_LEN` / `MAX_CONTEXT_LEN` | `8192` / `32768` | 每次调用的生成长度和上下文上限 |

全局训练 batch 为 `ROLLOUT_BATCH_SIZE * 8`。请保持每组 8 个样本与 agent server 配置一致。若 server 槽位不足以容纳全部常驻 Session，可能无法让完整 Group 到达首次模型请求；脚本会拒绝这种配置。开启 Eval 还需要对应数据集、SIF 镜像，以及 Eval setup 命令要求的 wheels。

Agent 的单次模型请求 timeout 和 Relax 的 agent 进程活跃时间上限均为 7200 秒，两者约束不同的执行环节。本示例默认关闭 prelaunch、partial rollout、Session KV lifecycle 和 program admission。

## PD 配置

脚本通过 `--sglang-config` 加载 [pd-2p2d-tp2.yaml](../../../examples/mini_swe_agent/pd-2p2d-tp2.yaml)：

```yaml
# 8 shared GPUs: two prefill engines and two decode engines, each with TP2.
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: prefill
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: false
          hicache_storage_backend: null
      - worker_type: decode
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: false
          hicache_storage_backend: null

```

`num_gpus` 表示一个角色分配到的 GPU 总量，`num_gpus_per_engine` 决定如何拆成多个 engine。每个角色分配 4 张卡、每个 engine 使用 2 张卡，因此分别得到 2 个 engine。P/D 两侧都显式关闭 hierarchical cache，且不配置其存储后端。普通的 GPU 缓存行为遵循 SGLang 默认值，本示例不引入多级缓存配置。

## Router 启动与故障恢复

使用 `power_of_two` decode 路由时，Relax 在创建 Agentic Session 前预留 router 地址，先启动实际的 prefill 和 decode engines，再以这些 worker 启动 router。确认预期 worker 的角色、bootstrap port 和健康注册状态都正确后，才发布 engine groups。Engine 接管已有的初始注册，不再重复添加自己。

这条托管启动路径要求使用框架启动的原生 SGLang engines，`dp_size=1`，至少一个 prefill engine 和两个 decode engines。本示例为两组各两个。外部 engines、自定义 engine 类、DP-aware worker identity，以及混合 regular/PD groups 不适用这条路径。请使用这里的固定拓扑；目前不支持 PD 动态扩容。

示例开启了 Relax 的容错机制。替换死亡的托管 PD engine 前，Manager 会先确认它的旧 router 注册已移除。新 engine 的注册必须达到健康状态，恢复才会返回；训练循环在下一轮 rollout 前更新恢复 engine 的权重。启动或注册校验失败时，Relax 清理本次拥有的 engines 和 router。正在执行的 Agent 请求能否继续取决于请求恢复条件，部分故障仍需重启任务。

若希望 decode 也采用 consistent hashing，可在启动命令末尾追加以下参数，任务和八卡拓扑保持不变：

```bash
--sglang-router-decode-policy consistent_hashing
```

此时使用已有的动态注册路径，上述托管静态启动流程仅适用于 `power_of_two`。

## 查看进度和停止任务

```bash
ray job status --address "$RAY_JOB_ADDRESS" "$RUN_ID"
ray job logs --address "$RAY_JOB_ADDRESS" "$RUN_ID"
tail -f "$EXP_ROOT/agent_server/server.log"

ray job stop --address "$RAY_JOB_ADDRESS" "$RUN_ID"
```

检查任务 setup、连续的模型／工具调用、带 reward 的 Session 完成、rollout 导出和 optimizer step。PD 运行还应确认注册了 2 个 prefill、2 个 decode engine，且每次更新完成后四个 engine 的权重版本一致。作业提交成功或 agent server 就绪，都不足以说明完整训练循环已经跑通。

Agent server 的日志和 Session 事件位于 `$EXP_ROOT/agent_server`，Relax 的 rollout 明细位于 `$EXP_ROOT/dump`。训练结束后 driver 关闭本次 server，server 的退出处理负责清理所属 Session 和 Apptainer 实例。

启动失败时，先查看作业日志和 server 日志再重新提交。端口绑定失败表示 agent 端口已有服务占用；parquet、SIF、wheel 缺失应检查 Mini SWE 准备步骤；PD 初始化或传输失败应检查模型与后端版本、Mooncake 依赖和通信网卡。遇到 GPU OOM 时可先降低 `ROLLOUT_BATCH_SIZE`，在保留 2P2D TP2 分配的前提下检查显存上限。

常驻 Session 与轨迹导出的契约见 [Agentic Rollout](./agentic-rollout.md)。训练和 rollout 的独立调度见 [Fully Async](./fully-async-training.md)。
