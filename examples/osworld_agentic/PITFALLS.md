# OSWorld agentic 踩坑清单

这份文档汇总了在把 hybrid-routing-agent 接入 Relax agentic pipeline 过程中
容易踩、但很难从代码本身看出来的坑。全部有据可查（对应到 memory 或过往 bug
调试记录）。想改 launcher / agent 前先扫一眼。

______________________________________________________________________

## 1. `RELAX_BASE_URL` 末尾斜杠 → 404

**症状**：session 干净启动、又干净退出，reward=0，
`Prepare-owned managed agent session completed before producing a chat IR`。

**根因**：`resolve_chat_api_base_url()`（`relax/agentic/pipeline/runtime.py`）
返回带尾斜杠的 base_url。任何用字符串拼接 `f"{base_url}/chat/completions"`
的裸 httpx 客户端会得到 `…//chat/completions`（双斜杠），撞不到
`AgenticChatAPIService` 任何一条路由 → 404。OpenAI SDK 自己会归一化，所以
症状只在自写客户端上出现。

**做法**：`run_agent_app.sh` 里用
`export OPENAI_BASE_URL="${RELAX_BASE_URL%/}/v1"` 显式剥掉尾斜杠。

（memory `f3c6b412`）

______________________________________________________________________

## 2. `/v1` 后缀不能省

**症状**：422 `Field 'request' required (loc=['query','request'])` 或
silent-success（exit 0，reward=0）。

**根因**：`AgenticChatAPIService` 用堆叠 `@app.post` 暴露 `/`、
`/chat/completions`、`/v1/chat/completions` 三条路径，**只有
`/v1/chat/completions` 稳定注册**。少了 `/v1` 会走到 fallback，字段名错报
422 或静默吞掉请求。

**做法**：base_url 一律加 `/v1`。上面第 1 条的示例已经带上了。

（memory `a61a163b`）

______________________________________________________________________

## 3. `--agent-env` 只写一遍

**症状**：launcher 明明设了环境变量，session 子进程读不到；用户可能看到
"启动 X 脚本但堆栈显示用了 Y adapter" 之类的错配。

**根因**：`relax/utils/arguments.py` 的 `--agent-env` 用 `nargs="+"`，不是
`action="append"`。argparse 默认行为是**后值覆盖前值**——多次 `--agent-env
"K=v"` 只有最后一次生效，前面的全被静默丢弃。

**做法**：所有 KV 必须放在**同一个** `--agent-env` flag 后面：

```bash
--agent-env
    "AGENT_DEBUG_LOG_DIR=..."
    "OSWORLD_CONFIG_PATH=..."
    "OSWORLD_CACHE_DIR=..."
    "OSWORLD_MCP_BONUS=..."
    "OPENAI_MODEL=..."
```

（memory `2a732e4e`）

______________________________________________________________________

## 4. SGLang router 必须换成 `consistent_hashing`

**症状**：rollout 时只有某 2 张卡（或 rollout_num_gpus_per_engine 张卡）满
载，其余 engine 0 流量。

**根因**：Relax launcher 不传 `--sglang-router-policy` 时 sglang_router 走默
认 `cache_aware`。OSWorld 的 system prompt 又长又高度一致（HybridAgentLocal
自建的几千 token 模板），所有 session 的前缀 hash 到同一 bucket → 集中
到一个 engine。

**做法**：agentic launcher 必加 `--sglang-router-policy consistent_hashing`。
它按 `session_id` 注入 `X-SMG-Routing-Key`，同 session 多轮锁定同 engine
（保 cache 复用），不同 session 散到不同 engine。**并发 session 数要 >>
engine 数**才能均匀（生日悖论）。

（memory `c6351b41`）

______________________________________________________________________

## 5. Session 冷启动风暴 → `--agentic-prepare-pool-size`

**症状**：rollout 0 时刻 Docker daemon CPU 100%、大量 container 同时
`docker run`，前几十秒每个 session 都 timeout。

**根因**：OSWorld 每个 session 都要新起一个 QEMU 容器 + 挂 qcow2 + 等 VM
guest 就绪，冷启动 30–40s。若 Relax 一次性把整批 session 全放出去，Docker /
KVM 会同时被踩爆。

**做法**：`--agentic-prepare-pool-size N`（当前默认 8）把并发准备数量夹
住。N 具体取值看 host 上 CPU + Docker daemon 承受能力：8–16 是 8 卡机的
经验区间。生产真的 100+ 并发的话建议加大 qcow2 拷贝到 tmpfs（见
`scripts/prepare_vm.sh` 的 `OSWORLD_STAGE_TMPFS=1`）。

______________________________________________________________________

## 6. 长任务和 SIGTERM 僵尸

**症状**：`prepare_gate_blocked_ir_count` 一路涨，rollout 卡在 248/256 之
类的接近完成态死循环。SGLang idle 但 chat completions 都 900s CANCELLED。

**根因**：Relax 超时时对 session 发 `SIGTERM forget=True`。httpx 事件循环
会把 SIGTERM 变成 KeyboardInterrupt 但不会真的打断进行中的 request；OpenAI
SDK 会把当前 chat 补齐 + retry，僵尸 request 能占满整个客户端 timeout
（默认 900s）× retries。

**做法**：本 recipe 已内置 3 道防线：

1. `RelaxHybridAgent` 里 `max_llm_call_retries` / `llm_call_timeout_s` 都
   可以从 yaml 调（默认 3 次 × 600s，比默认 2×900 更保守）。
2. `--agent-timeout 1500`（约 5 倍单 chat timeout）在 launcher 里显式设过。
3. `session_discarded` / `context_length_exceeded` 的错误码在
   `hybrid_agent.py` 里被解析并即时抛出，避免 retry 一个 404。

如果还是踩，逐步调低 `llm_call_timeout_s`（memory `f8489f91` 建议 180s）
而不是把 `--agent-timeout` 调大——治僵尸的关键是让**单次 chat 快速失败**。

（memory `f8489f91`）

______________________________________________________________________

## 7. 用 `tee` 做 per-session 日志会累积僵尸

**症状**：跑到几百个 session 后 host 上 PID 用尽 / `fork: retry: Resource
temporarily unavailable`。

**根因**：`tee` 在进程替换里创建的 tee 子进程，session 主进程退出后 tee
不一定被回收；OSWorld 用的容器 PID 1 通常是 `sleep`，不会收养/reap 孤儿。

**做法**：`run_agent_app.sh` 已强制用 `exec >> "$FILE" 2>&1`（纯
append），千万别改回 `2>&1 | tee -a`。

______________________________________________________________________

## 8. VM cache 位置：`/tmp` 会被清

**症状**：训练跑了一两天，再次启动时 qcow2 消失、每个 session 都在重新
下载。

**根因**：`/tmp` 在多数 host 上有 tmpfs / systemd-tmpfiles 定期清理。

**做法**：`OSWORLD_CACHE_DIR` 指到持久盘（比如 `${DATA_DIR}/osworld_cache`）
或者 `/dev/shm/osworld`（后者需要显式 `OSWORLD_STAGE_TMPFS=1`）。

______________________________________________________________________

## 9. tool_calls 不是结构化字段

**症状**：模型明明输出了 `<tool_call>...</tool_call>`，agent 却觉得没工
具调用、1 turn 就收尾。

**根因**：Relax 的 chat-completions endpoint 只填 `content`，
**永不填 `tool_calls`**——它假设下游 agent 自己在 content 里 regex 解析
tool call XML/JSON。

**做法**：我们复用 hybrid-routing-agent 的 HybridAgentLocal，它内部就是走
的 content-parse 路径，与 Relax 的语义一致。写自己的 agent 时**不要**去读
`choices[0].message.tool_calls`。

（memory `effb0f58`）

______________________________________________________________________

## 排查顺序

看到 session 静默失败 / reward 全 0 时按这个顺序看：

1. `${AGENT_DEBUG_LOG_DIR}/<session_id>.log`——真实 traceback 在这里，别
   看 Ray/Relax 上层堆栈。
2. `rollout_result/train/0.jsonl` 的 `reward` 和 `agent_turns`：全 0 + 全 1
   turn = 客户端根本没进 agent loop（多半是坑 #1/#2/#3）；有 reward 但都
   `agent_turns=1` = 客户端进 loop 但每次都退 = HybridAgentLocal 解析失败
   或 stop token 不对。
3. `nvidia-smi`：只集中在某几张卡 → 坑 #4。
4. `docker ps -a | grep osworld`：一堆 exited container 或 CPU 100% → 坑
   #5，调小 pool size。
