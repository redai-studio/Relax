# Workplace Assistant PITFAIL：接入踩坑记录

本文件记录 Workplace Assistant 接入过程中最容易踩到的坑。完整流程见 [README.md](README.md)。

## 未评分的 deadline 截断不能导出为训练样本

Gateway 达到 deadline 时可能返回 `status=truncated`、`reward=null`、`error.code=deadline_exceeded`。
这表示 verifier 没有产生奖励。薄客户端必须失败退出，让 Relax 丢弃对应采样组；不能把空奖励
正常导出，否则会触发本地 RM fallback，并因未配置 `rm_type` 导致整个 rollout 失败。
已有奖励的截断结果（包括 `reward=0.0`）仍正常导出。

持续的模型回调 500 应查看 Gateway 的 `Retrying NeMo Gym model callback` 日志，其中的 `error`
包含上游错误详情，关闭 verbose 时也会记录。不要只增加 deadline 或用零分掩盖未评分的 trial。

## 1. 工具数量不能只抄上游描述

固定 Gym commit 的上游 README 和公开 dataset card 写明 5 个数据库、26 个工具，但同一 commit
的内置 example 行实际包含 27 个 tool schema。文档描述和任务 artifact 存在一项差异，因此训练
和验收必须以每行 `responses_create_params.tools` 为准，不能硬编码 26 或 27。

## 2. 数据集条数不要硬编码

上游文字说明曾写 1260 条 prompts；当前 Hugging Face 文件可见约 1255 train、545 validation。
prepare 脚本没有固定 dataset revision，远端 artifact 还可能更新。验收下载应检查：

- 文件非空；
- 每行是 JSON object；
- 原始和转换文件非空记录数一致；
- tools、ground truth、category 未丢失。

不要用单纯 `wc -l` 判断内置 example；文件末尾没有换行时会少算一条。

## 3. 不需要额外下载五个数据库

数据库 schema 和初始数据在 Gym resource server 内按 session 构造。用户要准备的是任务 JSONL；
其中包含 prompt、tools、ground truth 和 category。不要寻找不存在的额外 database artifact。

## 4. Verifier 比较最终状态，不比较固定 action trace

模型可以先执行只读搜索，再执行正确写操作。只要最终数据库状态与 ground truth 执行后的状态等价，
reward 应为 1。用字符串或 action list 精确相等来检查会误判。

`verify_workplace_assistant.py` 专门覆盖这个语义。

## 5. Converter 必须保留 tools 和 ground truth

原始字段分布在：

```text
responses_create_params.input
responses_create_params.tools
ground_truth
category
environment_name
```

转换后 messages 进入 `.input`，其余字段进入 `.metadata`。丢掉 tools 会让模型看不到函数定义；丢掉
ground truth 会让 verifier 无法判分。

训练入口使用原始 `workplace_assistant_train.jsonl`，不是 `_relax.jsonl`。

## 6. 同一 session 的工具结果必须回到下一轮 callback

只看到一次 tool call 不代表链路正确。deterministic trial 应依次看到 callback history 中 tool
result 数量 `[0, 1, 2]`。如果第二轮没有第一轮结果，检查：

- opaque rollout prefix 是否保留；
- Gateway callback 是否命中同一个 `rollout_id`；
- simple agent 是否使用当前 Gateway model；
- `observability_enabled=true` 是否生效。

## 7. 终态后要清理 resource session

上游 resource server 原先没有 Gateway 所需的 rollout cleanup contract。本集成 patch 增加
`/cleanup/{rollout_id}` 和映射。没有 patch 的旧镜像可能：

- trial 完成后仍保留数据库 session；
- 同一进程长期训练时内存增长；
- abort 最终变成 `cleanup_unverified`。

镜像必须由当前 Dockerfile 构建。

## 8. Gym 必须使用独立私有 Ray

Gym 和 Relax 共用 Ray 时，Relax `ray-job.sh` 的 `ray serve shutdown` 和残留进程清理会杀掉 Gym
graph。Workplace start 脚本默认启动独立 `:6382`，训练使用另一套 `:6379`。

## 9. 端口残留会导致 `policy_model finished unexpectedly`

重复启动容器或 Ray job 后，旧进程可能仍占用 Gateway 端口：

```text
[Errno 98] address already in use
RuntimeError: Process `policy_model` finished unexpectedly!
```

先定位并停止旧 Gym 容器/job，不要盲目重试启动第二套 graph。`/readyz` 返回的 commit 和 trial
计数也能帮助识别访问的是不是旧服务。

不要把服务端口放在主机的临时端口区间内。Linux 常见的
`ip_local_port_range` 是 `32768 60999`；NeMo 子进程连接私有 Ray 时可能瞬时分配该区间内的
源端口，导致预检查时端口空闲、随后 Uvicorn 却报 `address already in use`。本 recipe 默认使用
区间外的 `29000`–`29003`。可用下面的命令查看当前主机范围：

```bash
cat /proc/sys/net/ipv4/ip_local_port_range
```

## 10. callback allowlist 和代理

`NEMO_GYM_CALLBACK_ALLOWED_NETWORKS` 要覆盖 Relax callback URL 中的 IP，填写严格 CIDR。Gym 和 Relax IP 必须加入
`NO_PROXY`。否则请求可能被代理转发，表现为超时、403/404 或 callback IP 不在允许网段内。

## 11. `max_steps=6` 是 agent 循环上限

固定 config 的 simple agent 最多运行 6 step。复杂任务在预算内未完成会 truncated 或 reward=0。
不要仅靠提高 Relax response token 长度来突破 agent step 上限；若要修改上游 graph 配置，应单独
评估并发、deadline 和轨迹长度。

## 12. 1 sample smoke 没有有效 GRPO advantage

`NEMO_GYM_N_SAMPLES_PER_PROMPT=1` 只适合验证链路。真实 GRPO 需要同 prompt 多 sample，例如 4，
并检查 reward 有组内差异、Actor metrics 和 checkpoint。

## 13. Ray Job `SUCCEEDED` 不是最终训练证据

必须同时检查 rollout JSONL、Actor OOM/traceback、step metrics 和 Gateway cleanup。driver 正常
退出不保证每个远端 Actor 都完成了 optimizer step。

## 14. 不要用 Ray working-dir 上传整个仓库

训练节点应能看到同一个共享 Relax checkout。recipe 会把 `run_training.sh`、agent command 和
agent cwd 都展开为该 checkout 的绝对路径，因此不需要设置 `WORKING_DIR`。设置 `WORKING_DIR=./`
会让 Ray 打包整个仓库；上传较慢时，worker 可能报 runtime-env package 已从 GCS 过期或下载失败。

## 15. submission ID 和 Dashboard 地址

固定复用同一个 submission ID 会被 Ray 拒绝，即使旧任务已经结束。使用脚本生成的带时间、进程号和
随机后缀的 ID，并记录提交输出。训练脚本从 `RAY_ADDRESS=<head>:6379` 推导
`http://<head>:8265`；Dashboard 不在默认端口时设置 `RAY_DASHBOARD_ADDRESS`。

## 16. 上下文预算必须相容

默认 8K context、最多 8191 prompt 和单轮最多 2K response。Agentic 多轮生成会根据每一轮已经
占用的上下文，把实际生成上限截断到剩余空间；`response` 不是额外预留在 prompt 之外的固定区间。
若覆盖这些值，必须满足：

```text
rollout_max_prompt_len < rollout_max_context_len
rollout_max_response_len <= rollout_max_context_len
max_tokens_per_gpu >= rollout_max_context_len
```

脚本会在提交 Ray Job 前检查这些关系。接近 context 上限的 prompt 虽然可以进入 rollout，但只会
剩下很少生成空间；真实数据应结合 token 统计选择更小的 prompt 上限。
