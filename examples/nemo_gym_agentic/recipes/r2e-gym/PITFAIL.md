# R2E-Gym PITFAIL：接入踩坑记录

R2E-Gym 同时跨越数据转换、OpenHands、Ray、Docker、Apptainer、网络 callback 和长序列训练。本文件
记录本次接入中真实遇到的问题。完整流程见 [README.md](README.md)。

## 1. 数据准备不需要 Ray

prepare 只做三件事：

1. 流式读取 Hugging Face row；
2. 转成 NeMo Gym/Relax 所需 JSONL；
3. 将 row 的 OCI image 转成 SIF。

它可以在任意能写共享存储的机器运行。把 `RAY_ADDRESS`、远程 Ray 或训练平台引入数据准备只会增加
故障面。

## 2. 一题一个镜像，磁盘会快速增长

R2E-Gym 上游说明每个 environment image 常见约 300—500 MB；本次单题 SIF 为 433 MB。`LIMIT=N`
通常意味着 N 个独立 SIF，还会有 Apptainer cache。先用 1 题通过 golden，再扩数据。

## 3. `instance_id`、SIF 名和 formatter 必须完全一致

prepare 从：

```text
<owner>/<repo>_final:<commit>
```

生成：

```text
instance_id = <owner>__<repo>-<commit>
sif_name    = <repo>_final_<commit>.sif
```

启动脚本使用：

```text
${R2E_GYM_SIF_DIR}/${R2E_GYM_SIF_PREFIX}{instance_id}.sif
```

固定 commit 的 SWE agent 会在识别到 `R2E-Gym` dataset name 后完成该映射。若报
`No container file found`，先同时检查 task JSONL、manifest、实际 SIF 文件名和 resolved config，
不要只确认目录存在。

共享目录使用额外前缀时，不要为数千个 SIF 建软链。分别设置 `--sif-dir` 和
`--sif-prefix r2egym_`；remote launcher 会只读挂载该目录，local launcher 会直接读取它。prefix
只能是文件名前缀，不能包含 `/`。

Docker-in-Docker 下要区分“当前开发容器可见”和“外层 Docker daemon 可见”。如果共享目录是只在
开发容器内挂载的 FUSE 文件系统，`ls` 能看到它并不代表 remote launcher 能 bind mount。remote
launcher 会在替换已有容器前用临时容器检查 repo、数据和 SIF 路径；检查失败时先在 Docker 宿主
挂载这些路径，不能通过批量软链规避 mount namespace 问题。

## 4. 正确 base commit 不是 fix commit

数据行的 `commit_hash` 是修复后的 commit。agent 应从修复前状态开始，因此正确 base 优先取：

```text
parsed_commit_content.old_commit_hash
```

缺失时才退化为：

```text
${commit_hash}^
```

直接把 `commit_hash` 当 base 会让 golden patch 或模型修改落在错误仓库状态，通常 reward=0。

## 5. `instance_dict`、`repo`、`version` 不能在 converter 中丢失

SWE agent 和 evaluator 依赖：

- `responses_create_params.metadata.instance_dict`；
- `repo`；
- `version`；
- `base_commit`；
- `problem_statement`。

早期 converter 只保留 input/tools，导致：

```text
KeyError: instance_dict
```

当前 converter 会把未知 `responses_create_params` 字段放到
`metadata.responses_create_params`，adapter 再还原。修改 schema 时必须保留通用 passthrough。

## 6. 本地 driver 直接 attach 远程 Ray 可能失败

曾出现：

```text
RuntimeError: No node info found matching attributes
ConnectionError: Failed to connect to Ray cluster
```

`ray status --address=<head>:6379` 成功不代表本地 Python driver 能作为远程 node driver attach。
网络、多网卡、node IP 和 Ray 版本都会影响。当前推荐让 Gym 在本地容器的私有 Ray 上运行，只通过
HTTP 与远程 Relax 连接。

如果确实把 Gym 跑在远程 Ray，应使用 Ray Jobs dashboard 提交，且所有 worker 环境一致。
共置模式还要求 NeMo Gym venv 与 Relax Ray 集群的 Ray 版本完全一致。local launcher 会使用
`uv` 将 Gym、Gateway 和 SWE venv 自动对齐到 Relax 系统 Python 的 Ray 版本；首次对齐需要可用的
包索引/代理。安装或复验失败时应修复镜像或网络，不能用 `RAY_IGNORE_VERSION_MISMATCH=1` 绕过。

local launcher 必须运行在 Ray head 容器中，并复用已有 `RAY_ADDRESS`。不要在共置容器中再启动
`:6381` 私有 Ray。共置 Gym 启动后不能再执行 `ray-job.sh`：其残留进程清理会通过
`kill_for_ray.sh` 把 Gym 的 Python 进程一并杀掉。必须先确保训练集群干净，再启动共置 Gym；Gym
ready 后直接执行训练 recipe。当前尚未提供把“清理、启动 Gym、提交训练、按 PID 回收 Gym”串起来的
一体化 supervisor。

## 7. Ray 端口范围不能重叠

曾把 dashboard `18265` 放进 worker port range `10002—19999`，Ray 直接报：

```text
Ray component worker_ports is trying to use a port number 18265
that is used by other components
```

本地 wrapper 使用私有 GCS 6381、dashboard 28265，并关闭 dashboard serving。不要随意把 Ray
组件端口放进 worker range。

## 8. 旧 `policy_model` 占用 28100

重复提交 Gym job 会出现：

```text
[Errno 98] address already in use
RuntimeError: Process `policy_model` finished unexpectedly!
```

同时还可能有旧 setup lock。启动前确认只有一套 Gym：

```bash
docker ps --filter name=nemo-gym-r2e
ss -ltnp | rg ':28100|:28101|:6381'
```

`start_r2e_gym_remote.sh` 会删除同名旧容器，但不会杀掉其他名字的旧容器或远程 Ray job。
`start_r2e_gym_local.sh` 会按 `28100/28101/28103` 和对应 NeMo Gym 服务 cwd 精确识别并回收上次
失败遗留的本地服务；若端口属于其他 cwd，则只报错，不会执行宽泛 `pkill`。

## 9. 首次启动需要代理，并且 setup 要持久化

首次 SWE agent 启动会下载：

- uv/miniforge；
- pinned NVIDIA R2E evaluator fork；
- pinned OpenHands fork；
- Python/Conda 依赖。

没有代理时可能卡在 `curl` 或 `git clone`。代理必须传进 Gym 容器，同时把 Gym/Relax 内网 IP 加入
`NO_PROXY`。本地 wrapper 用 named volumes 持久化 R2E 和 OpenHands setup，后续启动应看到
`already set up`。

失败的 setup 可能留下 lockdir。只有确认没有活跃 setup 进程后才能处理 stale lock，不能运行中
直接删除。

## 10. `platform.release()` 可能不是合法 PEP 440 版本

某些平台 kernel release 带内部后缀，OpenHands build 将其作为版本解析后失败。当前
`openhands_platform_release.patch` 和 `sitecustomize.py` 会清理该值。旧镜像没有这个 patch 时，
setup 可能在看似无关的 Python package build 阶段失败。

## 11. rollout prefix 必须跨进程、Ray 和 Apptainer 保留

OpenHands 最初直接请求：

```text
/v1/chat/completions
```

Gateway 无法知道属于哪个 Relax session，会返回 404/410。正确路径是：

```text
/ng-rollout/<opaque-id>/v1/chat/completions
```

opaque prefix 必须从 Gateway `/run` payload 进入 SWE agent，再进入 OpenHands config/container。
如果首轮能回调、后续轮断链，检查 rollout-prefix patch 是否真的在运行镜像中。

## 12. callback networks 必须覆盖实际 IP

只配置 `NEMO_GYM_CALLBACK_ALLOWED_NETWORKS` 或 `--callback-network`，使用逗号分隔的严格 CIDR。
网段必须覆盖 Relax callback URL 中的 IP；单地址使用 `/32` 或 `/128`，禁止 `/0`。
Golden 模式虽然不调用模型，协议仍校验 callback URL，因此也必须配置覆盖 verifier callback IP 的 CIDR。

## 13. `cleanup_unverified` 不是模型 reward=0

`cleanup_unverified` 表示任务取消或失败后，Gateway 无法证明远端 agent/sandbox 已经清理。它是
生命周期错误，不是模型没解出题。对于 protected trial，不要把本地 HTTP coroutine 取消等同于
Apptainer 已停止。

## 14. Golden status completed 但 reward=0 仍是失败

早期 smoke 只检查 HTTP/terminal status，得到：

```json
{"status":"completed","reward":0.0,"tool_calls":0}
```

这不能证明 R2E 正确。Golden 的唯一通过标准是 reference patch 在该 SIF/evaluator 中 reward=1。
health、200 OK、server ready 都不能替代它。

## 15. 真实模型 reward=0 不一定是集成失败

本次 Qwen3-4B rollout 实际完成 12 个 turn 和 12 个 `<tool_call>`，然后 evaluator 返回
`resolved=false`、reward=0。模型反复使用错误 `str_replace` 且没有生成 patch。此时：

- callback 正常；
- OpenHands 正常；
- Apptainer evaluator 正常；
- 模型能力失败。

区分方法是看 status、turn/tool call、agent/evaluator 日志和 patch/test 结果，而不是只看 reward。

## 16. Relax JSONL 中 tool call 是序列化文本

当前 `rollout_result` 的 `.response` 是字符串。下面的 jq 会错误得到 0：

```bash
jq '[.. | objects | select(.type? == "function_call")] | length'
```

应使用：

```bash
jq -r '.response' result.jsonl | rg -o '<tool_call>' | wc -l
```

并检查真实 shell/file command。

## 17. 2-GPU 长序列训练的 OOM 边界

真实 20K token rollout 已进入 Actor `train_one_step`，先后暴露：

1. 未分块 full log-prob/CE 需要额外约 3.68 GiB；
2. 加 `--log-probs-chunk-size 1024` 后，`entropy_coef=0` 仍计算 full entropy，又申请约 290 MiB。

当前分支已经：

- 默认设置 log-prob chunk 1024；
- 在 `entropy_coef == 0` 时跳过 entropy 计算；
- 添加对应单元测试。

但用户停止任务后，最终 2-GPU optimizer step 尚未重新验证。因此：

- 不能写“2-GPU E2E 训练已跑通”；
- 优先用 8-GPU reference；
- 若继续 2-GPU，必须重新观察 Actor step 和显存，而不是复用旧 Ray Job 状态。

## 18. Ray Job `SUCCEEDED` 可能掩盖 Actor OOM

外层 controller/driver 可能记录 `Main func successfully`，而远端 Actor 已经 OOM。最终验收必须同时
满足：

- rollout JSONL 正常；
- Actor 日志无 OOM/traceback；
- 出现 optimizer/metrics step；
- checkpoint/参数更新符合预期；
- Gym active trial 和 Apptainer 进程已归零。

## 19. 一条 sample 不是有效 GRPO 训练

R2E 默认 smoke 是 1 prompt × 1 sample。即使 optimizer step 运行，GRPO 组内 advantage 也没有
有效差异。真实训练至少需要多 sample，但 24K response × 4 会显著扩大资源开销。先逐层验证，再从
concurrency 1 开始扩。

## 20. 平台远程 Gym 只适合可控 runtime

不可修改的平台 worker 曾同时暴露代理缺失、镜像不一致、端口残留、setup 不持久和驱逐问题。远程
`submit_r2e_gym.sh` 不是错误，但它要求你能控制所有 environment Ray worker。否则本地私有 Gym +
远程 Relax 的 HTTP 拓扑更简单、更可复现。

## 21. 重跑前必须同时清理 Ray Job、Ray Serve、SGLang 和残留 sandbox

只执行 `pkill sglang`、只停止训练 driver，或者只重启 NeMo Gym 都不够。旧 Ray Job、Ray Serve
deployment 和 managed agent 可能继续存活或重新部署服务，产生下面这种跨代状态：

```text
Connecting to existing Serve app
Failed to register engine to router: <RAY_HEAD_IP>:3778 Connection refused
Router launched locally at <RAY_HEAD_IP>:4484
RuntimeDomain activation did not take ownership:
leased_requests=8, activated_sessions=0, started_sessions=0
```

这时旧 SGLang Engine 仍携带上一轮 Router 端口，新任务却已经创建了另一端口。新 Router 的
`/workers` 为空，agentic service 随后回退访问 `/list_workers` 并出现 404。该 404 是二次表现，
不是根因。GPU 通常保持 0 利用率，NeMo Gym 侧只剩 trial renew 或失败清理。

Ray Dashboard 还可能把 driver 已消失的 submission 显示为 `RUNNING`。不能仅凭 Jobs API 判断进程
仍然存活，应同时核对 driver PID、Ray Actor 和系统进程。

在**独占、可随时重建的单节点 TorchJob** 上，可以用下面的一条命令执行全量清理。它会停止该
Ray 集群的全部 submission 和 Serve 服务，并杀掉本节点的 R2E-Gym sandbox；不要在共享 Ray
集群或承载其他 Apptainer 任务的宿主机执行：

```bash
(ray job list 2>/dev/null | grep RUNNING | grep -oP "submission_id='\K[^']+" | sort -u | xargs -r -n1 ray job stop 2>/dev/null || true); ray serve shutdown -y 2>/dev/null || true; ray stop --force 2>/dev/null || true; pkill -9 -f '[s]glang' 2>/dev/null || true; pkill -9 -f '[p]ython(3)? -m app.client' 2>/dev/null || true; pkill -9 -f '[A]pptainer runtime parent: aiohttp_final_' 2>/dev/null || true; pkill -9 -f '/container_scripts/[e]val_script.sh' 2>/dev/null || true; pkill -9 -x appinit 2>/dev/null || true
```

正确恢复顺序是：

1. 先清理 Relax/Ray 和本节点残留 sandbox；
2. 再重启 NeMo Gym，清空其进程内 trial registry；
3. 检查 NeMo Gym 为零 trial；
4. 最后重新创建 Ray 集群并提交唯一一个训练任务。

## 22. 新启动的 NeMo Gym 必须先检查 `service_epoch` 和 trial 计数

Gateway 的 trial registry 是进程内内存，`service_epoch` 也会在 Gateway 进程启动时重新生成。
因此，新 epoch 不会从磁盘恢复旧 trial。启动后先执行：

```bash
curl --noproxy "*" -sS "http://${GYM_HOST}:28100/readyz" |
  jq '{service_epoch, trials, active_trials, terminal_trials, ready}'
```

还没有启动训练时，正确的干净基线是：

```json
{
  "trials": 0,
  "active_trials": 0,
  "terminal_trials": 0,
  "ready": true
}
```

本次实际遇到的异常表现是：

```json
{
  "service_epoch": "03612432c6c48cb0992e5b534148e630",
  "trials": 16,
  "active_trials": 0,
  "terminal_trials": 16,
  "ready": true
}
```

这不表示 NeMo Gym 恢复了 16 条历史记录，而表示残留 Relax agent 在本次 Gateway 启动后创建了
16 个 trial；清理任务后，它们全部进入 terminal 状态。必须先清理请求方，再重启 Gateway，否则
残留客户端会立刻重新填充刚清空的 registry。

训练侧清理完成后，可重启并等待 Gateway ready：

```bash
docker restart nemo-gym-r2e-local >/dev/null &&
until curl --noproxy "*" -fsS "http://${GYM_HOST}:28100/readyz" >/dev/null; do
  sleep 2
done
curl --noproxy "*" -sS "http://${GYM_HOST}:28100/readyz" |
  jq '{service_epoch, trials, active_trials, terminal_trials, ready}'
```

如果 trial 在没有新训练任务时继续增长，说明仍有 agent client 存活。此时查看：

```bash
docker logs nemo-gym-r2e-local 2>&1 | grep 'POST /v1/trials'
```

定位请求时间和来源 IP。正常训练开始后 terminal 计数会随已结束 trial 累积，所以“必须为 0”只
用于 Gateway 刚重启、训练尚未提交时的基线检查。

## 23. `base_commit=<hash>^` 在 cleanup 后失效会让所有模型 reward 变成 0

R2E-Gym 数据转换会把任务基线写成 `<commit_hash>^`。OpenHands 初始化 sandbox 时先将这个表达式
解析为真实父提交，随后为防止 agent 读取未来提交，会删除后继 refs、过期 reflog，并执行
`git prune`/`git gc --prune=now`。cleanup 结束后，表达式中的 child commit 已经不存在。

旧实现完成 agent 后仍执行：

```bash
git diff --cached <commit_hash>^
```

因此即使模型已经正确修改代码并调用 `finish`，该命令也会返回 `fatal: bad revision`。OpenHands
连续重试后报：

```text
Failed to get git diff (None)
```

最终 artifact 表现为：

```json
{
  "patch_exists": false,
  "model_patch": null,
  "resolved": false,
  "reward": 0
}
```

这时 evaluator 根本没有运行，不能把 reward=0 归因于模型能力。Golden 验证直接使用 reference
patch，会绕过 agent completion 的 patch 收集阶段，因此 golden reward=1 也不能发现这个问题。

当前修复在 cleanup/prune 前创建 `swebench_baseline`，并让完成阶段优先相对该稳定 tag 生成 diff。
nuclear cleanup fallback 也优先从该 tag 恢复已解析的基线，并在删除其他 tag 后立即重建它。启动脚本
会把修复注入已有 OpenHands setup cache；新构建的 NeMo Gym 镜像也会内置该 patch。

本次用同一题执行确定性“正确编辑 + `finish`”重放，修复后的结果是：

```json
{
  "patch_exists": true,
  "resolved": true,
  "reward": 1.0,
  "status": "completed"
}
```

实际 artifact 路径由 trial result 的 `artifact_ref` 返回，不要在 recipe 中硬编码共享存储路径。

另外，OpenHands replay retry 原来会复用已被 EventStream 写入 ID 的可变 Event，第二次 attempt 报：

```text
Event already has an ID:1
```

当前修复在每次 attempt 前分别对 replay event 列表和 initial action 执行 `deepcopy`。不要通过清空
单个 `_id` 或吞掉 retry 异常绕过；ReplayManager 还可能原地修改其他 Event 字段。
