# reasoning-gym-cc

这个 recipe 使用 NeMo Gym 的 `claude_code_agent` 驱动 Claude Code CLI，通过 Anthropic Messages
向 Relax 请求 Qwen3-4B，并使用 reasoning-gym scorer 计算 reward。Claude Code 在 Gym 容器内自行执行
Bash 工具。

接入踩坑见 [PITFAIL.md](PITFAIL.md)。

以下步骤覆盖从零运行。

```text
Relax managed app/client.py
  -> NeMo Gym trial
  -> claude_code_agent
  -> POST /ng-rollout/<rollout-id>/v1/messages
  -> Gateway adds the Relax Bearer session token
  -> Relax /v1/messages
  -> canonical messages/tools/chat_template_kwargs
  -> SessionForest
```

Gateway 按 Messages 协议转发请求和 Relax Buffered SSE。每次请求携带完整 `system + messages` 历史。
Claude Code 以 bare 模式启动并显式加载 recipe settings。当前因 bwrap 与运行内核不兼容，
Bash sandbox 已关闭，Bash 直接在 Gym 容器内执行。
每个 trial 使用 `~/.claude_code_agent/<uuid>/workspace/` 作为独立工作目录，
`TMPDIR`、`TMP`、`TEMP` 指向同一 trial 的 `tmp/`；退出清理时一并删除这些目录，
减少并发同名文件覆盖和跨 trial 文件残留。绝对路径、共享 HOME 和显式 `/tmp/...` 仍可跨 trial 访问，
这不是严格的文件系统沙箱。

## 构建 Gym 镜像

Callback 白名单只配置 `NEMO_GYM_CALLBACK_ALLOWED_NETWORKS`，默认为 `10.0.0.0/8`，可覆盖为逗号分隔的 CIDR；
填写实际覆盖 Relax callback IP 的网段，单个 IPv4/IPv6 地址使用 `/32` 或 `/128`。
远程启动脚本也可重复传入 `--callback-network`。CIDR 按 URL 中的 IP 匹配，不解析域名，且不接受 `/0`。

```bash
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"
docker build \
  -f examples/nemo_gym_agentic/service/Dockerfile \
  -t relax-nemo-gym:reasoning-gym-cc-dev \
  .
```

镜像固定 NeMo Gym commit `a85670eb167ba9b48cc53a36a070eed815e6c40d` 和 Claude Code
`2.1.237`，安装 `bubblewrap` 与 `socat`，并为 Claude Code 子进程提供 abort、force-cleanup 和 cleanup
probe。

工作目录隔离由镜像构建时应用的 `claude_code_agent_cleanup.patch` 提供。
更新此补丁后，需要重新构建镜像并重新创建 Gym 容器；仅更新挂载的 Relax checkout 或重启旧容器不会生效。

## 准备数据

```bash
DATA_DIR=/共享目录/nemo-gym/reasoning-gym-cc \
NEMO_GYM_IMAGE=relax-nemo-gym:reasoning-gym-cc-dev \
  bash examples/nemo_gym_agentic/recipes/reasoning-gym-cc/prepare_reasoning_gym_cc.sh
```

训练数据输出到 `${DATA_DIR}/reasoning_gym_cc_train.jsonl`。设置
`REASONING_GYM_CC_SPLIT=example` 可准备 pinned Gym 自带的 5 条 smoke 数据。

## 启动 Gym

```bash
bash examples/nemo_gym_agentic/recipes/reasoning-gym-cc/start_reasoning_gym_cc_gym_remote.sh \
  --gym-host "${GYM_HOST}" \
  --callback-network "${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS}" \
  --data-jsonl "${DATA_DIR}/reasoning_gym_cc_train.jsonl" \
  --port-base 29200 \
  --image relax-nemo-gym:reasoning-gym-cc-dev
```

默认端口为 Gateway `29200`、Claude Code agent `29201`、reasoning-gym resource `29202` 和
Gym head `29203`。

启动后验证完整 Claude Code/Bash 链路：

```bash
docker exec nemo-gym-reasoning-gym-cc \
  /opt/nemo-gym/resources_servers/reasoning_gym/.venv/bin/python \
  /opt/relax-integration/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/verify_reasoning_gym_cc_trial.py \
  --gateway-url http://127.0.0.1:29200 \
  --task-jsonl "${DATA_DIR}/reasoning_gym_cc_train.jsonl"
```

## 启动训练

```bash
RAY_ADDRESS="${RELAX_HOST}:6379" \
MODEL_DIR=/共享模型目录 \
NEMO_GYM_GATEWAY_PORT=29200 \
NEMO_GYM_SOURCE_DATA=${DATA_DIR}/reasoning_gym_cc_train.jsonl \
GYM_HOST="${GYM_HOST}" \
  bash examples/nemo_gym_agentic/recipes/reasoning-gym-cc/run-qwen3-4B-8xgpu-nemo-gym-reasoning-gym-cc.sh
```

训练脚本关闭 checkpoint 保存，并把 `--dump-details` 写到 `${SAVE_DIR}/dump`。缩小实验时直接修改
脚本中的 `--num-rollout`、数据条数和 `--n-samples-per-prompt`。
标准配置同时驻留 8 个 prompt Group，每组 8 个 Session，对应 64 个 Claude Code trial。

当前 recipe 通过 Claude Code settings 关闭 context compaction。正常线性历史使用 implicit export；若
客户端仍改写旧历史，Relax 会记录真实 SessionForest 分叉，无法唯一 finalization 的 Session 会被丢弃并
补采。
