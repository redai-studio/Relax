# Relax NeMo Gym Gateway

本目录实现 Relax managed-agent runtime 与长生命周期 NeMo Gym graph 之间的服务边界。用户启动流程
请从[顶层文档](../README.md)和各 recipe README 开始；本文只描述 Gateway 设计和扩展约束。

## 组件

| 文件                      | 职责                                                                          |
| ------------------------- | ----------------------------------------------------------------------------- |
| `app.py`                  | FastAPI 路由、health/readiness 和错误映射                                     |
| `registry.py`             | admission、幂等 create、queue、lease、deadline、终态竞争、callback capability |
| `callback_provider.py`    | NeMo Gym Chat、Responses、Messages 原协议转发到 request-scoped Relax endpoint |
| `run_adapter.py`          | trial task 转成 NeMo Gym agent `/run` payload，处理完成和 cleanup             |
| `config.py`               | environment registry、callback allowlist 和 graph 校验                        |
| `nemo_gym_gateway_model/` | 将 Gateway 托管在 NeMo Gym graph 内的 model-server plugin                     |
| `Dockerfile`              | 固定 Gym commit、应用 patch、预建各 server venv                               |
| `patches/`                | 固定 Gym/OpenHands/R2E 版本的兼容和生命周期修正                               |

## HTTP 契约

```text
POST /v1/trials
GET  /v1/trials/{request_id}
POST /v1/trials/{request_id}/renew
POST /v1/trials/{request_id}/abort

POST /ng-rollout/{rollout_id}/v1/chat/completions
POST /ng-rollout/{rollout_id}/v1/responses
POST /ng-rollout/{rollout_id}/v1/messages

GET  /healthz
GET  /readyz
```

协议版本为：

```text
relax-nemo-gym/v1
```

### Trial create

create payload 包含：

- 稳定、attempt-scoped 的 `request_id`；
- session/group/mode；
- environment name、config 和 task；
- request-scoped model endpoint、Bearer token 和模型名；
- generation overrides；
- interrupt policy；
- deadline 和 lease。

create 对同一个 request payload 幂等；相同 ID 配不同 payload 会拒绝。

### Session capability

Registry 为每个 trial 生成随机 opaque `rollout_id`，只在内存中保存：

```text
rollout_id -> callback URL + token + model + sampling params
```

NeMo Gym agent 必须通过带 prefix 的 callback route 请求模型。未带 prefix 的请求不能安全关联到
Relax session，因而不会被接受。trial 终态后 capability 立即移除。

### Heartbeat、deadline 和 abort

thin client 在运行期每 `lease_s / 3` 续租。lease expiry、deadline、显式 abort 和正常 completion
在 trial lock 下竞争，首个被接受的终态获胜。

abort 返回不等于远端 sandbox 已清理。只有 environment cleanup probe 返回：

```json
{"clean": true}
```

才确认清理；否则终态保守报告 `cleanup_unverified`。

## Environment 注册

Gateway 不接受客户端传任意 agent URL。服务端通过
`NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON` 注册允许的 `(environment, config)`：

```json
{
  "workplace-assistant-v1": {
    "environment": "workplace_assistant",
    "agent_name": "workplace_assistant_simple_agent",
    "agent_url": "http://GYM_HOST:28101",
    "readiness_urls": [
      "http://GYM_HOST:28101",
      "http://GYM_HOST:28102"
    ],
    "abort_url": "http://GYM_HOST:28102/cleanup/{rollout_id}",
    "force_cleanup_url": "http://GYM_HOST:28102/cleanup/{rollout_id}",
    "cleanup_probe_url": "http://GYM_HOST:28102/cleanup/{rollout_id}",
    "interrupt_policy": "protected",
    "max_concurrency": 8,
    "queue_capacity": 32,
    "max_deadline_s": 1800
  }
}
```

启动时还必须：

- `observability_enabled=true`；
- 显式 `ray_head_node_address`；
- `skip_venv_if_present=true`；
- agent 的 `model_server.name` 指向当前 Gateway model；
- 注册的 `agent_url` 与 resolved graph host/port 一致；
- readiness URLs 覆盖 agent 和必要 resource server。

## Callback 安全边界

`NEMO_GYM_CALLBACK_ALLOWED_NETWORKS` 是逗号分隔的 CIDR 白名单，启动脚本默认使用 `10.0.0.0/8`，必须覆盖 callback URL 中的 IP：

```bash
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"
```

单个 IPv4/IPv6 地址使用 `/32` 或 `/128`；网段必须是严格 CIDR，不允许 `/0`。
CIDR 仅匹配 URL 中的 IP，不对域名做 DNS 解析。不支持 wildcard、userinfo 或非 HTTP(S) URL。token 不写日志，record 终态后清除。三种 callback
都支持 JSON 与 Buffered SSE；Messages 在等待终态时发送 ping。

## 镜像和 Python 环境

Dockerfile 基于 Relax 训练镜像构建，但不能全局把 `/opt/nemo-gym/.venv/bin` 放到 PATH 前面。
原因是：

- NeMo Gym 固定的 Ray/Python 依赖用于 Gym graph；
- Relax 系统 Python 包含训练所需 `loguru`、Megatron、SGLang 和 `ray serve`；
- 两套依赖由 launcher 的绝对路径显式选择。

验证：

```bash
docker run --rm "${NEMO_GYM_IMAGE}" bash -lc '
  /opt/nemo-gym/.venv/bin/gym --help >/dev/null
  /usr/bin/python3 -c "import loguru, ray"
  ray serve --help >/dev/null
'
```

## 增加新 environment

新增 recipe 至少完成：

01. 研究上游 task schema、agent、resource、verifier 和 sandbox 生命周期。
02. 在 Dockerfile 构建对应 server venv；运行时不下载普通 server 依赖。
03. 分配固定可路由端口。
04. 注册精确 environment/config 和 readiness URL。
05. 确认 agent 的完整 Chat、Responses 或 Messages payload 能由 Relax 对应 endpoint 接收。
06. 提供 deterministic verifier contract：正确 reward 与错误 reward 都要测。
07. 工具环境提供完整 trial test，证明多轮 callback 和 tool result 回灌。
08. 有状态或 sandbox 环境提供 cleanup/abort/probe，或明确标记 protected 和限制。
09. 新建 recipe 中文 `README.md`。
10. 最后再做真实模型 rollout 和 optimizer step，分别记录证据。

不要只用 health 200 或“agent `/run` 返回”作为 correctness 验收。

## 当前限制

- Registry 是单进程内存状态，Uvicorn 必须 `--workers 1`。
- 进程重启不会恢复运行中的 trial 或 durable tombstone。
- streaming callback 使用 Buffered SSE。
- 通用 upstream `/run` 没有统一 cancellation handle，cleanup 能力按 environment 定制。
- Gateway 支持按 trial 的 environment/config 路由；multienv recipe 从每行 metadata 选择该二元组。
- 固定 commit 上的 patch 不能假设适用于 NeMo Gym `main`；升级时必须重新审计和跑全套 contract test。

当前真实验证边界以[顶层文档的“当前验证边界”](../README.md#当前验证边界)为准。本文不再保留早期
`example_multi_step` 的“optimizer step 已通过”表述。
