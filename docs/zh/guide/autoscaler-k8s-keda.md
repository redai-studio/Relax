# K8s HPA + KEDA 弹性扩缩容集成方案

## 概述

本文档描述如何将 Relax 弹性 Rollout 扩缩容与 Kubernetes HPA（通过 KEDA）结合，实现 **K8s 自动扩缩 GPU 资源 + Relax 自动注册/注销引擎** 的完整链路。

### 核心思路

| 层次 | 职责 | 实现方式 |
|---|---|---|
| **K8s 资源层** | 根据 SGLang 指标自动扩缩 Pod（GPU 资源） | KEDA ScaledObject + Prometheus |
| **应用注册层** | Pod ready 后调 `scale_out`(external)，缩容前调 `scale_in` drain | Pod Lifecycle Hooks |

Relax 代码 **零修改**：所有能力（external scale_out/scale_in API、幂等性、权重同步、drain）已在 `relax/components/rollout.py` 和 `relax/distributed/ray/rollout.py` 中实现。

### 前置条件

- 训练使用 **全异步模式**（`--fully-async`）
- Rollout 引擎使用 **SGLang** 作为推理后端
- K8s 集群已安装 [KEDA](https://keda.sh/) 和 [Prometheus](https://prometheus.io/)
- SGLang Pod 与 Ray 集群网络互通（NCCL 权重同步需要 GPU 直连或 RoCE/InfiniBand）
- 启动训练时 **不要** 传 `--autoscaler-config`（禁用内置 Autoscaler，避免冲突）

______________________________________________________________________

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                           │
│                                                                 │
│  ┌────────────┐      Prometheus Query       ┌───────────────┐   │
│  │   KEDA     │◄────────────────────────────│  Prometheus   │   │
│  │  Operator  │                             └───────┬───────┘   │
│  └─────┬──────┘                                     │ scrape    │
│        │ scale replicas                             │ /metrics  │
│        ▼                                            │           │
│  ┌──────────────────────────────────────────────────┤           │
│  │        Deployment: sglang-external-engines       │           │
│  │                                                  │           │
│  │  ┌──────────────────────────────────────────┐    │           │
│  │  │  Pod (SGLang Engine)                     │    │           │
│  │  │                                          │    │           │
│  │  │  postStart hook ──► scale_out(external)  │────┘           │
│  │  │                     engine_urls=[my IP]  │                │
│  │  │                                          │                │
│  │  │  preStop hook ───► scale_in              │                │
│  │  │                    engine_urls=[my IP]   │                │
│  │  │                    wait drain complete   │                │
│  │  └──────────────────────────────────────────┘                │
│  └──────────────────────────────────────────────────────────────┘
│        │                                                        │
│        │  HTTP API                                              │
│        ▼                                                        │
│  ┌──────────────────────────────────────┐                       │
│  │  Relax Training Cluster (Ray)        │                       │
│  │  ┌─────────────────────────────┐     │                       │
│  │  │ Rollout Service (FastAPI)   │     │                       │
│  │  │ POST /rollout/scale_out     │     │                       │
│  │  │ POST /rollout/scale_in      │     │                       │
│  │  │ GET  /rollout/engines       │     │                       │
│  │  └─────────────────────────────┘     │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

**数据流：**

1. Prometheus 抓取所有 SGLang Pod 的 `/metrics`
2. KEDA 通过 PromQL 查询聚合指标，驱动 HPA 调整 Deployment replicas
3. 新 Pod 启动后，`postStart` hook 等待 SGLang 就绪，调用 Relax `scale_out`(external 模式) 注册引擎
4. 缩容时，`preStop` hook 调用 Relax `scale_in` drain 流量，等待完成后允许 Pod 终止

______________________________________________________________________

## 与 Relax 内置 Autoscaler 的关系

内置 `AutoscalerService`（`relax/utils/autoscaler/`）和 K8s KEDA 方案不应同时启用。两者职责对比：

| 职责 | 内置 Autoscaler | K8s KEDA 方案 |
|---|---|---|
| 指标采集 | `MetricsCollector` 轮询 `/metrics` | Prometheus 抓取 |
| 扩缩决策 | `ScalingDecisionEngine` | KEDA ScaledObject |
| 资源分配 | `scale_out(num_replicas=N)` ray_native 模式 | K8s Deployment replicas |
| 引擎注册 | 内部自动完成 | Pod lifecycle hooks 调 external 模式 |

内置 Autoscaler 使用 **ray_native** 模式在 Ray 集群内创建 Actor 和 PlacementGroup；KEDA 方案使用 **external** 模式将 K8s 管理的 Pod 注册为外部引擎。

______________________________________________________________________

## 为什么选 KEDA

| 维度 | Prometheus Adapter | KEDA |
|---|---|---|
| 配置复杂度 | 需要写复杂的 relabeling 规则和 custom metrics API | 一个 ScaledObject YAML |
| 多指标组合 | HPA 原生支持但配置繁琐 | 天然支持多 trigger（任一触发即扩容） |
| 缩容到零 | 不支持 | 支持 |
| 扩缩策略 | 标准 HPA behavior | 支持 HPA behavior + 高级策略 |
| 社区成熟度 | K8s 官方附属项目 | CNCF 毕业项目，活跃维护 |

KEDA 的多 trigger 默认使用 OR 逻辑（任一触发即扩容），与 Relax 内置 Autoscaler 的扩容策略一致。

______________________________________________________________________

## K8s 资源配置

### 1. KEDA ScaledObject

根据 SGLang Prometheus 指标自动调整 Pod 数量。

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sglang-engine-scaler
  namespace: relax-training
spec:
  scaleTargetRef:
    name: sglang-external-engines  # 指向 SGLang Deployment
  minReplicaCount: 2               # 最小引擎数
  maxReplicaCount: 16              # 最大引擎数
  cooldownPeriod: 300              # 缩容冷却期（秒）
  pollingInterval: 30              # 评估间隔（秒）
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 30   # 扩容稳定窗口
          policies:
          - type: Pods
            value: 4                       # 单次最多扩 4 个 Pod
            periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 120  # 缩容稳定窗口（保守）
          policies:
          - type: Pods
            value: 1                       # 单次最多缩 1 个 Pod
            periodSeconds: 300
  triggers:
    # 触发条件 1: KV Cache 利用率 > 85%
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: sglang_token_usage_avg
        threshold: "0.85"
        query: |
          avg(sglang:token_usage{job="sglang-engines"})
    # 触发条件 2: 每引擎排队请求数 > 10
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: sglang_queue_per_engine
        threshold: "10"
        query: |
          sum(sglang:num_queue_reqs{job="sglang-engines"})
          /
          count(sglang:num_queue_reqs{job="sglang-engines"})
```

**指标对应关系（与 Relax 内置 Autoscaler 一致）：**

| KEDA trigger | 对应 Relax 条件 | 含义 |
|---|---|---|
| `sglang_token_usage_avg > 0.85` | `token_usage_high` | KV Cache 利用率过高 |
| `sglang_queue_per_engine > 10` | `queue_backlog` | 排队请求积压 |

可根据需要添加更多 trigger（如 `queue_time_p95 > 5s`、`ttft_p95 > 10s`）。

### 2. SGLang Engine Deployment（含 Lifecycle Hooks）

通过 `postStart` 和 `preStop` hook 打通 K8s Pod 生命周期与 Relax 引擎注册/注销。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-external-engines
  namespace: relax-training
spec:
  replicas: 2  # 初始引擎数，由 KEDA 动态调整
  selector:
    matchLabels:
      app: sglang-engine
  template:
    metadata:
      labels:
        app: sglang-engine
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "30000"
        prometheus.io/path: "/metrics"
    spec:
      terminationGracePeriodSeconds: 180  # 必须足够长，等待 drain 完成

      containers:
      - name: sglang
        image: your-registry/sglang:latest
        args:
        - "--model-path"
        - "/models/your-model"
        - "--port"
        - "30000"
        - "--tp"
        - "1"
        ports:
        - containerPort: 30000
          name: sglang

        readinessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 60    # SGLang 模型加载较慢
          periodSeconds: 10
          failureThreshold: 30       # 容忍 5 分钟启动

        resources:
          limits:
            nvidia.com/gpu: 1        # 按 TP 并行度调整
          requests:
            nvidia.com/gpu: 1

        env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: RELAX_ROLLOUT_URL
          value: "http://relax-rollout-service:8000/rollout"

        lifecycle:
          # ========== 扩容：引擎就绪后注册到 Relax ==========
          postStart:
            exec:
              command:
              - "/bin/sh"
              - "-c"
              - |
                ENGINE_URL="http://${POD_IP}:30000"
                MAX_WAIT=600
                WAITED=0

                # 等待 SGLang 引擎就绪
                # postStart 和容器 ENTRYPOINT 并行执行，必须等 SGLang 真正 ready
                while [ $WAITED -lt $MAX_WAIT ]; do
                  if curl -sf "${ENGINE_URL}/health" > /dev/null 2>&1; then
                    break
                  fi
                  sleep 5
                  WAITED=$((WAITED + 5))
                done

                if [ $WAITED -ge $MAX_WAIT ]; then
                  echo "ERROR: SGLang engine did not become healthy within ${MAX_WAIT}s"
                  exit 1
                fi

                # 调用 Relax scale_out (external 模式)
                # Relax 会执行: 健康检查 -> 权重同步 -> Router 注册
                # scale_out 是幂等的，重复注册返回 NOOP
                echo "Registering engine ${ENGINE_URL} with Relax..."
                RESPONSE=$(curl -sf -X POST "${RELAX_ROLLOUT_URL}/scale_out" \
                  -H "Content-Type: application/json" \
                  -d "{\"engine_urls\": [\"${ENGINE_URL}\"]}" \
                  --max-time 30)

                echo "Scale-out response: ${RESPONSE}"

          # ========== 缩容：优雅注销引擎，等待 drain 完成 ==========
          preStop:
            exec:
              command:
              - "/bin/sh"
              - "-c"
              - |
                ENGINE_URL="http://${POD_IP}:30000"

                echo "Initiating scale-in for ${ENGINE_URL}..."

                # Step 1: 调用 scale_in，指定要移除的引擎地址
                RESPONSE=$(curl -sf -X POST "${RELAX_ROLLOUT_URL}/scale_in" \
                  -H "Content-Type: application/json" \
                  -d "{\"engine_urls\": [\"${ENGINE_URL}\"]}" \
                  --max-time 30)

                REQ_ID=$(echo "${RESPONSE}" | grep -o '"request_id":"[^"]*"' | cut -d'"' -f4)

                if [ -z "${REQ_ID}" ]; then
                  echo "WARN: No request_id returned, engine may already be removed"
                  exit 0
                fi

                echo "Scale-in request: ${REQ_ID}"

                # Step 2: 轮询等待 drain + remove 完成
                # Relax 会: 停止流量 -> 等待在途请求 -> 注销引擎 -> 释放资源
                MAX_WAIT=150  # 需小于 terminationGracePeriodSeconds
                WAITED=0

                while [ $WAITED -lt $MAX_WAIT ]; do
                  STATUS=$(curl -sf "${RELAX_ROLLOUT_URL}/scale_in/${REQ_ID}" \
                    --max-time 10 | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

                  echo "Scale-in status: ${STATUS} (${WAITED}s elapsed)"

                  if [ "${STATUS}" = "COMPLETED" ] || [ "${STATUS}" = "FAILED" ]; then
                    break
                  fi

                  sleep 5
                  WAITED=$((WAITED + 5))
                done

                echo "Scale-in finished (status=${STATUS}), allowing pod termination"
```

### 3. Prometheus ServiceMonitor

让 Prometheus 自动发现并抓取所有 SGLang Pod 的 `/metrics`。

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sglang-engines
  namespace: relax-training
spec:
  selector:
    matchLabels:
      app: sglang-engine
  endpoints:
  - port: sglang
    path: /metrics
    interval: 10s
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-engines-metrics
  namespace: relax-training
  labels:
    app: sglang-engine
spec:
  selector:
    app: sglang-engine
  ports:
  - name: sglang
    port: 30000
    targetPort: 30000
  clusterIP: None  # Headless Service，让 Prometheus 发现每个 Pod
```

### 4. Relax Rollout Service（K8s Service）

将 Ray 集群中的 Rollout Service 暴露为 K8s Service，供 SGLang Pod 调用。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: relax-rollout-service
  namespace: relax-training
spec:
  selector:
    app: relax-ray-head   # 指向 Ray head node
  ports:
  - name: rollout-api
    port: 8000
    targetPort: 8000      # Ray Serve 默认端口
```

______________________________________________________________________

## 时序与生命周期

### 扩容时序

```
K8s 创建 Pod
  → 容器启动，SGLang 开始加载模型
  → postStart hook 并行启动，轮询等待 /health 就绪
  → SGLang ready
  → postStart 调用 POST /rollout/scale_out {"engine_urls": ["http://<pod-ip>:30000"]}
  → Relax RolloutManager 执行:
      CONNECTING → HEALTH_CHECKING → WEIGHT_SYNCING → READY → ACTIVE
  → 引擎接入流量
```

### 缩容时序

```
K8s 发送 SIGTERM
  → preStop hook 拦截
  → 调用 POST /rollout/scale_in {"engine_urls": ["http://<pod-ip>:30000"]}
  → Relax RolloutManager 执行:
      PENDING → DRAINING (停止新流量，等待在途请求)
             → REMOVING (注销引擎)
             → COMPLETED
  → preStop 轮询到 COMPLETED，退出
  → K8s 终止 Pod
```

### 关键时间参数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `terminationGracePeriodSeconds` | 180s | 必须 > drain timeout + shutdown timeout + 轮询开销 |
| `--scale-in-drain-timeout` (Relax) | 30s (默认) | 等待在途请求完成的超时 |
| `--scale-in-shutdown-timeout` (Relax) | 20s (默认) | 引擎优雅关闭超时 |
| preStop `MAX_WAIT` | 150s | 需 < `terminationGracePeriodSeconds` |
| KEDA `cooldownPeriod` | 300s | 缩容冷却期，防止频繁扩缩 |

______________________________________________________________________

## 幂等性与安全

### scale_out 幂等性

Relax 的 external 模式 `scale_out` 天然幂等（`relax/components/rollout.py:29`）：

- 已注册的 `engine_urls` 会被自动过滤，返回 `NOOP`
- 正在处理中的 in-flight 请求中的地址也会被过滤
- Pod 重启后 `postStart` 重新注册是安全的

### scale_in 安全性

- 初始引擎（由 `--rollout-num-gpus` 启动参数定义）受保护，不会被缩容
- 缩容前会检查权重同步状态，如果正在进行权重更新则等待完成
- 按 LIFO（后进先出）策略优先移除最近扩容的引擎

### 互斥保护

同一时刻只允许一个扩/缩容操作执行（HTTP 409），KEDA 的 `cooldownPeriod` 和 HPA `stabilizationWindowSeconds` 进一步防止并发冲突。

______________________________________________________________________

## 权重同步

通过 external 模式扩容的引擎，权重同步走 **Remote Instance Sync**：从 seed engine（初始引擎）通过 NCCL Broadcast 直接传输权重到新引擎。

### 网络要求

- SGLang Pod 与 Ray 集群中的 seed engine 之间需要 **GPU 直连**（NCCL 通信）
- 如果使用 RoCE/InfiniBand，需要确保 NCCL 端口开放
- 如果跨网段，需要设置 `NCCL_SOCKET_IFNAME` 等环境变量

### 跨集群场景

如果 SGLang Pod 和 Ray 集群不在同一网络（跨集群联邦推理），NCCL Broadcast 无法进行：

- 引擎会使用初始模型权重运行
- 后续权重更新在 Actor 的 `update_weights_fully_async()` 完成后自动触发
- 如果对权重一致性要求严格，参考 [弹性 Rollout 扩缩容文档](./elastic-rollout.md) 中的权重同步机制

______________________________________________________________________

## 完整部署步骤

### 1. 安装 KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

### 2. 部署 Prometheus（如果未安装）

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

### 3. 部署 SGLang Engines + KEDA

```bash
kubectl create namespace relax-training

# 部署 SGLang Engine Deployment（含 lifecycle hooks）
kubectl apply -f sglang-deployment.yaml

# 部署 ServiceMonitor（Prometheus 抓取）
kubectl apply -f sglang-servicemonitor.yaml

# 部署 Relax Rollout Service（K8s Service）
kubectl apply -f relax-rollout-service.yaml

# 部署 KEDA ScaledObject
kubectl apply -f sglang-scaledobject.yaml
```

### 4. 启动 Relax 训练（不启用内置 Autoscaler）

```bash
ray job submit -- python3 relax/entrypoints/train.py \
    --fully-async \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 1 \
    --scale-out-timeout 600 \
    --scale-out-partial-success-policy keep_partial \
    --scale-in-drain-timeout 60 \
    --scale-in-shutdown-timeout 30 \
    ... # 其他训练参数（不传 --autoscaler-config）
```

### 5. 验证

```bash
# 查看 KEDA ScaledObject 状态
kubectl get scaledobject sglang-engine-scaler -n relax-training

# 查看 HPA 状态
kubectl get hpa -n relax-training

# 查看 SGLang Pod 状态
kubectl get pods -l app=sglang-engine -n relax-training

# 查询 Relax 引擎状态
curl http://<rollout-host>:8000/rollout/engines
```

______________________________________________________________________

## 监控与排障

### 关键监控项

| 监控项 | 查看方式 |
|---|---|
| KEDA 扩缩事件 | `kubectl describe scaledobject sglang-engine-scaler` |
| HPA 当前指标 | `kubectl get hpa -n relax-training -o wide` |
| Pod 扩缩历史 | `kubectl get events -n relax-training --field-selector reason=SuccessfulRescale` |
| Relax 引擎列表 | `GET /rollout/engines` |
| Relax scale_out 请求 | `GET /rollout/scale_out` |
| Relax scale_in 请求 | `GET /rollout/scale_in` |

### 常见问题

| 问题 | 原因 | 解决方案 |
|---|---|---|
| Pod 启动但未注册到 Relax | postStart hook 失败 | 检查 Pod events，确认 `RELAX_ROLLOUT_URL` 可达 |
| 缩容时 Pod 被强制终止 | `terminationGracePeriodSeconds` 太短 | 增大到 180s 以上 |
| 权重同步失败 | NCCL 网络不通 | 检查 GPU 网络互通，确认 NCCL 端口开放 |
| KEDA 不触发扩容 | Prometheus 未抓取到指标 | 检查 ServiceMonitor 和 Prometheus targets |
| scale_out 返回 CONFLICT | 有进行中的扩缩操作 | 等待当前操作完成，KEDA cooldownPeriod 会自动处理 |

______________________________________________________________________

## 延伸阅读

- [弹性 Rollout 扩缩容](./elastic-rollout.md) — Relax 弹性扩缩容完整文档
- [全异步训练流水线](./fully-async-training.md) — 弹性扩缩容的基础运行模式
- [KEDA 官方文档](https://keda.sh/docs/) — KEDA ScaledObject 配置参考
- [Prometheus Operator](https://prometheus-operator.dev/) — ServiceMonitor 配置参考
