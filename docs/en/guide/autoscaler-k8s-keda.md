# K8s HPA + KEDA Autoscaling Integration

## Overview

This guide shows how to combine Relax's elastic Rollout scaling with Kubernetes HPA through KEDA. Kubernetes automatically scales GPU resources, while Relax automatically registers and deregisters rollout engines.

### How It Works

| Layer | Responsibility | Implementation |
|---|---|---|
| **K8s resource layer** | Scale Pods (GPU resources) based on SGLang metrics | KEDA ScaledObject + Prometheus |
| **Engine registration layer** | Call `scale_out` in external mode after a Pod is ready, and call `scale_in` to drain traffic before scale-down | Pod Lifecycle Hooks |

**No Relax code changes are required**: the external `scale_out`/`scale_in` APIs, idempotency, weight synchronization, and draining are already implemented in `relax/components/rollout.py` and `relax/distributed/ray/rollout.py`.

### Prerequisites

- Training uses **Fully Async mode** (`--fully-async`).
- Rollout engines use **SGLang** as the inference backend.
- [KEDA](https://keda.sh/) and [Prometheus](https://prometheus.io/) are installed in the K8s cluster.
- SGLang Pods can reach the Ray cluster over the network. NCCL weight synchronization requires direct GPU connectivity or RoCE/InfiniBand.
- Do **not** pass `--autoscaler-config` when starting training. This keeps the built-in Autoscaler disabled and avoids conflicts.

______________________________________________________________________

## Architecture

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
│  │  │ Rollout Gateway (/rollout)  │     │                       │
│  │  │ POST /rollout/scale_out     │     │                       │
│  │  │ POST /rollout/scale_in      │     │                       │
│  │  │ GET  /rollout/engines       │     │                       │
│  │  └─────────────────────────────┘     │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:**

1. Prometheus scrapes `/metrics` from all SGLang Pods.
2. KEDA queries aggregated metrics with PromQL and drives HPA to adjust the Deployment replicas.
3. After a new Pod starts, the `postStart` hook waits for SGLang to become ready, then calls Relax `scale_out` in external mode to register the engine.
4. During scale-down, the `preStop` hook calls Relax `scale_in` to drain traffic and waits for completion before allowing the Pod to terminate.

______________________________________________________________________

## Comparison with the Built-in Relax Autoscaler

The built-in `AutoscalerService` (`relax/utils/autoscaler/`) and the K8s KEDA setup should not be enabled at the same time. Their responsibilities are:

| Responsibility | Built-in Autoscaler | K8s KEDA |
|---|---|---|
| Metrics collection | `MetricsCollector` polls `/metrics` | Prometheus scraping |
| Scaling decisions | `ScalingDecisionEngine` | KEDA ScaledObject |
| Resource allocation | `scale_out(num_replicas=N)` in ray_native mode | K8s Deployment replicas |
| Engine registration | Handled automatically | Pod lifecycle hooks call external mode |

The built-in Autoscaler uses **ray_native** mode to create Actors and PlacementGroups in the Ray cluster. The KEDA setup uses **external** mode to register K8s-managed Pods as external engines.

______________________________________________________________________

## Why KEDA

| Area | Prometheus Adapter | KEDA |
|---|---|---|
| Setup complexity | Requires complex relabeling rules and the custom metrics API | One ScaledObject YAML |
| Multiple metrics | Supported by HPA, but configuration is verbose | Supports multiple triggers directly; any trigger can scale up |
| Scale to zero | Not supported | Supported |
| Scaling policy | Standard HPA behavior | HPA behavior plus advanced policies |
| Project maturity | Kubernetes ecosystem project | CNCF graduated project with active maintenance |

KEDA uses OR logic across multiple triggers by default, so any trigger can scale up. This matches the scale-out behavior of the built-in Relax Autoscaler.

______________________________________________________________________

## K8s Resource Configuration

### 1. KEDA ScaledObject

Automatically adjusts the number of Pods using SGLang Prometheus metrics.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sglang-engine-scaler
  namespace: relax-training
spec:
  scaleTargetRef:
    name: sglang-external-engines  # Target SGLang Deployment
  minReplicaCount: 2               # Minimum engine count
  maxReplicaCount: 16              # Maximum engine count
  cooldownPeriod: 300              # Applies only when scaling to 0
  pollingInterval: 30              # Polling interval in seconds
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 30   # Scale-up stabilization window
          policies:
          - type: Pods
            value: 4                       # Scale up by at most 4 Pods at a time
            periodSeconds: 60
        scaleDown:
          stabilizationWindowSeconds: 120  # Conservative scale-down stabilization window
          policies:
          - type: Pods
            value: 1                       # Scale down by at most 1 Pod at a time
            periodSeconds: 300
  triggers:
    # Trigger 1: KV Cache usage > 85%
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: sglang_token_usage_avg
        threshold: "0.85"
        query: |
          avg(sglang:token_usage{job="sglang-engines"})
    # Trigger 2: queued requests per engine > 10
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

**Metric mapping (aligned with the built-in Relax Autoscaler):**

| KEDA trigger | Relax condition | Meaning |
|---|---|---|
| `sglang_token_usage_avg > 0.85` | `token_usage_high` | High KV Cache usage |
| `sglang_queue_per_engine > 10` | `queue_backlog` | Queued request backlog |

Add more triggers as needed, such as `queue_time_p95 > 5s` or `ttft_p95 > 10s`.

### 2. SGLang Engine Deployment (with Lifecycle Hooks)

Use `postStart` and `preStop` hooks to connect the K8s Pod lifecycle with Relax engine registration and deregistration.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-external-engines
  namespace: relax-training
spec:
  replicas: 2  # Initial engine count, adjusted by KEDA
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
      terminationGracePeriodSeconds: 180  # Must be long enough for draining to finish

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
          initialDelaySeconds: 60    # SGLang model loading can be slow
          periodSeconds: 10
          failureThreshold: 30       # Allow up to 5 minutes for startup

        resources:
          limits:
            nvidia.com/gpu: 1        # Adjust based on TP size
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
          # ========== Scale out: register the engine after it is ready ==========
          postStart:
            exec:
              command:
              - "/bin/sh"
              - "-c"
              - |
                ENGINE_URL="http://${POD_IP}:30000"
                MAX_WAIT=600
                WAITED=0

                # Wait for the SGLang engine to become ready
                # postStart runs in parallel with the container ENTRYPOINT, so wait until SGLang is actually ready
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

                # Call Relax scale_out in external mode
                # Relax runs: health check -> weight sync -> Router registration
                # scale_out is idempotent; duplicate registration returns NOOP
                echo "Registering engine ${ENGINE_URL} with Relax..."
                RESPONSE=$(curl -sf -X POST "${RELAX_ROLLOUT_URL}/scale_out" \
                  -H "Content-Type: application/json" \
                  -d "{\"engine_urls\": [\"${ENGINE_URL}\"]}" \
                  --max-time 30)

                echo "Scale-out response: ${RESPONSE}"

          # ========== Scale in: gracefully deregister the engine and wait for draining ==========
          preStop:
            exec:
              command:
              - "/bin/sh"
              - "-c"
              - |
                ENGINE_URL="http://${POD_IP}:30000"

                echo "Initiating scale-in for ${ENGINE_URL}..."

                # Step 1: Call scale_in with the engine URL to remove
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

                # Step 2: Poll until draining and removal are complete
                # Relax: stop new traffic -> wait for in-flight requests -> deregister the engine -> release resources
                MAX_WAIT=150  # Must be less than terminationGracePeriodSeconds
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

Lets Prometheus automatically discover all SGLang Pods and scrape their `/metrics` endpoints.

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
  clusterIP: None  # Headless Service so Prometheus can discover each Pod
```

### 4. Relax Rollout Service (K8s Service)

Expose the Rollout Service in the Ray cluster as a K8s Service so SGLang Pods can call it.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: relax-rollout-service
  namespace: relax-training
spec:
  selector:
    app: relax-ray-head   # Target the Ray head node
  ports:
  - name: rollout-api
    port: 8000
    targetPort: 8000      # Default Ray Serve port
```

______________________________________________________________________

## Lifecycle and Flow

### Scale-Out Flow

```
K8s creates the Pod
  → The container starts and SGLang begins loading the model
  → The postStart hook starts in parallel and polls /health until it is ready
  → SGLang ready
  → postStart calls POST /rollout/scale_out {"engine_urls": ["http://<pod-ip>:30000"]}
  → Relax InferenceManager (RolloutEnginePool) runs:
      CONNECTING → HEALTH_CHECKING → WEIGHT_SYNCING → READY → ACTIVE
  → The engine starts receiving traffic
```

### Scale-In Flow

```
K8s sends SIGTERM
  → The preStop hook handles the signal
  → Call POST /rollout/scale_in {"engine_urls": ["http://<pod-ip>:30000"]}
  → Relax InferenceManager (RolloutEnginePool) runs:
      PENDING → DRAINING (stop new traffic and wait for in-flight requests)
             → REMOVING (deregister the engine)
             → COMPLETED
  → preStop polls until COMPLETED, then exits
  → K8s terminates the Pod
```

### Key Timing Settings

| Parameter | Recommended value | Description |
|---|---|---|
| `terminationGracePeriodSeconds` | 180s | Must be greater than drain timeout + shutdown timeout + polling overhead |
| `--scale-in-drain-timeout` (Relax) | 30s (default) | Timeout for in-flight requests to finish |
| `--scale-in-shutdown-timeout` (Relax) | 30s (default) | Graceful engine shutdown timeout |
| preStop `MAX_WAIT` | 150s | Must be less than `terminationGracePeriodSeconds` |
| KEDA `cooldownPeriod` | 300s | Cooldown before scaling to 0; not used when `minReplicaCount` > 0 |

______________________________________________________________________

## Idempotency and Safety

### scale_out Idempotency

Relax's external-mode `scale_out` is idempotent by design (`relax/components/rollout.py:29`):

- Already registered `engine_urls` are filtered automatically, and the request returns `NOOP`.
- Engine URLs already being handled by in-flight requests are also filtered.
- It is safe for `postStart` to register the engine again after a Pod restart.

### scale_in Safety

- Initial engines defined by `--rollout-num-gpus` are protected and cannot be scaled in.
- Before scaling in, Relax checks the weight synchronization state and waits if a weight update is still running.
- Relax uses LIFO (last in, first out), so the most recently added engines are removed first.

### Concurrency Protection

Only one scale-out or scale-in operation can run at a time (HTTP 409). HPA `stabilizationWindowSeconds` helps reduce rapid scaling changes.

______________________________________________________________________

## Weight Synchronization

For engines added in external mode, Relax uses **Remote Instance Sync**. Weights are sent directly from a seed engine (an initial engine) to the new engine through NCCL Broadcast.

### Network Requirements

- SGLang Pods need **direct GPU connectivity** to the seed engine in the Ray cluster for NCCL communication.
- If you use RoCE/InfiniBand, make sure the required NCCL ports are open.
- If traffic crosses network segments, configure environment variables such as `NCCL_SOCKET_IFNAME`.

### Cross-Cluster Setup

If the SGLang Pods and Ray cluster are not on the same network, such as in cross-cluster federated inference, NCCL Broadcast cannot be used:

- The engine runs with the initial model weights.
- Later weight updates are triggered automatically after the Actor finishes `update_weights_fully_async()`.
- If strict weight consistency is required, see the weight synchronization section in [Elastic Rollout Scaling](./elastic-rollout.md).

______________________________________________________________________

## Deployment Steps

### 1. Install KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

### 2. Deploy Prometheus (if not already installed)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

### 3. Deploy SGLang Engines + KEDA

```bash
kubectl create namespace relax-training

# Deploy the SGLang Engine Deployment with lifecycle hooks
kubectl apply -f sglang-deployment.yaml

# Deploy the ServiceMonitor for Prometheus scraping
kubectl apply -f sglang-servicemonitor.yaml

# Deploy the Relax Rollout Service as a K8s Service
kubectl apply -f relax-rollout-service.yaml

# Deploy the KEDA ScaledObject
kubectl apply -f sglang-scaledobject.yaml
```

### 4. Start Relax Training Without the Built-in Autoscaler

```bash
ray job submit -- python3 relax/entrypoints/train.py \
    --fully-async \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 1 \
    --scale-out-timeout 600 \
    --scale-out-partial-success-policy keep_partial \
    --scale-in-drain-timeout 60 \
    --scale-in-shutdown-timeout 30 \
    ... # Other training arguments (do not pass --autoscaler-config)
```

### 5. Verify

```bash
# Check KEDA ScaledObject status
kubectl get scaledobject sglang-engine-scaler -n relax-training

# Check HPA status
kubectl get hpa -n relax-training

# Check SGLang Pod status
kubectl get pods -l app=sglang-engine -n relax-training

# Check Relax engine status
curl http://<rollout-host>:8000/rollout/engines
```

______________________________________________________________________

## Monitoring and Troubleshooting

### What to Monitor

| Item | How to check |
|---|---|
| KEDA scaling events | `kubectl describe scaledobject sglang-engine-scaler` |
| Current HPA metrics | `kubectl get hpa -n relax-training -o wide` |
| Pod scaling history | `kubectl get events -n relax-training --field-selector reason=SuccessfulRescale` |
| Relax engine list | `GET /rollout/engines` (v2 format; add `?schema_version=1` for the legacy format) |
| Relax scale_out requests | `GET /rollout/scale_out` |
| Relax scale_in requests | `GET /rollout/scale_in` |

### Common Issues

| Issue | Cause | Solution |
|---|---|---|
| Pod starts but is not registered with Relax | The postStart hook fails | Check Pod events and make sure `RELAX_ROLLOUT_URL` is reachable |
| Pod is force-terminated during scale-down | `terminationGracePeriodSeconds` is too short | Increase it to more than 180s |
| Weight synchronization fails | NCCL network connectivity fails | Check GPU network connectivity and make sure NCCL ports are open |
| KEDA does not scale out | Prometheus is not scraping the metrics | Check the ServiceMonitor and Prometheus targets |
| scale_out returns CONFLICT | Another scaling operation is in progress | Wait for the current operation to finish before retrying. |

______________________________________________________________________

## Further Reading

- [Elastic Rollout Scaling](./elastic-rollout.md) — Full documentation for Relax elastic scaling
- [Fully Async Training Pipeline](./fully-async-training.md) — Base runtime mode for elastic scaling
- [KEDA documentation](https://keda.sh/docs/) — ScaledObject configuration reference
- [Prometheus Operator](https://prometheus-operator.dev/) — ServiceMonitor configuration reference
