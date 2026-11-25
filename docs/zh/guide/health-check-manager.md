# 健康检查管理器

健康检查管理器为 Relax 分布式训练系统提供全面的服务健康监控和自动恢复能力。它支持两种恢复策略：**就地重启**（针对单个服务）和**全局重启**（全系统重新初始化）。

## 概述

健康管理系统监控所有已部署的服务（Actor、Rollout、Critic、ActorFwd、Advantages 等），并在检测到故障时自动触发恢复。两种检测机制协同工作：

- **错误上报检测**：服务主动上报错误（延迟约 1 秒）
- **心跳超时检测**：心跳超时判定（超时阈值 120 秒）

## 快速开始

Rollout 和独立 ActorFwd 还有一条针对明确异常的局部恢复路径，可以保留训练 Actor。它的默认开关、支持条件以及与健康检查的关系，见[推理服务局部自愈](./inference-recovery.md)。

在训练命令中添加 `--use-health-check` 即可启用健康检查：

```bash
python3 relax/entrypoints/train.py \
    --use-health-check \
    --max-global-restart 3 \
    --fully-async \
    ...
```

或在启动脚本中：

```bash
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 relax/entrypoints/train.py \
   --use-health-check \
   --max-global-restart 3 \
   ...
```

## 架构

### 组件设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Controller (Consumer)                        │
│  uses: HealthManager                                            │
│  owns: restart_serve() → in-place restart / global restart      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│          HealthManager (Coordinator Layer)                      │
│  Owns and coordinates internal components                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
            ┌────────┴──────────┐
            │                   │
┌───────────▼──────┐  ┌────────▼────────────────┐
│  HealthStatus    │  │  HealthChecker          │
│  (Remote Actor)  │  │  (Background Thread)    │
│                  │  │                         │
│ Stores per-svc   │  │ Periodic checking       │
│ health state     │  │ Unhealthy + stale       │
│                  │  │ detection               │
└──────────────────┘  └─────────────────────────┘
```

### 组件说明

| 组件                   | 职责                                                       | 实现方式         |
| ---------------------- | ---------------------------------------------------------- | ---------------- |
| **ServiceHealthState** | 每个服务的健康数据（健康状态、错误、心跳、步数、重启次数） | Python dataclass |
| **HealthStatus**       | 存储和查询所有服务的健康状态                               | Ray 远程 Actor   |
| **HealthChecker**      | 定期检查健康状态，检测心跳超时，触发回调                   | 后台守护线程     |
| **HealthManager**      | 协调上述组件，对外提供统一接口                             | 组合包装类       |

## 健康状态

每个服务拥有一个 `ServiceHealthState` 数据类：

```python
@dataclass
class ServiceHealthState:
    healthy: bool = True              # 当前健康状态
    error: Optional[str] = None       # 不健康时的错误信息
    last_heartbeat: float = time.time() # 上次心跳时间戳
    current_step: int = 0             # 最新上报的训练步数
    task_running: bool = False        # 服务任务是否正在运行
    restart_count: int = 0            # 该服务的重启次数
```

## 检测机制

### 错误上报检测

服务通过 `report_error()` 主动上报错误。HealthChecker 每秒轮询 `get_unhealthy_services()`，一旦发现异常立即触发恢复。

```python
# 服务后台循环内部（如 Actor._background_run）：
try:
    ray.get(self.actor_model.train_fully_async(self.step))
    # 成功时上报心跳
    self.healthy.update_heartbeat.remote("actor", self.step + 1)
except Exception as e:
    # 失败时上报错误 —— 约 1 秒内触发恢复
    self.healthy.report_error.remote("actor", f"Training failed: {e}")
```

### 心跳超时检测

每个服务定期发送心跳。如果超过 120 秒（`HEARTBEAT_TIMEOUT`）未收到心跳，则判定该服务已失联并触发恢复。

这可以捕获服务无声崩溃（没有调用 `report_error()`）的情况。

## 恢复策略

### 两级恢复

| 条件                                    | 策略         | 说明                                         |
| --------------------------------------- | ------------ | -------------------------------------------- |
| **Actor / Rollout / ActorFwd** 故障     | **全局重启** | 完整的 Controller 销毁 + 从零重新初始化      |
| 任意角色重启次数 **≥ 3**                | **全局重启** | 系统不稳定，需要全新起步                     |
| 其他角色故障（如 Critic、Advantages）   | **就地重启** | 复用 placement group，恢复步数，重新运行任务 |
| 全局重启次数超过 `--max-global-restart` | **终止**     | `os._exit(1)` —— 系统无法恢复                |

### 全局重启

全局重启会完全销毁所有资源（Ray Serve 部署、数据系统、DCS 协调器、Ray 本身），然后通过 `Controller.__init__()` 从零重新初始化。

```
阶段一 —— 销毁：
  1. 停止 HealthManager
  2. 删除所有 Ray Serve 部署
  3. 删除 metrics 和 DCS 部署
  4. 终止数据系统 Actor
  5. serve.shutdown() + ray.shutdown()
  6. 等待进程清理
  7. ray.init() + serve.start()

阶段二 —— 重新初始化：
  8. Controller.__init__(config, runtime_env)
  9. 通知主线程 → 重新运行训练循环
```

### 就地重启

就地重启复用现有的 placement group，将服务恢复到当前步数：

```
1. 保存当前步数（通过 HTTP GET /get_step）
2. 停止心跳线程
3. 停止旧部署（HTTP POST /stop_service）
4. 删除 Ray Serve 部署
5. 验证/重建 placement group
6. 重新部署服务
7. 恢复步数（HTTP POST /set_step）
8. 重新运行服务任务
```

## 配置参数

| 参数                   | 类型 | 默认值 | 说明                           |
| ---------------------- | ---- | ------ | ------------------------------ |
| `--use-health-check`   | 标志 | False  | 启用全局健康检查系统           |
| `--max-global-restart` | 整数 | 3      | 达到最大全局重启次数后强制终止 |

### 编程接口

```python
from relax.utils.health_system import HealthManager

# 创建健康管理器（自定义检查间隔）
health_manager = HealthManager(check_interval=1.0)

# 启动（设置回调）
health_manager.start(on_unhealthy=handle_unhealthy_service)

# 查询健康状态
health = health_manager.get_service_health("actor")
print(health)  # {"healthy": True, "error": None, "current_step": 42, ...}

# 停止
health_manager.stop(timeout=2.0)
```

## 健康日志

启用健康检查后，您将看到如下日志：

```
[INFO] Global health check system enabled
[INFO] Health checker started
[INFO] HealthManager initialized
[WARNING] Service actor is unhealthy: RayTaskError(OutOfMemoryError), triggering restart
[INFO] Restarting service 'actor'...
[WARNING] Triggering global restart due to: actor failure
[INFO] === Starting GLOBAL restart #1 (max=3) ===
[INFO] [Global Restart] Health manager stopped
[INFO] [Global Restart] Deleted Ray Serve deployment 'actor'
[INFO] [Global Restart] Ray shutdown completed
[INFO] [Global Restart] Ray re-initialized
[INFO] [Global Restart] Controller re-initialized from zero via __init__
[INFO] === Global restart completed, main thread signaled ===
[INFO] Global restart completed, re-running training loop
```

## 调试端点

服务暴露 HTTP 端点用于调试和恢复：

| 端点                                 | 方法 | 说明                             |
| ------------------------------------ | ---- | -------------------------------- |
| `/{role}/get_step`                   | GET  | 获取当前训练步数                 |
| `/{role}/set_step?step=N`            | POST | 设置训练步数                     |
| `/{role}/stop_service`               | POST | 优雅停止服务                     |
| `/{role}/mark_unhealthy_for_testing` | GET  | 强制服务上报不健康状态（调试用） |

示例：

```bash
# 检查 actor 步数
curl http://localhost:8000/actor/get_step

# 强制 actor 不健康（触发重启）
curl http://localhost:8000/actor/mark_unhealthy_for_testing
```

## 最佳实践

1. **生产环境务必启用**：在生产训练中使用 `--use-health-check`
2. **设置合适的最大重启次数**：`--max-global-restart 3` 是合理的默认值；长时间训练可适当增大
3. **监控日志**：关注重复出现的重启模式，这可能表明存在持续性问题
4. **频繁保存检查点**：使用 `--save-interval` 确保重启不会丢失太多进度
5. **使用异步保存**：`--async-save` 防止检查点保存阻塞训练

## 故障排除

### 健康检查未触发恢复

1. 确认启动命令中包含 `--use-health-check`
2. 检查服务是否在故障时正确调用了 `report_error()`
3. 对于心跳超时检测，服务崩溃后需等待至少 120 秒

### 全局重启失败

1. 检查日志中的 `[Global Restart] Failed to ...` 消息
2. 如果 `ray.shutdown()` 失败，残留进程可能占用 GPU 显存
3. 如果超过最大重启次数，需排查反复失败的根本原因

### 服务重启后未恢复

1. 检查 placement group 是否成功重建
2. 验证服务步数是否正确恢复
3. 检查新服务是否有 NCCL/分布式通信组问题

## 相关文档

- [Metrics 服务](/zh/guide/metrics-service-detailed) - 监控服务指标
- [通知系统](/zh/guide/notification-system) - 健康问题告警
- [架构设计](/zh/guide/architecture) - 了解系统设计
