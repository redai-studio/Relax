# Health Check Manager

The Health Check Manager provides comprehensive service health monitoring and automatic recovery for distributed training in Relax. It supports two recovery strategies: **in-place restart** for individual services and **global restart** for full system re-initialization.

## Overview

The health management system monitors all deployed services (Actor, Rollout, Critic, ActorFwd, Advantages, etc.) and automatically triggers recovery when failures are detected. Two detection mechanisms work together:

- **Error-based detection**: Services self-report errors (~1s latency)
- **Heartbeat-based detection**: Stale heartbeat timeout (~120s latency)

## Quick Start

Rollout and standalone ActorFwd also have a separate explicit-error recovery path that can preserve the training Actor. See [Local Inference Recovery](./inference-recovery.md) for its defaults, supported configurations, and interaction with health checks.

Enable health checking by adding `--use-health-check` to your training command:

```bash
python3 relax/entrypoints/train.py \
    --use-health-check \
    --max-global-restart 3 \
    --fully-async \
    ...
```

Or in your launch script:

```bash
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 relax/entrypoints/train.py \
   --use-health-check \
   --max-global-restart 3 \
   ...
```

## Architecture

### Component Design

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

### Components

| Component              | Responsibility                                                           | Implementation           |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------ |
| **ServiceHealthState** | Per-service health data (healthy, error, heartbeat, step, restart_count) | Python dataclass         |
| **HealthStatus**       | Store and query all service health state                                 | Ray remote actor         |
| **HealthChecker**      | Periodically check health, detect stale heartbeat, trigger callbacks     | Background daemon thread |
| **HealthManager**      | Coordinate both components, expose unified interface                     | Composite wrapper        |

## Health State

Each service has a `ServiceHealthState` dataclass:

```python
@dataclass
class ServiceHealthState:
    healthy: bool = True              # Current health status
    error: Optional[str] = None       # Error message if unhealthy
    last_heartbeat: float = time.time() # Last heartbeat timestamp
    current_step: int = 0             # Latest reported training step
    task_running: bool = False        # Whether the service task is running
    restart_count: int = 0            # Number of restarts for this service
```

## Detection Mechanisms

### Error-Based Detection

Services self-report errors via `report_error()`. The HealthChecker polls `get_unhealthy_services()` every second and triggers recovery immediately.

```python
# Inside service background loop (e.g., Actor._background_run):
try:
    ray.get(self.actor_model.train_fully_async(self.step))
    # Report heartbeat on success
    self.healthy.update_heartbeat.remote("actor", self.step + 1)
except Exception as e:
    # Report error on failure — triggers recovery within ~1s
    self.healthy.report_error.remote("actor", f"Training failed: {e}")
```

### Heartbeat-Based Detection

Each service sends periodic heartbeats. If no heartbeat is received for 120 seconds (`HEARTBEAT_TIMEOUT`), the service is considered stale and recovery is triggered.

This catches cases where a service crashes silently without calling `report_error()`.

## Recovery Strategies

### Two-Tier Recovery

| Condition                                     | Strategy             | Description                                      |
| --------------------------------------------- | -------------------- | ------------------------------------------------ |
| **Actor / Rollout / ActorFwd** fails          | **Global restart**   | Full Controller teardown + re-init from zero     |
| Any role restart_count **≥ 3**                | **Global restart**   | System is unstable, need clean slate             |
| Other roles fail (e.g., Critic, Advantages)   | **In-place restart** | Reuse placement group, restore step, re-run task |
| Global restart count > `--max-global-restart` | **Terminate**        | `os._exit(1)` — system is unrecoverable          |

### Global Restart

Global restart performs a full teardown of all resources (Ray Serve deployments, data system, DCS coordinator, Ray itself) and re-initializes everything from zero via `Controller.__init__()`.

```
Phase 1 — Teardown:
  1. Stop HealthManager
  2. Delete all Ray Serve deployments
  3. Delete metrics & DCS deployments
  4. Kill data system actors
  5. serve.shutdown() + ray.shutdown()
  6. Wait for process cleanup
  7. ray.init() + serve.start()

Phase 2 — Re-initialize:
  8. Controller.__init__(config, runtime_env)
  9. Signal main thread → re-run training loop
```

### In-Place Restart

In-place restart reuses the existing placement group and restores the service to its current step:

```
1. Save current step (via HTTP GET /get_step)
2. Stop heartbeat thread
3. Stop old deployment (HTTP POST /stop_service)
4. Delete Ray Serve deployment
5. Validate / rebuild placement group
6. Redeploy service
7. Restore step (HTTP POST /set_step)
8. Re-run service task
```

## Configuration

| Argument               | Type | Default | Description                                       |
| ---------------------- | ---- | ------- | ------------------------------------------------- |
| `--use-health-check`   | flag | False   | Enable the global health check system             |
| `--max-global-restart` | int  | 3       | Maximum global restarts before forced termination |

### Programmatic Configuration

```python
from relax.utils.health_system import HealthManager

# Create health manager with custom check interval
health_manager = HealthManager(check_interval=1.0)

# Start with callback
health_manager.start(on_unhealthy=handle_unhealthy_service)

# Query health
health = health_manager.get_service_health("actor")
print(health)  # {"healthy": True, "error": None, "current_step": 42, ...}

# Stop
health_manager.stop(timeout=2.0)
```

## Health Logs

When health checking is enabled, you'll see logs like:

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

## Debugging Endpoints

Services expose HTTP endpoints for debugging and recovery:

| Endpoint                             | Method | Description                               |
| ------------------------------------ | ------ | ----------------------------------------- |
| `/{role}/get_step`                   | GET    | Get current training step                 |
| `/{role}/set_step?step=N`            | POST   | Set training step                         |
| `/{role}/stop_service`               | POST   | Gracefully stop the service               |
| `/{role}/mark_unhealthy_for_testing` | GET    | Force service to report unhealthy (debug) |

Example:

```bash
# Check actor step
curl http://localhost:8000/actor/get_step

# Force actor unhealthy (triggers restart)
curl http://localhost:8000/actor/mark_unhealthy_for_testing
```

## Best Practices

1. **Always enable for production**: Use `--use-health-check` in production training runs
2. **Set appropriate max restarts**: `--max-global-restart 3` is a good default; increase for long runs
3. **Monitor logs**: Watch for repeated restart patterns which may indicate a persistent issue
4. **Save checkpoints frequently**: Use `--save-interval` so restarts don't lose too much progress
5. **Use async save**: `--async-save` prevents checkpoint saving from blocking training

## Troubleshooting

### Health Check Not Triggering Recovery

1. Verify `--use-health-check` is in your launch command
2. Check that the service properly calls `report_error()` on failure
3. For heartbeat-based detection, wait at least 120s after service crash

### Global Restart Fails

1. Check logs for `[Global Restart] Failed to ...` messages
2. If `ray.shutdown()` fails, stale processes may hold GPU memory
3. If max restarts exceeded, check the root cause of repeated failures

### Service Not Recovering After Restart

1. Check if placement group was recreated successfully
2. Verify the service step was restored correctly
3. Check for NCCL/distributed group issues in the new service

## Next Steps

- [Metrics Service](/en/guide/metrics-service-detailed) - Monitor service metrics
- [Notification System](/en/guide/notification-system) - Alert on health issues
- [Architecture](/en/guide/architecture) - Understand the system design
