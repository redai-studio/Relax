# Case: Rollout Eval 等待 onload 状态导致 Hang

> 历史案例：下文代码片段来自统一推理重构之前的 `RolloutManager`，该类已删除。现在 eval 由 `RolloutWorker.eval` 执行，引擎状态归 `InferenceManager`（`RolloutEnginePool`）管理。排查思路（条件等待必须有可满足的退出条件）仍然适用，但不要按片段里的行号或方法名去找代码。

## 环境

- Ray 集群: 1 节点, 8 GPU
- 训练框架: Megatron + SGLang
- 配置: `--fully-async`, `offload_rollout=False`

## 现象

训练任务运行一段时间后 hang 住，GPU 利用率降至接近 0%。

## 排查过程

### 1. 集群状态

```bash
RAY_ADDRESS=10.x.x.x:6379 ray status --address 10.x.x.x:6379
```

结果: CPU 17.4/170.0, GPU 2.4/8.0，资源使用率低。

### 2. 运行中 Tasks

```bash
ray list tasks --address="10.x.x.x:6379" --filter "JOB_ID=02000000" --filter "state=RUNNING"
```

发现 4 个 RUNNING tasks:
- 2x `MegatronTrainRayActor.train_async`
- 1x `MegatronTrainRayActor.compute_actor_log_prob`
- 1x `MegatronTrainRayActor.compute_ref_log_prob`

### 3. 训练端调用栈

从 task yaml 获取 node_id，然后指定节点执行：

```bash
# 获取 node_id: 3d678d2c27a929a3e16e457e80d98cb95dc5ce94fab4fbc84ee3ed98
# PID: 2630709
ray job submit --working-dir "./" --address="10.x.x.x:6379" -- \
  python scripts/tools/run_on_each_ray_node.py -n 3d678d2c27a929a3e16e457e80d98cb95dc5ce94fab4fbc84ee3ed98 "py-spy dump --pid 2630709"
```

阻塞点:
```
transfer_queue/dataloader/streaming_dataset.py:217, in __iter__
    time.sleep(1)
```

结论: 训练端在等待数据，数据迭代器无数据。

### 4. Rollout 端调用栈

```bash
# PID: 2629242, 同一节点
ray job submit --working-dir "./" --address="10.x.x.x:6379" -- \
  python scripts/tools/run_on_each_ray_node.py -n 3d678d2c27a929a3e16e457e80d98cb95dc5ce94fab4fbc84ee3ed98 "py-spy dump --pid 2629242"
```

```bash
py-spy dump --pid 2630709  # train_async
```

阻塞点:
```
transfer_queue/dataloader/streaming_dataset.py:217, in __iter__
    time.sleep(1)
```

结论: 训练端在等待数据，数据迭代器无数据。

### 4. Rollout 端调用栈

```bash
py-spy dump --pid 2629242  # Rollout Actor
```

阻塞点:
```
relax/components/rollout.py:190, in _background_run
    time.sleep(1)
```

调用链:
```
_background_run → _should_eval(local_step) → while True: if get_status() == "onload"
```

### 5. 根因分析

**代码逻辑**:
```python
# relax/components/rollout.py:186-190
if self._should_eval(local_step):
    while True:
        if ray.get(self.rollout_manager.get_status.remote()) == "onload":
            break
        time.sleep(1)
```

**状态设置**:
```python
# relax/distributed/ray/rollout.py:594-596
def onload_kv(self):
    self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
    self.status = "onload"
```

**问题**:
- `onload_kv()` 只在 `offload_rollout=True` 时调用
- 当前配置 `offload_rollout=False`
- `status` 永远是 `None`，无法满足 `== "onload"` 条件
- 导致无限循环等待

## 修复方案

```python
# relax/components/rollout.py
if self._should_eval(local_step):
    # Only wait for onload status when offload_rollout is enabled
    if self.config.offload_rollout:
        while True:
            if ray.get(self.rollout_manager.get_status.remote()) == "onload":
                break
            time.sleep(1)
    self._logger.info(f"Evaluating after rollout {local_step}")
    ray.get(self.rollout_manager.eval.remote(rollout_id=local_step))
```

## 阻塞链条

```
Train (等待数据)
  ↑ 数据流
Rollout (等待 onload 状态)
  ↑ 条件
RolloutManager.status = None (因 offload_rollout=False 未设置)
```

## 关键启示

1. **条件等待必须有退出路径**: `while True` 循环等待必须有条件可满足
2. **配置与代码逻辑一致性**: 检查条件分支依赖的配置项是否有默认值/预期值不匹配
3. **状态机初始化**: 确保状态初始值与业务逻辑匹配
