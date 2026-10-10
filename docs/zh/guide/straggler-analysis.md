# Straggler 分析（慢卡观测）

默认关闭的轻量慢卡 / 阶段观测。打开后复用 Megatron 已有 `config.timers` 调用点，用非阻塞 CUDA Event 记录粗粒度阶段耗时，按窗口跨 rank 汇总，并上报 `straggler/*` 标量。

v1 **只观测**：不踢卡、不改拓扑、不挂起训练。

## 开启方式

```bash
python3 relax/entrypoints/train.py \
  --straggler-analysis \
  --straggler-interval 10 \
  --straggler-relative-threshold 0.10 \
  --straggler-absolute-ms-threshold 5.0 \
  --straggler-persist-windows 3 \
  # ... 其他训练参数
```

可选：`--straggler-enable-module-stages` 在 timer 名存在时额外记录 attention / MoE。

关闭时（默认）`config.timers` 保持 `None`，训练路径与现有行为一致。评测 / log-prob 路径仍强制 `config.timers = None`。

## 行为摘要

| 项 | 行为 |
| --- | --- |
| 计时 | start/stop 只 `record` Event；用 `event.query()` 读取已完成事件，热路径不 `cuda.synchronize()` |
| 汇总 | 默认每 10 个 rollout，经独立 Gloo `all_gather_object` 汇总固定 CPU 标量 |
| 判定 | 同一 PP stage 内与组中位数比较；连续 N 个窗口过线才告警 |
| 原因 | 优先级：`data_imbalance` → `cpu_bound` → `slow_device` → `late_arrival` → `upstream_wait` |
| 上报 | `straggler/*` 经 `tracking_utils.log`，与 `perf/*` 并存 |

## 本地验证（非 recipe）

```bash
# 判定逻辑单测
python -m pytest tests/utils/test_straggler_detector.py tests/utils/test_straggler_config.py -q

# 单卡 Event smoke
python scripts/tools/smoke_straggler_events.py

# 4 卡合成慢卡 + 开销
torchrun --nproc_per_node=4 scripts/tools/bench_straggler_4gpu.py \
  --out /tmp/straggler_4gpu.json
```

合成脚本会注入 rank3 额外计算，并打印 `overhead_pct` 与 `alerts`。这是机制可行性证据，不是官方 recipe 终验。

## 相关

- RFC：[#334](https://github.com/redai-studio/Relax/issues/334)
- 任务看板：[#321](https://github.com/redai-studio/Relax/issues/321)
