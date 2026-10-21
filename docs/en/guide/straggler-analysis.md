# Straggler analysis

Optional always-on, coarse-grained slow-rank observation for the Megatron actor path. When enabled, Relax replaces the usual `config.timers = None` with a non-blocking CUDA Event timer shim that reuses Megatron's existing timer call sites, aggregates fixed CPU scalars over Gloo every N rollouts, and emits `straggler/*` metrics.

v1 is **observe-only**: no eviction, topology changes, or training stalls.

## Enable

```bash
python3 relax/entrypoints/train.py \
  --straggler-analysis \
  --straggler-interval 10 \
  --straggler-relative-threshold 0.10 \
  --straggler-absolute-ms-threshold 5.0 \
  --straggler-persist-windows 3 \
  # ... other training args
```

Optional: `--straggler-enable-module-stages` records attention / MoE timer names when present.

When disabled (default), `config.timers` stays `None`. Evaluation / log-prob paths still force `config.timers = None`.

## Behavior

| Item | Behavior |
| --- | --- |
| Timing | start/stop only `record` events; completed events are drained with `event.query()`; no `cuda.synchronize()` on the hot path |
| Aggregation | Every 10 rollouts by default via a dedicated Gloo `all_gather_object` |
| Judgement | Compare within the same PP stage against the group median; alert only after N consecutive windows |
| Reasons | Priority: `data_imbalance` → `cpu_bound` → `slow_device` → `late_arrival` → `upstream_wait` |
| Metrics | `straggler/*` through `tracking_utils.log`, alongside `perf/*` |

## Local checks (not a recipe)

```bash
python -m pytest tests/utils/test_straggler_detector.py tests/utils/test_straggler_config.py -q
python scripts/tools/smoke_straggler_events.py
torchrun --nproc_per_node=4 scripts/tools/bench_straggler_4gpu.py --out /tmp/straggler_4gpu.json
```

The synthetic bench injects extra work on rank 3 and prints `overhead_pct` / `alerts`. Treat it as mechanism evidence, not official recipe acceptance.

## Related

- RFC: [#334](https://github.com/redai-studio/Relax/issues/334)
- Board: [#321](https://github.com/redai-studio/Relax/issues/321)
