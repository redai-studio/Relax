# 配置检查与 Dry-run

Config Doctor 在启动 Ray、SGLang 或 GPU worker 之前，复用 Relax 现有训练参数解析器和各 backend 校验器完成前置检查。它适合本地检查和 CI，不替代正式训练入口。

## 使用方法

在 `--` 后传入原本跟在 `python -m relax.entrypoints.train` 后面的完整参数：

```bash
python -m relax.entrypoints.doctor -- \
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --colocate \
  --hf-checkpoint /models/Qwen3-4B \
  --prompt-data /data/train.jsonl \
  # ...其余训练参数
```

CI 可使用 JSON 输出：

```bash
python -m relax.entrypoints.doctor --format json -- <完整训练参数>
```

配置正确时返回码为 `0`，并输出最终合并配置、实际角色、GPU 需求和预计训练命令。配置或参数错误时返回非零状态，并同时输出错误原因和修复建议；未知或拼错的参数也会失败。

该入口不会执行 Shell 脚本。请先展开 recipe，再直接传入训练参数。报告会自动隐藏 API key、token、password、credential 和 `--wandb-key` 等敏感值。
