# ChunkFlow

核心位于 `relax/backends/megatron/chunkflow/`。默认关闭，模型构建后由
`model_provider.py` 按 `--use-chunkflow` 显式安装。

| 模块 | 职责 |
| --- | --- |
| `scheduler.py`、`units.py` | 切分、装箱、依赖排序及流水线计划 |
| `schedules.py`、`pipeline_schedules.py` | 单阶段与流水线前后向 |
| `attention_ops.py`、`flash_backends.py` | 分块注意力及 FA2/FA3 |
| `state_store.py`、`runtime.py` | 状态缓存与释放 |
| `unit_batch.py`、`rope.py` | 目标、mask、位置对齐 |
| `integration.py`、`gdn.py` | 注意力安装及 GDN 状态梯度传递 |
| `metadata.py` | 序列元数据 |

MoE 继续使用 Megatron 原生路由、dispatch/combine 和专家计算，不替换专家内核。
保留 dense 注意力、GDN 混合架构及反序 backward 的梯度累积处理。

## 唯一训练示例

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/train.parquet
export CHUNK_SIZE=4096 KEEP_LAST_K=0
bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-4B-8xgpu-chunkflow.sh chunkflow
```

模型目录为 `${MODEL_DIR}/Qwen3-4B`；数据列默认 `question`、`answer`，可用
`INPUT_KEY`、`LABEL_KEY` 覆盖。将最后的 `chunkflow` 换成 `baseline` 可运行对照。
默认 8 卡 TP2/PP4、GBS8、32K 长度上限、100 步、FA3；FA3 不可用时设
`RELAX_CHUNKFLOW_FLASH_BACKEND=fa2`。日志位置由终端打印的 `LOG_FILE` 指示。

## 兼容边界

- 当前路径要求 CP=1、packed THD、单 model chunk、因果注意力及 per-token loss。
- 示例使用 `staggered`、dropout=0、关闭 RoPE fusion；多依赖段 `lockstep` RoPE 不支持。
- `KEEP_LAST_K=0` 保留全部激活；正数启用较早 chunk 的 Fp/Fr 重计算。
- GDN 需要支持初始/末状态梯度的算子及 Megatron `_forward_compute` 接口；
  关闭预处理融合，不支持 deterministic 模式或 full-layer recompute。
- MoE 辅助损失、容量限制及丢弃 token 的配置需单独验证，不能假设切块前后等价。
- PP=1 暂不更新上游 PackingMetrics，性能对比应使用实际 step 耗时。
- loss、耗时和显存以 GPU 对照为准；诊断使用框架原有 profiler。

## 测试

```bash
python3 -m pytest -q tests/cpu/backends/megatron/test_chunkflow_*.py
python3 -m pytest -q tests/backends/megatron/test_chunkflow_cuda.py
```

CPU 测试覆盖切分、状态和梯度；CUDA、Megatron、FA/FLA 相关测试需要对应环境。
