# 权重版本与 FULL checkpoint

训练端可以将一次训练更新导出为独立的完整权重包。包内使用 Hugging Face 的逻辑张量名、dtype 和 shape；消费端从共享目录重建 safetensors checkpoint，再交给 SGLang 的现有 Loader。

## 训练端导出

在训练模型仍处于运行态时，将该调用作为训练 step 的最后一个分布式操作：optimizer step 完成后、立即 sleep/offload 前，对训练组的每个 `MegatronTrainRayActor` 调用 `export_weight_checkpoint`。该方法不会自动唤醒已 offload 的模型。调用期间保持训练权重稳定；方法同步完成导出，并在各 rank 返回相同的版本回执。

该 FULL 导出 API 暂不支持 LoRA adapter 模式。即使只有一个 rank 请求该模式，全部训练 rank 也会在转换或发布前一致拒绝调用。adapter 模式训练继续使用已有的适配器同步路径；本 API 不导出独立 adapter artifact。首个 PR 的验收覆盖非 LoRA 模型。

```python
receipt = actor.export_weight_checkpoint(
    store_path="/shared/weights",
    stream="training-run",
    version=0,
    update_id="optimizer-step-0",
    scratch="/local/weight-export",
    disk_budget=200 * 1024**3,
    max_weight_bytes=32 * 1024**3,
    metrics_dir="/local/weight-metrics",
)
```

以上容量是配置示例，按实际模型与保留版本数量设置。`version` 标识该 stream 中的权重版本，`update_id` 对应调用方的训练更新。回执包含逻辑内容摘要 `root` 和交付包标识 `artifact_id`。

导出沿用 Relax 的 Direct／Bridge 转换器：源端先在训练并行组内汇聚参数，转为逻辑张量，由 rank 0 保存并提交 FULL 包。写入期间各 rank 协调进度；文件错误会返回给全部参与者，同一目标可以重试。训练进程退出后，消费端仍可以读取已提交的包。

## 消费端重建与加载

共享目录挂载到消费端后，执行：

```bash
python -m relax.distributed.weight_sync.checkpoint   --store /shared/weights --stream training-run --version 0   --output /local/checkpoint-0   --disk-budget 214748364800 --max-weight-bytes 34359738368

python -m sglang.launch_server   --model-path /local/checkpoint-0   --tokenizer-path /models/original-model
```

重建输出包括 `model.safetensors`、`config.json` 和 `weight-version.json`。权重与模型配置随版本保存，tokenizer 和 processor 使用原模型目录。SGLang 根据目标端的并行配置加载完整 checkpoint。

## 格式与计量

版本清单绑定张量目录、块内容和模型配置。存储层提供 FULL／差量块表示、不可变版本、保留引用与完整重建；训练端的 `export_weight_checkpoint` 每次生成 FULL 包。

每个 rank 的计量文件记录源端转换时间、应用层发送字节、文件写入和进程内存。集合通信的发送字节按发送方计量，文件写入单列；网络接口和文件系统观测保留各自的统计范围。

## 验证

CPU 测试覆盖位模式重建、版本身份、checkpoint 格式、容量竞争、并行文件操作，以及两个训练参与者在写入失败后的共同返回与重试。

```bash
python -m pytest -q tests/cpu/distributed/weight_sync

torchrun --standalone --nproc-per-node=1   tests/distributed/weight_sync/run_full_export.py --root /tmp/full-direct
torchrun --standalone --nproc-per-node=1   tests/distributed/weight_sync/run_full_export.py --root /tmp/full-bridge --bridge
```

两个 GPU 驱动使用小型 Qwen3：执行一次初始化和五次真实 Megatron-Core optimizer 更新，每次导出完整权重，再由 Transformers 加载并比较前向结果。

`tests/distributed/weight_sync/run_full_checkpoint_loader.py` 在独立进程中重建指定版本，通过 SGLang Loader 启动并执行生成；可用 `--probe` 提供文本或图文请求。
