# 视觉分帧与FSDP组内负载均衡

CPU保留vision grid metadata，训练和NLL评估均传递；分帧执行并在FSDP组内均衡视觉计算。

基于FSDPTurbo后端的Qwen3.8配方，在现有turbo YAML中添加以下选项；未配置时保持主仓默认行为。

```yaml
relax_performance:
  vision_frame_chunk_size: 4096
  vision_load_balance: true
```

只支持SFT/TP1及完整CP组包含在FSDP组内的布局。图像/视频、空视觉rank、CP重复样本及前反向均需分布式验证；文本9.13秒不构成本PR性能证据。

启用后需验证loss/梯度、吞吐和HBM/allocator峰值。历史9.13秒不是本PR单独收益；本补丁尚未经过NPU验证。
