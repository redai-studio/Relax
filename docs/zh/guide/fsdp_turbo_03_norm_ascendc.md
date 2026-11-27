# 普通/分组RMSNorm与官方QHC后端

提供独立RMSNorm及官方QHC替代实现，复用现有模型权重。

基于FSDPTurbo后端的Qwen3.8配方，在现有turbo YAML中添加以下选项；未配置时保持主仓默认行为。

```yaml
relax_performance:
  rmsnorm_implementation: triton
  grouped_rmsnorm_implementation: native
  qhc_implementation: ascendc
```

官方QHC需cann_ops_transformer_qhcofficial；与01组合须合并qhc枚举并保留互斥选择，不允许两个installer覆盖同一模块。

启用后需验证loss/梯度、吞吐和HBM/allocator峰值。历史9.13秒不是本PR单独收益；本补丁尚未经过NPU验证。
