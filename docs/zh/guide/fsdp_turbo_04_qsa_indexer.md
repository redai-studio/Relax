# QSA LightningIndexer与掩码校验

索引器返回mask/token IDs/block IDs，包含前缀mask校验和完整第三方许可证。

基于FSDPTurbo后端的Qwen3.8配方，在现有turbo YAML中添加以下选项；未配置时保持主仓默认行为。

```yaml
relax_performance:
  qsa_indexer_implementation: ascendc
```

独立支持CP1与eager id2mask；CP模式依赖05 attention适配，未合入则提前报错。需要cann_ops_transformer。

启用后需验证loss/梯度、吞吐和HBM/allocator峰值。历史9.13秒不是本PR单独收益；本补丁尚未经过NPU验证。
