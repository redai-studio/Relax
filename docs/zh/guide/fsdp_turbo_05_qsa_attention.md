# QSA NPU/grouped attention、id2mask及CP

Triton id2mask、NPU与custom grouped attention、Ulysses委派和序列长度传递；保留选择性重计算。

基于FSDPTurbo后端的Qwen3.8配方，在现有turbo YAML中添加以下选项；未配置时保持主仓默认行为。

```yaml
relax_performance:
  qsa_indexer_implementation: ascendc
  qsa_id2mask_implementation: triton
  attention_implementation: npu
```

AscendC/grouped模式依赖先合04；本patch不重复索引器源码。grouped另需qwen38_fused_train_ops。CP要求MTP=0；TP1/MBS1及几何约束仍保留。

启用后需验证loss/梯度、吞吐和HBM/allocator峰值。历史9.13秒不是本PR单独收益；本补丁尚未经过NPU验证。
