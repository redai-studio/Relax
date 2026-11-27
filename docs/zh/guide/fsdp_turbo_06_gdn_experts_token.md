# GDN/conv、MegaMoE与token分块

合并GDN/causal conv后端、grouped QK、MegaMoE、token分块及SFT microbatch上限；包含Turbo配套补丁。

基于FSDPTurbo后端的Qwen3.8配方，在现有turbo YAML中添加以下选项；未配置时保持主仓默认行为。

```yaml
relax_performance:
  gdn_implementation: ascendc
  causal_conv1d_implementation: ascendc
  gdn_grouped_qk: true
  skip_gdn_recompute: false
  moe_implementation: megamoe
  token_chunk_size: 4096
```

仅SFT/TP1/Ulysses；保留GDN默认重计算策略，显式配置按算子能力校验。MegaMoE需hyper_multicore。与QHC组合先安装QHC再包token；Turbo补丁需与PR05合并。

启用后需验证loss/梯度、吞吐和HBM/allocator峰值。历史9.13秒不是本PR单独收益；本补丁尚未经过NPU验证。
