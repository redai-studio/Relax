# Gated RMSNorm Triton前反向

保留原前反向kernel，按配置安装适配；完整保留第三方许可证。

基于FSDPTurbo后端的Qwen3.8配方，在现有turbo YAML中添加以下选项；未配置时保持主仓默认行为。

```yaml
relax_performance:
  rmsnormgated_implementation: triton
```

需要triton-ascend；默认native，不启用时不加载kernel。

启用后需验证loss/梯度、吞吐和HBM/allocator峰值。历史9.13秒不是本PR单独收益；本补丁尚未经过NPU验证。
