# FLOPs-aware scheduling

For text Qwen3.5 Dense and MoE training, FLOPs-aware scheduling weights samples by estimated training work instead of sequence length alone. It keeps the token capacity constraint separate from the cost estimate. The estimate describes theoretical model work, not measured GPU time.

The feature is opt-in. Add the following arguments to an existing Megatron training configuration with dynamic batching:

```bash
--use-dynamic-batch-size \
--flops-scheduling both
```

`--flops-scheduling` accepts four modes:

| Mode | Microbatch construction | DP assignment |
| --- | --- | --- |
| `off` (default) | Existing behavior | Existing behavior |
| `microbatch` | FLOPs-aware | Existing behavior |
| `dp` | Existing behavior | FLOPs-aware |
| `both` | FLOPs-aware | FLOPs-aware |

The default order is `--flops-scheduling-order dp-first`: assign rows to data-parallel ranks, then construct each rank's microbatches. For global packing before DP assignment, use:

```bash
--flops-scheduling both \
--flops-scheduling-order pack-first \
--polling-mode True \
--qkv-format thd
```

Pack-first selects a complete cohort through the existing sampler, fixes its microbatch membership, and distributes those microbatches across DP ranks. Training consumes that plan instead of independently repacking each local batch. Zero-contribution dummy microbatches align the execution counts. Queue consumption is distinct from completion of an optimizer update.

## Supported configuration

Enabled modes require synchronous, dynamic-batch, ordinary text causal-language-model training without MTP. Multimodal inputs, classification/preference heads, LoRA, custom parameter freezing, Wholepack, CPU prepack, fully asynchronous training and dynamic global batch sizes are rejected.

SFT DP assignment requires `n_samples_per_prompt=1`. RL DP assignment requires `--balance-data` and the existing Balanced identity sampler; it preserves complete logical identity admission even when an identity has multiple physical rows.

Pack-first additionally requires polling and THD packing. It rejects hybrid execution, dynamic CP, SFT data prefetch, debug-data bypasses and routing/indexer replay. SFT supports one optimizer mini per partition (`--num-steps-per-rollout 1`); Balanced RL retains explicit multiple-mini boundaries. Changing checkpoint/data identity or execution geometry invalidates plan replay. Do not reuse a saved plan with different rows or model configuration.

The SFT adapter requires the repository's pinned TransferQueue `assignment_transform` patch. Use the project's matching dependency build. Runtime restrictions are checked before publishing a plan; CPU tests alone do not establish GPU or distributed-topology support.

Choose a mode by comparing the same effective batch, checkpoint, optimizer state and training configuration. An estimated FLOPs balance does not guarantee throughput improvement. Measure the full training loop and validate loss, reduced gradients and FP32 parameter updates before making a performance claim.

See [SFT Training](./sft-training.md) for the training setup and [Performance Tuning](./performance-tuning.md) for measurement tools.
