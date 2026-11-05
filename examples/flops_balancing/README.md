# FLOPs-aware SFT scheduling

Both training paths are wired locally: `--flops-microbatch-packing` changes
microbatch construction; `--flops-dp-balancing` changes TransferQueue placement.
Both default to false. **Target Qwen3.5 multi-GPU correctness and real throughput
remain unverified.**

See the [reproduction guide](GPU_RUNBOOK.md) for pinned assets, commands and
reporting rules.

## Supported scope

- Megatron, causal-LM SFT, text-only, dynamic THD packing.
- Qwen3.5 Dense/MoE text branches, with coefficients frozen once from HF config.
- Ordinary iterators and the existing restricted SFT async-prepack path.
- Static TP/PP/CP; microbatch counts follow existing DP/VPP agreement.
- No dynamic CP, streaming fully-async sampler, MTP, expert token dropping,
  router expert-bias updates or nonzero MoE auxiliary/z losses.
- Wholepack is a separate scheduling policy and cannot be combined with either
  FLOPs flag. FLOPs microbatch packing replaces `balanced`; DP-only placement
  may also use the existing `binpack` microbatch strategy.

`SequenceFlopsEstimator.from_config(hf_config).estimate_per_sample(lengths)`
returns raw integer work `a*L + b*L**2`, without time or hardware queries. It
reuses existing accounting approximations. Visual work, padding computation,
recomputation, optimizer time, communication and routing imbalance are unmodeled.

`pack_by_workload(capacity_lengths, costs, max_capacity)` uses first-fit
decreasing as a feasible fallback, then tries workload-first placement. Every
real sample appears once and each group satisfies additive capacity. The result
is heuristic, not optimal. A count below FFD may fail even when a solution exists.
More groups than samples are rejected at this low level; the training planner
splits real groups and adds marked zero-loss dummy execution slots. It accounts
for CP alignment and final TP padding. A token bound does not guarantee GPU memory.

The DP sampler delegates through a private FLOPs metadata view. Complete prompt
groups, equal group counts, chosen global batch, replay/polling caches and
checkpoints are preserved. Real lengths and payloads are unchanged.

## Reference and adaptations

[Slime PR #2017](https://github.com/THUDM/slime/pull/2017) substitutes per-sequence
work for lengths in KK and balances microbatch work across DP. Its illustrative
`coeff*L + L**2` has the same relative weights as our formula when `b > 0` and
`coeff = a/b`. Relax deliberately differs in these ways:

1. Cache raw integer coefficients; handle `b == 0` without division.
2. Retain TransferQueue's existing group/global-batch/replay protocol, placing
   samples across DP before local packing.
3. Constrain final capacity: the reference's `ceil(sum(lengths)/budget)` is a
   lower bound. `[6,6,6]` at budget `10` needs three bins, though that bound is two.
4. Preserve optimizer-step boundaries and actual VPP grouping. Expand validated
   plans by splitting; clearly mark repeated dummy execution slots.
5. Reject auxiliary-loss modes because masking main loss does not necessarily
   mask independently injected auxiliary gradients.

These adaptations have CPU regression coverage; native parallel validation is pending.

## CPU regression checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest \
  tests/cpu/utils/data/test_workload_balancing.py \
  tests/cpu/utils/data/test_flops_schedule.py \
  tests/cpu/utils/data/test_flops_sampler.py \
  tests/cpu/backends/megatron/test_flops_iterator.py \
  tests/cpu/utils/test_flops_counter.py \
  tests/cpu/utils/test_optimizer_step_capture.py \
  tests/cpu/examples/test_flops_validation_tools.py
```

These cover packing and sample boundaries, real TransferQueue replay, production
iterator logic with CPU metadata, actual FP32 AdamW deltas and report rejection
paths. They do not prove Qwen3.5/TP/PP/CP/EP correctness. The opt-in numerical
exporter intentionally synchronizes/writes large files and must be disabled for
timing. Chunked optimizer-state offload and FSDP capture are unsupported.
