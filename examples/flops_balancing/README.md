# FLOPs-aware SFT scheduling

Both training paths are wired locally: `--flops-microbatch-packing` changes
microbatch construction; `--flops-dp-balancing` changes TransferQueue placement.
Both default to false. On 4×RTX 6000D PCIe, commit `a7eb426` improved
Qwen3-4B SFT effective supervised-token throughput by 11.39% on a three-length
workload and 21.01% on a preselected long/short workload, each over three
independent paired runs. The long/short case failed the frozen A/C gradient
and FP32-update tolerance despite identical samples and masks, so its 21.01%
is a throughput observation, not a validated extra-star result. Length-only
variable-row placement also gives the same assignment in that case, so it
does not isolate the quadratic FLOPs term.
Earlier 4×A800 NVLink results were 15.61% and 26.84%, respectively.
The first effective batch of the measured three-length workload passed
same-scene one-step full-gradient and actual FP32-update comparisons with the
same stabilized mathematics on both sides. A separate frozen one-step Dense
workload passed as well.
Default native BF16 did not pass the frozen actual-update tolerance and is
not claimed as validated.

See the [reproduction guide](GPU_RUNBOOK.md) for pinned assets, commands and
reporting rules.

## Supported scope

- Megatron, causal-LM SFT, text-only, dynamic THD packing.
- Qwen3 Dense, Qwen3.5 Dense/MoE and OLMoE text branches, with coefficients frozen once from HF config.
- Ordinary iterators and the existing restricted SFT async-prepack path.
- Static TP/PP/CP; microbatch counts follow existing DP/VPP agreement.
- No dynamic CP, streaming fully-async sampler, MTP, expert token dropping,
  router expert-bias updates, LoRA, language-model parameter freezing or nonzero
  MoE auxiliary/z losses.
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
An individual sample exceeding capacity fails FLOPs microbatch planning.
More groups than samples are rejected at this low level; the training planner
splits real groups and adds marked zero-loss dummy execution slots. It accounts
for CP alignment and final TP padding. A token bound does not guarantee GPU memory.

The DP sampler preserves the parent's exact selected global batch, complete
prompt groups, replay/polling caches and checkpoint contract. General parallel
layouts retain equal group counts. Restricted synchronous fixed-window SFT with
TP1/PP1/CP1/EP1/ETP1 permits variable row counts in the Relax sampler without a
TransferQueue patch. If KK exceeds the existing token budget, capacity-aware placement is
tried; infeasible windows retain KK and native multi-microbatch/dummy agreement.
Real lengths and payloads are unchanged.

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

These adaptations have CPU regression coverage and the scoped four-GPU evidence
above. Untested parallel layouts and the failed native BF16 profile are not
claimed as numerically accepted.

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
