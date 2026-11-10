# FLOPs-aware SFT scheduling

`--balance-by-flops` opts ordinary text causal-LM SFT into FLOPs-aware DP assignment and capacity-safe dynamic microbatch packing. It is disabled by default. It does not change the selected effective batch, optimizer, loss normalization, or the meaning of `--max-tokens-per-gpu`.

## Configuration

Use the Megatron backend, `--loss-type sft`, `--use-dynamic-batch-size`, and `--n-samples-per-prompt 1`. Select the ordinary SFT path explicitly; for example, `--sft-max-in-flight-steps 1` prevents the automatically selected lookahead prepack path. The new flag does not change that setting for you.

The official Docker image supplies the pinned TransferQueue `assignment_transform` API. An installation without this API fails explicitly. The adapter delegates sample selection, cache publication, replay and checkpoint recovery to the existing sampler. A FLOPs KK candidate is accepted only if it strictly lowers max/mean rank FLOPs and does not increase predicted DP MAX microbatch count; otherwise it preserves the token assignment. The guard uses the actual packing strategy and checks both possible raw/padded CP document layouts before worker model construction.

The initial model cost contract covers Qwen2/Qwen3 text Dense/MoE and Qwen3.5 text Dense/MoE (including their text configuration names). It reuses the existing FLOPs formulas. Other or malformed model configurations fail rather than using a fallback estimate. Qwen3.5 configuration files can contain a vision configuration, but image inputs are not supported by this scheduler.

MoE requires zero auxiliary-loss/z-loss coefficients and unlimited expert capacity (`moe_expert_capacity_factor=None`). Auxiliary loss and expert token dropping depend on microbatch composition; auxiliary autograd paths can also produce gradients for dummy batches. These configurations are rejected to preserve training semantics.

Microbatch-dependent Sinkhorn/quantile routing and nonzero expert-bias updates are also rejected. Ordinary top-k routing with zero auxiliary coefficients remains available; enabling the feature does not silently change router settings.

## Capacity and synchronization

FLOPs are the balancing objective; token lengths are the capacity constraint. The existing capacity-safe token scheduler constructs the feasible baseline at the original DP MAX/VPP-aligned count. FLOPs repair never increases this count. Existing oversized singleton behavior is retained and frozen during repair; such a sample may still exceed available GPU memory.

At fixed K, repair accepts only strictly improving single moves or pair swaps, ordered by maximum bin FLOPs, spread, then sum of squared FLOPs. Sources stay nonempty and both affected bins respect token capacity. The search examines two heaviest/two lightest bins, at most eight samples per bin, for at most 32 iterations. Stable indices break ties; an unsuccessful search returns the exact baseline.

Existing DP MAX, VPP grouping, physical CP padding and dummy construction remain in place. Real samples appear exactly once per optimizer step. Dummy copies and offsets remain those of the token baseline and are excluded from repair, retaining zero effective loss/token semantics.

## Scope and validation status

| Combination | Implementation boundary | Hardware validation |
| --- | --- | --- |
| Ordinary text SFT, dynamic batching, Dense/MoE | Included | Pending |
| DP assignment and local microbatch construction | Included | Multi-GPU pending |
| TP, static PP/CP, EP, VPP | Existing topology/count contracts preserved | Multi-GPU pending |
| RL, fully async, hybrid, wholepack, async prepack | Explicitly rejected with the flag | Not supported |
| Preference, multimodal, dynamic CP, routing replay, MTP, LoRA or custom language freezing | Explicitly rejected with the flag | Not supported |

CPU planning tests and mocked collective tests do not prove NCCL ordering, expert routing behavior, peak GPU memory, distributed numerical correctness, or training throughput. No GPU throughput improvement is claimed.

## Inspecting scheduling

With `LOG_LEVEL=DEBUG`, the existing logger reports DP before/after token and FLOPs totals, max/mean imbalance, microbatch capacity-token/FLOPs totals, synchronized counts, dummy counts, move/swap/iteration counts, DP candidate decisions and scheduling time. FLOPs summaries count real samples; dummy forward/backward overhead is identified separately by the dummy count. Timing is diagnostic only and never selects the plan.

A CPU-only comparison tool accepts a model `config.json` and a JSON list of token lengths from one fixed effective batch:

```bash
python scripts/tools/benchmark_flops_balancing.py \
  --model-config /path/to/config.json \
  --lengths /path/to/lengths.json \
  --dp-size 2 --token-capacity 8192 \
  --output /path/to/scheduling.json
```

This tool measures scheduling, not training throughput. Synthetic inputs are mechanism evidence only.

## Preparing a training A/B comparison

Keep the same checkpoint and initial optimizer state, seed, data order, effective batch, precision, topology, token cap, dropout policy and existing baseline length-balancing/dynamic-batch settings. The only scheduling difference should be the new flag. Keep both runs on the same ordinary data path; do not compare baseline prepack against an optimized run with prepack disabled.

First repeat the baseline to establish numerical variation and choose tolerances before inspecting optimized results. Compare sample IDs, effective loss-mask tokens, full-effective-batch loss, accumulated/reduced gradients and FP32 parameter deltas. Retain raw logs, exact revisions, commands and model/data identifiers. Record warmup separately and measure the complete training loop, including scheduling and data preparation. CPU workload balance is not a substitute for the required real Dense/MoE training A/B tests and at least 10% throughput improvement.
