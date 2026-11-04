# H200 frozen-condition cache results

Three paired repetitions per workload, in AB/BA/AB order. Each segment uses 16 warmup and 64 measured optimizer updates,
global batch 8, LoRA rank 32, BF16 computation, FP32 masters/reductions, and a 64 GiB CPU cache per model job.

| Workload                             | Paired throughput gains   | Median      |
| ------------------------------------ | ------------------------- | ----------- |
| SenseNova, repeated conditions       | +9.29%, +5.49%, +7.78%    | **+7.78%**  |
| Qwen-Image-Edit, repeated conditions | +14.43%, +15.23%, +14.54% | **+14.54%** |
| SenseNova, low reuse                 | -20.68%, -19.38%, -22.31% | **-20.68%** |

Both models' stored full comparisons report strict equality for loss, accumulated/reduced gradients and FP32 updates,
including cold/warm cache paths. Complete per-parameter reports are in `numerical/`.

Two independent world1 model lanes ran concurrently: SenseNova on GPU2/CPU40–59 and Qwen on GPU3/CPU60–79. A/B stayed
on the same GPU and CPU affinity within each workload; the lanes shared host memory bandwidth. These are per-model
throughput comparisons, not world2 distributed-training scaling results. Timing covers the native SFT training loop,
including data/conditions, forward, backward, reduction and optimizer updates; Ray/TQ and checkpoint I/O are outside it.

## Files

- `summary.json`: factual result index, every performance pair, and model/data references.
- `performance/*.json.gz`: all18 complete per-segment results, including all80 update times and low-reuse regressions.
- `logs/`: all28 original console logs and28 execution receipts, gzip-compressed; the two empty compare logs are retained.
- `numerical/sensenova.json.gz`, `numerical/qwen_edit.json.gz`: complete stored numerical comparison reports.
- `original-summary.json.gz`: complete original summary after private-string redaction.
- `provenance/`: model/data revisions and file identities, real input selection, environment, hardware and execution scope.
  `environment.json` is the installation/import preflight; actual GPU execution is recorded in the stage results.
- `history/pre-performance03-summary.json.gz`: the earlier deterministic campaign summary, including its performance failure.
- `MANIFEST.json`: original-source and public-copy SHA256/byte counts, plus redaction scope.

Private paths, usernames, hostnames and GPU UUIDs use consistent placeholders. Console logs retain all other text,
including numeric values and timestamps; receipt `log_sha256` fields refer to the original logs, while `MANIFEST.json`
hashes the public copies. Complete JSON copies retain every numeric
value, tensor parameter name, timestamp and SHA field; the manifest distinguishes these copies from derived indices.

## Inspect and recompute

No model, PyTorch or GPU is needed to inspect this evidence:

```bash
python3 examples/frozen_condition_cache/results/h200-20261103/recompute.py
python3 examples/frozen_condition_cache/results/h200-20261103/verify.py
```

The command recomputes all paired gains, mean/median/range/sample standard deviation, and cumulative observed payback
from the18 original segment records. It also displays the stored numerical verdicts. The numerical tensor comparisons
are reported in full, but this command does not regenerate them from raw tensors.

To unpack a compressed report without changing the original:

```bash
python3 -c 'import gzip, pathlib; p=pathlib.Path("numerical/sensenova.json.gz"); print(gzip.decompress(p.read_bytes()).decode())'
```

Run that unpack command from this directory. `MANIFEST.json` hashes refer to the published bytes, including gzip framing.
`verify.py --source /path/to/received` additionally checks the retained original hashes and numeric/string-copy invariants
against the unredacted packet, when that packet is available locally.

Raw trajectory tensors are absent from the lite packet; the full archive's reported SHA is recorded in `MANIFEST.json`,
and its download URL and contents remain to be independently verified.
