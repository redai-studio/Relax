# Speculative Decoding Metrics

Enable `--sglang-speculative-algorithm` to log speculative decoding metrics. Each rollout metric batch counts the committed generations covered by its exported samples, deduplicates shared generations, and sums counters before computing ratios.

## Logged Metrics

The existing logging entry reports these keys under `rollout/` for training and `eval/<dataset>/` for evaluation. When speculative decoding is disabled, these metrics are omitted.

| Metric | Meaning |
|---|---|
| `spec_accept_rate` | Sum of accepted draft tokens divided by sum of proposed draft tokens, using generations with both counters available. |
| `spec_accept_length` | Sum of completion tokens divided by sum of verify steps, using generations with both counters available and a positive verify count. |
| `spec_nodes_total` | Number of counted generations after deduplication. A sample without generation identities contributes one record. |
| `spec_nodes_missing_counts` | Counted records missing at least one of the four counters. |
| `spec_coverage` | Fraction of counted records with all four counters available. |
| `spec_accept_rate_coverage` | Fraction with both accepted and proposed counters available. |
| `spec_accept_length_coverage` | Fraction with both completion and verify counters available. |
| `spec_legacy_samples` | Number of legacy samples without recoverable generation identities; omitted when zero. |

Coverage uses the same deduplicated record count as `spec_nodes_total`. Explicit zeros count as available fields. A ratio is omitted when its paired denominator sums to zero. An explicitly reported zero numerator with a positive denominator produces a valid zero ratio. Missing, `null`, negative, non-integer, or otherwise invalid counters remain unknown; they never become fabricated zero numerators. An empty batch returns no metrics.

For a partial report, one ratio can remain valid while the other is omitted. For example, accepted/proposed = 1/2 with verify = 1 but no completion count produces `spec_accept_rate=0.5`, omits `spec_accept_length`, and reports coverage values of 0, 1, and 0 for complete counters, acceptance, and length respectively.

## Generation and Resume Accounting

A generation is a request committed to a Session, identified by `req_<session_id>_<sequence>`. Exported samples retain sparse per-request counter records on their lineage. The batch counts each identity once: `A -> B` and `A -> C` share A, while independent requests that generate identical text remain separate. Different Sessions also remain separate. Committed sibling branches outside the exported lineages are excluded.

An interrupted request keeps its identity when resumed. A counter is a known request total only if every backend attempt reported that field. Its values are then summed across attempts. A missing field in any attempt remains unknown, regardless of attempt order. For example, an attempt with completion = 5 followed by a complete report with completion = 2 preserves completion = 7, but cannot establish complete accepted, proposed, or verify totals. It therefore emits neither ratio. This avoids reporting partial work as a fully covered generation.

Repeated exported copies can supply missing fields for the same committed generation; an already recorded field is never added again. Aggregation does not modify the samples.

## Compatibility

`spec_accept_rate` and `spec_accept_length` retain their names and units, but change from arithmetic means of per-sample ratios to ratios of summed, deduplicated counters. A small sample no longer has the same weight as a large sample. Historical curves from the two accounting methods should not be directly combined. The coverage and record-count metrics are new.

The four integer totals in `Sample.SpecInfo` remain available to existing callers. New serialization also preserves generation records and `available_counters`, so a missing field remains unknown through JSON round trips even though the compatibility integer defaults to zero.

Old payloads without generation identities are counted as whole samples. Shared Agentic work in those payloads cannot be deduplicated, and `spec_legacy_samples` makes that limitation visible. Only fields actually present with valid values are available. Positive proposed or verify counts provide evidence of a speculative report; all-zero or completion-only legacy data cannot distinguish a missing report from explicit zeros and remains unknown. Partial legacy payloads can still contribute to a ratio whose two fields are known. New and legacy samples can coexist in one batch.

## Human-Checkable Examples

Two independent generations with accepted/proposed = 1/2 and 9/10 produce:

```text
accepted = 1 + 9 = 10
proposed = 2 + 10 = 12
spec_accept_rate = 10 / 12 ≈ 0.8333
previous sample-average result = (1/2 + 9/10) / 2 = 0.7
```

For the exported trajectories `A -> B` and `A -> C`:

| Generation | Accepted | Proposed | Verify | Completion |
|---|---|---|---|---|
| A (shared) | 4 | 8 | 2 | 6 |
| B | 1 | 2 | 1 | 2 |
| C | 1 | 1 | 1 | 1 |
| Sum, A counted once | 6 | 11 | 4 | 9 |

The result is `spec_accept_rate=6/11≈0.5455`, `spec_accept_length=9/4=2.25`, `spec_nodes_total=3`, `spec_nodes_missing_counts=0`, and all three coverage metrics equal 1. The previous implementation averaged the two sample ratios, `5/10` and `5/9`, and reported approximately 0.5278.

`tests/utils/test_spec_decoding_metrics.py` asserts these examples and boundary behavior. CPU tests in `tests/test_agentic_rollout.py` exercise backend metadata processing, request commit, trajectory export, `TrainingFieldArtifact` serialization, and batch aggregation, including partial resumed requests and unexported sibling branches.

## Related Guides

- [MTP Training](./mtp-rl-training.md)
- [Agentic Rollout](./agentic-rollout.md)
