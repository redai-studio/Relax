# Agentic speculative decoding metrics

Agentic rollout speculative decoding metrics are aggregated from the committed
generation requests covered by the samples exported in the current rollout
batch.

## Metric semantics

The existing metric names are retained, but their aggregation semantics change.

| Metric | Previous aggregation | Current aggregation |
| --- | --- | --- |
| `spec_accept_rate` | Mean of per-sample acceptance ratios | Sum of accepted draft tokens divided by sum of proposed draft tokens |
| `spec_accept_length` | Mean of per-sample completion/verify ratios | Sum of completion tokens divided by sum of verify calls |

Two coverage metrics are added:

- `spec_accept_rate_coverage`: fraction of accounting records that provide both
  accepted and proposed draft counters.
- `spec_accept_length_coverage`: fraction of accounting records that provide
  both verify and completion counters.

A zero counter is valid reported data. An absent counter is treated as missing.
If the aggregate denominator is zero, the corresponding ratio metric is omitted
rather than reported as a fabricated zero-percent value.

## Agentic generation identity and deduplication

Each committed generation is identified by `(session_id, request_id)`.

`resp_state_hash` identifies conversation state, not generation execution.
Independent requests may produce the same state and are still counted
independently.

When multiple exported trajectories share a committed generation, that
generation is counted once within the rollout metric batch.

Only committed generations whose response states occur in the exported sample
trajectories are included. Generations on branches not covered by the current
export are not counted.

## Exported metadata

New Agentic exports include sparse speculative accounting records under
`metadata.agentic_trace.spec_generations`.

Each record contains `request_id`, `resp_state_hash`, and any speculative
counters actually supplied by the backend.

Supported counter fields are:

- `spec_accept_token_num`
- `spec_draft_token_num`
- `spec_verify_ct`
- `completion_token_num`

Counter keys that were not supplied remain absent.

## Legacy compatibility

Samples serialized before generation-level accounting do not contain
`agentic_trace.spec_generations`. They continue to use `Sample.spec_info` as a
fallback.

Historical zero-valued `spec_info` cannot distinguish an explicitly reported
zero from a missing counter that was defaulted to zero. Therefore a legacy
counter pair is considered covered only when its denominator is positive. This
avoids inventing a zero-percent metric from information that was never
recorded.

## Manual verification example

Suppose the current export contains two trajectories:

    A -> B
    A -> C

The committed generation counters are:

| Generation | Accepted | Proposed | Verify | Completion |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 2 | 1 | 2 |
| B | 2 | 4 | 2 | 3 |
| C | 9 | 10 | 3 | 6 |

The exported samples contain `[A, B]` and `[A, C]`, but shared generation `A`
is counted once.

The unique totals are:

    accepted   = 1 + 2 + 9  = 12
    proposed   = 2 + 4 + 10 = 16
    verify     = 1 + 2 + 3  = 6
    completion = 2 + 3 + 6  = 11

Therefore:

    spec_accept_rate        = 12 / 16 = 75%
    spec_accept_length      = 11 / 6 = 1.833333...
    acceptance coverage     = 3 / 3 = 100%
    accept-length coverage  = 3 / 3 = 100%

For the acceptance criterion with two independent generations whose counters
are `1/2` and `9/10`, aggregation is:

    (1 + 9) / (2 + 10) = 10 / 12 = 83.33%
