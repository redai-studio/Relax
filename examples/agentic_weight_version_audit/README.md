# Agentic async weight-version tracking & lag audit (No.15)

Side-channel audit for policy provenance on samples that actually enter training.

## What is recorded

- Each successfully committed generation call / agentic turn stores a `WeightVersionsPerCall`
  with absolute token spans (`version`, `abs_start`, `abs_end`) on `Sample.weight_versions`.
- Partial-rollout abort/resume keeps spans for accepted prefixes on the in-flight request;
  only `append_resp` commits them into the Forest / exported Sample.
- Cancelled or replaced prefixes that never commit do not enter training stats.

## Metric definitions (`rollout/policy_provenance/*`)

Staleness uses **published rollout weight-version sequence numbers** for the same policy
(the integer string returned by SGLang), not wall-clock time.

| Metric | Meaning |
| --- | --- |
| `avg_staleness` / `max_staleness` | `current - oldest` numeric version in each selected group |
| `avg_post_generation_staleness` / `max_post_generation_staleness` | `current - newest` (wait after generation finished) |
| `avg_generation_version_span` / `max_generation_version_span` | `newest - oldest` within the group (updates during generation / multi-turn) |
| `token_weighted_staleness` | token-weighted mean of `current - span.version` |
| `weight_version_sample_coverage` | fraction of selected samples with ≥1 numeric span |
| `weight_version_missing_rate` | `1 - coverage`; unknown non-numeric versions are counted separately |

Missing / non-numeric versions are **not** coerced to `0` or to the latest version.

Optional: set `args.current_weight_version` when logging so consume lag uses the real
published version. If unset, the logger falls back to the newest stamped version in the
selected batch (a lower bound on train-time lag).

## Reproducible example

```bash
python examples/agentic_weight_version_audit/reproduce_tool_wait_audit.py
```

The script builds a two-turn Session where the agent waits on a tool while two weight
updates land, then prints batch metrics and a Session/turn/segment diagnostic report.
