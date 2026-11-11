# Agentic parameter preflight

Read the final parsed arguments after Relax applies defaults and validation. Report every applicable row as `PASS`,
`NEEDS_CHANGES`, or `UNVERIFIED`, including when a bad configuration is detected.

## Agent application and response

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--use-agentic-rollout` | Selects Agentic train/Eval functions; custom generate is ignored | Startup normalization, without a conflict error |
| `--agent-command`, `--agent-cwd` | Command runs through `/bin/bash -lc`; cwd and dependencies must exist on every possible SessionShard node | Basic values at startup; node availability at process launch |
| `--agent-env` | `KEY=VALUE`; `RELAX_*` is reserved; duplicate keys resolve to the last value | Startup validation; effective login-shell environment needs runtime evidence |
| `--agent-resource-owner-path` | Default process-group scope, or explicit external receipt/custom owner covering containers, remote jobs and detached children | Owner class is loaded before launch; physical cleanup needs runtime evidence |
| `--agent-timeout` | Positive Runtime active-process safety budget for agent-side hangs; separate from each Chat request's wall-clock timeout | Positivity at startup; adequacy only during execution |
| reasoning/tool parsers | Match the exact model/template and installed SGLang | First matching model response, not argument parsing |

## Resident capacity

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--agentic-concurrency` and `--n-samples-per-prompt` | Resolve logical train Groups and `L = C * N`; do not equate this with occupied physical slots | Positive values at startup; capacity at runtime |
| `--agentic-eval-concurrency` and per-dataset `--n-samples-per-eval-prompt` | Resolve each dataset and its peak Session count | Positive values at startup; capacity at runtime |
| `--agentic-session-capacity-per-node` | Unset disables member scheduling; `0` is invalid; `H >= N`; per-host train budget, not per GPU or Eval | Startup value check; actual Shard host identities, owner proof and host/executor headroom at runtime |
| `--sglang-server-concurrency` | Distinguish fleet generation permits from resident Sessions | Runtime queue/throughput evidence |

Use [external-agent-capacity.md](external-agent-capacity.md) when physical Session capacity or a capacity-limited
external platform is used. Eval defaults still derive from logical `C * N`, not the capped train occupancy. If
`--agentic-prelaunch`, `--partial-rollout`, or `--fully-async` is enabled, read
[partial-and-async-lifecycle.md](partial-and-async-lifecycle.md). Fully async has its own retention/backfill path; it
does not require `--partial-rollout`. Agentic dynamic GBS requires partial rollout, strict oversampling, and a non-fully-async
configuration. Keep it distinct from dynamic token packing. Read
[large-scale-rollout.md](large-scale-rollout.md) only when fixed defaults may be stressed.

## Optional KV lifecycle and admission

If neither `--agentic-session-lifecycle` nor `--agentic-program-admission` is enabled, report this section `N/A` and
skip its rows; continue with export checks when applicable.

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--agentic-session-lifecycle` and radix-cache dependencies | Installed SGLang support, session radix cache, priority eviction | Startup error |
| `--agentic-program-admission` | Enable only for measured KV-bound scheduling | Startup wiring; effectiveness at runtime |
| `--agentic-admission-headroom`, `--agentic-admission-pressure-threshold` | Values in `(0, 1]` | Startup validation when admission is enabled |
| `--agentic-admission-expected-decode-cap`, `--agentic-admission-max-wait-s`, `--agentic-admission-scope` | Positive decode cap, nonnegative wait, and intended train/all scope | Startup validation/defaulting when admission is enabled |
| Admission effectiveness | Router/engine discovery, pressure and queue behavior | Runtime `agentic_kv/*` metrics |

When lifecycle or admission is enabled, verify router/engine discovery and `agentic_kv/*` metrics. Rising
`agentic_kv/session/close_failure`, persistent `agentic_kv/admission/waiting`, frequent
`agentic_kv/admission/bypass_aged`, or `agentic_kv/budget/degraded == 1` prevents a `PASS` verdict. Protected work
bypasses admission.

## Conditional export fanout

Read this section only when the agent uses explicit export, nonlinear history, multiple contexts, or custom advantage.

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--agentic-custom-advantage-path` | Requires explicit export. Every export name needs a scalar or list; a list must match that export's assistant-turn count | Export names, fanout, and any list lengths are runtime evidence |
| Dynamic batch and token budget | Required for multiple physical rows per Session | Token-budget dependency is checked at startup only after dynamic batching is enabled; actual fanout is runtime evidence |
| Response length fields | `max_completion_tokens` takes precedence over legacy `max_tokens`; either overrides the turn default | Request validation |

Use [agentic-training-contract.md](agentic-training-contract.md) for identity, reward, Eval, and batching semantics.

## Report

```text
Agentic parameters: PASS | NEEDS_CHANGES | UNVERIFIED
Effective values:
Resolved defaults:
Dependency failures:
Unsafe interactions:
Missing evidence:
```

Source anchors: `relax/utils/arguments.py`, `relax/backends/sglang/arguments.py`,
`relax/agentic/runner/ipc.py`, `relax/agentic/rollout.py`, `relax/agentic/session/service.py`, and
`relax/agentic/session/admission.py`.
