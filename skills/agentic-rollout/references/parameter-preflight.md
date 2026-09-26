# Agentic parameter preflight

Read the final parsed arguments after Relax applies defaults and validation. Report every applicable row as `PASS`,
`NEEDS_CHANGES`, or `UNVERIFIED`, including when a bad configuration is detected.

## Agent application and response

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--use-agentic-rollout` | Selects Agentic train/Eval functions; custom generate is ignored | Startup normalization, without a conflict error |
| `--agent-command`, `--agent-cwd` | Command runs through `/bin/bash -lc`; cwd and dependencies must exist on every possible SessionShard node | Basic values at startup; node availability at process launch |
| `--agent-env` | `KEY=VALUE`; `RELAX_*` is reserved; duplicate keys resolve to the last value | Startup validation; effective login-shell environment needs runtime evidence |
| `--agent-timeout` | Positive Runtime active-process safety budget for agent-side hangs; separate from each Chat request's wall-clock timeout | Positivity at startup; adequacy only during execution |
| reasoning/tool parsers | Match the exact model/template and installed SGLang | First matching model response, not argument parsing |

## Resident capacity

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--agentic-concurrency` and `--n-samples-per-prompt` | Resolve resident train Groups and resulting Session/process count | Positive values at startup; capacity at runtime |
| `--agentic-eval-concurrency` and per-dataset `--n-samples-per-eval-prompt` | Resolve each dataset and its peak Session count | Positive values at startup; capacity at runtime |
| `--sglang-server-concurrency` | Distinguish fleet generation permits from resident Sessions | Runtime queue/throughput evidence |

Use [external-agent-capacity.md](external-agent-capacity.md) only for a capacity-limited external platform. If
`--agentic-prelaunch`, `--partial-rollout`, or `--fully-async` is enabled, read
[partial-and-async-lifecycle.md](partial-and-async-lifecycle.md). Treat partial rollout and fully async as mutually
exclusive. Otherwise skip those rules. Read
[large-scale-rollout.md](large-scale-rollout.md) only when fixed defaults may be stressed.

## Optional KV lifecycle and admission

If neither `--agentic-session-lifecycle` nor `--agentic-program-admission` is enabled, report this section `N/A` and
stop here.

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

Read this section only when the agent uses explicit export, nonlinear history, multiple contexts, or custom credit.

| Parameters | Required check | Detection |
| --- | --- | --- |
| `--agentic-custom-advantage-path` | Required for multiple exports; ordinarily avoid custom RM | Export fanout is not validated at startup; failures or silent ownership changes appear during rollout/training |
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
