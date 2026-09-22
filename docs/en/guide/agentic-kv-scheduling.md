# Agentic KV Scheduling

Two independent, opt-in features that reduce KV-cache pressure during agentic rollout: **session KV lifecycle** releases a finished session's KV immediately, and **program-aware admission** bounds the active working set at request boundaries.

## Overview

In agentic rollout a session is a long-lived multi-turn program, not a single request. Between turns the agent executes tools — which can take seconds to minutes — while the KV prefix of that session stays resident in the engine. With many concurrent sessions the KV pool saturates, the engine falls back to eviction and recompute, and newly arriving requests queue behind work that is not actually decoding.

Relax addresses this from both ends of a session's life:

| Feature | Flag | Acts at | Effect |
|---|---|---|---|
| Session KV lifecycle | `--agentic-session-lifecycle` | End of a session | Releases the session's KV instead of waiting for priority-aware radix-cache eviction (LRU within equal priority) |
| Program-aware admission | `--agentic-program-admission` | Start of each request | Bounds how much KV the cluster commits to at once |

Both features are **off by default**, are **independent** (either can be enabled alone), and **fail open** — any missing, stale, or failing signal falls back to the existing behaviour. Neither changes generation results: the full replay payload (`input_ids`) is always sent, so a cold cache still serves correctly.

::: tip
These features target *scheduling* overhead, not model quality. Enable them when the rollout is KV-bound — high engine `token_usage`, frequent eviction, requests queueing while the GPU is not saturated.
:::

## Session KV Lifecycle

### What it does

When enabled, the SGLang backend adapter sends the Relax session ID as a top-level `session_id` field on every `generate` call. Concurrent requests and SessionForest branches from the same session share that ID, while each backend attempt keeps a unique request ID. When the session becomes terminal or is dropped, Relax aborts or joins its in-flight requests and then issues an idempotent `/close_session`, releasing that session's KV immediately. Partial rollout and fully async retention keep unfinished sessions alive across steps and therefore do not close their KV.

### Routing

`/close_session` is **not** proxied by the sgl-router. Relax therefore fans out directly to each engine base URL, mirroring how request aborts are handled. The engine's DP controller broadcasts the release across all DP ranks; ranks that do not hold the session no-op.

```
                    ┌──────────────────┐
   generate ───────►│   sgl-router     │───────► engine (placement decided here)
   (session_id)     └──────────────────┘
                    ┌──────────────────┐
   /close_session ─X│   sgl-router     │   not proxied
                    └──────────────────┘
                             │
   /close_session ───────────┴──────────────► every engine base URL directly
                                              (DP controller broadcasts to all DP ranks)
```

### Requirements

The supported SGLang target is 0.5.15.post1. The server must run with `--sglang-enable-session-radix-cache` and `--sglang-radix-eviction-policy priority`; Relax validates this combination at startup.

::: warning
`--sglang-enable-session-radix-cache` and `--sglang-radix-eviction-policy` are not defined by Relax. They are SGLang `ServerArgs` auto-exposed with a `--sglang-` prefix (see `relax/backends/sglang/arguments.py`), so they exist only if your installed SGLang provides them. Both are present in SGLang 0.5.15.post1.
:::

### Failure behaviour

Close is a KV-release optimisation only. Failures never block the logical terminal state; affected sessions remain subject to SGLang's configured priority-aware radix-cache eviction (LRU within equal priority). Monitor `agentic_kv/session/close_failure` for failed close fan-outs.

### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--agentic-session-lifecycle` | flag | `False` | Enable the feature |

## Program-Aware Admission

### What it does

At each request boundary — before taking the SGLang permit or calling `generate()` — admission uses one global FIFO queue and token ledger:

| Action | Behaviour |
|---|---|
| **Bypass** | Continue without a lease when work is protected, signals are degraded, or the maximum wait has elapsed. |
| **Admit** | Hold a cluster-wide execution-token lease until the backend attempt finishes. |
| **Wait** | Remain in FIFO order without taking a fleet request permit. |

Two invariants hold regardless of configuration:

- Admission **never selects a worker**. Final placement stays with the SGLang router.
- Admission **never interrupts in-flight decode**. It only gates work that has not started.

### Architecture

```
┌──────────────────────┐   acquire / release   ┌────────────────────────────────┐
│  AgenticSessionShard │ ────────────────────► │  AdmissionCoordinator          │
│  (16 Ray actors)     │ ◄──────────────────── │  single Ray actor, one writer  │
│                      │        lease          │  global FIFO + BudgetState     │
└──────────┬───────────┘                       └───────────────┬────────────────┘
           │                                                   │ poll /metrics
           │ fleet permit → generate                           ▼
           ▼                                   ┌────────────────────────────────┐
┌──────────────────────┐                       │        SGLang engines          │
│     sgl-router       │ ────────────────────► └────────────────────────────────┘
└──────────────────────┘
```

| Component | Responsibility | Implementation |
|---|---|---|
| **Decision logic** | Pure Admit/Bypass policy and the token ledger, no Ray or I/O | `relax/agentic/session/admission.py` |
| **Coordinator** | Single-writer Ray actor, global FIFO, `/metrics` poller, lease TTL reclaim | `relax/agentic/session/admission_coordinator.py` |
| **Shard integration** | Cancellation-safe lease acquisition before the fleet permit | `relax/agentic/session/service.py` |

`BudgetState` is separated from Ray, so capacity, pressure, lease, TTL, and usage accounting are deterministic and CPU-testable with an injected monotonic `now`. Global FIFO ordering, aging, cancellation, and metrics polling live in the Ray-backed `AdmissionCoordinator`; tests exercise its underlying class without starting Ray (`tests/test_agentic_rollout.py`).

### The reservation

A reservation is the **processor-expanded training prefix + remaining completion budget**. Text training and backend prefix lengths are naturally equal. Multimodal capacity accounting uses the processor-expanded training length while the SGLang payload continues to use backend tokenizer IDs and media data. `--agentic-admission-expected-decode-cap` can tighten the completion portion without changing the actual generation limit.

### Decision order

Admission follows this order:

1. Feature disabled or scope not selected → existing path, without admission
2. Protected work → **Bypass** (`protected`)
3. Missing or stale capacity snapshot → **Bypass** (`degraded`)
4. An older waiter exists, or a capacity or pressure limit is reached → **Wait** in the global FIFO
5. Maximum wait reached → **Bypass** (`aged`)
6. FIFO empty and capacity available → **Admit** and hold a lease

The Coordinator queues work for either of these ledger conditions:

| Refusal reason | Ledger condition | Caller behaviour |
|---|---|---|
| `pressure_guard` | Worst-case engine `token_usage` reaches the pressure threshold | **Wait** |
| `capacity_exhausted` | `reserved + tokens` exceeds the admission ceiling | **Wait** |

The ceiling is `sum(max_total_num_tokens of healthy engines) × headroom`.

### Leases

Leases are idempotent per request ticket, so a retried acquire does not double-charge the budget. Cancellation atomically removes a waiter or releases a concurrently granted lease. A TTL reclaims leases stranded by a dead shard, and a worker-set change advances the ledger epoch.

### Anti-starvation

Requests wait oldest-first in one Coordinator queue. Lease release and periodic metric reconciliation advance that queue. If the ledger becomes degraded, queued requests bypass. After `--agentic-admission-max-wait-s`, the oldest request also bypasses, ensuring admission cannot block progress indefinitely.

### Requirements

The coordinator discovers engines through the SGLang router (`/workers`, falling back to `/list_workers`) and scrapes each engine's Prometheus `/metrics`. If the router address is unset or unreachable there are no snapshots, the ledger reports `degraded`, and every request bypasses.

::: tip
Because the coordinator reads engine gauges, TP replication matters. SGLang emits `sglang:max_total_num_tokens` and friends once per rank with an identical per-engine value, so Relax aggregates them by **max**, not sum. Summing would inflate capacity and deflate usage by the TP degree, making a saturated engine read as nearly idle.
:::

### Options

| Flag | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `--agentic-program-admission` | flag | `False` | — | Enable the feature |
| `--agentic-admission-headroom` | float | `0.90` | `(0, 1]` | Fraction of aggregate KV capacity usable as the ceiling |
| `--agentic-admission-pressure-threshold` | float | `0.92` | `(0, 1]` | Per-worker `token_usage` at/above which new requests wait |
| `--agentic-admission-expected-decode-cap` | int | `--rollout-max-response-len` | `> 0` | Upper bound on expected decode tokens per reservation |
| `--agentic-admission-max-wait-s` | float | `30.0` | `>= 0` | Max FIFO wait before aging bypass |
| `--agentic-admission-scope` | str | `train` | `train` \| `all` | Apply to train only, or train + eval |

The numeric constraints are enforced only when `--agentic-program-admission` is set.

## Quick Start

Add to an existing agentic rollout launch script:

```bash
AGENTIC_ARGS=(
   --use-agentic-rollout
   # ... existing agent flags ...

   # Session KV lifecycle: requires the server-side session radix cache
   --sglang-enable-session-radix-cache
   --sglang-radix-eviction-policy priority
   --agentic-session-lifecycle

   # Program-aware admission: defaults are a reasonable starting point
   --agentic-program-admission
   --agentic-admission-headroom 0.90
   --agentic-admission-pressure-threshold 0.92
)
```

A complete example lives in `examples/mini_swe_agent/run_mini_swe_agent.sh`.

## Metrics

Both features report once per rollout step, alongside the existing `rollout/` and `perf/` metrics. Every series shares the `agentic_kv/` prefix, so trackers that group by the first path segment — ClearML splits a key into `(title, series)` on its first `/` — render them as one panel instead of three.

| Metric | Meaning |
|---|---|
| `agentic_kv/session/lifecycle_enabled` | `1.0` when session lifecycle is on; absent otherwise |
| `agentic_kv/session/close` / `close_failure` | Session close attempts and failures |
| `agentic_kv/admission/admit` / `bypass` | Per-step admission outcomes; absent when no matching outcome occurred |
| `agentic_kv/admission/wait` / `waiting` / `cancelled` | Enqueued, currently waiting, and cancelled requests |
| `agentic_kv/admission/bypass_protected` / `bypass_degraded` / `bypass_aged` | Fail-open bypass reasons |
| `agentic_kv/admission/defer_rate` | `wait / (admit + wait + bypass)`; the proportion of requests placed in the FIFO wait queue by admission control |
| `agentic_kv/admission/degraded_rate` | `bypass_degraded / (admit + wait + bypass)`; the proportion of requests allowed to proceed directly when capacity signals are unavailable or stale |
| `agentic_kv/admission/wait_seconds_mean` | Mean queue time for requests granted after waiting |
| `agentic_kv/budget/ceiling` / `reserved` / `available_tokens` | Admission ceiling and token ledger state |
| `agentic_kv/budget/reserved_utilization` | Reserved tokens divided by the admission ceiling |
| `agentic_kv/budget/lease_count` / `lease_expired` | Current leases and leases reclaimed by TTL |
| `agentic_kv/budget/kv_token_usage_mean` / `kv_token_usage_max` | In-window mean and peak engine KV token usage |
| `agentic_kv/budget/epoch` / `degraded` | Worker-set generation and capacity-snapshot health |

::: warning
`agentic_kv/budget/kv_token_usage_*` are sampled over a running window that is drained once per step, because an instantaneous read at log time lands after the rollout has drained and understates the real peak. The engine-side release gains — pool size, forced evictions, freed tokens — are not in this table; read them from the engine's own Prometheus `/metrics`.
:::

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Everything bypasses; `agentic_kv/admission/bypass_degraded` keeps increasing | Ledger degraded | Check `agentic_kv/budget/degraded == 1.0`. The router address is unset or unreachable, or worker `/metrics` scraping failed. |
| `agentic_kv/admission/waiting` remains high | Capacity or pressure bound | Check reservation sizes, concurrency, headroom, and engine KV usage. |
| `agentic_kv/admission/bypass_aged` keeps increasing | Requests routinely hit the aging deadline | Loosen headroom, reduce concurrency, or reduce the expected decode cap. |
| `agentic_kv/budget/reserved_utilization` remains near `1.0` | Genuinely capacity-bound | Raise headroom or reduce concurrent sessions. |
| Session lifecycle enabled but KV never drops | Server-side cache not enabled | Confirm the engine was started with `--sglang-enable-session-radix-cache`. |
| `agentic_kv/session/close_failure` increases | Worker discovery or close fan-out failed | Check router discovery and direct `/close_session` connectivity. |

## Next Steps

- Read [Agentic Rollout](./agentic-rollout.md) for the session lifecycle these features hook into.
- Read [Performance Tuning](./performance-tuning.md) for the wider rollout throughput checklist.
- Read [OOM Troubleshooting](./oom-troubleshooting.md) when KV pressure turns into out-of-memory failures.
