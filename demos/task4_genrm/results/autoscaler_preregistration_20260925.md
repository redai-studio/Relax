# Autoscaler round preregistration — 2026-09-25

Written and committed **before** the runs. No threshold, cooldown or
assertion may change after a round starts; a failed round is recorded as a
failure, not re-tuned. Decision times are taken from `/scale_history`
(`triggered_at`, unix epoch), never from poll-discovery times; phase
boundaries come from run events on the same clock.

## Frozen configuration

Deployment: single GenRM instance (Qwen3-0.6B, 1 engine initial), one
AutoscalerService with a `genrm` service target. Identical numbers to the
v3 working configuration — frozen, including the runtime cooldown PATCH
(`scale_out_cooldown_secs=20`, `scale_in_cooldown_secs=60`; per-service
cooldowns are not yet part of `ServiceScalingPolicy`):

| Setting                                | Value                                                                                                                                                                                                                                                                                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| min / max engines (genrm)              | 1 / 2                                                                                                                                                                                                                                                                                                                                     |
| metrics_interval / evaluation_interval | 3 s / 5 s                                                                                                                                                                                                                                                                                                                                 |
| condition_window                       | 30 s                                                                                                                                                                                                                                                                                                                                      |
| genrm scale-out policy                 | token_usage > 0.3, queue_depth > 4/engine, queue_time_p95 > 3 s, ttft_p95 > 3 s, window 15 s, max_delta 1                                                                                                                                                                                                                                 |
| genrm scale-in policy                  | token_usage \< 0.05, queue = 0, throughput_variance \< 1.0, window 45 s, max_delta 1, projected_usage_max 0.9                                                                                                                                                                                                                             |
| load (Round A)                         | 48 workers; LOW = 16-token replies (20 s); HIGH = ~1.8k-token prompts, 512-token replies until scale-out observed (300 s wait cap); STEADY = 256-token replies (60 s sampling, then screenshot capture ≤360 s, still under STEADY load); LOW′ = 16-token replies until scale-in observed (500 s wait cap, then ≤90 s history-record poll) |
| load (Round B)                         | none (true idle; no load generator, no timeline — scale evidence comes from `/scale_history`)                                                                                                                                                                                                                                             |

## Round A — with-traffic cycle (preregistered assertions)

| #   | Assertion                                                                                                                                             |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | No scale-out decided during the LOW phase (decision ts, not poll ts)                                                                                  |
| A2  | Scale-out decided while HIGH is active (decision ts ∈ \[high_start, high_end\])                                                                       |
| A3  | The elastic engine serves > 0 requests during STEADY                                                                                                  |
| A4  | Scale-in decided outside HIGH (decision ts ∈ \[steady_start, run_end\])                                                                               |
| A5  | Scale-in COMPLETED; final capacity == initial; initial engine survived                                                                                |
| A6  | Zero failed load requests across the whole round                                                                                                      |
| A7  | Resources returned: Ray free GPUs == baseline; per-GPU memory within 500 MiB of the pre-scale-out snapshot; placement-group table count == baseline   |
| A8  | Both TUI screenshots (`--service genrm` and `--service rollout`) captured live during STEADY and non-empty on disk — a failed capture fails the round |

## Round B — true-idle scale-in (preregistered assertions)

| #   | Assertion                                                                                                                                                                                                                                                      |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B1  | Manual scale-out to 2 with zero traffic reaches ACTIVE                                                                                                                                                                                                         |
| B2  | The autoscaler decides scale-in on valid idle evidence (queue/running observed 0, token_usage 0 \< 0.05, throughput stable) within 300 s of the manual scale-out completing, and the decision snapshot records all three conditions with zero running requests |
| B3  | Scale-in COMPLETED; final capacity == 1; initial engine survived; A7-style resource assertions hold                                                                                                                                                            |

## Stopping rules

Per-phase caps are the stopping rules: HIGH scale-out wait ≤ 300 s; STEADY
sampling exactly 60 s (screenshots after sampling, still under STEADY load);
LOW′ scale-in wait ≤ 500 s followed by a ≤ 90 s poll for the scale-in history
record; Round B scale-in decision within 300 s of the manual scale-out
reaching ACTIVE. A timeout is a functional failure of the round; no mid-run
tuning, no selective re-runs. The Round B manual scale-out reaching FAILED or
PARTIAL aborts the round as a functional failure immediately. Round B's
scale-in record is identified by `triggered_at` later than the Round B
scale-out ACTIVE moment (Round A's own scale-in is already in history).
Cleanup (serve apps, driver PG, ray shutdown) runs in an outer finally with
functional/cleanup verdicts reported separately.
