# Large-scale Agentic rollout

Load this reference if and only if projected or observed load may approach a fixed Relax width or exceed a known
validated scale. A production label, prelaunch, or the phrase "large-scale" is not enough.

First calculate train and per-dataset Eval Session ceilings with
[external-agent-capacity.md](external-agent-capacity.md). If the target and measured distribution leave clear headroom,
return `Internal scale: N/A` without naming internal constants.

## Establish the possible mismatch

One resident Session owns one agent process. Runtime assigns a complete Group to one SessionShard by Group ID
hash, so inspect the maximum observed load on any Shard instead of dividing the total by the Shard count.

Escalate this check only when at least one signal exists:

- projected per-Shard long-lived Chat calls approach the current Shard method width;
- requests queue before reaching their Session owner or backend permit;
- first-request progress, health, cleanup, or permit return stops advancing at target fanout; or
- the requested scale exceeds a measured known-good run without demonstrated headroom.

## Inspect only the implicated defaults

Read values from the exact checkout. Report only rows connected to the observed or projected mismatch.

| Symbol | Meaning and action |
| --- | --- |
| `_DEFAULT_SESSION_SHARD_COUNT` | Controls both SessionShard actor count and Serve ingress replica count. Every ingress replica can route to every Shard. Increase only when group-affine per-Shard load needs redistribution, then validate the changed deployment topology. |
| `_AGENTIC_CHAT_MAX_ONGOING_REQUESTS` | Per-ingress-replica ongoing request ceiling. Excess work queues because no separate Serve queue limit is configured. Increase only when ingress admission is the measured bottleneck. |
| `_AGENTIC_SHARD_MAX_CONCURRENCY` | Default Ray actor lane used by long Chat and ordinary Session control methods. Increase when this lane delays first requests or lifecycle progress on a loaded Shard. |
| `_SGLANG_PERMIT_CONCURRENCY` | Remote permit-acquire lane on the limiter hosted by Shard 0; Shard 0 local acquisition does not traverse this Ray lane. Increase when remote acquires queue before permit arbitration. |
| `_SGLANG_PERMIT_CONTROL_CONCURRENCY` | Separate lane for remote permit release plus health, debug, trim, and Agentic KV metric RPCs. It does not own SGLang `close_session`. Increase only when this lane delays those operations. |
| `_LAUNCHER_SERVER_BACKLOG` | Per-node launcher socket burst backlog, also subject to the host kernel limit. Inspect only when launch bursts produce socket accept failures. |
| admission coordinator module `_MAX_CONCURRENCY` | Optional admission actor method width. Inspect only when program admission is enabled and the coordinator itself is demonstrably saturated. |

The actual fleet generation-permit capacity is:

```text
sglang_server_concurrency * rollout_num_gpus // rollout_num_gpus_per_engine
```

Increasing Shard, Serve, or RPC widths cannot compensate for insufficient generation permits or external agent slots.
These fixed values have no public Agentic tuning flags; a required change is `NEEDS_CODE_CHANGE`, followed by
target-load validation.

## Validate the change

1. Record the exact revision, relevant current values, target Session counts, and maximum observed per-Shard load.
2. Prove one complete Group and its first-request barrier.
3. Reach the target Prepare/prelaunch fanout without stalled first-request progress.
4. Complete a target-size train rollout and optimizer step.
5. Exercise train/Eval overlap when configured.
6. Soak repeated pause/resume and drain cycles while checking only the implicated queues and lanes.

| Result | Required evidence |
| --- | --- |
| `PASS` | Target scale completes with measured headroom in every implicated width |
| `UNSAFE` | A named default is below measured demand or loses liveness |
| `UNVERIFIED` | The relevant target-load or per-Shard evidence is unavailable |

Source anchors: `relax/agentic/session/service.py`, `relax/agentic/pipeline/runtime.py`,
`relax/agentic/runner/ipc.py`, and `relax/agentic/session/admission_coordinator.py`.
