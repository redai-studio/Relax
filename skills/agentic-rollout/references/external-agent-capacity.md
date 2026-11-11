# Session and external agent capacity

Calculate logical and physical train limits when selecting Session capacity. Check the external quota as well when
agents submit work to a capacity-limited service; direct local execution still needs a host resource budget.

## Resolve effective values

Read the final recipe and parsed values:

```text
C = agentic_concurrency -> over_sampling_batch_size -> rollout_batch_size
N = n_samples_per_prompt
H = agentic_session_capacity_per_node, or unset
M = distinct hosts actually running training SessionShards
agent_resource_owner_path
agentic_eval_concurrency
n_samples_per_eval_prompt for each Eval dataset
agentic_prelaunch, partial_rollout, fully_async
Eval enabled/disabled and shared/dedicated executor pools
```

`C` counts logical Groups across Prepare and Runtime. `H` counts physical train Session slots reserved or started and
not yet confirmed released on one host. Shards on the same host share `H` in this pipeline. It is not per GPU, per Shard,
a decode-request limit, a KV-token budget, or a node-wide semaphore shared by independent jobs.

Omitting `H` disables member scheduling; `0` is invalid and `H >= N` is required. Capacity is not inferred from hardware
or discovered from the remote service. Recipe defaults may select it from a known executor limit. Validate host CPU,
memory, container/process costs, and the owner's release contract before choosing a value.

With `C=32, N=8, H=128` on one host, the logical ceiling is 256 members and the physical ceiling is 128 slots. Confirmed
members return slots immediately, a new Group can start only its admitted members, and leased ready members generate
without waiting for all first requests. Reward behavior is unchanged; filtering, export, and replenishment retain
complete Groups.

For a same-capacity ordinary baseline on that host, omit `H` and normally use `C=16` with the same 128-slot external
pool. Keep other settings fixed and disclose the Group-concurrency change. Do not claim that a larger `H` fills GPU
memory or guarantees a throughput improvement.

## Calculate train and Eval limits

```text
L = C * N
T = L if H is unset, otherwise min(L, M * H)
G_d = dataset_d.n_samples_per_eval_prompt
C_d = explicit agentic_eval_concurrency or ceil(L / G_d)
E_d = C_d * G_d
E_peak = max(E_d across Eval datasets)
```

`T` is the physical train upper bound, not a prediction of exact occupancy or balanced placement. Check local quotas
per host as well as any shared platform limit. Each complete Group remains on one Shard/host.

Eval is not governed by `H`; its default still uses `L`, so it can exceed the capped training occupancy. Ordinary RM
uses singleton Runtime Groups and Group RM keeps multi-Session Groups; both retain the same `E_d` Session ceiling.
Eval datasets run serially.

For a shared train/Eval executor, use the capacity available to this job after other users/jobs:

| Eval enabled | Train Sessions may remain during Eval | Required slots |
| --- | --- | --- |
| No | — | `external_slots >= T` |
| Yes | No | `external_slots >= max(T, E_peak)` |
| Yes | Yes | `external_slots >= T + E_peak` |

Prelaunch, partial rollout, or fully async can retain training resources during Eval. Prelaunch uses the same train
budgets; it creates overlap, not another allowance. Dedicated pools require `train_slots >= T` and
`eval_slots >= E_peak`. An explicit smaller Eval concurrency may be needed even with training capacity enabled.

## Find the external limit and release proof

Inspect the platform executor, sandbox pool, server settings, and deployment quota. Do not infer slots from GPU count,
Ray concurrency, or a successful small run. Verify whether all Shard hosts submit to the same pool and whether the
adapter covers remote resources with `ReceiptResourceOwner` or a custom owner. See
[integration-contract.md](integration-contract.md#resource-ownership).

With ordinary Group scheduling, insufficient slots can be split across incomplete Groups so none passes the
all-member first-request barrier. Group-RM Eval still has this risk; singleton Eval Groups primarily lose throughput.
Capacity-managed train generation removes that barrier for ready members, but does not create remote slots, remove
external dependency cycles, or authorize early refunds when cleanup is unconfirmed.

## Verdict and evidence

| Result | Condition |
| --- | --- |
| `PASS` | Known available quotas satisfy the relevant bounds and the owner confirms physical release |
| `UNSAFE` | Known quotas are insufficient, or the owner does not cover the resources being counted |
| `UNVERIFIED` | Placement, external quota, or release behavior is unknown |

Report `C`, `N`, `H`, `M`, `L`, `T`, each `E_d`, overlap, configured quotas, and the owner contract. Keep logical Group
metrics separate from `session_capacity/outstanding`, `session_capacity/peak`, and
`session_capacity/confirmed_releases`. Outstanding slots may be valid across steps; final cleanup must retire them.

Source anchors:

- `relax/utils/arguments.py`: concurrency defaults and capacity validation
- `relax/agentic/pipeline/session_capacity.py`: host ledger and identity-bound confirmations
- `relax/agentic/pipeline/runtime.py::agentic_eval_concurrency_from_args`: Eval still derives from `C * N`
- `relax/agentic/rollout.py::_agentic_eval_runtime_concurrency`: ordinary/Group RM Eval
