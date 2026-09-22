# Resident lifecycle

Use this lifecycle to reason about ownership, barriers, capacity, and cleanup.

## Ownership path

```text
DataSource GroupInput
-> Prepare owns an unleased Group
-> SessionShard owns every Session, process, SessionForest, and first request
-> all Sessions reach the first-request barrier
-> Runtime leases the Group and opens generation
-> each terminal Session cleans up and publishes its result
-> Runtime gathers the Group and releases resident capacity
-> Reward finalizes the complete Group
-> Transfer accepts the scored Group
```

Ordinary per-Session RM may overlap Runtime's Group gather; it does not delay publication of the Session result.

## Core invariants

| Boundary | Required invariant |
| --- | --- |
| Resident capacity | Prepare and Runtime share `agentic_concurrency` Group permits |
| First-request barrier | Every original Session is active and has at least one live IR |
| Runtime lease | A Group cannot lease before the all-Session barrier |
| Group completion | Runtime gathers every original Session result; a controlled Session drop drops the Group |
| Result publication | Backend attempts, process ownership, and Session cleanup finish before the result is published |
| Permit release | Completed, rejected, cancelled, and dropped Groups release their resident ownership exactly once |

## Prelaunch boundary

Prelaunch fills available resident capacity with next-step Groups before their Runtime lease. It starts every agent
process in those Groups and lets every Session reach its first request, while the requests remain held at the Prepare
gate. Warming and ready Groups can remain resident across the current rollout tail, the optimizer step, and Eval before
the next train lease.

Prelaunch changes timing and overlap rather than adding another train resident allowance: Prepare and Runtime still
share the same `agentic_concurrency` permits. It can make the train allowance overlap with Eval and external agent
capacity.

The agent process `--agent-timeout` is not active while a Group is owned by Prepare. The agent application's client
request to `RELAX_BASE_URL` is already open and its own wall-clock timeout continues. Read
[runtime-operations.md](runtime-operations.md) before enabling prelaunch.

## Check before launch

- Verify each source Group contains the expected `n_samples_per_prompt` Sessions.
- Verify every agent process can reach its first model request without waiting for another Group.
- Verify the Relax-facing agent client can keep that first request open through the longest prelaunch wait.
- For centralized external executors, run the capacity audit before assuming the all-Session barrier can complete.
- Treat Group-level failure and replenishment separately from individual Session errors.

## Evidence

Use `warming_prepare_group_count`, `ready_prepare_group_count`, `runtime_reward_group_count`, `finalized_group_count`, `resident_group_count`, and the first-request barrier state. A progress bar at its target does not prove all resident Groups are gone.

Source anchors:

- `relax/agentic/pipeline/prepare.py`
- `relax/agentic/pipeline/runtime.py::RuntimeDomain`
- `relax/agentic/session/service.py::ResidentGroup`
- `relax/agentic/session/service.py::AgenticSessionShard`
- `relax/agentic/rollout.py::AgenticResidentPipeline`
