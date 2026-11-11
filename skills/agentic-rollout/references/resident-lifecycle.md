# Resident lifecycle

Use this lifecycle to reason about logical Group ownership, physical slots, generation gates, and cleanup.

## Ownership and scheduling

```text
DataSource GroupInput -> Prepare owns an unleased Group -> SessionShard owns its Sessions
  ordinary: start all members -> all-member first-request barrier -> Runtime lease
  capacity: reserve slots -> start admitted members -> Runtime lease -> run ready members
            -> start remaining members as confirmed slots become available
-> each terminal Session joins requests and releases owned resources
-> confirm its physical slot return (capacity mode)
-> Runtime gathers every original result and retires the logical Group
-> Reward and Transfer retain complete-Group semantics
```

Ordinary per-Session RM can overlap Runtime gather. Reward outside the agent is distinct from a harness that runs tests
inside its Session; the latter still owns its task resources until confirmed cleanup.

## Core invariants

| Boundary | Required invariant |
| --- | --- |
| Logical residency | Prepare and Runtime share `agentic_concurrency` Group permits |
| Physical capacity | Train only; all Shards on one host share `agentic_session_capacity_per_node` within the Runtime |
| Placement | One complete Group fits one host; allocate slots before submitting the corresponding start RPC |
| Ordinary lease | Every member has reached its first live request before the Group leases |
| Capacity-managed lease | Members may be waiting for slots or first requests; only leased, ready members with an open generation gate can run |
| Prelaunch | Preparation does not bypass the Runtime lease or weight-update generation gate |
| Group completion | Gather every original Session result; a controlled member drop still drops the complete Group |
| Physical release | Match the exact launch identity and owner proof; timeouts, lost RPC responses, and logical removal are not proof |
| Logical release | Retire a Group once; preserve uncertain owners and their debits for later cleanup |

## Prelaunch and cross-step boundaries

Prelaunch fills the same logical and physical budgets with next-step work. Without Session capacity it starts all
members and waits at the ordinary first-request barrier; with capacity it starts admitted members and leaves the rest
pending. Neither path allows unleased members to generate.

Prepare-owned work may survive a synchronous weight boundary only when prelaunch retains it. Runtime-owned work must
drain for that synchronous boundary. Partial rollout and fully async may retain the same Sessions, pending startup,
requests, and unreleased debits across steps. Step changes never create another host allowance or reset a receipt's
identity. Read [partial-and-async-lifecycle.md](partial-and-async-lifecycle.md) for mode-specific transitions.

Prepare waits do not consume the Session's Runtime active-time budget, but the agent client's open request still has
a wall-clock deadline. Read [runtime-operations.md](runtime-operations.md) before choosing the client timeout.

## Check the integration

- Verify Group size, eligible host identities, and each resource owner's scope.
- Verify members can reach model requests without an application-level dependency on unstarted siblings or another Group.
- Check ordinary barriers or incremental-start/eager-generation behavior according to the selected capacity mode.
- Observe normal completion, cancellation, failed startup, and cleanup retry without duplicate launches or refunds.
- Keep Group-level failure/replenishment separate from member resource release.

Use Prepare/Runtime Group metrics, Shard state, and physical-capacity metrics together. A target progress count proves
neither that every resident Group has finished nor that all task resources have been released.

Source anchors: `relax/agentic/pipeline/prepare.py`, `relax/agentic/pipeline/runtime.py`,
`relax/agentic/pipeline/session_capacity.py`, `relax/agentic/session/service.py`,
`relax/agentic/runner/resources.py`, and `relax/agentic/rollout.py`.
