# Resident lifecycle

```text
DataSource GroupInput -> Prepare registers every original Session on one Shard
-> Runtime leases the Group -> ready members generate independently under the step gate
-> each terminal Session publishes its logical result and delegates background cleanup
-> Runtime gathers all original results and releases the Group permit
-> Reward and Transfer preserve complete-Group semantics
```

Prepare and Runtime share `agentic_concurrency` Group permits. Oversampling determines legal launch demand; independent
member generation adds no new demand or budget. Unleased members cannot generate. A member's active timer starts after
its first request is ready and its leased generation gate opens. A Group counts as interrupted only after every member
has reached its first IR, so an executor-queued member cannot supply fully-async close credit.

Prelaunch may retain unleased registered work across a synchronous boundary. Partial rollout and fully async retain
their original Session/request identity and partition rules. Surplus prepared Groups without a future lease must be
dropped, including completed Prepare tasks. See [partial-and-async-lifecycle.md](partial-and-async-lifecycle.md).

The launched `--agent-command` cleans up its resources on natural exit and when Relax sends SIGTERM because of a timeout
or another reason. Relax publishes logical results and refunds Group
capacity before best-effort physical cleanup. Abort, backend close, and process cleanup errors warn independently;
they do not invalidate successful exports. Retired records fence late callbacks. Internal backend permits and token
leases must still be compensated using their exact identities.

Check normal completion, cancellation, failed startup, late sibling requests, pause/resume, and Eval. A controlled
member failure drops the whole Group. Physical executor occupancy can outlive logical residency; verify external quotas
and `agent-command` cleanup independently. See [external-agent-capacity.md](external-agent-capacity.md).

Source anchors: `relax/agentic/pipeline/prepare.py`, `relax/agentic/pipeline/runtime.py`,
`relax/agentic/session/service.py`, `relax/agentic/runner/ipc.py`, and `relax/agentic/rollout.py`.
