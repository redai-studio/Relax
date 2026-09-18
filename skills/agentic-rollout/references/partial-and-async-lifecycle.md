# Partial and fully-async lifecycle

Load this reference when prelaunch, partial rollout, or fully async is enabled.

Partial rollout and fully async are mutually exclusive. Apply the sections for the selected mode.

## Partial rollout transition

```text
generation open
-> close the train generation gate
-> abort interruptible backend attempts
-> accumulate returned token prefixes and increment abort_count
-> park interrupted IRs
-> pause normal Session active-time budgets
-> next resume updates rollout_id and restores timeouts
-> dispatch parked IRs with their accumulated prefix
```

When the abort threshold is reached, protection is pending for the next resume. Resume promotes every live IR in that Session to protected. Active protected Sessions bypass later interruption and remain protected until Session finalization.

Do not describe the threshold as preventing the first qualifying abort. It prevents the next interruption after the pending state is activated.

Protection does not disable `--agent-timeout`. A protected Session bypasses later partial-rollout aborts while its active
process timer continues; timeout still terminates the agent process. Step close waits for protected Groups to
finalize both before and after the Shard pause transition. Process timeout does not bound every backend or finalization
RPC.

## Timeout boundary

One call made by the agent application's Relax-facing Chat Completions, Responses, or Messages client remains the same
HTTP request while it is held by prelaunch, or while its backend attempt is aborted, parked, and resumed. Fully async
can likewise retain an unfinished request across a training boundary. The request's read/overall timeout continues on
wall-clock time throughout that wait. Buffered SSE connection frames and heartbeats do not change this lifecycle.

`--agent-timeout` is a separate active process budget for containing defects in the agent itself, such as a stuck loop
or tool execution. Gated normal Sessions pause it and resumed Sessions restore it. Raising it does not extend the
Relax-facing client request timeout, bound the Prepare first-request barrier, or guarantee that a backend abort RPC
returns.

When repeated cross-step waits have no known finite wall-clock bound, use a client configuration that intentionally has
no read/overall deadline and retain explicit cancellation and the agent process timeout. Read
[runtime-operations.md](runtime-operations.md) for the client check.

## Fully-async close

Fully async may close a step with physical debt only when resident Groups provide interrupted credit. Unfinished Groups remain resident for later steps. Final backfill closes remaining partition debt without opening a new train partition.

Dynamic global batch close is a separate partial-rollout path: it pauses generation, collects a DP-aligned completed surplus, seals the partition, and waits for Transfer writes. Do not combine its accounting with fully-async backfill.

## Check before launch

- Confirm which Sessions may survive a step boundary.
- Confirm Eval overlap and external slots using the retained-Session capacity rule.
- Confirm protection threshold semantics and timeout behavior.
- Confirm the Relax-facing client timeout spans every applicable prelaunch, abort, park, resume, and fully-async hold.
- Confirm `--use-rollout-routing-replay` is not combined with partial rollout.
- Do not assume generic `--mask-offpolicy-in-partial-rollout` changes resident Agentic response masks; verify an Agentic consumer before claiming it applies.

Source anchors:

- `relax/agentic/session/service.py::pause_generation`
- `relax/agentic/session/service.py::resume_generation`
- `relax/agentic/pipeline/runtime.py::RuntimeDomain.pause_generation`
- `relax/agentic/rollout.py::_step_can_close`
- `relax/agentic/rollout.py::_close_rollout_step`
- `relax/agentic/pipeline/transfer.py`
