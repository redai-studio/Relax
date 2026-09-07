# Runtime operations

Contents: [process](#agent-process-contract), [client timeout](#relax-facing-agent-client-timeout),
[model APIs](#model-api-contract), [parsers](#parser-preflight), [optional controls](#optional-runtime-controls),
[errors](#error-and-cleanup-categories), [evidence](#observable-evidence).

## Agent process contract

Relax injects `RELAX_INPUT_JSON`, `RELAX_OUTPUT_JSON`, `RELAX_SESSION_IO_DIR`, `RELAX_BASE_URL`, `RELAX_SESSION_ID`, `RELAX_ROLLOUT_MODE`, and `RELAX_GROUP_ID`. The `RELAX_` prefix is reserved from `--agent-env` overrides.

`--agent-timeout` is an active-time safety budget for the agent process. It starts after the Group is leased to Runtime,
pauses for gated normal Sessions, and continues for protected Sessions. Exhausting it terminates the process regardless
of whether the agent is defective or merely slow. It cannot catch an agent stuck in Prepare before its first request,
and it does not bound the first-request barrier, backend abort, or finalization cleanup RPCs.

## Relax-facing agent client timeout

The **agent client** in this skill means the client inside the agent application that sends Chat Completions, Responses,
or Messages requests to `RELAX_BASE_URL` and authenticates with `RELAX_SESSION_ID`. Its timeout applies to each
individual request.

One Relax-facing request can remain open without a final response while:

- a prelaunched first request waits at the Prepare gate through the current rollout, optimizer work, or Eval;
- a partial-rollout request keeps the same HTTP waiter across backend abort, park, and next-step resume;
- fully-async execution retains unfinished work across a training boundary.

A protected Session follows a different path: it bypasses later interruption and continues generating instead of
parking until the next resume.

The request's read/overall timeout and intermediary idle timeouts continue on wall-clock time. Configure them to let the
same request survive every applicable hold. `--agent-timeout` has a separate failure-containment role and does not set
or extend this per-request deadline.

Inspect the actual client construction and every enclosing deadline. Configure its read/overall timeout to exceed the
longest possible request lifetime, or disable that deadline intentionally when no finite bound exists. Connect, write,
and pool acquisition may remain bounded when the client library supports separate timeout fields. A large example value
such as `timeout=9999` is evidence only after comparing it with the real run's worst-case wait.

| Stage C result | Condition |
| --- | --- |
| `PASS` | The Relax-facing client and every intermediary deadline are known to cover the worst-case lifetime of one request |
| `UNSAFE` | A known timeout is shorter than a possible prelaunch, partial-rollout, or fully-async wait |
| `UNVERIFIED` | Client construction or intermediary idle timeouts are unknown |

## Model API contract

| Capability | Shared contract |
| --- | --- |
| Authentication | `Authorization: Bearer <RELAX_SESSION_ID>` |
| Context | Complete history on every request; stable tools and applicable template arguments |
| Canonical state | Protocol payload projects to `messages + function tools + chat_template_kwargs` before SessionForest matching |
| Transport | JSON for missing/false `stream`; Buffered SSE for `stream=true` |
| Buffered SSE | Connection frame and 15-second heartbeats while pending; complete terminal events after generation |
| Model | Request `model` is echoed and does not select the backend |
| Sampling | Training supplies temperature/top-p; request values do not control them |
| Response parsing | Optional reasoning parser runs before the optional tool-call parser; tool parsing requires normalized tools |

| Interface | Consumed request shape | Response shape |
| --- | --- | --- |
| Chat Completions | Complete `messages`; nested function `tools`; `chat_template_kwargs`; `max_completion_tokens` or legacy `max_tokens`; `stop`; `seed`; `logprobs` | Chat completion JSON, or one complete chunk with usage plus `[DONE]`; `include_usage` does not change this shape |
| Responses | Optional `instructions`; complete typed `input` with message/reasoning/function-call Items; flat function `tools`; `max_output_tokens` | Response JSON, or created/output-item-done/completed, incomplete, or failed events |
| Messages | Complete `system + messages` with text/image/thinking/tool blocks; `input_schema` tools; required `max_tokens`; `stop_sequences` | Anthropic message JSON, or ping/content-block/message terminal events |

Pass `RELAX_BASE_URL` unchanged to a standard client that appends its own resource path. For a custom HTTP client, build
the final URL as `RELAX_BASE_URL.rstrip("/") + endpoint`, using the endpoint from the interface table. Verify the final
URL rather than the configured base alone, and preserve the existing Relax route prefix during URL resolution.

Responses `previous_response_id` does not continue Relax state. Explicit export uses canonical Chat-shaped records for
all request protocols. Verify the current source before adapting a client; a successful basic request does not prove
that every client payload shape is consumed.

## Parser preflight

Read `_decode_response_payload()` in the current `relax/agentic/session/service.py` and the parser implementations in
the installed SGLang version. Compare them with the exact model and chat-template output.

| Result | Evidence |
| --- | --- |
| `PASS` | Required parsers are configured and a real tool turn plus final-answer turn produce the expected structured messages |
| `NEEDS_CHANGES` | Parser is missing, incompatible, ordered incorrectly for the output, or produces unusable tool arguments |
| `UNVERIFIED` | Raw model output, installed parser support, or parsed Relax response was not inspected |
| `N/A` | The verified model/template emits neither separate reasoning syntax nor structured tool calls |

## Optional runtime controls

Read this section only when Session lifecycle or program admission is enabled. Resolve values and dependencies with
[parameter-preflight.md](parameter-preflight.md).

Session lifecycle terminal cleanup performs bounded `close_session` calls.

Program admission runs before the fleet request permit. Protected requests bypass admission, stale or unavailable metrics
fail open, and aged waiters bypass after the configured maximum wait.

SGLang request permits are independent from external agent slots and program-admission token reservations. Do not collapse these three capacities.

## Error and cleanup categories

| Evidence | Meaning |
| --- | --- |
| Agent exits/fails under the process contract | Controlled Session failure; Group is dropped |
| Runtime/backend/transport failure | `RuntimeGroupError` |
| Explicit export misses committed state | Non-finalizable Session; its Runtime Group is dropped |
| Context length exceeded | HTTP 400 in the endpoint's JSON error envelope, or terminal SSE error after streaming starts |
| Unknown/discarded Session | HTTP 404 in the endpoint's error envelope |
| Process active-time exhausted | Process-group termination |
| Relax-facing client deadline expires | Agent-side request failure or disconnect before Relax completes the Session request |

## Observable evidence

Check `accounting_start`, `accounting_end`, `idle_heartbeat`, progress `scored/prepared`, `resident_group_count`,
`interrupted_runtime_group_count`, Shard health/debug state, and the agent log tail. When KV features are enabled, use
the exact `agentic_kv/*` metric names.

Use `--save-debug-rollout-data <path-with-{rollout_id}>` when finalized Sample evidence is needed. It stores exported
Samples and their state hashes, turns, abort counts, timings, and weight versions; it does not capture the complete live
forest or resident/parked/dropped Sessions. Inspect those through `/debug_state` and logs. Avoid full dumps when their
volume would distort a scale experiment.

`Rollout {rollout_id} generation: 100%` means the target finalized Sessions were collected. It does not prove that every
resident Group or agent process has exited.

Source anchors:

- `relax/agentic/runner/ipc.py`
- `relax/agentic/session/service.py`
- `relax/agentic/session/admission.py`
- `relax/agentic/session/admission_coordinator.py`
- `relax/agentic/rollout.py`
