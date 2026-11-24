# Fault event history

Fault history provides bounded, in-memory records owned by the training driver. It is separate from MetricsService and the performance timeline. The driver can keep the same history while Ray or a Controller is rebuilt; history is not persisted after the driver exits.

HealthStatus supplies stable fault identities and completion receipts through its existing health queries. Controller records handling decisions, local and global recovery operations, skipped roles, fatal termination and global restart budget exhaustion. A `handling_started` or `recovery_handoff` record alone does not establish success.

## Enable the query endpoint

The HTTP endpoint requires Uvicorn 0.36.0 or newer, as declared in `requirements.txt`. Earlier versions cannot load the custom event loop used for connection deadlines.

Add `--use-health-check --fault-events-enabled` to your existing training command. Both flags are required. The feature is off by default; requesting fault history without health checks prints a fixed warning and leaves collection disabled.

The driver logs the query URL and `run_id` after HTTP startup succeeds. `--fault-events-port` accepts 0–65535; the default 0 selects an available port. The endpoint binds only `127.0.0.1`. If the port is occupied or HTTP startup fails, in-memory recording remains available and training follows its existing path. A pure HTTP failure does not imply that history was lost.

Run queries on the driver machine, replacing `PORT` with the logged port:

```bash
curl 'http://127.0.0.1:PORT/fault-events/v1/status'
curl 'http://127.0.0.1:PORT/fault-events/v1/events?role=actor&limit=50'
```

Use an existing SSH tunnel for remote access. The run ID and cursor are identifiers, not authentication credentials.

## Read pages correctly

`GET /fault-events/v1/events` accepts `role`, `fault_id`, `limit` and `cursor`. Role and fault filters intersect. The default page size is 50 and the maximum is 100. Unknown parameters, duplicate parameters and invalid values are rejected.

The first page fixes `snapshot_end_seq`. Carry `next_cursor` and the same filters into subsequent requests until `done` is true. The page size may change. Newer events are excluded from this snapshot; start a new query to see them. A page scans at most 512 original sequence positions, so an empty filtered page can still contain a continuation cursor.

Snapshot pagination does not pin retained history. For example, after reading sequences 1–2 of a snapshot ending at 5, concurrent appends can evict sequence 3. The continuation returns sequences 4–5 and `missing_range: [3, 3]`; it never inserts newer sequences into that snapshot. If every unread position was evicted, an empty final page still reports the missing range.

Interpret these fields separately:

| Field | Meaning |
| --- | --- |
| `retention_truncated` | This run has evicted previously committed records because of capacity. |
| `missing_range` | Unread positions missing from this page's global sequence interval. |
| `gap_scope: global` | The gap is measured before applying any filter. |
| `filter_match_unknown` | A gap exists with active filters; the number of missing matches is unknown. |
| `admission_rejections` | Local admission attempts rejected, including temporary saturation. |
| `known_lost_events` | Events confirmed uncommitted and finally abandoned. Temporary rejection alone is not final loss. |
| `coverage_unknown`, `unknown_reasons` | Fixed diagnostics for observation or commit uncertainty. False does not prove complete distributed coverage. |

`GET /fault-events/v1/status` returns the run ID, history capacity/count/latest sequence, queue capacity/use, one in-flight slot, `query_available` and the same fixed diagnostics. It does not return configuration, raw errors or training data.

Application errors use fixed JSON codes: 422 `INVALID_QUERY`, 409 `RUN_MISMATCH`, 503 `UNAVAILABLE`, 405 `METHOD_NOT_ALLOWED`, and 404 `NOT_FOUND`. Uvicorn may reject malformed HTTP or excess connections before the application with fixed 400/503 text; clients should inspect the status and content type. No write endpoint, request body, compression or automatic API documentation is exposed.

## Local recovery events

For a local recovery, filter by the `fault_id` returned by a role query. With no delivery loss, a successful attempt has this sequence:

```text
fault_detected
handling_started    expected=7 observed=0 failed=0
recovery_handoff    expected=7 observed=7 failed=0
recovery_succeeded  expected=7 observed=7 failed=0
```

Evidence bits are redeployment (1), step restoration (2), and task submission with a saved reference (4). Local recovery requires all three, including in debug mode. Handoff describes control flow; success describes these recovery operations, not the numerical correctness of model weights or a completed training step.

If the original Service catches a `/set_step` failure and continues, the timeline instead records `recovery_failed` with `reason_code=step_restore_failed`, `observed_evidence=1` and `failed_evidence=2`. Later task submission or `mark_healthy` does not reverse that terminal result. Missing evidence stays unverified and sets `OUTCOME_UNOBSERVED`; it is not guessed to be success or a business failure.

Each continuous unhealthy period keeps the same discovery ID (D) and first reason. A later nonfatal-to-fatal transition allocates one escalation ID (U); a first fatal report needs only D. Repeated health queries retry unaccepted D/U once per ID per query round. After health clears, cached IDs may still be published, but never trigger another recovery. Replacing or retiring a source settles each unaccepted D/U once; accepted IDs remain the consumer's responsibility. For example, D accepted and U rejected followed by retirement adds one to `known_lost_events`; both rejected adds two; a successful retry adds none. Active publication state prevents D from reappearing after its stored record is evicted.

An unregistered role produces `handling_skipped`; Controller still treats the missing recovery handler as an error, stops the health checker and cancels pending tasks. Fatal handling preserves the existing callback/exit policy. Global members share a recovery ID. A new recovery can retry the same still-unhealthy fault in a later health round using a new recovery ID.

## Global recovery and completion receipts

Before rebuilding, the plan freezes the role graph and the configuration branch that determines recovery operations. Handoff atomically compares the rebuilt plan, merges local evidence and selects a terminal result. Different role graphs invalidate the comparison even if their evidence masks match. Ordinary checkpoint paths, router addresses and step values may change.

| Global startup branch | Required evidence |
| --- | --- |
| Colocate or hybrid, with actor | Initialization (1), all service step setters (2), task submission (4): 7. |
| Pure fully async, actor only | The same operations plus actor send (8): 15. |
| Pure fully async, actor + actor_fwd | Also actor_fwd receive (16): 31. |
| Pure fully async, actor + reference | Also reference receive (32): 47. |
| Pure fully async, all three | Both receives: 63. |
| No actor, or either global debug gate | Initialization and task submission: 5. |

Local remains 7 in debug mode. In colocate/hybrid, the rollout-manager setter waits for initial synchronization only when actor and rollout exist and the existing non-offline branch executes it. No weight receipt is invented for a skipped branch. Task evidence requires references for every service; a partial or empty task set cannot prove submission. It does not prove that training finished.

In pure async recovery, Actor/ActorFwd report after the actual worker-group call returns or raises. Service awaits the actual remote response, and Controller starts senders and receivers concurrently with `asyncio.gather` before restoring steps. The new HealthStatus has its fixed receipt token before deployment. Existing unhealthy queries merge the three receipt slots before callbacks, including empty unhealthy lists. Runtime retains no training responses and adds no driver polling RPC.

For example, handoff for actor + actor_fwd may record:

```json
{"event_type":"recovery_handoff","expected_evidence":31,"observed_evidence":7,"failed_evidence":0,"availability":"unverified"}
```

After both receipts arrive, a new `recovery_succeeded` record has expected=observed=31; earlier records stay unchanged. An explicit operation failure records `recovery_failed/operation_failed` with its failed bit. A failed remote response leaves operation evidence unverified until a completion receipt arrives; the original exception still propagates to training. Missing receipts stay unverified; elapsed time does not create a terminal result or increase `known_lost_events`. Conflicting slots prove neither success nor failure, but do not hide another operation's definite failure. `active_recoveries` counts pending contexts for the run rather than exposing per-recovery pending state.

Fault source and receipt receiver generations are independent. A fault from G0 starts a recovery targeting G1; destroying G0 does not interrupt that recovery. Starting another recovery alone does not end it either. Only actually retiring G1 interrupts its pending result with `source_retired`. Source observation failure does not discard the independent receipt slots. Abandoned startup before handoff records `superseded`. Stale tokens cannot affect newer attempts. A successful terminal never reverses: a later opposite receipt only sets `RECEIPT_CONFLICT`. `SOURCE_RETIRED` may coexist with verified success.

## Python storage API

This synchronous example exercises the storage contract without starting Ray or a model:

```python
from relax.utils.fault_events import EventDraft, FaultEventStore
from relax.utils.fault_events.models import new_id

store = FaultEventStore(new_id())
draft = EventDraft(
    run_id=store.run_id,
    fault_id=new_id(),
    event_id=new_id(),
    role="actor",
    source_generation=0,
    event_type="fault_detected",
    reason_code="reported_error",
)
assert store.append(draft) == "accepted"
assert store.append(draft) == "duplicate"
page = store.query(role="actor")
assert len(page["events"]) == 1
```

`FaultEventRuntime.record(draft)` is the asynchronous admission API for internal producers. True means the FIFO accepted the draft, not that the Store committed it. Producers retain a stable event ID across a permitted retry. Storage deduplicates IDs only while their records are retained; the integrated health lifecycle publication state prevents an active fault from being republished after retention eviction.

Identical pending IDs share an admission slot; conflicting contents for a pending ID are rejected. Before settling abandoned delivery, the consumer checks retained history so an already committed duplicate is not counted as lost. If it cannot establish commit status, it reports uncertainty instead of guessing a loss count.

The schema allows only IDs, an allowed service role, fixed event/reason codes, sequence/time and evidence masks. All fields are present in a returned record; absent context is JSON null. It rejects unknown fields, non-native integer types such as bool, invalid combinations and oversized records. It does not store exception text, traceback, URLs, samples, model data, configuration or environment variables. `EventDraft.from_dict` provides a fixed-error validation boundary for dictionaries.

## Fault logs through OpenTelemetry

Fault-log export keeps the existing event schema, local query API and recovery criteria. It does not install a global OTel provider, instrument ordinary Python logging or enable other observability signals.

The driver owns one private official SDK `LoggerProvider`, with an explicit `service.name=relax` resource, and one official OTLP HTTP exporter. Only a newly accepted Store commit supplies an immutable record snapshot to a separate bounded export queue: at most 512 queued records and one in flight. `offer` constructs the SDK `LogRecord` on the Runtime consumer, outside Store and admission locks. Only the export worker emits the SDK record, invokes `SimpleLogRecordProcessor` and performs export/network I/O. There is no second SDK batch queue.

The dependencies are `opentelemetry-sdk>=1.45.0,<1.46` and `opentelemetry-exporter-otlp-proto-http>=1.45.0,<1.46` for Python 3.10 and newer, plus the `opentelemetry-exporter-otlp` umbrella in the same range so a preinstalled gRPC exporter stays on the SDK's release; export itself uses HTTP only. The 1.45 range preserves the exporter's owned-client diagnostic interface used to suppress potentially sensitive transport error details, without changing any global logger. If that compatibility contract is unavailable, export startup fails closed while local history continues. The private provider and exporter are created only when this fault-log export is enabled.

Export requires both fault-history flags and an explicitly configured endpoint. For an existing local OTLP HTTP receiver, configure the driver environment before starting the existing training command:

```bash
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://127.0.0.1:4318/v1/logs
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL=http/protobuf
# Add --use-health-check --fault-events-enabled to the training command.
```

The logs-specific endpoint is a complete URL; alternatively, `OTEL_EXPORTER_OTLP_ENDPOINT` is a base URL to which `/v1/logs` is appended. This interface supports `http/protobuf` only, honors `OTEL_SDK_DISABLED`, and uses the official exporter's standard logs-specific/general header and timeout variables. Header values must use the OTLP environment syntax; malformed raw controls or delimiters are rejected before SDK parsing, while URL-encoded values are accepted. Credential-provider entry points are intentionally rejected so optional authentication code cannot block training startup. No endpoint is inferred from the exporter's localhost default. With no configured endpoint, no export worker is started and local recording/querying continue normally. Configuration or export failures leave the local result unchanged. A rejected setting logs one warning naming the setting and constraint, for example `INVALID_OTEL_CONFIG: headers must use the SDK name=value grammar`, never its value.

Every exported record uses native `event_name` and `body` equal to `relax.fault.<event_type>`. All 17 local record fields have a whitelisted attribute; no arbitrary dictionary, exception, environment, configuration or training content is attached:

| Local field | OTel attribute | Value |
| --- | --- | --- |
| `schema_version` | `relax.fault.schema_version` | Integer |
| `run_id` | `relax.fault.run_id` | 32-character lowercase hex string |
| `fault_id` | `relax.fault.fault_id` | 32-character lowercase hex string |
| `event_id` | `relax.fault.event_id` | 32-character lowercase hex string |
| `source_generation` | `relax.fault.source_generation` | Decimal string |
| `role` | `relax.fault.role` | Fixed role string |
| `event_type` | `relax.fault.event_type` | Fixed event string |
| `reason_code` | `relax.fault.reason_code` | Fixed reason string; omit if null |
| `recovery_id` | `relax.fault.recovery_id` | 32-character lowercase hex string; omit if null |
| `recovery_mode` | `relax.fault.recovery_mode` | `local` or `global`; omit if null |
| `expected_evidence` | `relax.fault.expected_evidence` | Integer bitmask |
| `observed_evidence` | `relax.fault.observed_evidence` | Integer bitmask |
| `failed_evidence` | `relax.fault.failed_evidence` | Integer bitmask |
| `sequence` | `relax.fault.sequence` | Decimal string |
| `recorded_at_ms` | `relax.fault.recorded_at_ms` | Decimal string |
| `result_scope` | `relax.fault.result_scope` | Fixed scope string; omit if null |
| `availability` | `relax.fault.availability` | Fixed availability string |

Native severity is also fixed: `INFO` for `handling_started`, `recovery_handoff` and `recovery_succeeded`; `WARN` for `fault_detected`, `recovery_interrupted` and `handling_skipped`; and `ERROR` for `fatal_escalated`, `recovery_failed` and `termination_requested`.

Decimal strings preserve the full unsigned 64-bit range without OTLP's signed-integer limit. Native timestamp is `recorded_at_ms * 1_000_000` when it fits OTLP's unsigned 64-bit nanoseconds; otherwise that timestamp is unset while the exact attribute remains. Severity is a fixed mapping from event type/reason, not caller-provided text. Valid trace ID, span ID and trace flags are captured where each discovery, fatal escalation or recovery event is produced, then carried through the health/Ray sidecar and async queues. The sidecar retains at most two valid contexts per role and clears them when that role's source is cleared. Missing or invalid context remains absent and does not change local health or history. No baggage is copied, no span is created, and business IDs are never used as trace IDs. This sidecar does not change the persisted business schema.

Local duplicate commits do not enqueue another export; network retries can still deliver the same stable `event_id` more than once. The receiver can use that ID for correlation. Export queue saturation, transport failure and shutdown abandonment use separate `otel_*` diagnostics; they do not increment local `known_lost_events` or disable the Store. Transport success does not prove durable receiver storage. In particular, the official exporter's success return cannot distinguish a receiver's OTLP `partial_success` rejection, so a successful attempt is not a per-record persistence receipt.

Shutdown stops export admission and waits at most 50 ms for the export worker. `otel_shutdown_dropped` counts only queued records abandoned at shutdown. A request already in flight is not counted as dropped: after the 50 ms wait its outcome remains unknown until the worker returns, when the actual result updates the transport counters. It must not be described as failed or delivered while pending; hard process exit can still lose its record. Export shutdown does not extend the local Runtime's queue-drain policy or wait for Ray.

Global recovery reuses this same driver-owned provider, committed-record mapping and export queue across Controller rebuilding. Receipt and trace sidecars share the existing health envelope; completion evidence creates no second exporter.

To check a receiver, use its own documented health command or `curl`. Posting plain JSON to `/v1/logs` is not a valid check, because this interface sends binary `http/protobuf`.

## Bounds, failures and shutdown

History and its ID index retain at most 4,096 records each. A final compact UTF-8 event is at most 1,024 bytes. Runtime admission holds at most 512 drafts plus one in flight, 18 publication slots, nine health-snapshot members and two recovery contexts. Sequence and diagnostic counters use unsigned 64-bit bounds and never wrap. Recording time is for display; sequence is the ordering authority.

Admission waits at most 5 ms for its lock and does not call the Store, HTTP or Ray. Storage operations wait at most 50 ms for their lock. These are lock-wait budgets, not hard real-time bounds on total call duration. A single consumer retries a confirmed pre-commit busy result once with the same ID. If a commit becomes uncertain, it disables the original Store's append and query paths and reports uncertainty; it does not replace history with an apparently successful empty store.

HTTP has one server thread, four active business slots, backlog/concurrency limits, a 2 KiB request target limit, a 1 KiB cursor limit and a 128 KiB response limit. A business slot remains occupied through response sending and cancellation cleanup. Connection lifetime and request processing/sending have a three-second deadline; keep-alive is disabled and h11's incomplete-event buffer is limited. These controls apply to trusted loopback use and do not promise a hard bound on OS TCP connections or whole-process RSS.

Each relevant replica lazily creates one receipt sender with at most eight queued receipts and one in flight; each encoded receipt is at most 256 bytes. The business method only attempts local enqueue. A slow confirmation keeps occupying its slot without replacement requests or threads. Ordinary calls and disabled collection start no sender. Receiver loss or saturation may leave evidence unverified without changing training results.

Driver shutdown signals Runtime and HTTP closure without waiting for queue drain or adding Ray waits. A hard exit can lose tail records. The history does not establish a business failure merely because a record or completion receipt is absent.
