# Fault event history

Fault history provides bounded, in-memory records owned by the training driver. It is separate from MetricsService and the performance timeline. The driver can keep the same history while Ray or a Controller is rebuilt; history is not persisted after the driver exits.

This first implementation batch provides the event model, storage, asynchronous admission, read-only queries and driver lifecycle. Automatic HealthStatus/Controller event production and operation receipts are introduced in the following batches. Enabling this batch alone does not yet produce a complete fault/recovery timeline.

## Enable the query endpoint

The HTTP endpoint requires Uvicorn 0.36.0 or newer, as declared in `requirements.txt`. Earlier versions cannot load the custom event loop used for connection deadlines. The validated environment uses Uvicorn 0.54.0.

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

`FaultEventRuntime.record(draft)` is the asynchronous admission API for internal producers. True means the FIFO accepted the draft, not that the Store committed it. Producers retain a stable event ID across a permitted retry. Storage deduplicates IDs only while their records are retained; lifecycle publication state in the later integration batch prevents an active fault from being republished after retention eviction.

Identical pending IDs share an admission slot; conflicting contents for a pending ID are rejected. Before settling abandoned delivery, the consumer checks retained history so an already committed duplicate is not counted as lost. If it cannot establish commit status, it reports uncertainty instead of guessing a loss count.

The schema allows only IDs, an allowed service role, fixed event/reason codes, sequence/time and evidence masks. All fields are present in a returned record; absent context is JSON null. It rejects unknown fields, non-native integer types such as bool, invalid combinations and oversized records. It does not store exception text, traceback, URLs, samples, model data, configuration or environment variables. `EventDraft.from_dict` provides a fixed-error validation boundary for dictionaries.

## Bounds, failures and shutdown

History and its ID index retain at most 4,096 records each. A final compact UTF-8 event is at most 1,024 bytes. Runtime admission holds at most 512 drafts plus one in flight. Sequence and diagnostic counters use unsigned 64-bit bounds and never wrap. Recording time is for display; sequence is the ordering authority.

Admission waits at most 5 ms for its lock and does not call the Store, HTTP or Ray. Storage operations wait at most 50 ms for their lock. These are lock-wait budgets, not hard real-time bounds on total call duration. A single consumer retries a confirmed pre-commit busy result once with the same ID. If a commit becomes uncertain, it disables the original Store's append and query paths and reports uncertainty; it does not replace history with an apparently successful empty store.

HTTP has one server thread, four active business slots, backlog/concurrency limits, a 2 KiB request target limit, a 1 KiB cursor limit and a 128 KiB response limit. A business slot remains occupied through response sending and cancellation cleanup. Connection lifetime and request processing/sending have a three-second deadline; keep-alive is disabled and h11's incomplete-event buffer is limited. These controls apply to trusted loopback use and do not promise a hard bound on OS TCP connections or whole-process RSS.

Driver shutdown signals Runtime and HTTP closure without waiting for queue drain or adding Ray waits. A hard exit can lose tail records. The history does not establish a business failure merely because a record or completion receipt is absent.

## CPU verification

Run from the repository root in an environment with its existing test and HTTP dependencies:

```bash
PYTHONPATH=. pytest -q \
  tests/utils/test_fault_event_store.py \
  tests/utils/test_fault_event_runtime.py \
  tests/utils/test_fault_query_server.py \
  tests/entrypoints/test_fault_events_lifecycle.py
pre-commit run --all-files
```

These tests use the actual Store/Runtime and real loopback sockets. Entrypoint tests execute the production entrypoint with explicit substitutes for model/Ray initialization and process exit. They check foundation contracts and lifecycle isolation, not GPU numerical behavior, full training, or the recovery integrations scheduled for subsequent batches. The existing health-thread regression remains a prerequisite check; its fix is already present on main.
