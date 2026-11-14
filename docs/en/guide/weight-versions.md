# Independent weight publication and consumption

The versioned weight path separates training export from inference installation. Megatron exports logical weights through its existing converter. A consumer retrieves a complete version or delta package over SharedFS or TCP, reconstructs a complete safetensors checkpoint locally, and installs it through the SGLang Loader.

Downloads and reconstruction precede the inference pause. The installation gate reopens after every worker finishes loading, physical weight verification, cache flushing and an internal generation probe. The consumer then acknowledges its durable cursor.

## Configuration

Source file operations use `source_io_timeout` (3600 seconds by default). Training threads exchange participant completion states while worker threads operate on CPU tensors and files. Device copies and training collectives stay on the training thread. Running file operations drain before scratch is released. When every participant completes successfully, the actual result is returned with an elapsed-time warning; worker errors still fail the operation.

Publication candidates, exports and shard caches have stream/rank ownership and exclusive leases. Restart recovery protects the journal's pending candidate and reclaims abandoned copies from the same owner. Failed removals are recorded in `cleanup-*.json` and retried at startup. Legacy directories without ownership markers require separate ownership verification.

TCP `timeout` bounds connection and frame transfer; control operations have a separate finite `control_timeout` (120 seconds by default) for storage admission and durable acknowledgement. Retries share that operation budget. Set it in the consumer transport configuration and with `serve --control-timeout` on the source. Capacity scans use bounded metadata workers and still count hard-linked content once. `full_timeout` (600 seconds by default) bounds same-target FULL repair. Configure the server with `serve --full-timeout`. Concurrent requests share an in-flight repair, while later requests revalidate existing FULL objects. Before closing generation admission, consumers check checkpoint names, shapes and dtypes against each runtime TP/PP/EP layout, including integer buffers. Device weights are still verified bit for bit after installation. Verification retains at most four safetensors readers within one target, avoiding repeated parsing of large MoE headers; all readers close after success or failure and are reopened for the next verification.

Both flags are opt-in. Omitting them preserves the existing synchronization path.

```bash
python -m relax.entrypoints.train \
  --weight-version-config /path/to/publisher.json \
  --weight-consumer-config /path/to/consumer.json \
  ...
```

Example `publisher.json`:

```json
{
  "store": "/path/to/published",
  "scratch": "/path/to/publication-scratch",
  "stream": "policy",
  "disk_budget": 34359738368,
  "max_weight_bytes": 8589934592,
  "anchor_interval": 20,
  "compression": false,
  "full_only": false,
  "source_delta": false
}
```

Capacity fields use bytes; these values illustrate the format. Size each store for its actual model and retained history. Source snapshots, publication scratch, consumer caches and reconstructed checkpoints have separate storage lifetimes. Measure their combined node memory and storage usage.

Retained logical weights default to `materialized/` inside the store and share its `disk_budget`. Set `canonical_root` and `canonical_budget` before the first publication to place them on an independent persistent volume with a separate per-store quota. Each store has its own namespace on that volume. Local placement metadata survives restarts; delivery manifests contain no publisher paths. Collection follows the same version references and consumer cursors and removes the corresponding canonical data. Existing stores keep their placement; migrate data separately while publication, consumption and collection are stopped.

For example, add `"canonical_root":"/path/to/local-canonical"` and `"canonical_budget":34359738368` to keep canonical data on the publication owner's persistent disk while placing delivery packages in a shared directory. The owner's TCP service uses the recorded placement for same-target FULL repair. A SharedFS client can add `"repair_owner":{"host":"publisher-host","port":9010}` to its `transport` configuration: the owner handles FULL repair while object reads continue through SharedFS. Generating FULL directly on the SharedFS client requires access to the canonical volume. Configure the publisher and FULL-serving process with the same budget.

`source_delta=false` uses complete export. Setting it to `true` enables REUSE/PATCH/FULL selection for registered raw/direct and Bridge mappings. Bridge applies exact replacements in gathered Megatron coordinates, then uses the existing BridgeConverter for QKV, gated MLP, expert and visual conversion. Missing source caches, unregistered mappings and dense changes use the complete converter. Replica mismatches identify the source parameter and shard so the training runtime can repair consistency before recapture.

Set `source_patch=false` to retain group reuse while sending changed groups through the complete converter, for a source ablation. `source_cache_root` optionally places reconstructible complete exports, per-rank shard bases and adaptive conversion caches on a separate local filesystem. Complete and adaptive exporters use the same configured placement. Publication candidates, journals and retained canonical weights still use `scratch` and the store configuration. Losing the source cache requires complete conversion again; it does not remove committed targets. Capacity admission checks both filesystems conservatively.

Independent Ray clusters on one host can set different `RELAX_SERVE_HTTP_PORT` values (default `8000`). The resolved port is shared by the driver, service clients, workers and Controller restart. The validation driver exposes this as `--serve-port`; each run still needs separate GPU assignments, stores and consumer directories.

PATCH selection includes remote frame bytes, padded plan exchange and per-frame confirmation exchanges. It compares this payload model with a lower bound on the complete export's TP/PP/EP sends. The route receipt keeps these budgets alongside measured counters. A single-rank source with no export communication uses FULL for changed tensors and REUSE for unchanged tensors.

Example `consumer.json`:

```json
{
  "directory": "/path/to/node-local-consumers",
  "consumer_prefix": "rollout",
  "stream": "policy",
  "cache_budget": 34359738368,
  "checkpoint_budget": 34359738368,
  "max_weight_bytes": 8589934592,
  "start_version": 0,
  "mode": "all_versions",
  "transport": {
    "type": "sharedfs",
    "store": "/path/to/published",
    "disk_budget": 34359738368
  }
}
```

The engine and its workers must see the same node-local consumer directory. Engines on that node share verified objects and reconstructed checkpoints, with separate installation state and subscriptions. The last owner releases the checkpoint. Installations run sequentially on each node to avoid overlapping Loader workspaces. The managed RolloutManager consumes available versions before generation or evaluation. Restarted engines and engines with released memory first restore their acknowledged version.

For TCP, use `{"type":"tcp","host":"publisher-host","port":9010}`. Run the object service independently so retained versions remain available after training exits:

```bash
python -m relax.distributed.weight_sync \
  --store /path/to/published --disk-budget 34359738368 \
  --max-weight-bytes 8589934592 \
  serve --bind 127.0.0.1 --port 9010
```

The TCP service is intended for a trusted deployment network. Provide access control and encryption at the deployment layer; use an address reachable through that protected channel for remote access.

The publisher uses `encode_workers=4` by default (any positive integer). CPU block reads, encoding, hashing and immutable content writes run in a bounded pool. Per-content locks deduplicate writes, capacity reservations are serialized, and one owner commits the ordered manifest only after every write finishes. Source admission reserves 64 MiB per configured codec worker on the publication owner and one window on each other source rank. Use the same worker count for paired latency measurements.

## Lifecycle and recovery

### Sealed offline sets

After the selected versions finish publication, seal their delivery files:

```bash
python -m relax.distributed.weight_sync \
  --store /path/to/published --disk-budget 34359738368 \
  --max-weight-bytes 8589934592 \
  seal --name evaluation --stream policy --versions 1 3
```

The command verifies every encoded object and reconstructed block in the dependency closure, then atomically writes `seals/evaluation.json`. The inventory records the requested versions, dependency versions, content roots, package identities and encoded size. Model configuration is bound by the content root. Collection preserves the complete set until `release-seal --name evaluation` is run. Repeating the same seal verifies its files again; a different set needs a new name.

Once sealing succeeds, the training producer can exit. For a detached archive, copy the inventory, its listed `versions/<stream>/<version>.json` files and their referenced `objects/` files into a fresh directory. These are sufficient to reconstruct weights; source caches and externally placed canonical weights are unnecessary. Keep the archive writable for fetch retention records. A fresh consumer can use `fetch --source-store /path/to/archive --stream policy --version 3 --checkpoint-output /path/to/checkpoint`, with its own store and capacity arguments. The published configuration is restored automatically. Installation into an inference engine remains a separate step.

An interrupted or failed seal leaves no completed inventory. Restore missing or damaged delivery files and repeat the command. Releasing a seal removes its retention reference; deleting versions still requires the explicit `collect` operation.

The publisher and managed consumers default to `canonical_blocks=true`. Decoded blocks are immutable and content-addressed; retained versions share unchanged blocks. A partial tensor update adds only changed blocks. Checkpoint construction streams those blocks into standard files. Frozen source snapshots and standard checkpoints still contain complete weights and need separate temporary-space budgets. The persisted format marker also supports a transport process opened before the first publication.

Each manifest's `source` record identifies the run, publication generation, capture sequence, rollout, optimizer attempts and successful updates. Initialization and the first update have distinct update IDs. Skipped updates or unchanged BF16 content may share a content root while retaining distinct provenance. Source ranks exchange waiting status while storage publication runs, separating storage latency from a single collective timeout. Publication retries preserve the candidate identity and increment a durable `attempt` field separately. Optimizer counts describe the current training process, separately from the global step in a model checkpoint.

Consumers on one node share checkpoint-cache locks during preparation. With the same `node_budget`, each group reserves its preparation and installation resources; groups progress concurrently when the combined demand fits, while each group remains serial. Without a node budget, shared directories and the node installation lock retain serial execution.

Optional `node_budget` configuration adds byte reservations: `limits` describes node capacities, `resident` includes models, KV cache and CUDA Graph allocations, and `operation` describes the measured additional peak. Each is a mapping from a resource name such as `host` or a node-level GPU identifier to bytes. Consumers share identical limits and resident allocations. Interrupted installations retain reservations until their owner completes device recovery.

Standalone consumers set `node_directory` to the same shared directory on a node for installation locks and reservations. Its default is the cache parent directory. Capacity failures preserve the installed version and record required capacity, current limits and recovery actions. Recover the retained installation or explicitly expand capacity, then retry the same target.

Unchanged chunks use COPY. For changed chunks, the default codec compares SPARSE and RAW using payload bytes plus their encoded metadata in the manifest, preserving BF16 bit patterns. `compression: true` additionally compares zlib and XOR-zlib. Ties prefer fewer decoding steps. Profile complete frozen sequences to weigh encoding time against transfer savings. New on-demand FULL recovery packages inherit the publication’s compression choice, including when its DELTA contains only COPY chunks. Legacy manifests without this policy use raw blocks. Existing packages are repaired using their recorded encoding to preserve artifact identity.

| State or result | Meaning and action |
| --- | --- |
| Publication handle | The candidate was accepted; encoding may still be running. |
| `STORE_COMMITTED` | Content is durable and discoverable; source snapshot confirmation follows. |
| `SOURCE_CONFIRMED` | Training confirmed its snapshot base and returns PublishedVersion. |
| `ACTIVE` | Installation and the generation probe completed. |
| `SUSPENDED` | Inference memory was released. Resume memory and reinstall the acknowledged version. |
| `WAITING_ACTION` | Inspect the recorded stage, target version, error and recovery action. |

A missing object or failed content check triggers one same-target FULL request within the current operation deadline. Explicit FULL requests remain available for recovery. The owner rebuilds damaged objects of an existing FULL package from retained canonical content. Package identity may differ; reconstructed content has the same root.

Model configuration, logical parameter inventory and dtype are checked before closing admission. Installation commands bind the installation, worker identities and operation content. After loading, verification and the generation probe, the commit decision and its checkpoint are persisted together. Recovery after commit restores that checkpoint; recovery before commit can restore the previous active checkpoint. Consumption acknowledgement is idempotent.

An `all_versions` subscription retains unacknowledged history. Acknowledgement follows ACTIVE and is idempotent. If retained data exhausts the quota, publication returns a capacity error; increase capacity or explicitly retire unused subscriptions before retrying.

With `mode="latest"`, each rollout boundary fixes the newest target and installs it directly, reconstructing all required dependencies. This subscription retains its acknowledged version and dependencies; the ordinary anchor policy retains recent history. Use `all_versions` to observe every training update. A subscription keeps its mode; create a new consumer identity when changing modes.

The consumer's `operation_timeout` (seconds, default 1800) covers preparation, installation and acknowledgement for one target. Each target in a catch-up range receives its own deadline. TCP requests, worker RPCs, generation probes and file-block boundaries share the remaining time. A preparation timeout leaves the previous engine active and retains verified cached objects for retry. An installation timeout closes admission and records the target and recovery actions. Retention release and admission closure have a separate cleanup budget of up to 5 seconds. Filesystem and device calls return control when the underlying runtime completes or reports an error; restart an unresponsive engine and restore its retained checkpoint as directed by the operation status.

### Source capacity and accounting

Optional publication `source_limits` sets `ram_bytes` per source host, `gpu_bytes` per source rank and `scratch_bytes` for the aggregate source workspace across all ranks. Admission runs before snapshot cloning. It combines frozen tensors, the largest conversion group, codec windows, shard bases, replacement frames, logical exports, publication candidates and Bridge gathered-source caches, including retained files and store occupancy.

The Bridge gathered-source cache uses actual Megatron shapes and TP shard counts. Logical exports reserve the checkpoint limit separately, accounting for differences introduced by vocabulary padding, cropping and tied-weight conversion.

Available RAM and GPU memory are instantaneous observations. An estimated shortfall produces a warning and publication continues; configured source limits and storage budgets remain enforced. Actual allocation or I/O failures report the affected phase and retain the target for retry. Optional capacity and source-route diagnostics use the bounded diagnostic writer.

Host checks include available RAM and ancestor Slurm/cgroup limits. A GPU shortfall first releases unused process-local PyTorch cache and rechecks driver availability. Scratch and publication writes on the same disk or shared mount are combined. `scratch/capacity-*.json` records demand, availability and recovery actions. After releasing unused caches or increasing capacity, retry the same update.

The RAM estimate includes clean inactive disk-backed cache that the kernel can reclaim. Shared memory, active cache, dirty pages and pending writeback retain their occupied capacity. Each ancestor cgroup limit and host `MemAvailable` caps the estimate; missing cache statistics use the unadjusted headroom.

Source `metrics-v*-rank-*.json` and `publisher-metrics-*.json` retain gather, file-write, background publication, control and retry events. `aggregate_events` deduplicates event IDs and keeps collective payloads, file writes and socket bytes separate. Retry events remain; repeated collection of a receipt is idempotent.

Consumer `delivery_stage` selects `X` (default, first target node) or `T` (other target nodes); consumers sharing a node use the same assignment. Managed nodes sharing one configuration can set `first_target_host`; other nodes are assigned T. TCP object and manifest bodies belong to delivery; frame headers and commands belong to C. Failed preparation retains completed transfers in `checkpoint_root/failed-metrics/`. Reconcile each direction with the server socket ledger. HTTP installation control records serialized request and response bodies, including error responses, in `audit/control-metrics-*.json` using the separate `http_body` scope.

Loader observations retain process I/O, actual storage type and NFS client-mount RPC counters. Set `RELAX_WEIGHT_METRIC_INTERFACES` to verified comma-separated interface names for node counters. Mount and interface observations include concurrent traffic and need isolated calibration intervals and storage-side records. Application bytes, file writes, local CPU/GPU movement and network sends remain separate views. Complete comparisons require the same sequence, measurement layer and evidence-backed G/W/X/T/C bounds.

Observations include host, process, interval endpoints and original counters so overlapping observations of the same mount can be reconciled. Memory-backed scratch and storage also consume the host RAM budget. Add `serve --metrics-output /path/to/service-metrics.json` to record the TCP service's socket counters and writes made while constructing FULL repair packages. The file is saved on orderly shutdown; an abrupt service exit requires the external link observations for the missing interval.

## Integration and verification

The worker extension targets SGLang 0.5.17 through the plugin registered when installing this checkout. The physical verifier covers unquantized Qwen3 / Qwen3-VL Dense and MoE layouts. The installation adapter uses one node. Small-model GPU regressions cover Dense TP2, PP2 and MoE TP2/EP2, comparing FULL and DELTA installation, image-text generation and recovery over the same six versions. Expert parallelism uses static placement. Dynamic expert placement and quantized layouts require corresponding verification adapters.

Start the engine with the checkpoint's model configuration and dtype. Configuration changes require a new engine. Each worker verifies device parameters in bounded row windows against the local checkpoint's expected layout. The coordinator checks complete member and logical coverage, including tied embedding/head aliases. Device dumps remain an independent reference for integration tests.

CPU regressions cover version reconstruction, publication retries, consumer cursors, installation recovery and communication accounting:

```bash
python -m pytest -q tests/cpu/distributed/weight_sync
python -m pytest -q tests/cpu/backends/sglang/test_weight_version_verification.py
```

The [FULL checkpoint guide](./weight-checkpoints.md) provides GPU entrypoints for real training export and an independent Loader. End-to-end validation uses the same frozen training sequence to measure gathering, publication writes, reads, target distribution and control traffic, alongside synchronization time, inference pauses and throughput.

### Diagnostics and operation status

Memory sampling, I/O observations and independent diagnostic files use bounded waits. Each observation waits at most 0.2 seconds by default; delayed or failed observations produce warnings and coverage errors while weight updates proceed. Diagnostic worker counts are bounded, including when storage stalls. `memory_sampling.status` and `errors` describe coverage, and existing trace files are preserved. Complete performance acceptance requires the corresponding observation records.

Generation returns 503 during installation, with `status`, `reason`, `closed_seconds` and `action`. Control failures expose a checkpoint recovery action. Posting `{"action":"status"}` to `/relax_weight_version` queries the current phase, generation availability and recovery information independently of the installation control lock. The standalone consumer periodically reports the target version and remaining time while waiting for publication.

Weight integrity, version identity, installation receipts and actual capacity remain part of the operation. Running file writes retain their inputs until completion; long waits report the stage and inspection guidance. Diagnostic files are separate from these durable protocol records.

Status queries return the existing consumer state, or NOT_INITIALIZED before creation, without creating a session or pausing the engine. An active, acknowledged version returns success; checkpoint cleanup failures are reported as cleanup=PENDING and retried on subsequent consumption. operation.json is a display record; delayed or failed writes do not change installation receipts or subscription cursors.

Version mode adds its required plugin while spawning the engine, preserves existing plugins, and restores the caller's environment afterward. SGLang 0.5.17 is the integration baseline; compatibility checks use actual interfaces and weight layouts. Native pause, continue and LoRA endpoints retain engine behavior. Conflicting operations during installation report the reason; base-weight updates use the installation adapter.

The standalone consumer's wait_seconds bounds the wait for each target publication and accepts any finite nonnegative value. Worker commands, generation probes and file locks use the current operation's remaining time. With node_budget, independent groups may progress concurrently when their summed reservations fit; installation remains serial within each group. Without that configuration, node-wide serial installation is retained.
