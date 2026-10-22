# CUA successful trajectory archive

Disabled by default. Each complete task attempt is one candidate trajectory. OSWorld captures actual model requests, environment actions and evaluator evidence independently of rewards, advantages, sample filtering and SessionOutput. Other recipes create no archive, threads or extra I/O automatically.

## Deployment and enablement

One standalone HTTP process owns the archive. Agents on all training nodes connect to that process; its budget covers the **whole library**, not each node. No Ray service or training argument parser changes are required. The writer does not create sandboxes or call models.

Choose a dedicated directory on local persistent storage on a host reachable by all Agents. Substitute actual paths and addresses:

```bash
export CUA_ARCHIVE_TOKEN='replace-with-a-private-token'
python -m relax.agentic.archive.server \
  --archive-dir /local-disk/cua-library \
  --max-bytes 10737418240 --per-task-limit 5 \
  --host 0.0.0.0 --port 8765
```

The required byte budget above is 10 GiB; the per-task limit defaults to 5. A non-loopback listener requires a token. Use HTTP only within a trusted private network, or put an HTTPS reverse proxy in front. Do not commit credentials or include them in logs.

Add these settings to OSWorld's private `env.local.sh`, readable on every Agent node:

```bash
export CUA_ARCHIVE_ENABLE=1
export CUA_ARCHIVE_URL='http://actual-archive-host:8765'
export CUA_ARCHIVE_TOKEN='same-private-token-as-the-writer'
```

Use the existing OSWorld training entrypoint. `run_agent_app.sh` already reads that file; no credentials need to enter logged Ray runtime-env JSON. Unset `CUA_ARCHIVE_ENABLE` or set it to `0` to disable capture.

The server serializes requests and reserves event and new asset bytes before writing. Client RPCs have a five-second timeout and no automatic retries. Connection failures, unsupported data or disk errors abandon that candidate with a logged reason without changing training results. Synchronous capture adds boundary latency. Ingress is bounded to min(twice the library budget, 64 MiB) per request; larger candidates are rejected explicitly.

Supported storage: Linux local ext2/3/4, XFS, Btrfs, tmpfs and overlay. Startup checks mountinfo and rejects NFS, shared FUSE mounts and unknown types. tmpfs and ephemeral container layers are suitable for tests, not durable archives. Multiple nodes must not open the same directory directly. Separate servers on separate nodes have separate budgets, never a global quota.

## Records, admission and capacity

The public implementation in `relax/agentic/archive/` imports no OSWorld code. `app/archive.py` provides the task adapter.

- `archive.json`: schema, byte budget and task quota; restart cannot silently change them.
- `staging/<trajectory_id>.jsonl`: incrementally written trajectory, observation, model_call, call_result/call_parse, action/action_result and outcome events. Incomplete files are never exported.
- `trajectories/<trajectory_id>.jsonl`: validated candidates atomically renamed from staging into the retained library.
- `assets/<sha256>`: unique image bytes. Filenames have no extension; PIL detects the format from content.

Stable run, Session, trajectory, observation, call and action IDs preserve associations. A step may contain several calls, and a call may lead to several ordered actions. Actual request payloads and raw HTTP replies are recorded. Observations include screenshots and fields returned by the environment. Inline model images are externalized, including resized/cropped inputs and historical images, rather than replaced with desktop screenshots. `asset://<hash>` can be reconstructed as bytes or a data URI. `observation_ids` identifies the environment observation at request time; `model_input_asset_ids` enumerates actual input resources. Unknown crop-to-original mappings are not invented.

Task quotas use `(task_source, task_version, task_id)`; OSWorld task versions hash the full task configuration. Admission requires explicit success, no evaluation error, complete associations and verified resources. The public layer has no reward threshold. The OSWorld archive conservatively maps evaluator `1` to success, `0` to failure, `(0,1)` to partial, and other/non-finite/missing/error results to unknown. This does not alter training rewards or success metrics. Advantages are never used for admission.

Exact behavior deduplication hashes the task key, ordered action types and full parameters, before/after screenshot hashes and observation fields. It excludes record IDs, timestamps, scores and reasoning text. For the same path, retain the better record. Deterministic priority is: fewer known execution errors, fewer environment steps, fewer actions, fewer model calls, fewer actions with unknown execution status, then trajectory ID. There is no semantic similarity model.

The budget counts logical bytes of every file inside the library, including retained records, staging, unique assets and configuration. It does not measure allocation blocks, filesystem metadata or free disk space. Repeated images have one file. Every serialized write reserves all additional event and image bytes before mutation.

Under pressure, evict lower-priority extras from tasks with multiple retained trajectories, favoring task coverage. Never evict a task's last representative or another live staging candidate. Reject the candidate if capacity remains insufficient and log the reason. GC checks both staging and retained references, keeping shared assets. Consequently, a full archive with only unique representatives or active staging may reject a new task. Evictions already performed are not rolled back if a subsequent candidate write fails. ENOSPC is handled separately.

## Recovery

A local exclusive owner lock prevents simultaneous servers, recovery or offline export. Restart keeps staging charged against capacity; no heartbeat timeout is used to guess that a Session is dead. Only after **all Agents, training and the archive server have stopped**, explicitly clean abandoned candidates:

```bash
python -m relax.agentic.archive.export recover \
  --archive-dir /local-disk/cua-library --confirm-no-active-sessions
```

This command acquires the owner lock, removes only this library's staging and collects unreferenced assets. The flag is the operator's confirmation that no Agents remain active, including temporarily idle ones. Export ignores staging even without cleanup.

## Offline export and SFT

Stop the archive server after training, then run:

```bash
python -m relax.agentic.archive.export export \
  --archive-dir /local-disk/cua-library --output-dir /data/cua-export-001
```

The output directory must not exist. Export holds exclusive ownership throughout, preventing eviction while copying assets. It writes `manifest.json`, `trajectories.jsonl`, `samples.jsonl` and the referenced `assets/`. The manifest includes schema/export versions, sources, selected/skipped/duplicate/missing counts, reasons and supervision rules. A failed partial export has no manifest; choose a fresh directory instead of overwriting it.

Full exports retain retries and corrections. SFT conservatively excludes entire trajectories with unselected/unconfirmed parsing calls or known execution failures. Targets must have actual associated actions; harness-forced terminal actions are not targets. OSWorld records successful RPC return without semantic confirmation as unknown + rpc_completed. Final evaluator success, no known execution failures and no retries can qualify those calls for SFT without claiming each action was semantically verified.

`samples.jsonl` uses Relax's existing messages format: historical messages have `learn=false`, only the current final assistant reply has `learn=true`. Context content, tools and `metadata.apply_chat_template_kwargs` are preserved. Samples needing unsupported loader fields such as `tool_call_id/name` or the developer role are skipped with counts; full trajectories retain them.

Images use portable paths relative to the export directory. The existing loader resolves paths from process CWD. After placing the export at a path shared by all SFT nodes, generate a deployment-local absolute-path version:

```bash
python -m relax.agentic.archive.export convert-sft \
  --samples /data/cua-export-001/samples.jsonl \
  --output /data/cua-export-001/sft.local.jsonl
```

Use these data options in an existing SFT launch script appropriate for the image model:

```bash
--loss-type sft \
--prompt-data /data/cua-export-001/sft.local.jsonl \
--input-key messages --metadata-key metadata --tool-key tools \
--multimodal-keys '{"image":"images"}'
```

Do not set `--label-key` or inject an extra system prompt. Inline image_url parts provide the images, so no duplicate images column mapping is needed. Reconvert paths after moving the export. Select a matching processor, chat template and resource configuration yourself; no automatic training feedback is implemented.

## Validation and external limits

Tests use temporary files, mock models/environments and local HTTP to cover capacity, deduplication, recovery, multiprocess writes, completeness, interrupted writes, asset protection during eviction and CLI export. Export requires a complete commit record; damaged records are skipped and counted. CLI tests confirm that an active owner lock prevents export. The existing SFT loader test confirms supervision only on the current reply. No real sandboxes, multi-node training or model training are run.

The external OSWorld fork is not implemented here. Environment and policy weight versions remain unknown unless reliable metadata exists; DesktopEnv/Nex properties are not guessed. Result files are marked unavailable when no reliable accessor exists. Actual request image bytes are retained, not processor tensors. Remote/local-file model images are rejected rather than downloaded later because their original bytes cannot be established. Fine-grained execution failures not exposed by the environment cannot be inferred from normal RPC return.
