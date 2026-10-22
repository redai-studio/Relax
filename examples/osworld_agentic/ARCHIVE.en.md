# CUA successful trajectory and image storage

The OSWorld training script saves complete successful trajectories and images directly under each run's output directory by default. No standalone writer, URL or token is required. Capture does not change rewards, advantages, sample filtering or SessionOutput. Other recipes do not enable capture automatically.

## Training options

After configuring MODEL_DIR, DATA_DIR, SAVE_DIR and private Nex settings, run:

```bash
bash examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh \
  --save-trajectories \
  --trajectory-max-bytes 10737418240 \
  --trajectory-per-task-limit 5 \
  --trajectory-task-limits /path/to/task-limits.json
```

Saving is enabled by default; use `--no-save-trajectories` to disable it. The defaults are 10 GiB and at most 5 trajectories per task. The optional override file is a JSON object such as `{"task-id-a": 2, "task-id-b": 0}`. Keys are actual task IDs and values are nonnegative integers; 0 skips capture for that task. The default quota must be at least 1 and the budget at least 1024 bytes. These options belong to the OSWorld script, not the framework training parser.

Each run has this layout:

```text
rollout_result/<run_tag>/
├── train/*.jsonl
├── reward_dump.jsonl
└── sessions/
    ├── archive.json
    ├── staging/<trajectory_id>.jsonl
    ├── trajectories/<trajectory_id>.jsonl
    └── assets/<task_id>/<trajectory_id>/
        ├── screenshots/000000-<sha256>.png
        ├── screenshots/000001-<sha256>.png
        └── model_inputs/000000/000000-<sha256>.png
```

The trajectory header contains `session_id`, `task_id` and the complete task configuration, linking it to `reward_dump.jsonl`. New runs group images by task and trajectory, with image extensions so they can be opened directly. Special task ID characters are escaped into safe directory names; unusually long or special IDs use hashed directory names. The original ID remains in the trajectory header.

The six-digit sequence in `screenshots/` follows observation order: `000000` is the initial screenshot, followed by observations after actual actions. A single model turn may execute several actions, so screenshot indices are not environment turn numbers. Action events still link before/after images through `before_observation_id` and `after_observation_id`; `outcome.final_observation_id` identifies the final screenshot.

`model_inputs/000000/` contains images for the first actual model request. Later requests, parsing retries and transport retries receive successive indices. Images within a request follow traversal order, including historical images; this is not desktop observation order. These are the actual cropped or resized inputs, not replacement desktop screenshots. Original bytes are preserved, with extensions detected from the image format: `.png` for PNG and `.jpg` for JPEG. Images attached to other event types use `events/<event_index>/`.

The `asset_layout` field in `archive.json` records the layout. New runs use `task`; existing libraries without this field retain the legacy `assets/<sha256>` layout and remain readable and exportable. Historical images are not rearranged during a running experiment.

**Long-term retention still requires a complete trajectory with evaluator score exactly 1.** Failed, partial, incomplete, duplicate or lower-priority candidates are removed; their temporary images are not a permanent screenshot log. Changes apply to new launches. Images from historical candidates whose capture was abandoned cannot be recovered this way.

A single node can use a local output directory. Multiple nodes must access the same directory at the same path, on a filesystem with reliable cross-node `flock`, atomic rename and close-to-open visibility. Separate node-local directories do not provide a global budget. Each write briefly holds the common `.owner.lock` while checking total usage, deduplication and task quotas; model calls hold no archive lock. Lock acquisition times out after 30 seconds. Lock timeouts, disk errors and unsupported images abandon the candidate without stopping training. The lock deadline does not bound filesystem I/O itself. Synchronous shared-filesystem writes add latency; assess it for the actual concurrency and filesystem.

Budget and quotas are fixed in each run's `archive.json`. Agents sharing that directory must use identical settings; do not change them while running. Each new run_tag has its own directory and budget, excluding other runs, training JSONL and manually exported copies. Launcher settings take precedence when Agents reload an older `env.local.sh`; old writer URLs and tokens are unused in directory mode.

## Standalone HTTP writer compatibility

The public HTTP API remains available, but the official OSWorld training script selects directory mode. Custom Agent invocations can explicitly set `CUA_ARCHIVE_BACKEND=http`, `CUA_ARCHIVE_ENABLE=1`, `CUA_ARCHIVE_URL` and `CUA_ARCHIVE_TOKEN`. The HTTP writer still requires local Linux ext2/3/4, XFS, Btrfs, tmpfs or overlay storage, and rejects shared storage. Non-loopback listeners require a token.

```bash
python -m relax.agentic.archive.server \
  --archive-dir /local-disk/cua-library \
  --max-bytes 10737418240 --per-task-limit 5 \
  --host 0.0.0.0 --port 8765
```

Supply the token through private configuration before starting. HTTP requests time out after five seconds without retries. Each request is limited to min(twice the budget, 64 MiB). Do not use an HTTP writer and direct Agent writes on the same library concurrently.

## Records, admission and capacity

The public implementation in `relax/agentic/archive/` imports no OSWorld code. `app/archive.py` provides the task adapter.

- `archive.json`: schema, byte budget and task quota; restart cannot silently change them.
- `staging/<trajectory_id>.jsonl`: incrementally written trajectory, observation, model_call, call_result/call_parse, action/action_result and outcome events. Incomplete files are never exported.
- `trajectories/<trajectory_id>.jsonl`: validated candidates atomically renamed from staging into the retained library.
- `assets/<task_id>/<trajectory_id>/`: ordered screenshots and model input images for new training runs. Existing libraries and default HTTP libraries still use `assets/<sha256>`.

Stable run, Session, trajectory, observation, call and action IDs preserve associations. A step may contain several calls, and a call may lead to several ordered actions. Actual request payloads and raw HTTP replies are recorded. Observations include screenshots and fields returned by the environment. Inline model images are externalized, including resized/cropped inputs and historical images, rather than replaced with desktop screenshots. `asset://<relative_path>` identifies an image under `assets/`, which can be reconstructed as bytes or a data URI. Legacy hash references remain supported. `observation_ids` identifies the environment observation at request time; `model_input_asset_ids` enumerates actual input resources. Unknown crop-to-original mappings are not invented.

Task quotas use `(task_source, task_version, task_id)`; OSWorld task versions hash the full task configuration. Admission requires explicit success, no evaluation error, complete associations and verified resources. The public layer has no reward threshold. The OSWorld archive conservatively maps evaluator `1` to success, `0` to failure, `(0,1)` to partial, and other/non-finite/missing/error results to unknown. This does not alter training rewards or success metrics. Advantages are never used for admission.

Exact behavior deduplication hashes the task key, ordered action types and full parameters, before/after screenshot hashes and observation fields. It excludes record IDs, timestamps, scores and reasoning text. For the same path, retain the better record. Deterministic priority is: fewer known execution errors, fewer environment steps, fewer actions, fewer model calls, fewer actions with unknown execution status, then trajectory ID. There is no semantic similarity model.

The budget counts logical bytes of every file inside the library, including retained records, staging, image files and configuration. It does not measure allocation blocks, filesystem metadata or free disk space. The new layout retains an ordered file for each image occurrence rather than sharing files across trajectories; all copies count toward the budget. The legacy flat layout still deduplicates image files by content. Complete trajectory deduplication still compares content hashes and is independent of task directories, trajectory IDs and file indices. Every serialized write reserves all additional event and image bytes before mutation.

Under pressure, evict lower-priority extras from tasks with multiple retained trajectories, favoring task coverage. Never evict a task's last representative or another live staging candidate. Reject the candidate if capacity remains insufficient and log the reason. GC checks both staging and retained references, keeping shared assets. Consequently, a full archive with only unique representatives or active staging may reject a new task. Evictions already performed are not rolled back if a subsequent candidate write fails. ENOSPC is handled separately.

## Recovery

Direct writes hold the owner lock during each operation; the HTTP writer holds it for its lifetime. Offline export holds the lock for the complete snapshot and blocks Agent writes, so do not export during training. Restart leaves staging charged against capacity without guessing that an inactive Session is dead. Stop training, all Agents and any HTTP writer before explicit cleanup. Replace the example paths below with actual paths:

```bash
python -m relax.agentic.archive.export recover \
  --archive-dir /data/outputs/rollout_result/example-run/sessions \
  --shared-filesystem --confirm-no-active-sessions
```

This command acquires the owner lock, removes only this library's staging and collects unreferenced assets. The flag is the operator's confirmation that no Agents remain active, including temporarily idle ones. Export ignores staging even without cleanup.

## Offline export and SFT

Stop the archive server after training, then run:

```bash
python -m relax.agentic.archive.export export \
  --archive-dir /data/outputs/rollout_result/example-run/sessions \
  --shared-filesystem --output-dir /data/cua-export-001
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
