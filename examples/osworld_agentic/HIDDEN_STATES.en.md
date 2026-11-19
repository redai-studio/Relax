# OSWorld decision hidden states

[中文](HIDDEN_STATES.md)

Collection is off by default. Set `RELAX_DECISION_HIDDEN_DIR` to collect decision representations from the actual rollout policy's existing generation forward. There is no training forward hook, replay, or additional model forward. Inputs, image windows, numerical sampling settings, action parsing, rewards, and training samples remain unchanged. Reasoning and action text are pooled together.

## Representation and alignment

Given prompt `p` and generated tokens `y1, y2, …, yn`, the vector predicting `y1` comes from the last prompt position in prefill. The vector predicting `yi` comes from the decode position consuming `yi-1`. Each vector is taken **after the final normalization and before the LM head**. It is not the representation obtained after feeding `yi` into the model.

Only rows used to sample new tokens are selected. Other prompt, history, and image positions do not contribute. EOS, additional model termination IDs, and request stop token IDs are excluded. Reasoning/tool delimiters are not excluded merely because the tokenizer considers them special tokens. Ordinary tokens forming a string stop remain included. GPU FP32 sum/count accumulation exports one mean and count at completion, never all token vectors. A zero count produces a zero vector; exclude these records from analysis.

## Supported configuration and setup

The current installation's package metadata and Git tag report SGLang **0.5.9**, HEAD `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`, but its source has extensive local modifications. Repository Docker defaults use **0.5.17**. The implementation checks both package version and SHA256 of four critical source files. **Only the inspected local source snapshot is accepted**; neither stock 0.5.9 nor Docker 0.5.17 is claimed to work. Review interfaces and alignment before updating these checks for another snapshot; do not simply remove them.

The existing `return_hidden_states` interface transfers per-token vectors to CPU and can prefer pre-normalization states. Instead, the existing scheduler subprocess entry installs two scoped hooks: LogitsProcessor retains post-norm sampling rows, and the scheduler pools and saves them. Existing `custom_params` carries task/session/turn metadata without changing numerical sampling settings.

Only `Qwen3VLForConditionalGeneration`, ordinary autoregressive generation, eager execution, a non-overlap scheduler, and nonstreaming backend requests are supported. Ordinary TP is supported: this model's final hidden dimension is replicated across TP ranks, and only TP rank 0 pools and writes. PP and DP must both be 1. Speculative decoding, LoRA, DP attention, torch compile, prefill/decode disaggregation, grammar, `return_hidden_states`, partial rollout, fully async rollout, and agentic prelaunch are rejected. Independent engines can each write their own shards.

1. Use the matching engine source and this Relax checkout. The engine, SessionShard, API, and agent processes must all receive the variable. Use a persistent shared directory unique to the experiment, or collect all node-local shards before analysis. Setting the variable only on the agent is insufficient.

2. In `scripts/run-qwen3vl-8B-osworld-8xgpu.sh`, remove `--agentic-prelaunch` and add `--sglang-disable-cuda-graph --sglang-disable-overlap-schedule` to `SGLANG_ARGS` for the collection run. Preserve the original configuration for ordinary runs with collection disabled.

3. Pass the switch using the existing Ray runtime environment and submit training as usual. Merge this entry into existing `RUNTIME_ENV_JSON.env_vars`:

   ```json
   {"RELAX_DECISION_HIDDEN_DIR": "/path/to/run/decision-hidden"}
   ```

   For example, when no other runtime environment configuration is needed:

   ```bash
   export RUNTIME_ENV_JSON='{"env_vars":{"RELAX_DECISION_HIDDEN_DIR":"/path/to/run/decision-hidden"}}'
   ```

Unset the variable to disable collection. No hooks are installed, no extra hidden tensors are retained, and no collection files are created. The agent Python environment must be able to import `relax.agentic.decision_hidden` from this checkout.

## Files and joins

The existing success archive removes failures and entries exceeding quotas. Research data therefore uses independent process shards, without a database, service, queue, or shared-file write lock:

```text
decision-hidden/
  requests-<worker_uuid>-<pid>.jsonl
  outcomes-<worker_uuid>-<pid>.jsonl
```

Each engine worker owns its requests file; each agent process owns its outcomes file. Requests are saved at completion regardless of HTTP response delivery. This example is shortened to three dimensions and is not actual model output:

```json
{"task_id":"task-a","session_id":"session-a","trajectory_id":"session-a","turn_id":0,"backend_request_id":"req_session-a_0:0","request_id":"req_session-a_0:0:unique-generation-id","policy_version":"engine-instance:7","checkpoint":"/path/to/checkpoint","vector":[0.1,0.2,0.3],"token_count":12,"hidden_dim":3,"layer":"final_norm","semantics":"mean_final_norm_pre_lm_head_predicting_generated_tokens_v1","finish_reason":{"type":"length","length":12}}
{"task_id":"task-a","session_id":"session-a","reward":0.0,"success":false,"evaluation_error":null,"stop_reason":"env_done"}
```

`turn_id` is the OSWorld action turn, shared by parsing retries. `backend_request_id` is Relax's stable request ID. `request_id` adds a generation instance UUID, so even an HTTP retry that executes the same backend ID has a distinct record. `trajectory_id` uses the resident session ID, unlike the success archive's independent random trajectory ID; join that archive through `session_id`.

`policy_version` combines an engine instance UUID and a monotonically increasing revision tied to successful weight mutations. Disk loads, tensor/distributed/IPC updates, and weight postprocessing advance it. It is neither a training global step nor a content hash. Chunked updates may advance it multiple times; replicas with identical weights still have different engine instance IDs. `checkpoint` is the initial or most recent disk load path, not a claim that this path contains subsequent online updates. A revision change within a request raises an error rather than mixing versions.

The outcome preserves the original OSWorld evaluator reward. `success` means reward equals 1 with no evaluator error; evaluator errors produce null success. Cancellation or process failure before evaluation can leave requests without an outcome. Do not interpret a missing outcome as failure. Filesystem errors are surfaced instead of silently losing records. A forcibly terminated process may leave a partial final line; inspect files before reading.

Join the files offline:

```bash
python examples/osworld_agentic/scripts/join_decision_hidden.py /path/to/run/decision-hidden > joined.jsonl
```

Or read vectors directly:

```python
from pathlib import Path
from examples.osworld_agentic.scripts.join_decision_hidden import joined_records

for row in joined_records(Path("/path/to/run/decision-hidden")):
    if row["outcome_available"] and row["success"] is not None and row["token_count"] > 0:
        vector, success, reward = row["vector"], row["success"], row["reward"]
```

This feature only collects and joins data; it does not cluster embeddings, train diagnostic models, or synthesize tasks.
