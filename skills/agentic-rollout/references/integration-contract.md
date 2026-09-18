# Integration contract

Contents: [runtime path](#choose-the-runtime-path), [environment](#agent-environment),
[dataset](#dataset-mapping), [request/parser](#model-request-contract), [export](#export-contract),
[multimodal](#multimodal-boundary), [validation](#integration-validation).

## Choose the runtime path

Use resident Agentic rollout for an external agent application that owns its model/tool loop:

```text
--use-agentic-rollout
--agent-command <adapter command>
--agent-cwd <working directory>
```

`--use-agentic-rollout` unconditionally selects Agentic train and Eval generation functions. A configured
`--custom-generate-function-path` is ignored; use the legacy path instead when Relax-side custom code must own the loop.

## Agent environment

Relax provides:

| Variable | Contract |
| --- | --- |
| `RELAX_INPUT_JSON` | Session input path; payload may provide messages, metadata, or both |
| `RELAX_OUTPUT_JSON` | Optional terminal output and explicit export records |
| `RELAX_SESSION_IO_DIR` | Per-Session temporary directory owned and removed by the agent process runtime |
| `RELAX_BASE_URL` | Base URL for the supported model API endpoints |
| `RELAX_SESSION_ID` | Session route token and API credential |
| `RELAX_ROLLOUT_MODE` | `train` or `eval`, for mode-specific agent/export behavior |
| `RELAX_GROUP_ID` | Logical Runtime Group identifier |

The adapter must preserve the agent's normal task lifecycle, write terminal output atomically, and exit after cleanup.
The command runs through `/bin/bash -lc`; inspect login-profile behavior if it may replace injected environment values.

## Dataset mapping

`--input-key`, `--metadata-key`, and `--multimodal-keys` map dataset fields into Samples. The harness may receive
ready-to-use messages, or read task data from metadata and construct messages at runtime. Every Session retains its own
metadata and logical indices. Eval resolves input and metadata keys per dataset.

## Model request contract

Choose the interface used by the actual agent client and send its complete history on every request:

| Interface | Endpoint | Complete-history input |
| --- | --- | --- |
| OpenAI Chat Completions | `/v1/chat/completions` | `messages` |
| OpenAI Responses | `/v1/responses` | typed `input` Items |
| Anthropic Messages | `/v1/messages` | `system + messages` |

Every protocol is projected to canonical `messages`, nested function `tools`, and `chat_template_kwargs` before
SessionForest matching. Reuse stable tools and applicable template arguments for a continuous lineage. Preserve
assistant reasoning, tool calls, call IDs, and tool results exactly as observed by the model. Responses
`previous_response_id` is not a continuation handle; replay the complete typed `input`.

Inspect the actual HTTP payload. Canonical history accepts `user`, `assistant`, `tool`, and `system`. Chat clients must
convert `developer` to `system`; Responses projection performs that conversion. User, system, and tool messages require
present, non-`null`, non-zero-length content. Assistant may omit text content when the message carries tool calls or
nonempty reasoning. When a tool returns `None` or `""`, choose a stable nonempty representation and keep it in every
later turn and export.

Chat request `chat_template_kwargs` must not set `add_generation_prompt`, `tokenize`, or `tools`. Send tools through the
top-level `tools` field. Any adapter-side canonicalization becomes the SessionForest history and must also be used by
explicit export.

Verify the endpoint contract, supported Items/blocks, and transport behavior in
`relax/agentic/session/service.py`; the client library's full schema does not extend the Relax contract.

Omitting `stream` or setting it to `false` returns JSON. `stream=true` returns Buffered SSE: connection frames and
heartbeats may arrive while generation is pending, then all model-content terminal events arrive after complete
generation. A successful Chat stream has one terminal chunk containing complete usage followed by `[DONE]`, independent
of `stream_options.include_usage`. This transport does not change normalization, SessionForest state, sampling, or
export.

## Reasoning and tool-call parser contract

Determine parser requirements from the exact model output and chat template. `--agentic-reasoning-parser` parses the
decoded response first; `--agentic-tool-call-parser` then parses the remaining text and runs only when the request's
lineage contains `tools`. Verify that both parser names exist in the installed SGLang version. A parser name from a
different model recipe is not compatibility evidence.

Relax commits this parsed canonical assistant message to SessionForest, then renders it in the request protocol:

```text
content
optional reasoning_content
optional tool_calls with stable call IDs, names, and arguments
finish_reason = tool_calls | stop | length
```

A missing or incompatible parser can leave reasoning or tool syntax inside plain `content`, remove ordinary answer
text, produce malformed arguments, or prevent the agent's tool loop from continuing. Capture one raw decoded model
response and its protocol-rendered Relax response; test both a real tool-call turn and a non-tool final-answer turn when
the model can produce both.

The model client created inside the agent application is the Relax-facing agent client. Configure the timeout of each
model API request for the full lifetime of that request. Prelaunch can hold the first request before lease;
partial rollout and fully async can retain a request across runtime boundaries. `--agent-timeout` separately limits the
agent process's Runtime active time to contain agent-side hangs. Use
[runtime-operations.md](runtime-operations.md) for the decision rule.

## Export contract

Use implicit export only after verifying that the model-visible history is strictly linear. Intentional forks, multiple
agents, deleted thinking, changing tools, or any other nonlinear history require explicit export.

One explicit record may be a JSON object containing `messages`; several records use JSONL. A single JSON object without
`messages` is an implicit output payload that only contributes top-level metadata/reward. Every JSONL explicit record
must include `messages`. JSON arrays are rejected.

Each explicit record needs:

```text
unique name
complete messages
exact tools when used
exact chat_template_kwargs when used
metadata needed for credit
optional reward
```

Explicit records always use canonical Chat-shaped `messages`, nested function `tools`, and `chat_template_kwargs`,
including for Sessions whose agent used Responses or Messages.

Every record must resolve to a committed SessionForest state. A JSON array is not the explicit export format. Keep task outcome reporting separate from per-context training credit.

## Multimodal boundary

Initial dataset media follows the normal Relax placeholder and `--multimodal-keys` path. Agent-added training media
currently supports images in this exact shape:

```json
{"type": "image_url", "image_url": {"url": "..."}}
```

Agent-added audio and video are not exported as training multimodal inputs. Verify processor-expanded training length
and lineage media for image tasks.

## Integration validation

Before experiment preflight, verify:

1. the agent runs independently on one real task;
2. the adapter reads input and reaches the selected Relax protocol endpoint with Bearer Session authentication;
3. one complete JSON or Buffered SSE model/tool loop exits normally;
4. the Relax-facing client and outer deadlines cover prelaunch, partial-rollout, and fully-async request lifetimes;
5. request payloads pass the context-linearity audit;
6. configured reasoning/tool-call parsers produce the expected structured assistant messages, call IDs, and
   protocol-specific finish reasons;
7. implicit or explicit export resolves to committed state;
8. export count and training credit match `agentic-training-contract.md`;
9. agent, tool, sandbox, and external resources clean up.

Source anchors: `docs/en/guide/agentic-rollout.md`, `relax/agentic/runner/ipc.py`,
`relax/agentic/session/service.py`, `relax/agentic/session/state.py`, `relax/agentic/pipeline/reward.py`, and
`examples/mini_swe_agent/`.
