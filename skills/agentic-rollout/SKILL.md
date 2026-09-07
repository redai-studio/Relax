---
name: agentic-rollout
description: Assess an agent, bridge it into Relax, check launch readiness, and demonstrate resident Agentic rollout with evidence. Use when evaluating an agent app before integration, connecting an external agent with --use-agentic-rollout, auditing model-visible context linearity or SessionForest export, checking remote or large-scale resident concurrency, or preparing an Agentic experiment.
argument-hint: "[assess|bridge|check|demonstrate] [agent-or-script-path]"
---

# Agentic rollout

Work in stages. Identify the user's current stage from the request and available artifacts before reading or changing code.

## Stage router

| User state | Enter |
| --- | --- |
| Agent is not connected to Relax | Stage A: Assess the Agent |
| User asks to implement the connection | Stage A quick gate, then Stage B: Bridge into Relax |
| Integration exists and user asks whether it is safe to run | Stage C: Check before Launch |
| User explicitly asks to run an experiment | Stage C must pass, then Stage D: Demonstrate with an Experiment |
| Experiment is stalled or failed | Diagnose the Agentic layer first, then route Ray evidence to `debug-hang` |

Start every response with the current stage letter and name, evidence available, and blocking unknowns. Do not repeat a completed stage when its evidence is still current; recheck drift-prone configuration and runtime state.

## Non-negotiable boundaries

- Inspect the current checkout, diff, and exact source before making claims.
- Treat model-facing `messages`, `tools`, and `chat_template_kwargs` as the context contract. Python calls and internal agent objects are not substitutes.
- Treat reasoning and tool-call parsers as part of the model/chat-template contract. A basic text response does not
  prove parser compatibility.
- Distinguish logical prompt Groups, resident Sessions, model requests, exported contexts, and physical training rows.
- Apply export, credit, and dynamic-batching rules to exported contexts per Session. The number of resident Sessions is a different dimension.
- Choosing resident Agentic rollout (`--use-agentic-rollout`) gives Agentic ownership of train and Eval generation;
  `--custom-generate-function-path` is ignored on this path.
- Distinguish the timeout of each Relax-facing model API request from `--agent-timeout`. The former must let one request
  survive prelaunch, partial, or fully-async holds; the latter is a Runtime active-time safety budget intended to
  contain a stuck agent loop or tool execution.
- Identify whether the agent uses Chat Completions, OpenAI Responses, or Anthropic Messages, and compare its payload
  with the supported fields, Items, and blocks. Record the request's `stream` setting and require complete history on
  every request. Use [runtime-operations.md](references/runtime-operations.md) for transport-specific response checks.
- Inspect fixed internal widths only when the large-scale risk gate is triggered.
- Keep this skill Agentic-specific. Route RL algorithms, OPD/TIS/loss details, generic TransferQueue sampling, model backends, and Ray scheduling to their dedicated experts or skills after checking the Agentic boundary.
- Never assume a remote agent platform has enough slots. Unknown capacity produces an `UNVERIFIED` Stage C result.
- Do not launch a remote experiment unless the user explicitly requests it. Load `relax-dev-debug` for code changes or remote validation.
- Respect explicit edit and test boundaries. Validation does not authorize adding tests or changing unrelated files.

## Stage A: Assess the Agent

Goal: decide whether the existing agent can connect without corrupting context, export, or execution semantics.

1. Read the agent entry point, Relax-facing agent client, its request and outer task timeouts, tool loop, context store,
   compaction/retry logic, and final-output path.
2. Capture or reconstruct exact model request and response payloads. Prefer the HTTP payload over internal message
   classes. Record the endpoint, protocol, `stream` value, authentication header, complete-history representation,
   reasoning/tool blocks, and missing, `null`, or zero-length content. Resolve the final request URL after
   client-specific path handling; determine whether the client appends its resource path or constructs the endpoint
   itself, and verify that the Relax route prefix is preserved.
3. Determine the raw reasoning and tool-call syntax produced by the model under the exact chat template. Identify the
   compatible SGLang parser names or prove that no parser is required.
4. Read [context-linearity.md](references/context-linearity.md) and classify every transition as append, intentional fork, accidental fork, or reset.
5. Read [integration-contract.md](references/integration-contract.md) and compare the agent with the Relax input, API, and output contracts.
6. Read [runtime-operations.md](references/runtime-operations.md) and check whether the agent client relies on unsupported API or process behavior.
7. Determine whether execution is local per Session or submitted to a centralized remote platform. Record any train/Eval slot pools and hard concurrency limits.

Return:

```text
Current stage: A
Stage name: Assess the Agent
Integration readiness: READY | NEEDS_CHANGES | BLOCKED
Context topology: linear | intentional branches | accidental branches | unknown
Execution shape: local | remote shared | remote dedicated | unknown
Required adaptations:
Blocking unknowns:
Next allowed stage:
```

Do not modify Relax or start an experiment in this stage.

## Stage B: Bridge into Relax

Enter only when the user asks for implementation and Stage A has no unresolved blocker.

1. Preserve the agent's normal model-and-tool loop.
2. Map dataset fields with `--input-key`, `--metadata-key`, and `--multimodal-keys`; decide whether the harness receives
   ready-to-use messages or constructs them from metadata at runtime.
3. Wire the complete agent environment in [integration-contract.md](references/integration-contract.md).
4. Configure the agent application's client that calls `RELAX_BASE_URL` so the timeout of each model API request can
   span every applicable prelaunch, partial-rollout, or fully-async hold. Keep this client timeout separate from
   `--agent-timeout`, which bounds the agent process's Runtime active time.
5. Send complete histories on every model request: Chat uses `messages`, Responses uses typed `input` Items, and
   Messages uses `system + messages`. Keep tools and applicable template arguments stable. Do not use Responses
   `previous_response_id` as a continuation mechanism.
6. When canonicalization is needed, ask before changing `developer` semantics, choose a stable nonempty representation
   for empty tool results, and reuse the canonical payload in later turns and explicit export.
7. Configure `--agentic-reasoning-parser` and `--agentic-tool-call-parser` when required by the verified model/template
   format. Do not copy parser names from a different model recipe.
8. Use [agentic-training-contract.md](references/agentic-training-contract.md) for export, credit, logical identity, and dynamic-batching decisions. Implicit export is reserved for audited linear history; nonlinear history requires explicit export.
   Explicit export remains canonical Chat-shaped `messages`, nested function `tools`, and `chat_template_kwargs`
   regardless of the request protocol.
9. Define reward ownership. One exported context needs an exported reward or configured reward producer. Multiple
   exported contexts require `--agentic-custom-advantage-path` and dynamic batching; ordinarily avoid
   `--custom-rm-path` in this mode.
10. Keep the change at the adapter and recipe boundary unless the verified contract requires a core `relax/agentic/**` change.

Start with:

- `docs/en/guide/agentic-rollout.md`
- the closest maintained example, such as `examples/mini_swe_agent/`
- [integration-contract.md](references/integration-contract.md)

Read the exact core source only when a verified incompatibility or requested core change requires it.

## Stage C: Check before Launch

Goal: produce a launch verdict without starting the job.

1. Re-read the final agent adapter and launch script.
2. Read [parameter-preflight.md](references/parameter-preflight.md). Resolve every applicable Agentic parameter after
   defaults and validation, then check its dependencies and runtime evidence.
3. Audit context linearity again from the integrated HTTP payloads.
4. Read [resident-lifecycle.md](references/resident-lifecycle.md). Verify Group size, first-request barrier, and cleanup.
5. When agents use a capacity-limited external platform, read
   [external-agent-capacity.md](references/external-agent-capacity.md). Otherwise report `External capacity: N/A`.
6. Read [large-scale-rollout.md](references/large-scale-rollout.md) if and only if the projected or observed load is
   large enough to approach or exceed a fixed internal width or known validated scale envelope.
   Otherwise report `Internal scale: N/A`; prelaunch, production use, or a "large-scale" label alone does not activate
   this check.
7. When the agent uses explicit export, nonlinear history, multiple contexts, custom credit, `--log-passrate`, a reward
   object or `--reward-key`, a configured RM, or `--group-rm`, read
   [agentic-training-contract.md](references/agentic-training-contract.md).
8. Read [runtime-operations.md](references/runtime-operations.md). Verify protocol and transport compatibility, the
   per-request Relax-facing client timeout, reasoning/tool-call parsers, the agent process timeout, optional
   KV/admission flags, errors, and observable evidence.
9. When prelaunch, partial rollout, or fully async is enabled, read [partial-and-async-lifecycle.md](references/partial-and-async-lifecycle.md) and verify the applicable cross-step state transitions.
10. Report unknown remote capacity, internal scale evidence, or networking as blockers rather than optimistic assumptions.

Return:

```text
Current stage: C
Stage name: Check before Launch
Verdict: PASS | UNSAFE | UNVERIFIED
Agentic parameters: PASS | NEEDS_CHANGES | UNVERIFIED
Context linearity: PASS | NEEDS_CHANGES | UNVERIFIED
External capacity: PASS | UNSAFE | UNVERIFIED
Internal scale: PASS | UNSAFE | UNVERIFIED | N/A
Internal width changes: N/A | see conditional large-scale result
Relax-facing per-request timeout: PASS | UNSAFE | UNVERIFIED
Reasoning/tool-call parsers: PASS | NEEDS_CHANGES | UNVERIFIED | N/A
Export/credit/batching: PASS | NEEDS_CHANGES | UNVERIFIED
Required slots:
Configured slots:
Blocking items:
Ready to launch: yes | no
```

## Stage D: Demonstrate with an Experiment

Enter only after explicit user authorization and a passing Stage C check. Use the environment-specific launch skill when one exists; otherwise follow `relax-dev-debug` and `ssh-ray-cluster` constraints.

Validate in order:

1. one complete Group reaches the first-request barrier;
2. the full model/tool loop completes, with expected text, reasoning, tool calls, call IDs, arguments, finish reasons,
   usage, and protocol-specific JSON or Buffered SSE terminal shape in the Relax response;
3. request payloads preserve context lineage and stable tools/template arguments;
4. SessionForest commits the intended leaf or branches;
5. export, reward, and custom credit match the selected contexts; if custom advantage can return `None`, the whole-Group drop and replenishment path is observed;
6. agent and external resources clean up;
7. a meaningful multi-sample run reaches transfer and an optimizer step;
8. partial/resume or Eval overlap behavior is exercised when configured.
9. large-scale runs reach their target resident fanout and soak duration without saturating ingress, Shard, permit, or
   cleanup concurrency lanes.

Use the log markers and state evidence listed in [runtime-operations.md](references/runtime-operations.md). Keep algorithm-specific proof in the corresponding algorithm review rather than expanding this skill.

A one-sample success proves plumbing only. Do not claim training success without meaningful Group behavior, reward evidence, and an optimizer step.

## Failure routing

Trace failures through this ownership order:

```text
DataSource -> Prepare Group -> first-request barrier -> Runtime lease
-> Session -> model/backend request -> agent process
-> external executor -> Reward -> Transfer
```

Stay in Agentic diagnosis for context mismatches, external slot exhaustion, first-request barriers, protected Sessions, agent timeouts, exports, rewards, or cleanup. Use `debug-hang` after evidence points to Ray scheduling, Actor state, placement, resources, or distributed collectives.

## References

- [context-linearity.md](references/context-linearity.md): model-visible history and branch audit
- [external-agent-capacity.md](references/external-agent-capacity.md): mandatory remote slot calculation
- [integration-contract.md](references/integration-contract.md): adapter, request, export, and validation contracts
- [large-scale-rollout.md](references/large-scale-rollout.md): conditional target-scale readiness
- [parameter-preflight.md](references/parameter-preflight.md): effective Agentic flags, dependencies, and evidence
- [resident-lifecycle.md](references/resident-lifecycle.md): resident Group and Session lifecycle
- [partial-and-async-lifecycle.md](references/partial-and-async-lifecycle.md): cross-step park, resume, protection, and close semantics
- [agentic-training-contract.md](references/agentic-training-contract.md): Agentic export fanout, identity, credit, Eval, and batching
- [runtime-operations.md](references/runtime-operations.md): process, API, timeout, optional controls, errors, and evidence
