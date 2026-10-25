# Visual Context Segmentation

[中文](VCS.md)

VCS preserves text history and prunes old screenshots in batches **before model requests**, exporting several consecutive answers as one training segment. It is opt-in. Baseline retains the upstream two-image window and exports each call. Task execution, evaluation, reward shaping and logical GRPO grouping remain unchanged.

## Enable

Edit this recipe's `app/osworld_config.yaml`, then use the existing training entry point:

```yaml
context_mode: vcs
context_policy: baseline
vcs_keep_images: 2
vcs_prune_images: 2
```

Set `context_mode: baseline` to restore the original path. Setting both image parameters to `4` is an optional comparison, not a new production default. Both must be positive. VCS cannot be combined with upstream screenshot-skipping policies. `max_image_history_length` only controls the upstream temporary prompt in VCS; the actual visual window is controlled by VCS.

Keep the recipe's explicit exports, custom advantage callback, dynamic batching and token budgets. Do not switch multiple segments to ordinary per-row reward normalization. Preserved text and tool definitions can still exceed the model context limit: VCS bounds images, not arbitrary episode length.

The worker/agent environment can also enable the shared switch (explicit environment values override YAML):

```bash
export RELAX_VCS=1
export RELAX_VCS_KEEP_IMAGES=2
export RELAX_VCS_PRUNE_IMAGES=2
```

`RELAX_VCS=0` forces VCS off; an unset switch preserves YAML behavior. Forward these variables through the Ray runtime environment and into agent subprocesses. Exporting them only on the submission host is insufficient.

The shared implementation is `relax.utils.visual_context.VisualContext`; the OSWorld import remains a compatibility adapter. Other CUA adapters use the default `full_history` mode with one instance per sequential conversation: call `prepare(payload)` before the actual request, `commit(record, usage)` with the exact request and real assistant response, then `finish(reason)` and explicitly export `records`. Call `begin_turn(turn)` once per environment turn, retaining its state across parsing retries. Full history supports user/tool screenshots, tool calls/results and the initial task text. Disable upstream screenshot collapse so VCS owns pruning. The `latest_observation` mode retains OSWorld's rebuilt-prompt contract.

## Exact boundary and example

Following [SCALECUA §3.4, Figure 4 / Algorithm 1 and Appendix C.1 / Algorithm 2](https://arxiv.org/pdf/2607.11185), pruning triggers when `images in the next request > K + Δ`. The prose also says “reaches”; this implementation follows the strict inequality in the pseudocode. Count both the initial and current screenshot, including repeated identical observations. Remove the oldest Δ images, rather than immediately reducing the count to K. With one new screenshot per observation, the first request after pruning has K+1 images.

For K=2, Δ=2 and eight turns without retries or tool changes, I1 is the initial screenshot, Ot the upstream observation text and At the actual answer. Each segment also contains the system rules and a distinct segment marker.

| Turn | Request images | Request text history | Training ownership after inference                    |
| ---- | -------------- | -------------------- | ----------------------------------------------------- |
| 1    | I1             | O1                   | A1 belongs to segment 0                               |
| 2    | I1–I2          | O1,A1,O2             | A2 belongs to segment 0                               |
| 3    | I1–I3          | O1,A1,O2,A2,O3       | A3 belongs to segment 0                               |
| 4    | I1–I4          | O1,A1,…,O4           | A4 belongs to segment 0                               |
| 5    | I3–I5          | O1,A1,…,O5           | Save segment 0, then generate A5 in segment 1         |
| 6    | I3–I6          | O1,A1,…,O6           | A6 belongs to segment 1                               |
| 7    | I5–I7          | O1,A1,…,O7           | Save segment 1, then generate A7 in segment 2         |
| 8    | I5–I8          | O1,A1,…,O8           | A8 belongs to segment 2; save the tail at termination |

The three training rows supervise A1–A4, A5–A6 and A7–A8 respectively. Earlier answers in subsequent rows are input only. Each boundary exports the previous committed answer state, excluding the pending observation. Within a segment, requests strictly append: previous Instruction, Previous actions and tool-result text are never rewritten. Those fields from the upstream current observation are appended as new text, potentially repeating instructions or summaries. There is no extra summarization model.

## Data and loss ownership

`RelaxHybridAgent.call_llm` calls `VisualContext.prepare` before HTTP. Actual replies still enter `turn_records`; VCS separately records segment leaves. `agent.py` exports the exact messages, tools and chat_template_kwargs for each leaf. `AgenticSessionShard` resolves the committed Forest state and `SessionForest.build_sample` concatenates actual generated tokens and rollout log probabilities: response nodes receive mask=1, subsequent observations mask=0.

A fresh `[Visual context segment N]` user message appears after the system rules and before all history. This prevents longest-prefix matching from reusing any previous response ancestor, including when tools switch back or parse retries return identical text. Replayed history at the new root is entirely prompt context (mask=0). Old answers are never re-tokenized and paired with old probabilities. Only the last committed state of each segment is exported. The adapter never constructs token or probability arrays.

All segments retain the same Session / `Sample.index` and trajectory advantage. Sample-mean loss already sums effective tokens by logical index across segments, so no extra segment-count divisor is needed. The launcher keeps `OSWORLD_PER_TRAJ_NORM=0`. If an older configuration enables `=1`, VCS uses the original model-call count to preserve baseline scaling, not the segment count. Legacy `rollout/raw_reward` retains call-count weighting too. `agent_turns` remains environment turns; success metrics remain per Session and only the first export receives reward.

## Boundaries, diagnostics and limits

- This adapter targets the [upstream revision pinned in the README](https://github.com/fxsc03/OSWorld-main-nex/tree/50b4a85778cfc13ce00e3bf63a3cd08bd260fb96). New observations/execution feedback are in the last user message, and tool definitions mainly live in system text. System, tools or template changes close the old segment and start a fresh context with current definitions and preserved text history.
- Upstream parsing retries resample the same prompt up to five times. Failed answers remain training targets in the preceding segment; they are not inserted into retry input. Every returned nonempty content is retained, including length-truncated content. Empty responses or failed transport attempts do not create synthetic targets. If another upstream version rewrites retry messages, the adapter raises an explicit error instead of dropping correction text.
- Completion, normal step exhaustion and handled `predict_error` / context-length termination save the last nonempty segment. Cancellation, discarded Sessions and errors that prevent output retain the existing lifecycle; export is not guaranteed. Lost HTTP responses still follow existing transport semantics; this feature adds no exactly-once transport protocol.
- `metadata.vcs_segment` includes boundary reason, call indices, environment turns, parse attempts, new answer count, image/prune counts, triggering image count and input/answer token counts. The tail uses the task stop reason. Existing `_session_metrics` adds `osworld/vcs_segments`, `osworld/vcs_pruned_images`, `osworld/input_tokens` and `osworld/answer_tokens`. Existing model-call, retry and environment-step metrics retain their meaning. Token diagnostics use API usage; the Forest-generated Sample/dump is authoritative for exact masks, tokens and probabilities.
- The VCS algorithm is shared by a Relax utility module and CUA adapters, without new dependencies or changes to SessionForest token ownership. Archiving still captures actual calls, not segments as substitute calls. The external `HybridAgentLocal.trajectory` debug field still contains its temporary baseline prompt; use Relax dumps or the archive for actual VCS context.

CPU regression command, with mock models and environments and no Nex/GPU:

```bash
python3 -m pytest -q tests/examples/test_osworld_visual_context.py \
  tests/examples/test_osworld_agentic_advantage.py \
  tests/examples/test_osworld_agentic_monitoring.py \
  tests/examples/test_osworld_archive.py tests/test_agentic_rollout.py
```

Tests use real Forest matching, commits, explicit exports and training-data conversion, with mock encoding and generation. Deterministic comparison results are written to `comparison-*.json` in pytest's temporary directory. They do not represent Qwen token costs or speed. Later experiments should fix the initial model, task pool, rollout count, hardware and token/time budget, then compare held-out evaluation, useful-trajectory rate, rollout time, update time and full-cycle time. Real multimodal encoding, GPU training stability and throughput require separate validation; no reproduction of the paper's 2.83× speedup is claimed.
